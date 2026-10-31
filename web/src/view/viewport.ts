/**
 * The 2D viewport: one canvas, the image drawn with its pixel spacing, W/L through a lookup table, overlays in the
 * four corners like dcv/slice_view.py. Pointer events cover mouse, touch and Apple Pencil:
 *   one finger / left button  = the active tool (W/L, pan, zoom, scroll, length, DSA shift / auto shift)
 *   two fingers               = pinch zoom + pan (the one-finger action that had started is undone)
 *   double tap / double click = fit to the window
 *   mouse: middle = pan, right = zoom, wheel = frames, Ctrl + wheel / trackpad pinch = zoom
 */
import type { DsaHint } from "../dicom/header";
import type { Pixels } from "../dicom/volume";
import { DSA, quantiles } from "../dsa";
import { ellipseStats, measureLines, REQUIRED, type MeasureKind, type Measurement } from "./measure";

/** What a viewport shows: a loaded series (dicom/volume.ts Volume) or an MPR plane (view/mpr.ts). */
export interface ViewSource {
  readonly n: number;
  readonly rows: number;
  readonly cols: number;
  readonly spacing: [number, number, number];
  readonly spacingKnown: boolean;
  readonly ww: number | null;
  readonly wl: number | null;
  readonly frameTimeMs: number | null;
  /** "CT": region statistics are in HU. MPR planes of a CT say so too. */
  readonly modality?: string;
  readonly monochrome1: boolean;
  readonly patient: string;
  readonly description: string;
  readonly dsaHint: DsaHint | null;
  readonly rgb: Uint8Array | null;
  readonly complete: boolean;
  readonly readyCount: number;
  readonly errors: string[];
  onFrame: ((k: number) => void) | null;
  frame(k: number): Pixels;
  rgbFrame(k: number): Uint8Array | null;
  isReady(k: number): boolean;
  want(ks: number[]): Promise<void>;
  /** Changes whenever frame(k) would return other pixels for the same k (an MPR plane that moved). */
  version?: number;
  /** Key of the measurements drawn with frame k (MPR: the plane's position and orientation). */
  measureKey?(k: number): string | number;
  /** Zoom of "fit": an MPR plane is larger than the volume (it covers every rotation). */
  fitZoom?(): number;
}

/** Extra drawing and pointer handling on top of a viewport (MPR axis lines). */
export interface ViewHooks {
  overlay?(ctx: CanvasRenderingContext2D): void;
  /** A press the hook takes over (true): its moves and release then go to drag / end, not to the tool. */
  down?(x: number, y: number, e: PointerEvent): boolean;
  drag?(x: number, y: number): void;
  end?(): void;
  hover?(x: number, y: number): string | null;          // CSS cursor
  index?(k: number): void;                              // the frame index was changed by the viewport
}

export type Tool = "wl" | "pan" | "zoom" | "scroll" | MeasureKind | "locate" | "seed" | "dsaShift" | "dsaAuto";

const isMeasure = (t: Tool): t is MeasureKind => t === "length" || t === "ellipse" || t === "angle" || t === "cobb";

interface Single {
  tool: Tool;
  x0: number;
  y0: number;
  x: number;
  y: number;
  ww0: number;
  wl0: number;
  zoom0: number;
  pan0: [number, number];
  index0: number;
  shift0: [number, number] | null;
  moved: boolean;
  acc: number;
  /** The measurement being drawn or moved: point `idx` follows the finger, -1 = all of it. */
  edit: { list: Measurement[]; m: Measurement; idx: number; created: boolean; orig: [number, number][] } | null;
}

interface Pinch {
  d0: number;
  zoom0: number;
  anchor: [number, number];      // image point under the fingers' midpoint
}

const MEASURE_COLOR = "#FFD24A";
const SELECTED_COLOR = "#7DB6F5";
const SQUARE_COLOR = "rgb(255,220,0)";

export class Viewport {
  readonly canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  vol: ViewSource | null = null;
  hooks: ViewHooks = {};
  /** Highlighted frame (the active viewport of several). */
  active = false;
  private hookDrag = false;
  index = 0;
  ww = 400;
  wl = 40;
  invert = false;
  zoom = 1;
  pan: [number, number] = [0, 0];
  tool: Tool = "wl";
  dsa: DSA | null = null;
  private nativeWindow: [number, number] | null = null;
  measurements = new Map<string | number, Measurement[]>();
  selected: Measurement | null = null;
  /** An angle / Cobb that still needs points: the next press continues it. */
  private draft: Measurement | null = null;

  cinePlaying = false;
  fps = 10;
  private cineAcc = 0;
  private cineLast = 0;

  /** A seed placed (Seed tool, or Ctrl / Cmd + click), in continuous image coordinates. */
  onSeed: ((r: number, c: number) => void) | null = null;
  /** What lies over the image, a canvas of the image's size (segments), null = nothing. */
  layer: ((k: number) => HTMLCanvasElement | null) | null = null;

  /** Called after anything the toolbar / cine bar shows has changed. */
  onChange: () => void = () => undefined;
  onStatus: (text: string) => void = () => undefined;

  private img = document.createElement("canvas");
  private imgData: ImageData | null = null;
  private lut = new Uint32Array(65536);
  private lutKey = "";
  private drawnKey = "";
  private dirty = false;
  private raf = 0;
  private pointers = new Map<number, { x: number; y: number; type: string }>();
  private single: Single | null = null;
  private pinch: Pinch | null = null;
  private gestureDone = false;        // after a pinch, ignore the remaining finger until all are lifted
  private hover: [number, number] | null = null;
  private flashSquare: { r: number; c: number; until: number } | null = null;
  private lastTap = { t: 0, x: 0, y: 0 };
  private wheelAcc = 0;
  busy = "";

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d", { alpha: false })!;
    new ResizeObserver(() => this.resize()).observe(canvas.parentElement!);
    canvas.addEventListener("pointerdown", (e) => this.down(e));
    canvas.addEventListener("pointermove", (e) => this.move(e));
    canvas.addEventListener("pointerup", (e) => this.up(e));
    canvas.addEventListener("pointercancel", (e) => this.up(e, true));
    canvas.addEventListener("pointerleave", () => {
      this.hover = null;
      if (this.tool === "dsaAuto") this.requestRender();
    });
    canvas.addEventListener("wheel", (e) => this.wheel(e), { passive: false });
    canvas.addEventListener("dblclick", () => this.resetView());
    canvas.addEventListener("contextmenu", (e) => e.preventDefault());
    // Safari's own pinch-to-zoom of the page
    for (const t of ["gesturestart", "gesturechange", "gestureend"]) canvas.addEventListener(t, (e) => e.preventDefault());
    this.resize();
  }

  // ================================================================ data
  setVolume(vol: ViewSource | null, first = 0): void {
    this.stopCine();
    this.vol = vol;
    this.dsa = null;
    this.nativeWindow = null;
    this.measurements = new Map();
    this.selected = null;
    this.draft = null;
    this.index = first;
    this.invert = !!vol?.monochrome1;
    this.drawnKey = "";
    if (vol) {
      const other = vol.onFrame;                      // the same series may be shown in another viewport too
      vol.onFrame = (k) => {
        other?.(k);
        if (this.vol !== vol) return;
        if (k === this.index || this.dsa?.mask.includes(k) || !vol.complete && vol.readyCount % 8 === 0) this.requestRender();
        if (vol.complete) {
          this.requestRender();                       // the corner's "Decoding n/N" goes away
          this.onChange();
        }
      };
      if (vol.rgb) [this.ww, this.wl] = [255, 127.5];
      else if (vol.ww !== null && vol.wl !== null) [this.ww, this.wl] = [vol.ww, vol.wl];
      else this.autoWindow(false);
      this.fps = vol.frameTimeMs ? Math.max(1, Math.min(60, Math.round(1000 / vol.frameTimeMs))) : 10;
    }
    this.resetView();
    this.onChange();
  }

  count(): number {
    return this.vol?.n ?? 0;
  }

  setIndex(k: number): void {
    const n = this.count();
    if (!n) return;
    k = Math.max(0, Math.min(n - 1, k));
    if (k === this.index) return;
    this.index = k;
    this.selected = null;
    this.hooks.index?.(k);
    this.requestRender();
    this.onChange();
  }

  setWindow(ww: number, wl: number): void {
    this.ww = Math.max(1, ww);
    this.wl = wl;
    this.requestRender();
    this.onChange();
  }

  /** The 1–99 % range of the image on screen (subtracted when DSA is on). */
  autoWindow(notify = true): void {
    const v = this.vol;
    if (!v || v.rgb) return;
    if (!v.isReady(this.index)) {
      if (!v.ww) [this.ww, this.wl] = [400, 40];
      v.want([this.index]).then(() => this.autoWindow(notify));
      return;
    }
    const plane = this.dsa ? this.dsa.subtract(this.index, true) : v.frame(this.index);
    const [lo, hi] = quantiles(plane, 0.01, 0.99);
    this.ww = Math.max(1, hi - lo);
    this.wl = (hi + lo) / 2;
    if (notify) {
      this.requestRender();
      this.onChange();
    }
  }

  resetView(): void {
    this.zoom = this.vol?.fitZoom?.() ?? 1;
    this.pan = [0, 0];
    this.requestRender();
    this.onChange();
  }

  // ================================================================ DSA
  dsaAvailable(): boolean {
    return !!this.vol && this.vol.n > 1 && !this.vol.rgb;
  }

  /** Subtraction on / off. On: frames minus the mask with the DSA window; off: the window from before. */
  async setDsa(on: boolean): Promise<boolean> {
    const v = this.vol;
    if (on && !this.dsa && v && this.dsaAvailable()) {
      const dsa = new DSA(v, v.dsaHint);
      this.busy = "Decoding the mask…";
      this.requestRender();
      await v.want(dsa.needs(this.index));
      this.busy = "";
      if (this.vol !== v) return false;
      this.dsa = dsa;
      this.nativeWindow = [this.ww, this.wl];
      const [ww, wl] = dsa.suggestWindow(this.ww, this.index);
      this.setWindow(ww, wl);
      this.onStatus(`DSA on — ${dsa.label()[0]}`);
    } else if (!on && this.dsa) {
      this.dsa = null;
      if (this.nativeWindow) this.setWindow(...this.nativeWindow);
      if (this.tool === "dsaShift" || this.tool === "dsaAuto") this.tool = "wl";
    }
    this.drawnKey = "";
    this.requestRender();
    this.onChange();
    return !!this.dsa;
  }

  async dsaMaskCurrent(): Promise<void> {
    if (!(await this.setDsa(true))) return;
    this.dsa!.setMask([this.index]);
    this.onStatus(`DSA mask = image ${this.index + 1}`);
    this.requestRender();
    this.onChange();
  }

  dsaReset(): void {
    if (!this.dsa) return;
    this.dsa.reset();
    this.vol?.want(this.dsa.needs(this.index)).then(() => this.requestRender());
    this.onStatus("DSA mask reset (default mask, no shift)");
    this.requestRender();
    this.onChange();
  }

  async dsaMove(dy: number, dx: number): Promise<void> {
    if (!(await this.setDsa(true))) return;
    this.dsa!.move(dy, dx);
    this.onStatus(this.dsa!.label()[1]);
    this.requestRender();
    this.onChange();
  }

  async dsaAutoAt(r: number, c: number): Promise<void> {
    if (!(await this.setDsa(true)) || !this.vol) return;
    const dsa = this.dsa!;
    this.flashSquare = { r, c, until: performance.now() + 900 };
    this.busy = "Auto pixel-shift…";
    this.requestRender();
    await new Promise((res) => setTimeout(res, 30));        // let the "busy" text paint first
    await this.vol.want([this.index]);
    const t0 = performance.now();
    dsa.autoShift(this.index, [r, c], dsa.square());
    this.busy = "";
    this.onStatus(`Auto pixel-shift: ${dsa.label()[1]} (${Math.round(performance.now() - t0)} ms)`);
    this.requestRender();
    this.onChange();
  }

  // ================================================================ cine
  toggleCine(): void {
    if (this.cinePlaying) this.stopCine();
    else this.playCine();
  }

  playCine(): void {
    if (this.count() < 2 || this.cinePlaying) return;
    this.cinePlaying = true;
    this.cineAcc = 0;
    this.cineLast = performance.now();
    this.requestRender();
    this.onChange();
  }

  stopCine(): void {
    if (!this.cinePlaying) return;
    this.cinePlaying = false;
    this.onChange();
    this.requestRender();
  }

  setFps(fps: number): void {
    this.fps = Math.max(1, Math.min(60, Math.round(fps)));
    this.onChange();
  }

  private cineStep(now: number): void {
    const v = this.vol;
    if (!v) return;
    this.cineAcc += now - this.cineLast;
    this.cineLast = now;
    const dur = 1000 / this.fps;
    if (this.cineAcc > 1000) this.cineAcc = dur;              // tab was hidden: no burst of frames
    while (this.cineAcc >= dur) {
      const next = (this.index + 1) % v.n;
      const needs = this.dsa ? this.dsa.needs(next) : [next];
      if (!needs.every((k) => v.isReady(k))) {           // still decoding: wait for it rather than skip
        v.want(needs);
        this.cineAcc = 0;
        break;
      }
      this.cineAcc -= dur;
      this.index = next;
      this.selected = null;
      this.onChange();
    }
  }

  // ================================================================ geometry
  private resize(): void {
    const p = this.canvas.parentElement!;
    const dpr = window.devicePixelRatio || 1;
    const w = p.clientWidth, h = p.clientHeight;
    this.canvas.style.width = `${w}px`;
    this.canvas.style.height = `${h}px`;
    this.canvas.width = Math.max(1, Math.round(w * dpr));
    this.canvas.height = Math.max(1, Math.round(h * dpr));
    this.requestRender();
  }

  private cssSize(): [number, number] {
    const dpr = window.devicePixelRatio || 1;
    return [this.canvas.width / dpr, this.canvas.height / dpr];
  }

  /** CSS pixels per mm, and the top-left corner of the image on screen. */
  private layout(): { scale: number; x0: number; y0: number; sx: number; sy: number } {
    const v = this.vol!;
    const [W, H] = this.cssSize();
    const sy = v.spacing[1], sx = v.spacing[2];
    const mmW = v.cols * sx, mmH = v.rows * sy;
    const scale = Math.min(W / mmW, H / mmH) * this.zoom;
    return { scale, sx, sy, x0: W / 2 + this.pan[0] - (mmW * scale) / 2, y0: H / 2 + this.pan[1] - (mmH * scale) / 2 };
  }

  toScreen(r: number, c: number): [number, number] {
    const L = this.layout();
    return [L.x0 + c * L.sx * L.scale, L.y0 + r * L.sy * L.scale];
  }

  toImage(x: number, y: number): [number, number] {
    const L = this.layout();
    return [(y - L.y0) / (L.sy * L.scale), (x - L.x0) / (L.sx * L.scale)];
  }

  private zoomAbout(x: number, y: number, zoom: number): void {
    if (!this.vol) return;
    const [r, c] = this.toImage(x, y);
    this.zoom = Math.max(0.05, Math.min(60, zoom));
    const [nx, ny] = this.toScreen(r, c);
    this.pan = [this.pan[0] + x - nx, this.pan[1] + y - ny];
  }

  // ================================================================ input
  private pos(e: PointerEvent | WheelEvent | MouseEvent): [number, number] {
    const b = this.canvas.getBoundingClientRect();
    return [e.clientX - b.left, e.clientY - b.top];
  }

  private down(e: PointerEvent): void {
    if (!this.vol) return;
    e.preventDefault();
    try {
      this.canvas.setPointerCapture(e.pointerId);
    } catch {
      // a pointer the browser no longer tracks (or a synthetic one): carry on without capture
    }
    const [x, y] = this.pos(e);
    this.pointers.set(e.pointerId, { x, y, type: e.pointerType });
    if (this.pointers.size === 2) {
      this.cancelSingle();
      if (this.hookDrag) {
        this.hookDrag = false;
        this.hooks.end?.();
      }
      const [p, q] = [...this.pointers.values()];
      const mid: [number, number] = [(p.x + q.x) / 2, (p.y + q.y) / 2];
      this.pinch = { d0: Math.max(10, Math.hypot(p.x - q.x, p.y - q.y)), zoom0: this.zoom, anchor: this.toImage(...mid) };
      this.gestureDone = true;
      return;
    }
    if (this.pointers.size > 2 || this.gestureDone) return;
    if (this.onSeed && (e.pointerType !== "mouse" || e.button === 0) && (this.tool === "seed" || e.ctrlKey || e.metaKey)) {
      this.onSeed(...this.toImage(x, y));
      return;
    }
    if ((e.pointerType !== "mouse" || e.button === 0) && this.hooks.down?.(x, y, e)) {
      this.hookDrag = true;
      return;
    }
    let tool = this.tool;
    if (e.pointerType === "mouse") {
      if (e.button === 1) tool = "pan";
      else if (e.button === 2) tool = "zoom";
      else if (e.button !== 0) return;
    }
    this.single = {
      tool, x0: x, y0: y, x, y, ww0: this.ww, wl0: this.wl, zoom0: this.zoom, pan0: [...this.pan], index0: this.index,
      shift0: this.dsa ? [...this.dsa.shift] : null, moved: false, acc: 0, edit: null,
    };
    if (isMeasure(tool)) this.startMeasure(tool, x, y, e.pointerType);
    if (tool === "dsaShift" && !this.dsa) this.setDsa(true);
    if (tool === "scroll" || tool === "dsaShift") this.stopCine();
  }

  private move(e: PointerEvent): void {
    const [x, y] = this.pos(e);
    if (!this.pointers.has(e.pointerId)) {
      if (e.pointerType === "mouse" && this.hooks.hover && this.vol) this.canvas.style.cursor = this.hooks.hover(x, y) ?? "";
      if (e.pointerType === "mouse" && this.tool === "dsaAuto" && this.vol) {
        this.hover = this.toImage(x, y);
        this.requestRender();
      }
      return;
    }
    this.pointers.set(e.pointerId, { x, y, type: e.pointerType });
    if (this.pinch && this.pointers.size >= 2) {
      const [p, q] = [...this.pointers.values()];
      const mid: [number, number] = [(p.x + q.x) / 2, (p.y + q.y) / 2];
      this.zoom = Math.max(0.05, Math.min(60, (this.pinch.zoom0 * Math.hypot(p.x - q.x, p.y - q.y)) / this.pinch.d0));
      const [ax, ay] = this.toScreen(...this.pinch.anchor);
      this.pan = [this.pan[0] + mid[0] - ax, this.pan[1] + mid[1] - ay];
      this.requestRender();
      this.onChange();
      return;
    }
    if (this.hookDrag) {
      this.hooks.drag?.(x, y);
      return;
    }
    const s = this.single;
    if (!s) return;
    const dx = x - s.x, dy = y - s.y;
    if (Math.hypot(x - s.x0, y - s.y0) > 4) s.moved = true;
    switch (s.tool) {
      case "wl": {
        const sens = Math.max(0.5, s.ww0 / 250);          // dcv/slice_view.py: right = wider, down = higher level
        this.setWindow(s.ww0 + (x - s.x0) * sens, s.wl0 + (y - s.y0) * sens);
        break;
      }
      case "pan":
        this.pan = [s.pan0[0] + x - s.x0, s.pan0[1] + y - s.y0];
        this.requestRender();
        break;
      case "zoom": {
        const pan = this.pan;
        this.zoom = s.zoom0;
        this.pan = [...s.pan0];
        this.zoomAbout(s.x0, s.y0, s.zoom0 * Math.exp(-(y - s.y0) / 150));
        if (!Number.isFinite(this.pan[0])) this.pan = pan;
        this.requestRender();
        this.onChange();
        break;
      }
      case "scroll": {
        s.acc += dy;
        const per = e.pointerType === "touch" ? 10 : 4;
        const step = Math.trunc(s.acc / per);
        if (step) {
          s.acc -= step * per;
          this.setIndex(this.index + step);
        }
        break;
      }
      case "length": case "ellipse": case "angle": case "cobb":
        if (s.edit) {
          const p = this.toImage(x, y);
          if (s.edit.idx < 0) {
            const [r0, c0] = this.toImage(s.x0, s.y0);
            s.edit.m.pts = s.edit.orig.map(([r, c]) => [r + p[0] - r0, c + p[1] - c0]);
          } else s.edit.m.pts[s.edit.idx] = p;
          this.requestRender();
        }
        break;
      case "dsaShift":
        if (this.dsa) {
          const [r0, c0] = this.toImage(s.x, s.y);
          const [r1, c1] = this.toImage(x, y);
          this.dsa.move(r1 - r0, c1 - c0);          // the mask follows the finger (zoom aware)
          this.requestRender();
          this.onChange();
        }
        break;
      case "dsaAuto":
        this.hover = this.toImage(x, y);
        this.requestRender();
        break;
    }
    s.x = x;
    s.y = y;
  }

  private up(e: PointerEvent, cancelled = false): void {
    if (!this.pointers.has(e.pointerId)) return;
    const [x, y] = this.pos(e);
    this.pointers.delete(e.pointerId);
    if (this.pinch) {
      if (this.pointers.size < 2) this.pinch = null;
    }
    if (this.hookDrag) {
      this.hookDrag = false;
      this.hooks.end?.();
    }
    const s = this.single;
    if (s && !this.gestureDone) {
      this.single = null;
      if (cancelled) this.cancel(s);
      else this.finish(s, x, y, e.pointerType);
    }
    if (this.pointers.size === 0) {
      this.gestureDone = false;
      this.single = null;
    }
  }

  private finish(s: Single, x: number, y: number, type: string): void {
    if (isMeasure(s.tool) && s.edit) {
      this.finishMeasure(s.edit);
      this.requestRender();
      this.onChange();
    } else if (s.tool === "dsaAuto" && !s.moved) {
      const [r, c] = this.toImage(x, y);
      if (type !== "mouse") this.hover = null;
      this.dsaAutoAt(r, c);
    }
    // double tap (touch / pen) = fit to the window; the mouse has dblclick
    if (!s.moved && type !== "mouse") {
      const now = performance.now();
      if (now - this.lastTap.t < 320 && Math.hypot(x - this.lastTap.x, y - this.lastTap.y) < 30 && !isMeasure(s.tool)) {
        this.resetView();
        this.lastTap.t = 0;
      } else this.lastTap = { t: now, x, y };
    }
  }

  /** Undo what a single-finger drag did when a second finger turns it into a pinch. */
  private cancelSingle(): void {
    if (this.single) this.cancel(this.single);
    this.single = null;
  }

  private cancel(s: Single): void {
    if (s.tool === "wl") this.setWindow(s.ww0, s.wl0);
    if (s.tool === "scroll") this.setIndex(s.index0);
    if (s.tool === "pan" || s.tool === "zoom") {
      this.zoom = s.zoom0;
      this.pan = [...s.pan0];
    }
    if (s.tool === "dsaShift" && this.dsa && s.shift0) this.dsa.shift = s.shift0;
    if (s.edit) {
      if (s.edit.created) {
        s.edit.list.splice(s.edit.list.indexOf(s.edit.m), 1);
        if (this.draft === s.edit.m) this.draft = null;
      } else s.edit.m.pts = s.edit.orig.map((p) => [...p] as [number, number]);
    }
    this.requestRender();
    this.onChange();
  }

  /**
   * A press with a measuring tool. Length, ellipse: one drag. Angle: drag the first arm, then tap or drag the third
   * point. Cobb: drag one line, then the other. Tapping point by point works too. A press on a point moves that
   * point, on the line (or inside the ellipse) moves the whole measurement.
   */
  private startMeasure(kind: MeasureKind, x: number, y: number, type: string): void {
    const s = this.single!;
    const list = this.measurements.get(this.mkey()) ?? [];
    this.measurements.set(this.mkey(), list);
    const p = this.toImage(x, y);
    const d = this.draft;
    if (d && (d.kind !== kind || !list.includes(d))) this.cancelDraft();     // the draft stayed on another image
    const draft = this.draft;
    if (draft) {
      const orig = draft.pts.map((q) => [...q] as [number, number]);
      if (draft.kind === "cobb" && draft.pts.length === 2) draft.pts.push(p);
      draft.pts.push([...p]);
      s.edit = { list, m: draft, idx: draft.pts.length - 1, created: false, orig };
      this.requestRender();
      return;
    }
    const tol = type === "mouse" ? 9 : 22;
    for (const m of [...list].reverse()) {
      for (let idx = 0; idx < m.pts.length; idx++) {
        const [px, py] = this.toScreen(...m.pts[idx]);
        if (Math.hypot(px - x, py - y) < tol) {
          this.selected = m;
          s.edit = { list, m, idx, created: false, orig: m.pts.map((q) => [...q] as [number, number]) };
          return;
        }
      }
    }
    for (const m of [...list].reverse()) {
      if (this.onBody(m, x, y, tol * 0.7)) {
        this.selected = m;
        s.edit = { list, m, idx: -1, created: false, orig: m.pts.map((q) => [...q] as [number, number]) };
        this.requestRender();
        this.onChange();
        return;
      }
    }
    const m: Measurement = { kind, pts: [p, [...p]] };
    list.push(m);
    this.selected = m;
    if (REQUIRED[kind] > 2) this.draft = m;
    s.edit = { list, m, idx: 1, created: true, orig: [] };
  }

  private onBody(m: Measurement, x: number, y: number, tol: number): boolean {
    const W = m.pts.map((q) => this.toScreen(...q));
    if (m.kind === "ellipse") {
      return x > Math.min(W[0][0], W[1][0]) - tol && x < Math.max(W[0][0], W[1][0]) + tol
        && y > Math.min(W[0][1], W[1][1]) - tol && y < Math.max(W[0][1], W[1][1]) + tol;
    }
    const step = m.kind === "cobb" ? 2 : 1;
    for (let i = 0; i + 1 < W.length; i += step) if (distToSegment([x, y], W[i], W[i + 1]) < tol) return true;
    return false;
  }

  private finishMeasure(edit: NonNullable<Single["edit"]>): void {
    const m = edit.m;
    if (edit.idx < 1) return;
    const [ax, ay] = this.toScreen(...m.pts[edit.idx - 1]), [bx, by] = this.toScreen(...m.pts[edit.idx]);
    const tap = Math.hypot(ax - bx, ay - by) < 4;
    const need = REQUIRED[m.kind];
    if (tap && need === 2 && edit.created) {
      edit.list.splice(edit.list.indexOf(m), 1);             // a tap, not a measurement
      this.selected = null;
      return;
    }
    if (this.draft !== m) return;
    if (tap && (edit.idx === 1 || edit.idx === 3)) m.pts.pop();    // a tap placed one point: the next one follows
    if (m.pts.length >= need) {
      this.draft = null;
      return;
    }
    this.onStatus(m.kind === "angle" ? (m.pts.length === 1 ? "Angle: now the vertex" : "Angle: now the end of the second arm")
      : m.pts.length < 3 ? (m.pts.length === 1 ? "Cobb: now the end of the first line" : "Cobb: now draw the second line") : "Cobb: now the end of the second line");
  }

  private mkey(): string | number {
    return this.vol?.measureKey?.(this.index) ?? this.index;
  }

  /** Drop an angle / Cobb that was not finished (another tool, Esc). */
  cancelDraft(): void {
    const d = this.draft;
    if (!d) return;
    this.draft = null;
    for (const list of this.measurements.values()) {
      const i = list.indexOf(d);
      if (i >= 0) list.splice(i, 1);
    }
    if (this.selected === d) this.selected = null;
    this.requestRender();
  }

  deleteSelected(): boolean {
    for (const list of this.measurements.values()) {
      const i = this.selected ? list.indexOf(this.selected) : -1;
      if (i >= 0) {
        list.splice(i, 1);
        if (this.draft === this.selected) this.draft = null;
        this.selected = null;
        this.requestRender();
        this.onChange();
        return true;
      }
    }
    return false;
  }

  clearMeasurements(): void {
    this.draft = null;
    this.measurements.delete(this.mkey());
    this.selected = null;
    this.requestRender();
    this.onChange();
  }

  private wheel(e: WheelEvent): void {
    if (!this.vol) return;
    e.preventDefault();
    const [x, y] = this.pos(e);
    if (e.ctrlKey) {                                        // Ctrl + wheel, or a trackpad pinch
      this.zoomAbout(x, y, this.zoom * Math.exp(-e.deltaY * (e.deltaMode ? 0.05 : 0.01)));
      this.requestRender();
      this.onChange();
      return;
    }
    this.wheelAcc += e.deltaMode === 1 ? e.deltaY / 3 : e.deltaMode === 2 ? e.deltaY * 10 : e.deltaY / 100;
    const step = Math.trunc(this.wheelAcc);
    if (step) {
      this.wheelAcc -= step;
      this.setIndex(this.index + step);
    }
  }

  /** Keyboard (desktop / iPad keyboard). Returns whether the key was used. */
  key(e: KeyboardEvent): boolean {
    if (!this.vol) return false;
    const arrows: Record<string, [number, number]> = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] };
    if (this.dsa && arrows[e.key] && e.shiftKey) {
      const step = e.ctrlKey || e.metaKey ? 0.1 : 1;       // Shift 1 px, Ctrl+Shift 0.1 px
      this.dsaMove(arrows[e.key][0] * step, arrows[e.key][1] * step);
      return true;
    }
    switch (e.key) {
      case "ArrowUp": case "ArrowLeft": this.setIndex(this.index - 1); return true;
      case "ArrowDown": case "ArrowRight": this.setIndex(this.index + 1); return true;
      case "PageUp": this.setIndex(this.index - 10); return true;
      case "PageDown": this.setIndex(this.index + 10); return true;
      case "Home": this.setIndex(0); return true;
      case "End": this.setIndex(this.count() - 1); return true;
      case " ": this.toggleCine(); return true;
      case "Delete": case "Backspace": return this.deleteSelected();
      case "Escape":
        this.cancelDraft();
        this.selected = null;
        this.requestRender();
        return true;
    }
    return false;
  }

  // ================================================================ rendering
  requestRender(): void {
    if (this.dirty) return;
    this.dirty = true;
    this.raf = requestAnimationFrame((t) => this.frame(t));
  }

  private frame(now: number): void {
    this.dirty = false;
    if (this.cinePlaying) {
      this.cineStep(now);
      this.requestRender();
    } else if (this.flashSquare && now < this.flashSquare.until) this.requestRender();
    this.draw();
  }

  private buildLut(): void {
    const key = `${this.ww}|${this.wl}|${this.invert}`;
    if (key === this.lutKey) return;
    this.lutKey = key;
    const lo = this.wl - this.ww / 2, sc = 255 / this.ww;
    for (let i = 0; i < 65536; i++) {
      let y = (i - 32768 - lo) * sc;
      y = y < 0 ? 0 : y > 255 ? 255 : y | 0;
      if (this.invert) y = 255 - y;
      this.lut[i] = 0xff000000 | (y << 16) | (y << 8) | y;
    }
  }

  /** Window the frame on screen into the offscreen image (only when something it depends on changed). */
  private compose(): boolean {
    const v = this.vol!;
    const k = this.index;
    const needs = this.dsa ? this.dsa.needs(k) : [k];
    if (!needs.every((j) => v.isReady(j))) {
      v.want(needs);
      return false;
    }
    const dsaKey = this.dsa ? `${this.dsa.mask.join(",")}|${this.dsa.shift.join(",")}` : "";
    const key = `${k}|${this.ww}|${this.wl}|${this.invert}|${dsaKey}|${v.rows}x${v.cols}|${v.version ?? 0}`;
    if (key === this.drawnKey && this.imgData) return true;
    this.drawnKey = key;
    if (!this.imgData || this.imgData.width !== v.cols || this.imgData.height !== v.rows) {
      this.imgData = new ImageData(v.cols, v.rows);
      this.img.width = v.cols;
      this.img.height = v.rows;
    }
    const px = new Uint32Array(this.imgData.data.buffer);
    const rgb = v.rgbFrame(k);
    if (rgb) {
      const lo = this.wl - this.ww / 2, sc = 255 / this.ww;
      const f = (u: number) => {
        let y = (u - lo) * sc;
        y = y < 0 ? 0 : y > 255 ? 255 : y | 0;
        return this.invert ? 255 - y : y;
      };
      const t = new Uint8Array(256).map((_, i) => f(i));
      for (let i = 0, j = 0; i < px.length; i++, j += 3) px[i] = 0xff000000 | (t[rgb[j + 2]] << 16) | (t[rgb[j + 1]] << 8) | t[rgb[j]];
    } else {
      const plane = this.dsa ? this.dsa.subtract(k) : v.frame(k);
      if (plane instanceof Int16Array) {
        this.buildLut();
        const lut = this.lut;
        for (let i = 0; i < px.length; i++) px[i] = lut[plane[i] + 32768];
      } else {
        const lo = this.wl - this.ww / 2, sc = 255 / this.ww, inv = this.invert;
        for (let i = 0; i < px.length; i++) {
          let y = (plane[i] - lo) * sc;
          y = y < 0 ? 0 : y > 255 ? 255 : y | 0;
          if (inv) y = 255 - y;
          px[i] = 0xff000000 | (y << 16) | (y << 8) | y;
        }
      }
    }
    this.img.getContext("2d")!.putImageData(this.imgData, 0, 0);
    return true;
  }

  /** Frames that must be decoded to draw frame k (with DSA on: its mask frames too). */
  needs(k: number): number[] {
    return this.dsa ? this.dsa.needs(k) : [k];
  }

  /** Draw frame k as this view windows it, in a rectangle of another canvas (collage tiles); the view keeps its frame. */
  drawFrame(k: number, ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number): void {
    const old = this.index;
    this.index = k;
    if (this.vol && this.compose()) {
      ctx.drawImage(this.img, x, y, w, h);
      const layer = this.layer?.(k);
      if (layer) ctx.drawImage(layer, x, y, w, h);
    }
    this.index = old;
  }

  private draw(): void {
    const ctx = this.ctx;
    const dpr = window.devicePixelRatio || 1;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    const v = this.vol;
    if (!v) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const ok = this.compose();
    const L = this.layout();
    if (this.imgData && (ok || this.drawnKey)) {
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "high";
      ctx.drawImage(this.img, L.x0, L.y0, v.cols * L.sx * L.scale, v.rows * L.sy * L.scale);
      const layer = this.layer?.(this.index);
      if (layer) ctx.drawImage(layer, L.x0, L.y0, v.cols * L.sx * L.scale, v.rows * L.sy * L.scale);
    }
    this.drawMeasurements(ctx);
    this.drawSquare(ctx);
    this.hooks.overlay?.(ctx);
    if (this.active) {
      const [W, H] = this.cssSize();
      ctx.strokeStyle = "#E8B400";                 // yellow frame of the active viewport (dcv/theme.py)
      ctx.lineWidth = 2;
      ctx.strokeRect(1, 1, W - 2, H - 2);
    }
    this.drawCorners(ctx, ok);
  }

  private roiStats(m: Measurement): ReturnType<typeof ellipseStats> {
    const v = this.vol!;
    const k = this.index;
    if (m.kind !== "ellipse" || v.rgb || !v.isReady(k) || m.pts.length < 2) return null;
    const dsaKey = this.dsa ? `${this.dsa.mask.join(",")}|${this.dsa.shift.join(",")}` : "";
    const key = `${k}|${v.version ?? 0}|${dsaKey}|${m.pts.join(",")}`;
    if (m.stats?.key !== key) {
      const plane = this.dsa ? this.dsa.subtract(k) : v.frame(k);
      m.stats = { key, value: ellipseStats(plane, v.rows, v.cols, m.pts) };
    }
    return m.stats.value;
  }

  private drawMeasurements(ctx: CanvasRenderingContext2D): void {
    const v = this.vol!;
    const list = this.measurements.get(this.mkey()) ?? [];
    const [cw, ch] = this.cssSize();
    const unit = v.modality === "CT" && !this.dsa ? " HU" : "";
    ctx.lineWidth = 1.5;
    ctx.font = "600 13px system-ui, -apple-system, sans-serif";
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
    for (const m of list) {
      const col = m === this.selected ? SELECTED_COLOR : MEASURE_COLOR;
      const W = m.pts.map((q) => this.toScreen(...q));
      ctx.strokeStyle = col;
      ctx.beginPath();
      if (m.kind === "ellipse" && W.length >= 2) {
        ctx.ellipse((W[0][0] + W[1][0]) / 2, (W[0][1] + W[1][1]) / 2, Math.abs(W[1][0] - W[0][0]) / 2, Math.abs(W[1][1] - W[0][1]) / 2, 0, 0, Math.PI * 2);
      } else if (m.kind === "cobb") {
        for (let i = 0; i + 1 < W.length; i += 2) {
          ctx.moveTo(W[i][0], W[i][1]);
          ctx.lineTo(W[i + 1][0], W[i + 1][1]);
        }
      } else W.forEach(([px, py], i) => (i ? ctx.lineTo(px, py) : ctx.moveTo(px, py)));
      ctx.stroke();
      for (const [px, py] of W) {
        ctx.beginPath();
        ctx.arc(px, py, m === this.selected ? 5 : 3, 0, Math.PI * 2);
        ctx.stroke();
      }
      const lines = measureLines(m, v.spacing[1], v.spacing[2], v.spacingKnown, unit, this.roiStats(m));
      if (!lines.length) continue;
      const lh = 16;
      const w = Math.max(...lines.map((t) => ctx.measureText(t).width)) + 8, h = lines.length * lh + 4;
      const xs = W.map((q) => q[0]), ys = W.map((q) => q[1]);
      let tx = Math.max(...xs) + 10;
      if (tx + w > cw - 2) tx = Math.max(2, Math.min(...xs) - w - 10);
      if (tx + w > cw - 2) tx = Math.max(2, cw - w - 2);
      const ty = Math.max(2, Math.min(ch - h - 2, (Math.min(...ys) + Math.max(...ys)) / 2 - h / 2));
      ctx.fillStyle = "rgba(0,0,0,0.6)";
      ctx.fillRect(tx, ty, w, h);
      lines.forEach((t, i) => shadowText(ctx, t, tx + 4, ty + 14 + i * lh, col));
    }
  }

  private drawSquare(ctx: CanvasRenderingContext2D): void {
    let at: [number, number] | null = null;
    if (this.flashSquare && performance.now() < this.flashSquare.until) at = [this.flashSquare.r, this.flashSquare.c];
    else if (this.tool === "dsaAuto" && this.hover && this.dsaAvailable()) at = this.hover;
    if (!at || !this.vol) return;
    const hs = Math.max(16, Math.floor(Math.min(this.vol.rows, this.vol.cols) / 10));
    const [x0, y0] = this.toScreen(at[0] - hs, at[1] - hs);
    const [x1, y1] = this.toScreen(at[0] + hs, at[1] + hs);
    ctx.strokeStyle = SQUARE_COLOR;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
  }

  private drawCorners(ctx: CanvasRenderingContext2D, ok: boolean): void {
    const v = this.vol!;
    const [W, H] = this.cssSize();
    const m = 8, lh = 16;
    ctx.font = "13px system-ui, -apple-system, sans-serif";
    const tl = [v.patient, v.description].filter(Boolean);
    if (this.cinePlaying) tl.push(`Cine: ${this.fps} fps`);
    const tr: string[] = [];
    if (this.dsa) tr.push("DSA (subtraction)");
    if (!v.complete) tr.push(`Decoding ${v.readyCount}/${v.n}`);
    if (!ok) tr.push(`Loading image ${this.index + 1}…`);
    if (this.busy) tr.push(this.busy);
    if (v.errors.length) tr.push(`${v.errors.length} image(s) could not be read`);
    const bl = [`WW: ${Math.round(this.ww)}  WL: ${Math.round(this.wl)}${this.invert ? "  (invert)" : ""}`];
    if (this.dsa) bl.push(...this.dsa.label());
    const br = [`Zoom: ${Math.round(this.zoom * 100)}%`, `Img: ${this.index + 1}/${v.n}`];
    ctx.textBaseline = "top";
    tl.forEach((t, i) => shadowText(ctx, t, m, m + i * lh, "#EBEBEB"));
    ctx.textAlign = "right";
    tr.forEach((t, i) => shadowText(ctx, t, W - m, m + i * lh, "#EBEBEB"));
    ctx.textBaseline = "bottom";
    [...br].reverse().forEach((t, i) => shadowText(ctx, t, W - m, H - m - i * lh, "#EBEBEB"));
    ctx.textAlign = "left";
    [...bl].reverse().forEach((t, i) => shadowText(ctx, t, m, H - m - i * lh, "#EBEBEB"));
    ctx.textBaseline = "alphabetic";
  }
}

function shadowText(ctx: CanvasRenderingContext2D, t: string, x: number, y: number, color: string): void {
  ctx.fillStyle = "rgba(0,0,0,0.85)";
  ctx.fillText(t, x + 1, y + 1);
  ctx.fillStyle = color;
  ctx.fillText(t, x, y);
}

function distToSegment(p: [number, number], a: [number, number], b: [number, number]): number {
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const l2 = dx * dx + dy * dy;
  const t = l2 ? Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2)) : 0;
  return Math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy);
}
