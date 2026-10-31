/**
 * The 3D view: camera, touch / mouse interaction and the state of the rendering (preset, gain, blend mode,
 * quality, bone lock), for one volume at a time. Rendering is lighter while the view is being turned (half
 * resolution, longer ray steps) and full once it settles, like the desktop's interactive / still update rates.
 */
import type { Volume } from "../dicom/volume";
import { add, cross, mulMtV, rot, mulMV, scale, sub, unit, type V3 } from "../geometry";
import { boneRegion, BONE_OFFSET, CT_FLOOR, gridStep, VEIN_OFFSET, vesselMasks, type BoneRegion, type VesselMasks } from "./bone";
import { cutVoxels, fillLasso } from "./cut";
import { pickSurface } from "./pick";
import { fmtLen } from "../view/measure";
import { contentBox, VolumeRenderer, type Box, type SegmentMask } from "./renderer";
import { buildTable, builtinPresets, copyTF, genericPreset, RIM_GRAD_MAX, usesVeins, type TransferFunction } from "./tf";

// ------------------------------------------------------------------ 4×4 matrices (column-major, as WebGL wants)
function mul4(a: Float32Array, b: Float32Array): Float32Array {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    let s = 0;
    for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
    o[c * 4 + r] = s;
  }
  return o;
}

function inv4(m: Float32Array): Float32Array {
  const a = Array.from(m), o = new Float32Array(16);
  const [a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33] = a;
  const b00 = a00 * a11 - a01 * a10, b01 = a00 * a12 - a02 * a10, b02 = a00 * a13 - a03 * a10, b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11, b05 = a02 * a13 - a03 * a12, b06 = a20 * a31 - a21 * a30, b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30, b09 = a21 * a32 - a22 * a31, b10 = a21 * a33 - a23 * a31, b11 = a22 * a33 - a23 * a32;
  const det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06 || 1e-12;
  const d = 1 / det;
  o[0] = (a11 * b11 - a12 * b10 + a13 * b09) * d; o[1] = (a02 * b10 - a01 * b11 - a03 * b09) * d;
  o[2] = (a31 * b05 - a32 * b04 + a33 * b03) * d; o[3] = (a22 * b04 - a21 * b05 - a23 * b03) * d;
  o[4] = (a12 * b08 - a10 * b11 - a13 * b07) * d; o[5] = (a00 * b11 - a02 * b08 + a03 * b07) * d;
  o[6] = (a32 * b02 - a30 * b05 - a33 * b01) * d; o[7] = (a20 * b05 - a22 * b02 + a23 * b01) * d;
  o[8] = (a10 * b10 - a11 * b08 + a13 * b06) * d; o[9] = (a01 * b08 - a00 * b10 - a03 * b06) * d;
  o[10] = (a30 * b04 - a31 * b02 + a33 * b00) * d; o[11] = (a21 * b02 - a20 * b04 - a23 * b00) * d;
  o[12] = (a11 * b07 - a10 * b09 - a12 * b06) * d; o[13] = (a00 * b09 - a01 * b07 + a02 * b06) * d;
  o[14] = (a31 * b01 - a30 * b03 - a32 * b00) * d; o[15] = (a20 * b03 - a21 * b01 + a22 * b00) * d;
  return o;
}

/** Orbit camera around a centre (mm, texture frame). right / up / back are the camera axes in that frame. */
export class Camera {
  center: V3 = [0, 0, 0];
  dist = 500;
  right: V3 = [1, 0, 0];
  up: V3 = [0, 1, 0];
  back: V3 = [0, 0, 1];
  radius = 200;
  fov = (30 * Math.PI) / 180;

  fit(): void {
    this.dist = (this.radius / Math.sin(this.fov / 2)) * 1.02;
  }

  set(back: V3, up: V3): void {
    this.back = unit(back);
    this.right = unit(cross(up, this.back));
    this.up = cross(this.back, this.right);
  }

  orbit(ax: number, ay: number): void {
    const q1 = rot(this.up, -ax), q2 = rot(this.right, -ay);
    for (const q of [q1, q2]) {
      this.right = unit(mulMV(q, this.right));
      this.up = unit(mulMV(q, this.up));
      this.back = unit(mulMV(q, this.back));
    }
  }

  /** Turn the image in the screen plane (two-finger twist). */
  roll(a: number): void {
    const q = rot(this.back, a);
    this.right = unit(mulMV(q, this.right));
    this.up = unit(mulMV(q, this.up));
  }

  pan(dx: number, dy: number, heightPx: number): void {
    const mmPerPx = (2 * this.dist * Math.tan(this.fov / 2)) / heightPx;
    this.center = add(this.center, add(scale(this.right, -dx * mmPerPx), scale(this.up, dy * mmPerPx)));
  }

  invVP(aspect: number): Float32Array {
    return inv4(this.vp(aspect));
  }

  /** Forward view-projection (mm → clip), column-major. */
  vp(aspect: number): Float32Array {
    const eye = add(this.center, scale(this.back, this.dist));
    const v = new Float32Array([
      this.right[0], this.up[0], this.back[0], 0,
      this.right[1], this.up[1], this.back[1], 0,
      this.right[2], this.up[2], this.back[2], 0,
      -dotv(this.right, eye), -dotv(this.up, eye), -dotv(this.back, eye), 1,
    ]);
    const near = Math.max(1, this.dist - this.radius * 1.2), far = this.dist + this.radius * 1.2;
    const f = 1 / Math.tan(this.fov / 2);
    const p = new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) / (near - far), -1, 0, 0, (2 * far * near) / (near - far), 0]);
    return mul4(p, v);
  }

  state(): CameraState {
    return { center: [...this.center], dist: this.dist, right: [...this.right], up: [...this.up], back: [...this.back] };
  }

  restore(c: CameraState): void {
    this.center = [...c.center];
    this.dist = c.dist;
    this.right = [...c.right];
    this.up = [...c.up];
    this.back = [...c.back];
  }
}

export interface CameraState {
  center: V3;
  dist: number;
  right: V3;
  up: V3;
  back: V3;
}

/** What a 3D snapshot keeps (like dcv/snapshots.py): camera, crop box, colours, mode. */
export interface Snapshot {
  camera: CameraState;
  crop: { on: boolean; lo: V3; hi: V3 };
  tf: TransferFunction;
  gain: number;
  blend: Blend;
  thumb: string;                 // small PNG data URL (kept in memory only)
}

const FACES: [number, 0 | 1][] = [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]];   // (axis, low / high side)

export interface Measure3D {
  a: V3;
  b: V3;
}

const dotv = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

/** Camera directions of the desktop's views (patient LPS): from the centre to the camera, and up. */
export const VIEWS: Record<string, [string, V3, V3]> = {
  anterior: ["Anterior", [0, -1, 0], [0, 0, 1]], posterior: ["Posterior", [0, 1, 0], [0, 0, 1]],
  right: ["Right", [-1, 0, 0], [0, 0, 1]], left: ["Left", [1, 0, 0], [0, 0, 1]],
  superior: ["Superior", [0, 0, 1], [0, -1, 0]], inferior: ["Inferior", [0, 0, -1], [0, -1, 0]],
};

export type Blend = 0 | 1 | 2;
export type Resolution = "auto" | "full" | "half";

const MOBILE = /iPad|iPhone|iPod/.test(navigator.userAgent) || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1)
  || matchMedia("(pointer: coarse)").matches;

/**
 * Bytes of volume texture this device is trusted with at full resolution. Browsers on iOS do not tell the memory;
 * a tab there gets more than 1 GB on any iPhone / iPad of the last years, and the decoded series is in it as well.
 */
export function gpuLimitBytes(): number {
  const ios = /iPad|iPhone|iPod/.test(navigator.userAgent) || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
  if (ios) return Math.min(screen.width, screen.height) < 600 ? 350e6 : 600e6;
  const gb = (navigator as Navigator & { deviceMemory?: number }).deviceMemory ?? 8;
  return gb <= 2 ? 200e6 : gb <= 4 ? 500e6 : 1500e6;
}

/** In-plane step for a choice: Auto is full unless the texture would pass the device's limit, or memory ran out before. */
export function resolutionFactor(choice: Resolution, textureBytes: number, limit: number, ranOut: boolean): 1 | 2 {
  if (choice !== "auto") return choice === "half" ? 2 : 1;
  return ranOut || textureBytes > limit ? 2 : 1;
}
export const QUALITY: Record<string, number> = { Fast: 1.0, Normal: 0.6, High: 0.35 };   // ray step × smallest spacing

export class VRView {
  readonly renderer: VolumeRenderer;
  readonly cam = new Camera();
  vol: Volume | null = null;
  tf!: TransferFunction;
  gain = 1;
  blend: Blend = 0;
  quality: keyof typeof QUALITY = "Normal";
  window: [number, number] = [-1024, 3071];
  boneLock = true;
  boneHu = 500;
  resolution: 1 | 2 = 1;
  ct = false;
  lo = 0;
  hi = 1;
  hist: Float32Array | null = null;
  private boneOn = false;
  private bone: BoneRegion | null = null;
  private boneKey = "";
  /** Arteries / veins, when the curves have a vein layer; `arteryHu` overrides the automatic arterial core. */
  private vessels: VesselMasks | null = null;
  private vesselKey = "";
  arteryHu: number | null = null;
  /** The arterial core in use, for the panel (null = no vein layer). */
  get arteryCore(): number | null {
    return this.renderer.vessels?.coreHu ?? null;
  }
  /** Crop box in texture coordinates; `crop.on` applies it, `cropEdit` shows its handles. */
  crop = { on: false, lo: [0, 0, 0] as V3, hi: [1, 1, 1] as V3 };
  cropEdit = false;
  /** Scalpel: removed voxels (bitset over the texture grid) and earlier states for undo. */
  removed: Uint32Array | null = null;
  /** Undo: each entry = the bits a cut changed (XOR), deflated: a few kB instead of ~10 MB per level. */
  private undoStack: Promise<Uint8Array>[] = [];
  private box: Box | null = null;
  /** Share of the view's pixels rendered while turning; adapts to keep frames near 30 per second. */
  private iscale = 0.5;
  private lastDraw = 0;
  /** "rotate" = one finger turns; "pan" = one finger moves the model; "scalpel" = the next stroke draws a lasso and
   *  cuts; "measure" = the next stroke draws a length between two points of the surface. */
  tool: "rotate" | "pan" | "scalpel" | "measure" = "rotate";
  /** Lengths between surface points, in mm of the volume: real 3D, so they follow every turn. */
  measures: Measure3D[] = [];
  selectedMeasure: Measure3D | null = null;
  private measureDrag: { m: Measure3D; end: "a" | "b"; created: boolean; orig: [V3, V3] } | null = null;
  private pickTable: { alpha: Float32Array; w: number; h: number; x0: number; x1: number } | null = null;
  /** Milliseconds of the last surface pick (for the report in the about box / checks). */
  lastPickMs = 0;
  private segMasks: SegmentMask[] = [];
  /** Auto / Full / Half, and whether Auto must stay at half because memory ran out earlier. */
  resChoice: Resolution = "auto";
  ranOut = false;
  /** The browser took the graphics memory away: this view is finished. */
  onLost: () => void = () => undefined;
  cutOutside = false;
  snapshots: Snapshot[] = [];
  private lasso: [number, number][] | null = null;
  private cropDrag: { axis: number; side: 0 | 1; x: number; y: number } | null = null;
  private overlay: HTMLCanvasElement | null = null;
  onStatus: (t: string) => void = () => undefined;
  /** The scalpel finished a cut (the tool releases itself, like the desktop). */
  onToolDone: () => void = () => undefined;
  private tfCt: boolean | null = null;
  private interactive = false;
  private idleTimer = 0;
  private dirty = false;
  private pointers = new Map<number, { x: number; y: number }>();
  private pinch: { d: number; mx: number; my: number; a: number } | null = null;
  private dragBtn = 0;
  private lastTap = { t: 0, x: 0, y: 0 };
  onChange: () => void = () => undefined;
  lastFrameMs = 0;

  constructor(readonly canvas: HTMLCanvasElement) {
    this.renderer = new VolumeRenderer(canvas);
    this.renderer.onLost = () => this.onLost();
    this.overlay = document.createElement("canvas");
    this.overlay.className = "vr-overlay";
    canvas.parentElement!.append(this.overlay);
    new ResizeObserver(() => this.requestRender()).observe(canvas.parentElement!);
    canvas.addEventListener("pointerdown", (e) => this.down(e));
    canvas.addEventListener("pointermove", (e) => this.move(e));
    canvas.addEventListener("pointerup", (e) => this.up(e));
    canvas.addEventListener("pointercancel", (e) => this.up(e));
    canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      this.cam.dist *= Math.exp(e.deltaY * (e.deltaMode ? 0.05 : 0.0015));
      this.touched();
    }, { passive: false });
    canvas.addEventListener("dblclick", () => this.tool === "rotate" && this.view("anterior"));
    canvas.addEventListener("contextmenu", (e) => e.preventDefault());
    for (const t of ["gesturestart", "gesturechange", "gestureend"]) canvas.addEventListener(t, (e) => e.preventDefault());
  }

  /** Upload a (fully decoded) volume, at the resolution of `resChoice` (half = ¼ of the GPU memory). */
  open(vol: Volume): void {
    if (this.vol !== vol) {
      this.removed = null;
      this.box = null;
      this.measures = [];
      this.selectedMeasure = null;
    }
    this.vol = vol;
    this.ct = vol.modality === "CT";
    const [lo, hi] = vol.valueRange();
    this.lo = this.ct ? Math.max(lo, CT_FLOOR) : lo;
    this.hi = Math.max(hi, this.lo + 1);
    this.window = vol.ww && vol.wl !== null ? [vol.wl - vol.ww / 2, vol.wl + vol.ww / 2] : [this.lo, this.hi];
    if (!this.tf || this.tfCt !== this.ct) {        // CT: the vessels + bone preset; MR / other: one curve
      this.tf = copyTF(this.ct ? builtinPresets()[1] : genericPreset(this.lo, this.hi));
      this.tfCt = this.ct;
    }
    this.hist = histogram(vol, this.lo, this.hi);
    this.box ??= contentBox(vol, this.ct);
    const f = this.factor();
    if (this.resolution !== f) this.removed = null;
    this.resolution = f;
    this.bone = null;
    this.boneKey = "";
    this.vessels = null;
    this.vesselKey = "";
    this.undoStack = [];
    this.crop = { on: false, lo: [0, 0, 0], hi: [1, 1, 1] };
    this.cropEdit = false;
    this.snapshots = [];
    this.tool = "rotate";
    this.upload();
    const [sz, sy, sx] = vol.spacing;
    const size: V3 = [vol.cols * sx, vol.rows * sy, vol.n * sz];
    this.cam.center = scale(size, 0.5);
    this.cam.radius = Math.hypot(...size) / 2;
    this.view("anterior");
  }

  /** (Re)upload with the bone lock state; falls back to half resolution when the GPU refuses the full volume. */
  upload(): void {
    const vol = this.vol!;
    this.boneOn = this.ct && this.boneLock;
    const key = `${this.boneHu}`;
    if (this.boneOn && (!this.bone || this.boneKey !== key)) {
      this.bone = boneRegion(vol, this.boneHu, 3, 1.5, gridStep(vol.rows * vol.cols * vol.n, MOBILE));
      this.boneKey = key;
    }
    const bone = this.boneOn ? this.bone : null;
    this.syncVessels();
    const before = this.resolution;
    this.box ??= contentBox(vol, this.ct);
    let ok = this.renderer.upload(vol, this.resolution, bone, this.ct, this.removed, this.box);
    if (!ok && this.resolution === 1) {
      this.resolution = 2;
      this.removed = null;                       // the cut grid belongs to the other resolution
      ok = this.renderer.upload(vol, 2, bone, this.ct, null, this.box);
      if (ok) this.onStatus("Not enough graphics memory for full resolution: showing half resolution");
    }
    if (this.resolution !== before) this.undoStack = [];
    if (!ok) throw new Error("the graphics memory cannot hold this volume");
    this.applyTF(false);
  }

  get gpuBytes(): number {
    return this.renderer.gpuBytes;
  }

  /** Bring the artery / vein masks in line with the curves. True when what the texture should hold changed. */
  private syncVessels(): boolean {
    const vol = this.vol!;
    const want = this.ct && usesVeins(this.tf);
    const before = this.renderer.vessels;
    if (!want) {
      this.renderer.vessels = null;
      return before !== null;
    }
    const key = `${this.boneOn ? this.boneKey : "off"}|${this.arteryHu}`;
    if (!this.vessels || this.vesselKey !== key) {
      this.vessels = vesselMasks(vol, this.boneOn ? this.bone : null, this.arteryHu, gridStep(vol.rows * vol.cols * vol.n, MOBILE));
      this.vesselKey = key;
    }
    this.renderer.vessels = this.vessels;
    return before !== this.vessels;
  }

  /** The value axis of the stored volume, and its table: wider with the veins' domain far above the others. */
  private tableSpec(width: number) {
    const bone = this.boneOn, vein = this.renderer.vessels !== null;
    const x0 = bone ? -BONE_OFFSET - this.hi : this.lo, x1 = vein ? VEIN_OFFSET + 3000 : this.hi;
    return { x0, x1, width, lo: this.lo, bone, boneOffset: BONE_OFFSET, vein, veinOffset: VEIN_OFFSET, gain: this.gain };
  }

  /** Another arterial core (null = automatic). */
  setArteryHu(hu: number | null): void {
    this.arteryHu = hu;
    this.applyTF(false);
    this.onChange();
  }

  applyTF(fast: boolean): void {
    if (!this.vol) return;
    if (this.syncVessels()) this.renderer.fill(this.vol, this.boneOn ? this.bone : null, this.ct, this.removed, 0, this.renderer.dims[2]);
    const vein = this.renderer.vessels !== null;
    const limit = this.renderer.maxTexture2D;
    const width = Math.min(limit, vein ? (fast ? 2048 : 8192) : fast ? 1024 : 4096);
    const { x0, x1 } = this.tableSpec(width);
    const t = buildTable(this.tf, this.tableSpec(width), fast ? 32 : 64);
    this.pickTable = null;
    this.renderer.setTable(t.data, width, t.rows, x0, x1);
    this.requestRender();
  }

  private factor(): 1 | 2 {
    const b = this.box!;
    return resolutionFactor(this.resChoice, (b.x1 - b.x0) * (b.y1 - b.y0) * (b.z1 - b.z0) * 2, gpuLimitBytes(), this.ranOut);
  }

  /** Auto / Full / Half (cuts are kept only on the grid they were made on). */
  setResChoice(choice: Resolution): void {
    this.resChoice = choice;
    if (!this.vol) return;
    const f = this.factor();
    if (f === this.resolution) return;
    if (this.hasCuts) this.onStatus("The scalpel cuts were made at the other resolution and are cleared");
    this.resolution = f;
    this.removed = null;
    this.undoStack = [];
    this.upload();
  }

  // ------------------------------------------------------------ crop box
  setCrop(on: boolean): void {
    this.crop.on = on;
    this.cropEdit = on;
    this.requestRender();
    this.onChange();
  }

  /** Show / hide the box and its handles; the crop itself stays. */
  setCropBox(show: boolean): void {
    if (show && !this.crop.on) this.crop.on = true;
    this.cropEdit = show && this.crop.on;
    this.requestRender();
    this.onChange();
  }

  resetCrop(): void {
    this.crop.lo = [0, 0, 0];
    this.crop.hi = [1, 1, 1];
    this.requestRender();
    this.onChange();
  }

  private screenOf(tc: V3, W: number, H: number): [number, number] | null {
    const s = this.renderer.full;
    return this.screenOfMm([tc[0] * s[0], tc[1] * s[1], tc[2] * s[2]], W, H);
  }

  private screenOfMm(p: V3, W: number, H: number): [number, number] | null {
    const m = this.cam.vp(W / H);
    const [X, Y, Z] = p;
    const cw = m[3] * X + m[7] * Y + m[11] * Z + m[15];
    if (cw <= 0) return null;
    const cx = (m[0] * X + m[4] * Y + m[8] * Z + m[12]) / cw, cy = (m[1] * X + m[5] * Y + m[9] * Z + m[13]) / cw;
    return [((cx + 1) / 2) * W, ((1 - cy) / 2) * H];
  }

  private faceCenter(axis: number, side: 0 | 1): V3 {
    const c: V3 = [(this.crop.lo[0] + this.crop.hi[0]) / 2, (this.crop.lo[1] + this.crop.hi[1]) / 2, (this.crop.lo[2] + this.crop.hi[2]) / 2];
    c[axis] = side ? this.crop.hi[axis] : this.crop.lo[axis];
    return c;
  }

  private hitFace(x: number, y: number, touch: boolean): { axis: number; side: 0 | 1 } | null {
    if (!this.cropEdit) return null;
    const r = this.canvas.getBoundingClientRect();
    let best: { axis: number; side: 0 | 1; d: number } | null = null;
    for (const [axis, side] of FACES) {
      const p = this.screenOf(this.faceCenter(axis, side), r.width, r.height);
      if (!p) continue;
      const d = Math.hypot(p[0] - x, p[1] - y);
      if (d < (touch ? 26 : 12) && (!best || d < best.d)) best = { axis, side, d };
    }
    return best;
  }

  /** Move a crop face with the pointer: along the screen direction of its axis. */
  private dragFace(x: number, y: number): void {
    const d = this.cropDrag!;
    const r = this.canvas.getBoundingClientRect();
    const c = this.faceCenter(d.axis, d.side);
    const c2: V3 = [...c];
    c2[d.axis] += 0.1;
    const a = this.screenOf(c, r.width, r.height), b = this.screenOf(c2, r.width, r.height);
    if (!a || !b) return;
    const vx = b[0] - a[0], vy = b[1] - a[1], l2 = vx * vx + vy * vy;
    if (l2 < 1) return;                                    // the axis points at the viewer: no useful direction
    const dt = (((x - d.x) * vx + (y - d.y) * vy) / l2) * 0.1;
    const lo = this.crop.lo, hi = this.crop.hi, k = d.axis;
    if (d.side) hi[k] = Math.min(1, Math.max(lo[k] + 0.02, hi[k] + dt));
    else lo[k] = Math.max(0, Math.min(hi[k] - 0.02, lo[k] + dt));
    d.x = x;
    d.y = y;
    this.touched();
  }

  // ------------------------------------------------------------ 3D measurement
  /** The surface under a point of the view, in mm of the volume; null = nothing visible there. */
  pick(x: number, y: number): V3 | null {
    const vol = this.vol;
    if (!vol) return null;
    const t0 = performance.now();
    const r = this.renderer;
    const b = this.canvas.getBoundingClientRect();
    if (!this.pickTable) {
      // the opacity only, at low resolution: ~130 kB, built when a measurement starts and after a curve changed
      const w = r.vessels ? 4096 : 1024;
      const spec = this.tableSpec(w);
      const { x0, x1 } = spec;
      const t = buildTable(this.tf, spec, 32);
      const alpha = new Float32Array(w * t.rows);
      for (let i = 0; i < alpha.length; i++) alpha[i] = t.data[i * 4 + 3];
      this.pickTable = { alpha, w, h: t.rows, x0, x1 };
    }
    const T = this.pickTable;
    const inv = this.cam.invVP(b.width / b.height);
    const nx = (x / b.width) * 2 - 1, ny = 1 - (y / b.height) * 2;
    const un = (z: number): V3 => {
      const w = inv[3] * nx + inv[7] * ny + inv[11] * z + inv[15];
      return [(inv[0] * nx + inv[4] * ny + inv[8] * z + inv[12]) / w, (inv[1] * nx + inv[5] * ny + inv[9] * z + inv[13]) / w,
        (inv[2] * nx + inv[6] * ny + inv[10] * z + inv[14]) / w];
    };
    const ro = un(-1), far = un(1);
    const rd = unit(sub(far, ro));
    const F = r.full;
    const lo: V3 = this.crop.on ? [this.crop.lo[0] * F[0], this.crop.lo[1] * F[1], this.crop.lo[2] * F[2]] : [0, 0, 0];
    const hi: V3 = this.crop.on ? [this.crop.hi[0] * F[0], this.crop.hi[1] * F[1], this.crop.hi[2] * F[2]] : [...F];
    const p = pickSurface({
      data: vol.data, cols: vol.cols, rows: vol.rows, dims: r.dims, start: r.start, f: r.f, spacing: r.spacing, origin: r.origin,
      clipLo: lo, clipHi: hi, bone: this.boneOn ? this.bone : null, vessels: r.vessels, ct: this.ct, removed: this.removed,
      air: this.ct ? CT_FLOOR : vol.valueRange()[0], blend: this.blend, alpha: T.alpha, alphaW: T.w, alphaH: T.h, x0: T.x0, x1: T.x1,
      gradMax: RIM_GRAD_MAX, solid: this.segMasks.length ? (px, py, pz) => this.onSegment(px, py, pz) : undefined,
    }, ro, rd);
    this.lastPickMs = performance.now() - t0;
    return p;
  }

  private onSegment(px: number, py: number, pz: number): boolean {
    const [sz, sy, sx] = this.vol!.spacing;
    const X = Math.floor(px / sx), Y = Math.floor(py / sy), Z = Math.floor(pz / sz);
    for (const m of this.segMasks) {
      const z = Z - m.off[0], y = Y - m.off[1], x = X - m.off[2];
      if (z >= 0 && y >= 0 && x >= 0 && z < m.dims[0] && y < m.dims[1] && x < m.dims[2] && m.sub[(z * m.dims[1] + y) * m.dims[2] + x]) return true;
    }
    return false;
  }

  /** The segments to show; `fast` = while a range is being dragged (no surface smoothing). */
  setSegments(list: SegmentMask[], fast = false): void {
    this.segMasks = list;
    if (!this.vol) return;
    const [sz, sy, sx] = this.vol.spacing;
    this.renderer.setSegments(list, [sz, sy, sx], fast);
    this.requestRender();
  }

  private hitMeasure(x: number, y: number, touch: boolean): { m: Measure3D; end: "a" | "b" } | null {
    const b = this.canvas.getBoundingClientRect();
    const tol = touch ? 24 : 10;
    for (const m of [...this.measures].reverse()) {
      for (const end of ["a", "b"] as const) {
        const p = this.screenOfMm(m[end], b.width, b.height);
        if (p && Math.hypot(p[0] - x, p[1] - y) < tol) return { m, end };
      }
    }
    return null;
  }

  /** Delete the selected measurement, or the last one. */
  deleteMeasure(): void {
    const m = this.selectedMeasure ?? this.measures[this.measures.length - 1];
    if (!m) return;
    this.measures.splice(this.measures.indexOf(m), 1);
    this.selectedMeasure = null;
    this.drawOverlay();
    this.onChange();
  }

  // ------------------------------------------------------------ scalpel
  setTool(t: VRView["tool"]): void {
    this.tool = t;
    this.lasso = null;
    this.drawOverlay();
    this.onChange();
  }

  /** Cut with the lasso drawn in CSS pixels of the view (the whole depth, like the desktop scalpel). */
  cut(points: [number, number][]): number {
    if (!this.vol || points.length < 3) return 0;
    const r = this.canvas.getBoundingClientRect();
    const W = r.width, H = r.height;
    const k = Math.min(1, 640 / Math.max(W, H));             // the lasso grid: at most 640 px across
    const mw = Math.max(1, Math.round(W * k)), mh = Math.max(1, Math.round(H * k));
    const L = fillLasso(points.map(([x, y]) => [x * k, y * k] as [number, number]), mw, mh);
    const [nx, ny, nz] = this.renderer.dims;
    const words = Math.ceil((nx * ny * nz) / 32);
    if (!this.removed) this.removed = new Uint32Array(words);
    const keep = this.removed.slice();
    const t0 = performance.now();
    const res = cutVoxels(this.removed, this.renderer.dims, this.renderer.spacing, this.cam.vp(W / H), L, this.cutOutside,
      this.renderer.origin);
    if (!res.count) return 0;
    for (let i = 0; i < keep.length; i++) keep[i] ^= this.removed[i];          // the bits this cut set
    this.undoStack.push(deflate(keep));
    while (this.undoStack.length > 10) this.undoStack.shift();
    this.renderer.fill(this.vol, this.boneOn ? this.bone : null, this.ct, this.removed, res.z0, res.z1);
    this.onStatus(`Cut ${res.count.toLocaleString()} voxels (${Math.round(performance.now() - t0)} ms)`);
    this.requestRender();
    this.onChange();
    return res.count;
  }

  get canUndo(): boolean {
    return this.undoStack.length > 0;
  }

  get hasCuts(): boolean {
    return !!this.removed && this.removed.some((w) => w !== 0);
  }

  /** Undo the last cut (or reset): XOR its bits back, upload only the slices they touch. */
  async undoCut(): Promise<void> {
    const entry = this.undoStack.pop();
    if (!entry || !this.vol) return;
    const delta = new Uint32Array((await inflate(entry)).buffer);
    if (!this.vol) return;
    const [nx, ny] = this.renderer.dims;
    this.removed ??= new Uint32Array(delta.length);
    let w0 = -1, w1 = -1;
    for (let i = 0; i < delta.length; i++) {
      if (!delta[i]) continue;
      this.removed[i] ^= delta[i];
      if (w0 < 0) w0 = i;
      w1 = i;
    }
    if (w0 >= 0) {
      const slice = nx * ny;
      this.renderer.fill(this.vol, this.boneOn ? this.bone : null, this.ct, this.removed, Math.floor((w0 * 32) / slice),
        Math.min(this.renderer.dims[2], Math.floor((w1 * 32 + 31) / slice) + 1));
    }
    this.requestRender();
    this.onChange();
  }

  resetCuts(): void {
    if (!this.removed || !this.vol) return;
    this.undoStack.push(deflate(this.removed));            // XOR with nothing = everything that was cut
    this.removed = null;
    this.renderer.fill(this.vol, this.boneOn ? this.bone : null, this.ct, null, 0, this.renderer.dims[2]);
    this.requestRender();
    this.onChange();
  }

  // ------------------------------------------------------------ snapshots
  /** Keep the current view (camera, crop box, colours, mode) with a thumbnail. */
  addSnapshot(): Snapshot {
    this.draw();                                            // the frame on the canvas is the still one
    const t = document.createElement("canvas");
    const s = 96 / Math.max(this.canvas.width, this.canvas.height);
    t.width = Math.max(1, Math.round(this.canvas.width * s));
    t.height = Math.max(1, Math.round(this.canvas.height * s));
    t.getContext("2d")!.drawImage(this.canvas, 0, 0, t.width, t.height);
    const snap: Snapshot = {
      camera: this.cam.state(), crop: { on: this.crop.on, lo: [...this.crop.lo], hi: [...this.crop.hi] },
      tf: copyTF(this.tf), gain: this.gain, blend: this.blend, thumb: t.toDataURL("image/png"),
    };
    this.snapshots.push(snap);
    this.onChange();
    return snap;
  }

  applySnapshot(s: Snapshot): void {
    this.cam.restore(s.camera);
    this.crop = { on: s.crop.on, lo: [...s.crop.lo], hi: [...s.crop.hi] };
    this.cropEdit = false;
    this.tf = copyTF(s.tf);
    this.gain = s.gain;
    this.blend = s.blend;
    this.applyTF(false);
    this.onChange();
  }

  /** The still 3D view as a PNG (full quality frame with the measurements, without the crop handles). */
  async png(): Promise<Blob | null> {
    this.interactive = false;
    this.draw();
    let out = this.canvas;
    if (this.measures.length) {
      const b = this.canvas.getBoundingClientRect();
      out = document.createElement("canvas");
      out.width = this.canvas.width;
      out.height = this.canvas.height;
      const ctx = out.getContext("2d")!;
      ctx.drawImage(this.canvas, 0, 0);
      ctx.scale(out.width / b.width, out.height / b.height);
      this.drawMeasures(ctx, b.width, b.height);
    }
    return new Promise((res) => out.toBlob(res, "image/png"));
  }

  setPreset(tf: TransferFunction): void {
    this.tf = copyTF(tf);
    this.applyTF(false);
    this.onChange();
  }

  view(name: string): void {
    const v = VIEWS[name];
    if (!v || !this.vol) return;
    const D = this.vol.direction;
    this.cam.set(mulMtV(D, v[1]), mulMtV(D, v[2]));       // patient directions into the image frame
    const [sz, sy, sx] = this.vol.spacing;
    this.cam.center = scale([this.vol.cols * sx, this.vol.rows * sy, this.vol.n * sz], 0.5);
    this.cam.fit();
    this.requestRender();
  }

  // ------------------------------------------------------------ interaction
  private pos(e: PointerEvent): [number, number] {
    const b = this.canvas.getBoundingClientRect();
    return [e.clientX - b.left, e.clientY - b.top];
  }

  private touched(): void {
    this.interactive = true;
    clearTimeout(this.idleTimer);
    this.idleTimer = window.setTimeout(() => {
      this.interactive = false;
      this.requestRender();
    }, 140);
    this.requestRender();
  }

  private down(e: PointerEvent): void {
    e.preventDefault();
    try {
      this.canvas.setPointerCapture(e.pointerId);
    } catch { /* synthetic pointer */ }
    const [x, y] = this.pos(e);
    this.pointers.set(e.pointerId, { x, y });
    this.dragBtn = e.pointerType === "mouse" ? (e.shiftKey ? 1 : e.button) : 0;
    if (this.pointers.size === 1 && this.dragBtn === 0) {
      const f = this.hitFace(x, y, e.pointerType !== "mouse");
      if (f) {
        this.cropDrag = { ...f, x, y };
        return;
      }
      if (this.tool !== "scalpel") {
        const h = this.hitMeasure(x, y, e.pointerType !== "mouse");
        if (h) {
          this.selectedMeasure = h.m;
          this.measureDrag = { ...h, created: false, orig: [[...h.m.a], [...h.m.b]] };
          this.drawOverlay();
          this.onChange();
          return;
        }
      }
      if (this.tool === "scalpel") {
        this.lasso = [[x, y]];
        return;
      }
      if (this.tool === "measure") {
        const p = this.pick(x, y);
        if (!p) {
          this.onStatus("Nothing visible here to measure on");
          this.dragBtn = -1;                               // no turning from this press
          return;
        }
        const m: Measure3D = { a: p, b: [...p] };
        this.measures.push(m);
        this.selectedMeasure = m;
        this.measureDrag = { m, end: "b", created: true, orig: [[...p], [...p]] };
        this.drawOverlay();
        return;
      }
    }
    if (this.pointers.size === 2) {
      this.cropDrag = null;
      this.cancelMeasureDrag();
      this.lasso = null;                                   // a second finger: zoom / pan instead of cutting
      this.drawOverlay();
      const [p, q] = [...this.pointers.values()];
      this.pinch = { d: Math.max(10, Math.hypot(p.x - q.x, p.y - q.y)), mx: (p.x + q.x) / 2, my: (p.y + q.y) / 2, a: Math.atan2(q.y - p.y, q.x - p.x) };
    }
    if (e.pointerType !== "mouse" && this.pointers.size === 1 && this.tool === "rotate") {
      const now = performance.now();
      if (now - this.lastTap.t < 320 && Math.hypot(x - this.lastTap.x, y - this.lastTap.y) < 30) {
        this.lastTap.t = 0;
        this.view("anterior");
      } else this.lastTap = { t: now, x, y };
    }
  }

  private move(e: PointerEvent): void {
    const prev = this.pointers.get(e.pointerId);
    if (!prev) return;
    const [x, y] = this.pos(e);
    this.pointers.set(e.pointerId, { x, y });
    const h = this.canvas.getBoundingClientRect().height || 1;
    if (this.pinch && this.pointers.size >= 2) {
      const [p, q] = [...this.pointers.values()];
      const d = Math.max(10, Math.hypot(p.x - q.x, p.y - q.y));
      const mx = (p.x + q.x) / 2, my = (p.y + q.y) / 2, a = Math.atan2(q.y - p.y, q.x - p.x);
      this.cam.dist *= this.pinch.d / d;
      this.cam.pan(mx - this.pinch.mx, my - this.pinch.my, h);
      this.cam.roll(a - this.pinch.a);
      this.pinch = { d, mx, my, a };
      this.touched();
      return;
    }
    if (this.pointers.size !== 1 || this.pinch) return;
    if (this.cropDrag) {
      this.dragFace(x, y);
      return;
    }
    if (this.measureDrag) {
      const p = this.pick(x, y);
      if (p) this.measureDrag.m[this.measureDrag.end] = p;     // off the surface: the point waits where it was
      this.drawOverlay();
      return;
    }
    if (this.dragBtn === -1) return;
    if (this.lasso) {
      const last = this.lasso[this.lasso.length - 1];
      if (Math.hypot(x - last[0], y - last[1]) > 3) this.lasso.push([x, y]);
      this.drawOverlay();
      return;
    }
    const dx = x - prev.x, dy = y - prev.y;
    if (this.dragBtn === 2) this.cam.dist *= Math.exp(dy / 150);
    else if (this.dragBtn === 1 || this.tool === "pan") this.cam.pan(dx, dy, h);
    else this.cam.orbit(dx * 0.008, dy * 0.008);
    this.touched();
  }

  private up(e: PointerEvent): void {
    if (!this.pointers.has(e.pointerId)) return;
    this.pointers.delete(e.pointerId);
    if (this.pointers.size === 0) {
      this.cropDrag = null;
      this.finishMeasureDrag();
      const lasso = this.lasso;
      this.lasso = null;
      if (lasso && lasso.length >= 3) {
        this.onStatus("Cutting…");
        setTimeout(() => {
          this.cut(lasso);
          this.tool = "rotate";                            // one cut, then back to turning (like the desktop)
          this.drawOverlay();
          this.onToolDone();
          this.onChange();
        }, 20);
      }
      this.drawOverlay();
    }
    if (this.pointers.size < 2) this.pinch = null;
    if (this.pointers.size === 0) this.pinch = null;
  }

  private cancelMeasureDrag(): void {
    const d = this.measureDrag;
    if (!d) return;
    this.measureDrag = null;
    if (d.created) {
      this.measures.splice(this.measures.indexOf(d.m), 1);
      this.selectedMeasure = null;
    } else [d.m.a, d.m.b] = d.orig;
  }

  /** One length, then back to turning (like the desktop); a tap without a drag draws nothing. */
  private finishMeasureDrag(): void {
    const d = this.measureDrag;
    if (!d) {
      if (this.dragBtn === -1) this.dragBtn = 0;
      return;
    }
    this.measureDrag = null;
    if (d.created) {
      if (Math.hypot(...sub(d.m.a, d.m.b)) < 0.5) {
        this.measures.splice(this.measures.indexOf(d.m), 1);
        this.selectedMeasure = null;
      } else {
        this.tool = "rotate";
        this.onToolDone();
      }
    }
    this.drawOverlay();
    this.onChange();
  }

  // ------------------------------------------------------------ rendering
  requestRender(): void {
    if (this.dirty) return;
    this.dirty = true;
    requestAnimationFrame(() => this.draw());
  }

  private draw(): void {
    this.dirty = false;
    if (!this.vol) return;
    const p = this.canvas.parentElement!;
    const W = p.clientWidth, H = p.clientHeight;
    if (!W || !H) return;
    // pixels when still: at most ~3.2 M (2.2 M on phones and tablets: one frame at rest may take its time, and fewer
    // than that looked pixelated on retina screens). While turning, a share that adapts so frames come every
    // ~25–45 ms (rays through a whole CT are heavy on a tablet).
    const dpr = window.devicePixelRatio || 1;
    const now = performance.now();
    if (this.interactive && now - this.lastDraw < 250) {
      const dt = now - this.lastDraw;
      if (dt > 45) this.iscale = Math.max(0.22, this.iscale * 0.85);
      else if (dt < 25) this.iscale = Math.min(0.6, this.iscale * 1.05);
    }
    this.lastDraw = now;
    let s = Math.min(dpr, Math.sqrt((MOBILE ? 2.2e6 : 3.2e6) / (W * H)));
    if (this.interactive) s *= this.iscale;
    const cw = Math.max(1, Math.round(W * s)), ch = Math.max(1, Math.round(H * s));
    if (this.canvas.width !== cw || this.canvas.height !== ch) {
      this.canvas.width = cw;
      this.canvas.height = ch;
    }
    const minSp = Math.min(...this.renderer.spacing);
    const step = minSp * QUALITY[this.quality] * (this.interactive ? 2 : 1);
    const tf = this.tf;
    const t0 = performance.now();
    this.renderer.render({
      invVP: this.cam.invVP(W / H), step, blend: this.blend, shade: tf.shading,
      light: [tf.ambient, tf.diffuse, tf.specular, tf.specular_power], background: tf.background,
      window: this.window, clipLo: this.crop.on ? this.crop.lo : [0, 0, 0], clipHi: this.crop.on ? this.crop.hi : [1, 1, 1],
      fastGradient: this.interactive,
    }, RIM_GRAD_MAX);
    this.lastFrameMs = performance.now() - t0;
    this.drawOverlay();
  }

  /** Crop box wireframe with its face handles, and the lasso being drawn (a 2D canvas over the WebGL one). */
  private drawOverlay(): void {
    const o = this.overlay;
    if (!o) return;
    const r = this.canvas.getBoundingClientRect();
    const W = r.width, H = r.height, dpr = window.devicePixelRatio || 1;
    if (o.width !== Math.round(W * dpr) || o.height !== Math.round(H * dpr)) {
      o.width = Math.round(W * dpr);
      o.height = Math.round(H * dpr);
    }
    const ctx = o.getContext("2d")!;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, o.width, o.height);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (this.vol && this.cropEdit) {
      const { lo, hi } = this.crop;
      const corner = (i: number): V3 => [i & 1 ? hi[0] : lo[0], i & 2 ? hi[1] : lo[1], i & 4 ? hi[2] : lo[2]];
      ctx.strokeStyle = "rgba(255,220,0,0.85)";
      ctx.lineWidth = 1.2;
      for (let i = 0; i < 8; i++) {
        for (const bit of [1, 2, 4]) {
          if (i & bit) continue;
          const a = this.screenOf(corner(i), W, H), b = this.screenOf(corner(i | bit), W, H);
          if (!a || !b) continue;
          ctx.beginPath();
          ctx.moveTo(a[0], a[1]);
          ctx.lineTo(b[0], b[1]);
          ctx.stroke();
        }
      }
      const colors = ["rgb(40,170,255)", "rgb(255,60,150)", "rgb(255,215,0)"];      // x, y, z like the MPR planes
      for (const [axis, side] of FACES) {
        const p = this.screenOf(this.faceCenter(axis, side), W, H);
        if (!p) continue;
        ctx.fillStyle = colors[axis];
        ctx.strokeStyle = "#000";
        ctx.beginPath();
        ctx.arc(p[0], p[1], 7, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
    }
    this.drawMeasures(ctx, W, H);
    if (this.lasso && this.lasso.length > 1) {
      ctx.strokeStyle = this.cutOutside ? "rgb(80,200,255)" : "rgb(255,80,80)";
      ctx.fillStyle = this.cutOutside ? "rgba(80,200,255,0.12)" : "rgba(255,80,80,0.15)";
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      this.lasso.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  private drawMeasures(ctx: CanvasRenderingContext2D, W: number, H: number): void {
    ctx.font = "600 13px system-ui, -apple-system, sans-serif";
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
    for (const m of this.measures) {
      const a = this.screenOfMm(m.a, W, H), b = this.screenOfMm(m.b, W, H);
      if (!a || !b) continue;
      const col = m === this.selectedMeasure ? "#7DB6F5" : "#FFD24A";
      ctx.strokeStyle = col;
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.moveTo(a[0], a[1]);
      ctx.lineTo(b[0], b[1]);
      ctx.stroke();
      for (const p of [a, b]) {
        ctx.beginPath();
        ctx.arc(p[0], p[1], m === this.selectedMeasure ? 5 : 3.5, 0, Math.PI * 2);
        ctx.stroke();
      }
      const t = fmtLen(Math.hypot(...sub(m.a, m.b)));
      const w = ctx.measureText(t).width + 8;
      const x = Math.max(2, Math.min(W - w - 2, (a[0] + b[0]) / 2 + 8)), y = Math.max(2, Math.min(H - 22, (a[1] + b[1]) / 2 - 10));
      ctx.fillStyle = "rgba(0,0,0,0.6)";
      ctx.fillRect(x, y, w, 20);
      ctx.fillStyle = col;
      ctx.fillText(t, x + 4, y + 15);
    }
  }

  /** Wait for the GPU to finish the frame on screen (for timing checks). */
  finish(): void {
    this.renderer.gl.finish();
  }

  /** Remove this view from the page (its graphics context is gone). */
  destroy(): void {
    this.dispose();
    this.overlay?.remove();
    this.overlay = null;
  }

  dispose(): void {
    this.renderer.release();
    this.renderer.releaseSegments();
    this.renderer.vessels = null;
    this.vessels = null;
    this.vesselKey = "";
    this.segMasks = [];
    this.vol = null;
    this.box = null;
    this.hist = null;
    this.removed = null;
    this.undoStack = [];
    this.bone = null;
    this.pickTable = null;
    this.measures = [];
    this.selectedMeasure = null;
    this.snapshots = [];                                   // thumbnails are patient images: not kept
    this.drawOverlay();
  }
}

async function deflate(a: Uint32Array): Promise<Uint8Array> {
  const s = new Blob([a as Uint32Array<ArrayBuffer>]).stream().pipeThrough(new CompressionStream("deflate-raw"));
  return new Uint8Array(await new Response(s).arrayBuffer());
}

async function inflate(p: Promise<Uint8Array>): Promise<Uint8Array> {
  const s = new Blob([(await p) as Uint8Array<ArrayBuffer>]).stream().pipeThrough(new DecompressionStream("deflate-raw"));
  return new Uint8Array(await new Response(s).arrayBuffer());
}

/** log(1 + count) histogram of the data over [lo, hi], sampled (at most ~2 M voxels). */
export function histogram(vol: Volume, lo: number, hi: number, bins = 512): Float32Array {
  const h = new Float32Array(bins);
  const d = vol.data;
  const step = Math.max(1, Math.floor(d.length / 2e6));
  const k = bins / Math.max(1, hi - lo);
  for (let i = 0; i < d.length; i += step) {
    const b = Math.floor((d[i] - lo) * k);
    if (b >= 0 && b < bins) h[b]++;
  }
  for (let i = 0; i < bins; i++) h[i] = Math.log1p(h[i]);
  return h;
}


