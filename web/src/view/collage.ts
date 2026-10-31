/**
 * Collage (dcv/collage.py; toolbar Collage, Ctrl+G): one image of 5 × 5, 4 × 4, 4 × 3… tiles from the active viewport,
 * one slice every N mm (5 mm by default). The run of slices starts centred in the series — the middle of a head, not
 * the first slices below the skull base — and a slider moves it. A series whose slices are already that far apart
 * keeps every slice; a movie (XA / US run) has its frames spread over the image. Tiles are drawn by the viewport
 * itself, so window / level, invert and DSA look exactly as on screen. A stack can also be shown in another patient
 * plane (Axial / Coronal / Sagittal): those tiles are reformatted from the volume with the view's window.
 */
import { Volume } from "../dicom/volume";
import { col, dot, mulMtV, type M3, type V3 } from "../geometry";
import { Layer, type Segments } from "../seg/segments";
import { roundEven } from "../vr/bone";
import type { Viewport } from "./viewport";

/** (columns, rows) */
export const GRIDS: [number, number][] = [[5, 5], [4, 4], [4, 3], [3, 3], [3, 2], [6, 5]];
export const DEFAULT_INTERVAL_MM = 5;
const HEADER_PX = 30;
const TILE_WIDTHS = [256, 384, 512];
// iOS Safari leaves a canvas above ~16.7 million pixels blank: larger collages get narrower tiles
const MAX_PIXELS = 16e6;

/** Slices between two tiles: 1 when the slices are already interval mm or more apart. */
export function stepFor(spacing: number, interval: number): number {
  if (spacing <= 0 || interval <= 0) return 1;
  return Math.max(1, roundEven(interval / spacing));
}

/** At most `cells` indices, one every interval mm, starting at `first` (null = centred in the series). */
export function sliceIndices(count: number, spacing: number, interval: number, cells: number, first: number | null = null): number[] {
  const step = stepFor(spacing, interval);
  const n = Math.min(cells, Math.floor((count - 1) / step) + 1);
  const maxFirst = count - 1 - (n - 1) * step;
  const start = first === null ? Math.floor(maxFirst / 2) : Math.max(0, Math.min(maxFirst, Math.trunc(first)));
  return Array.from({ length: n }, (_, i) => start + i * step);
}

/** A movie has no slice spacing: spread the frames evenly over the image. */
export function frameIndices(count: number, cells: number): number[] {
  if (count <= cells) return Array.from({ length: count }, (_, i) => i);
  if (cells < 2) return [0];
  const out = new Set<number>();
  for (let i = 0; i < cells; i++) out.add(roundEven((i * (count - 1)) / (cells - 1)));
  return [...out].sort((a, b) => a - b);
}

/** Tile size in pixels: the image's shape in mm, narrower when the page would not fit a canvas. */
export function tileSize(tileW: number, cols: number, usedRows: number, mmW: number, mmH: number, top: number): [number, number] {
  const aspect = mmH / Math.max(1e-6, mmW);
  let w = tileW;
  let h = Math.max(16, Math.round(w * aspect));
  if (cols * w * (top + usedRows * h) > MAX_PIXELS) {
    w = Math.max(16, Math.floor(Math.sqrt(MAX_PIXELS / (cols * usedRows * aspect)) * 0.98));
    h = Math.max(16, Math.round(w * aspect));
  }
  return [w, h];
}

function slicePosition(view: Viewport, k: number): number | null {
  const v = view.vol;
  const p = v instanceof Volume && !v.frameTimeMs ? v.positions?.[k] : null;
  return p && v instanceof Volume ? dot(p, col(v.direction, 2)) : null;
}

export type PlaneName = "axial" | "coronal" | "sagittal";
export const PLANES: PlaneName[] = ["axial", "coronal", "sagittal"];
// right, up and normal of each patient plane in LPS (+x = L, +y = P, +z = S): the display of the MPR views
const PLANE_AXES: Record<PlaneName, [V3, V3, V3]> = {
  axial: [[1, 0, 0], [0, -1, 0], [0, 0, 1]], coronal: [[1, 0, 0], [0, 0, 1], [0, 1, 0]], sagittal: [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
};

/** The patient plane closest to the acquired slices. */
export function ownPlane(direction: M3): PlaneName {
  const n = col(direction, 2).map(Math.abs);
  return (["sagittal", "coronal", "axial"] as const)[n.indexOf(Math.max(...n))];
}

/** What a reformat needs of a volume (voxel i of an axis is centred at i · spacing, as in the MPR). */
export interface StackGeometry {
  cols: number;
  rows: number;
  n: number;
  spacing: [number, number, number];     // (slice, row, column) mm
  direction: M3;
  origin: V3;                            // patient position of the first voxel
}

/** A patient plane through a stack: how many slices fit, how far apart, and where each pixel of a slice lies. */
export class Reformat {
  readonly a: V3;
  readonly b: V3;
  readonly nrm: V3;
  readonly mmW: number;
  readonly mmH: number;
  readonly count: number;
  /** mm between two slices: the voxel size along the plane's normal. */
  readonly spacing: number;
  private readonly u0: number;
  private readonly v1: number;
  private readonly w0: number;
  private readonly posBase: number;

  constructor(readonly g: StackGeometry, plane: PlaneName) {
    const [right, up, normal] = PLANE_AXES[plane];
    this.a = mulMtV(g.direction, right);
    this.b = mulMtV(g.direction, up);
    this.nrm = mulMtV(g.direction, normal);
    const [sz, sy, sx] = g.spacing;
    const X = (g.cols - 1) * sx, Y = (g.rows - 1) * sy, Z = (g.n - 1) * sz;
    const range = (d: V3): [number, number] => {
      let lo = Infinity, hi = -Infinity;
      for (const x of [0, X]) for (const y of [0, Y]) for (const z of [0, Z]) {
        const t = d[0] * x + d[1] * y + d[2] * z;
        if (t < lo) lo = t;
        if (t > hi) hi = t;
      }
      return [lo, hi];
    };
    const [u0, u1] = range(this.a), [v0, v1] = range(this.b), [w0, w1] = range(this.nrm);
    this.u0 = u0;
    this.v1 = v1;
    this.w0 = w0;
    this.mmW = u1 - u0;
    this.mmH = v1 - v0;
    this.spacing = 1 / Math.hypot(this.nrm[0] / sx, this.nrm[1] / sy, this.nrm[2] / sz);
    this.count = Math.floor((w1 - w0) / this.spacing + 1e-6) + 1;
    this.posBase = dot(g.origin, normal) + w0;
  }

  /** Slice position along the plane's normal in patient space (mm). */
  position(k: number): number {
    return this.posBase + k * this.spacing;
  }

  /** Slice k as w × h pixels: voxel coordinates (x, y, z) of pixel (0, 0) and the steps per column and per row. */
  pixelGrid(k: number, w: number, h: number): { o: V3; du: V3; dv: V3 } {
    const [sz, sy, sx] = this.g.spacing;
    const ru = this.mmW / Math.max(1, w - 1), rv = this.mmH / Math.max(1, h - 1);
    const t = this.w0 + k * this.spacing;
    const m = (i: number) => this.a[i] * this.u0 + this.b[i] * this.v1 + this.nrm[i] * t;
    return {
      o: [m(0) / sx, m(1) / sy, m(2) / sz],
      du: [(this.a[0] * ru) / sx, (this.a[1] * ru) / sy, (this.a[2] * ru) / sz],
      dv: [(-this.b[0] * rv) / sx, (-this.b[1] * rv) / sy, (-this.b[2] * rv) / sz],
    };
  }

  /** Trilinear samples of slice k; outside the stack = `outside`. */
  sample(data: ArrayLike<number>, k: number, w: number, h: number, out: Float32Array, outside: number): void {
    const { cols: nx, rows: ny, n: nz } = this.g;
    const nxy = nx * ny, xm = nx - 1, ym = ny - 1, zm = nz - 1;
    const { o, du, dv } = this.pixelGrid(k, w, h);
    for (let r = 0, i = 0; r < h; r++) {
      let X = o[0] + dv[0] * r, Y = o[1] + dv[1] * r, Z = o[2] + dv[2] * r;
      for (let c = 0; c < w; c++, i++, X += du[0], Y += du[1], Z += du[2]) {
        if (X < -0.5 || Y < -0.5 || Z < -0.5 || X > xm + 0.5 || Y > ym + 0.5 || Z > zm + 0.5) {
          out[i] = outside;
          continue;
        }
        const x = X < 0 ? 0 : X > xm ? xm : X, y = Y < 0 ? 0 : Y > ym ? ym : Y, z = Z < 0 ? 0 : Z > zm ? zm : Z;
        const x0 = x | 0, y0 = y | 0, z0 = z | 0;
        const fx = x - x0, fy = y - y0, fz = z - z0;
        const ax = x0 < xm ? 1 : 0, ay = y0 < ym ? nx : 0, az = z0 < zm ? nxy : 0;
        const b = z0 * nxy + y0 * nx + x0;
        const c00 = data[b] + (data[b + ax] - data[b]) * fx;
        const c10 = data[b + ay] + (data[b + ay + ax] - data[b + ay]) * fx;
        const c01 = data[b + az] + (data[b + az + ax] - data[b + az]) * fx;
        const c11 = data[b + az + ay] + (data[b + az + ay + ax] - data[b + az + ay]) * fx;
        const c0 = c00 + (c10 - c00) * fy, c1 = c01 + (c11 - c01) * fy;
        out[i] = c0 + (c1 - c0) * fz;
      }
    }
  }
}

/** Draws one reformatted tile with the view's window, and the segments over it. */
class ReformatTiles {
  private canvas = document.createElement("canvas");
  private img: ImageData | null = null;
  private plane: Float32Array | null = null;
  private layer = new Layer();

  constructor(private readonly view: Viewport, private readonly vol: Volume, private readonly rf: Reformat, private readonly segs: Segments | null) {}

  draw(k: number, ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number): void {
    if (!this.img || this.img.width !== w || this.img.height !== h) {
      this.img = new ImageData(w, h);
      this.plane = new Float32Array(w * h);
      this.canvas.width = w;
      this.canvas.height = h;
    }
    const plane = this.plane!;
    this.rf.sample(this.vol.data, k, w, h, plane, this.vol.valueRange()[0]);
    const lo = this.view.wl - this.view.ww / 2, sc = 255 / this.view.ww, inv = this.view.invert;
    const px = new Uint32Array(this.img.data.buffer);
    for (let i = 0; i < px.length; i++) {
      let g = (plane[i] - lo) * sc;
      g = g < 0 ? 0 : g > 255 ? 255 : g | 0;
      if (inv) g = 255 - g;
      px[i] = 0xff000000 | (g << 16) | (g << 8) | g;
    }
    this.canvas.getContext("2d")!.putImageData(this.img, 0, 0);
    ctx.drawImage(this.canvas, x, y, w, h);
    const segs = this.segs;
    if (segs?.list(this.vol).length) {
      const { o, du, dv } = this.rf.pixelGrid(k, w, h);
      const over = this.layer.get(`${segs.version}|${k}|${w}x${h}`, w, h, (im) => segs.paintPlane(this.vol, o, du, dv, im));
      if (over) ctx.drawImage(over, x, y, w, h);
    }
  }
}

/** What renderCollage draws: the viewport's own frames, or reformatted slices. */
interface TileSource {
  n: number;
  mmW: number;
  mmH: number;
  draw(k: number, ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number): void;
  position(k: number): number | null;
}

/** Draw the tiles (data must be decoded) on `page`. Returns the tile width used. */
function renderCollage(src: TileSource, page: HTMLCanvasElement, indices: number[], cols: number, tileW: number, labels: boolean, title: string): number {
  const top = labels ? HEADER_PX : 0;
  const usedRows = Math.ceil(indices.length / cols);        // no empty rows when the series has fewer slices
  const [w, h] = tileSize(tileW, cols, usedRows, src.mmW, src.mmH, top);
  page.width = cols * w;
  page.height = top + usedRows * h;
  const ctx = page.getContext("2d", { alpha: false })!;
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, page.width, page.height);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  const font = Math.max(8, Math.floor(w / 32));
  indices.forEach((k, i) => {
    const x = (i % cols) * w, y = top + Math.floor(i / cols) * h;
    src.draw(k, ctx, x, y, w, h);
    if (labels) {
      ctx.font = `${font}px system-ui, -apple-system, sans-serif`;
      ctx.fillStyle = "rgb(235,235,235)";
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      ctx.fillText(`${k + 1}/${src.n}`, x + 5, y + 4);
      const pos = src.position(k);
      if (pos !== null) {
        ctx.textAlign = "right";
        ctx.textBaseline = "bottom";
        ctx.fillText(`${pos.toFixed(1)} mm`, x + w - 5, y + h - 4);
      }
    }
    ctx.strokeStyle = "rgb(60,60,60)";
    ctx.lineWidth = 1;
    ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
  });
  if (labels) {
    ctx.font = "15px system-ui, -apple-system, sans-serif";
    ctx.fillStyle = "rgb(235,235,235)";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(title, 8, HEADER_PX / 2);
  }
  return w;
}

/** Settings and a live preview; Save writes the image, Copy puts it on the clipboard. iOS saves through its share sheet. */
export function openCollage(view: Viewport, toast: (t: string) => void, useShareSheet = false, segs: Segments | null = null): void {
  const vol = view.vol;
  if (!vol) return;
  const movie = vol.frameTimeMs !== null;
  // another plane needs a stack with geometry, without subtraction or colour
  const stack = vol instanceof Volume && !!vol.positions && vol.n >= 3 && !vol.rgb && !movie && !view.dsa ? vol : null;
  const own: PlaneName | null = stack ? ownPlane(stack.direction) : null;
  let plane = own;
  let spacing = vol.spacing[0];
  let n = vol.n;
  let reformat: Reformat | null = null;
  let tiles: ReformatTiles | null = null;
  const dlg = document.getElementById("collage") as HTMLDialogElement;
  dlg.innerHTML = `
    <h2>Collage</h2>
    <div class="col-form">
      ${stack ? `<label>Plane <select id="col-plane">${PLANES.map((p) => `<option value="${p}"${p === own ? " selected" : ""}>${p[0].toUpperCase() + p.slice(1)}${p === own ? " (as acquired)" : ""}</option>`).join("")}</select></label>` : ""}
      <label>Tiles <select id="col-grid">
        ${GRIDS.map(([c, r]) => `<option value="${c},${r}">${c} × ${r}  (${c} columns, ${r} rows)</option>`).join("")}
        <option value="custom">Custom…</option>
      </select></label>
      <label id="col-custom" hidden><input id="col-cols" type="number" min="1" max="12" value="5" /> columns ×
        <input id="col-rows" type="number" min="1" max="12" value="5" /> rows</label>
      <label>One image every <input id="col-mm" type="number" min="0.1" max="100" step="0.5" value="${DEFAULT_INTERVAL_MM}" /> mm</label>
      <label class="col-wide">Position <input id="col-pos" type="range" min="0" max="0" value="0" />
        <span id="col-range"></span> <button id="col-centre" type="button" class="chip">Centre</button></label>
      <label>Tile width <select id="col-size">${TILE_WIDTHS.map((px) => `<option value="${px}"${px === 384 ? " selected" : ""}>${px} px</option>`).join("")}</select></label>
      <label class="check"><input id="col-labels" type="checkbox" checked /> Header, image number and position</label>
    </div>
    <p id="col-info"></p>
    <div class="col-preview"><canvas id="col-page"></canvas></div>
    <div class="col-actions">
      <button id="col-png" type="button" class="chip">Save PNG</button>
      <button id="col-jpg" type="button" class="chip">Save JPEG</button>
      <button id="col-copy" type="button" class="chip">Copy</button>
      <button id="col-close" type="button" class="big secondary">Close</button>
    </div>`;
  const el = <T extends HTMLElement>(id: string) => dlg.querySelector<T>(`#${id}`)!;
  const grid = el<HTMLSelectElement>("col-grid");
  const colsIn = el<HTMLInputElement>("col-cols"), rowsIn = el<HTMLInputElement>("col-rows");
  const mm = el<HTMLInputElement>("col-mm"), pos = el<HTMLInputElement>("col-pos");
  const size = el<HTMLSelectElement>("col-size"), labels = el<HTMLInputElement>("col-labels");
  const info = el("col-info"), page = el<HTMLCanvasElement>("col-page");
  let cols = 5, rows = 5;
  let first: number | null = null;          // null = centred
  let mid = 0;                              // centre kept when the grid or the interval changes
  let ticket = 0;
  let raf = 0;
  let ready = false;

  const clampInt = (s: string, lo: number, hi: number, old: number) => {
    const v = Math.round(+s);
    return Number.isFinite(v) && s !== "" ? Math.max(lo, Math.min(hi, v)) : old;
  };
  const interval = () => {
    const v = +mm.value;
    return Number.isFinite(v) && v >= 0.1 ? Math.min(100, v) : DEFAULT_INTERVAL_MM;
  };
  const indices = () => (movie ? frameIndices(n, cols * rows) : sliceIndices(n, spacing, interval(), cols * rows, first));
  const title = () => {
    const parts = [vol.patient, vol.description, plane?.toUpperCase() ?? ""];
    if (!movie) parts.push(`every ${(stepFor(spacing, interval()) * spacing).toFixed(1)} mm`);
    parts.push(`WW ${view.ww.toFixed(0)} / WL ${view.wl.toFixed(0)}`);
    return parts.filter(Boolean).join("   ·   ");
  };

  const refresh = () => {
    const idx = indices();
    mid = (idx[0] + idx[idx.length - 1]) / 2;
    const step = stepFor(spacing, interval());
    let how: string;
    if (movie) how = `${idx.length} frames spread over the run of ${n}`;
    else {
      const maxFirst = n - 1 - (idx.length - 1) * step;
      pos.max = String(Math.max(0, maxFirst));
      pos.value = String(idx[0]);
      el("col-range").textContent = `images ${idx[0] + 1} to ${idx[idx.length - 1] + 1} of ${n}`;
      if (idx.length === n && step === 1 && spacing >= interval() - 1e-3) how = `all ${n} slices — they are ${spacing.toFixed(1)} mm apart already`;
      else if (idx.length < cols * rows) how = `the whole series fits: ${idx.length} images, one every ${(step * spacing).toFixed(1)} mm`;
      else how = `${idx.length} images, one every ${(step * spacing).toFixed(1)} mm — images ${idx[0] + 1}–${idx[idx.length - 1] + 1} of ${n}; move the slider to shift them`;
    }
    const t = ++ticket;
    ready = false;
    info.textContent = `${how}.`;
    const paint = () => {
      const src: TileSource = reformat && tiles
        ? { n, mmW: reformat.mmW, mmH: reformat.mmH, draw: (k, ctx, x, y, w, h) => tiles!.draw(k, ctx, x, y, w, h), position: (k) => reformat!.position(k) }
        : { n, mmW: vol.cols * vol.spacing[2], mmH: vol.rows * vol.spacing[1], draw: (k, ctx, x, y, w, h) => view.drawFrame(k, ctx, x, y, w, h), position: (k) => slicePosition(view, k) };
      const used = renderCollage(src, page, idx, cols, +size.value, labels.checked, title());
      ready = true;
      info.textContent = `${how}.${used < +size.value ? ` Tiles reduced to ${used} px to fit one image.` : ""}`;
    };
    if (reformat && stack && stack.spacing[0] > 3 && stack.spacing[0] > 2.5 * Math.min(stack.spacing[1], stack.spacing[2])) {
      how += ` Reformatted from slices ${stack.spacing[0].toFixed(1)} mm apart, so it looks coarse`;
    }
    const need = reformat ? Array.from({ length: vol.n }, (_, k) => k) : [...new Set(idx.flatMap((k) => view.needs(k)))];
    cancelAnimationFrame(raf);
    if (need.every((k) => vol.isReady(k))) raf = requestAnimationFrame(paint);   // one drawing per slider step
    else {
      info.textContent = `${how}. Decoding…`;
      vol.want(need).then(() => {
        if (t === ticket) raf = requestAnimationFrame(paint);
      });
    }
  };

  /** Grid or interval changed: keep the run centred on the same slice. */
  const layoutChanged = () => {
    if (first !== null) {
      const step = stepFor(spacing, interval());
      const count = Math.min(cols * rows, Math.floor((n - 1) / step) + 1);
      first = Math.round(mid - ((count - 1) * step) / 2);
    }
    refresh();
  };

  dlg.querySelector<HTMLSelectElement>("#col-plane")?.addEventListener("change", (e) => {
    plane = (e.target as HTMLSelectElement).value as PlaneName;
    first = null;
    if (plane === own || !stack) {
      reformat = null;
      tiles = null;
      n = vol.n;
      spacing = vol.spacing[0];
    } else {
      reformat = new Reformat({ cols: stack.cols, rows: stack.rows, n: stack.n, spacing: stack.spacing, direction: stack.direction, origin: stack.positions![0] }, plane);
      tiles = new ReformatTiles(view, stack, reformat, segs);
      n = reformat.count;
      spacing = reformat.spacing;
    }
    refresh();
  });
  grid.addEventListener("change", () => {
    const custom = grid.value === "custom";
    el("col-custom").hidden = !custom;
    if (!custom) {
      [cols, rows] = grid.value.split(",").map(Number);
      colsIn.value = String(cols);
      rowsIn.value = String(rows);
    }
    layoutChanged();
  });
  colsIn.addEventListener("change", () => {
    cols = clampInt(colsIn.value, 1, 12, cols);
    colsIn.value = String(cols);
    layoutChanged();
  });
  rowsIn.addEventListener("change", () => {
    rows = clampInt(rowsIn.value, 1, 12, rows);
    rowsIn.value = String(rows);
    layoutChanged();
  });
  mm.addEventListener("change", () => {
    mm.value = String(interval());
    layoutChanged();
  });
  pos.addEventListener("input", () => {
    first = +pos.value;
    refresh();
  });
  el("col-centre").addEventListener("click", () => {
    first = null;
    refresh();
  });
  size.addEventListener("change", refresh);
  labels.addEventListener("change", refresh);
  if (movie) {
    mm.disabled = pos.disabled = el<HTMLButtonElement>("col-centre").disabled = true;
    mm.title = "A movie run has no slice spacing: its frames are spread over the image.";
    el("col-range").textContent = "";
  }

  const blob = (type: string) => new Promise<Blob>((res, rej) => page.toBlob((b) => (b ? res(b) : rej(new Error("no image"))), type, 0.92));
  const save = async (type: string, ext: string) => {
    if (!ready) return;
    const file = new File([await blob(type)], `collage.${ext}`, { type });
    if (useShareSheet && navigator.canShare?.({ files: [file] })) {
      try {
        await navigator.share({ files: [file] });
        return;
      } catch (e) {
        if ((e as Error).name === "AbortError") return;        // the user closed the sheet
      }
    }
    const a = document.createElement("a");
    a.href = URL.createObjectURL(file);
    a.download = file.name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    toast(`Collage saved: ${file.name}`);
  };
  el("col-png").addEventListener("click", () => save("image/png", "png"));
  el("col-jpg").addEventListener("click", () => save("image/jpeg", "jpg"));
  const copy = el<HTMLButtonElement>("col-copy");
  copy.hidden = !navigator.clipboard?.write || typeof ClipboardItem === "undefined";    // https only
  copy.addEventListener("click", async () => {
    if (!ready) return;
    try {
      await navigator.clipboard.write([new ClipboardItem({ "image/png": blob("image/png") })]);
      toast("Collage copied");
    } catch (e) {
      toast(`Could not copy: ${(e as Error).message ?? e}`);
    }
  });
  el("col-close").addEventListener("click", () => dlg.close());
  dlg.addEventListener("close", () => {
    ticket++;
    cancelAnimationFrame(raf);
    page.width = page.height = 0;            // frees the image at once (iOS counts canvas memory)
    dlg.innerHTML = "";
  }, { once: true });

  dlg.showModal();
  refresh();
}
