/**
 * MPR (like dcv/mpr_window.py): sagittal / coronal / axial planes through one centre, with
 * axes that can be shifted and tilted (oblique MPR) and a slab (MIP / MinIP / average). Each plane is an ordinary
 * Viewport (W/L, zoom, pan, pinch, length all work); on top of that, in every view:
 *   drag the line intersection = move the centre · drag a line = shift that plane · drag a circle = tilt the axes ·
 *   wheel / scroll tool = move this plane along its normal.
 * Planes are resampled here in JavaScript (trilinear); while something is dragged every other pixel is sampled.
 */
import type { Volume, Pixels } from "../dicom/volume";
import { add, col, cross, dot, letter, mulMM, mulMV, mulMtV, norm, rot, scale, sub, unit, I3, type M3, type V3 } from "../geometry";
import type { ViewHooks, ViewSource, Viewport } from "./viewport";

export type Kind = "sagittal" | "coronal" | "axial";
export const KINDS: Kind[] = ["sagittal", "coronal", "axial"];
export const TITLES: Record<Kind, string> = { sagittal: "Sagittal", coronal: "Coronal", axial: "Axial" };
export const PLANE_COLORS: Record<Kind, string> = { sagittal: "rgb(40,170,255)", coronal: "rgb(255,60,150)", axial: "rgb(255,215,0)" };
// reference up / right of each view in patient LPS (+x = L, +y = P, +z = S): radiological display
const REF: Record<Kind, [V3, V3]> = {
  sagittal: [[0, 0, 1], [0, 1, 0]], coronal: [[0, 0, 1], [1, 0, 0]], axial: [[0, -1, 0], [1, 0, 0]],
};

export type SlabMode = "max" | "min" | "mean";
export const SLABS: [string, number, SlabMode][] = [
  ["Off", 0, "max"], ["MIP 5 mm", 5, "max"], ["MIP 10 mm", 10, "max"], ["MIP 20 mm", 20, "max"], ["MIP 40 mm", 40, "max"],
  ["MinIP 10 mm", 10, "min"], ["Average 5 mm", 5, "mean"], ["Average 10 mm", 10, "mean"],
];

export class MPRModel {
  readonly D: M3;
  readonly extent: V3;           // mm, image frame (x = columns, y = rows, z = slices)
  readonly vcenter: V3;
  readonly step: number;         // mm per index step along a normal
  readonly res: number;          // mm per plane pixel
  readonly N: number;            // plane = (2N+1)² pixels
  readonly half: number;
  readonly lo: number;
  center!: V3;
  R: M3 = [...I3];
  roll: Record<Kind, number> = { sagittal: 0, coronal: 0, axial: 0 };
  slabMm = 0;
  slabMode: SlabMode = "max";
  showLines = true;
  /** Sample every other pixel (while dragging), the full plane once things settle. */
  fast = false;
  private listeners = new Set<() => void>();

  constructor(readonly vol: Volume) {
    this.D = vol.direction;
    const [sz, sy, sx] = vol.spacing;
    this.extent = [vol.cols * sx, vol.rows * sy, vol.n * sz];
    this.vcenter = scale(sub(this.extent, [sx, sy, sz]), 0.5);
    this.step = Math.min(sx, sy, sz);
    const diag = norm(this.extent);
    this.res = Math.max(this.step, diag / 900);            // a plane covers every rotation: at most ~900² pixels
    this.N = Math.ceil(diag / 2 / this.res);
    this.half = Math.ceil(diag / 2 / this.step);
    this.lo = vol.valueRange()[0];
    this.reset(false);
  }

  onChange(f: () => void): () => void {
    this.listeners.add(f);
    return () => this.listeners.delete(f);
  }

  changed(): void {
    for (const f of this.listeners) f();
  }

  reset(notify = true): void {
    this.center = [...this.vcenter];
    this.R = [...I3];
    this.roll = { sagittal: 0, coronal: 0, axial: 0 };
    if (notify) this.changed();
  }

  /** (right, up, normal) of a view in the image frame. "Up" follows the patient reference, so tilting
   *  the axes in one view does not turn that view's own image; it makes the other views oblique. */
  axes(kind: Kind): [V3, V3, V3] {
    const n = col(this.R, KINDS.indexOf(kind));
    const upRef = mulMtV(this.D, REF[kind][0]);
    const rightRef = mulMtV(this.D, REF[kind][1]);
    let b = sub(upRef, scale(n, dot(upRef, n)));
    let a: V3;
    if (norm(b) < 1e-3) {
      a = unit(sub(rightRef, scale(n, dot(rightRef, n))));
      b = cross(n, a);
    } else {
      b = unit(b);
      a = cross(b, n);
    }
    if (dot(a, rightRef) < 0) a = scale(a, -1);
    const th = this.roll[kind];
    if (th) {
      const c = Math.cos(th), s = Math.sin(th);
      [a, b] = [add(scale(a, c), scale(b, s)), add(scale(a, -s), scale(b, c))];
    }
    return [a, b, n];
  }

  normal(kind: Kind): V3 {
    return this.axes(kind)[2];
  }

  origin(kind: Kind): V3 {
    const n = this.normal(kind);
    return add(this.vcenter, scale(n, dot(sub(this.center, this.vcenter), n)));
  }

  index(kind: Kind): number {
    return Math.round(dot(sub(this.center, this.vcenter), this.normal(kind)) / this.step) + this.half;
  }

  setIndex(kind: Kind, k: number): void {
    const n = this.normal(kind);
    this.center = add(this.center, scale(n, (k - this.index(kind)) * this.step));
    this.clamp();
    this.changed();
  }

  clamp(): void {
    this.center = this.center.map((v, i) => Math.min(Math.max(v, 0), this.extent[i])) as V3;
  }

  rotate(axis: V3, theta: number): void {
    this.R = mulMM(rot(axis, theta), this.R);
  }

  /** Plane pixel (row, column) ↔ image-frame mm. */
  toWorld(kind: Kind, r: number, c: number): V3 {
    const [a, b] = this.axes(kind);
    const u = (c - this.N) * this.res, v = (this.N - r) * this.res;
    return add(this.origin(kind), add(scale(a, u), scale(b, v)));
  }

  toPlane(kind: Kind, w: V3): [number, number] {
    const [a, b] = this.axes(kind);
    const d = sub(w, this.origin(kind));
    return [this.N - dot(d, b) / this.res, dot(d, a) / this.res + this.N];
  }

  planeKey(kind: Kind): string {
    const [a, , n] = this.axes(kind);
    const r3 = (v: V3) => v.map((x) => x.toFixed(4)).join(",");
    return `${kind}|${r3(n)}|${r3(a)}|${dot(this.origin(kind), n).toFixed(2)}|${this.slabMm}${this.slabMode}`;
  }

  /** The plane image (2N+1)², trilinear, slab combined; outside the volume = the lowest value. */
  sample(kind: Kind, out: Float32Array, fast: boolean): void {
    const vol = this.vol;
    const d = vol.data;
    const nx = vol.cols, ny = vol.rows, nz = vol.n, nxy = nx * ny;
    const [sz, sy, sx] = vol.spacing;
    const [a, b, n] = this.axes(kind);
    const o = this.origin(kind);
    const res = this.res, N = this.N, W = 2 * N + 1, lo = this.lo;
    // slab: samples along the normal, at most 24
    const ks = this.slabMm > 0 ? Math.max(2, Math.min(24, Math.round(this.slabMm / this.step))) : 1;
    const offs: V3[] = [];
    for (let j = 0; j < ks; j++) {
      const t = ks === 1 ? 0 : (j / (ks - 1) - 0.5) * this.slabMm;
      offs.push([n[0] * t / sx, n[1] * t / sy, n[2] * t / sz]);
    }
    const mode = this.slabMode;
    const stepC: V3 = [a[0] * res / sx, a[1] * res / sy, a[2] * res / sz];
    const by = fast ? 2 : 1;
    const xm = nx - 1, ym = ny - 1, zm = nz - 1;
    const tri = (X: number, Y: number, Z: number): number => {
      if (X < -0.5 || Y < -0.5 || Z < -0.5 || X > xm + 0.5 || Y > ym + 0.5 || Z > zm + 0.5) return NaN;
      X = X < 0 ? 0 : X > xm ? xm : X;
      Y = Y < 0 ? 0 : Y > ym ? ym : Y;
      Z = Z < 0 ? 0 : Z > zm ? zm : Z;
      const x0 = X | 0, y0 = Y | 0, z0 = Z | 0;
      const fx = X - x0, fy = Y - y0, fz = Z - z0;
      const dx = x0 < xm ? 1 : 0, dy = y0 < ym ? nx : 0, dz = z0 < zm ? nxy : 0;
      const i = z0 * nxy + y0 * nx + x0;
      const c00 = d[i] + (d[i + dx] - d[i]) * fx;
      const c10 = d[i + dy] + (d[i + dy + dx] - d[i + dy]) * fx;
      const c01 = d[i + dz] + (d[i + dz + dx] - d[i + dz]) * fx;
      const c11 = d[i + dz + dy] + (d[i + dz + dy + dx] - d[i + dz + dy]) * fx;
      const c0 = c00 + (c10 - c00) * fy, c1 = c01 + (c11 - c01) * fy;
      return c0 + (c1 - c0) * fz;
    };
    for (let r = 0; r < W; r += by) {
      const v = (N - r) * res, u0 = -N * res;
      const wx = o[0] + a[0] * u0 + b[0] * v, wy = o[1] + a[1] * u0 + b[1] * v, wz = o[2] + a[2] * u0 + b[2] * v;
      let X = wx / sx, Y = wy / sy, Z = wz / sz;
      const row = r * W;
      for (let c = 0; c < W; c += by, X += stepC[0] * by, Y += stepC[1] * by, Z += stepC[2] * by) {
        let val: number;
        if (ks === 1) {
          val = tri(X, Y, Z);
          if (val !== val) val = lo;
        } else {
          let acc = mode === "max" ? -Infinity : mode === "min" ? Infinity : 0, cnt = 0;
          for (const of of offs) {
            const s = tri(X + of[0], Y + of[1], Z + of[2]);
            if (s !== s) continue;
            cnt++;
            if (mode === "max") { if (s > acc) acc = s; } else if (mode === "min") { if (s < acc) acc = s; } else acc += s;
          }
          val = cnt ? (mode === "mean" ? acc / cnt : acc) : lo;
        }
        out[row + c] = val;
        if (by === 2) {
          if (c + 1 < W) out[row + c + 1] = val;
          if (r + 1 < W) {
            out[row + W + c] = val;
            if (c + 1 < W) out[row + W + c + 1] = val;
          }
        }
      }
    }
  }
}

/** One MPR plane as a viewport source. */
export class PlaneSource implements ViewSource {
  readonly n: number;
  readonly rows: number;
  readonly cols: number;
  readonly spacing: [number, number, number];
  readonly spacingKnown: boolean;
  readonly frameTimeMs = null;
  readonly modality: string;
  readonly monochrome1: boolean;
  readonly patient: string;
  readonly dsaHint = null;
  readonly rgb = null;
  readonly complete = true;
  readonly readyCount: number;
  readonly errors: string[] = [];
  onFrame: ((k: number) => void) | null = null;
  readonly ww: number | null;
  readonly wl: number | null;
  private img: Float32Array;
  private key = "";
  private fastDrawn = false;
  private ver = 0;

  constructor(readonly m: MPRModel, readonly kind: Kind) {
    const W = 2 * m.N + 1;
    this.n = 2 * m.half + 1;
    this.rows = this.cols = W;
    this.spacing = [m.step, m.res, m.res];
    this.spacingKnown = m.vol.spacingKnown;
    this.modality = m.vol.modality;
    this.monochrome1 = m.vol.monochrome1;
    this.patient = m.vol.patient;
    this.ww = m.vol.ww;
    this.wl = m.vol.wl;
    this.readyCount = this.n;
    this.img = new Float32Array(W * W);
  }

  get description(): string {
    const m = this.m;
    const slab = m.slabMm > 0 ? `   ${{ max: "MIP", min: "MinIP", mean: "Average" }[m.slabMode]} ${m.slabMm} mm` : "";
    return `MPR ${TITLES[this.kind]}${slab}`;
  }

  /** Bumped when the plane moved, turned or the slab changed — the viewport then windows the image again. */
  get version(): number {
    const k = this.m.planeKey(this.kind);
    if (k !== this.key || (this.fastDrawn && !this.m.fast)) {
      this.key = k;
      this.fastDrawn = this.m.fast;
      this.m.sample(this.kind, this.img, this.m.fast);
      this.ver++;
    }
    return this.ver;
  }

  frame(): Pixels {
    void this.version;
    return this.img;
  }

  rgbFrame(): null {
    return null;
  }

  isReady(): boolean {
    return true;
  }

  want(): Promise<void> {
    return Promise.resolve();
  }

  measureKey(): string {
    return this.m.planeKey(this.kind);
  }

  fitZoom(): number {
    return ((2 * this.m.N + 1) * this.m.res) / (Math.max(...this.m.extent) * 1.05);
  }
}

/** Axis lines, their handles and the drags that move / tilt the planes (dcv/mpr_window.py ObliqueView). */
export function mprHooks(m: MPRModel, kind: Kind, vp: Viewport, settle: () => void): ViewHooks {
  let drag: { what: "center" | "move" | "rotate"; other: Kind | null; x: number; y: number } | null = null;
  const toScreen = (w: V3) => vp.toScreen(...m.toPlane(kind, w));
  const toWorld = (x: number, y: number) => m.toWorld(kind, ...vp.toImage(x, y));
  const lines = () => {
    const n = m.normal(kind);
    const [cx, cy] = toScreen(m.center);
    return KINDS.filter((o) => o !== kind).map((other) => {
      const dir = unit(cross(n, m.normal(other)));
      const [px, py] = toScreen(add(m.center, scale(dir, 20)));
      const L = Math.hypot(px - cx, py - cy) || 1;
      return { other, cx, cy, dx: (px - cx) / L, dy: (py - cy) / L };
    });
  };
  const handleDist = () => {
    const r = vp.canvas.getBoundingClientRect();
    return Math.min(r.width, r.height) * 0.33;
  };
  const hit = (x: number, y: number, touch: boolean): { what: "center" | "move" | "rotate"; other: Kind | null } | null => {
    if (!m.showLines) return null;
    const ls = lines();
    const k = touch ? 2 : 1;
    if (Math.hypot(x - ls[0].cx, y - ls[0].cy) < 10 * k) return { what: "center", other: null };
    const hd = handleDist();
    for (const l of ls) {
      for (const s of [1, -1]) if (Math.hypot(x - (l.cx + s * l.dx * hd), y - (l.cy + s * l.dy * hd)) < 10 * k) return { what: "rotate", other: l.other };
    }
    if (vp.tool === "length") return null;          // with the length tool a line press starts a measurement
    for (const l of ls) if (Math.abs((x - l.cx) * l.dy - (y - l.cy) * l.dx) < 5 * k) return { what: "move", other: l.other };
    return null;
  };
  return {
    overlay(ctx) {
      const r = vp.canvas.getBoundingClientRect();
      const W = r.width, H = r.height;
      ctx.fillStyle = PLANE_COLORS[kind];
      ctx.fillRect(W - 16, 6, 10, 10);
      // patient orientation letters at the edges
      const [a, b] = m.axes(kind);
      ctx.font = "600 13px system-ui, -apple-system, sans-serif";
      ctx.fillStyle = "rgb(255,170,60)";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(letter(mulMV(m.D, b)), W / 2, 40);
      ctx.fillText(letter(mulMV(m.D, scale(b, -1))), W / 2, H - 40);
      ctx.fillText(letter(mulMV(m.D, scale(a, -1))), 14, H / 2);
      ctx.fillText(letter(mulMV(m.D, a)), W - 16, H / 2);
      ctx.textAlign = "left";
      ctx.textBaseline = "alphabetic";
      if (!m.showLines) return;
      const L = Math.max(W, H) * 2, hd = handleDist();
      ctx.lineWidth = 1.4;
      const ls = lines();
      for (const l of ls) {
        ctx.strokeStyle = PLANE_COLORS[l.other!];
        ctx.beginPath();
        ctx.moveTo(l.cx - l.dx * L, l.cy - l.dy * L);
        ctx.lineTo(l.cx + l.dx * L, l.cy + l.dy * L);
        ctx.stroke();
        for (const s of [1, -1]) {
          ctx.beginPath();
          ctx.arc(l.cx + s * l.dx * hd, l.cy + s * l.dy * hd, 6, 0, Math.PI * 2);
          ctx.stroke();
        }
      }
      ctx.fillStyle = PLANE_COLORS[kind];
      ctx.beginPath();
      ctx.arc(ls[0].cx, ls[0].cy, 4, 0, Math.PI * 2);
      ctx.fill();
    },
    down(x, y, e) {
      const h = hit(x, y, e.pointerType !== "mouse");
      if (!h) return false;
      drag = { ...h, x, y };
      m.fast = true;
      return true;
    },
    drag(x, y) {
      if (!drag) return;
      const w0 = toWorld(drag.x, drag.y), w1 = toWorld(x, y);
      const delta = sub(w1, w0);
      if (drag.what === "center") m.center = add(m.center, delta);
      else if (drag.what === "move") {
        const no = m.normal(drag.other!);
        m.center = add(m.center, scale(no, dot(delta, no)));
      } else {
        const [cx, cy] = toScreen(m.center);
        const a0 = Math.atan2(-(drag.y - cy), drag.x - cx), a1 = Math.atan2(-(y - cy), x - cx);
        const [a, b] = m.axes(kind);
        m.rotate(cross(a, b), a1 - a0);
      }
      drag.x = x;
      drag.y = y;
      m.clamp();
      m.changed();
    },
    end() {
      drag = null;
      settle();
    },
    hover(x, y) {
      const h = hit(x, y, false);
      return h ? (h.what === "rotate" ? "crosshair" : "move") : null;
    },
    index(k) {
      m.fast = true;
      m.setIndex(kind, k);
      settle();
    },
  };
}
