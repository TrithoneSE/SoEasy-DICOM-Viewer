/**
 * Digital subtraction angiography for XA / fluoroscopy runs — a port of dcv/dsa.py (same maths, same defaults):
 * subtraction on / off, the displayed frame as the mask, reset, manual pixel-shift and auto pixel-shift in a square
 * around the tap.
 *
 * A mask given by the scanner (Mask Subtraction Sequence, AVG_SUB: mean of the mask frames, Mask Sub-pixel Shift)
 * is used when present, else frame 2. LOG / DISP data are already logarithmic, so live − mask is the attenuation
 * of the contrast; LIN data are log-transformed first.
 */
import type { DsaHint } from "./dicom/header";

export const LOG_SCALE = 1000;        // LIN data: log difference × this, so a window of a few hundred shows the vessels
// The display is linear in live − mask with WW ≈ 0.77 × the run's window and "no change" at 64 % grey;
// 0.6 × gives a little more contrast, as the owner asked (dcv/dsa.py).
export const DSA_WW_FACTOR = 0.6;
export const DSA_BACKGROUND = 0.64;

/** What DSA needs from a loaded run: frames as stored values (after rescale), row-major. */
export interface Frames {
  n: number;
  rows: number;
  cols: number;
  frame(k: number): ArrayLike<number>;
  isReady?(k: number): boolean;
}

/** img moved by (dy, dx) pixels (rows down, columns right), bilinear for sub-pixel shifts, edges repeated. */
export function shiftImage(img: Float32Array, h: number, w: number, dy: number, dx: number,
  out = new Float32Array(h * w)): Float32Array {
  const iy = Math.floor(dy), ix = Math.floor(dx);
  const fy = dy - iy, fx = dx - ix;
  const w00 = (1 - fy) * (1 - fx), w01 = (1 - fy) * fx, w10 = fy * (1 - fx), w11 = fy * fx;
  const c0 = new Int32Array(w), c1 = new Int32Array(w);
  for (let c = 0; c < w; c++) {                       // out[r, c] = img[r - oy, c - ox], clamped at the edges
    c0[c] = Math.min(w - 1, Math.max(0, c - ix));
    c1[c] = Math.min(w - 1, Math.max(0, c - ix - 1));
  }
  for (let r = 0; r < h; r++) {
    const r0 = Math.min(h - 1, Math.max(0, r - iy)) * w;
    const r1 = Math.min(h - 1, Math.max(0, r - iy - 1)) * w;
    const o = r * w;
    if (fx === 0 && fy === 0) {
      for (let c = 0; c < w; c++) out[o + c] = img[r0 + c0[c]];
    } else {
      for (let c = 0; c < w; c++) {
        out[o + c] = img[r0 + c0[c]] * w00 + img[r0 + c1[c]] * w01 + img[r1 + c0[c]] * w10 + img[r1 + c1[c]] * w11;
      }
    }
  }
  return out;
}

const round = (v: number, d: number) => Math.round(v * 10 ** d) / 10 ** d;

export class DSA {
  readonly defaultMask: number[];
  readonly defaultShift: [number, number];
  readonly log: boolean;
  mask: number[];
  shift: [number, number];
  private cache: { key: string; img: Float32Array } | null = null;
  private out: Float32Array;

  constructor(private vol: Frames, hint: DsaHint | null) {
    const n = vol.n;
    this.defaultMask = hint?.mask?.length ? [...hint.mask] : [Math.min(1, n - 1)];    // the second frame
    this.defaultShift = hint ? [hint.shift[0], hint.shift[1]] : [0, 0];
    this.log = !!hint?.log;
    this.mask = [...this.defaultMask];
    this.shift = [...this.defaultShift];
    this.out = new Float32Array(vol.rows * vol.cols);
  }

  setMask(frames: number[]): void {
    this.mask = frames.map((f) => Math.round(f));
  }

  /** Back to the default mask and its original shift. */
  reset(): void {
    this.mask = [...this.defaultMask];
    this.shift = [...this.defaultShift];
  }

  move(dy: number, dx: number): void {
    this.shift = [round(this.shift[0] + dy, 3), round(this.shift[1] + dx, 3)];
  }

  /** Frames the subtraction needs decoded before frame k can be shown. */
  needs(k: number): number[] {
    return [...this.mask, k];
  }

  private prep(k: number): Float32Array {
    const f = this.vol.frame(k);
    const out = new Float32Array(f.length);
    if (this.log) for (let i = 0; i < f.length; i++) out[i] = Math.log(Math.max(f[i], 0) + 1) * LOG_SCALE;
    else for (let i = 0; i < f.length; i++) out[i] = f[i];
    return out;
  }

  private base(): Float32Array {
    const m = this.prep(this.mask[0]);
    for (const f of this.mask.slice(1)) {
      const g = this.prep(f);
      for (let i = 0; i < m.length; i++) m[i] += g[i];
    }
    if (this.mask.length > 1) for (let i = 0; i < m.length; i++) m[i] /= this.mask.length;
    return m;
  }

  maskImage(): Float32Array {
    const key = `${this.mask.join(",")}|${this.shift[0]},${this.shift[1]}`;
    if (!this.cache || this.cache.key !== key) {
      const m = this.base();
      const img = this.shift[0] || this.shift[1] ? shiftImage(m, this.vol.rows, this.vol.cols, ...this.shift) : m;
      this.cache = { key, img };
    }
    return this.cache.img;
  }

  /** Frame k minus the (shifted) mask: about 0 for background, negative where contrast attenuates.
   *  The result is a buffer reused by the next call unless `fresh`. */
  subtract(k: number, fresh = false): Float32Array {
    const m = this.maskImage();
    const out = fresh ? new Float32Array(m.length) : this.out;
    const f = this.vol.frame(k);
    if (this.log) for (let i = 0; i < m.length; i++) out[i] = Math.log(Math.max(f[i], 0) + 1) * LOG_SCALE - m[i];
    else for (let i = 0; i < m.length; i++) out[i] = f[i] - m[i];
    return out;
  }

  /** [ww, wl] of the subtracted run: DSA_WW_FACTOR × the window the run was shown with, "no change" at
   *  DSA_BACKGROUND grey. LIN data (other units): 5 × the 99th percentile of |live − mask| in a few frames. */
  suggestWindow(nativeWW: number | null, current?: number): [number, number] {
    let ww: number;
    if (!this.log && nativeWW) {
      ww = DSA_WW_FACTOR * nativeWW;
    } else {
      const n = this.vol.n;
      const ks = new Set<number>();
      for (let i = 0; i < Math.min(n, 8); i++) ks.add(Math.round((i * (n - 1)) / Math.max(1, Math.min(n, 8) - 1)));
      let list = [...ks].filter((k) => !this.vol.isReady || this.vol.isReady(k));
      if (current !== undefined) list.push(current);
      if (!list.length) list = [n - 1];
      let p = 0;
      for (const k of new Set(list)) p = Math.max(p, percentileAbs(this.subtract(k), 0.99));
      ww = Math.max(5 * p, 10);
    }
    return [ww, -(DSA_BACKGROUND - 0.5) * ww];
  }

  label(): [string, string] {
    const m = this.mask.map((f) => f + 1).join("+");
    const f = (v: number) => (v >= 0 ? "+" : "") + v.toFixed(1);
    return [`Mask: ${m}`, `Shift: x ${f(this.shift[1])}  y ${f(this.shift[0])} px`];
  }

  /** Half side (image px) of the auto pixel-shift area (dcv/slice_view.py _dsa_square). */
  square(): number {
    return Math.max(16, Math.floor(Math.min(this.vol.rows, this.vol.cols) / 10));
  }

  /** Mask shift that best removes the background in the square (centre [row, col], half side) of frame k: least
   *  total variation of live − mask there (misaligned bone edges leave strong edges; vessel contrast is the same
   *  for every shift). Integer search ±`search` px around the current shift, then 0.1 px steps. */
  autoShift(k: number, center: [number, number], half: number, search = 12): [number, number] {
    const live = this.prep(k);
    const base = this.base();
    const h = this.vol.rows, w = this.vol.cols;
    half = Math.floor(Math.min(half, Math.floor(h / 2) - 1, Math.floor(w / 2) - 1));
    const r = Math.min(Math.max(Math.round(center[0]), half), h - half - 1);
    const c = Math.min(Math.max(Math.round(center[1]), half), w - half - 1);
    const m = half + search + 3;
    const r0 = Math.max(0, r - m), r1 = Math.min(h, r + m + 1), c0 = Math.max(0, c - m), c1 = Math.min(w, c + m + 1);
    const ch = r1 - r0, cw = c1 - c0;
    const liveC = new Float32Array(ch * cw), baseC = new Float32Array(ch * cw);
    for (let y = 0; y < ch; y++) {
      liveC.set(live.subarray((r0 + y) * w + c0, (r0 + y) * w + c1), y * cw);
      baseC.set(base.subarray((r0 + y) * w + c0, (r0 + y) * w + c1), y * cw);
    }
    const ys = r - half - r0, xs = c - half - c0, side = 2 * half + 1;
    const shifted = new Float32Array(ch * cw);
    const d = new Float32Array(side * side);
    const cost = (dy: number, dx: number) => {
      shiftImage(baseC, ch, cw, dy, dx, shifted);
      for (let y = 0; y < side; y++) {
        const o = (ys + y) * cw + xs;
        for (let x = 0; x < side; x++) d[y * side + x] = liveC[o + x] - shifted[o + x];
      }
      let s = 0;
      for (let y = 0; y < side; y++) {
        const o = y * side;
        for (let x = 0; x < side - 1; x++) s += Math.abs(d[o + x + 1] - d[o + x]);
        if (y < side - 1) for (let x = 0; x < side; x++) s += Math.abs(d[o + side + x] - d[o + x]);
      }
      return s;
    };
    const y0 = Math.round(this.shift[0]), x0 = Math.round(this.shift[1]);
    let best = [Infinity, y0, x0];
    for (let dy = -search; dy <= search; dy++) {
      for (let dx = -search; dx <= search; dx++) {
        const v = cost(y0 + dy, x0 + dx);
        if (v < best[0]) best = [v, y0 + dy, x0 + dx];
      }
    }
    const [, by, bx] = best;
    let fine = [Infinity, by, bx];
    for (let dy = -9; dy <= 9; dy++) {
      for (let dx = -9; dx <= 9; dx++) {
        const v = cost(by + dy / 10, bx + dx / 10);
        if (v < fine[0]) fine = [v, by + dy / 10, bx + dx / 10];
      }
    }
    this.shift = [round(fine[1], 2), round(fine[2], 2)];
    return this.shift;
  }
}

/** q-quantile of |a| (on a sample of at most ~200k values). */
export function percentileAbs(a: ArrayLike<number>, q: number): number {
  const step = Math.max(1, Math.floor(a.length / 200000));
  const s = new Float32Array(Math.ceil(a.length / step));
  for (let i = 0, j = 0; i < a.length; i += step, j++) s[j] = Math.abs(a[i]);
  s.sort();
  return s[Math.min(s.length - 1, Math.floor(q * (s.length - 1)))] ?? 0;
}

/** [lo, hi] quantiles of a (sampled), for auto window and thumbnails. */
export function quantiles(a: ArrayLike<number>, lo: number, hi: number): [number, number] {
  const step = Math.max(1, Math.floor(a.length / 200000));
  const s = new Float32Array(Math.ceil(a.length / step));
  for (let i = 0, j = 0; i < a.length; i += step, j++) s[j] = a[i];
  s.sort();
  const at = (q: number) => s[Math.min(s.length - 1, Math.max(0, Math.round(q * (s.length - 1))))] ?? 0;
  return [at(lo), at(hi)];
}
