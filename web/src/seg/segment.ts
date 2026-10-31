/**
 * Seed segmentation (dcv/segment.py region_grow / auto_grow): the voxels connected to a seed with values in a range,
 * searched in a block of ±70 mm around it. The anti-leak step works as in the desktop (erode the thresholded block by
 * one voxel so thin bridges break, take what is connected to the seed, dilate back within the threshold), but voxels
 * are only looked at as the region reaches them: no threshold or erosion images of the whole block, so the time
 * follows the size of the segment and the range can be re-grown while it is dragged.
 */

/** A cropped binary mask: one byte per voxel of the segment's bounding box (a 30 ml tumour: a few MB). */
export interface Grown {
  sub: Uint8Array;
  off: [number, number, number];      // (z, y, x) of its first voxel
  dims: [number, number, number];     // (nz, ny, nx)
  count: number;
}

const RADIUS_MM = 70;

export class RegionGrower {
  private readonly x0: number;
  private readonly y0: number;
  private readonly z0: number;
  private readonly bw: number;
  private readonly bh: number;
  private readonly bd: number;
  private flags: Uint8Array | null = null;     // the block: 1 region, 2 not in the eroded set, 4 dilated, 8 final
  private queue: Int32Array = new Int32Array(1 << 16);

  /** spacing = (z, y, x) mm, seed = (z, y, x) voxel. */
  constructor(readonly data: ArrayLike<number>, readonly nx: number, readonly ny: number, readonly nz: number,
    readonly spacing: [number, number, number], readonly seed: [number, number, number], radiusMm = RADIUS_MM) {
    const [sz, sy, sx] = spacing;
    const [z, y, x] = seed;
    const rz = Math.ceil(radiusMm / sz), ry = Math.ceil(radiusMm / sy), rx = Math.ceil(radiusMm / sx);
    this.z0 = Math.max(0, z - rz);
    this.y0 = Math.max(0, y - ry);
    this.x0 = Math.max(0, x - rx);
    this.bd = Math.min(nz, z + rz + 1) - this.z0;
    this.bh = Math.min(ny, y + ry + 1) - this.y0;
    this.bw = Math.min(nx, x + rx + 1) - this.x0;
  }

  /** Bytes of the search block held between grows. */
  get blockBytes(): number {
    return this.bw * this.bh * this.bd;
  }

  private dataIndex(z: number, y: number, x: number): number {
    return ((this.z0 + z) * this.ny + this.y0 + y) * this.nx + this.x0 + x;
  }

  grow(lo: number, hi: number, antiLeak = 1): Grown | null {
    const { data, bw, bh, bd, nx } = this;
    const nxy = nx * this.ny;
    const sz = this.seed[0] - this.z0, sy = this.seed[1] - this.y0, sx = this.seed[2] - this.x0;
    const seedValue = data[this.dataIndex(sz, sy, sx)];
    if (!(seedValue >= lo && seedValue <= hi)) return null;
    const flags = (this.flags ??= new Uint8Array(bw * bh * bd));
    flags.fill(0);
    const bwh = bw * bh;
    const seedBlock = sz * bwh + sy * bw + sx;
    // anti-leak radii: in-plane voxels; z scaled so the distance in mm is the same
    const rx = antiLeak, ry = antiLeak, rz = antiLeak > 0 ? Math.round((antiLeak * this.spacing[2]) / this.spacing[0]) : 0;

    /** True when every voxel of the box around one (inside the block) is in range: the eroded threshold. */
    const eroded = (z: number, y: number, x: number): boolean => {
      const zb = z - rz < 0 ? 0 : z - rz, ze = z + rz >= bd ? bd - 1 : z + rz;
      const yb = y - ry < 0 ? 0 : y - ry, ye = y + ry >= bh ? bh - 1 : y + ry;
      const xb = x - rx < 0 ? 0 : x - rx, xe = x + rx >= bw ? bw - 1 : x + rx;
      for (let k = zb; k <= ze; k++) {
        for (let j = yb; j <= ye; j++) {
          const row = (this.z0 + k) * nxy + (this.y0 + j) * nx + this.x0;
          for (let i = xb; i <= xe; i++) {
            const v = data[row + i];
            if (!(v >= lo && v <= hi)) return false;
          }
        }
      }
      return true;
    };

    let queue = this.queue;
    const useEroded = antiLeak > 0 && eroded(sz, sy, sx);
    let tail = 0;
    if (useEroded) {
      // what is connected to the seed in the eroded threshold
      flags[seedBlock] = 1;
      queue[tail++] = seedBlock;
      for (let head = 0; head < tail; head++) {
        const b = queue[head];
        const z = Math.floor(b / bwh), rest = b - z * bwh, y = Math.floor(rest / bw), x = rest - y * bw;
        if (tail + 6 > queue.length) queue = this.queue = grown32(queue);
        for (let n = 0; n < 6; n++) {
          const nz = z + DZ[n], ny = y + DY[n], nx2 = x + DX[n];
          if (nz < 0 || ny < 0 || nx2 < 0 || nz >= bd || ny >= bh || nx2 >= bw) continue;
          const c = nz * bwh + ny * bw + nx2;
          if (flags[c]) continue;
          if (eroded(nz, ny, nx2)) {
            flags[c] = 1;
            queue[tail++] = c;
          } else flags[c] = 2;
        }
      }
      // dilated back, within the threshold
      for (let head = 0; head < tail; head++) {
        const b = queue[head];
        const z = Math.floor(b / bwh), rest = b - z * bwh, y = Math.floor(rest / bw), x = rest - y * bw;
        for (let k = Math.max(0, z - rz); k <= Math.min(bd - 1, z + rz); k++) {
          for (let j = Math.max(0, y - ry); j <= Math.min(bh - 1, y + ry); j++) {
            const row = k * bwh + j * bw;
            const dat = (this.z0 + k) * nxy + (this.y0 + j) * nx + this.x0;
            for (let i = Math.max(0, x - rx); i <= Math.min(bw - 1, x + rx); i++) {
              const f = flags[row + i];
              if (f & 4) continue;
              const v = data[dat + i];
              if (v >= lo && v <= hi) flags[row + i] = f | 4;
            }
          }
        }
      }
    }

    // the final region: connected to the seed in the dilated set (or, with no anti-leak, in the threshold)
    tail = 0;
    flags[seedBlock] |= 8;
    queue[tail++] = seedBlock;
    let x0 = bw, x1 = 0, y0 = bh, y1 = 0, z0 = bd, z1 = 0;
    for (let head = 0; head < tail; head++) {
      const b = queue[head];
      const z = Math.floor(b / bwh), rest = b - z * bwh, y = Math.floor(rest / bw), x = rest - y * bw;
      if (x < x0) x0 = x;
      if (x > x1) x1 = x;
      if (y < y0) y0 = y;
      if (y > y1) y1 = y;
      if (z < z0) z0 = z;
      if (z > z1) z1 = z;
      if (tail + 6 > queue.length) queue = this.queue = grown32(queue);
      for (let n = 0; n < 6; n++) {
        const nz = z + DZ[n], ny = y + DY[n], nx2 = x + DX[n];
        if (nz < 0 || ny < 0 || nx2 < 0 || nz >= bd || ny >= bh || nx2 >= bw) continue;
        const c = nz * bwh + ny * bw + nx2;
        const f = flags[c];
        if (f & 8) continue;
        if (useEroded) {
          if (!(f & 4)) continue;
        } else {
          const v = data[this.dataIndex(nz, ny, nx2)];
          if (!(v >= lo && v <= hi)) continue;
        }
        flags[c] = f | 8;
        queue[tail++] = c;
      }
    }
    const w = x1 - x0 + 1, h = y1 - y0 + 1, d = z1 - z0 + 1;
    const sub = new Uint8Array(w * h * d);
    for (let i = 0; i < tail; i++) {
      const b = queue[i];
      const z = Math.floor(b / bwh), rest = b - z * bwh, y = Math.floor(rest / bw), x = rest - y * bw;
      sub[((z - z0) * h + y - y0) * w + x - x0] = 1;
    }
    return { sub, off: [this.z0 + z0, this.y0 + y0, this.x0 + x0], dims: [d, h, w], count: tail };
  }

  /** Mean and SD around the seed: ±2 voxels in-plane, ±1 slice (local_stats). */
  localStats(radius = 2): [number, number] {
    const [z, y, x] = this.seed;
    let n = 0, sum = 0, sq = 0;
    for (let k = Math.max(0, z - 1); k <= Math.min(this.nz - 1, z + 1); k++) {
      for (let j = Math.max(0, y - radius); j <= Math.min(this.ny - 1, y + radius); j++) {
        const row = (k * this.ny + j) * this.nx;
        for (let i = Math.max(0, x - radius); i <= Math.min(this.nx - 1, x + radius); i++) {
          const v = this.data[row + i];
          n++;
          sum += v;
          sq += v * v;
        }
      }
    }
    const m = sum / n;
    return [m, Math.sqrt(Math.max(0, sq / n - m * m))];
  }

  /** Automatic mode: range from the statistics around the seed (mean ± k·SD), refined once (auto_grow). */
  autoGrow(k = 2.5, antiLeak = 1): { grown: Grown | null; lo: number; hi: number } {
    const [m, s0] = this.localStats();
    const s = Math.max(s0, 4);
    const lo = m - k * s, hi = m + k * s;
    const res = this.grow(lo, hi, antiLeak);
    if (res && res.count > 50) {
      const [d, h, w] = res.dims, [oz, oy, ox] = res.off;
      let n = 0, sum = 0, sq = 0;
      for (let z = 0; z < d; z++) {
        for (let y = 0; y < h; y++) {
          const row = ((oz + z) * this.ny + oy + y) * this.nx + ox, mr = (z * h + y) * w;
          for (let x = 0; x < w; x++) {
            if (!res.sub[mr + x]) continue;
            const v = this.data[row + x];
            n++;
            sum += v;
            sq += v * v;
          }
        }
      }
      const m2 = sum / n, s2 = Math.max(Math.sqrt(Math.max(0, sq / n - m2 * m2)), 4);
      let lo2 = m2 - k * s2, hi2 = m2 + k * s2;
      const w0 = hi - lo;
      if (hi2 - lo2 > 1.5 * w0) {                 // do not let the range widen too much (prevents gradual leaking)
        const c = (lo2 + hi2) / 2;
        lo2 = c - 0.75 * w0;
        hi2 = c + 0.75 * w0;
      }
      const res2 = this.grow(lo2, hi2, antiLeak);
      if (res2) return { grown: res2, lo: lo2, hi: hi2 };
    }
    return { grown: res, lo, hi };
  }

  /**
   * Values around the seed (a cube of ±radius mm, at most ~1 M voxels sampled): `bins` counts over [min, max],
   * for the range editor.
   */
  histogram(radiusMm = 30, bins = 2048): { counts: Float32Array; min: number; max: number } {
    const [z, y, x] = this.seed, [sz, sy, sx] = this.spacing;
    const z0 = Math.max(0, z - Math.ceil(radiusMm / sz)), z1 = Math.min(this.nz - 1, z + Math.ceil(radiusMm / sz));
    const y0 = Math.max(0, y - Math.ceil(radiusMm / sy)), y1 = Math.min(this.ny - 1, y + Math.ceil(radiusMm / sy));
    const x0 = Math.max(0, x - Math.ceil(radiusMm / sx)), x1 = Math.min(this.nx - 1, x + Math.ceil(radiusMm / sx));
    const total = (z1 - z0 + 1) * (y1 - y0 + 1) * (x1 - x0 + 1);
    const every = Math.max(1, Math.round(Math.cbrt(total / 1e6)));
    let min = Infinity, max = -Infinity;
    const visit = (f: (v: number) => void) => {
      for (let k = z0; k <= z1; k += every) {
        for (let j = y0; j <= y1; j += every) {
          const row = (k * this.ny + j) * this.nx;
          for (let i = x0; i <= x1; i += every) f(this.data[row + i]);
        }
      }
    };
    visit((v) => {
      if (v < min) min = v;
      if (v > max) max = v;
    });
    if (max <= min) max = min + 1;
    const counts = new Float32Array(bins);
    const c = bins / (max - min);
    visit((v) => counts[Math.min(bins - 1, Math.floor((v - min) * c))]++);
    return { counts, min, max };
  }
}

const DZ = [-1, 1, 0, 0, 0, 0], DY = [0, 0, -1, 1, 0, 0], DX = [0, 0, 0, 0, -1, 1];

function grown32(q: Int32Array): Int32Array {
  const out = new Int32Array(q.length * 2);
  out.set(q);
  return out;
}

/** A dragged handle of the range editor: the value at the seed stays inside, so the segment never disappears. */
export function dragRange(handle: "lo" | "hi", v: number, lo: number, hi: number, seedValue: number): [number, number] {
  if (handle === "lo") return [Math.min(v, hi - 1, seedValue), hi];
  return [lo, Math.max(v, lo + 1, seedValue)];
}

export function volumeMl(count: number, spacing: [number, number, number]): number {
  return (count * spacing[0] * spacing[1] * spacing[2]) / 1000;
}

export function fmtMl(ml: number): string {
  return `${ml.toFixed(ml < 10 ? 2 : 1)} ml`;
}

/**
 * The columns of the range editor: the part of the histogram [view0, view1] resampled to `width` values, log(1 + n),
 * scaled to 0..1.
 */
export function histColumns(h: { counts: Float32Array; min: number; max: number }, view0: number, view1: number, width: number): Float32Array {
  const out = new Float32Array(width);
  const B = h.counts.length, k = B / (h.max - h.min);
  let top = 0;
  for (let i = 0; i < width; i++) {
    const a = (view0 + ((view1 - view0) * i) / width - h.min) * k, b = (view0 + ((view1 - view0) * (i + 1)) / width - h.min) * k;
    let sum = 0;
    for (let j = Math.max(0, Math.floor(a)); j < Math.min(B, Math.max(Math.floor(a) + 1, Math.ceil(b))); j++) sum += h.counts[j];
    out[i] = Math.log1p(sum);
    if (out[i] > top) top = out[i];
  }
  if (top > 0) for (let i = 0; i < width; i++) out[i] /= top;
  return out;
}
