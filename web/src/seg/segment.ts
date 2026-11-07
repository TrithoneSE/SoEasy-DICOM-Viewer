/**
 * Seed segmentation (dcv/segment.py region_grow / auto_grow): the voxels connected to a seed with values in a range,
 * searched in a block of ±70 mm around it. The anti-leak cut works as in the desktop: contacts narrower than twice
 * ANTI_LEAK_MM are cut (erode, take what is connected to the seed, grow back to the edge through the range), on values
 * smoothed by CUT_SMOOTH_MM, as on 0.4 mm voxels the noise otherwise breaks the core apart (real CT: a sinonasal tumour
 * touching muscle of the same HU grew to 209 ml from a ~20 ml mass). As in the desktop, a small window around the click
 * is grown first and a larger one only when the segment reaches its side, and the cut erodes the window one axis at a
 * time: a lesion costs tens of milliseconds, not the whole block (a 0.625 mm CT froze an iPhone for minutes).
 */

export const ANTI_LEAK_MM = 1.6;
const CUT_SMOOTH_MM = 0.7;
const SQRT3 = Math.sqrt(3);

/** A cropped binary mask: one byte per voxel of the segment's bounding box (a 30 ml tumour: a few MB). */
export interface Grown {
  sub: Uint8Array;
  off: [number, number, number];      // (z, y, x) of its first voxel
  dims: [number, number, number];     // (nz, ny, nx)
  count: number;
}

const RADIUS_MM = 70;
/** Windows around the click tried before the whole block (dcv/segment.py WINDOWS_MM). */
const WINDOWS_MM = [10, 20, 35];
/** tissueGuard: the neighbourhood whose tissues bound the automatic range (dcv/segment.py GUARD_MM). */
const GUARD_MM = 30;
/** autoGrow: how far the first look (the segment's values, for the refined range) may reach from the click. */
const FIRST_LOOK_MM = 20;
// flags bits: 1 core reached from the click, 8 final; CORE / THR the eroded threshold and its working copy
const CORE = 16, THR = 32;

/** One window of the block: a box around the click (block voxels), with its smoothed values once they are made. */
interface Win {
  z0: number;
  y0: number;
  x0: number;
  d: number;
  h: number;
  w: number;
  s: [number, number, number];        // the click in the window
  inner: [boolean, boolean][];        // per axis (z, y, x): its low / high side lies inside the block
  smooth: Float32Array | Int16Array | null;
}

export class RegionGrower {
  private readonly x0: number;
  private readonly y0: number;
  private readonly z0: number;
  private readonly bw: number;
  private readonly bh: number;
  private readonly bd: number;
  private readonly radiusMm: number;
  private flags: Uint8Array | null = null;     // the window grown in (bits above)
  private readonly wins = new Map<string, Win>();      // the last few windows, with their smoothed values
  // per side (z low, z high, y low, y high, x low, x high): the size the last grow needed; a dragged handle starts there
  private lastIdx = [0, 0, 0, 0, 0, 0];
  private queue: Int32Array = new Int32Array(1 << 16);

  /** spacing = (z, y, x) mm, seed = (z, y, x) voxel. */
  constructor(readonly data: ArrayLike<number>, readonly nx: number, readonly ny: number, readonly nz: number,
    readonly spacing: [number, number, number], readonly seed: [number, number, number], radiusMm = RADIUS_MM) {
    const [sz, sy, sx] = spacing;
    const [z, y, x] = seed;
    const rz = Math.ceil(radiusMm / sz), ry = Math.ceil(radiusMm / sy), rx = Math.ceil(radiusMm / sx);
    this.radiusMm = radiusMm;
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

  /** The window with sides at sizes()[idx[k]] mm from the click (z low, z high, y low, ...), within the block. */
  private window(idx: number[]): Win {
    const key = idx.join();
    const hit = this.wins.get(key);
    if (hit) {
      this.wins.delete(key);                      // most recent last
      this.wins.set(key, hit);
      return hit;
    }
    const sizes = this.sizes();
    const s = [this.seed[0] - this.z0, this.seed[1] - this.y0, this.seed[2] - this.x0];
    const n = [this.bd, this.bh, this.bw];
    const a0 = [0, 0, 0], a1 = [0, 0, 0];
    for (let a = 0; a < 3; a++) {
      a0[a] = Math.max(0, s[a] - Math.ceil(sizes[idx[2 * a]] / this.spacing[a]));
      a1[a] = Math.min(n[a], s[a] + Math.ceil(sizes[idx[2 * a + 1]] / this.spacing[a]) + 1);
    }
    const win: Win = {
      z0: a0[0], y0: a0[1], x0: a0[2], d: a1[0] - a0[0], h: a1[1] - a0[1], w: a1[2] - a0[2],
      s: [s[0] - a0[0], s[1] - a0[1], s[2] - a0[2]],
      inner: [0, 1, 2].map((a) => [a0[a] > 0, a1[a] < n[a]] as [boolean, boolean]), smooth: null,
    };
    this.wins.set(key, win);
    // a phone's memory: a dragged handle reuses the last window; older ones go
    while (this.wins.size > 3) this.wins.delete(this.wins.keys().next().value!);
    return win;
  }

  /**
   * The window's values smoothed by CUT_SMOOTH_MM ([1 2 1] / 4 passes per axis, as smooth_block), made on its first
   * grow. Smoothed on the window plus the voxels the passes reach, so they are what smoothing the whole block gives.
   */
  private smoothed(win: Win): Float32Array | Int16Array {
    if (win.smooth) return win.smooth;
    const passes = this.spacing.map((k) => Math.round(2 * (CUT_SMOOTH_MM / k) ** 2));
    const b0 = [win.z0, win.y0, win.x0], bn = [win.d, win.h, win.w], n = [this.bd, this.bh, this.bw];
    const e0 = b0.map((v, a) => Math.max(0, v - passes[a]));
    const e1 = b0.map((v, a) => Math.min(n[a], v + bn[a] + passes[a]));
    const ed = e1[0] - e0[0], eh = e1[1] - e0[1], ew = e1[2] - e0[2];
    // half the memory for a CT (~55 MB for the full ±70 mm block of a 0.4 mm scan); 1 HU steps are plenty for a range
    const int = this.data instanceof Int16Array;
    const box = int ? new Int16Array(ew * eh * ed) : new Float32Array(ew * eh * ed);
    for (let z = 0; z < ed; z++) {
      for (let y = 0; y < eh; y++) {
        const src = this.dataIndex(e0[0] + z, e0[1] + y, e0[2]), dst = (z * eh + y) * ew;
        for (let x = 0; x < ew; x++) box[dst + x] = this.data[src + x];
      }
    }
    blurLines(box, ew, eh * ed, passes[2], int);
    blurStrided(box, ed, ew * eh, eh, ew, passes[1], int);
    blurStrided(box, 1, 0, ed, ew * eh, passes[0], int);
    if (ed === win.d && eh === win.h && ew === win.w) return (win.smooth = box);
    const out = int ? new Int16Array(win.w * win.h * win.d) : new Float32Array(win.w * win.h * win.d);
    for (let z = 0; z < win.d; z++) {
      for (let y = 0; y < win.h; y++) {
        const src = ((win.z0 - e0[0] + z) * eh + win.y0 - e0[1] + y) * ew + win.x0 - e0[2];
        out.set(box.subarray(src, src + win.w), (z * win.h + y) * win.w);
      }
    }
    return (win.smooth = out);
  }

  /** The windows' sizes (mm), the last one the block. */
  private sizes(): number[] {
    return [...WINDOWS_MM.filter((w) => w < this.radiusMm), this.radiusMm];
  }

  /**
   * The cut: contacts narrower than twice `antiLeak` mm break; 0 = off (thin vessels). Grown in a small window around
   * the click first; a side of it moves out to the next size only when the segment comes near that side (region_grow
   * does it with cubes): the time follows the segment, not the ±70 mm block (real CT, a frontal tumour 52 mm above the
   * click: a quarter of the block), and what a window holds away from its sides is what the whole block gives.
   */
  grow(lo: number, hi: number, antiLeak = ANTI_LEAK_MM): Grown | null {
    return this.growUpTo(lo, hi, antiLeak, Infinity);
  }

  /**
   * grow, with no side beyond sizes()[cap]: what is held there is only part of a larger segment (autoGrow's first look,
   * which needs its values, not all of it). A side still too narrow there starts the next grow one size further.
   */
  private growUpTo(lo: number, hi: number, antiLeak: number, cap: number): Grown | null {
    const sizes = this.sizes(), top = sizes.length - 1, lim = Math.min(top, cap);
    const margin = Math.max(...this.spacing.map((k) => Math.max(1, Math.round(antiLeak / k)))) + 2;
    const idx = this.lastIdx.map((i) => Math.min(i, lim));
    const stuck = [false, false, false, false, false, false];
    for (;;) {
      const win = this.window(idx);
      const tail = this.growIn(win, lo, hi, antiLeak);
      if (tail < 0) return null;
      const { w: bw, h: bh, d: bd } = win, bwh = bw * bh, queue = this.queue;
      let x0 = bw, x1 = 0, y0 = bh, y1 = 0, z0 = bd, z1 = 0;
      for (let q = 0; q < tail; q++) {
        const b = queue[q];
        const z = Math.floor(b / bwh), rest = b - z * bwh, y = Math.floor(rest / bw), x = rest - y * bw;
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
        if (z < z0) z0 = z;
        if (z > z1) z1 = z;
      }
      // a side the segment comes near (inside the block) moves out to the next size; the others stay
      let wider = false;
      [[z0, z1, bd], [y0, y1, bh], [x0, x1, bw]].forEach(([a, b, n], ax) => {
        const near = [win.inner[ax][0] && a < margin, win.inner[ax][1] && b >= n - margin];
        for (const side of [0, 1]) {
          if (near[side] && idx[2 * ax + side] < lim) {
            idx[2 * ax + side]++;
            wider = true;
          }
          stuck[2 * ax + side] = near[side] && idx[2 * ax + side] >= lim;
        }
      });
      if (wider) continue;
      const w = x1 - x0 + 1, h = y1 - y0 + 1, d = z1 - z0 + 1;
      const sub = new Uint8Array(w * h * d);
      for (let q = 0; q < tail; q++) {
        const b = queue[q];
        const z = Math.floor(b / bwh), rest = b - z * bwh, y = Math.floor(rest / bw), x = rest - y * bw;
        sub[((z - z0) * h + y - y0) * w + x - x0] = 1;
      }
      const off: [number, number, number] = [this.z0 + win.z0 + z0, this.y0 + win.y0 + y0, this.x0 + win.x0 + x0];
      // per side, the smallest size that would have held it starts the next grow
      const need = (vox: number, a: number) => {
        const j = sizes.findIndex((s) => s >= Math.abs(vox) * this.spacing[a] + margin * Math.max(...this.spacing));
        return j < 0 ? top : j;
      };
      for (let a = 0; a < 3; a++) {
        const o = off[a], m = [d, h, w][a], c = this.seed[a];
        this.lastIdx[2 * a] = stuck[2 * a] ? Math.min(top, lim + 1) : need(c - o, a);
        this.lastIdx[2 * a + 1] = stuck[2 * a + 1] ? Math.min(top, lim + 1) : need(o + m - 1 - c, a);
      }
      return { sub, off, dims: [d, h, w], count: tail };
    }
  }

  /** region_grow inside one window: the segment's voxels (window indices) in this.queue[0, n) → n, or -1. */
  private growIn(win: Win, lo: number, hi: number, antiLeak: number): number {
    const { w: bw, h: bh, d: bd } = win, bwh = bw * bh, nxy = this.nx * this.ny;
    const [sz, sy, sx] = win.s;
    const zb = this.z0 + win.z0, yb = this.y0 + win.y0, xb = this.x0 + win.x0;
    const data = this.data;
    const own = (z: number, y: number, x: number) => data[(zb + z) * nxy + (yb + y) * this.nx + xb + x];
    const S = antiLeak > 0 ? this.smoothed(win) : null;
    const seedBlock = sz * bwh + sy * bw + sx;
    const seedValue = own(sz, sy, sx);
    if (!(seedValue >= lo && seedValue <= hi) && !(S && S[seedBlock] >= lo && S[seedBlock] <= hi)) return -1;
    const n = bwh * bd;
    if (!this.flags || this.flags.length < n) this.flags = new Uint8Array(n);
    const flags = this.flags;
    /** Smoothed or own value in range: the way back out to the edge, which smoothing blurs into the background. */
    const edgeIn = (b: number, z: number, y: number, x: number) => {
      const v = own(z, y, x);
      if (v >= lo && v <= hi) return true;
      return S !== null && S[b] >= lo && S[b] <= hi;
    };

    let queue = this.queue;
    let tail = 0;
    const push = (c: number) => {
      if (tail + 1 > queue.length) queue = this.queue = grown32(queue);
      queue[tail++] = c;
    };

    // core: connected to the click through voxels whose whole box of ±r is in range (smoothed); then back out through
    // the range one box step at a time: follows the shape, no box corners. A narrower cut when the click finds no core.
    let cut = antiLeak, done = false;
    while (S && cut > 0 && !done) {
      const rx = Math.max(1, Math.round(cut / this.spacing[2])), ry = Math.max(1, Math.round(cut / this.spacing[1]));
      const rz = Math.max(1, Math.round(cut / this.spacing[0]));
      for (let i = 0; i < n; i++) flags[i] = S[i] >= lo && S[i] <= hi ? THR : 0;
      // box_erode, one axis at a time (the box ends at the window's sides, as it did at the block's)
      erodeLines(flags, bw, bh * bd, rx, THR, CORE);
      erodeStrided(flags, bd, bwh, bh, bw, ry, CORE, THR);
      erodeStrided(flags, 1, 0, bd, bwh, rz, THR, CORE);
      // the nearest core voxel to the click, within twice the cut
      let start = -1, best = Infinity;
      for (let k = Math.max(0, sz - 2 * rz); k <= Math.min(bd - 1, sz + 2 * rz); k++) {
        for (let j = Math.max(0, sy - 2 * ry); j <= Math.min(bh - 1, sy + 2 * ry); j++) {
          for (let i = Math.max(0, sx - 2 * rx); i <= Math.min(bw - 1, sx + 2 * rx); i++) {
            const d = ((k - sz) * this.spacing[0]) ** 2 + ((j - sy) * this.spacing[1]) ** 2 + ((i - sx) * this.spacing[2]) ** 2;
            if (d < best && flags[k * bwh + j * bw + i] & CORE) {
              best = d;
              start = k * bwh + j * bw + i;
            }
          }
        }
      }
      if (start < 0) {
        cut = Math.max(rx, ry) > 1 ? cut / 2 : 0;
        continue;
      }
      tail = 0;
      flags[start] |= 1 | 8;
      push(start);
      for (let head = 0; head < tail; head++) {
        const b = queue[head];
        const z = Math.floor(b / bwh), rest = b - z * bwh, y = Math.floor(rest / bw), x = rest - y * bw;
        for (let q = 0; q < 6; q++) {
          const nz = z + DZ[q], ny = y + DY[q], nx2 = x + DX[q];
          if (nz < 0 || ny < 0 || nx2 < 0 || nz >= bd || ny >= bh || nx2 >= bw) continue;
          const c = nz * bwh + ny * bw + nx2;
          if (flags[c] & 1 || !(flags[c] & CORE)) continue;
          flags[c] |= 1 | 8;
          push(c);
        }
      }
      // one box step at a time, as the desktop's box_dilate: diagonals too, or the corners of a round lesion stay out
      // √3 · r steps: a box of ±r reaches only so far into a round surface, so r steps would leave a shell out
      const back = (k: number) => Math.ceil(SQRT3 * k);
      let front = queue.slice(0, tail);
      for (let step = 0; step < back(Math.max(rx, ry, rz)); step++) {
        const dz = step < back(rz) ? 1 : 0, dy = step < back(ry) ? 1 : 0, dx = step < back(rx) ? 1 : 0;
        const first = tail;
        for (let f = 0; f < front.length; f++) {
          const b = front[f];
          const z = Math.floor(b / bwh), rest = b - z * bwh, y = Math.floor(rest / bw), x = rest - y * bw;
          for (let k = Math.max(0, z - dz); k <= Math.min(bd - 1, z + dz); k++) {
            for (let j = Math.max(0, y - dy); j <= Math.min(bh - 1, y + dy); j++) {
              for (let i = Math.max(0, x - dx); i <= Math.min(bw - 1, x + dx); i++) {
                const c = k * bwh + j * bw + i;
                if (flags[c] & 8 || !edgeIn(c, k, j, i)) continue;
                flags[c] |= 8;
                push(c);
              }
            }
          }
        }
        front = queue.slice(first, tail);
      }
      done = true;
    }
    if (!done) {
      // thinner than any cut: what is connected in the range
      flags.fill(0, 0, n);
      tail = 0;
      if (!edgeIn(seedBlock, sz, sy, sx)) return -1;
      flags[seedBlock] = 8;
      push(seedBlock);
      for (let head = 0; head < tail; head++) {
        const b = queue[head];
        const z = Math.floor(b / bwh), rest = b - z * bwh, y = Math.floor(rest / bw), x = rest - y * bw;
        for (let q = 0; q < 6; q++) {
          const nz = z + DZ[q], ny = y + DY[q], nx2 = x + DX[q];
          if (nz < 0 || ny < 0 || nx2 < 0 || nz >= bd || ny >= bh || nx2 >= bw) continue;
          const c = nz * bwh + ny * bw + nx2;
          if (flags[c] & 8 || !edgeIn(c, nz, ny, nx2)) continue;
          flags[c] = 8;
          push(c);
        }
      }
    }
    return tail;
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

  /**
   * [lowest lo, highest hi] the automatic range may take on CT, so it stays out of the large tissue next to the seed's
   * own (tissue_guard). A heterogeneous enhancing tumour has a wide SD, so mean − 2.5·SD reached ~38 HU and the segment
   * ran into the brain (~35 HU): real CT, a 0.625 mm frontal tumour grew to 100–700 ml and froze a phone. On the
   * smoothed values within GUARD_MM, the histogram peak nearest the seed's level is its tissue; the range stops at the
   * valley between it and the largest peak below (above): ~56 HU there, the tumour alone.
   */
  tissueGuard(): [number, number] {
    const sizes = this.sizes();
    const j = sizes.findIndex((w) => w >= GUARD_MM), i = j < 0 ? sizes.length - 1 : j;
    const win = this.window([i, i, i, i, i, i]), S = this.smoothed(win);
    const { w: bw, h: bh, d: bd } = win, [sz, sy, sx] = win.s;
    const r = (mm: number, a: number) => Math.max(1, Math.round(mm / this.spacing[a]));
    // the seed's level: median of the smoothed values ±2 mm around it
    const near: number[] = [];
    for (let z = Math.max(0, sz - r(2, 0)); z <= Math.min(bd - 1, sz + r(2, 0)); z++)
      for (let y = Math.max(0, sy - r(2, 1)); y <= Math.min(bh - 1, sy + r(2, 1)); y++)
        for (let x = Math.max(0, sx - r(2, 2)); x <= Math.min(bw - 1, sx + r(2, 2)); x++) near.push(S[(z * bh + y) * bw + x]);
    near.sort((a, b) => a - b);
    const m = near[near.length >> 1], atSeed = S[(sz * bh + sy) * bw + sx];
    // histogram (1 HU bins from -1024) of the smoothed values within GUARD_MM, itself smoothed (σ 2 HU)
    const base = -1024, raw = new Float64Array(4096);
    const g = (a: number) => Math.ceil(GUARD_MM / this.spacing[a]);
    for (let z = Math.max(0, sz - g(0)); z <= Math.min(bd - 1, sz + g(0)); z++)
      for (let y = Math.max(0, sy - g(1)); y <= Math.min(bh - 1, sy + g(1)); y++) {
        const row = (z * bh + y) * bw;
        for (let x = Math.max(0, sx - g(2)); x <= Math.min(bw - 1, sx + g(2)); x++)
          raw[Math.min(4095, Math.max(0, Math.round(S[row + x]) - base))]++;
      }
    raw[0] = raw[4095] = 0;                           // out of the HU scale
    const kern = Array.from({ length: 13 }, (_, j) => Math.exp(-0.5 * ((j - 6) / 2) ** 2));
    const ks = kern.reduce((a, b) => a + b);
    const h = new Float64Array(4096);
    let total = 0;
    for (let b = 0; b < 4096; b++) {
      let v = 0;
      for (let j = 0; j < 13; j++) {
        const q = b + j - 6;
        if (q >= 0 && q < 4096) v += raw[q] * kern[j];
      }
      total += (h[b] = v / ks);
    }
    const peaks: number[] = [];
    for (let b = 3; b < 4093; b++) {
      if (h[b] <= 0.002 * total) continue;
      let top = true;
      for (let j = -3; j <= 3 && top; j++) if (h[b + j] > h[b]) top = false;
      if (top) peaks.push(b);
    }
    let lo = -Infinity, hi = Infinity;
    if (!peaks.length) return [lo, hi];
    const mb = Math.round(m) - base;
    const own = peaks.reduce((a, b) => (Math.abs(b - mb) < Math.abs(a - mb) ? b : a));
    const largest = (l: number[]) => l.reduce((a, b) => (h[b] > h[a] ? b : a));
    const valley = (a: number, b: number) => {
      let at = a;
      for (let q = a; q <= b; q++) if (h[q] < h[at]) at = q;
      return at;
    };
    const below = peaks.filter((p) => p < own), above = peaks.filter((p) => p > own);
    if (below.length) {
      const v = base + valley(largest(below), own);
      if (v < Math.min(m, atSeed)) lo = v;
    }
    if (above.length) {
      const v = base + valley(own, largest(above));
      if (v > Math.max(m, atSeed)) hi = v;
    }
    return [lo, hi];
  }

  /** Automatic mode: range from the statistics around the seed (mean ± k·SD), refined once (auto_grow). guard (CT):
   * the range stays out of the neighbouring tissue (tissueGuard). */
  autoGrow(k = 2.5, antiLeak = ANTI_LEAK_MM, guard = false): { grown: Grown | null; lo: number; hi: number } {
    const [gLo, gHi] = guard ? this.tissueGuard() : [-Infinity, Infinity];
    const [m, s0] = this.localStats();
    const s = Math.max(s0, 4);
    const lo = Math.max(m - k * s, gLo), hi = Math.min(m + k * s, gHi);
    // the first look only gives the values of the segment: up to FIRST_LOOK_MM around the click, not all of a large one
    // (a phone: the whole of a 100 ml tumour twice took twice as long)
    const sizes = this.sizes(), j = sizes.findIndex((w) => w >= FIRST_LOOK_MM);
    const res = this.growUpTo(lo, hi, antiLeak, j < 0 ? sizes.length - 1 : j);
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
      lo2 = Math.max(lo2, gLo);
      hi2 = Math.min(hi2, gHi);
      const res2 = this.grow(lo2, hi2, antiLeak);
      if (res2) return { grown: res2, lo: lo2, hi: hi2 };
    }
    return { grown: res && this.grow(lo, hi, antiLeak), lo, hi };       // all of it this time
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

/** What Segments needs of a grower. */
export type Grower = Pick<RegionGrower, "grow" | "autoGrow" | "histogram">;

/** In-plane pixels below this (mm) grow on the halved grid on phones and tablets (HalfGrower). */
export const HALF_GRID_BELOW_MM = 0.6;

/**
 * A RegionGrower on the grid halved in-plane (the mean of 2 × 2 pixels; slices unchanged), as the bone area on phones:
 * a quarter of the work for a thin-pixel CT (0.44 → 0.88 mm). Real CT, a 100 ml frontal tumour on a 0.625 mm series:
 * ~30 s on an iPhone at full size. The mask comes back at full size, each cell as its 2 × 2 pixels.
 */
export class HalfGrower implements Grower {
  private readonly g: RegionGrower;
  private readonly z0: number;
  private readonly y0: number;
  private readonly x0: number;

  constructor(data: ArrayLike<number>, readonly nx: number, readonly ny: number, nz: number,
    spacing: [number, number, number], seed: [number, number, number], radiusMm = RADIUS_MM) {
    const [sz, sy, sx] = spacing, [z, y, x] = seed;
    // only the block the grower searches, starting on an even pixel so the cells line up with the seed's
    this.z0 = Math.max(0, z - Math.ceil(radiusMm / sz));
    this.y0 = Math.max(0, y - Math.ceil(radiusMm / sy)) & ~1;
    this.x0 = Math.max(0, x - Math.ceil(radiusMm / sx)) & ~1;
    const z1 = Math.min(nz, z + Math.ceil(radiusMm / sz) + 1);
    const y1 = Math.min(ny, y + Math.ceil(radiusMm / sy) + 1), x1 = Math.min(nx, x + Math.ceil(radiusMm / sx) + 1);
    const d = z1 - this.z0, h = Math.ceil((y1 - this.y0) / 2), w = Math.ceil((x1 - this.x0) / 2);
    const int = data instanceof Int16Array;
    const half = int ? new Int16Array(w * h * d) : new Float32Array(w * h * d);
    for (let k = 0; k < d; k++) {
      for (let j = 0; j < h; j++) {
        const ya = this.y0 + 2 * j, yb = Math.min(ny - 1, ya + 1);
        const ra = ((this.z0 + k) * ny + ya) * nx, rb = ((this.z0 + k) * ny + yb) * nx, o = (k * h + j) * w;
        for (let i = 0; i < w; i++) {
          const xa = this.x0 + 2 * i, xb = Math.min(nx - 1, xa + 1);
          const v = (data[ra + xa] + data[ra + xb] + data[rb + xa] + data[rb + xb]) / 4;
          half[o + i] = int ? Math.round(v) : v;
        }
      }
    }
    this.g = new RegionGrower(half, w, h, d, [sz, 2 * sy, 2 * sx],
      [z - this.z0, (y - this.y0) >> 1, (x - this.x0) >> 1], radiusMm);
  }

  /** A mask on the halved grid at full size. */
  private full(r: Grown | null): Grown | null {
    if (!r) return null;
    const [d, h, w] = r.dims, [oz, oy, ox] = r.off;
    const Y0 = this.y0 + 2 * oy, X0 = this.x0 + 2 * ox;
    const H = Math.min(2 * h, this.ny - Y0), W = Math.min(2 * w, this.nx - X0);
    const sub = new Uint8Array(d * H * W);
    let count = 0;
    for (let z = 0; z < d; z++) {
      for (let y = 0; y < H; y++) {
        const src = (z * h + (y >> 1)) * w, dst = (z * H + y) * W;
        for (let x = 0; x < W; x++) {
          if (r.sub[src + (x >> 1)]) {
            sub[dst + x] = 1;
            count++;
          }
        }
      }
    }
    return { sub, off: [this.z0 + oz, Y0, X0], dims: [d, H, W], count };
  }

  grow(lo: number, hi: number, antiLeak = ANTI_LEAK_MM): Grown | null {
    return this.full(this.g.grow(lo, hi, antiLeak));
  }

  autoGrow(k = 2.5, antiLeak = ANTI_LEAK_MM, guard = false): { grown: Grown | null; lo: number; hi: number } {
    const r = this.g.autoGrow(k, antiLeak, guard);
    return { ...r, grown: this.full(r.grown) };
  }

  histogram(radiusMm = 30, bins = 2048): { counts: Float32Array; min: number; max: number } {
    return this.g.histogram(radiusMm, bins);
  }
}

const DZ = [-1, 1, 0, 0, 0, 0], DY = [0, 0, -1, 1, 0, 0], DX = [0, 0, 0, 0, -1, 1];

function grown32(q: Int32Array): Int32Array {
  const out = new Int32Array(q.length * 2);
  out.set(q);
  return out;
}

/** The kernel of `passes` [1 2 1] / 4 passes (smooth_block), in one: the binomial weights C(2p, k) / 4^p. */
function binomial(passes: number): Float32Array {
  let w = [1];
  for (let p = 0; p < 2 * passes; p++) w = [...w, 0].map((v, k) => (v + (k > 0 ? w[k - 1] : 0)) / 2);
  return Float32Array.from(w);
}

/** That kernel along `lines` contiguous lines of `len` (the ends repeat); Int16 rounds. */
function blurLines(box: Float32Array | Int16Array, len: number, lines: number, passes: number, int: boolean): void {
  if (passes <= 0) return;
  if (int) return blurLinesI16(box as Int16Array, len, lines, passes);
  const wts = binomial(passes), taps = wts.length;
  const a = new Float32Array(len + 2 * passes);
  for (let l = 0; l < lines; l++) {
    const o = l * len;
    for (let i = 0; i < a.length; i++) a[i] = box[o + Math.min(len - 1, Math.max(0, i - passes))];
    for (let i = 0; i < len; i++) {
      let v = 0;
      for (let k = 0; k < taps; k++) v += wts[k] * a[i + k];
      box[o + i] = int ? Math.round(v) : v;
    }
  }
}

/** The same along an axis of `n` rows of `seg` contiguous values (y: rows of a slice, z: whole slices), row by row:
 * a ring holds the original rows the kernel reaches, so memory access stays contiguous. */
function blurStrided(box: Float32Array | Int16Array, groups: number, groupStride: number, n: number, seg: number,
  passes: number, int: boolean): void {
  if (passes <= 0) return;
  if (int) return blurStridedI16(box as Int16Array, groups, groupStride, n, seg, passes);
  const wts = binomial(passes), taps = wts.length;
  const ring = Array.from({ length: taps }, () => new Float32Array(seg));
  const acc = new Float32Array(seg);
  const row = (r: number) => Math.min(n - 1, Math.max(0, r)) * seg;
  for (let g = 0; g < groups; g++) {
    const base = g * groupStride;
    for (let k = 0; k < taps; k++) ring[k].set(box.subarray(base + row(k - passes), base + row(k - passes) + seg));
    for (let r = 0; r < n; r++) {
      acc.fill(0);
      for (let k = 0; k < taps; k++) {
        const wk = wts[k], rk = ring[(r + k) % taps];
        for (let x = 0; x < seg; x++) acc[x] += wk * rk[x];
      }
      const o = base + r * seg;
      if (int) for (let x = 0; x < seg; x++) box[o + x] = Math.round(acc[x]);
      else box.set(acc, o);
      // the row the next one reaches is still the original: only rows up to r are written
      const q = base + row(r + passes + 1);
      ring[r % taps].set(box.subarray(q, q + seg));
    }
  }
}

/** blurLines on CT (Int16): integer weights C(2p, k) and one rounding shift, which a phone runs much faster. */
function blurLinesI16(box: Int16Array, len: number, lines: number, passes: number): void {
  const wts = Int32Array.from(binomial(passes), (v) => Math.round(v * 4 ** passes)), taps = wts.length;
  const sh = 2 * passes, half = 1 << (sh - 1);
  const a = new Int32Array(len + 2 * passes);
  for (let l = 0; l < lines; l++) {
    const o = l * len;
    for (let i = 0; i < a.length; i++) a[i] = box[o + Math.min(len - 1, Math.max(0, i - passes))];
    for (let i = 0; i < len; i++) {
      let v = 0;
      for (let k = 0; k < taps; k++) v += wts[k] * a[i + k];
      box[o + i] = (v + half) >> sh;
    }
  }
}

/** blurStrided on CT (Int16), the same way. */
function blurStridedI16(box: Int16Array, groups: number, groupStride: number, n: number, seg: number, passes: number): void {
  const wts = Int32Array.from(binomial(passes), (v) => Math.round(v * 4 ** passes)), taps = wts.length;
  const sh = 2 * passes, half = 1 << (sh - 1);
  const ring = Array.from({ length: taps }, () => new Int16Array(seg));
  const acc = new Int32Array(seg);
  const row = (r: number) => Math.min(n - 1, Math.max(0, r)) * seg;
  for (let g = 0; g < groups; g++) {
    const base = g * groupStride;
    for (let k = 0; k < taps; k++) ring[k].set(box.subarray(base + row(k - passes), base + row(k - passes) + seg));
    for (let r = 0; r < n; r++) {
      acc.fill(half);
      for (let k = 0; k < taps; k++) {
        const wk = wts[k], rk = ring[(r + k) % taps];
        for (let x = 0; x < seg; x++) acc[x] += wk * rk[x];
      }
      const o = base + r * seg;
      for (let x = 0; x < seg; x++) box[o + x] = acc[x] >> sh;
      const q = base + row(r + passes + 1);
      ring[r % taps].set(box.subarray(q, q + seg));
    }
  }
}

/** Bit `dst` where every voxel within ±r along contiguous lines has bit `src` (box_erode along one axis). */
function erodeLines(f: Uint8Array, len: number, lines: number, r: number, src: number, dst: number): void {
  for (let l = 0; l < lines; l++) {
    const o = l * len;
    let miss = 0;                    // voxels of [i - r, i + r] without src
    for (let i = 0; i < Math.min(r, len); i++) if (!(f[o + i] & src)) miss++;
    for (let i = 0; i < len; i++) {
      if (i + r < len && !(f[o + i + r] & src)) miss++;
      if (i - r - 1 >= 0 && !(f[o + i - r - 1] & src)) miss--;
      f[o + i] = miss === 0 ? f[o + i] | dst : f[o + i] & ~dst;
    }
  }
}

/** The same along an axis of `n` rows of `seg` contiguous voxels, a row at a time. */
function erodeStrided(f: Uint8Array, groups: number, groupStride: number, n: number, seg: number, r: number,
  src: number, dst: number): void {
  const miss = new Int32Array(seg);
  const count = (o: number, d: number) => {
    for (let x = 0; x < seg; x++) if (!(f[o + x] & src)) miss[x] += d;
  };
  for (let g = 0; g < groups; g++) {
    const base = g * groupStride;
    miss.fill(0);
    for (let k = 0; k < Math.min(r, n); k++) count(base + k * seg, 1);
    for (let k = 0; k < n; k++) {
      if (k + r < n) count(base + (k + r) * seg, 1);
      if (k - r - 1 >= 0) count(base + (k - r - 1) * seg, -1);
      const o = base + k * seg;
      for (let x = 0; x < seg; x++) f[o + x] = miss[x] === 0 ? f[o + x] | dst : f[o + x] & ~dst;
    }
  }
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
