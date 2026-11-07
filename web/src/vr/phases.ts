/**
 * Arterial and venous phase of one CTA (dcv/phases.py; its comment has the measurements). One series cannot tell arteries
 * from veins: per voxel the time course can, and the anatomy where the "arterial" series was late and the sinuses already
 * full. Seeds: as bright as the large arteries and washing out = artery; brighter on the venous series, keeping more of
 * its enhancement than this patient's arteries, or a long thick vessel against the skull in the upper head = vein. Both grow through the vessels and meet where the trees touch. Bone
 * is what stays bright on both series, and what is already bright before contrast is no vessel. On a grid of about 1 mm.
 */
import { classifyHead } from "./atlas";
import { dilateMask, erodeMask, pack, type BoneRegion, type VesselMasks } from "./bone";
import type { Volume } from "../dicom/volume";

const MAX_SHIFT_MM = 6, PEAK_MAX_HU = 1000, VESSEL_MIN_HU = 150, DIFF_MIN_HU = 60, PIECE_MIN_MM3 = 8, ARTERY_SHARE = 0.85, ARTERY_SEED_DEPTH_MM = 70;
const SINUS_THICK_MM = 3, SINUS_REACH_MM = 3, SINUS_DEPTH_MM = 130, SINUS_MIN_MM3 = 150;
const WASHOUT_MARGIN = 0.2, WASHOUT_MIN_MM3 = 60, UNENHANCED_HU = 40;
// a late scan (dcv/phases.py _late_scan) and where its seeds go; bone that moved between the phases
const LATE_GAP = 0.12, LATE_MIN_ML = 2, ARTERY_BASE_DEPTH_MM = 90, ARTERY_BASE_FRONT = 0.55, AORTA_DEPTH_MM = 270;
const JUGULAR_GAP_MM = 14, MOVED_BONE_HU = 700, MOVED_BONE_MM = 3, SKULL_BASE_HALF_MM = 30;
const VENOUS = /(^|[^A-Z])(VENA|VENOUS|VENOSA|VENOGRAM|VEN|CTV|DELAY|DELAYED|LATE)($|[^A-Z])/;
const ARTERIAL = /(^|[^A-Z])(ARTER|ARTERI|ARTERIAL|ARTERY)/;
const NATIVE = /(^|[^A-Z])(PRE|NK|NC|NON ?CONTRAST|PLAIN|NATIVE|UNENHANCED|POLOS)($|[^A-Z])/;

export type Phase = "arterial" | "venous" | "native" | null;

/**
 * HU of the brightest 0.5 % inside the skull, 15-90 mm under the vertex, away from bone: the vessels and sinuses
 * (dcv/phases.py intracranial_top). Null with too little head in the scan. Real CTs: without contrast 42-66 HU, with
 * contrast 74-258. On ~1 mm cells: the mean of each f × f × fz block, as the desktop's grid.
 */
export function intracranialTop(vol: Volume): number | null {
  const { cols, rows, n } = vol, d = vol.data;
  const f = Math.max(1, Math.round(1 / vol.spacing[2])), fz = Math.max(1, Math.round(1 / vol.spacing[0]));
  const nx = Math.floor(cols / f), ny = Math.floor(rows / f), nz = Math.floor(n / fz);
  const mean = new Float32Array(nx * ny * nz), top = new Float32Array(nx * ny * nz).fill(-Infinity);
  for (let z = 0; z < nz * fz; z++) {
    for (let y = 0; y < ny * f; y++) {
      const o = (Math.floor(z / fz) * ny + Math.floor(y / f)) * nx, row = (z * rows + y) * cols;
      for (let x = 0; x < nx * f; x++) {
        const c = o + Math.floor(x / f), v = d[row + x];
        mean[c] += v / (f * f * fz);
        if (v > top[c]) top[c] = v;
      }
    }
  }
  const sz = vol.spacing[0] * fz, sxy = vol.spacing[2] * f, nxy = nx * ny;
  let vtx = -1;
  for (let z = nz - 1; z >= 0 && vtx < 0; z--) {
    let c = 0;
    for (let i = z * nxy; i < (z + 1) * nxy; i++) if (top[i] >= 500) c++;
    if (c >= 20) vtx = z;
  }
  if (vtx < 0) return null;
  const z0 = Math.max(0, vtx - Math.floor(90 / sz)), z1 = Math.max(1, vtx - Math.floor(15 / sz));
  const r = Math.max(1, Math.round(3 / sxy)), rz = Math.max(1, Math.round(3 / sz));
  const vals: number[] = [];
  for (let z = z0; z < z1; z++) {
    const bone = (y: number, x: number) => top[(z * ny + y) * nx + x] >= 500;
    // between bone on both sides, along x and along y
    const xl = new Int32Array(ny).fill(nx), xr = new Int32Array(ny).fill(-1), yt = new Int32Array(nx).fill(ny), yb = new Int32Array(nx).fill(-1);
    for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) if (bone(y, x)) {
      if (x < xl[y]) xl[y] = x;
      if (x > xr[y]) xr[y] = x;
      if (y < yt[x]) yt[x] = y;
      if (y > yb[x]) yb[x] = y;
    }
    for (let y = 0; y < ny; y++) {
      for (let x = xl[y] + 1; x < xr[y]; x++) {
        if (y <= yt[x] || y >= yb[x]) continue;
        const i = (z * ny + y) * nx + x;
        if (mean[i] <= 0) continue;
        let near = false;                                   // away from bone: 3 mm
        for (let k = Math.max(0, z - rz); k <= Math.min(nz - 1, z + rz) && !near; k++) {
          for (let j = Math.max(0, y - r); j <= Math.min(ny - 1, y + r) && !near; j++) {
            for (let q = Math.max(0, x - r); q <= Math.min(nx - 1, x + r); q++) if (top[(k * ny + j) * nx + q] >= 500) { near = true; break; }
          }
        }
        if (!near) vals.push(mean[i]);
      }
    }
  }
  if (vals.length < 20000) return null;
  vals.sort((p, q) => p - q);
  return vals[Math.floor(0.995 * (vals.length - 1))];
}

/** Both contrast phases? A plain head CT next to its '+ CM' series is not: pairing them painted a whole skull red and
 *  blue. Unknown (no head in the scan) counts as a pair: the names decided it. */
export function isPhasePair(a: Volume, b: Volume): boolean {
  const ta = intracranialTop(a), tb = intracranialTop(b);
  return ta === null || tb === null || Math.min(ta, tb) >= 70;
}

export function phaseOfName(description: string): Phase {
  const d = (description || "").toUpperCase();
  return NATIVE.test(d) ? "native" : VENOUS.test(d) ? "venous" : ARTERIAL.test(d) ? "arterial" : null;
}

export interface PhaseResult extends VesselMasks {
  bone: BoneRegion;
  /** Voxel values shown for each: arteries from the arterial series, veins from the venous one. */
  artData: ArrayLike<number>;
  venData: ArrayLike<number>;
  /** Both series after the arterial peak: anatomy decided (dcv/phases.py classify_phases). */
  late: boolean;
}

type D3 = [number, number, number];

/** Mean and max of each cell of the in-plane step `f` and every `fz`-th plane pair. */
export function pool(data: ArrayLike<number>, cols: number, rows: number, n: number, f: number, fz: number) {
  const nx = Math.floor(cols / f), ny = Math.floor(rows / f), nz = Math.floor(n / fz);
  const mean = new Float32Array(nx * ny * nz), top = new Float32Array(nx * ny * nz).fill(-Infinity);
  const share = 1 / (f * f * fz);
  if (f === 2) {
    // the same sums in the same order, two voxels a step and no division per voxel (~3× quicker on a 111 M voxel CTA)
    for (let z = 0; z < nz * fz; z++) {
      const Z = Math.floor(z / fz);
      for (let y = 0; y < ny * 2; y++) {
        const o = (Z * ny + (y >> 1)) * nx, row = (z * rows + y) * cols;
        for (let i = 0; i < nx; i++) {
          const a = data[row + 2 * i], b = data[row + 2 * i + 1], c = o + i;
          mean[c] += a * share;
          mean[c] += b * share;
          let t = top[c];
          if (a > t) t = a;
          if (b > t) t = b;
          top[c] = t;
        }
      }
    }
    return { mean, top, d: [nx, ny, nz] as D3 };
  }
  for (let z = 0; z < nz * fz; z++) {
    const Z = Math.floor(z / fz);
    for (let y = 0; y < ny * f; y++) {
      const o = (Z * ny + Math.floor(y / f)) * nx, row = (z * rows + y) * cols;
      for (let x = 0; x < nx * f; x++) {
        const c = o + Math.floor(x / f), v = data[row + x];
        mean[c] += v * share;
        if (v > top[c]) top[c] = v;
      }
    }
  }
  return { mean, top, d: [nx, ny, nz] as D3 };
}

/** A (2r + 1) box along each axis, on bits (bone.ts): the same cells as a running count, far fewer steps. */
function dilate(m: Uint8Array, d: D3, r: D3): Uint8Array {
  return dilateMask(m, d, r);
}

/** Erode = dilate the inverse; outside the grid counts as inside the mask. */
function erode(m: Uint8Array, d: D3, r: D3): Uint8Array {
  return erodeMask(m, d, r);
}

/** Each 6-connected piece of a mask: `visit` gets its voxels. */
export function pieces(mask: Uint8Array, [nx, ny, nz]: D3, visit: (list: Int32Array, n: number) => void): void {
  const seen = new Uint8Array(mask.length);
  let list = new Int32Array(1 << 16);
  const nxy = nx * ny;
  for (let s = 0; s < mask.length; s++) {
    if (!mask[s] || seen[s]) continue;
    seen[s] = 1;
    list[0] = s;
    let n = 1;
    for (let h = 0; h < n; h++) {
      if (n + 6 > list.length) {
        const more = new Int32Array(list.length * 2);
        more.set(list);
        list = more;
      }
      const i = list[h], x = i % nx, y = Math.floor(i / nx) % ny, z = Math.floor(i / nxy);
      if (x > 0 && mask[i - 1] && !seen[i - 1]) { seen[i - 1] = 1; list[n++] = i - 1; }
      if (x < nx - 1 && mask[i + 1] && !seen[i + 1]) { seen[i + 1] = 1; list[n++] = i + 1; }
      if (y > 0 && mask[i - nx] && !seen[i - nx]) { seen[i - nx] = 1; list[n++] = i - nx; }
      if (y < ny - 1 && mask[i + nx] && !seen[i + nx]) { seen[i + nx] = 1; list[n++] = i + nx; }
      if (z > 0 && mask[i - nxy] && !seen[i - nxy]) { seen[i - nxy] = 1; list[n++] = i - nxy; }
      if (z < nz - 1 && mask[i + nxy] && !seen[i + nxy]) { seen[i + nxy] = 1; list[n++] = i + nxy; }
    }
    visit(list, n);
  }
}

/** vesselTrees: the atlas share of arteries for an artery seed (1 - this for a vein seed), and the lumen that thick
 *  around it (dcv/phases.py TREE_*). */
const TREE_SURE = 0.9, TREE_ARTERY_MM = 1.2, TREE_VEIN_MM = 1.5;

/**
 * Arteries and veins of ONE contrast series as two trees that compete for its vessels (dcv/phases.py vessel_trees):
 * veins from the traced sinuses and the thick vessels the atlas is sure are veins, arteries from the thick ones it is
 * sure are arteries; every vessel goes to the tree that reaches it first along the vessels (compete). Real CTAs, the
 * venous series alone against its own two phases: 84 % of the head's vessel cells right with the atlas, 88 % as trees.
 * Vessels neither reaches keep their label. In place on arteries / veins (cells of grid d, spacing sp: x, y, z mm).
 */
export function vesselTrees(arteries: Uint8Array, veins: Uint8Array, sinus: Uint8Array, prior: (m: Uint8Array) => Float32Array,
  d: D3, sp: number[]): void {
  const N = arteries.length, vessel = new Uint8Array(N);
  for (let i = 0; i < N; i++) vessel[i] = arteries[i] | veins[i];
  const pa = prior(vessel), thickA = thick(vessel, d, sp, TREE_ARTERY_MM), thickV = thick(vessel, d, sp, TREE_VEIN_MM);
  const labels = new Uint8Array(N);
  let nA = 0, nV = 0;
  for (let i = 0; i < N; i++) {
    if (!vessel[i]) labels[i] = 3;
    else if (sinus[i] || (pa[i] <= 1 - TREE_SURE && thickV[i])) {
      labels[i] = 2;
      nV++;
    } else if (pa[i] >= TREE_SURE && thickA[i]) {
      labels[i] = 1;
      nA++;
    }
  }
  if (!nA || !nV) return;
  compete(labels, d);
  for (let i = 0; i < N; i++) {
    if (labels[i] === 1) [arteries[i], veins[i]] = [1, 0];
    else if (labels[i] === 2) [arteries[i], veins[i]] = [0, 1];
  }
}

/** Cells of m with no cell outside it closer than mm (the grid's edge counts as inside; dcv/phases.py _thick). Only m's
 *  own cells are looked at: a vessel mask is ~1 % of a head's grid. */
function thick(m: Uint8Array, [nx, ny, nz]: D3, sp: number[], mm: number): Uint8Array {
  const out = new Uint8Array(m.length);
  const r = sp.map((s) => Math.ceil(mm / s));
  const offs: [number, number, number][] = [];
  for (let dz = -r[2]; dz <= r[2]; dz++) for (let dy = -r[1]; dy <= r[1]; dy++) for (let dx = -r[0]; dx <= r[0]; dx++) {
    const dist = Math.hypot(dx * sp[0], dy * sp[1], dz * sp[2]);
    if (dist > 0 && dist < mm) offs.push([dx, dy, dz]);
  }
  const nxy = nx * ny;
  for (let i = 0; i < m.length; i++) {
    if (!m[i]) continue;
    const x = i % nx, y = Math.floor(i / nx) % ny, z = Math.floor(i / nxy);
    let ok = 1;
    for (const [dx, dy, dz] of offs) {
      const X = x + dx, Y = y + dy, Z = z + dz;
      if (X >= 0 && X < nx && Y >= 0 && Y < ny && Z >= 0 && Z < nz && !m[(Z * ny + Y) * nx + X]) {
        ok = 0;
        break;
      }
    }
    out[i] = ok;
  }
  return out;
}

/** Grow both seeds through `labels` (0 = free, 3 = no vessel, 1 / 2 = seeds): first front to reach a voxel takes it. */
export function compete(labels: Uint8Array, [nx, ny]: D3): void {
  const nxy = nx * ny, N = labels.length;
  const offsets = [1, -1, nx, -nx, nxy, -nxy];
  const collect = (tag: number) => {
    const out: number[] = [];
    for (let i = 0; i < N; i++) if (labels[i] === tag) out.push(i);
    return out;
  };
  let a = collect(1), b = collect(2);
  const step = (front: number[], tag: number) => {
    const next: number[] = [];
    for (const i of front) {
      for (const o of offsets) {
        const j = i + o;
        if (j >= 0 && j < N && labels[j] === 0) {
          labels[j] = tag;
          next.push(j);
        }
      }
    }
    return next;
  };
  while (a.length || b.length) {
    a = step(a, 1);
    b = step(b, 2);
  }
}

/** Mean over a 3 × 3 × 3 box (edges repeat), one axis at a time. */
function boxMean(m: Float32Array, [nx, ny, nz]: D3): Float32Array {
  let cur = m;
  for (const [stride, len] of [[1, nx], [nx, ny], [nx * ny, nz]] as const) {
    const out = new Float32Array(m.length);
    for (let i = 0; i < m.length; i++) {
      const k = Math.floor(i / stride) % len;
      const lo = k > 0 ? i - stride : i, hi = k < len - 1 ? i + stride : i;
      out[i] = (cur[lo] + cur[i] + cur[hi]) / 3;
    }
    cur = out;
  }
  return cur;
}

/** How many voxels `a` and `b` moved by (z, y, x) have in common, on every second plane. */
function overlap(a: Uint8Array, b: Uint8Array, [nx, ny, nz]: D3, [dz, dy, dx]: D3): number {
  let c = 0;
  for (let z = Math.max(0, dz); z < nz + Math.min(0, dz); z += 2) {
    for (let y = Math.max(0, dy); y < ny + Math.min(0, dy); y++) {
      const ra = (z * ny + y) * nx, rb = ((z - dz) * ny + y - dy) * nx;
      for (let x = Math.max(0, dx); x < nx + Math.min(0, dx); x++) if (a[ra + x] && b[rb + x - dx]) c++;
    }
  }
  return c;
}

/** Whole-cell shift (z, y, x) of b relative to a from their bone, uphill one cell at a time; and how much of a it covers. */
function alignment(a: Uint8Array, b: Uint8Array, d: D3, sp: number[]): { shift: D3; cover: number } {
  let n = 0;
  for (let z = 0; z < d[2]; z += 2) for (let i = z * d[0] * d[1], e = i + d[0] * d[1]; i < e; i++) n += a[i];
  let best: D3 = [0, 0, 0], score = overlap(a, b, d, best);
  if (!n) return { shift: best, cover: 0 };
  const reach = [MAX_SHIFT_MM / sp[2], MAX_SHIFT_MM / sp[1], MAX_SHIFT_MM / sp[0]].map(Math.ceil);
  for (let better = true; better;) {
    better = false;
    for (const ax of [1, 2, 0]) {
      for (const s of [-1, 1]) {
        const t = [...best] as D3;
        t[ax] += s;
        if (Math.abs(t[ax]) > reach[ax]) continue;
        const c = overlap(a, b, d, t);
        if (c > score) {
          best = t;
          score = c;
          better = true;
        }
      }
    }
  }
  return { shift: best, cover: score / n };
}

/** np.roll: the value at p moves to p + (z, y, x), wrapping round. */
function roll<T extends Uint8Array | Float32Array>(m: T, [nx, ny, nz]: D3, [dz, dy, dx]: D3): T {
  if (!dz && !dy && !dx) return m;
  const out = m.slice() as T;
  for (let z = 0; z < nz; z++) {
    const Z = (z + dz + nz) % nz;
    for (let y = 0; y < ny; y++) {
      const Y = (y + dy + ny) % ny, from = (z * ny + y) * nx, to = (Z * ny + Y) * nx;
      for (let x = 0; x < nx; x++) out[to + (x + dx + nx) % nx] = m[from + x];
    }
  }
  return out;
}

function percentile90(v: number[]): number {
  v.sort((p, q) => p - q);
  return v[Math.floor(0.9 * (v.length - 1))];
}

/** Arteries, veins and bone of two series on the same grid; `native` = the pre-contrast series when there is one. */
export function classifyPhases(arterial: Volume, venous: Volume, native: Volume | null, boneHu = 500, ownIsVenous = true): PhaseResult {
  const [sz, sy, sx] = venous.spacing;
  const { cols, rows, n } = venous;
  const coarse = cols * rows * n >= 30e6 && cols % 2 === 0 && rows % 2 === 0;    // small volumes keep every voxel
  const f = coarse && sx < 0.75 ? 2 : 1, fz = coarse ? Math.max(1, Math.round(1 / sz)) : 1;
  const a = pool(arterial.data, cols, rows, n, f, fz), v = pool(venous.data, cols, rows, n, f, fz);
  const d = a.d, [nx, ny, nz] = d, N = nx * ny * nz, nxy = nx * ny;
  const sp = [sx * f, sy * f, sz * fz], voxel = sp[0] * sp[1] * sp[2];
  const radii = (mm: number): D3 => [Math.round(mm / sp[0]), Math.round(mm / sp[1]), Math.round(mm / sp[2])];
  const atLeast1 = (r: D3): D3 => r.map((k) => Math.max(1, k)) as D3;

  // a patient who moved between the scans: everything is compared where the arterial series has it
  const above = (x: Float32Array, t: number) => {
    const out = new Uint8Array(N);
    for (let i = 0; i < N; i++) out[i] = x[i] >= t ? 1 : 0;
    return out;
  };
  const highA = above(a.top, boneHu);
  const { shift } = alignment(highA, above(v.top, boneHu), d, sp);
  v.mean = roll(v.mean, d, shift);
  v.top = roll(v.top, d, shift);
  let P: Float32Array | null = null;
  if (native) {
    // the native scan is a minute or two earlier (real CTA: 4 mm); used only when its bone can be matched
    const pre = pool(native.data, cols, rows, n, f, fz).mean;
    const m = alignment(highA, above(pre, 0.6 * boneHu), d, sp);
    if (m.cover >= 0.5) P = roll(pre, d, m.shift);
  }

  const bone = new Uint8Array(N);
  let top = 0;
  const count = new Int32Array(nz);
  for (let i = 0; i < N; i++) {
    if (a.top[i] >= boneHu && v.top[i] >= boneHu && Math.min(a.mean[i], v.mean[i]) >= 0.6 * boneHu) {
      bone[i] = 1;
      count[Math.floor(i / nxy)]++;
    }
  }
  for (let z = nz - 1; z >= 0; z--) if (count[z] >= 20) { top = z; break; }

  // The sinuses traced along the venous series, as on one contrast series (sinusMasks): the whole transverse and sigmoid
  // sinuses, the straight sinus and torcula.
  const traced = traceSinuses(v.mean, v.top, d, sp)?.sinus ?? new Uint8Array(N);
  // Bone that moved on its own between the phases (jaw, shoulders) is bright on one series and soft tissue on the other,
  // which looked like a vessel filling or washing out (real CTA: blue blocks on the teeth). Whole cells of bone only, and
  // never beside a traced sinus: a skull that turned a little also moves its inner table.
  const denseA = above(a.top, MOVED_BONE_HU), denseV = above(v.top, MOVED_BONE_HU);
  const nearA = dilate(denseA, d, radii(MOVED_BONE_MM)), nearV = dilate(denseV, d, radii(MOVED_BONE_MM));
  const keep = dilate(traced, d, radii(MOVED_BONE_MM + 1));
  const solid = new Uint8Array(N);
  for (let i = 0; i < N; i++) {
    solid[i] = bone[i];
    if (keep[i]) continue;
    if (denseA[i] && a.mean[i] >= boneHu && v.mean[i] < 0.5 * boneHu && nearV[i]) solid[i] = 1;
    if (denseV[i] && v.mean[i] >= boneHu && a.mean[i] < 0.5 * boneHu && nearA[i]) solid[i] = 1;
  }
  const near = dilate(solid, d, atLeast1(radii(1)));
  const body = new Uint8Array(N);
  for (let i = 0; i < N; i++) body[i] = a.mean[i] > -200 ? 1 : 0;
  let largest = 0;
  pieces(body, d, (_, m) => { if (m > largest) largest = m; });
  const patient = new Uint8Array(N);
  pieces(body, d, (list, m) => { if (m === largest) for (let i = 0; i < m; i++) patient[list[i]] = 1; });

  const seen = new Uint8Array(N);
  for (let z = 0; z < nz; z++) {
    // what the shift carried round from the far edge of the volume compares two unrelated places
    if (shift[0] > 0 ? z < shift[0] : shift[0] < 0 && z >= nz + shift[0]) continue;
    for (let y = 0; y < ny; y++) {
      if (shift[1] > 0 ? y < shift[1] : shift[1] < 0 && y >= ny + shift[1]) continue;
      for (let x = 0; x < nx; x++) {
        if (shift[2] > 0 ? x < shift[2] : shift[2] < 0 && x >= nx + shift[2]) continue;
        const i = (z * ny + y) * nx + x;
        if (patient[i] && !near[i] && Math.max(a.mean[i], v.mean[i]) >= VESSEL_MIN_HU) seen[i] = 1;
      }
    }
  }
  // The sinuses first, before the native rule: a head can also turn a little between the scans (real CTA: 63 % of the
  // bone overlapped), and then the native inner table may fall on the sinus beside it. A piece that was mostly bright
  // before contrast is no sinus. A weakly filled sinus (~209 HU arterial, ~120 venous) has no core here: the washout finds it.
  const sinusZ = Math.max(0, top - Math.floor(SINUS_DEPTH_MM / sp[2]));
  const core = erode(seen, d, atLeast1(radii(SINUS_THICK_MM / 2)));
  const wall = dilate(bone, d, radii(SINUS_REACH_MM + SINUS_THICK_MM / 2));
  for (let i = 0; i < N; i++) if (!wall[i] || i < sinusZ * nxy) core[i] = 0;
  const longCore = new Uint8Array(N);
  pieces(core, d, (list, m) => {
    if (m * voxel < SINUS_MIN_MM3) return;
    let clear = 0;
    for (let i = 0; i < m; i++) if (!P || P[list[i]] < VESSEL_MIN_HU) clear++;
    if (clear >= 0.5 * m) for (let i = 0; i < m; i++) longCore[list[i]] = 1;
  });
  const lumen = dilate(longCore, d, atLeast1(radii(SINUS_THICK_MM)));

  const raw = new Uint8Array(N);
  for (let i = 0; i < N; i++) if (seen[i] && (lumen[i] || !(P && P[i] >= VESSEL_MIN_HU))) raw[i] = 1;
  const vessel = new Uint8Array(N);
  pieces(raw, d, (list, m) => { if (m * voxel >= PIECE_MIN_MM3) for (let i = 0; i < m; i++) vessel[list[i]] = 1; });

  // the smoothed difference and the large arteries' HU
  const rawDiff = new Float32Array(N);
  for (let i = 0; i < N; i++) rawDiff[i] = v.mean[i] - a.mean[i];
  const diff = boxMean(rawDiff, d);
  const z0 = Math.max(0, top - Math.floor(220 / sp[2])), z1 = Math.max(1, top - Math.floor(120 / sp[2]));
  const thick = erode(vessel, d, [1, 1, 1]);
  const neck: number[] = [];
  for (let i = z0 * nxy; i < z1 * nxy; i++) if (thick[i] && a.mean[i] < PEAK_MAX_HU) neck.push(a.mean[i]);
  const peak = neck.length >= 50 ? percentile90(neck) : 400;

  const artZ = Math.max(0, top - Math.floor(ARTERY_SEED_DEPTH_MM / sp[2]));

  // Veins by the share of their enhancement left on the venous series: thin or weakly filled vessels are found this way,
  // as a share does not change with partial volume. This patient's arteries set the bar (dcv/phases.py washout_seeds).
  const enhA = new Float32Array(N), enhV = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const base = P ? P[i] : UNENHANCED_HU;
    enhA[i] = a.mean[i] - base;
    enhV[i] = v.mean[i] - base;
  }
  const smoothA = boxMean(enhA, d), smoothV = boxMean(enhV, d);
  const ratio = new Float32Array(N);
  for (let i = 0; i < N; i++) ratio[i] = smoothV[i] / Math.max(smoothA[i], 20);
  const farBone = dilate(bone, d, radii(12));
  const refNeck: number[] = [], refInner: number[] = [];
  for (let i = 0; i < N; i++) {
    if (!vessel[i]) continue;
    const z = Math.floor(i / nxy);
    if (z >= z0 && z < z1 && a.mean[i] >= ARTERY_SHARE * peak) refNeck.push(ratio[i]);
    if (z >= artZ && !farBone[i]) refInner.push(ratio[i]);
  }
  const refs = [refNeck, refInner].filter((r) => r.length >= 50).map((r) => r.sort((p, q) => p - q)[Math.floor(r.length / 2)]);
  const slow = new Uint8Array(N);
  if (refs.length) {
    const bar = Math.min(...refs) + WASHOUT_MARGIN;
    const brightA = new Uint8Array(N);
    for (let i = 0; i < N; i++) brightA[i] = a.mean[i] >= ARTERY_SHARE * peak ? 1 : 0;
    const shell = dilate(brightA, d, radii(3));   // the edge of a large artery is partly tissue: its share drifts up
    const cand = new Uint8Array(N);
    for (let i = 0; i < N; i++) if (vessel[i] && ratio[i] >= bar && !shell[i]) cand[i] = 1;
    pieces(cand, d, (list, m) => { if (m * voxel >= WASHOUT_MIN_MM3) for (let i = 0; i < m; i++) slow[list[i]] = 1; });
  }

  for (let i = 0; i < N; i++) traced[i] &= vessel[i];

  // A late scan: the traced sinuses keep about the share of their enhancement that the neck arteries keep (real CTAs:
  // 0.36 against 0.40), so the time course separates nothing and anatomy decides.
  let late = false, tracedN = 0;
  for (let i = 0; i < N; i++) tracedN += traced[i];
  if (tracedN * voxel >= LATE_MIN_ML * 1000) {
    const share = (i: number) => (v.mean[i] - (P ? P[i] : UNENHANCED_HU)) / Math.max(a.mean[i] - (P ? P[i] : UNENHANCED_HU), 20);
    const s: number[] = [], k: number[] = [];
    for (let i = 0; i < N; i++) {
      if (traced[i]) s.push(share(i));
      const z = Math.floor(i / nxy);
      if (vessel[i] && z >= z0 && z < z1 && a.mean[i] >= ARTERY_SHARE * peak) k.push(share(i));
    }
    const median = (x: number[]) => x.sort((p, q) => p - q)[Math.floor(x.length / 2)];
    late = k.length >= 50 && median(s) - median(k) < LATE_GAP;
  }

  const labels = new Uint8Array(N);
  if (late) {
    // Arteries start where they enter the head: the medial neck vessels, the centre of the skull base under the circle
    // of Willis (carotid siphons inside the cavernous sinus, basilar artery) and the descending aorta. Veins from every
    // traced sinus and the lateral neck vessels (the jugulars, as bright as the carotids here).
    const cand = new Uint8Array(N);
    for (let i = 0; i < N; i++) cand[i] = vessel[i] && thick[i] && a.mean[i] >= 0.6 * peak ? 1 : 0;
    const neckArt = new Uint8Array(N), neckVen = new Uint8Array(N);
    neckSides(cand, body, top, d, sp, neckArt, neckVen);
    const entry = arteryEntry(insideOf(bone, d), top, d, sp);
    const aorta = descendingAorta(cand, bone, body, top, d, sp);
    const sinusOrTraced = new Uint8Array(N);
    for (let i = 0; i < N; i++) sinusOrTraced[i] = traced[i] | (lumen[i] & vessel[i]);
    const besideSinus = dilate(sinusOrTraced, d, [1, 1, 1]);
    for (let i = 0; i < N; i++) {
      if (!vessel[i]) labels[i] = 3;
      else if ((neckArt[i] || entry[i] || aorta[i]) && !besideSinus[i]) labels[i] = 1;
      else if (diff[i] >= DIFF_MIN_HU || sinusOrTraced[i] || neckVen[i]) labels[i] = 2;
    }
  } else {
    for (let i = 0; i < N; i++) {
      const artSeed = a.mean[i] >= ARTERY_SHARE * peak && diff[i] <= -DIFF_MIN_HU && !lumen[i] && !traced[i] && i < artZ * nxy;
      if (!vessel[i]) labels[i] = 3;
      else if (artSeed) labels[i] = 1;
      else if (lumen[i] || traced[i] || diff[i] >= DIFF_MIN_HU || slow[i]) labels[i] = 2;
    }
  }
  compete(labels, d);
  // what the two fronts could not reach (a piece cut off by bone): the time course, or on a late scan whether it lies
  // against the inside of the skull, where the sinuses are
  const dural = late ? dilate(bone, d, radii(SINUS_REACH_MM + 1)) : null;
  const insideBone = late ? insideOf(bone, d) : null;
  const arteries = new Uint8Array(N), veins = new Uint8Array(N);
  for (let i = 0; i < N; i++) {
    if (labels[i] === 1) arteries[i] = 1;
    else if (labels[i] === 2) veins[i] = 1;
    else if (labels[i] === 0) {
      const toVein = late ? insideBone![i] && dural![i] : diff[i] >= 0;
      if (toVein) veins[i] = 1;
      else arteries[i] = 1;
    }
  }
  // a late scan: inside the head the vessel atlas decides (atlas.ts), with the venous series' own HU where it still tells
  // them apart; the neck and chest keep the rules above
  let byAtlas = false;
  if (late) {
    const res = classifyHead(vessel, (i) => [v.mean[i]], bone, d, sp);
    if (res) {
      for (let i = 0; i < N; i++) if (res.decided[i]) {
        arteries[i] = res.arteries[i];
        veins[i] = res.veins[i];
      }
      byAtlas = true;
    }
  }

  // the bone area: closed (diploe), widened, without the vessels and their wall
  const rc = radii(3);
  const closed = erode(dilate(bone, d, rc), d, rc);
  for (let i = 0; i < N; i++) closed[i] |= bone[i];
  const area = dilate(closed, d, radii(1.5));
  const any = new Uint8Array(N);
  for (let i = 0; i < N; i++) any[i] = arteries[i] | veins[i];
  const wallV = dilate(any, d, atLeast1(radii(1)));
  for (let i = 0; i < N; i++) if (wallV[i] && !bone[i]) area[i] = 0;

  // one cell wider, so the full-resolution HU, not the coarse grid, draws the surface
  const wideA = dilate(arteries, d, [1, 1, 1]), wideV = dilate(veins, d, [1, 1, 1]);
  for (let i = 0; i < N; i++) {
    if (veins[i]) wideA[i] = 0;
    if (arteries[i] || wideA[i]) wideV[i] = 0;
    if (area[i]) wideA[i] = wideV[i] = 0;
  }
  // bone outside the bone area is drawn as bone (encodeWith); where it moved (bright on this series only: teeth), its
  // partial-volume edge is no vein either. The inner table beside a sinus is bright on both and keeps the sinus.
  const ownTop = ownIsVenous ? v.top : a.top, otherTop = ownIsVenous ? a.top : v.top, hard = new Uint8Array(N);
  for (let i = 0; i < N; i++) hard[i] = ownTop[i] >= boneHu && otherTop[i] < boneHu && !area[i] && !arteries[i] ? 1 : 0;
  const besideHard = dilate(hard, d, [1, 1, 1]);
  for (let i = 0; i < N; i++) if (besideHard[i]) wideV[i] = 0;
  // a cell cuts through a wide lumen (a sinus against the skull): the vein takes the rest of the lumen within ~2 cells,
  // so its surface is drawn by the HU and not in steps of the grid
  for (let step = 0; step < 2; step++) {
    const grown = dilate(wideV, d, [1, 1, 1]);
    for (let i = 0; i < N; i++) {
      if (grown[i] && !wideV[i] && !wideA[i] && !arteries[i] && !area[i] && !besideHard[i] && v.mean[i] >= 110 && v.mean[i] < boneHu) wideV[i] = 1;
    }
  }
  // where the bone area meets a vessel (its cells and the wall taken out around them), the voxels decide (encodeCT)
  const cells = new Uint8Array(N);
  for (let i = 0; i < N; i++) cells[i] = wideA[i] | wideV[i];
  const nearArea = dilate(area, d, [1, 1, 1]), nearVessel = dilate(cells, d, [2, 2, 2]);
  const edge = new Uint8Array(N);
  for (let i = 0; i < N; i++) edge[i] = nearVessel[i] && nearArea[i] ? 1 : 0;
  // the masks lie where the arterial series has things: on the window of the venous series they move back to it
  const back: D3 = ownIsVenous ? [-shift[0], -shift[1], -shift[2]] : [0, 0, 0];
  const region = (m: Uint8Array): BoneRegion => ({ ...stretchZ(roll(m, d, back), d, fz, n), f });
  const voxels = shift[0] * fz * rows * cols + shift[1] * f * cols + shift[2] * f;
  return {
    bone: region(area), arteries: region(wideA), veins: region(wideV), edge: region(edge), coreHu: peak,
    artData: arterial.data, venData: venous.data,
    artOffset: ownIsVenous ? voxels : 0, venOffset: ownIsVenous ? 0 : -voxels,
    shiftMm: Math.hypot(shift[0] * sp[2], shift[1] * sp[1], shift[2] * sp[0]),
    late, boneHu, atlas: byAtlas,
  };
}

/** Cells with bone on both sides along x and along y in their plane: the inside of the head (dcv/phases.py inside_skull). */
export function insideOf(bone: Uint8Array, [nx, ny, nz]: D3): Uint8Array {
  const inside = new Uint8Array(bone.length);
  const xl = new Int32Array(ny), xr = new Int32Array(ny), yt = new Int32Array(nx), yb = new Int32Array(nx);
  for (let z = 0; z < nz; z++) {
    xl.fill(nx); xr.fill(-1); yt.fill(ny); yb.fill(-1);
    for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) if (bone[(z * ny + y) * nx + x]) {
      if (x < xl[y]) xl[y] = x;
      if (x > xr[y]) xr[y] = x;
      if (y < yt[x]) yt[x] = y;
      if (y > yb[x]) yb[x] = y;
    }
    for (let y = 0; y < ny; y++) for (let x = xl[y]; x <= xr[y]; x++) if (y >= yt[x] && y <= yb[x]) inside[(z * ny + y) * nx + x] = 1;
  }
  return inside;
}

/** Where a late scan's arteries start inside the head: 90-130 mm under the vertex, within 30 mm of the midline, in the
 *  front 55 % of the head (carotid siphons, basilar artery); the vein of Galen and the straight sinus are higher and
 *  further back (dcv/phases.py _artery_entry). */
function arteryEntry(inside: Uint8Array, vtx: number, [nx, ny]: D3, sp: number[]): Uint8Array {
  const out = new Uint8Array(inside.length);
  const z0 = Math.max(0, vtx - Math.floor(SINUS_DEPTH_MM / sp[2])), z1 = Math.max(0, vtx - Math.floor(ARTERY_BASE_DEPTH_MM / sp[2]));
  for (let z = z0; z < z1; z++) {
    let xmin = nx, xmax = -1, ymin = ny, ymax = -1, c = 0;
    for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) if (inside[(z * ny + y) * nx + x]) {
      c++;
      if (x < xmin) xmin = x;
      if (x > xmax) xmax = x;
      if (y < ymin) ymin = y;
      if (y > ymax) ymax = y;
    }
    if (c < 100) continue;
    const xc = (xmin + xmax) / 2, half = SKULL_BASE_HALF_MM / sp[0], front = ymin + ARTERY_BASE_FRONT * (ymax - ymin);
    for (let y = 0; y < Math.floor(front); y++) {
      for (let x = Math.max(0, Math.floor(xc - half)); x <= Math.min(nx - 1, Math.floor(xc + half)); x++) out[(z * ny + y) * nx + x] = 1;
    }
  }
  return out;
}

/** The middle column of the body in plane z, or null. */
function midline(body: Uint8Array, z: number, [nx, ny]: D3): number | null {
  let lo = nx, hi = -1;
  for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) if (body[(z * ny + y) * nx + x]) {
    if (x < lo) lo = x;
    if (x > hi) hi = x;
  }
  return hi - lo + 1 >= 10 ? 0.5 * (lo + hi) : null;
}

/** The bright thick vessels of the neck, 14-24 cm under the vertex, split by where they lie: on each side the carotid
 *  is medial and the internal jugular posterolateral to it (dcv/phases.py _neck_sides). */
function neckSides(cand: Uint8Array, body: Uint8Array, top: number, d: D3, sp: number[], art: Uint8Array, ven: Uint8Array): void {
  const [nx, ny] = d, nxy = nx * ny, cell = sp[0] * sp[1];
  const z0 = Math.max(0, top - Math.floor(240 / sp[2])), z1 = Math.max(0, top - Math.floor(140 / sp[2]));
  const plane = new Uint8Array(nxy);
  for (let z = z0; z < z1; z++) {
    const mid = midline(body, z, d);
    if (mid === null) continue;
    for (const left of [false, true]) {
      plane.fill(0);
      let c = 0;
      for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
        if (cand[z * nxy + y * nx + x] && (x >= mid) === left) { plane[y * nx + x] = 1; c++; }
      }
      if (c * cell < 10) continue;
      // the two largest pieces of this side, by size
      const found: { list: Int32Array; out: number }[] = [];
      pieces(plane, [nx, ny, 1], (list, m) => {
        let out = 0;
        for (let i = 0; i < m; i++) out += Math.abs((list[i] % nx) - mid) * sp[0] + Math.floor(list[i] / nx) * sp[1];
        found.push({ list: list.slice(0, m), out: out / m });
      });
      found.sort((p, q) => q.list.length - p.list.length);
      const mark = (to: Uint8Array, list: ArrayLike<number>) => { for (let i = 0; i < list.length; i++) to[z * nxy + list[i]] = 1; };
      if (found.length > 1 && found[1].list.length * cell >= 10) {
        const [jug, car] = found[0].out > found[1].out ? [found[0], found[1]] : [found[1], found[0]];
        mark(art, car.list);
        mark(ven, jug.list);
        continue;
      }
      const one = found[0].list;
      let dmin = Infinity, dmax = -Infinity, omin = Infinity, omax = -Infinity;
      const outOf = (i: number) => Math.abs((i % nx) - mid) * sp[0] + Math.floor(i / nx) * sp[1];
      for (let i = 0; i < one.length; i++) {
        const dd = Math.abs((one[i] % nx) - mid), o = outOf(one[i]);
        dmin = Math.min(dmin, dd); dmax = Math.max(dmax, dd); omin = Math.min(omin, o); omax = Math.max(omax, o);
      }
      if ((dmax - dmin) * sp[0] < JUGULAR_GAP_MM) {
        mark(art, one);
        continue;
      }
      // carotid and jugular touching: one piece, split halfway back and out
      for (let i = 0; i < one.length; i++) (outOf(one[i]) > 0.5 * (omin + omax) ? ven : art)[z * nxy + one[i]] = 1;
    }
  }
}

/** The descending aorta where the scan reaches the chest: thick bright vessel beside the vertebral body, on the
 *  patient's left and in front of the spine's centre (dcv/phases.py _descending_aorta). */
function descendingAorta(cand: Uint8Array, bone: Uint8Array, body: Uint8Array, top: number, d: D3, sp: number[]): Uint8Array {
  const [nx, ny] = d, nxy = nx * ny, out = new Uint8Array(cand.length);
  const z1 = Math.max(0, top - Math.floor(AORTA_DEPTH_MM / sp[2]));
  if (!z1) return out;
  const wide = erode(cand, d, [2, 2, 0]);
  for (let z = 0; z < z1; z++) {
    const mid = midline(body, z, d);
    if (mid === null) continue;
    const ys: number[] = [];
    for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) if (bone[z * nxy + y * nx + x] && Math.abs(x - mid) * sp[0] < 10) ys.push(y);
    if (ys.length < 10) continue;
    ys.sort((p, q) => p - q);
    const back = ys.slice(Math.floor(ys.length / 2));
    const sy = back[Math.floor(back.length / 2)];
    for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
      const i = z * nxy + y * nx + x;
      if (!wide[i] || !cand[i]) continue;
      const left = (x - mid) * sp[0], front = (sy - y) * sp[1];
      if (left > 5 && left < 45 && front > 0 && front < 45) out[i] = 1;
    }
  }
  return out;
}

/**
 * The dural sinuses of ONE contrast series, by anatomy (dcv/phases.py sinus_tracing; its comment has the scores): thick
 * vessels inside the skull against the inner table in the upper head, in long pieces, followed along vessels that stay
 * against bone, and thick ones only high in the head; never in the middle of the skull base (carotid siphons, circle of
 * Willis, aneurysms). `arterial`: a named arterial CTA keeps its brightness rule for the other veins. Null = no head.
 * Arteries here = every other vessel above the threshold: arteries and small veins together.
 */
export function sinusMasks(vol: Volume, arterial: VesselMasks | null): VesselMasks | null {
  const { cols, rows, n } = vol;
  const coarse = cols * rows * n >= 30e6 && cols % 2 === 0 && rows % 2 === 0;
  const f = coarse && vol.spacing[2] < 0.75 ? 2 : 1, fz = coarse ? Math.max(1, Math.round(1 / vol.spacing[0])) : 1;
  const { mean, top, d } = pool(vol.data, cols, rows, n, f, fz);
  const N = d[0] * d[1] * d[2];
  const sp = [vol.spacing[2] * f, vol.spacing[1] * f, vol.spacing[0] * fz], voxel = sp[0] * sp[1] * sp[2];
  const t = traceSinuses(mean, top, d, sp);
  if (!t) return null;
  const { sinus, count, hu } = t;
  const other = new Uint8Array(N);
  for (let i = 0; i < N; i++) other[i] = mean[i] >= hu && !sinus[i] ? 1 : 0;
  const region = (m: Uint8Array): BoneRegion => ({ ...stretchZ(m, d, fz, n), f });
  // Inside the head the vessel atlas decides, with this series' own HU (atlas.ts; leave-one-out on real CTAs from one
  // series: 88-89 % of the vessel voxels right, the tracing alone 57-66 %). Elsewhere, and without a fitting skull, the above.
  const bone = new Uint8Array(N);
  for (let i = 0; i < N; i++) bone[i] = top[i] >= 500 ? 1 : 0;
  const inside = insideOf(bone, d), near = dilate(bone, d, [1, 1, 1]), vessel = new Uint8Array(N);
  for (let i = 0; i < N; i++) vessel[i] = mean[i] >= hu && inside[i] && !near[i] ? 1 : 0;
  const head = classifyHead(vessel, (i) => [mean[i]], bone, d, sp);
  const veinsOwn = sinus.slice(), artsOwn = other;
  if (head) {
    for (let i = 0; i < N; i++) if (head.decided[i]) {
      artsOwn[i] = head.arteries[i];
      veinsOwn[i] = head.veins[i];
    }
    // arteries and veins equally bright: each vessel the label of the tree it is joined to
    if (!arterial) vesselTrees(artsOwn, veinsOwn, sinus, head.prior, d, sp);
  }
  const ml = (count * voxel) / 1000;
  if (arterial) {
    // a named arterial CTA: its own arteries; veins = its dimmer vessels and the traced sinuses (or the atlas's veins)
    const veins = arterial.veins, arts = arterial.arteries;
    const s = region(veinsOwn), dec = head ? region(head.decided) : null, ha = head ? region(artsOwn) : null;
    const words = Math.min(veins.bits.length, s.bits.length);
    if (veins.f === s.f && veins.nx === s.nx && veins.ny === s.ny) {
      for (let w = 0; w < words; w++) {
        if (dec && ha) {
          veins.bits[w] = (veins.bits[w] & ~dec.bits[w]) | s.bits[w];
          arts.bits[w] = ((arts.bits[w] & ~dec.bits[w]) | (ha.bits[w] & dec.bits[w])) & ~s.bits[w];
        } else {
          veins.bits[w] |= s.bits[w];
          arts.bits[w] &= ~s.bits[w];
        }
      }
    }
    return { ...arterial, arteries: arts, veins, sinusMl: ml, atlas: !!head };
  }
  // the range of this series' vessels, for curves fitted to a routine contrast CT (sinuses ~100-140 HU, real CTs);
  // none for a CTA-bright series
  const sv: number[] = [];
  for (let i = 0; i < N; i++) if (sinus[i]) sv.push(mean[i]);
  sv.sort((p, q) => p - q);
  const peak = sv.length ? sv[sv.length >> 1] : Infinity;
  const fit = peak < 200 ? { floor: Math.max(hu + 10, 60), peak } : undefined;
  return { arteries: region(artsOwn), veins: region(veinsOwn), coreHu: hu, sinusMl: ml, atlas: !!head, fit };
}

/** sinusMasks' tracing on the mask grid: the sinus cells, how many, and the vessel threshold. Null = no head. */
function traceSinuses(mean: Float32Array, top: Float32Array, d: D3, sp: number[]): { sinus: Uint8Array; count: number; hu: number } | null {
  const [nx, ny, nz] = d, N = nx * ny * nz, nxy = nx * ny, voxel = sp[0] * sp[1] * sp[2];
  const radii = (mm: number): D3 => [Math.round(mm / sp[0]), Math.round(mm / sp[1]), Math.round(mm / sp[2])];
  const atLeast1 = (r: D3): D3 => r.map((k) => Math.max(1, k)) as D3;
  const bone = new Uint8Array(N);
  for (let i = 0; i < N; i++) bone[i] = top[i] >= 500 ? 1 : 0;
  let vtx = -1;
  for (let z = nz - 1; z >= 0 && vtx < 0; z--) {
    let c = 0;
    for (let i = z * nxy; i < (z + 1) * nxy; i++) c += bone[i];
    if (c >= 20) vtx = z;
  }
  if (vtx < 0) return null;
  const inside = insideOf(bone, d);
  const near = dilate(bone, d, atLeast1(radii(1)));
  // the threshold: halfway between this patient's brain and the brightest 0.5 % of the upper head, well clear of brain
  const far3 = dilate(bone, d, radii(3));
  const z0 = Math.max(0, vtx - Math.floor(90 / sp[2])), z1 = Math.max(1, vtx - Math.floor(15 / sp[2]));
  const vals: number[] = [];
  for (let i = z0 * nxy; i < z1 * nxy; i++) if (inside[i] && !far3[i] && mean[i] > 0) vals.push(mean[i]);
  if (vals.length < 1000) return null;
  vals.sort((p, q) => p - q);
  const brain = vals[Math.floor(vals.length / 2)], top995 = vals[Math.floor(0.995 * (vals.length - 1))];
  const hu = Math.max(brain + 35, 0.5 * (brain + top995));
  const raw = new Uint8Array(N);
  for (let i = 0; i < N; i++) if (inside[i] && !near[i] && mean[i] >= hu) raw[i] = 1;
  const vessel = new Uint8Array(N);
  pieces(raw, d, (list, m) => { if (m * voxel >= PIECE_MIN_MM3) for (let i = 0; i < m; i++) vessel[list[i]] = 1; });
  const dural = dilate(bone, d, radii(SINUS_REACH_MM + 1));
  const er = erode(vessel, d, [1, 1, 1]);
  const thick = new Uint8Array(N);
  for (let i = 0; i < N; i++) thick[i] = er[i] & vessel[i];
  const sinusZ = Math.max(0, vtx - Math.floor(SINUS_DEPTH_MM / sp[2])), highZ = Math.max(0, vtx - Math.floor(70 / sp[2]));
  const core = new Uint8Array(N);
  for (let i = sinusZ * nxy; i < N; i++) core[i] = thick[i] & dural[i];
  const seeds = new Uint8Array(N);
  pieces(core, d, (list, m) => { if (m * voxel >= SINUS_MIN_MM3) for (let i = 0; i < m; i++) seeds[list[i]] = 1; });
  const thickWide = dilate(thick, d, [1, 1, 1]);
  const way = new Uint8Array(N);
  for (let i = 0; i < N; i++) way[i] = vessel[i] && ((dural[i] && thickWide[i]) || (thick[i] && i >= highZ * nxy)) ? 1 : 0;
  const kept = new Uint8Array(N);
  pieces(way, d, (list, m) => {
    let hit = false;
    for (let i = 0; i < m && !hit; i++) hit = seeds[list[i]] === 1;
    if (hit) for (let i = 0; i < m; i++) kept[list[i]] = 1;
  });
  const sinus = dilate(kept, d, [1, 1, 1]);
  let count = 0;
  for (let i = 0; i < N; i++) {
    sinus[i] &= vessel[i];
    count += sinus[i];
  }
  // the middle of the skull base, below 70 mm under the vertex: within 30 mm of the midline, in front of the back fifth
  for (let z = 0; z < highZ; z++) {
    let xmin = nx, xmax = -1, ymin = ny, ymax = -1;
    for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) if (inside[(z * ny + y) * nx + x]) {
      if (x < xmin) xmin = x;
      if (x > xmax) xmax = x;
      if (y < ymin) ymin = y;
      if (y > ymax) ymax = y;
    }
    if (xmax < 0) continue;
    const xc = (xmin + xmax) / 2, half = 30 / sp[0], back = ymax - 0.2 * (ymax - ymin);
    for (let y = 0; y < back; y++) for (let x = Math.max(0, Math.floor(xc - half)); x <= Math.min(nx - 1, Math.ceil(xc + half)); x++) {
      const i = (z * ny + y) * nx + x;
      count -= sinus[i];
      sinus[i] = 0;
    }
  }
  if (count * voxel > 45000) count = 0, sinus.fill(0);   // far more than the sinuses hold: cortex, not sinus
  return { sinus, count, hu };
}

/** Bits on the (x, y) step `f`, every plane: the masks use the existing mask grid, which never halves planes. */
export function stretchZ(m: Uint8Array, [nx, ny, nz]: D3, fz: number, planes: number): Omit<BoneRegion, "f"> {
  const nxy = nx * ny;
  const out = new Uint8Array(nxy * planes);
  for (let z = 0; z < planes; z++) {
    const Z = Math.min(nz - 1, Math.floor(z / fz)) * nxy;
    out.set(m.subarray(Z, Z + nxy), z * nxy);
  }
  return { bits: pack(out), nx, ny, nz: planes };
}
