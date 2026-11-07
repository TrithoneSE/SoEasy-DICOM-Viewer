/**
 * Venography view of a contrast head CT (preset "CT Venography", dcv/venography.py): the dural sinuses and the cortical
 * and deep veins in blue over a faint brain, inside a faint skull and skin.
 *
 * One HU curve cannot do it: brain and scalp have the same HU, and the edge of bone passes through the vessels' HU. So the
 * inside of the skull gets a domain of its own in the 3D copy (VEIN_OFFSET + HU, where the vein curves read): there the
 * brain and the vessels have their own curves, and the scalp and face outside keep theirs.
 *   cavity: the inside of the skull on a ~1 mm grid, with the gaps in it closed first (an infant's open fontanelle left
 *           the front of the superior sagittal sinus outside it);
 *   lumen:  what the contrast made bright. With the study's plain scan on the same grid: brighter by >= 50 HU than there,
 *           and nothing bright around it there (a fraction of a voxel's move changes the bone's edge by hundreds of HU).
 *           Without one: bright and away from bone;
 *   bone:   the bone colour lock's area, minus the lumens inside the cavity, plus what was bright before contrast there;
 *   layer:  the voxels around the cavity's domain are stored as its 0 HU, so a vessel against the bone or the scalp ends
 *           in its own domain (a smooth surface) instead of on the jump to the other code (a ladder of voxel faces).
 * All masks are bits on the grid of the 3D copy (step f in-plane): a few tens of MB for a large CT.
 */
import { BONE_OFFSET, dilateMask, erodeMask, maskAt, THIN_BONE_HU, VEIN_OFFSET, type BoneRegion } from "./bone";
import { insideOf, pieces, pool, stretchZ } from "./phases";
import type { Volume } from "../dicom/volume";
import { add, col, I3, mulMtV, mulMV, rot, scale, sub, type M3, type V3 } from "../geometry";

const CLOSE_MM = 10;          // gaps in the skull up to twice this are closed for the cavity
const LUMEN_MIN_HU = 70;      // contrast in a routine head CT's veins: ~95-175 HU (real CT), brain 30-40
const LUMEN_MAX_HU = 350;
const ENHANCE_HU = 50;        // brighter than on the plain scan by at least this
const PLAIN_MAX_HU = 120;     // the voxel on the plain scan: blood, not bone or calcium
const PLAIN_NEAR_HU = 200;    // nothing this bright around it on the plain scan (a newborn's skull there: 200-400 HU)
const BEFORE_HU = 90;         // bright around it on the plain scan: bone edge, calcification
const OUTSIDE_MIN_HU = 120;   // no plain scan: a lumen outside the skull is at least this bright
const RIM_MM = 2;             // the cavity's edge, where the skull lies
const MEAN_MIN_HU = 60;       // a lumen with its 6 neighbours: at least this (a thin cortical vein ~85, noise less)
const BRAIN_TOP_HU = 55;      // what is no lumen inside the skull is stored at most this: under the vein curve
/** The plain scan is used from this share of matching bone on (linesUp): a newborn lined up by register() reached 74 %,
 *  and its sutures then still came out as veins; the sheets alone did better. */
export const MATCH_MIN = 0.85;
/** What is no vessel outside the skull, from LUMEN_MIN_HU on, is stored here: under the curves of the preset. */
export const HIDDEN_HU = LUMEN_MIN_HU - 1;

type D3 = [number, number, number];

/** Bits on the 3D copy's grid: voxel (x, y, z) of the copy ↔ (x·f, y·f, z) of the volume. */
export interface VenoMasks {
  nx: number;
  ny: number;
  nz: number;
  f: number;
  inner: Uint32Array;
  layer: Uint32Array;
  bone: Uint32Array;
  lumen: Uint32Array;
  /** The plain scan decided the lumens. */
  plain: boolean;
}

/** The inside of the skull (see the module comment) and the bright sheets a cell wider, as bits on a grid halved
 *  in-plane, every plane. The skull that bounds the cavity: its dense bone and its bright sheets (sheets(): a newborn's
 *  plates are 150-450 HU, real CT, and at 300 HU the cavity ran out through them and the open sutures into the scalp);
 *  the larger of the cavities bounded at 500 and at 300 HU. */
export function cavity(vol: Volume): { cav: BoneRegion; plates: BoneRegion } {
  const { f, fz, sp } = cavityGrid(vol);
  const { mean, top, d } = pool(vol.data, vol.cols, vol.rows, vol.n, f, fz);
  const { cav, plates } = cavityOn(mean, top, d, sp);
  return { cav: { ...stretchZ(cav, d, fz, vol.n), f }, plates: { ...stretchZ(dilateMask(plates, d, [1, 1, 1]), d, fz, vol.n), f } };
}

/** The grid cavity() works on: halved in-plane and ~1 mm planes for a large volume. */
export function cavityGrid(vol: Volume): { f: number; fz: number; sp: D3 } {
  const [sz, sy, sx] = vol.spacing;
  const big = vol.cols * vol.rows * vol.n >= 8e6 && vol.cols % 2 === 0 && vol.rows % 2 === 0;
  const f = big && sx < 0.75 ? 2 : 1, fz = big ? Math.max(1, Math.round(1 / sz)) : 1;
  return { f, fz, sp: [sx * f, sy * f, sz * fz] };
}

/** A newborn's skull plates on cavity()'s grid (pooled mean and top, phases.ts pool): the sheets outside the skull's
 *  cavity, from THIN_BONE_HU (dcv/segment.py thin_plates). The bone lock takes them as bone (bone.worker.ts). */
export function thinPlates(mean: Float32Array, top: Float32Array, d: D3, sp: D3): Uint8Array {
  const { cav, plates } = cavityOn(mean, top, d, sp);
  for (let i = 0; i < plates.length; i++) plates[i] = plates[i] && !cav[i] && top[i] >= THIN_BONE_HU ? 1 : 0;
  return plates;
}

/** cavity() on its pooled grid, cell by cell: the cavity and the bright sheets. */
function cavityOn(mean: Float32Array, top: Float32Array, d: D3, sp: D3): { cav: Uint8Array; plates: Uint8Array } {
  const plates = sheets(mean, d, sp);
  let best: Uint8Array | null = null, bestN = -1;
  for (const hu of [500, 300]) {
    const barrier = new Uint8Array(top.length);
    for (let i = 0; i < top.length; i++) barrier[i] = top[i] >= hu || plates[i] ? 1 : 0;
    const c = cavityAt(barrier, mean, d, sp);
    let n = 0;
    for (let i = 0; i < c.length; i++) n += c[i];
    if (n > bestN) [best, bestN] = [c, n];
  }
  return { cav: best!, plates };
}

function cavityAt(barrier: Uint8Array, mean: Float32Array, d: D3, sp: D3): Uint8Array {
  const N = barrier.length;
  const r = sp.map((k) => Math.max(1, Math.round(CLOSE_MM / k))) as D3;
  const shut = dilateMask(barrier, d, r);
  const ins = insideOf(shut, d);
  const open = new Uint8Array(N);
  for (let i = 0; i < N; i++) open[i] = ins[i] && !shut[i] ? 1 : 0;
  const core = erodeMask(open, d, [2, 2, 2]);
  // the largest piece: the brain's, not an orbit or the neck's vertebral canal
  let largest: Int32Array | null = null, size = 0;
  pieces(core, d, (list, n) => {
    if (n > size) {
      size = n;
      largest = list.slice(0, n);
    }
  });
  let out: Uint8Array = new Uint8Array(N);
  if (!largest) return out;
  for (const i of largest as Int32Array) out[i] = 1;
  // grown back to the skull a cell at a time, not across it nor into air: the core starts ~CLOSE_MM inside the skull
  const steps = Math.max(...r) + 6;       // ~CLOSE_MM + the erosion, and the box corners: to the inner table
  for (let s = 0; s < steps; s++) {
    out = dilateMask(out, d, [1, 1, 1]);
    for (let i = 0; i < N; i++) if (out[i] && !(ins[i] && !barrier[i] && mean[i] > -300)) out[i] = 0;
  }
  return out;
}

const SHEET_SIGMA_MM = 1;     // sheets(): the scale of a skull plate (a newborn's is 1-2 mm thick)
const SHEET_MIN_HU = 100;     // cells this bright and up are looked at
const SHEET_SHARE = 0.35;     // across it the curvature dominates: the second one is at most about this share of it
const SHEET_CURVE = 30;       // HU (Hessian x sigma^2): a weaker sheet is noise (a 2 mm plate of 250 HU: ~50)
const SHEET_RIDGE = 0.8;      // on the ridge: the gradient (x sigma) at most this share of the curvature (a 2 mm plate
                              // between voxels: 0.75); at the edge of a large bright thing (a sinus) ~1: no sheet

/** Bright sheets on a grid (dcv/venography.py sheets): skull plates, also dura. Their Hessian has one strong negative
 *  curvature (across) and little along them, where a vessel has two (a tube) and the inside of a sinus little at all.
 *  HU alone cannot tell a newborn's plates and their edges at the open sutures (150-450 HU) from veins. */
export function sheets(mean: Float32Array, [nx, ny, nz]: D3, sp: D3): Uint8Array {
  const N = mean.length, out = new Uint8Array(N);
  let any = false;
  for (let i = 0; i < N && !any; i++) any = mean[i] >= SHEET_MIN_HU;
  if (!any) return out;
  const g = gauss(mean, [nx, ny, nz], sp.map((k) => SHEET_SIGMA_MM / k) as D3);
  const nxy = nx * ny, w = SHEET_SIGMA_MM * SHEET_SIGMA_MM;
  const [hx, hy, hz] = sp;
  for (let z = 1; z < nz - 1; z++) {
    for (let y = 1; y < ny - 1; y++) {
      for (let x = 1; x < nx - 1; x++) {
        const i = (z * ny + y) * nx + x;
        if (mean[i] < SHEET_MIN_HU) continue;
        const c = g[i];
        const a = (g[i + 1] - 2 * c + g[i - 1]) / (hx * hx) * w;
        const b = (g[i + nx] - 2 * c + g[i - nx]) / (hy * hy) * w;
        const cc = (g[i + nxy] - 2 * c + g[i - nxy]) / (hz * hz) * w;
        const dxy = (g[i + nx + 1] - g[i + nx - 1] - g[i - nx + 1] + g[i - nx - 1]) / (4 * hx * hy) * w;
        const dxz = (g[i + nxy + 1] - g[i + nxy - 1] - g[i - nxy + 1] + g[i - nxy - 1]) / (4 * hx * hz) * w;
        const dyz = (g[i + nxy + nx] - g[i + nxy - nx] - g[i - nxy + nx] + g[i - nxy - nx]) / (4 * hy * hz) * w;
        const [, l2, l3] = eigSym(a, b, cc, dxy, dxz, dyz);
        const slope = Math.hypot((g[i + 1] - g[i - 1]) / (2 * hx), (g[i + nx] - g[i - nx]) / (2 * hy),
          (g[i + nxy] - g[i - nxy]) / (2 * hz)) * SHEET_SIGMA_MM;
        if (l3 < -SHEET_CURVE && Math.abs(l2) < SHEET_SHARE * Math.abs(l3) && slope < SHEET_RIDGE * Math.abs(l3)) out[i] = 1;
      }
    }
  }
  return out;
}

/** Gaussian blur, sigma in cells per axis (separable, +-2.5 sigma, edges repeated). */
function gauss(a: Float32Array, [nx, ny, nz]: D3, sig: D3): Float32Array {
  let src = Float32Array.from(a);
  const strides = [1, nx, nx * ny], lens = [nx, ny, nz];
  for (let ax = 0; ax < 3; ax++) {
    const s = sig[ax], r = Math.ceil(2.5 * s);
    if (r < 1) continue;
    const k = Array.from({ length: 2 * r + 1 }, (_, j) => Math.exp(-0.5 * ((j - r) / s) ** 2));
    const sum = k.reduce((p, q) => p + q, 0);
    for (let j = 0; j < k.length; j++) k[j] /= sum;
    const out = new Float32Array(src.length), st = strides[ax], len = lens[ax];
    for (let i = 0; i < src.length; i++) {
      const pos = Math.floor(i / st) % len;
      let acc = 0;
      for (let j = -r; j <= r; j++) {
        const p = Math.min(len - 1, Math.max(0, pos + j));
        acc += k[j + r] * src[i + (p - pos) * st];
      }
      out[i] = acc;
    }
    src = out;
  }
  return src;
}

/** Eigenvalues of the symmetric [[a, d, e], [d, b, f], [e, f, c]], sorted by absolute value. */
function eigSym(a: number, b: number, c: number, d: number, e: number, f: number): [number, number, number] {
  const p1 = d * d + e * e + f * f, q = (a + b + c) / 3;
  const p = Math.sqrt(((a - q) ** 2 + (b - q) ** 2 + (c - q) ** 2 + 2 * p1) / 6) + 1e-12;
  const b0 = (a - q) / p, b1 = d / p, b2 = e / p, b3 = (b - q) / p, b4 = f / p, b5 = (c - q) / p;
  const det = b0 * (b3 * b5 - b4 * b4) - b1 * (b1 * b5 - b4 * b2) + b2 * (b1 * b4 - b3 * b2);
  const phi = Math.acos(Math.min(1, Math.max(-1, det / 2))) / 3;
  const l1 = q + 2 * p * Math.cos(phi), l3 = q + 2 * p * Math.cos(phi + (2 * Math.PI) / 3), l2 = 3 * q - l1 - l3;
  return [l1, l2, l3].sort((u, v) => Math.abs(u) - Math.abs(v)) as [number, number, number];
}

/** A contrast word in a series name ("Head + CM", "POST", "KONTRAS": dcv/phases.py CONTRAST_NAME). */
export function contrastName(description: string): boolean {
  return /\+ ?(CM|K|C|KONTRAS|CONTRAST)($|[^A-Z])|(^|[^A-Z])(C\+|KONTRAS|CONTRAST|POST|ENHANCED|CE)($|[^A-Z])/.test(description.toUpperCase());
}

/** A newborn's skull: most of its bright bone below the bone threshold (cells >= 500 HU / >= 250 HU on a 2 x 2 grid:
 *  real newborn CT 0.29, an infant 0.76, adult CTAs 0.57-0.59; dcv/segment.py thin_skull). Its plates are told by their
 *  shape, and its plain scan is not read: lined up it still matched too little (real CT: 74 % after ~30 s). */
export function thinSkull(vol: Volume): boolean {
  if (vol.modality !== "CT") return false;
  const { cols, rows, n } = vol, d = vol.data;
  let dense = 0, bright = 0;
  for (let z = 0; z < n; z += 2) {
    for (let y = 0; y + 1 < rows; y += 2) {
      for (let x = 0; x + 1 < cols; x += 2) {
        const i = (z * rows + y) * cols + x;
        const m = Math.max(d[i], d[i + 1], d[i + cols], d[i + cols + 1]);
        if (m >= 250) bright++;
        if (m >= 500) dense++;
      }
    }
  }
  return bright > 0 && dense / bright < 0.4;
}

/** The study's plain scan, when it lies on exactly this grid (another field of view or tilt: not used here). */
export function sameGrid(plain: Volume, vol: Volume): boolean {
  if (plain.cols !== vol.cols || plain.rows !== vol.rows || plain.n !== vol.n || plain.space !== vol.space) return false;
  if (plain.spacing.some((k, i) => Math.abs(k - vol.spacing[i]) > 0.01)) return false;
  const a = plain.positions?.[0], b = vol.positions?.[0];
  return !a || !b || Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) < 0.5;
}

/** Share of this series' bone (>= 300 HU, every 2nd voxel) with bone (>= 200) within a voxel on the plain scan: a patient
 *  who moved between the scans (real newborn CT: ~40 % instead of ~99 %) would turn every bone edge into a vessel. */
export function linesUp(plain: ArrayLike<number>, vol: Volume): number {
  const { cols, rows, n } = vol, d = vol.data;
  let hit = 0, all = 0;
  for (let z = 1; z < n - 1; z += 2) {
    for (let y = 1; y < rows - 1; y += 2) {
      for (let x = 1; x < cols - 1; x += 2) {
        const i = (z * rows + y) * cols + x;
        if (d[i] < 300) continue;
        all++;
        // as the desktop: the plain scan's every 2nd voxel, ±1 of those around (dcv/venography.py lines_up)
        if (maxAround(plain, i, cols, cols * rows, 1, 1, 2) >= 200) hit++;
      }
    }
  }
  return all ? hit / all : 0;
}

/** Maximum of a over ±r voxels in-plane and ±rz planes around voxel i (rows of nx, planes of nxy), every step-th. */
function maxAround(a: ArrayLike<number>, i: number, nx: number, nxy: number, r: number, rz: number, step = 1): number {
  let m = -Infinity;
  const len = a.length;
  for (let dz = -rz; dz <= rz; dz++) {
    for (let dy = -r; dy <= r; dy++) {
      const base = i + (dz * nxy + dy * nx) * step;
      for (let dx = -r; dx <= r; dx++) {
        const j = base + dx * step;
        if (j >= 0 && j < len && a[j] > m) m = a[j];
      }
    }
  }
  return m;
}

/** The masks on the 3D copy's grid (step f). bone = the bone lock's area or null; plain = the plain scan on this grid;
 *  reach = voxels around a voxel on the plain scan that must hold nothing bright (2 after a motion was undone: what is
 *  left of it, ~1 mm on a newborn's skull, would turn its edge into vessels). */
export function venoMasks(vol: Volume, cav: BoneRegion, bone: BoneRegion | null, plain: ArrayLike<number> | null, f: number,
  reach = 1, plates: BoneRegion | null = null, arteries: BoneRegion | null = null): VenoMasks {
  const { cols, rows, n } = vol, d = vol.data, nxy = cols * rows;
  const nx = Math.ceil(cols / f), ny = Math.ceil(rows / f), nz = n, N = nx * ny * nz;
  const words = Math.ceil(N / 32);
  const inner = new Uint32Array(words), boneB = new Uint32Array(words), lumenB = new Uint32Array(words);
  const r = Math.max(1, Math.round(2 / f));        // no plain scan: ~2 voxels of the volume around, in-plane
  const [sz, sy, sx] = vol.spacing;
  const rim = [Math.max(1, Math.round(RIM_MM / sx)), Math.max(1, Math.round(RIM_MM / sy)), Math.max(1, Math.round(RIM_MM / sz))];
  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) {
      const Y = y * f;
      for (let x = 0; x < nx; x++) {
        const X = x * f, i = (z * rows + Y) * cols + X, v = d[i], k = (z * ny + y) * nx + x, w = k >>> 5, b = 1 << (k & 31);
        const inCav = maskAt(cav, X, Y, z);
        let lumen = false, before = false;
        // veins only: an artery (phases.ts vesselTrees on one series), a voxel wider, is no lumen (dcv/venography.py)
        const artery = arteries !== null && v >= LUMEN_MIN_HU && nearMask(arteries, X, Y, z, f, cols, rows, n);
        // skull plates and their edges, dura: no vessel (sheets())
        if (v >= LUMEN_MIN_HU && plates !== null && maskAt(plates, X, Y, z)) before = true;
        else if (v >= LUMEN_MIN_HU) {
          if (plain) {
            const near = maxAround(plain, i, cols, nxy, reach, reach);
            before = near >= BEFORE_HU;
            if (v < LUMEN_MAX_HU && plain[i] < PLAIN_MAX_HU) {
              // within RIM_MM of the cavity's edge lies the skull, on a newborn only 150-400 HU: clear of BEFORE_HU there
              const edge = inCav && nearEdge(cav, X, Y, z, rim, cols, rows, n);
              const sm = mean7(d, i, cols, nxy), pm = mean7(plain, i, cols, nxy);
              lumen = sm >= MEAN_MIN_HU && sm - pm >= ENHANCE_HU && near < (edge ? BEFORE_HU : PLAIN_NEAR_HU);
            }
          } else if (v < LUMEN_MAX_HU && (inCav || v >= OUTSIDE_MIN_HU)) {
            // away from bone; a newborn's skull is 150-450 HU (real CT), so anything from 200 HU outside the cavity counts
            lumen = mean7(d, i, cols, nxy) >= MEAN_MIN_HU && !brightAround(d, cav, i, X, Y, z, cols, rows, n, r);
            before = !lumen;
          } else before = true;
        }
        if (artery) lumen = false;
        if (lumen) lumenB[w] |= b;
        const isBone = ((bone !== null && maskAt(bone, X, Y, z)) || (inCav && v >= LUMEN_MIN_HU && before && !artery)) && !lumen;
        if (isBone) boneB[w] |= b;
        else if (inCav) inner[w] |= b;
      }
    }
  }
  return { nx, ny, nz, f, inner, layer: ring(inner, nx, ny, nz), bone: boneB, lumen: lumenB, plain: plain !== null };
}

/** The mask at voxel (X, Y, z) or one step of the copy's grid (f voxels in-plane, a plane) around it. */
function nearMask(m: BoneRegion, X: number, Y: number, z: number, f: number, cols: number, rows: number, n: number): boolean {
  for (let dz = -1; dz <= 1; dz++) {
    const Z = z + dz;
    if (Z < 0 || Z >= n) continue;
    for (let dy = -f; dy <= f; dy += f) {
      const Yy = Y + dy;
      if (Yy < 0 || Yy >= rows) continue;
      for (let dx = -f; dx <= f; dx += f) {
        const Xx = X + dx;
        if (Xx >= 0 && Xx < cols && maskAt(m, Xx, Yy, Z)) return true;
      }
    }
  }
  return false;
}

/** Mean of voxel i of a and its 6 neighbours (edges: the voxel itself): a lumen is more than a spike of noise. */
function mean7(a: ArrayLike<number>, i: number, nx: number, nxy: number): number {
  const len = a.length, c = a[i];
  let t = c;
  for (const s of [1, -1, nx, -nx, nxy, -nxy]) {
    const j = i + s;
    t += j >= 0 && j < len ? a[j] : c;
  }
  return t / 7;
}

/** Voxel (X, Y, Z) of the cavity has no cavity rim[0..2] voxels away along x, y or z: the cavity's edge. */
function nearEdge(cav: BoneRegion, X: number, Y: number, Z: number, rim: number[], cols: number, rows: number, n: number): boolean {
  return !maskAt(cav, Math.max(0, X - rim[0]), Y, Z) || !maskAt(cav, Math.min(cols - 1, X + rim[0]), Y, Z)
    || !maskAt(cav, X, Math.max(0, Y - rim[1]), Z) || !maskAt(cav, X, Math.min(rows - 1, Y + rim[1]), Z)
    || !maskAt(cav, X, Y, Math.max(0, Z - rim[2])) || !maskAt(cav, X, Y, Math.min(n - 1, Z + rim[2]));
}

/** Bone near voxel i: >= 300 HU, or >= 200 outside the cavity, within ±r in-plane and ±1 plane. */
function brightAround(d: ArrayLike<number>, cav: BoneRegion, i: number, X: number, Y: number, z: number, cols: number, rows: number,
  n: number, r: number): boolean {
  for (let dz = -1; dz <= 1; dz++) {
    const Z = z + dz;
    if (Z < 0 || Z >= n) continue;
    for (let dy = -r; dy <= r; dy++) {
      const YY = Y + dy;
      if (YY < 0 || YY >= rows) continue;
      for (let dx = -r; dx <= r; dx++) {
        const XX = X + dx;
        if (XX < 0 || XX >= cols) continue;
        const v = d[(Z * rows + YY) * cols + XX];
        if (v >= 300 || (v >= 200 && !maskAt(cav, XX, YY, Z))) return true;
      }
    }
  }
  return false;
}

/** The voxels 6-adjacent to the mask and not in it, on bits (index (z·ny + y)·nx + x; a row's end touches the next
 *  row's start, one voxel at the edge of the volume). */
function ring(m: Uint32Array, nx: number, ny: number, nz: number): Uint32Array {
  const out = m.slice();
  for (const k of [1, nx, nx * ny]) {
    orShifted(out, m, k);
    orShifted(out, m, -k);
  }
  for (let w = 0; w < out.length; w++) out[w] &= ~m[w];
  void nz;
  return out;
}

/** out |= m moved by k bits (bit i of m lands on bit i + k). */
function orShifted(out: Uint32Array, m: Uint32Array, k: number): void {
  const q = Math.floor(k / 32), s = ((k % 32) + 32) % 32, W = m.length;
  for (let w = 0; w < W; w++) {
    const a = w - q, lo = a >= 0 && a < W ? m[a] : 0, hi = a - 1 >= 0 && a - 1 < W ? m[a - 1] : 0;
    out[w] |= s === 0 ? lo : (lo << s) | (hi >>> (32 - s));
  }
}

/** Value in the 3D copy for voxel value v (already floored) at bit k of the masks; boneOn = bone lock on. */
export function encodeVeno(m: VenoMasks, v: number, k: number, boneOn: boolean): number {
  const w = k >>> 5, b = 1 << (k & 31);
  if (m.layer[w] & b) return VEIN_OFFSET;
  // only a lumen reaches the vein curve: noise in a thin newborn scan put ~15 % of the brain at 60-80 HU (a blue haze)
  if (m.inner[w] & b) return VEIN_OFFSET + (m.lumen[w] & b ? (v > 3000 ? 3000 : v) : v > BRAIN_TOP_HU ? BRAIN_TOP_HU : v);
  if (boneOn && m.bone[w] & b) return -BONE_OFFSET - v;
  if (v >= LUMEN_MIN_HU && !(m.lumen[w] & b)) return HIDDEN_HU;
  return v;
}

/** Bit index of volume voxel (X, Y, Z) in the masks. */
export function venoIndex(m: VenoMasks, X: number, Y: number, Z: number): number {
  return (Z * m.ny + Math.floor(Y / m.f)) * m.nx + Math.floor(X / m.f);
}

// ---------------------------------------------------------------- the plain scan on this grid, lined up

/** Patient coordinates of voxel (x, y, z): o + ex·x + ey·y + ez·z; inv = the inverse of the matrix [ex ey ez]. */
interface Grid {
  o: V3;
  ex: V3;
  ey: V3;
  ez: V3;
  inv: M3;
}

function gridOf(v: Volume): Grid | null {
  const p = v.positions;
  if (!p || !p.length) return null;
  const [sz, sy, sx] = v.spacing, D = v.direction;
  const ex = scale(col(D, 0), sx), ey = scale(col(D, 1), sy);
  // the step between planes from their positions (the order of the planes, whatever the normal says)
  const ez = v.n > 1 ? scale(sub(p[v.n - 1], p[0]), 1 / (v.n - 1)) : scale(col(D, 2), sz);
  const m = [ex[0], ey[0], ez[0], ex[1], ey[1], ez[1], ex[2], ey[2], ez[2]];
  return { o: p[0], ex, ey, ez, inv: inverse(m) };
}

function inverse(m: M3): M3 {
  const [a, b, c, d, e, f, g, h, i] = m;
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const det = a * A + b * B + c * C;
  return [A / det, -(b * i - c * h) / det, (b * f - c * e) / det, B / det, (a * i - c * g) / det, -(a * f - c * d) / det,
    C / det, -(a * h - b * g) / det, (a * e - b * d) / det];
}

const toWorld = (g: Grid, x: number, y: number, z: number): V3 =>
  [g.o[0] + g.ex[0] * x + g.ey[0] * y + g.ez[0] * z, g.o[1] + g.ex[1] * x + g.ey[1] * y + g.ez[1] * z,
    g.o[2] + g.ex[2] * x + g.ey[2] * y + g.ez[2] * z];
const toIndex = (g: Grid, w: V3): V3 => mulMV(g.inv, sub(w, g.o));

/** The patient's motion between the plain scan and this series: a point p of the plain scan lies at rot·p + shift here. */
export interface Motion {
  rot: M3;
  shift: V3;
}

/** The plain scan's HU on vol's grid: the nearest voxel (the lumen rule looks around it anyway); null without patient
 *  coordinates in common. motion: from register(). */
export function plainOnGrid(plain: Volume, vol: Volume, motion: Motion | null = null): Int16Array | null {
  if (plain.space !== vol.space) return null;
  const gp = gridOf(plain), gv = gridOf(vol);
  if (!gp || !gv) return null;
  const R = motion?.rot ?? I3, shift: V3 = motion?.shift ?? [0, 0, 0];
  const at = (x: number, y: number, z: number): V3 => toIndex(gp, mulMtV(R, sub(toWorld(gv, x, y, z), shift)));
  const a = at(0, 0, 0), dx = sub(at(1, 0, 0), a), dy = sub(at(0, 1, 0), a), dz = sub(at(0, 0, 1), a);
  const { cols, rows, n } = vol, pc = plain.cols, pr = plain.rows, pn = plain.n, src = plain.data;
  const out = new Int16Array(cols * rows * n).fill(-1024);
  for (let z = 0; z < n; z++) {
    for (let y = 0; y < rows; y++) {
      let qx = a[0] + y * dy[0] + z * dz[0], qy = a[1] + y * dy[1] + z * dz[1], qz = a[2] + y * dy[2] + z * dz[2];
      const row = (z * rows + y) * cols;
      for (let x = 0; x < cols; x++, qx += dx[0], qy += dx[1], qz += dx[2]) {
        const X = Math.round(qx), Y = Math.round(qy), Z = Math.round(qz);
        if (X >= 0 && X < pc && Y >= 0 && Y < pr && Z >= 0 && Z < pn) out[row + x] = src[(Z * pr + Y) * pc + X];
      }
    }
  }
  return out;
}

const REG_CELL_MM = [2, 1], REG_CAP_MM = [10, 4], REG_POINTS = 20000, REG_BONE_HU = 300;

/** The skull's motion between the plain scan and this series (dcv/venography.py register): the plain scan's bone (points)
 *  moved onto this series' bone (a map of the distance to it), first on ~2 mm cells from a search of shifts, then
 *  rotations and shifts by halving steps, then again on ~1 mm cells. null when either has no bone. */
export function register(plain: Volume, vol: Volume): Motion | null {
  const gp = gridOf(plain), gv = gridOf(vol);
  if (!gp || !gv) return null;
  let motion: Motion = { rot: [...I3], shift: [0, 0, 0] };
  for (let k = 0; k < REG_CELL_MM.length; k++) {
    const cell = REG_CELL_MM[k], cap = REG_CAP_MM[k];
    const target = distanceMap(vol, cell, cap), pts = bonePoints(plain, gp, cell);
    if (!target || !pts) return null;
    let cx = 0, cy = 0, cz = 0;
    const np = pts.length / 3;
    for (let i = 0; i < np; i++) {
      cx += pts[3 * i];
      cy += pts[3 * i + 1];
      cz += pts[3 * i + 2];
    }
    const centre: V3 = [cx / np, cy / np, cz / np];
    const cost = (p: number[]) => meanDistance(gv, target, pts, motionOf(p, centre), cap);
    let params = paramsOf(motion, centre);
    if (k === 0) {
      // shifts first: a grid of ±16 mm
      let best = cost(params), bestP = params;
      for (let dz = -16; dz <= 16; dz += 4) for (let dy = -16; dy <= 16; dy += 4) for (let dx = -16; dx <= 16; dx += 4) {
        const p = [params[0], params[1], params[2], params[3] + dx, params[4] + dy, params[5] + dz];
        const c = cost(p);
        if (c < best) [best, bestP] = [c, p];
      }
      params = bestP;
    }
    params = descend(cost, params, k === 0 ? 4 : 1, 0.25);
    motion = motionOf(params, centre);
  }
  return motion;
}

interface DistMap {
  dist: Float32Array;
  d: D3;
  f: number;
  fz: number;
}

/** Distance (mm, at most cap) to this series' bone on cells of ~cell mm. */
function distanceMap(vol: Volume, cell: number, cap: number): DistMap | null {
  const [sz, , sx] = vol.spacing;
  const f = Math.max(1, Math.round(cell / sx)), fz = Math.max(1, Math.round(cell / sz));
  const { top, d } = pool(vol.data, vol.cols, vol.rows, vol.n, f, fz);
  const N = top.length;
  let cur: Uint8Array = new Uint8Array(N), any = false;
  for (let i = 0; i < N; i++) if (top[i] >= REG_BONE_HU) {
    cur[i] = 1;
    any = true;
  }
  if (!any) return null;
  const size = (2 * sx * f + sz * fz) / 3;
  const dist = new Float32Array(N).fill(cap);
  for (let i = 0; i < N; i++) if (cur[i]) dist[i] = 0;
  for (let s = 1; s <= Math.ceil(cap / size); s++) {
    const nxt = dilateMask(cur, d, [1, 1, 1]);
    for (let i = 0; i < N; i++) if (nxt[i] && !cur[i]) dist[i] = Math.min(cap, s * size);
    cur = nxt;
  }
  return { dist, d, f, fz };
}

/** Patient coordinates (x, y, z, x, y, z ...) of the plain scan's bone on cells of ~cell mm, at most REG_POINTS. */
function bonePoints(plain: Volume, g: Grid, cell: number): Float64Array | null {
  const [sz, , sx] = plain.spacing;
  const f = Math.max(1, Math.round(cell / sx)), fz = Math.max(1, Math.round(cell / sz));
  const { top, d } = pool(plain.data, plain.cols, plain.rows, plain.n, f, fz);
  const [nx, ny] = d;
  const idx: number[] = [];
  for (let i = 0; i < top.length; i++) if (top[i] >= REG_BONE_HU) idx.push(i);
  if (idx.length < 50) return null;
  const step = Math.max(1, idx.length / REG_POINTS), n = Math.floor(idx.length / step);
  const out = new Float64Array(3 * n);
  for (let k = 0; k < n; k++) {
    const i = idx[Math.floor(k * step)], x = i % nx, y = Math.floor(i / nx) % ny, z = Math.floor(i / (nx * ny));
    out.set(toWorld(g, x * f + (f - 1) / 2, y * f + (f - 1) / 2, z * fz + (fz - 1) / 2), 3 * k);
  }
  return out;
}

/** The motion of params = (rotation vector in degrees, shift in mm) about centre. */
function motionOf(p: number[], centre: V3): Motion {
  const v: V3 = [p[0], p[1], p[2]], a = Math.hypot(v[0], v[1], v[2]) * Math.PI / 180;
  const R = a < 1e-12 ? [...I3] : rot(v, a);
  return { rot: R, shift: sub(add(centre, [p[3], p[4], p[5]]), mulMV(R, centre)) };
}

/** params of a motion about centre (the inverse of motionOf, for the small rotations register builds). */
function paramsOf(m: Motion, centre: V3): number[] {
  const R = m.rot, c = Math.min(1, Math.max(-1, (R[0] + R[4] + R[8] - 1) / 2)), a = Math.acos(c);
  const axis: V3 = a < 1e-9 ? [0, 0, 0] : scale([R[7] - R[5], R[2] - R[6], R[3] - R[1]], 1 / (2 * Math.sin(a)));
  const t = sub(add(m.shift, mulMV(R, centre)), centre);
  const deg = a * 180 / Math.PI;
  return [axis[0] * deg, axis[1] * deg, axis[2] * deg, t[0], t[1], t[2]];
}

/** Mean distance (mm, at most cap) from the moved points to this series' bone, trilinear in the map. */
function meanDistance(g: Grid, m: DistMap, pts: Float64Array, motion: Motion, cap: number): number {
  const [nx, ny, nz] = m.d, R = motion.rot, t = motion.shift, dist = m.dist, nxy = nx * ny;
  let sum = 0;
  const n = pts.length / 3;
  for (let k = 0; k < n; k++) {
    const px = pts[3 * k], py = pts[3 * k + 1], pz = pts[3 * k + 2];
    const q = toIndex(g, [R[0] * px + R[1] * py + R[2] * pz + t[0], R[3] * px + R[4] * py + R[5] * pz + t[1],
      R[6] * px + R[7] * py + R[8] * pz + t[2]]);
    const X = (q[0] - (m.f - 1) / 2) / m.f, Y = (q[1] - (m.f - 1) / 2) / m.f, Z = (q[2] - (m.fz - 1) / 2) / m.fz;
    const x0 = Math.floor(X), y0 = Math.floor(Y), z0 = Math.floor(Z);
    if (x0 < 0 || y0 < 0 || z0 < 0 || x0 >= nx - 1 || y0 >= ny - 1 || z0 >= nz - 1) {
      sum += cap;
      continue;
    }
    const fx = X - x0, fy = Y - y0, fz = Z - z0, i = (z0 * ny + y0) * nx + x0;
    const c00 = dist[i] * (1 - fx) + dist[i + 1] * fx, c10 = dist[i + nx] * (1 - fx) + dist[i + nx + 1] * fx;
    const c01 = dist[i + nxy] * (1 - fx) + dist[i + nxy + 1] * fx, c11 = dist[i + nxy + nx] * (1 - fx) + dist[i + nxy + nx + 1] * fx;
    sum += (c00 * (1 - fy) + c10 * fy) * (1 - fz) + (c01 * (1 - fy) + c11 * fy) * fz;
  }
  return sum / n;
}

/** Coordinate descent on the 6 params (degrees, mm) from step down to last, halving. */
function descend(cost: (p: number[]) => number, params: number[], step: number, last: number): number[] {
  let best = cost(params);
  for (; step >= last; step /= 2) {
    for (let better = true; better;) {
      better = false;
      for (let i = 0; i < 6; i++) {
        for (const sgn of [1, -1]) {
          const p = params.slice();
          p[i] += sgn * step;
          const c = cost(p);
          if (c < best - 1e-6) [best, params, better] = [c, p, true];
        }
      }
    }
  }
  return params;
}

/** The plain scan on vol's grid as it lies (no registration), and how well it lines up there (linesUp). */
export function plainOnVol(plain: Volume, vol: Volume): { data: ArrayLike<number> | null; match: number | null } {
  const data = sameGrid(plain, vol) ? plain.data : plainOnGrid(plain, vol);
  return { data, match: data ? linesUp(data, vol) : null };
}

/** The plain scan for the venography (and the bone lock) of vol: on its grid, lined up when the patient moved (register;
 *  lineUp false: not), and used only when it lines up well enough (match >= MATCH_MIN). */
export function plainFor(plain: Volume, vol: Volume, lineUp = true): { data: ArrayLike<number> | null; match: number | null; moved: Motion | null } {
  let data: ArrayLike<number> | null = sameGrid(plain, vol) ? plain.data : plainOnGrid(plain, vol);
  if (!data) return { data: null, match: null, moved: null };
  let match = linesUp(data, vol), moved: Motion | null = null;
  if (match < MATCH_MIN && lineUp) {
    const m = register(plain, vol);
    const q = m ? plainOnGrid(plain, vol, m) : null;
    if (q && m) {
      const k = linesUp(q, vol);
      if (k > match) [data, match, moved] = [q, k, m];
    }
  }
  return { data: match >= MATCH_MIN ? data : null, match, moved };
}
