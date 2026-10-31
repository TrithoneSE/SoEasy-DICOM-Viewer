/**
 * Bone colour lock (dcv/segment.py bone_and_vessels + bone_region + update_render_data). One HU threshold cannot tell
 * bone from bright vessels on a CTA: contrast in a large artery or an aneurysm passes 500 HU (real CTA: arteries ~590,
 * max ~800; bone 1250-1650). Density and shape can. Two rules find the bright vessels, and neither is enough alone:
 *   by pieces: a 1 mm erosion cuts thin contacts; a piece with much of its volume at cortical density is bone;
 *   away from bone: for a vessel in broad contact with bone, take away dense bone and its edge; a thick remainder that
 *   runs away from dense bone is a vessel.
 * Bone = high minus vessels. Its area (closed by 3 mm: diploe, widened by a 1.5 mm rim, minus the vessels and their
 * wall) is coloured only by "bone" curves, everything else only by the other curves.
 *
 * Bone-area voxels are stored mirrored as -BONE_OFFSET - HU (not shifted): interpolating from the area edge (low
 * HU) to the tissue outside then passes only through bone-domain values below the edge HU and air (transparent).
 * The desktop uses 10000; the GPU copy here is half float, so 4096 keeps the bone domain at 4 HU precision.
 *
 * Arteries / veins (only with a curve of layer "vein"): arteries = bright voxels and what connects to them, veins =
 * the other vessels, stored as VEIN_OFFSET + HU. On a late phase both are equally bright: they cannot be separated.
 *
 * All masks are computed on a grid halved in-plane for large volumes on phones and tablets, from the brightest voxel of
 * each cell, and kept as 1 bit per voxel. Components are flood-filled with a typed-array list: no label volume.
 */
import type { Volume } from "../dicom/volume";

export const BONE_OFFSET = 4096;
/** CT air floor of the 3D copy: padding at -2048 / -3024 would otherwise reach the bone domain. */
export const CT_FLOOR = -1024;
/** Veins in the 3D copy: VEIN_OFFSET + HU, above the normal domain. Between normal and bone every bone surface would
 *  cross the vein values and get blue specks; here a vein surface crosses the arteries' range for ~2 % of a voxel. */
export const VEIN_OFFSET = 20000;
export const VEIN_MIN_HU = 110;
export const ARTERY_TOP_HU = 450;
/** Cortical bone: contrast in vessels stays well below it. */
const BONE_CORE_HU = 1000;
type R3 = [number, number, number];        // x, y, z
type Dims = [number, number, number];

export interface BoneRegion {
  /** 1 bit per voxel of the region grid (index (z·ny + y)·nx + x). */
  bits: Uint32Array;
  nx: number;
  ny: number;
  nz: number;
  f: number;                 // in-plane step of the region grid
}

/**
 * Byte masks are reused: one classification needs ~40 dilations of 3 passes each, and on a 37 M voxel grid leaving those
 * to the garbage collector held several times the memory of what is really in use.
 */
const pool: Uint8Array[] = [];

function take(n: number, zero = false): Uint8Array {
  const i = pool.findIndex((b) => b.length === n);
  if (i < 0) return new Uint8Array(n);
  const b = pool.splice(i, 1)[0];
  if (zero) b.fill(0);
  return b;
}

function give(...masks: Uint8Array[]): void {
  pool.push(...masks);
}

function dilateAxis(src: Uint8Array, out: Uint8Array, nx: number, ny: number, nz: number, axis: 0 | 1 | 2, r: number): void {
  if (axis === 0) {
    for (let l = 0; l < ny * nz; l++) {
      const start = l * nx;
      let count = 0;
      for (let i = 0; i < Math.min(r, nx); i++) count += src[start + i];
      for (let i = 0; i < nx; i++) {                  // window [i - r, i + r]
        if (i + r < nx) count += src[start + i + r];
        if (i - r - 1 >= 0) count -= src[start + i - r - 1];
        out[start + i] = count > 0 ? 1 : 0;
      }
    }
    return;
  }
  // y and z: a running count per column, plane after plane in memory order (3× faster than walking along each line)
  const plane = axis === 1 ? nx : nx * ny, len = axis === 1 ? ny : nz, blocks = axis === 1 ? nz : 1;
  const counts = new Uint16Array(plane);
  for (let b = 0; b < blocks; b++) {
    const base = b * nx * ny;
    counts.fill(0);
    for (let i = 0; i < Math.min(r, len); i++) {
      const o = base + i * plane;
      for (let j = 0; j < plane; j++) counts[j] += src[o + j];
    }
    for (let i = 0; i < len; i++) {
      if (i + r < len) {
        const o = base + (i + r) * plane;
        for (let j = 0; j < plane; j++) counts[j] += src[o + j];
      }
      if (i - r - 1 >= 0) {
        const o = base + (i - r - 1) * plane;
        for (let j = 0; j < plane; j++) counts[j] -= src[o + j];
      }
      const o = base + i * plane;
      for (let j = 0; j < plane; j++) out[o + j] = counts[j] > 0 ? 1 : 0;
    }
  }
}

/** A new mask (from the pool); `m` stays. */
function dilate(m: Uint8Array, d: [number, number, number], r: [number, number, number]): Uint8Array {
  let cur = m;
  for (const ax of [0, 1, 2] as const) {
    if (r[ax] <= 0) continue;
    const out = take(m.length);
    dilateAxis(cur, out, ...d, ax, r[ax]);
    if (cur !== m) give(cur);
    cur = out;
  }
  if (cur !== m) return cur;
  const copy = take(m.length);
  copy.set(m);
  return copy;
}

/** Python's round(): halves to even (mm_to_radii in the desktop), so both apps pick the same radii. */
export function roundEven(v: number): number {
  const r = Math.round(v);
  return Math.abs(v % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

function erode(m: Uint8Array, d: Dims, r: R3): Uint8Array {
  const inv = take(m.length);
  for (let i = 0; i < m.length; i++) inv[i] = m[i] ? 0 : 1;
  const out = dilate(inv, d, r);
  give(inv);
  for (let i = 0; i < out.length; i++) out[i] = out[i] ? 0 : 1;
  return out;
}

/** Each 6-connected component of a mask: `visit` gets its voxels (list[0..n)). */
function components(mask: Uint8Array, [nx, ny, nz]: Dims, visit: (list: Int32Array, n: number) => void): void {
  const seen = take(mask.length, true);
  let list = new Int32Array(1 << 16);
  const nxy = nx * ny;
  for (let s = 0; s < mask.length; s++) {
    if (!mask[s] || seen[s]) continue;
    seen[s] = 1;
    list[0] = s;
    let n = 1;
    for (let h = 0; h < n; h++) {
      const i = list[h];
      if (n + 6 > list.length) {
        const more = new Int32Array(list.length * 2);
        more.set(list);
        list = more;
      }
      const x = i % nx, y = Math.floor(i / nx) % ny, z = Math.floor(i / nxy);
      if (x > 0 && mask[i - 1] && !seen[i - 1]) { seen[i - 1] = 1; list[n++] = i - 1; }
      if (x < nx - 1 && mask[i + 1] && !seen[i + 1]) { seen[i + 1] = 1; list[n++] = i + 1; }
      if (y > 0 && mask[i - nx] && !seen[i - nx]) { seen[i - nx] = 1; list[n++] = i - nx; }
      if (y < ny - 1 && mask[i + nx] && !seen[i + nx]) { seen[i + nx] = 1; list[n++] = i + nx; }
      if (z > 0 && mask[i - nxy] && !seen[i - nxy]) { seen[i - nxy] = 1; list[n++] = i - nxy; }
      if (z < nz - 1 && mask[i + nxy] && !seen[i + nxy]) { seen[i + nxy] = 1; list[n++] = i + nxy; }
    }
    visit(list, n);
  }
  give(seen);
}

function countIn(list: Int32Array, n: number, m: Uint8Array): number {
  let c = 0;
  for (let i = 0; i < n; i++) c += m[list[i]];
  return c;
}

/** Spacing helpers of a grid: (x, y, z) mm per cell. */
function gridOf(sp: R3) {
  const radii = (mm: number): R3 => [roundEven(mm / sp[0]), roundEven(mm / sp[1]), roundEven(mm / sp[2])];
  return { radii, voxel: sp[0] * sp[1] * sp[2], atLeast1: (mm: number): R3 => radii(mm).map((r) => Math.max(1, r)) as R3 };
}

/** Thin contacts cut by 1 mm of erosion; a piece at cortical density is bone. Returns the leftovers that are vessels. */
export function vesselsByPieces(high: Uint8Array, core: Uint8Array, d: Dims, sp: R3, coreShare = 0.15, thinMm3 = 20): Uint8Array {
  const g = gridOf(sp);
  const r1 = g.atLeast1(1);
  const eroded = erode(high, d, r1);
  const seeds = take(high.length, true);
  components(eroded, d, (list, n) => {
    if (countIn(list, n, core) >= Math.max(20, coreShare * n)) for (let i = 0; i < n; i++) seeds[list[i]] = 1;
  });
  const grown = dilate(seeds, d, r1.map((r) => r + 1) as R3);
  give(seeds);
  const bone = take(high.length, true), rest = take(high.length, true);
  for (let i = 0; i < high.length; i++) {
    if (!high[i]) continue;
    if (grown[i]) bone[i] = 1;
    else rest[i] = 1;
  }
  give(grown);
  const near = dilate(bone, d, [1, 1, 1]);
  give(bone);
  const vessels = new Uint8Array(high.length);
  components(rest, d, (list, n) => {
    const small = n * g.voxel <= thinMm3;
    const isBone = countIn(list, n, core) >= Math.max(3, coreShare * n)
      || (small && countIn(list, n, near) > 0 && countIn(list, n, eroded) === 0);
    if (!isBone) for (let i = 0; i < n; i++) vessels[list[i]] = 1;
  });
  give(eroded, rest, near);
  return vessels;
}

/** A vessel in broad contact with bone: what remains without dense bone and its edge, thick and running away from it. */
export function vesselsAwayFromBone(high: Uint8Array, core: Uint8Array, d: Dims, sp: R3, reachMm = 4, farShare = 0.2): Uint8Array {
  const g = gridOf(sp);
  const rc = g.radii(1.5);
  const wide = dilate(core, d, rc);
  const solid = erode(wide, d, rc);
  give(wide);
  for (let i = 0; i < solid.length; i++) solid[i] |= core[i];
  const edge = dilate(solid, d, g.atLeast1(1).map((r) => r + 1) as R3);
  const rest = take(high.length);
  for (let i = 0; i < rest.length; i++) rest[i] = high[i] && !edge[i] ? 1 : 0;
  give(edge);
  const inner = erode(rest, d, [1, 1, 1]);
  const reach = dilate(solid, d, g.radii(reachMm));
  give(solid);
  const vessels = new Uint8Array(high.length);
  components(rest, d, (list, n) => {
    if (countIn(list, n, inner) < 3) return;
    let far = 0;
    for (let i = 0; i < n; i++) if (!reach[list[i]]) far++;
    if (far >= farShare * n) for (let i = 0; i < n; i++) vessels[list[i]] = 1;
  });
  give(rest, inner, reach);
  return vessels;
}

/** (bone, bright vessels) among the voxels >= hu of a grid. */
export function boneAndVessels(data: ArrayLike<number>, d: Dims, sp: R3, hu: number): { bone: Uint8Array; vessels: Uint8Array } {
  const N = d[0] * d[1] * d[2];
  const coreHu = Math.max(BONE_CORE_HU, hu + 200);
  const high = new Uint8Array(N), core = new Uint8Array(N);
  for (let i = 0; i < N; i++) {
    const v = data[i];
    if (v >= hu) high[i] = 1;
    if (v >= coreHu) core[i] = 1;
  }
  const vessels = vesselsByPieces(high, core, d, sp);
  const away = vesselsAwayFromBone(high, core, d, sp);
  for (let i = 0; i < N; i++) {
    if (away[i]) vessels[i] = 1;
    if (vessels[i]) high[i] = 0;
  }
  return { bone: high, vessels };
}

/** The bone area on a grid: bone closed and widened, minus the bright vessels and their wall. */
export function boneArea(data: ArrayLike<number>, d: Dims, sp: R3, hu = 500, closeMm = 3, rimMm = 1.5): Uint8Array {
  const { bone, vessels } = boneAndVessels(data, d, sp, hu);
  const g = gridOf(sp);
  const rc = g.radii(closeMm);
  const wide = dilate(bone, d, rc);
  const closed = erode(wide, d, rc);
  give(wide);
  for (let i = 0; i < closed.length; i++) closed[i] |= bone[i];
  const out = dilate(closed, d, g.radii(rimMm));
  give(closed);
  const wall = dilate(vessels, d, g.atLeast1(1));
  for (let i = 0; i < out.length; i++) if (wall[i] && !bone[i]) out[i] = 0;
  give(wall);
  return out;
}

/** Dev checks only: force the mask grid. */
export let gridStepForChecks: 1 | 2 | null = null;
export function forceGridStep(f: 1 | 2 | null): void {
  gridStepForChecks = f;
}

/** Above this many voxels the masks use the halved grid: phones and tablets early, as they hold several byte masks. */
export function gridStep(voxels: number, mobile: boolean): 1 | 2 {
  if (gridStepForChecks) return gridStepForChecks;
  return voxels > (mobile ? 20e6 : 100e6) ? 2 : 1;
}

/** The values on the mask grid: the volume itself, or the brightest voxel of each cell. */
function gridValues(vol: Volume, f: number): ArrayLike<number> {
  if (f === 1) return vol.data;
  const nx = Math.ceil(vol.cols / f), ny = Math.ceil(vol.rows / f), nz = vol.n;
  const out = vol.data instanceof Int16Array ? new Int16Array(nx * ny * nz).fill(-32768) : new Float32Array(nx * ny * nz).fill(-Infinity);
  const d = vol.data;
  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < vol.rows; y++) {
      const row = (z * vol.rows + y) * vol.cols, o = (z * ny + Math.floor(y / f)) * nx;
      for (let x = 0; x < vol.cols; x++) {
        const c = o + Math.floor(x / f);
        if (d[row + x] > out[c]) out[c] = d[row + x];
      }
    }
  }
  return out;
}

function gridSpacing(vol: Volume, f: number): R3 {
  return [vol.spacing[2] * f, vol.spacing[1] * f, vol.spacing[0]];
}

/** The bone area of a CT volume (see the module comment). `f` = in-plane step of the grid; by default from the size. */
export function boneRegion(vol: Volume, hu = 500, closeMm = 3, rimMm = 1.5, f: 1 | 2 = gridStep(vol.rows * vol.cols * vol.n, false)): BoneRegion {
  const nx = Math.ceil(vol.cols / f), ny = Math.ceil(vol.rows / f), nz = vol.n;
  const bits = pack(boneArea(gridValues(vol, f), [nx, ny, nz], gridSpacing(vol, f), hu, closeMm, rimMm));
  pool.length = 0;
  return { bits, nx, ny, nz, f };
}

/** Auto arterial core (HU): 60 % of the bright end of the soft-tissue side of the scan, within sane limits. */
export function arteryThreshold(data: ArrayLike<number>, free: Uint8Array): number {
  const counts = new Map<number, number>();
  let n = 0;
  for (let i = 0; i < free.length; i++) {
    const v = data[i];
    if (v >= 150 && free[i]) {
      const b = Math.round(v);
      counts.set(b, (counts.get(b) ?? 0) + 1);
      n++;
    }
  }
  if (n < 100) return 300;
  const keys = [...counts.keys()].sort((a, b) => a - b);
  const pos = 0.9 * (n - 1), lo = Math.floor(pos);
  let seen = 0, a = keys[0], b = keys[0], gotA = false;
  for (const k of keys) {
    seen += counts.get(k)!;
    if (!gotA && seen > lo) {
      a = k;
      gotA = true;
    }
    if (seen > lo + 1) {
      b = k;
      break;
    }
    b = k;
  }
  return Math.min(500, Math.max(180, 0.6 * (a + (b - a) * (pos - lo))));
}

/** Voxels >= core grown through >= 0.55 core (the arterial tree), 3 mm further through >= 110 HU, then the wall. */
export function classifyArteries(data: ArrayLike<number>, d: Dims, sp: R3, free: Uint8Array, coreHu: number): Uint8Array {
  const [nx, ny, nz] = d;
  const N = nx * ny * nz, nxy = nx * ny;
  const low = take(N, true), art = take(N, true);
  const lowHu = 0.55 * coreHu;
  for (let i = 0; i < N; i++) if (free[i] && data[i] >= lowHu) low[i] = 1;
  components(low, d, (list, n) => {
    let hit = false;
    for (let i = 0; i < n && !hit; i++) hit = data[list[i]] >= coreHu;
    if (hit) for (let i = 0; i < n; i++) art[list[i]] = 1;
  });
  // one voxel per step through the faint voxels: only what the last step added can reach further
  let front = new Int32Array(1 << 16), count = 0;
  const push = (i: number) => {
    if (count === front.length) {
      const more = new Int32Array(front.length * 2);
      more.set(front);
      front = more;
    }
    front[count++] = i;
  };
  for (let i = 0; i < N; i++) if (art[i]) push(i);
  const steps = Math.max(1, roundEven(3 / Math.min(...sp)));
  for (let s = 0; s < steps; s++) {
    const cur = front.slice(0, count);
    count = 0;
    for (let c = 0; c < cur.length; c++) {
      const i = cur[c];
      const x = i % nx, y = Math.floor(i / nx) % ny, z = Math.floor(i / nxy);
      for (let dz = z > 0 ? -1 : 0; dz <= (z < nz - 1 ? 1 : 0); dz++) {
        for (let dy = y > 0 ? -1 : 0; dy <= (y < ny - 1 ? 1 : 0); dy++) {
          for (let dx = x > 0 ? -1 : 0; dx <= (x < nx - 1 ? 1 : 0); dx++) {
            const j = i + dz * nxy + dy * nx + dx;
            if (!art[j] && free[j] && data[j] >= VEIN_MIN_HU) {
              art[j] = 1;
              push(j);
            }
          }
        }
      }
    }
  }
  const out = dilate(art, d, [1, 1, 1]);
  give(low, art);
  for (let i = 0; i < N; i++) if (!free[i]) out[i] = 0;
  return out;
}

/** Other vessels: >= 110 HU outside the arteries, more than 2 mm from the bone area, in pieces of at least 30 mm3. */
export function classifyVeins(data: ArrayLike<number>, d: Dims, sp: R3, arteries: Uint8Array, region: Uint8Array | null): Uint8Array {
  const g = gridOf(sp);
  const near = region ? dilate(region, d, g.radii(2)) : null;
  const N = arteries.length;
  const m = take(N, true), out = new Uint8Array(N);
  for (let i = 0; i < N; i++) if (data[i] >= VEIN_MIN_HU && !arteries[i] && !(near && near[i])) m[i] = 1;
  components(m, d, (list, n) => {
    if (n * g.voxel >= 30) for (let i = 0; i < n; i++) out[list[i]] = 1;
  });
  give(m);
  if (near) give(near);
  return out;
}

export interface VesselMasks {
  arteries: BoneRegion;
  veins: BoneRegion;
  coreHu: number;
}

/** Arteries and veins of a CT, on the same grid as its bone area (`region`; null = bone lock off: step `f`). */
export function vesselMasks(vol: Volume, region: BoneRegion | null, coreOverride: number | null = null, f: 1 | 2 = 1): VesselMasks {
  const step = (region?.f ?? f) as 1 | 2;
  const nx = Math.ceil(vol.cols / step), ny = Math.ceil(vol.rows / step), nz = vol.n;
  const d: Dims = [nx, ny, nz];
  const data = gridValues(vol, step);
  const sp = gridSpacing(vol, step);
  const area = region ? unpack(region.bits, nx * ny * nz) : null;
  const free = new Uint8Array(nx * ny * nz);
  for (let i = 0; i < free.length; i++) free[i] = area && area[i] ? 0 : 1;
  const coreHu = coreOverride ?? arteryThreshold(data, free);
  const arteries = classifyArteries(data, d, sp, free, coreHu);
  const veins = classifyVeins(data, d, sp, arteries, area);
  pool.length = 0;
  return { arteries: { bits: pack(arteries), nx, ny, nz, f: step }, veins: { bits: pack(veins), nx, ny, nz, f: step }, coreHu };
}

export function unpack(bits: Uint32Array, n: number): Uint8Array {
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) if (bits[i >>> 5] & (1 << (i & 31))) out[i] = 1;
  return out;
}

/** Uint8 0/1 → bitset (8× smaller: 2.5 MB instead of 20 MB for a 300 × 512² CT on the half grid). */
export function pack(m: Uint8Array): Uint32Array {
  const out = new Uint32Array(Math.ceil(m.length / 32));
  for (let i = 0; i < m.length; i++) if (m[i]) out[i >>> 5] |= 1 << (i & 31);
  return out;
}

export function inBone(r: BoneRegion, x: number, y: number, z: number): boolean {
  const i = (z * r.ny + y) * r.nx + x;
  return (r.bits[i >>> 5] & (1 << (i & 31))) !== 0;
}

/** Value stored in the 3D copy for voxel value v (CT: floor at -1024; bone area mirrored). */
export function encode(v: number, inBone: boolean, ct: boolean): number {
  if (ct && v < CT_FLOOR) v = CT_FLOOR;
  return inBone ? -BONE_OFFSET - v : v;
}

/**
 * The same with the vein layer, outside the bone area: veins in their own domain; other voxels >= 110 HU that are no
 * artery are transparent (brain noise, bone edge); arteries clamped to 450 HU so a vein surface crosses little.
 */
export function encodeVessel(v: number, isArtery: boolean, isVein: boolean): number {
  if (isVein) return VEIN_OFFSET + Math.min(v, 3000);
  if (v >= VEIN_MIN_HU && !isArtery) return VEIN_MIN_HU - 1;
  return v > ARTERY_TOP_HU ? ARTERY_TOP_HU : v;
}

/** The bit of a mask for a voxel of the full-resolution grid. */
export function maskAt(r: BoneRegion, x: number, y: number, z: number): boolean {
  const i = (z * r.ny + Math.floor(y / r.f)) * r.nx + Math.floor(x / r.f);
  return (r.bits[i >>> 5] & (1 << (i & 31))) !== 0;
}
