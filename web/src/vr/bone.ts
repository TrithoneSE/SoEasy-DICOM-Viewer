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
 * the other vessels, stored as VEIN_OFFSET + HU. On a late phase both are equally bright: one series cannot separate
 * them, two can (phases.ts).
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
// the bone area from the plain scan (dcv/segment.py VolumeState.bone_source, _plain_area)
/** Bone on the plain scan (blood without contrast ~40 HU; a child's facial bones 190-760 with it). */
export const PLAIN_BONE_HU = 150;
/** The plain scan's bone counts within this: one voxel of shift between the scans moves a bone edge by hundreds of HU. */
export const PLAIN_TOL_MM = 0.9;
/** Contrast: brighter than anything within PLAIN_TOL_MM on the plain scan by this, */
export const ENHANCE_HU = 50;
/** and below this there: a lumen is blood before contrast (~40-60 HU, ~100 at its edge). Brighter is bone: the noise of
 *  dense bone passes ENHANCE_HU (rings on the skull), and a plain scan of thicker slices blurs thin bone (real CTA, 1.25
 *  vs 0.625 mm: the maxilla's walls 150-280 HU). */
export const LUMEN_PLAIN_MAX_HU = 130;
/** Of the 27 cells around one, this many enhanced too: alone it is noise (grain on the face). */
export const ENHANCE_NEIGHBOURS = 9;
/** A lumen against bone: brighter than the plain scan at the cell itself by this, within LUMEN_EDGE_MM of a lumen found
 *  by the test above (real CTA: an aneurysm and its ICA on the clinoid took bone; dcv/segment.py). */
export const LUMEN_EDGE_HU = 150, LUMEN_EDGE_MM = 1;
/** What did not get brighter is stored no higher than this: under every vessel curve. */
export const NOT_VESSEL_HU = 100;
/** slabPlain (dcv/venography.py slab_plain): the plain scan lined up slab by slab along the body, each slab moved by up to
 *  SLAB_SHIFT_MM, scored on SLAB_POINTS of its bone; a slab lines up from SLAB_MATCH (venography.ts MATCH_MIN). */
export const SLAB_MM = 40, SLAB_SHIFT_MM = 4, SLAB_POINTS = 6000, SLAB_MATCH = 0.85;
/** A newborn's skull plate is bone from this HU (real CT: 150-450), and its bone curves start here (dcv/segment.py). */
export const THIN_BONE_HU = 150;
/** phases.intracranialTop from this: a CTA (real: 161-208), its bright vessels may pass the bone threshold; a routine
 *  '+ CM' head CT (74-112) has none, and a child's facial bones and spine (no cortical core) were taken for vessels. */
export const CTA_TOP_HU = 140;
type R3 = [number, number, number];        // x, y, z
type Dims = [number, number, number];

/** What the bone area is made from (dcv/segment.py bone_source): the plain scan, a CTA's bone less its bright vessels,
 *  a newborn's skull (all from the threshold and its plates by shape), or (any other CT) all from the threshold. */
export type BoneSource = "plain" | "cta" | "thin" | "ct";

export interface BoneRegion {
  /** 1 bit per voxel of the region grid (index (z·ny + y)·nx + x). */
  bits: Uint32Array;
  nx: number;
  ny: number;
  nz: number;
  f: number;                 // in-plane step of the region grid
  /** Bone from the plain scan: the cells that got brighter with contrast. Outside the area every other one is no
   *  vessel (NOT_VESSEL_HU). */
  enhanced?: Uint32Array;
  /** Bone from the plain scan: how many planes it lined up with (the others keep the rule of the series without it). */
  plainPlanes?: number;
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

/**
 * Dilations work on bits, 32 cells of a row per word: x shifts within and across words, y and z OR whole rows. The bone
 * area takes ~36 axis passes; on bytes they were ~2.5 s of a 0.625 mm head CT's 3 s on a laptop (far more on a phone).
 */
function packRows(m: Uint8Array, nx: number, rows: number, invert: boolean): Uint32Array {
  const wpr = (nx + 31) >>> 5, out = new Uint32Array(wpr * rows);
  if (nx % 4 === 0 && m.byteOffset % 4 === 0) {
    // 4 cells (bytes of 0 / 1) per read: their low bits gathered into a nibble
    const m32 = new Uint32Array(m.buffer, m.byteOffset, m.length >>> 2), q = nx >>> 2;
    for (let r = 0; r < rows; r++) {
      const s = r * q, o = r * wpr;
      for (let w = 0; w < wpr; w++) {
        const n4 = Math.min(8, q - (w << 3));
        let bits = 0;
        for (let k = 0; k < n4; k++) {
          const v = m32[s + (w << 3) + k];
          bits |= ((v | (v >>> 7) | (v >>> 14) | (v >>> 21)) & 15) << (k << 2);
        }
        out[o + w] = invert ? ~bits & (n4 === 8 ? -1 : (1 << (n4 << 2)) - 1) : bits;
      }
    }
    return out;
  }
  for (let r = 0; r < rows; r++) {
    const s = r * nx, o = r * wpr;
    for (let w = 0; w < wpr; w++) {
      const x0 = w << 5, n = Math.min(32, nx - x0);
      let bits = 0;
      for (let b = 0; b < n; b++) if (m[s + x0 + b]) bits |= 1 << b;
      // inverted, the cells past the row's end stay 0: outside the grid counts as empty, as with bytes
      out[o + w] = invert ? ~bits & (n === 32 ? -1 : (1 << n) - 1) : bits;
    }
  }
  return out;
}

/** A nibble as 4 bytes of 0 / 1 (little-endian: the first cell is the low byte), and inverted. */
const NIB = Uint32Array.from({ length: 16 }, (_, n) => (n & 1) | ((n >> 1) & 1) << 8 | ((n >> 2) & 1) << 16 | ((n >> 3) & 1) << 24);
const NIB_INV = Uint32Array.from({ length: 16 }, (_, n) => NIB[~n & 15]);

function unpackRows(p: Uint32Array, nx: number, rows: number, out: Uint8Array, invert: boolean): void {
  const wpr = (nx + 31) >>> 5, on = invert ? 0 : 1;
  if (nx % 4 === 0 && out.byteOffset % 4 === 0 && new Uint8Array(Uint32Array.of(1).buffer)[0] === 1) {
    const o32 = new Uint32Array(out.buffer, out.byteOffset, out.length >>> 2), q = nx >>> 2, lut = invert ? NIB_INV : NIB;
    for (let r = 0; r < rows; r++) {
      const s = r * q, o = r * wpr;
      for (let k = 0; k < q; k++) o32[s + k] = lut[(p[o + (k >>> 3)] >>> ((k & 7) << 2)) & 15];
    }
    return;
  }
  for (let r = 0; r < rows; r++) {
    const s = r * nx, o = r * wpr;
    for (let x = 0; x < nx; x++) out[s + x] = (p[o + (x >>> 5)] >>> (x & 31)) & 1 ? on : 1 - on;
  }
}

/** Box dilation of a bit mask by (rx, ry, rz) cells; outside the grid is empty. */
function dilateBits(src: Uint32Array, [nx, ny, nz]: Dims, [rx, ry, rz]: R3): Uint32Array {
  const wpr = (nx + 31) >>> 5, rows = ny * nz, last = nx & 31 ? (1 << (nx & 31)) - 1 : -1;
  let cur = src;
  for (let left = rx; left > 0;) {                  // radii add up: 31 at a time
    const r = Math.min(31, left), out = new Uint32Array(cur.length);
    left -= r;
    for (let row = 0; row < rows; row++) {
      const o = row * wpr;
      for (let w = 0; w < wpr; w++) {
        const c = cur[o + w], p = w > 0 ? cur[o + w - 1] : 0, q = w < wpr - 1 ? cur[o + w + 1] : 0;
        let v = c;
        for (let s = 1; s <= r; s++) v |= (c << s) | (p >>> (32 - s)) | (c >>> s) | (q << (32 - s));
        out[o + w] = v;
      }
      out[o + wpr - 1] &= last;                     // nothing past the row's end
    }
    cur = out;
  }
  for (const [r, len, stride, blocks, block] of [[ry, ny, wpr, nz, ny * wpr], [rz, nz, ny * wpr, 1, 0]]) {
    if (r <= 0) continue;
    const out = new Uint32Array(cur.length);
    for (let b = 0; b < blocks; b++) {
      for (let i = 0; i < len; i++) {
        const o = b * block + i * stride, a = Math.max(0, i - r), e = Math.min(len - 1, i + r);
        for (let w = 0; w < stride; w++) {
          let v = 0;
          for (let k = a; k <= e; k++) v |= cur[b * block + k * stride + w];
          out[o + w] = v;
        }
      }
    }
    cur = out;
  }
  return cur;
}

/** Box dilation of a byte mask into a new array (phases.ts: the sinus tracing); outside the grid is empty. */
export function dilateMask(m: Uint8Array, d: Dims, r: R3): Uint8Array {
  const out = new Uint8Array(m.length);
  unpackRows(dilateBits(packRows(m, d[0], d[1] * d[2], false), d, r), d[0], d[1] * d[2], out, false);
  return out;
}

/** Box erosion of a byte mask into a new array; outside the grid counts as inside the mask. */
export function erodeMask(m: Uint8Array, d: Dims, r: R3): Uint8Array {
  const out = new Uint8Array(m.length);
  unpackRows(dilateBits(packRows(m, d[0], d[1] * d[2], true), d, r), d[0], d[1] * d[2], out, true);
  return out;
}

/** A new mask (from the pool); `m` stays. */
function dilate(m: Uint8Array, d: [number, number, number], r: [number, number, number]): Uint8Array {
  const out = take(m.length);
  unpackRows(dilateBits(packRows(m, d[0], d[1] * d[2], false), d, r), d[0], d[1] * d[2], out, false);
  return out;
}

/** Python's round(): halves to even (mm_to_radii in the desktop), so both apps pick the same radii. */
export function roundEven(v: number): number {
  const r = Math.round(v);
  return Math.abs(v % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/** The background dilated, inverted: outside the grid counts as inside the mask, as before. */
function erode(m: Uint8Array, d: Dims, r: R3): Uint8Array {
  const out = take(m.length);
  unpackRows(dilateBits(packRows(m, d[0], d[1] * d[2], true), d, r), d[0], d[1] * d[2], out, true);
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

/**
 * The bone area on a grid: bone closed and widened, minus the bright vessels and their wall. Every mask of it grows
 * from the cells >= hu, so it is made in their box plus more than all its dilations and erosions reach (the same cells;
 * a head CT's box leaves out the air around the head).
 */
export function boneArea(data: ArrayLike<number>, d: Dims, sp: R3, hu = 500, closeMm = 3, rimMm = 1.5, cta = true,
  plates: Uint8Array | null = null): Uint8Array {
  // a newborn's plates (bone_source 'thin') reach past the box of the cells >= hu: the whole grid
  if (plates) return boneAreaIn(data, d, sp, hu, closeMm, rimMm, cta, plates);
  const [nx, ny, nz] = d;
  const lo = [nx, ny, nz], hi = [-1, -1, -1];
  for (let z = 0, i = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++, i++) {
        if (!(data[i] >= hu)) continue;
        if (x < lo[0]) lo[0] = x;
        if (x > hi[0]) hi[0] = x;
        if (y < lo[1]) lo[1] = y;
        if (y > hi[1]) hi[1] = y;
        if (z < lo[2]) lo[2] = z;
        if (z > hi[2]) hi[2] = z;
      }
    }
  }
  const out = new Uint8Array(nx * ny * nz);
  if (hi[0] < 0) return out;
  const reach = 2 * closeMm + rimMm + 4 /* vesselsAwayFromBone reachMm */ + 3;
  const a = lo.map((v, k) => Math.max(0, v - Math.ceil(reach / sp[k]) - 2));
  const b = hi.map((v, k) => Math.min(d[k] - 1, v + Math.ceil(reach / sp[k]) + 2));
  const cd: Dims = [b[0] - a[0] + 1, b[1] - a[1] + 1, b[2] - a[2] + 1];
  if (cd[0] * cd[1] * cd[2] > 0.9 * nx * ny * nz) return boneAreaIn(data, d, sp, hu, closeMm, rimMm, cta);
  const sub = data instanceof Int16Array ? new Int16Array(cd[0] * cd[1] * cd[2]) : new Float32Array(cd[0] * cd[1] * cd[2]);
  for (let z = 0; z < cd[2]; z++) {
    for (let y = 0; y < cd[1]; y++) {
      const s = ((a[2] + z) * ny + a[1] + y) * nx + a[0], o = (z * cd[1] + y) * cd[0];
      for (let x = 0; x < cd[0]; x++) sub[o + x] = data[s + x];
    }
  }
  const m = boneAreaIn(sub, cd, sp, hu, closeMm, rimMm, cta);
  for (let z = 0; z < cd[2]; z++) {
    for (let y = 0; y < cd[1]; y++) {
      out.set(m.subarray((z * cd[1] + y) * cd[0], (z * cd[1] + y + 1) * cd[0]), ((a[2] + z) * ny + a[1] + y) * nx + a[0]);
    }
  }
  return out;
}

function boneAreaIn(data: ArrayLike<number>, d: Dims, sp: R3, hu: number, closeMm: number, rimMm: number, cta: boolean,
  plates: Uint8Array | null = null): Uint8Array {
  let bone: Uint8Array, vessels: Uint8Array | null = null;
  if (cta) {
    ({ bone, vessels } = boneAndVessels(data, d, sp, hu));
  } else {
    // any other CT: all from the threshold (bone_source 'ct'); a newborn's: and its skull plates ('thin')
    bone = new Uint8Array(d[0] * d[1] * d[2]);
    for (let i = 0; i < bone.length; i++) if (data[i] >= hu || (plates && plates[i] && data[i] >= THIN_BONE_HU)) bone[i] = 1;
  }
  const g = gridOf(sp);
  const rc = g.radii(closeMm);
  const wide = dilate(bone, d, rc);
  const closed = erode(wide, d, rc);
  give(wide);
  for (let i = 0; i < closed.length; i++) closed[i] |= bone[i];
  const out = dilate(closed, d, g.radii(rimMm));
  give(closed);
  if (vessels) {
    const wall = dilate(vessels, d, g.atLeast1(1));
    for (let i = 0; i < out.length; i++) if (wall[i] && !bone[i]) out[i] = 0;
    give(wall);
  }
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
  return voxels > (mobile ? 20e6 : 30e6) ? 2 : 1;
}

/** The values on the mask grid: the volume itself, or the brightest voxel of each cell. */
function gridValues(vol: Volume, f: number): ArrayLike<number> {
  return gridValuesOf(vol.data, vol.cols, vol.rows, vol.n, f);
}

/** gridValues of any values on a volume's grid (cols × rows × n): the plain scan's. */
function gridValuesOf(d: ArrayLike<number>, cols: number, rows: number, n: number, f: number): ArrayLike<number> {
  if (f === 1) return d;
  const nx = Math.ceil(cols / f), ny = Math.ceil(rows / f), nz = n;
  const out = d instanceof Int16Array ? new Int16Array(nx * ny * nz).fill(-32768) : new Float32Array(nx * ny * nz).fill(-Infinity);
  if (f === 2) {
    // one pass per cell: the brightest of its 2 × 2 (a quarter of the loop, no division per voxel)
    const W = cols, H = rows;
    for (let z = 0; z < nz; z++) {
      for (let j = 0; j < ny; j++) {
        const ra = (z * H + 2 * j) * W, rb = 2 * j + 1 < H ? ra + W : ra, o = (z * ny + j) * nx;
        for (let i = 0; i < nx; i++) {
          const xa = 2 * i, xb = xa + 1 < W ? xa + 1 : xa;
          let m = d[ra + xa];
          if (d[ra + xb] > m) m = d[ra + xb];
          if (d[rb + xa] > m) m = d[rb + xa];
          if (d[rb + xb] > m) m = d[rb + xb];
          out[o + i] = m;
        }
      }
    }
    return out;
  }
  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < rows; y++) {
      const row = (z * rows + y) * cols, o = (z * ny + Math.floor(y / f)) * nx;
      for (let x = 0; x < cols; x++) {
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
/** The bone area of a CT volume (see the module comment), from `source` (plain: `plain` = the plain scan's values on
 *  this volume's grid, venography.ts plainFor). `f` = in-plane step of the grid; by default from the size. */
export function boneRegion(vol: Volume, hu = 500, closeMm = 3, rimMm = 1.5, f: 1 | 2 = gridStep(vol.rows * vol.cols * vol.n, false),
  source: BoneSource = "cta", plain: ArrayLike<number> | null = null, plates: Uint8Array | null = null, rule: BoneSource = "cta",
  whole = true): BoneRegion {
  const nx = Math.ceil(vol.cols / f), ny = Math.ceil(vol.rows / f), nz = vol.n;
  const d: Dims = [nx, ny, nz], sp = gridSpacing(vol, f);
  if (source === "plain" && plain) {
    const r = plainBoneSlabs(gridValues(vol, f), gridValuesOf(plain, vol.cols, vol.rows, vol.n, f), d, sp, hu, rule, whole,
      rule === "thin" ? plates : null);
    return { bits: r.bits, enhanced: r.enhanced, plainPlanes: r.plainPlanes, nx, ny, nz, f };
  }
  return { bits: boneBits(gridValues(vol, f), d, sp, hu, closeMm, rimMm, source === "cta", source === "thin" ? plates : null), nx, ny, nz, f };
}

/** What boneRegion works on: the values of the mask grid, its size and spacing, and the plain scan's values on it (copies,
 *  so they can go to a worker). */
export function boneGrid(vol: Volume, f: 1 | 2, plain: ArrayLike<number> | null = null): {
  values: Int16Array | Float32Array; plain: Int16Array | Float32Array | null; dims: Dims; spacing: R3;
} {
  const own = (a: ArrayLike<number>, of: ArrayLike<number>) => (a === of ? (a as Int16Array | Float32Array).slice() : a as Int16Array | Float32Array);
  const values = own(gridValues(vol, f), vol.data);
  const p = plain ? own(gridValuesOf(plain, vol.cols, vol.rows, vol.n, f), plain) : null;
  return { values, plain: p, dims: [Math.ceil(vol.cols / f), Math.ceil(vol.rows / f), vol.n], spacing: gridSpacing(vol, f) };
}

/** The plain scan's values (on vol's grid) on the bone grid of step f, its own copy. */
export function plainGridOf(vol: Volume, f: 1 | 2, plain: ArrayLike<number>): Int16Array | Float32Array {
  const g = gridValuesOf(plain, vol.cols, vol.rows, vol.n, f);
  return g === plain ? (plain as Int16Array | Float32Array).slice() : g as Int16Array | Float32Array;
}

/** The bone area of a grid, 1 bit per cell (bone.worker.ts runs it off the page's thread). cta = less the bright
 *  vessels that pass hu (bone_source 'cta'); else all from hu ('ct'). */
export function boneBits(values: ArrayLike<number>, d: Dims, sp: R3, hu = 500, closeMm = 3, rimMm = 1.5, cta = true,
  plates: Uint8Array | null = null): Uint32Array {
  const bits = pack(boneArea(values, d, sp, hu, closeMm, rimMm, cta, plates));
  pool.length = 0;
  return bits;
}

/**
 * The bone area from the plain scan (bone_source 'plain', dcv/segment.py _plain_area): what was bone before contrast
 * (PLAIN_BONE_HU within PLAIN_TOL_MM) and did not get brighter, cell by cell, so holes stay holes; and the cells that got
 * brighter (enhanced: ENHANCE_HU over the brightest plain cell within PLAIN_TOL_MM, that below LUMEN_PLAIN_MAX_HU, with
 * ENHANCE_NEIGHBOURS around). No HU tells a weak contrast vessel (real infant '+ CM' CT: 150-220 HU) from young bone.
 */
export function plainBone(values: ArrayLike<number>, plain: ArrayLike<number>, d: Dims, sp: R3): { bits: Uint32Array; enhanced: Uint32Array } {
  const before = boxMax(plain, d, gridOf(sp).atLeast1(PLAIN_TOL_MM));
  const [nx, ny, nz] = d, nxy = nx * ny, N = nxy * nz;
  const enh = new Uint8Array(N);
  let any = false;
  for (let i = 0; i < N; i++) {
    if (values[i] >= before[i] + ENHANCE_HU && before[i] < LUMEN_PLAIN_MAX_HU) {
      enh[i] = 1;
      any = true;
    }
  }
  // an enhanced cell with too few enhanced around it is noise: counted only at the enhanced cells (a few % of a head)
  const kept = new Uint8Array(N);
  if (any) {
    for (let z = 0, i = 0; z < nz; z++) {
      const z0 = z > 0 ? -1 : 0, z1 = z < nz - 1 ? 1 : 0;
      for (let y = 0; y < ny; y++) {
        const y0 = y > 0 ? -1 : 0, y1 = y < ny - 1 ? 1 : 0;
        for (let x = 0; x < nx; x++, i++) {
          if (!enh[i]) continue;
          const x0 = x > 0 ? -1 : 0, x1 = x < nx - 1 ? 1 : 0;
          let c = 0;
          for (let k = z0; k <= z1; k++) for (let j = y0; j <= y1; j++) {
            const row = i + k * nxy + j * nx;
            for (let h = x0; h <= x1; h++) c += enh[row + h];
          }
          if (c >= ENHANCE_NEIGHBOURS) kept[i] = 1;
        }
      }
    }
  }
  // a lumen against bone: within PLAIN_TOL_MM the plain scan holds the bone beside it, so the test above calls it no
  // lumen; where the plain scan at the cell itself is blood and the series far brighter, joined to a lumen, it is one
  // (else the neck of an aneurysm on the clinoid and its ICA came out as bone). Joined: a bone edge the scans moved
  // apart is no lumen (rings on the skull)
  if (any) {
    const near = dilateMask(kept, d, gridOf(sp).atLeast1(LUMEN_EDGE_MM));
    for (let i = 0; i < N; i++) {
      if (near[i] && !kept[i] && values[i] >= plain[i] + LUMEN_EDGE_HU && plain[i] < LUMEN_PLAIN_MAX_HU && values[i] < BONE_CORE_HU) kept[i] = 1;
    }
  }
  const area = new Uint8Array(N);
  for (let i = 0; i < N; i++) if (before[i] >= PLAIN_BONE_HU && !kept[i]) area[i] = 1;
  return { bits: pack(area), enhanced: pack(kept) };
}

/** A mask of another grid (cells of `f` voxels in-plane, `fz` planes; nx × ny × nz) on the bone area's grid `d` (cells of
 *  `g` voxels in-plane, one plane each): the cell under each one's first voxel. */
export function maskOnGrid(m: Uint8Array, [nx, ny, nz]: Dims, f: number, fz: number, d: Dims, g: number): Uint8Array {
  const out = new Uint8Array(d[0] * d[1] * d[2]);
  for (let z = 0, i = 0; z < d[2]; z++) {
    const Z = Math.min(nz - 1, Math.floor(z / fz)) * nx * ny;
    for (let y = 0; y < d[1]; y++) {
      const Y = Z + Math.min(ny - 1, Math.floor((y * g) / f)) * nx;
      for (let x = 0; x < d[0]; x++, i++) out[i] = m[Y + Math.min(nx - 1, Math.floor((x * g) / f))];
    }
  }
  return out;
}

/**
 * The bone area from the plain scan where it lines up (dcv/segment.py bone_region 'plain'): as a whole (`whole`: it
 * lined up as it is), else slab by slab (slabPlain); the planes it does not line up with keep the rule of the series
 * without it (`rule`, `plates` for a newborn), with every cell there counted as enhanced (no plain scan to say otherwise).
 * plainPlanes = 0: none lined up, the rule's area alone.
 */
export function plainBoneSlabs(values: ArrayLike<number>, plain: ArrayLike<number>, d: Dims, sp: R3, hu: number, rule: BoneSource,
  whole: boolean, plates: Uint8Array | null): { bits: Uint32Array; enhanced?: Uint32Array; plainPlanes: number } {
  const [nx, ny, nz] = d, nxy = nx * ny, N = nxy * nz;
  let moved: ArrayLike<number> = plain, ok: Uint8Array;
  if (whole) ok = new Uint8Array(nz).fill(1);
  else ({ plain: moved, ok } = slabPlain(values, plain, d, sp, gridOf(sp).atLeast1(PLAIN_TOL_MM)));
  let planes = 0;
  for (let z = 0; z < nz; z++) planes += ok[z];
  if (!planes) return { bits: boneBits(values, d, sp, hu, 3, 1.5, rule === "cta", plates), plainPlanes: 0 };
  const r = plainBone(values, moved, d, sp);
  if (planes === nz) return { ...r, plainPlanes: nz };
  const area = unpack(r.bits, N), enh = unpack(r.enhanced, N);
  const ruleArea = boneArea(values, d, sp, hu, 3, 1.5, rule === "cta", plates);
  pool.length = 0;
  for (let z = 0; z < nz; z++) {
    if (ok[z]) continue;
    area.set(ruleArea.subarray(z * nxy, (z + 1) * nxy), z * nxy);
    enh.fill(1, z * nxy, (z + 1) * nxy);
  }
  return { bits: pack(area), enhanced: pack(enh), plainPlanes: planes };
}

/**
 * The plain scan (on the grid of `values`) lined up slab by slab (dcv/venography.py slab_plain): SLAB_MM along z, each
 * slab moved by up to SLAB_SHIFT_MM where it does not line up as it is, and which planes then line up: the share of the
 * series' bone (>= 300 HU, every k-th of the slab's cells) with plain bone (>= 200 HU) within `tol` cells reaches
 * SLAB_MATCH. A CTA from the chest to the vertex: the head lies still while the jaw and neck bend between the scans.
 * Returns the plain values moved per slab (-1024 where no slab lines up) and 1 per plane that lines up.
 */
export function slabPlain(values: ArrayLike<number>, plain: ArrayLike<number>, [nx, ny, nz]: Dims, sp: R3, tol: R3):
  { plain: Int16Array | Float32Array; ok: Uint8Array } {
  const near = boxMax(plain, [nx, ny, nz], tol);
  const nxy = nx * ny, N = nxy * nz;
  const out = plain instanceof Int16Array ? new Int16Array(N).fill(-1024) : new Float32Array(N).fill(-1024);
  const ok = new Uint8Array(nz);
  const n = Math.max(1, Math.round(SLAB_MM / sp[2]));
  const reach = [Math.round(SLAB_SHIFT_MM / sp[0]), Math.round(SLAB_SHIFT_MM / sp[1]), Math.round(SLAB_SHIFT_MM / sp[2])];
  for (let z0 = 0; z0 < nz; z0 += n) {
    const z1 = Math.min(nz, z0 + n);
    const all: number[] = [];
    for (let i = z0 * nxy; i < z1 * nxy; i++) if (values[i] >= 300) all.push(i);
    if (all.length < 200) continue;
    const step = Math.max(1, Math.floor(all.length / SLAB_POINTS));
    const px: number[] = [], py: number[] = [], pz: number[] = [];
    for (let k = 0; k < all.length; k += step) {
      const i = all[k];
      px.push(i % nx);
      py.push(Math.floor(i / nx) % ny);
      pz.push(Math.floor(i / nxy));
    }
    const share = (ox: number, oy: number, oz: number): number => {
      let hit = 0, inside = 0;
      for (let k = 0; k < px.length; k++) {
        const x = px[k] + ox, y = py[k] + oy, z = pz[k] + oz;
        if (x < 0 || y < 0 || z < 0 || x >= nx || y >= ny || z >= nz) continue;
        inside++;
        if (near[(z * ny + y) * nx + x] >= 200) hit++;
      }
      return inside ? hit / inside : 0;
    };
    let off = [0, 0, 0], best = share(0, 0, 0);
    // a slab that lines up as it is stays so: shifted alone it left a step in the bone between two slabs
    if (best < SLAB_MATCH) {
      for (const [st, rr] of [[2, reach], [1, [2, 2, 2]]] as [number, number[]][]) {   // every 2nd cell, then near the best
        const c = off.slice();
        for (let dz = -rr[2]; dz <= rr[2]; dz += st) for (let dy = -rr[1]; dy <= rr[1]; dy += st) for (let dx = -rr[0]; dx <= rr[0]; dx += st) {
          const o = [c[0] + dx, c[1] + dy, c[2] + dz];
          if (Math.abs(o[0]) > reach[0] || Math.abs(o[1]) > reach[1] || Math.abs(o[2]) > reach[2]) continue;  // within reach
          const s = share(o[0], o[1], o[2]), moved = Math.abs(o[0]) + Math.abs(o[1]) + Math.abs(o[2]);
          if (s > best || (s === best && moved < Math.abs(off[0]) + Math.abs(off[1]) + Math.abs(off[2]))) [best, off] = [s, o];   // a tie: the smaller move
        }
      }
    }
    if (best < SLAB_MATCH) continue;
    ok.fill(1, z0, z1);
    const [ox, oy, oz] = off;
    for (let z = z0; z < z1; z++) {
      const Z = z + oz;
      if (Z < 0 || Z >= nz) continue;
      for (let y = 0; y < ny; y++) {
        const Y = y + oy;
        if (Y < 0 || Y >= ny) continue;
        const row = (z * ny + y) * nx, src = (Z * ny + Y) * nx;
        for (let x = Math.max(0, -ox); x < Math.min(nx, nx - ox); x++) out[row + x] = plain[src + x + ox];
      }
    }
  }
  return { plain: out, ok };
}

/** Maximum over (2 rx + 1) × (2 ry + 1) × (2 rz + 1) cells: per axis, one cell per pass (dcv/segment.py box_max). */
export function boxMax(a: ArrayLike<number>, [nx, ny, nz]: Dims, r: R3): Int16Array | Float32Array {
  let out = a instanceof Int16Array ? a.slice() : Float32Array.from(a);
  let src = out instanceof Int16Array ? new Int16Array(out.length) : new Float32Array(out.length);
  for (let axis = 0; axis < 3; axis++) {
    const s = axis === 0 ? 1 : axis === 1 ? nx : nx * ny, n = axis === 0 ? nx : axis === 1 ? ny : nz;
    for (let pass = 0; pass < r[axis]; pass++) {
      [src, out] = [out, src];
      maxPass(src, out, s, n);
    }
  }
  return out;
}

/**
 * out = the maximum of each cell and its two neighbours along one axis. The volume is blocks of n lines of s cells (x: s
 * 1, y: s nx, z: s nx·ny), so the first and last line of a block lack a neighbour and the rest have both: no test per
 * cell (with one, this filter took twice as long as the whole bone area).
 */
function maxPass(src: Int16Array | Float32Array, out: Int16Array | Float32Array, s: number, n: number): void {
  const N = src.length, block = s * n;
  if (n === 1) {
    out.set(src);
    return;
  }
  for (let b = 0; b < N; b += block) {
    const e1 = b + s, e2 = b + block - s, end = b + block;
    for (let i = b; i < e1; i++) {
      const u = src[i], w = src[i + s];
      out[i] = u > w ? u : w;
    }
    for (let i = e1; i < e2; i++) {
      let m = src[i];
      const u = src[i - s], w = src[i + s];
      if (u > m) m = u;
      if (w > m) m = w;
      out[i] = m;
    }
    for (let i = e2; i < end; i++) {
      const u = src[i - s], w = src[i];
      out[i] = u > w ? u : w;
    }
  }
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
  /** Both phases of a CTA (phases.ts): each vessel with the HU of the series that shows it. */
  artData?: ArrayLike<number>;
  venData?: ArrayLike<number>;
  /** Where the other series has this voxel, when the patient moved between the scans. */
  artOffset?: number;
  venOffset?: number;
  shiftMm?: number;
  /** One series: what the sinus tracing found, ml (phases.ts sinusMasks). */
  sinusMl?: number;
  /** Both phases: from this HU on, a voxel that is no artery is bone, even outside the bone area. */
  boneHu?: number;
  /** The head's vessels were labelled by the vessel atlas (atlas.ts). */
  atlas?: boolean;
  /** One routine contrast series: its vessels' range (sinusMasks), for curves fitted to it; floor also lowers what
   *  is hidden (encodeVessel). */
  fit?: { floor: number; peak: number };
  /** Both phases: the cells where the bone area meets a vessel's cells, decided voxel by voxel (encodeCT). */
  edge?: BoneRegion;
}

/** As phases.ts: contrast changes a vessel by at least this between the phases, bone not; a lumen is at least this. */
const PHASE_DIFF_HU = 60, LUMEN_MIN_HU = 150;

/**
 * Value in the 3D copy of voxel `i` of the volume `d` (rows of `nx`, slices of `nxy`; v = its value, already floored),
 * the one rule of renderer.fill and pick. `boneOn` = bone lock on; inArea, isArtery, isVein, onEdge = its cell in each
 * mask.
 *
 * With both phases the masks are cells of ~1 mm, and alone they drew the bone area's edge in steps: a sinus or an artery
 * against the skull was cut off in terraces, the bone beside it in a vessel's cell came out in blocks of vessel colour,
 * and where a cell's face ran along the bone, the bone surface was lit by the jump of the coding (real CTA, zoomed in:
 * 'pixelated'). So the voxel decides, from its own two phases and whether bone is within 2 voxels (the shading's
 * gradient reaches one voxel each way): the bone's edge outside the area (near bone, unchanged by the contrast) is bone;
 * a lumen in a bone-area cell beside a vessel (onEdge: changed, clear of the bone) is a vessel, artery when brighter on
 * the arterial series. A single voxel's difference is noisy, so only these clear cases move, or the edge came out
 * speckled. Only voxels between 110 HU and the bone threshold are looked at: a few % of a head.
 */
export function encodeCT(m: VesselMasks | null, d: ArrayLike<number>, nx: number, nxy: number, i: number, v: number, boneOn: boolean,
  inArea: boolean, isArtery: boolean, isVein: boolean, onEdge: boolean): number {
  if (boneOn && m && byVoxel(m, v, inArea, isArtery, onEdge)) {
    const r = voxelRule(m, d, nx, nxy, i, v, inArea, isArtery, isVein, onEdge);
    if (r === IN_AREA) inArea = true;
    else if (r !== KEEP) {
      inArea = false;
      isArtery = r === ARTERY;
      isVein = r === VEIN;
    }
  }
  if (inArea) return -BONE_OFFSET - v;
  return m ? encodeWith(m, v, isArtery, isVein, i, boneOn) : v;
}

export const KEEP = 0, IN_AREA = 1, ARTERY = 2, VEIN = 3;

/** Whether encodeCT looks at the voxel itself (bone lock on): with both phases, a voxel the cells may have got wrong.
 *  Cheap, for the loop of renderer.fill: most voxels are below 110 HU. */
export function byVoxel(m: VesselMasks, v: number, inArea: boolean, isArtery: boolean, onEdge: boolean): boolean {
  if (v < VEIN_MIN_HU || !m.artData) return false;
  return v >= (m.boneHu ?? 500) ? !inArea && isArtery : onEdge || !inArea;
}

/** encodeCT's rule for one voxel (byVoxel true): KEEP the cells' answer, or IN_AREA, ARTERY, VEIN. */
export function voxelRule(m: VesselMasks, d: ArrayLike<number>, nx: number, nxy: number, i: number, v: number, inArea: boolean,
  isArtery: boolean, isVein: boolean, onEdge: boolean): number {
  const top = m.boneHu ?? 500;
  // the other series only when it decides: it is another large array, read out of step with this one
  const changed = () => {
    const a = m.artData![i + (m.artOffset ?? 0)] ?? v, w = m.venData![i + (m.venOffset ?? 0)] ?? v;
    return Math.abs(w - a) >= PHASE_DIFF_HU ? (a > w ? ARTERY : VEIN) : KEEP;
  };
  if (v >= top) {
    // bone in an artery's cell: dense, or unchanged by the contrast (an artery above the threshold changes, or is no
    // denser than contrast gets); it came out as artery blocks along the clivus
    return v >= BONE_CORE_HU || changed() === KEEP ? IN_AREA : KEEP;
  }
  // a lumen: contrast changed it and it is not the bone's own partial-volume edge. Right at the bone a sub-voxel move
  // between the scans changes the edge by hundreds of HU, so there the difference says nothing
  const n = d.length;
  const touch1 = hot(d, n, i, 1, top) || hot(d, n, i, nx, top) || hot(d, n, i, nxy, top);
  const touch = touch1 || hot(d, n, i, 2, top) || hot(d, n, i, 2 * nx, top) || hot(d, n, i, 2 * nxy, top);
  if (inArea) {
    if (!onEdge || touch || v < LUMEN_MIN_HU) return KEEP;
    return changed();
  }
  if (!touch) return KEEP;
  if (touch1 || v < LUMEN_MIN_HU || !(isArtery || isVein)) return IN_AREA;
  return changed() === KEEP ? IN_AREA : KEEP;
}

/** Bone (>= top) k voxels before or after voxel i of d (length n). Bounds checked: reading past a typed array is slow. */
function hot(d: ArrayLike<number>, n: number, i: number, k: number, top: number): boolean {
  return (i >= k && d[i - k] >= top) || (i + k < n && d[i + k] >= top);
}

/** Value in the 3D copy with vessel masks; `i` = index of the voxel in the volume; `boneOn` = bone lock on. */
export function encodeWith(m: VesselMasks, v: number, isArtery: boolean, isVein: boolean, i: number, boneOn = false): number {
  if (!m.artData || !m.venData) return encodeVessel(v, isArtery, isVein, m.fit ? Math.min(m.fit.floor, VEIN_MIN_HU) : VEIN_MIN_HU);
  // bone outside the bone area (a cell given to a vessel, bone that moved between the phases) is still bone: clamped into
  // the vessel range it was hidden, which left holes in the teeth (real CTA)
  if (boneOn && !isArtery && m.boneHu !== undefined && v >= m.boneHu) return -BONE_OFFSET - v;
  // `?? v`: a shift can point outside the volume at its very edge
  if (isArtery) return Math.min(m.artData[i + (m.artOffset ?? 0)] ?? v, ARTERY_TOP_HU);
  const w = m.venData[i + (m.venOffset ?? 0)] ?? v;
  // only the vein's own voxels: brain in a vein's cell stored as VEIN_OFFSET + 40 met its neighbours at 40 HU across the
  // whole range, and every face of the cell drew a thin red and blue sheet (a wire grid)
  if (isVein && w >= VEIN_MIN_HU) return VEIN_OFFSET + Math.min(w, 3000);
  return v >= VEIN_MIN_HU ? VEIN_MIN_HU - 1 : v;
}

/** Arteries and veins of a CT, on the same grid as its bone area (`region`; null = bone lock off: step `f`). */
export function vesselMasks(vol: Volume, region: BoneRegion | null, coreOverride: number | null = null, f: 1 | 2 = 1, venousOnly = false): VesselMasks {
  const step = (region?.f ?? f) as 1 | 2;
  const nx = Math.ceil(vol.cols / step), ny = Math.ceil(vol.rows / step), nz = vol.n;
  const d: Dims = [nx, ny, nz];
  const data = gridValues(vol, step);
  const sp = gridSpacing(vol, step);
  const area = region ? unpack(region.bits, nx * ny * nz) : null;
  const free = new Uint8Array(nx * ny * nz);
  for (let i = 0; i < free.length; i++) free[i] = area && area[i] ? 0 : 1;
  const coreHu = coreOverride ?? arteryThreshold(data, free);
  // one venous series: arteries have washed out, every bright vessel is shown as a vein
  const arteries = venousOnly ? new Uint8Array(free.length) : classifyArteries(data, d, sp, free, coreHu);
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
export function encodeVessel(v: number, isArtery: boolean, isVein: boolean, floor = VEIN_MIN_HU): number {
  if (isVein && v >= floor) return VEIN_OFFSET + Math.min(v, 3000);       // the vein's own voxels (encodeWith)
  if (v >= floor && !isArtery) return floor - 1;
  return v > ARTERY_TOP_HU ? ARTERY_TOP_HU : v;
}

/** The value under every vessel curve that a voxel the plain scan shows unchanged is stored with (BoneRegion.enhanced),
 *  for the vessel masks `m` (a routine contrast CT's fitted curves start lower). */
export function notVesselHu(m: VesselMasks | null | undefined): number {
  return Math.min(NOT_VESSEL_HU, (m?.fit ? Math.min(m.fit.floor, VEIN_MIN_HU) : VEIN_MIN_HU) - 1);
}

/** The bit of a mask for a voxel of the full-resolution grid. */
export function maskAt(r: BoneRegion, x: number, y: number, z: number): boolean {
  const i = (z * r.ny + Math.floor(y / r.f)) * r.nx + Math.floor(x / r.f);
  return (r.bits[i >>> 5] & (1 << (i & 31))) !== 0;
}
