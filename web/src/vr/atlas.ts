/**
 * Where arteries and where veins run in the head (dcv/atlas.py; its comment has the measurements): an atlas built from
 * multiphase CTAs (tools/build_vessel_atlas.py), matched to a new scan by its skull (affine, Gauss-Newton on the atlas's
 * distance map), then this patient's own HU learnt from the places the atlas is sure of, and the evidence pooled along
 * each vessel. Coordinates here are mm in (z, y, x) order, as in the atlas file; mask grids are indexed
 * (z * ny + y) * nx + x with spacing [sx, sy, sz], as in phases.ts.
 */

type D3 = [number, number, number];

const HEAD_MM = 120, ATLAS_DEPTH_MM = 150, MIN_FIT = 0.4, SURE = 0.85, LLR_CLIP = 4, SUPPORT_MIN = 0.001;
/** Head pitches (degrees) the registration starts from: a routine head CT is often tilted ~35 degrees from a CTA. */
const PITCHES = [0, -20, 20, -40, 40], PITCH_MARGIN = 0.02;   // a tilted start must fit clearly better than upright

export interface VesselAtlas {
  shape: D3;              // z, y, x
  sp: D3;                 // mm, z y x
  origin: D3;             // mm of cell 0
  centre: D3;             // the skull's centroid, atlas mm
  dist: Float32Array;     // to the atlas skull surface, mm
  prob: Float32Array;     // share of arteries among the vessels here
  support: Float32Array;  // how many vessels were here (per study)
}

let shipped: VesselAtlas | null = null;

/** The atlas the classifiers use (null: none, they fall back to the rules). */
export function setVesselAtlas(a: VesselAtlas | null): void {
  shipped = a;
}

export function vesselAtlas(): VesselAtlas | null {
  return shipped;
}

/** Parse vessel_atlas.bin: "SEVA", header length, JSON header, then dist / prob / support as uint8. */
export function parseAtlas(buf: ArrayBuffer): VesselAtlas {
  const u8 = new Uint8Array(buf);
  if (String.fromCharCode(u8[0], u8[1], u8[2], u8[3]) !== "SEVA") throw new Error("not a vessel atlas");
  const n = new DataView(buf).getUint32(4, true);
  const head = JSON.parse(new TextDecoder().decode(u8.subarray(8, 8 + n)));
  const shape = head.shape as D3, size = shape[0] * shape[1] * shape[2];
  const part = (k: number, scale: number) => {
    const out = new Float32Array(size), src = u8.subarray(8 + n + k * size, 8 + n + (k + 1) * size);
    for (let i = 0; i < size; i++) out[i] = src[i] * scale;
    return out;
  };
  return {
    shape, sp: head.sp, origin: head.origin, centre: head.centre,
    dist: part(0, 1 / 8), prob: part(1, 1 / 255), support: part(2, head.support_scale / 255),
  };
}

/** Load ./vessel_atlas.bin once (same origin: the app's own file, cached offline by the service worker). */
export async function loadVesselAtlas(url = "./vessel_atlas.bin"): Promise<VesselAtlas | null> {
  try {
    const r = await fetch(url);
    if (!r.ok) return null;
    shipped = parseAtlas(await r.arrayBuffer());
  } catch {
    shipped = null;
  }
  return shipped;
}

/** Trilinear value of an atlas grid at a mm point (z y x), and its gradient per mm when `g` is given. */
function sampleAt(a: VesselAtlas, vol: Float32Array, z: number, y: number, x: number, g?: number[]): number {
  const [nz, ny, nx] = a.shape;
  const clamp = (v: number, hi: number) => Math.min(Math.max(v, 0), hi - 1e-6);
  const gz = clamp((z - a.origin[0]) / a.sp[0], nz - 1), gy = clamp((y - a.origin[1]) / a.sp[1], ny - 1);
  const gx = clamp((x - a.origin[2]) / a.sp[2], nx - 1);
  const z0 = Math.floor(gz), y0 = Math.floor(gy), x0 = Math.floor(gx);
  const fz = gz - z0, fy = gy - y0, fx = gx - x0;
  const z1 = Math.min(z0 + 1, nz - 1), y1 = Math.min(y0 + 1, ny - 1), x1 = Math.min(x0 + 1, nx - 1);
  const at = (k: number, j: number, i: number) => vol[(k * ny + j) * nx + i];
  const c000 = at(z0, y0, x0), c001 = at(z0, y0, x1), c010 = at(z0, y1, x0), c011 = at(z0, y1, x1);
  const c100 = at(z1, y0, x0), c101 = at(z1, y0, x1), c110 = at(z1, y1, x0), c111 = at(z1, y1, x1);
  const c00 = c000 + (c001 - c000) * fx, c01 = c010 + (c011 - c010) * fx;
  const c10 = c100 + (c101 - c100) * fx, c11 = c110 + (c111 - c110) * fx;
  const c0 = c00 + (c01 - c00) * fy, c1 = c10 + (c11 - c10) * fy;
  if (g) {
    g[0] = (c1 - c0) / a.sp[0];
    g[1] = ((1 - fz) * (c01 - c00) + fz * (c11 - c10)) / a.sp[1];
    const d0 = (1 - fy) * (c001 - c000) + fy * (c011 - c010), d1 = (1 - fy) * (c101 - c100) + fy * (c111 - c110);
    g[2] = ((1 - fz) * d0 + fz * d1) / a.sp[2];
  }
  return c0 + (c1 - c0) * fz;
}

function topPlane(bone: Uint8Array, [nx, ny, nz]: D3): number {
  for (let z = nz - 1; z >= 0; z--) {
    let c = 0;
    for (let i = z * nx * ny; i < (z + 1) * nx * ny; i++) c += bone[i];
    if (c >= 20) return z;
  }
  return nz - 1;
}

/** Up to n skull surface points (mm, z y x) HEAD_MM under the vertex and up, chosen by a fixed pseudo-random sequence. */
function headPoints(bone: Uint8Array, d: D3, sp: number[], n = 8000): number[][] {
  const [nx, ny, nz] = d, nxy = nx * ny;
  const z0 = Math.max(0, topPlane(bone, d) - Math.floor(HEAD_MM / sp[2]));
  const all: number[] = [];
  for (let z = z0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
    const i = z * nxy + y * nx + x;
    if (!bone[i]) continue;
    const edge = x === 0 || x === nx - 1 || y === 0 || y === ny - 1 || z === 0 || z === nz - 1 ||
      !bone[i - 1] || !bone[i + 1] || !bone[i - nx] || !bone[i + nx] || !bone[i - nxy] || !bone[i + nxy];
    if (edge) all.push(i);
  }
  let seed = 12345;
  const rnd = () => (seed = (seed * 1103515245 + 12345) >>> 0) / 4294967296;
  for (let i = 0; i < Math.min(n, all.length); i++) {               // partial shuffle: the first n are a sample
    const j = i + Math.floor(rnd() * (all.length - i));
    [all[i], all[j]] = [all[j], all[i]];
  }
  const pick = all.slice(0, n);
  return pick.map((i) => [Math.floor(i / nxy) * sp[2], (Math.floor(i / nx) % ny) * sp[1], (i % nx) * sp[0]]);
}

/** Solve H x = b for a small dense symmetric system (Gaussian elimination with partial pivoting). */
function solve(H: number[][], b: number[]): number[] {
  const n = b.length, A = H.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    [A[c], A[p]] = [A[p], A[c]];
    for (let r = c + 1; r < n; r++) {
      const f = A[r][c] / A[c][c];
      for (let k = c; k <= n; k++) A[r][k] -= f * A[c][k];
    }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = A[r][n];
    for (let k = r + 1; k < n; k++) s -= A[r][k] * x[k];
    x[r] = s / A[r][r];
  }
  return x;
}

export interface AtlasFit { M: number[][]; c: number[]; t: number[]; fit: number }

/** Affine taking the scan's skull onto the atlas skull (scan mm -> atlas mm: q = M (p - c) + t), and the share of
 *  skull points then within 2 mm of it. Null with too little skull. */
export function registerSkull(a: VesselAtlas, bone: Uint8Array, d: D3, sp: number[], iters = 60): AtlasFit | null {
  const pts = headPoints(bone, d, sp);
  if (pts.length < 500) return null;
  const c = [0, 1, 2].map((k) => pts.reduce((s, p) => s + p[k], 0) / pts.length);
  const p = pts.map((q) => [q[0] - c[0], q[1] - c[1], q[2] - c[2]]);
  let best: { f: number; M: number[][]; t: number[] } | null = null;
  for (const deg of PITCHES) {
    const r = (deg * Math.PI) / 180;
    const s = descend(a, p, [[Math.cos(r), -Math.sin(r), 0], [Math.sin(r), Math.cos(r), 0], [0, 0, 1]], [...a.centre], 20);
    const f = fitOf(a, p, s.M, s.t);
    if (!best || f > best.f + PITCH_MARGIN) best = { f, ...s };
  }
  const { M, t } = descend(a, p, best!.M, best!.t, iters);
  return { M, c, t, fit: fitOf(a, p, M, t) };
}

const applyTo = (M: number[][], t: number[], q: number[]) =>
  [0, 1, 2].map((i) => M[i][0] * q[0] + M[i][1] * q[1] + M[i][2] * q[2] + t[i]);

function fitOf(a: VesselAtlas, p: number[][], M: number[][], t: number[]): number {
  let near = 0;
  for (const q of p) {
    const [z, y, x] = applyTo(M, t, q);
    if (sampleAt(a, a.dist, z, y, x) < 2) near++;
  }
  return near / p.length;
}

/** Gauss-Newton on the atlas's distance map, Huber weights shrinking from 10 to 3 mm. */
function descend(a: VesselAtlas, p: number[][], M0: number[][], t0: number[], iters: number): { M: number[][]; t: number[] } {
  let M = M0, t = t0;
  const g = [0, 0, 0];
  const apply = (q: number[]) => applyTo(M, t, q);
  for (let it = 0; it < iters; it++) {
    const k = Math.max(3, 10 * (1 - it / (0.6 * iters)));
    const H = Array.from({ length: 12 }, () => new Array(12).fill(0)), b = new Array(12).fill(0);
    const J = new Array(12);
    for (const q of p) {
      const [z, y, x] = apply(q);
      const r = sampleAt(a, a.dist, z, y, x, g);
      const w = r <= k ? 1 : k / Math.max(r, 1e-6);
      for (let i = 0; i < 3; i++) {
        J[i * 4] = g[i] * q[0];
        J[i * 4 + 1] = g[i] * q[1];
        J[i * 4 + 2] = g[i] * q[2];
        J[i * 4 + 3] = g[i];
      }
      for (let i = 0; i < 12; i++) {
        b[i] -= w * J[i] * r;
        for (let j = i; j < 12; j++) H[i][j] += w * J[i] * J[j];
      }
    }
    for (let i = 0; i < 12; i++) {
      H[i][i] += 1e-3;
      for (let j = 0; j < i; j++) H[i][j] = H[j][i];
    }
    const s = solve(H, b);
    M = M.map((row, i) => row.map((v, j) => v + s[i * 4 + j]));
    t = t.map((v, i) => v + s[i * 4 + 3]);
  }
  return { M, t };
}

/** Per column: log N(f | arteries) - log N(f | veins), each a Gaussian (median, MAD) of the cells where the atlas is
 *  sure; a column whose classes overlap (d' < 1, a late scan) adds nothing. */
function ownLlr(f: number[][], sureA: boolean[], sureV: boolean[]): number[] {
  const n = f.length, out = new Array(n).fill(0);
  const nA = sureA.filter(Boolean).length, nV = sureV.filter(Boolean).length;
  if (nA < 30 || nV < 30 || !n) return out;
  const median = (v: number[]) => {
    const s = [...v].sort((p, q) => p - q);
    const m = s.length >> 1;
    return s.length % 2 ? s[m] : 0.5 * (s[m - 1] + s[m]);
  };
  for (let c = 0; c < f[0].length; c++) {
    const fa: number[] = [], fv: number[] = [];
    for (let i = 0; i < n; i++) {
      if (sureA[i]) fa.push(f[i][c]);
      if (sureV[i]) fv.push(f[i][c]);
    }
    const ma = median(fa), mv = median(fv);
    const sa = Math.max(1.4826 * median(fa.map((v) => Math.abs(v - ma))), 1e-3);
    const sv = Math.max(1.4826 * median(fv.map((v) => Math.abs(v - mv))), 1e-3);
    if (Math.abs(ma - mv) < Math.sqrt(0.5 * (sa * sa + sv * sv))) continue;
    for (let i = 0; i < n; i++) {
      const la = -0.5 * ((f[i][c] - ma) / sa) ** 2 - Math.log(sa), lv = -0.5 * ((f[i][c] - mv) / sv) ** 2 - Math.log(sv);
      out[i] += Math.min(LLR_CLIP, Math.max(-LLR_CLIP, la - lv));
    }
  }
  return out;
}

/** A size change between heads: overall 0.7-1.45 (adult to small child), no axis past 0.6-1.7. A skull-like shape that
 *  is no head (a tube of bone) can fit the skull surface too (dcv/atlas.py plausible). */
export function plausible(M: number[][]): boolean {
  const det = M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
    + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
  const size = Math.cbrt(Math.abs(det));
  const cols = [0, 1, 2].map((j) => Math.hypot(M[0][j], M[1][j], M[2][j]));
  return size >= 0.7 && size <= 1.45 && cols.every((c) => c >= 0.6 && c <= 1.7);
}

/**
 * Arteries and veins among the vessel cells of the head, by the atlas and this patient's own features (`feats(i)` of a
 * cell). Null without an atlas or when the skull does not fit it. `decided` = the cells it labelled.
 */
export function classifyHead(vessel: Uint8Array, feats: ((i: number) => number[]) | null, bone: Uint8Array, d: D3,
  sp: number[], atlas: VesselAtlas | null = shipped):
  { arteries: Uint8Array; veins: Uint8Array; decided: Uint8Array; prior: (mask: Uint8Array) => Float32Array } | null {
  if (!atlas) return null;
  const T = registerSkull(atlas, bone, d, sp);
  if (!T || T.fit < MIN_FIT || !plausible(T.M)) return null;
  const [nx, ny, nz] = d, nxy = nx * ny, N = vessel.length;
  const z0 = Math.max(0, topPlane(bone, d) - Math.floor(ATLAS_DEPTH_MM / sp[2]));
  const cells: number[] = [];
  for (let i = z0 * nxy; i < N; i++) if (vessel[i]) cells.push(i);
  if (!cells.length) return null;
  const [nzA, nyA, nxA] = atlas.shape;
  const L = new Float32Array(N), sureA: boolean[] = [], sureV: boolean[] = [];
  // the atlas share of arteries at cell i (0.5 where it knows nothing), unclipped
  const share = (i: number): number => {
    const p = [Math.floor(i / nxy) * sp[2] - T.c[0], (Math.floor(i / nx) % ny) * sp[1] - T.c[1], (i % nx) * sp[0] - T.c[2]];
    const q = [0, 1, 2].map((r) => T.M[r][0] * p[0] + T.M[r][1] * p[1] + T.M[r][2] * p[2] + T.t[r]);
    const gz = (q[0] - atlas.origin[0]) / atlas.sp[0], gy = (q[1] - atlas.origin[1]) / atlas.sp[1];
    const gx = (q[2] - atlas.origin[2]) / atlas.sp[2];
    const inside = gz >= 0 && gy >= 0 && gx >= 0 && gz <= nzA - 1 && gy <= nyA - 1 && gx <= nxA - 1;
    const su = inside ? sampleAt(atlas, atlas.support, q[0], q[1], q[2]) : 0;
    return su > SUPPORT_MIN && inside ? sampleAt(atlas, atlas.prob, q[0], q[1], q[2]) : 0.5;
  };
  cells.forEach((i, k) => {
    const pa = Math.min(0.98, Math.max(0.02, share(i)));
    L[i] = Math.log(pa / (1 - pa));
    sureA[k] = pa > SURE;
    sureV[k] = pa < 1 - SURE;
  });
  if (feats) {
    const f = cells.map(feats);
    const add = ownLlr(f, sureA, sureV);
    cells.forEach((i, k) => { L[i] += add[k]; });
  }
  const decided = new Uint8Array(N);
  for (const i of cells) decided[i] = 1;
  const pooled = poolAlong(L, decided, d);
  const arteries = new Uint8Array(N), veins = new Uint8Array(N);
  for (const i of cells) (pooled[i] > 0 ? arteries : veins)[i] = 1;
  // for phases.ts vesselTrees: the atlas share at the cells of any mask of this grid (dcv/atlas.py classify_head)
  const prior = (mask: Uint8Array): Float32Array => {
    const out = new Float32Array(N).fill(0.5);
    for (let i = 0; i < N; i++) if (mask[i]) out[i] = share(i);
    return out;
  };
  return { arteries, veins, decided, prior };
}

/** Each vessel cell's evidence pulled half-way to the mean of its vessel neighbours, 8 times (dcv/atlas.py _pool_along). */
function poolAlong(L: Float32Array, vessel: Uint8Array, [nx, ny]: D3): Float32Array {
  const nxy = nx * ny, N = L.length;
  const list: number[] = [];
  for (let i = 0; i < N; i++) if (vessel[i]) list.push(i);
  const offs = [1, -1, nx, -nx, nxy, -nxy];
  const nbrs = list.map((i) => offs.map((o) => i + o).filter((j) => j >= 0 && j < N && vessel[j]));
  let cur = Float32Array.from(list, (i) => L[i]);
  const pos = new Map<number, number>();
  list.forEach((i, k) => pos.set(i, k));
  const nb = nbrs.map((js) => js.map((j) => pos.get(j)!));
  for (let it = 0; it < 8; it++) {
    const next = new Float32Array(list.length);
    for (let k = 0; k < list.length; k++) {
      let s = 0;
      for (const j of nb[k]) s += cur[j];
      next[k] = 0.5 * L[list[k]] + (nb[k].length ? 0.5 * s / nb[k].length : 0);
    }
    cur = next;
  }
  const out = new Float32Array(N);
  list.forEach((i, k) => { out[i] = cur[k]; });
  return out;
}
