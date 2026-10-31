/**
 * Which surface is under the pointer: one ray marched on the CPU through the decoded volume with the current colour
 * curves, the same way the shader does (renderer.ts) — so a 3D measurement lands on what is seen, with no extra
 * graphics memory and no read-back from the GPU. Only the opacity of the table is kept, at low resolution.
 */
import { BONE_OFFSET, CT_FLOOR, encodeVessel, maskAt, VEIN_OFFSET, type BoneRegion, type VesselMasks } from "./bone";

type V3 = [number, number, number];

export interface PickScene {
  data: ArrayLike<number>;
  cols: number;
  rows: number;
  /** Texture grid of the renderer: its size, first full-resolution voxel, in-plane step, mm per voxel, mm origin. */
  dims: V3;
  start: V3;
  f: number;
  spacing: V3;
  origin: V3;
  /** The part of the ray to march: content box ∩ crop box, in mm. */
  clipLo: V3;
  clipHi: V3;
  bone: BoneRegion | null;
  vessels?: VesselMasks | null;
  ct: boolean;
  removed: Uint32Array | null;
  air: number;
  /** VR: opacity table over value × gradient. MIP / MinIP (blend 1, 2): the brightest / darkest sample instead. */
  blend: 0 | 1 | 2;
  alpha: Float32Array;
  alphaW: number;
  alphaH: number;
  x0: number;
  x1: number;
  gradMax: number;
  /** A segment's surface at this point (mm): segments are opaque, the ray stops there. */
  solid?: (x: number, y: number, z: number) => boolean;
}

/** First point along the ray where half of the light is stopped, in mm; null when the ray meets nothing visible. */
export function pickSurface(s: PickScene, ro: V3, rd: V3): V3 | null {
  const [nx, ny, nz] = s.dims;
  let tn = 0, tf = Infinity;
  for (let a = 0; a < 3; a++) {
    const lo = Math.max(s.clipLo[a], s.origin[a]);
    const hi = Math.min(s.clipHi[a], s.origin[a] + s.dims[a] * s.spacing[a]);
    if (Math.abs(rd[a]) < 1e-12) {
      if (ro[a] < lo || ro[a] > hi) return null;
      continue;
    }
    const t0 = (lo - ro[a]) / rd[a], t1 = (hi - ro[a]) / rd[a];
    tn = Math.max(tn, Math.min(t0, t1));
    tf = Math.min(tf, Math.max(t0, t1));
  }
  if (tf <= tn) return null;

  const { data, cols, f, removed, bone, ct, air } = s;
  const [sx0, sy0, sz0] = s.start;
  /** The value the renderer stored for a texture voxel: cuts = air, CT floored, bone area mirrored. */
  const voxel = (x: number, y: number, z: number): number => {
    x = x < 0 ? 0 : x >= nx ? nx - 1 : x;
    y = y < 0 ? 0 : y >= ny ? ny - 1 : y;
    z = z < 0 ? 0 : z >= nz ? nz - 1 : z;
    if (removed) {
      const i = (z * ny + y) * nx + x;
      if (removed[i >>> 5] & (1 << (i & 31))) return air;
    }
    const X = sx0 + x * f, Y = sy0 + y * f, Z = sz0 + z;
    let v = data[(Z * s.rows + Y) * cols + X];
    if (ct && v < CT_FLOOR) v = CT_FLOOR;
    if (bone) {
      const b = (Z * bone.ny + Math.floor(Y / bone.f)) * bone.nx + Math.floor(X / bone.f);
      if (bone.bits[b >>> 5] & (1 << (b & 31))) return -BONE_OFFSET - v;
    }
    if (s.vessels) v = encodeVessel(v, maskAt(s.vessels.arteries, X, Y, Z), maskAt(s.vessels.veins, X, Y, Z));
    return v;
  };
  const [ox, oy, oz] = s.origin, [dx, dy, dz] = s.spacing;
  /** Trilinear, as the GPU's filter: voxel i is centred at origin + (i + 0.5) · spacing. */
  const sample = (px: number, py: number, pz: number): number => {
    const X = (px - ox) / dx - 0.5, Y = (py - oy) / dy - 0.5, Z = (pz - oz) / dz - 0.5;
    const x = Math.floor(X), y = Math.floor(Y), z = Math.floor(Z);
    const fx = X - x, fy = Y - y, fz = Z - z;
    const c00 = voxel(x, y, z) * (1 - fx) + voxel(x + 1, y, z) * fx;
    const c10 = voxel(x, y + 1, z) * (1 - fx) + voxel(x + 1, y + 1, z) * fx;
    const c01 = voxel(x, y, z + 1) * (1 - fx) + voxel(x + 1, y, z + 1) * fx;
    const c11 = voxel(x, y + 1, z + 1) * (1 - fx) + voxel(x + 1, y + 1, z + 1) * fx;
    const c0 = c00 * (1 - fy) + c10 * fy, c1 = c01 * (1 - fy) + c11 * fy;
    return c0 * (1 - fz) + c1 * fz;
  };

  const minSp = Math.min(dx, dy, dz);
  const step = minSp * 0.6;
  const W = s.alphaW, H = s.alphaH;
  const kx = (W - 1) / (s.x1 - s.x0);
  let acc = 0, best = -1, bestT = -1;
  let extreme = s.blend === 1 ? -Infinity : Infinity;
  for (let t = tn + step / 2; t <= tf; t += step) {
    const px = ro[0] + rd[0] * t, py = ro[1] + rd[1] * t, pz = ro[2] + rd[2] * t;
    if (s.blend === 0 && s.solid?.(px, py, pz)) return [px, py, pz];
    const v = sample(px, py, pz);
    if (s.blend !== 0) {
      const hu = v < -2048 ? -BONE_OFFSET - v : v > 10000 ? v - VEIN_OFFSET : v;
      if (s.blend === 1 ? hu > extreme : hu < extreme) {
        extreme = hu;
        bestT = t;
      }
      continue;
    }
    const col = Math.round((v - s.x0) * kx);
    if (col < 0 || col >= W) continue;
    let c = s.alpha[col];
    if (c <= 0.0005) continue;
    if (H > 1) {
      const gx = (sample(px + dx, py, pz) - sample(px - dx, py, pz)) / (2 * dx);
      const gy = (sample(px, py + dy, pz) - sample(px, py - dy, pz)) / (2 * dy);
      const gz = (sample(px, py, pz + dz) - sample(px, py, pz - dz)) / (2 * dz);
      const row = Math.min(H - 1, Math.floor((Math.hypot(gx, gy, gz) * minSp * H) / s.gradMax));
      c = s.alpha[row * W + col];
      if (c <= 0.0005) continue;
    }
    const a = 1 - Math.pow(1 - Math.min(c, 0.9999), step);
    if (a > best) {
      best = a;
      bestT = t;
    }
    acc += (1 - acc) * a;
    if (acc >= 0.5) return [px, py, pz];
  }
  if (bestT < 0 || (s.blend === 0 && acc < 0.02)) return null;   // nearly transparent all the way: nothing to hold on to
  return [ro[0] + rd[0] * bestT, ro[1] + rd[1] * bestT, ro[2] + rd[2] * bestT];
}
