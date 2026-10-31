/**
 * Scalpel (dcv/volume_view.py project_inside + segment.py apply_cut): the voxels whose screen projection falls
 * inside (or outside) a lasso drawn on the 3D view are removed through the whole depth. Removed voxels are a
 * bitset over the GPU texture grid; the renderer writes them as air. Rows whose projected segment cannot touch the
 * lasso are skipped, so a cut of a 300 × 512² CT takes a fraction of a second.
 */

export interface LassoMask {
  mask: Uint8Array;
  w: number;
  h: number;
  /** bounding box of the filled pixels */
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

/** Fill a closed polygon (screen points scaled to a w × h grid) with the even-odd rule, scanline by scanline. */
export function fillLasso(pts: [number, number][], w: number, h: number): LassoMask {
  const mask = new Uint8Array(w * h);
  let x0 = w, y0 = h, x1 = -1, y1 = -1;
  const n = pts.length;
  const xs: number[] = [];
  for (let y = 0; y < h; y++) {
    const yc = y + 0.5;
    xs.length = 0;
    for (let i = 0; i < n; i++) {
      const [ax, ay] = pts[i], [bx, by] = pts[(i + 1) % n];
      if ((ay <= yc && by > yc) || (by <= yc && ay > yc)) xs.push(ax + ((yc - ay) / (by - ay)) * (bx - ax));
    }
    xs.sort((a, b) => a - b);
    for (let k = 0; k + 1 < xs.length; k += 2) {
      const a = Math.max(0, Math.ceil(xs[k] - 0.5)), b = Math.min(w - 1, Math.floor(xs[k + 1] - 0.5));
      if (b < a) continue;
      mask.fill(1, y * w + a, y * w + b + 1);
      if (a < x0) x0 = a;
      if (b > x1) x1 = b;
      if (y < y0) y0 = y;
      if (y > y1) y1 = y;
    }
  }
  return { mask, w, h, x0, y0, x1, y1 };
}

/**
 * Mark voxels (texture grid dims, spacing in mm, voxel i centred at (i + 0.5)·spacing) whose projection with the
 * column-major view-projection matrix `vp` falls inside the lasso (outside when `outside`). Returns the slice range
 * that changed and how many voxels were newly removed.
 */
export function cutVoxels(removed: Uint32Array, dims: [number, number, number], spacing: [number, number, number],
  vp: Float32Array, L: LassoMask, outside: boolean, origin: [number, number, number] = [0, 0, 0]): { z0: number; z1: number; count: number } {
  const [nx, ny, nz] = dims;
  const [sx, sy, sz] = spacing;
  const m = vp;
  const { mask, w, h } = L;
  let z0 = nz, z1 = -1, count = 0;
  const empty = L.x1 < 0;
  if (empty && !outside) return { z0: 0, z1: 0, count: 0 };
  const proj = (X: number, Y: number, Z: number): [number, number, number] => {
    const cw = m[3] * X + m[7] * Y + m[11] * Z + m[15];
    const cx = m[0] * X + m[4] * Y + m[8] * Z + m[12];
    const cy = m[1] * X + m[5] * Y + m[9] * Z + m[13];
    return [((cx / cw + 1) / 2) * w, ((1 - cy / cw) / 2) * h, cw];
  };
  const dX = sx;                                      // one voxel along x in mm
  for (let z = 0; z < nz; z++) {
    const Z = origin[2] + (z + 0.5) * sz;
    for (let y = 0; y < ny; y++) {
      const Y = origin[1] + (y + 0.5) * sy;
      const X0 = origin[0] + 0.5 * sx;
      if (!outside) {                                 // row culling: its screen segment must reach the lasso box
        const [ax, ay, aw] = proj(X0, Y, Z), [bx, by, bw] = proj(X0 + (nx - 1) * sx, Y, Z);
        if (aw > 0 && bw > 0 && (Math.max(ax, bx) < L.x0 - 1 || Math.min(ax, bx) > L.x1 + 2 || Math.max(ay, by) < L.y0 - 1 || Math.min(ay, by) > L.y1 + 2)) continue;
      }
      // clip coordinates are linear in X: step them instead of a full matrix product per voxel
      let cx = m[0] * X0 + m[4] * Y + m[8] * Z + m[12];
      let cy = m[1] * X0 + m[5] * Y + m[9] * Z + m[13];
      let cw = m[3] * X0 + m[7] * Y + m[11] * Z + m[15];
      const ex = m[0] * dX, ey = m[1] * dX, ew = m[3] * dX;
      const base = (z * ny + y) * nx;
      let rowHit = false;
      for (let x = 0; x < nx; x++, cx += ex, cy += ey, cw += ew) {
        let inside = false;
        if (cw > 0) {
          const px = Math.floor(((cx / cw + 1) / 2) * w), py = Math.floor(((1 - cy / cw) / 2) * h);
          inside = px >= 0 && py >= 0 && px < w && py < h && mask[py * w + px] === 1;
        }
        if (inside === outside) continue;
        const i = base + x, word = i >>> 5, bit = 1 << (i & 31);
        if (removed[word] & bit) continue;
        removed[word] |= bit;
        count++;
        rowHit = true;
      }
      if (rowHit) {
        if (z < z0) z0 = z;
        if (z > z1) z1 = z;
      }
    }
  }
  return z1 < 0 ? { z0: 0, z1: 0, count: 0 } : { z0, z1: z1 + 1, count };
}
