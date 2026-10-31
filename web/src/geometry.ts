/** Small 3-vector / 3×3 matrix helpers (row-major 3×3 as number[9]). */
export type V3 = [number, number, number];
export type M3 = number[];

export const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const scale = (a: V3, s: number): V3 => [a[0] * s, a[1] * s, a[2] * s];
export const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const norm = (a: V3) => Math.hypot(a[0], a[1], a[2]);
export const unit = (a: V3): V3 => scale(a, 1 / (norm(a) || 1));

export const I3: M3 = [1, 0, 0, 0, 1, 0, 0, 0, 1];
export const col = (m: M3, j: number): V3 => [m[j], m[3 + j], m[6 + j]];
export const mulMV = (m: M3, v: V3): V3 => [
  m[0] * v[0] + m[1] * v[1] + m[2] * v[2], m[3] * v[0] + m[4] * v[1] + m[5] * v[2], m[6] * v[0] + m[7] * v[1] + m[8] * v[2]];
export const mulMtV = (m: M3, v: V3): V3 => [               // transpose(m) · v
  m[0] * v[0] + m[3] * v[1] + m[6] * v[2], m[1] * v[0] + m[4] * v[1] + m[7] * v[2], m[2] * v[0] + m[5] * v[1] + m[8] * v[2]];
export function mulMM(a: M3, b: M3): M3 {
  const o = new Array(9).fill(0);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) o[i * 3 + j] += a[i * 3 + k] * b[k * 3 + j];
  return o;
}

/** Rotation by theta around axis (Rodrigues), as dcv/mpr_window.py _rot. */
export function rot(axis: V3, theta: number): M3 {
  const [x, y, z] = unit(axis);
  const c = Math.cos(theta), s = Math.sin(theta), t = 1 - c;
  return [t * x * x + c, t * x * y - s * z, t * x * z + s * y,
    t * x * y + s * z, t * y * y + c, t * y * z - s * x,
    t * x * z - s * y, t * y * z + s * x, t * z * z + c];
}

/** Image axes → patient LPS: columns = row direction, column direction, slice normal (Image Orientation Patient). */
export function directionFromIop(iop: number[] | null): M3 {
  if (!iop) return [...I3];
  const r = unit([iop[0], iop[1], iop[2]]);
  const c = unit([iop[3], iop[4], iop[5]]);
  const n = cross(r, c);
  return [r[0], c[0], n[0], r[1], c[1], n[1], r[2], c[2], n[2]];
}

/** Patient orientation letters of a direction in LPS (+x = L, +y = P, +z = S), like _letter in the desktop. */
export function letter(v: V3): string {
  const names = [["R", "L"], ["A", "P"], ["I", "S"]];
  const order = [0, 1, 2].sort((i, j) => Math.abs(v[j]) - Math.abs(v[i]));
  let out = "";
  for (const i of order.slice(0, 2)) if (Math.abs(v[i]) > 0.25) out += names[i][v[i] > 0 ? 1 : 0];
  return out;
}
