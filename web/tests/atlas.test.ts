import { describe, expect, it } from "vitest";
import { classifyHead, registerSkull, type VesselAtlas } from "../src/vr/atlas";

/** In a skull centred at c (mm, z y x), scaled by s: on the egg-shaped vault (radii 70 / 85 / 65 mm), and the slab of
 *  the anterior skull base under its front half (what makes a skull not turn into itself). */
function skullDist(z: number, y: number, x: number, c: number[], s: number): number {
  const r = Math.hypot((z - c[0]) / (70 * s), (y - c[1]) / (85 * s), (x - c[2]) / (65 * s));
  const shell = Math.abs(r - 1) * 72 * s;
  const out = (v: number, lo: number, hi: number) => Math.max(lo - v, 0, v - hi);
  const slab = Math.hypot(out(z, c[0] - 45 * s, c[0] - 40 * s), out(y, c[1] - 80 * s, c[1] - 10 * s),
    out(x, c[2] - 35 * s, c[2] + 35 * s));
  return Math.min(shell, slab);
}

/** An atlas of that skull (2 mm grid): arteries in its front half, veins in the back. */
function eggAtlas(): VesselAtlas {
  const shape: [number, number, number] = [90, 110, 90], sp: [number, number, number] = [2, 2, 2];
  const centre: [number, number, number] = [90, 110, 90];
  const n = shape[0] * shape[1] * shape[2];
  const dist = new Float32Array(n), prob = new Float32Array(n), support = new Float32Array(n);
  for (let z = 0; z < shape[0]; z++) for (let y = 0; y < shape[1]; y++) for (let x = 0; x < shape[2]; x++) {
    const i = (z * shape[1] + y) * shape[2] + x;
    dist[i] = Math.min(30, skullDist(z * 2, y * 2, x * 2, centre, 1));
    prob[i] = y * 2 < centre[1] ? 0.95 : 0.05;
    support[i] = 1;
  }
  return { shape, sp, origin: [0, 0, 0], centre, dist, prob, support };
}

/** A scan of the same skull 5 % larger, 6 mm further along y, on a 1.5 mm grid (index order x fastest). */
function scan() {
  const nx = 110, ny = 130, nz = 110, sp = [1.5, 1.5, 1.5], N = nx * ny * nz;
  const bone = new Uint8Array(N);
  const c = [82, 95 + 6, 82];                            // z, y, x mm
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
    if (skullDist(z * 1.5, y * 1.5, x * 1.5, c, 1.05) < 1.5 && z * 1.5 > c[0] - 50) bone[(z * ny + y) * nx + x] = 1;
  }
  return { bone, d: [nx, ny, nz] as [number, number, number], sp, c };
}

describe("vessel atlas", () => {
  it("lays a scan's skull onto the atlas skull: shifted and larger", () => {
    const a = eggAtlas(), s = scan();
    const T = registerSkull(a, s.bone, s.d, s.sp)!;
    expect(T.fit).toBeGreaterThan(0.8);
    expect(Math.cbrt(Math.abs(
      T.M[0][0] * (T.M[1][1] * T.M[2][2] - T.M[1][2] * T.M[2][1]) - T.M[0][1] * (T.M[1][0] * T.M[2][2] - T.M[1][2] * T.M[2][0])
      + T.M[0][2] * (T.M[1][0] * T.M[2][1] - T.M[1][1] * T.M[2][0])))).toBeCloseTo(1 / 1.05, 1);
  });

  it("labels the vessels by where they lie", () => {
    const a = eggAtlas(), s = scan(), [nx, ny, nz] = s.d;
    const vessel = new Uint8Array(s.bone.length);
    const front: number[] = [], back: number[] = [];
    for (let z = 60; z < 80; z++) for (const [y, list] of [[50, front], [85, back]] as const) {
      for (let x = 50; x < 60; x++) {
        const i = (z * ny + y) * nx + x;
        vessel[i] = 1;
        list.push(i);
      }
    }
    const r = classifyHead(vessel, null, s.bone, s.d, s.sp, a)!;
    expect(front.every((i) => r.arteries[i])).toBe(true);
    expect(back.every((i) => r.veins[i])).toBe(true);
    expect(nz).toBe(110);
  });
});
