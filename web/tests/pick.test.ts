import { describe, expect, it } from "vitest";
import { pickSurface, type PickScene } from "../src/vr/pick";

const N = 80;                   // 1 mm voxels: voxel i is centred at i + 0.5 mm
const data = new Int16Array(N * N * N);
for (let z = 0; z < N; z++) for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
  data[(z * N + y) * N + x] = Math.hypot(x + 0.5 - 40, y + 0.5 - 40, z + 0.5 - 40) < 20 ? 1000 : 0;
}

/** Opaque above 500. */
function scene(over: Partial<PickScene> = {}): PickScene {
  const W = 256;
  const alpha = new Float32Array(W);
  for (let i = 0; i < W; i++) alpha[i] = (i / (W - 1)) * 2000 >= 500 ? 1 : 0;
  return {
    data, cols: N, rows: N, dims: [N, N, N], start: [0, 0, 0], f: 1, spacing: [1, 1, 1], origin: [0, 0, 0],
    clipLo: [0, 0, 0], clipHi: [N, N, N], bone: null, ct: false, removed: null, air: 0, blend: 0,
    alpha, alphaW: W, alphaH: 1, x0: 0, x1: 2000, gradMax: 320, ...over,
  };
}

describe("3D surface pick", () => {
  it("lands on the near surface of what is opaque", () => {
    const p = pickSurface(scene(), [40, 40, -100], [0, 0, 1])!;
    expect(p[2]).toBeGreaterThan(19);
    expect(p[2]).toBeLessThan(21.5);
    expect(p[0]).toBe(40);
  });

  it("works from the other side and misses beside the sphere", () => {
    expect(pickSurface(scene(), [40, 40, 200], [0, 0, -1])![2]).toBeGreaterThan(58.5);
    expect(pickSurface(scene(), [5, 5, -100], [0, 0, 1])).toBeNull();
    expect(pickSurface(scene(), [200, 40, -100], [0, 0, 1])).toBeNull();
  });

  it("the crop box and scalpel cuts move the hit deeper", () => {
    expect(pickSurface(scene({ clipLo: [0, 0, 40] }), [40, 40, -100], [0, 0, 1])![2]).toBeLessThan(41);
    const removed = new Uint32Array(Math.ceil((N * N * N) / 32));
    for (let z = 0; z < 30; z++) for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const i = (z * N + y) * N + x;
      removed[i >>> 5] |= 1 << (i & 31);
    }
    const z = pickSurface(scene({ removed }), [40, 40, -100], [0, 0, 1])![2];
    expect(z).toBeGreaterThan(29);
    expect(z).toBeLessThan(31.5);
  });

  it("a half-resolution grid gives nearly the same point", () => {
    const n = N / 2;
    const p = pickSurface(scene({ dims: [n, n, N], f: 2, spacing: [2, 2, 1] }), [40, 40, -100], [0, 0, 1])!;
    expect(p[2]).toBeGreaterThan(18);
    expect(p[2]).toBeLessThan(23);
  });

  it("MIP picks the brightest sample", () => {
    const p = pickSurface(scene({ blend: 1 }), [40, 40, -100], [0, 0, 1])!;
    expect(p[2]).toBeGreaterThan(19);
    expect(p[2]).toBeLessThan(61);
  });

  it("is fast enough to follow a finger", () => {
    const t0 = performance.now();
    for (let i = 0; i < 100; i++) pickSurface(scene(), [40, 40, -100], [0, 0, 1]);
    expect((performance.now() - t0) / 100).toBeLessThan(5);
  });
});
