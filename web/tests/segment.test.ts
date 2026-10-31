import { describe, expect, it } from "vitest";
import { dragRange, fmtMl, histColumns, RegionGrower, volumeMl } from "../src/seg/segment";

const N = 64;
/** Background 0, a ball of 100 ± noise at x = 20, a ball of 100 at x = 46, a 1-voxel bridge between them. */
function phantom(bridge: boolean): Int16Array {
  const d = new Int16Array(N * N * N);
  for (let z = 0; z < N; z++) for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const a = Math.hypot(x - 20, y - 32, z - 32) < 8, b = Math.hypot(x - 46, y - 32, z - 32) < 8;
    if (a || b) d[(z * N + y) * N + x] = 100 + ((x + y + z) % 3);
    if (bridge && y === 32 && z === 32 && x > 20 && x < 46) d[(z * N + y) * N + x] = 100;
  }
  return d;
}

const S: [number, number, number] = [1, 1, 1];
const ballVoxels = () => {
  let n = 0;
  for (let z = 0; z < N; z++) for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) if (Math.hypot(x - 20, y - 32, z - 32) < 8) n++;
  return n;
};

describe("region growing", () => {
  it("finds the connected ball and its bounding box", () => {
    const g = new RegionGrower(phantom(false), N, N, N, S, [32, 32, 20]).grow(90, 110)!;
    expect(g.count).toBe(ballVoxels());
    expect(g.off).toEqual([25, 25, 13]);
    expect(g.dims).toEqual([15, 15, 15]);
  });

  it("anti-leak stops at a thin bridge; without it the segment leaks", () => {
    const d = phantom(true);
    const safe = new RegionGrower(d, N, N, N, S, [32, 32, 20]).grow(90, 110, 1)!;
    expect(safe.count).toBeGreaterThan(ballVoxels() * 0.9);
    expect(safe.count).toBeLessThan(ballVoxels() * 1.1);
    const leak = new RegionGrower(d, N, N, N, S, [32, 32, 20]).grow(90, 110, 0)!;
    expect(leak.count).toBeGreaterThan(ballVoxels() * 1.9);
  });

  it("nothing when the seed is outside the range, and a seed on a thin part still grows", () => {
    const d = phantom(true);
    expect(new RegionGrower(d, N, N, N, S, [32, 32, 20]).grow(200, 300)).toBeNull();
    expect(new RegionGrower(d, N, N, N, S, [32, 32, 33]).grow(90, 110, 1)!.count).toBeGreaterThan(ballVoxels() * 1.9);
  });

  it("the same grower can grow again with another range (the range editor)", () => {
    const d = phantom(false);
    const g = new RegionGrower(d, N, N, N, S, [32, 32, 20]);
    const a = g.grow(90, 110)!.count, b = g.grow(100, 100, 0)!, c = g.grow(90, 110)!.count;
    expect(c).toBe(a);
    expect(b.count).toBeLessThan(a);
  });

  it("stays in the ±70 mm block", () => {
    const d = new Int16Array(N * N * N).fill(100);
    const g = new RegionGrower(d, N, N, N, [4, 4, 4], [32, 32, 32], 70).grow(0, 200, 0)!;
    expect(g.dims).toEqual([37, 37, 37]);             // 70 / 4 → ±18 voxels
  });

  it("automatic range from around the seed", () => {
    const g = new RegionGrower(phantom(false), N, N, N, S, [32, 32, 20]);
    const r = g.autoGrow();
    expect(r.grown!.count).toBeGreaterThan(ballVoxels() * 0.9);
    expect(r.lo).toBeGreaterThan(50);
    expect(r.hi).toBeLessThan(150);
  });

  it("follows the segment's size, not the block's", () => {
    const n = 300;
    const d = new Int16Array(n * n * n);
    for (let z = 140; z < 160; z++) for (let y = 140; y < 160; y++) for (let x = 140; x < 160; x++) d[(z * n + y) * n + x] = 100;
    const g = new RegionGrower(d, n, n, n, [0.5, 0.5, 0.5], [150, 150, 150]);
    const t0 = performance.now();
    for (let i = 0; i < 5; i++) g.grow(90, 110);
    expect((performance.now() - t0) / 5).toBeLessThan(80);
  });
});

describe("range editor", () => {
  it("the value at the seed stays inside the range", () => {
    expect(dragRange("lo", 500, 400, 600, 429)).toEqual([429, 600]);
    expect(dragRange("hi", 300, 400, 600, 429)).toEqual([400, 429]);
    expect(dragRange("lo", 350, 400, 600, 429)).toEqual([350, 600]);
    expect(dragRange("hi", 700, 400, 600, 429)).toEqual([400, 700]);
  });
});

describe("volume and histogram", () => {
  it("volume in ml", () => {
    expect(volumeMl(1000, [1, 1, 1])).toBeCloseTo(1);
    expect(volumeMl(1000, [2, 0.5, 0.5])).toBeCloseTo(0.5);
    expect(fmtMl(0.5)).toBe("0.50 ml");
    expect(fmtMl(12.34)).toBe("12.3 ml");
  });

  it("histogram around the seed covers its values and the view resamples it", () => {
    const g = new RegionGrower(phantom(false), N, N, N, S, [32, 32, 20]);
    const h = g.histogram(10);
    expect(h.min).toBe(0);
    expect(h.max).toBe(102);
    const cols = histColumns(h, 0, 102, 50);
    expect(cols.length).toBe(50);
    expect(Math.max(...cols)).toBe(1);
    expect(cols[25]).toBe(0);                          // nothing between background and ball
    expect(cols[49]).toBeGreaterThan(0.5);
  });
});
