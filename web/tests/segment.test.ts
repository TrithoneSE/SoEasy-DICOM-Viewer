import { describe, expect, it } from "vitest";
import { dragRange, fmtMl, HalfGrower, histColumns, RegionGrower, volumeMl } from "../src/seg/segment";

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

  it("re-grows (a handle drag) in the time of the segment's size, not the block's", () => {
    const n = 300;
    const d = new Int16Array(n * n * n);
    for (let z = 140; z < 160; z++) for (let y = 140; y < 160; y++) for (let x = 140; x < 160; x++) d[(z * n + y) * n + x] = 100;
    const g = new RegionGrower(d, n, n, n, [0.5, 0.5, 0.5], [150, 150, 150]);
    g.grow(90, 110);                                  // the first grow smooths the block, once per seed
    const t0 = performance.now();
    for (let i = 0; i < 5; i++) g.grow(90, 110);
    expect((performance.now() - t0) / 5).toBeLessThan(80);
  });

  it("on the halved grid (phones) a lesion comes out as on the full grid, mask at full size", () => {
    // 0.4 mm pixels, noise; a 10 mm ball
    const n = 120, sp: [number, number, number] = [0.625, 0.4, 0.4], nz = 80, d = new Int16Array(nz * n * n);
    let rnd = 9, ball = 0;
    const noise = () => ((rnd = (rnd * 1103515245 + 12345) & 0x7fffffff) % 17) - 8;
    for (let z = 0; z < nz; z++) for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
      const inB = ((z - 40) * 0.625) ** 2 + ((y - 60) * 0.4) ** 2 + ((x - 61) * 0.4) ** 2 < 100;
      if (inB) ball++;
      d[(z * n + y) * n + x] = (inB ? 80 : 35) + noise();
    }
    const full = new RegionGrower(d, n, n, nz, sp, [40, 60, 61]).grow(60, 100)!;
    const half = new HalfGrower(d, n, n, nz, sp, [40, 60, 61]).grow(60, 100)!;
    expect(Math.abs(half.count - full.count) / full.count).toBeLessThan(0.08);
    expect(Math.abs(half.count - ball) / ball).toBeLessThan(0.1);
    expect(half.sub.length).toBe(half.dims[0] * half.dims[1] * half.dims[2]);
    expect(half.sub.reduce((a, b) => a + b, 0)).toBe(half.count);
    const [oz, oy, ox] = half.off;
    expect(half.sub[((40 - oz) * half.dims[1] + 60 - oy) * half.dims[2] + 61 - ox]).toBe(1);     // the seed is in it
  });

  it("on CT the automatic range stays out of the brain next to a heterogeneous tumour", () => {
    // brain ~35 HU; a 12 mm tumour of two parts (65 and 95 HU) and a few bright vessels: mean − 2.5·SD reaches the brain
    const n = 120, sp: [number, number, number] = [0.5, 0.5, 0.5], d = new Int16Array(n * n * n);
    let rnd = 5;
    const noise = (a: number) => (((rnd = (rnd * 1103515245 + 12345) & 0x7fffffff) % (2 * a + 1)) - a);
    let ball = 0;
    for (let z = 0; z < n; z++) for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
      const r2 = ((z - 60) ** 2 + (y - 60) ** 2 + (x - 60) ** 2) * 0.25;
      const inT = r2 < 144;
      if (inT) ball++;
      const v = inT ? (x < 60 ? 65 : 95) + ((x * 7 + y * 3 + z) % 23 === 0 ? 120 : 0) : 35;
      d[(z * n + y) * n + x] = v + noise(inT ? 14 : 9);
    }
    const seed: [number, number, number] = [60, 60, 59];
    const plain = new RegionGrower(d, n, n, n, sp, seed).autoGrow(2.5, undefined, false);
    const guarded = new RegionGrower(d, n, n, n, sp, seed).autoGrow(2.5, undefined, true);
    expect(plain.grown!.count).toBeGreaterThan(ball * 3);        // without the guard: into the brain
    expect(guarded.lo).toBeGreaterThan(45);
    expect(guarded.grown!.count).toBeGreaterThan(ball * 0.6);
    expect(guarded.grown!.count).toBeLessThan(ball * 1.3);
  }, 30000);

  it("grows in windows around the click exactly as in the whole block, and a small lesion fast", () => {
    // 0.5 mm voxels, noise; a 4 mm lesion (first window) and a 14 mm one touching a 6 mm one (larger windows)
    const n = 160, sp: [number, number, number] = [0.5, 0.5, 0.5], d = new Int16Array(n * n * n);
    let rnd = 3;
    const noise = () => ((rnd = (rnd * 1103515245 + 12345) & 0x7fffffff) % 25) - 12;
    const ball = (z: number, y: number, x: number, c: number[], r: number) =>
      ((z - c[0]) ** 2 + (y - c[1]) ** 2 + (x - c[2]) ** 2) * 0.25 < r * r;
    for (let z = 0; z < n; z++) for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
      const lesion = ball(z, y, x, [80, 40, 80], 4) || ball(z, y, x, [80, 95, 70], 14) || ball(z, y, x, [80, 95, 104], 6);
      d[(z * n + y) * n + x] = (lesion ? 70 : 35) + noise();
    }
    for (const seed of [[80, 40, 80], [80, 95, 70]] as [number, number, number][]) {
      const g = new RegionGrower(d, n, n, n, sp, seed);
      const t0 = performance.now();
      const r = g.grow(55, 90)!;
      const ms = performance.now() - t0;
      const whole = new RegionGrower(d, n, n, n, sp, seed);
      (whole as unknown as { lastIdx: number[] }).lastIdx = [9, 9, 9, 9, 9, 9];     // straight to the ±70 mm block
      const w = whole.grow(55, 90)!;
      expect([r.count, ...r.off, ...r.dims]).toEqual([w.count, ...w.off, ...w.dims]);
      expect(r.sub).toEqual(w.sub);
      if (seed[1] === 40) expect(ms).toBeLessThan(150);
      expect(g.grow(50, 95)).not.toBeNull();       // a dragged handle starts from the window it needed
    }
  }, 30000);

  it("a mass touching a neighbour of the same values over a narrow neck does not take it in", () => {
    // two noisy blocks of the same HU joined by a 2-voxel neck (0.4 mm voxels: 0.8 mm), like a tumour touching muscle
    const nx = 120, ny = 60, nz = 40, d = new Int16Array(nx * ny * nz).fill(-100);
    let rnd = 7;
    const noise = () => ((rnd = (rnd * 1103515245 + 12345) & 0x7fffffff) % 15) - 7;
    for (let z = 5; z < 35; z++) for (let y = 10; y < 50; y++) for (let x = 5; x < 115; x++) {
      const inA = x < 50, inB = x > 70, neck = Math.abs(y - 30) < 1 && Math.abs(z - 20) < 1;
      if (inA || inB || neck) d[(z * ny + y) * nx + x] = 45 + noise();
    }
    const g = new RegionGrower(d, nx, ny, nz, [0.625, 0.4, 0.4], [20, 30, 25]);
    const a = 30 * 40 * 45;
    const cut = g.grow(25, 70)!;
    expect(cut.count).toBeGreaterThan(a * 0.9);
    expect(cut.count).toBeLessThan(a * 1.1);
    expect(g.grow(25, 70, 0)!.count).toBeGreaterThan(a * 1.9);
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
