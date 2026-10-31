import { describe, expect, it } from "vitest";
import { frameIndices, ownPlane, Reformat, sliceIndices, stepFor, tileSize } from "../src/view/collage";

const range = (a: number, b: number) => Array.from({ length: b - a }, (_, i) => a + i);

// the same cases as tests/test_collage.py of the desktop app
describe("collage slice selection", () => {
  it("one image every 5 mm, centred in the series", () => {
    const idx = sliceIndices(292, 0.625, 5, 25);
    expect(idx.length).toBe(25);
    expect(idx[1] - idx[0]).toBe(8);
    expect([idx[0] + 1, idx[24] + 1]).toEqual([50, 242]);
  });

  it("keeps slices that are already that far apart", () => {
    expect(sliceIndices(29, 5, 5, 25)).toEqual(range(2, 27));
    expect(sliceIndices(29, 5, 5, 36)).toEqual(range(0, 29));
    expect(sliceIndices(20, 6, 5, 25)).toEqual(range(0, 20));
    expect(stepFor(0, 5)).toBe(1);
  });

  it("clamps the start to the series", () => {
    expect(sliceIndices(292, 0.625, 5, 25, 0)[0]).toBe(0);
    expect(sliceIndices(292, 0.625, 5, 25, 999)[24]).toBe(291);
  });

  it("spreads movie frames over the image", () => {
    expect(frameIndices(10, 25)).toEqual(range(0, 10));
    const f = frameIndices(200, 25);
    expect(f.length).toBe(25);
    expect([f[0], f[24]]).toEqual([0, 199]);
  });
});

describe("collage page size", () => {
  it("keeps the shape of the image in mm", () => {
    expect(tileSize(384, 5, 5, 200, 100, 30)).toEqual([384, 192]);
  });

  it("narrows the tiles when the page would not fit a canvas", () => {
    const [w, h] = tileSize(512, 12, 12, 200, 200, 30);
    expect(w).toBeLessThan(512);
    expect(12 * w * (30 + 12 * h)).toBeLessThanOrEqual(16e6);
  });
});

describe("collage in another patient plane", () => {
  // an axial stack: 10 slices 4 mm apart, 20 x 30 pixels of 1 mm; value = x + 100 y + 10000 z in voxels
  const nx = 20, ny = 30, nz = 10;
  const g = { cols: nx, rows: ny, n: nz, spacing: [4, 1, 1] as [number, number, number], direction: [1, 0, 0, 0, 1, 0, 0, 0, 1], origin: [-10, -15, 100] as [number, number, number] };
  const data = new Float32Array(nx * ny * nz);
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) data[(z * ny + y) * nx + x] = x + 100 * y + 10000 * z;

  it("finds the acquired plane", () => {
    expect(ownPlane(g.direction)).toBe("axial");
    expect(ownPlane([0, 0, 1, 1, 0, 0, 0, -1, 0])).toBe("sagittal");
    expect(ownPlane([1, 0, 0, 0, 0, 1, 0, -1, 0])).toBe("coronal");
  });

  it("counts slices and spacing along the plane's normal", () => {
    const c = new Reformat(g, "coronal");
    expect([c.count, c.spacing, c.mmW, c.mmH]).toEqual([30, 1, 19, 36]);
    const s = new Reformat(g, "sagittal");
    expect([s.count, s.spacing, s.mmW, s.mmH]).toEqual([20, 1, 29, 36]);
    const a = new Reformat(g, "axial");
    expect([a.count, a.spacing]).toEqual([10, 4]);
  });

  it("a coronal slice: left to right = x, top = superior = last slice, at one y", () => {
    const c = new Reformat(g, "coronal");
    const out = new Float32Array(20 * 10);
    c.sample(data, 7, 20, 10, out, -1);
    expect(out[0]).toBeCloseTo(0 + 700 + 90000);       // top left: x 0, z 9
    expect(out[19]).toBeCloseTo(19 + 700 + 90000);     // top right
    expect(out[180]).toBeCloseTo(0 + 700 + 0);         // bottom left: z 0
    expect(c.position(7)).toBeCloseTo(-15 + 7);
  });

  it("a sagittal slice: left to right = posterior, top = superior", () => {
    const s = new Reformat(g, "sagittal");
    const out = new Float32Array(30 * 10);
    s.sample(data, 4, 30, 10, out, -1);
    expect(out[0]).toBeCloseTo(4 + 0 + 90000);
    expect(out[29]).toBeCloseTo(4 + 2900 + 90000);
    expect(s.position(4)).toBeCloseTo(-10 + 4);
  });

  it("axial: anterior is up, as in the MPR", () => {
    const a = new Reformat(g, "axial");
    const out = new Float32Array(20 * 30);
    a.sample(data, 3, 20, 30, out, -1);
    expect(out[0]).toBeCloseTo(30000);                 // top left: x 0, y 0 (anterior)
    expect(a.position(3)).toBeCloseTo(100 + 12);
  });
});
