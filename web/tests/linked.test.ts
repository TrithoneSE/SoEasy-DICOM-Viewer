import { describe, expect, it } from "vitest";
import { imageOf, referenceLine, worldOf, type Plane } from "../src/view/linked";

// 200 × 200 images, 1 mm pixels, in patient LPS
const axial = (z: number): Plane => ({ o: [-100, -100, z], r: [1, 0, 0], c: [0, 1, 0], n: [0, 0, 1], sx: 1, sy: 1, rows: 200, cols: 200 });
const sagittal = (x: number): Plane => ({ o: [x, -100, 100], r: [0, 1, 0], c: [0, 0, -1], n: [1, 0, 0], sx: 1, sy: 1, rows: 200, cols: 200 });
const coronal = (y: number, rows = 200): Plane => ({ o: [-100, y, 100], r: [1, 0, 0], c: [0, 0, -1], n: [0, 1, 0], sx: 1, sy: 1, rows, cols: 200 });

describe("reference lines", () => {
  it("image ↔ patient coordinates round-trip, pixel centres at i + 0.5", () => {
    const P = sagittal(3);
    expect(worldOf(P, 0.5, 0.5)).toEqual([3, -100, 100]);
    const [r, c, d] = imageOf(P, worldOf(P, 17.25, 42.5));
    expect(r).toBeCloseTo(17.25);
    expect(c).toBeCloseTo(42.5);
    expect(d).toBeCloseTo(0);
  });

  it("a sagittal slice cuts an axial image along one column", () => {
    const seg = referenceLine(axial(0), sagittal(0))!;
    expect(seg).not.toBeNull();
    for (const [, c] of seg) expect(c).toBeCloseTo(100.5);
    const rows = seg.map(([r]) => r).sort((a, b) => a - b);
    expect(rows[0]).toBeCloseTo(0);
    expect(rows[1]).toBeCloseTo(200);
  });

  it("an axial slice cuts a coronal image along one row", () => {
    const seg = referenceLine(coronal(0), axial(20))!;
    for (const [r] of seg) expect(r).toBeCloseTo(80.5);              // z = 100 − (row − 0.5)
  });

  it("is clipped to the other image and absent for parallel or missed planes", () => {
    // a coronal image only 50 mm high (z 100 … 51): on a sagittal image its line stops there
    const seg = referenceLine(sagittal(0), coronal(0, 50))!;
    const rows = seg.map(([r]) => r).sort((a, b) => a - b);
    expect(rows[0]).toBeCloseTo(0);
    expect(rows[1]).toBeCloseTo(50);
    expect(referenceLine(axial(0), axial(5))).toBeNull();            // parallel
    expect(referenceLine(axial(150), sagittal(0))).toBeNull();       // above the sagittal image
  });
});
