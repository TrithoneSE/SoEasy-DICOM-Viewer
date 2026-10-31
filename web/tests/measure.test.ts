import { describe, expect, it } from "vitest";
import { angleOf, cobbOf, ellipseAreaMm2, ellipseStats, fmtArea, measureLines, type Measurement } from "../src/view/measure";

describe("angle and Cobb, as dcv/measure.py", () => {
  it("measures the angle at the vertex in mm", () => {
    expect(angleOf([[0, 10], [0, 0], [10, 0]], 1, 1)).toBeCloseTo(90);
    expect(angleOf([[0, 10], [0, 0], [0, -10]], 1, 1)).toBeCloseTo(180);
    // non-square pixels: 1 row = 2 mm makes the arm steeper than 45°
    expect(angleOf([[0, 10], [0, 0], [10, 10]], 2, 1)).toBeCloseTo((Math.atan2(20, 10) * 180) / Math.PI);
  });

  it("Cobb is the acute angle between the two lines, whichever way they were drawn", () => {
    expect(cobbOf([[0, 0], [0, 10], [0, 0], [10, 10]], 1, 1)).toBeCloseTo(45);
    expect(cobbOf([[0, 10], [0, 0], [0, 0], [10, 10]], 1, 1)).toBeCloseTo(45);
    expect(cobbOf([[0, 0], [0, 10], [5, 0], [5, 10]], 1, 1)).toBeCloseTo(0);
  });

  it("captions", () => {
    const line = (kind: Measurement["kind"], pts: [number, number][]) => measureLines({ kind, pts }, 1, 1, true, " HU", null);
    expect(line("angle", [[0, 10], [0, 0], [10, 0]])).toEqual(["90.0°"]);
    expect(line("cobb", [[0, 0], [0, 10], [0, 0], [10, 10]])).toEqual(["Cobb: 45.0°"]);
    expect(line("length", [[0, 0], [0, 8.4]])).toEqual(["8.4 mm"]);
    expect(line("angle", [[0, 0], [0, 1]])).toEqual([]);          // not finished yet: no caption
  });
});

describe("ellipse ROI", () => {
  const rows = 100, cols = 100;
  const plane = new Int16Array(rows * cols);
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) plane[r * cols + c] = c < 50 ? 10 : 30;

  it("counts the pixels whose centre is inside", () => {
    const s = ellipseStats(plane, rows, cols, [[20, 20], [40, 40]])!;       // inside the left half, radius 10
    expect(s.mean).toBe(10);
    expect(s.sd).toBe(0);
    expect(Math.abs(s.n - Math.PI * 100)).toBeLessThan(12);
  });

  it("mean, SD, min, max across two values", () => {
    const s = ellipseStats(plane, rows, cols, [[40, 40], [60, 60]])!;       // half over each side
    expect(s.mean).toBeCloseTo(20, 5);
    expect(s.sd).toBeCloseTo(10, 5);
    expect([s.min, s.max]).toEqual([10, 30]);
  });

  it("stays inside the image and has no values when empty", () => {
    expect(ellipseStats(plane, rows, cols, [[-10, -10], [10, 10]])!.n).toBeGreaterThan(0);
    expect(ellipseStats(plane, rows, cols, [[5, 5], [5, 20]])).toBeNull();
  });

  it("area and caption match the desktop", () => {
    const P: [number, number][] = [[0, 0], [20, 40]];
    expect(ellipseAreaMm2(P, 0.5, 0.5)).toBeCloseTo(Math.PI * 5 * 10);
    expect(fmtArea(ellipseAreaMm2(P, 0.5, 0.5))).toBe("1.57 cm²");
    const lines = measureLines({ kind: "ellipse", pts: P }, 0.5, 0.5, true, " HU", { n: 9, mean: 35.26, sd: 6.04, min: 12, max: 60 });
    expect(lines).toEqual(["Area: 1.57 cm²", "Ø 2.00 × 1.00 cm", "Mean: 35.3 HU  SD: 6.0", "Min: 12  Max: 60"]);
  });
});
