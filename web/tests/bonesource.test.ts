import { describe, expect, it } from "vitest";
import { boneRegion, boxMax, maskAt, notVesselHu, NOT_VESSEL_HU, plainBone, slabPlain, THIN_BONE_HU } from "../src/vr/bone";
import { BONE_CURVE_FROM_HU, boneCurveFrom, builtinPresets, fitBoneCurves, unchangedBuiltin } from "../src/vr/tf";
import type { Volume } from "../src/dicom/volume";

/** As tests/test_bone_source.py: soft tissue 40, noisy dense bone (1200), a thin young plate (300) beside it, a vessel
 *  along the plate (200 with contrast, 40 without), a bright piece without a cortical core (600 on both), and tissue at
 *  140 HU on both. Seeded noise. */
function scan(contrast: boolean, seed: number): Int16Array {
  const nx = 80, ny = 80, nz = 60, d = new Int16Array(nx * ny * nz);
  let s = seed;
  const noise = (sd: number) => {
    s = (s * 1103515245 + 12345) % 2147483648;
    return ((s / 2147483648) * 2 - 1) * sd * 1.7;
  };
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
    const i = (z * ny + y) * nx + x;
    let v = 40 + noise(8);
    if (z >= 10 && z < 50 && y >= 10 && y < 30 && x >= 10 && x < 30) v = 1200 + noise(60);
    if (z >= 10 && z < 50 && y >= 10 && y < 30 && x >= 34 && x < 37) v = 300;
    if (z >= 10 && z < 50 && y >= 14 && y < 17 && x >= 38 && x < 41) v = contrast ? 200 : 40;
    if (z >= 20 && z < 40 && y >= 50 && y < 62 && x >= 50 && x < 62) v = 600;
    if (z >= 20 && z < 40 && y >= 50 && y < 60 && x >= 10 && x < 20) v = 140;
    d[i] = Math.round(v);
  }
  return d;
}

const vol = (data: Int16Array, description: string) =>
  ({ data, cols: 80, rows: 80, n: 60, spacing: [1, 1, 1], description, modality: "CT" }) as unknown as Volume;
const share = (r: ReturnType<typeof boneRegion>, z: [number, number], y: [number, number], x: [number, number]) => {
  let hit = 0, all = 0;
  for (let k = z[0]; k < z[1]; k++) for (let j = y[0]; j < y[1]; j++) for (let i = x[0]; i < x[1]; i++, all++) if (maskAt(r, i, j, k)) hit++;
  return hit / all;
};

describe("the bone lock's bone source (dcv/segment.py bone_source)", () => {
  const cm = scan(true, 1), before = scan(false, 2);
  it("from the plain scan: thin bone is bone, the vessel against it not, dense bone has no hole", () => {
    const r = boneRegion(vol(cm, "Head + CM"), 500, 3, 1.5, 1, "plain", before);
    expect(share(r, [10, 50], [10, 30], [34, 37])).toBeGreaterThan(0.99);
    expect(share(r, [10, 50], [14, 17], [38, 41])).toBeLessThan(0.05);
    expect(share(r, [12, 48], [12, 28], [12, 28])).toBe(1);
    expect(share(r, [20, 40], [50, 62], [50, 62])).toBeGreaterThan(0.99);
    // what did not get brighter (the 140 HU tissue) is not enhanced: stored under the vessel curves by renderer.fill
    const enh = { ...r, bits: r.enhanced! };
    expect(share(enh, [22, 38], [52, 58], [12, 18])).toBe(0);
    expect(share(enh, [12, 48], [15, 16], [39, 40])).toBeGreaterThan(0.9);
    expect(notVesselHu(null)).toBe(NOT_VESSEL_HU);
  });
  it("without it: a routine contrast CT keeps a bright piece without a cortical core as bone, a CTA does not", () => {
    expect(share(boneRegion(vol(cm, "Head + CM"), 500, 3, 1.5, 1, "ct"), [20, 40], [50, 62], [50, 62])).toBeGreaterThan(0.99);
    expect(share(boneRegion(vol(cm, "CTA ARTERI"), 500, 3, 1.5, 1, "cta"), [20, 40], [50, 62], [50, 62])).toBeLessThan(0.05);
  });
  it("a newborn's skull: its plates (given by shape) are bone from 150 HU, the CTA rule is off", () => {
    const plates = new Uint8Array(80 * 80 * 60);
    for (let z = 10; z < 50; z++) for (let y = 10; y < 30; y++) for (let x = 34; x < 37; x++) plates[(z * 80 + y) * 80 + x] = 1;
    const young = new Int16Array(cm);
    for (let i = 0; i < young.length; i++) if (plates[i]) young[i] = 200;          // a plate of 200 HU, under the threshold
    const r = boneRegion(vol(young, "AX THIN ARTERY"), 500, 3, 1.5, 1, "thin", null, plates);
    expect(share(r, [10, 50], [10, 30], [34, 37])).toBeGreaterThan(0.99);
    expect(share(r, [20, 40], [50, 62], [50, 62])).toBeGreaterThan(0.99);
    expect(share(boneRegion(vol(young, "x"), 500, 3, 1.5, 1, "ct"), [12, 48], [10, 30], [35, 36])).toBeLessThan(0.05);
    expect(boneCurveFrom("thin")).toBe(THIN_BONE_HU);
    expect(boneCurveFrom("cta")).toBeNull();
  });
  it("from the plain scan: a vessel lying on bone keeps its edge against it, the bone stays bone", () => {
    // as tests/test_bone_source.py: a cortical plate (x 10-12) and a vessel on it (x 13-16), 30 cells a side
    const m = 30, t = new Int16Array(m * m * m).fill(40);
    for (let z = 5; z < 25; z++) for (let y = 5; y < 25; y++) for (let x = 10; x < 13; x++) t[(z * m + y) * m + x] = 900;
    const tp = t.slice();
    for (let z = 5; z < 25; z++) for (let y = 10; y < 14; y++) for (let x = 13; x < 17; x++) t[(z * m + y) * m + x] = 220;
    const r = plainBone(t, tp, [m, m, m], [1, 1, 1]);
    const v = { bits: r.bits, nx: m, ny: m, nz: m, f: 1 };
    let edge = 0, plate = 0;
    for (let z = 6; z < 24; z++) {
      for (let y = 11; y < 13; y++) edge += maskAt(v, 13, y, z) ? 1 : 0;
      for (let y = 6; y < 24; y++) for (let x = 10; x < 12; x++) plate += maskAt(v, x, y, z) ? 0 : 1;
    }
    expect(edge).toBe(0);
    expect(plate).toBe(0);
  });
  it("is the same in the worker's form (plainBone on a grid)", () => {
    const a = boneRegion(vol(cm, "x"), 500, 3, 1.5, 1, "plain", before);
    const b = plainBone(cm, before, [80, 80, 60], [1, 1, 1]);
    expect(Array.from(b.bits)).toEqual(Array.from(a.bits));
  });
});

describe("the plain scan slab by slab (dcv/venography.py slab_plain)", () => {
  // thin plates of bone up a column (as cortex is), 1 mm cells; the lowest slab (40 planes) of the plain scan moved along x
  const nx = 40, ny = 40, nz = 100;
  const col = new Int16Array(nx * ny * nz).fill(40);
  for (let zc = 5; zc < nz; zc += 10) for (let z = zc; z < zc + 5; z++) for (let y = 10; y < 30; y++) for (let x = 18; x < 20; x++) {
    col[(z * ny + y) * nx + x] = 900;
  }
  const moved = (shift: number) => {
    const p = col.slice();
    for (let z = 0; z < 40; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
      p[(z * ny + y) * nx + x] = col[(z * ny + y) * nx + ((x - shift + nx) % nx)];
    }
    return p;
  };
  it("finds a slab moved 3 mm by its shift, its bone where the series has it", () => {
    const { plain, ok } = slabPlain(col, moved(3), [nx, ny, nz], [1, 1, 1], [1, 1, 1]);
    expect(Array.from(ok).every((v) => v === 1)).toBe(true);
    for (let z = 0; z < 40; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
      if (col[(z * ny + y) * nx + x] < 500) continue;
      const near = [-1, 0, 1].some((dx) => x + dx >= 0 && x + dx < nx && plain[(z * ny + y) * nx + x + dx] >= 500);
      expect(near).toBe(true);
    }
  });
  it("leaves a slab moved beyond its reach out, the others in", () => {
    const { ok } = slabPlain(col, moved(10), [nx, ny, nz], [1, 1, 1], [1, 1, 1]);
    expect(Array.from(ok.subarray(0, 40)).every((v) => v === 0)).toBe(true);
    expect(Array.from(ok.subarray(40)).every((v) => v === 1)).toBe(true);
  });
});

describe("box maximum", () => {
  it("maximum over the box around a cell, along each axis", () => {
    const d = new Int16Array(5 * 5 * 5);
    d[62] = 5;                                   // the centre of 5 × 5 × 5
    const m = boxMax(d, [5, 5, 5], [1, 1, 1]);
    let set = 0;
    for (let i = 0; i < m.length; i++) if (m[i] === 5) set++;
    expect(set).toBe(27);
    expect(m[62 - 25 - 5 - 1]).toBe(5);
    expect(m[62 - 2]).toBe(0);
  });
});

describe("bone curves for a bone area from the plain scan", () => {
  it("start at BONE_CURVE_FROM_HU, in order; the venography keeps its own", () => {
    for (const p of builtinPresets()) {
      const f = fitBoneCurves(p);
      for (const c of f.curves) {
        const xs = c.points.map((q) => q.x);
        expect(xs).toEqual([...xs].sort((u, v) => u - v));
        if (c.layer === "bone" && !p.venography) expect(c.points[0].x).toBeLessThanOrEqual(BONE_CURVE_FROM_HU);
      }
      if (p.venography) expect(f).toEqual(p);
    }
    const vb = builtinPresets()[1];
    expect(unchangedBuiltin(vb)).toBe(true);
    expect(unchangedBuiltin(fitBoneCurves(vb))).toBe(false);
  });
});
