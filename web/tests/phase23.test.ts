import { describe, expect, it } from "vitest";
import type { Volume } from "../src/dicom/volume";
import { directionFromIop, letter, mulMV, scale } from "../src/geometry";
import { MPRModel } from "../src/view/mpr";
import { boneRegion, BONE_OFFSET, inBone, roundEven } from "../src/vr/bone";
import { cutVoxels, fillLasso } from "../src/vr/cut";
import { contentBox } from "../src/vr/renderer";
import { buildTable, builtinPresets, combine, curveSamples, fromDict, makeTF, RIM_GRAD_MAX, sampleCurve } from "../src/vr/tf";

/** A Volume-like object with data in memory (what MPR and the bone region read). */
function fakeVolume(nx: number, ny: number, nz: number, f: (x: number, y: number, z: number) => number,
  spacing: [number, number, number] = [1, 1, 1], modality = "CT"): Volume {
  const data = new Int16Array(nx * ny * nz);
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) data[(z * ny + y) * nx + x] = f(x, y, z);
  return {
    n: nz, rows: ny, cols: nx, data, spacing, direction: directionFromIop([1, 0, 0, 0, 1, 0]), modality,
    spacingKnown: true, monochrome1: false, patient: "", ww: 400, wl: 40, complete: true,
    frame: (k: number) => data.subarray(k * nx * ny, (k + 1) * nx * ny),
    valueRange: () => [Math.min(...data), Math.max(...data)] as [number, number],
  } as unknown as Volume;
}

describe("transfer functions (dcv/tf_model.py)", () => {
  const tf = builtinPresets()[1];                      // CT Angio - Vessels + Bone

  it("curve sampling: linear between points, 0 outside, colour clamped", () => {
    const c = tf.curves[0];                            // Vessels: (130,0) (190,.18) (280,.45) (380,0)
    const { a, rgb } = sampleCurve(c, Float64Array.from([100, 130, 160, 280, 330, 400]));
    expect(Array.from(a).map((v) => +v.toFixed(4))).toEqual([0, 0, 0.09, 0.45, 0.45, 0.45]);
    expect(rgb[0]).toBeCloseTo(200 / 255);             // left of the curve: first point's colour
  });

  it("merge: opacity = max, colour = opacity-weighted mean", () => {
    const t = makeTF("t", [
      { name: "a", layer: "normal", hide_edges: false, points: [{ x: 0, y: 0.2, color: [1, 0, 0] }, { x: 10, y: 0.2, color: [1, 0, 0] }] },
      { name: "b", layer: "normal", hide_edges: false, points: [{ x: 0, y: 0.6, color: [0, 0, 1] }, { x: 10, y: 0.6, color: [0, 0, 1] }] },
    ]);
    const m = combine(curveSamples(t, Float64Array.from([5]), null), 1, true);
    expect(m.a[0]).toBeCloseTo(0.6);
    expect([m.rgb[0], m.rgb[2]].map((v) => +v.toFixed(3))).toEqual([0.25, 0.75]);
  });

  it("bone lock table: bone curves only in the mirrored domain, the others only outside it", () => {
    const hi = 3071, lo = -1024, W = 4096;
    const x0 = -BONE_OFFSET - hi;
    const t = buildTable(tf, { x0, x1: hi, width: W, lo, bone: true, boneOffset: BONE_OFFSET, gain: 1 });
    const at = (v: number) => {
      const i = Math.round(((v - x0) / (hi - x0)) * (W - 1));
      return t.data[i * 4 + 3];
    };
    expect(t.rows).toBe(1);
    expect(at(-BONE_OFFSET - 900)).toBeGreaterThan(0.6);   // bone area at 900 HU: the bone curve (0.7)
    expect(at(1500)).toBe(0);                                // outside the bone area the bone curve is off
    expect(at(250)).toBeGreaterThan(0.3);                    // vessels outside the bone area
    expect(at(-BONE_OFFSET - 250)).toBe(0);                  // ...but not inside it (no vessel-coloured bone rim)
    expect(at(-2000)).toBe(0);                               // the gap between the domains is transparent
  });

  it("rims hidden: a 2D table that drops those curves above the gradient threshold", () => {
    const tumor = builtinPresets()[6];
    const W = 2048;
    const t = buildTable(tumor, { x0: -1024, x1: 3071, width: W, lo: -1024, bone: false, boneOffset: BONE_OFFSET, gain: 1 }, 64);
    expect(t.rows).toBe(64);
    const col = Math.round(((110 + 1024) / 4095) * (W - 1));          // 110 HU: the enhancing lesion curve
    const alpha = (row: number) => t.data[(row * W + col) * 4 + 3];
    const rowOf = (g: number) => Math.floor((g / RIM_GRAD_MAX) * 64);
    expect(alpha(0)).toBeGreaterThan(0.2);
    expect(alpha(rowOf(30))).toBeCloseTo(alpha(0));                    // below 0.6 × 60 HU: kept
    expect(alpha(rowOf(100))).toBe(0);                                 // above 1.4 × 60 HU: hidden
  });

  it("desktop preset JSON loads (layer guessed from the name when missing)", () => {
    const t = fromDict({ name: "Mine", edge_threshold: 40, curves: [
      { name: "Bone 2", points: [{ x: 300, y: 0, color: [1, 1, 1] }, { x: 900, y: 0.5, color: [1, 1, 0.9] }] },
      { name: "Lesion", hide_edges: true, points: [{ x: 120, y: 0.3 }, { x: 80, y: 0 }] },
    ] });
    expect(t.curves[0].layer).toBe("bone");
    expect(t.curves[1].points.map((p) => p.x)).toEqual([80, 120]);
    expect([t.edge_threshold, t.ambient, t.curves[1].hide_edges]).toEqual([40, 0.15, true]);
  });
});

describe("bone area (dcv/segment.py bone_region)", () => {
  it("closes small gaps in bone and adds a rim", () => {
    // a 'skull' slab at x = 8..9 and x = 12..13 with a 2-voxel gap (diploe) between them
    const vol = fakeVolume(24, 8, 8, (x) => ((x >= 8 && x <= 9) || (x >= 12 && x <= 13) ? 1000 : 0));
    const r = boneRegion(vol, 500, 3, 1.5);
    const at = (x: number) => (inBone(r, x, 4, 4) ? 1 : 0);
    expect(r.f).toBe(1);
    expect([10, 11].map(at)).toEqual([1, 1]);             // gap filled by the 3 mm closing
    expect([5, 6, 7, 14, 15, 16].map(at)).toEqual([0, 1, 1, 1, 1, 0]);   // 1.5 mm rim: round(1.5) = 2 voxels
    expect([roundEven(2.5), roundEven(1.5), roundEven(-0.5), roundEven(2.4)]).toEqual([2, 2, -0, 2]);
    expect(at(2)).toBe(0);
  });
});

describe("MPR (dcv/mpr_window.py)", () => {
  const vol = fakeVolume(5, 5, 5, (x, y, z) => 100 * z + 10 * y + x);

  it("an axial plane through the centre is that slice; letters are radiological", () => {
    const m = new MPRModel(vol);
    const W = 2 * m.N + 1;
    const out = new Float32Array(W * W);
    m.sample("axial", out, false);
    // plane pixel (r, c) → voxel x = 2 + c - N, y = 2 + r - N (up = anterior = -y) at z = 2
    for (const [x, y] of [[0, 0], [4, 1], [2, 2], [3, 4]]) {
      expect(out[(m.N + y - 2) * W + (m.N + x - 2)]).toBeCloseTo(200 + 10 * y + x);
    }
    const [a, b] = m.axes("axial");
    expect([letter(mulMV(m.D, b)), letter(mulMV(m.D, a))]).toEqual(["A", "L"]);
    const [sa, sb] = m.axes("sagittal");
    expect([letter(mulMV(m.D, sb)), letter(mulMV(m.D, sa))]).toEqual(["S", "P"]);
  });

  it("index along a normal and tilting the axes", () => {
    const m = new MPRModel(vol);
    const k = m.index("axial");
    m.setIndex("axial", k + 1);
    expect(m.center[2]).toBeCloseTo(m.vcenter[2] + m.step);
    m.reset(false);
    m.rotate([0, 0, 1], Math.PI / 2);                       // turn the axes in the axial plane by 90°
    const n = m.normal("sagittal");
    expect(Math.abs(n[1])).toBeCloseTo(1);                  // the sagittal plane now faces along y
    const out = new Float32Array((2 * m.N + 1) ** 2);
    m.slabMm = 4;
    m.slabMode = "max";
    m.reset(false);
    m.sample("axial", out, false);
    expect(out[m.N * (2 * m.N + 1) + m.N]).toBeCloseTo(422);   // MIP over z = 0..4 at the centre: z = 4
    expect(scale([1, 1, 1], 2)).toEqual([2, 2, 2]);
  });
});

describe("scalpel (dcv/volume_view.py project_inside)", () => {
  // a 10 × 10 × 4 mm volume seen straight on: clip x = X/5 - 1, clip y = 1 - Y/5 (screen y down), w = 1
  const vp = new Float32Array([0.2, 0, 0, 0, 0, -0.2, 0, 0, 0, 0, 0, 0, -1, 1, 0, 1]);
  const dims: [number, number, number] = [10, 10, 4];
  const bits = (r: Uint32Array) => r.reduce((a, w) => a + (w.toString(2).match(/1/g)?.length ?? 0), 0);

  it("lasso fill: even-odd scanlines with a bounding box", () => {
    const L = fillLasso([[0, 0], [50, 0], [50, 100], [0, 100]], 100, 100);
    expect([L.x0, L.x1, L.y0, L.y1]).toEqual([0, 49, 0, 99]);
    expect(L.mask.reduce((a, v) => a + v, 0)).toBe(5000);
  });

  it("cuts inside through the whole depth, or outside; a second cut removes nothing new", () => {
    const L = fillLasso([[0, 0], [50, 0], [50, 100], [0, 100]], 100, 100);   // left half of the view
    const r = new Uint32Array(Math.ceil(400 / 32));
    expect(cutVoxels(r, dims, [1, 1, 1], vp, L, false)).toEqual({ z0: 0, z1: 4, count: 200 });
    expect(r[0] & 1).toBe(1);                                   // voxel (0, 0, 0) is on the left
    expect(cutVoxels(r, dims, [1, 1, 1], vp, L, false).count).toBe(0);
    const o = new Uint32Array(Math.ceil(400 / 32));
    expect(cutVoxels(o, dims, [1, 1, 1], vp, L, true).count).toBe(200);
    expect(bits(o)).toBe(200);
    expect(o[0] & 1).toBe(0);                                   // keeping the inside leaves (0, 0, 0)
  });
});

describe("less GPU memory: the texture covers the content box only", () => {
  it("air around the body is left out (CT above -600 HU, 3-voxel margin)", () => {
    const vol = fakeVolume(40, 30, 20, (x, y, z) => (x >= 10 && x < 20 && y >= 5 && y < 25 && z >= 4 && z < 16 ? 40 : -1000));
    expect(contentBox(vol, true)).toEqual({ x0: 7, x1: 23, y0: 2, y1: 28, z0: 1, z1: 19 });
    const empty = fakeVolume(4, 4, 4, () => -1000);
    expect(contentBox(empty, true)).toEqual({ x0: 0, x1: 4, y0: 0, y1: 4, z0: 0, z1: 4 });
  });

  it("the scalpel uses the box origin", () => {
    // same straight-on view as above; texture starts 5 mm to the right: its 5 left voxels are at X 5..10 (right half)
    const vp = new Float32Array([0.2, 0, 0, 0, 0, -0.2, 0, 0, 0, 0, 0, 0, -1, 1, 0, 1]);
    const L = fillLasso([[0, 0], [50, 0], [50, 100], [0, 100]], 100, 100);
    const r = new Uint32Array(4);
    expect(cutVoxels(r, [5, 10, 2], [1, 1, 1], vp, L, false, [5, 0, 0]).count).toBe(0);
    expect(cutVoxels(r, [5, 10, 2], [1, 1, 1], vp, L, false, [0, 0, 0]).count).toBe(100);
  });
});
