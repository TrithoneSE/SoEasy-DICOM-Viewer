import { describe, expect, it } from "vitest";
import { BONE_OFFSET, encodeCT, encodeWith, maskAt, VEIN_MIN_HU, VEIN_OFFSET, type VesselMasks } from "../src/vr/bone";
import { classifyPhases, compete, isPhasePair, phaseOfName, sinusMasks } from "../src/vr/phases";
import { measureLines, polylineMm } from "../src/view/measure";
import type { Volume } from "../src/dicom/volume";

describe("phase names", () => {
  it("reads the series description", () => {
    expect(phaseOfName("CTA Carotis/COW 0.625mm VENA")).toBe("venous");
    expect(phaseOfName("CTA Carotis/COW 0.625mm ARTER")).toBe("arterial");
    expect(phaseOfName("PRE KONTRAS 0.625 mm NK")).toBe("native");
    expect(phaseOfName("Dose Report")).toBeNull();
    expect(phaseOfName("Ventricle")).toBeNull();
  });
});

describe("a plain series next to its contrast series", () => {
  function head(contrast: boolean): Volume {
    const cols = 100, rows = 100, n = 150, data = new Int16Array(cols * rows * n).fill(-1000);
    for (let z = 0; z < n; z++) for (let y = 0; y < rows; y++) for (let x = 0; x < cols; x++) {
      const r = Math.hypot(y - 50, x - 50, (z - 40) * 0.8), i = (z * rows + y) * cols + x;
      if (r < 48) data[i] = 1500;
      if (r < 44) data[i] = 35;
      // vessels over > 0.5 % of the inside, so they are what the top 0.5 % holds
      if (r < 44 && z > 20 && [35, 50, 65].some((c) => Math.hypot(y - 50, x - c) < 2.5)) data[i] = contrast ? 300 : 45;
    }
    return { data, cols, rows, n, spacing: [1, 1, 1], description: "x" } as unknown as Volume;
  }
  it("is no pair of phases; two contrast series are", () => {
    expect(isPhasePair(head(false), head(true))).toBe(false);
    expect(isPhasePair(head(true), head(true))).toBe(true);
    expect(phaseOfName("Brain 0.625 mm  NC")).toBe("native");
  });
});

describe("one routine contrast series", () => {
  it("traces the sinus against the skull; an artery of the same HU stays with the other vessels", () => {
    const cols = 100, rows = 100, n = 150, data = new Int16Array(cols * rows * n).fill(40);
    const sinus: number[] = [], artery: number[] = [];
    for (let z = 0; z < n; z++) for (let y = 0; y < rows; y++) for (let x = 0; x < cols; x++) {
      const i = (z * rows + y) * cols + x, r = Math.hypot(x - 50, y - 50);
      if (r > 44) data[i] = 1500;
      else if (Math.hypot(x - 88, y - 50) < 6) { data[i] = 160; sinus.push(i); }
      else if (Math.hypot(x - 30, y - 50) < 3) { data[i] = 160; artery.push(i); }
    }
    const vol = { data, cols, rows, n, spacing: [1, 1, 1], description: "Head + CM" } as unknown as Volume;
    const r = sinusMasks(vol, null)!;
    const share = (m: Parameters<typeof maskAt>[0], list: number[]) =>
      list.filter((i) => maskAt(m, i % 100, Math.floor(i / 100) % 100, Math.floor(i / 10000))).length / list.length;
    expect(share(r.veins, sinus)).toBeGreaterThan(0.5);
    expect(share(r.veins, artery)).toBe(0);
    expect(share(r.arteries, artery)).toBeGreaterThan(0.9);
  });
});

describe("compete", () => {
  it("gives a voxel to the front that reaches it first", () => {
    const l = new Uint8Array([1, 0, 0, 0, 2, 3, 0]);
    compete(l, [7, 1, 1]);
    expect([...l]).toEqual([1, 1, 1, 2, 2, 3, 0]);
  });
});

/** Skull shell, an artery bright on the arterial series, a vein bright on the venous one. */
function phantom(): { a: Volume; v: Volume; artery: number[]; vein: number[] } {
  const cols = 60, rows = 60, n = 60;
  const A = new Int16Array(cols * rows * n).fill(40), V = new Int16Array(cols * rows * n).fill(40);
  const at = (x: number, y: number, z: number) => (z * rows + y) * cols + x;
  for (let z = 0; z < n; z++) for (let y = 0; y < rows; y++) for (let x = 0; x < cols; x++) {
    const r = Math.hypot(x - 30, y - 30);
    if (r > 26) A[at(x, y, z)] = V[at(x, y, z)] = 1500;
    else if (Math.hypot(x - 20, y - 30) < 3) { A[at(x, y, z)] = 450; V[at(x, y, z)] = 120; }
    else if (Math.hypot(x - 40, y - 30) < 3) { A[at(x, y, z)] = 70; V[at(x, y, z)] = 250; }
  }
  const vol = (data: Int16Array, description: string) => ({ data, cols, rows, n, spacing: [1, 1, 1], description }) as unknown as Volume;
  return { a: vol(A, "ARTER"), v: vol(V, "VENA"), artery: [20, 30, 30], vein: [40, 30, 30] };
}

describe("classifyPhases", () => {
  it("separates by the time course and keeps bone out of both", () => {
    const { a, v, artery, vein } = phantom();
    const r = classifyPhases(a, v, null);
    expect(maskAt(r.arteries, artery[0], artery[1], artery[2])).toBe(true);
    expect(maskAt(r.veins, artery[0], artery[1], artery[2])).toBe(false);
    expect(maskAt(r.veins, vein[0], vein[1], vein[2])).toBe(true);
    expect(maskAt(r.arteries, vein[0], vein[1], vein[2])).toBe(false);
    expect(maskAt(r.bone, 2, 30, 30)).toBe(true);
    expect(maskAt(r.arteries, 2, 30, 30)).toBe(false);
  });
});

describe("by the share of enhancement kept", () => {
  function head() {
    const cols = 100, rows = 100, n = 150, N = cols * rows * n;
    const A = new Int16Array(N).fill(40), V = new Int16Array(N).fill(40);
    const weak: number[] = [], artery: number[] = [], sac: number[] = [];
    for (let z = 0; z < n; z++) for (let y = 0; y < rows; y++) for (let x = 0; x < cols; x++) {
      const i = (z * rows + y) * cols + x, r = Math.hypot(x - 50, y - 50);
      if (r > 44) A[i] = V[i] = 1500;
      else if (Math.hypot(x - 90, y - 50) < 2.2) { A[i] = 210; V[i] = 125; weak.push(i); }
      else if (Math.hypot(x - 30, y - 50) < 3) { A[i] = 450; V[i] = 120; artery.push(i); }
      else if (Math.hypot(z - 40, y - 50, x - 38) < 7) { A[i] = 450; V[i] = 260; sac.push(i); }
    }
    const vol = (data: Int16Array) => ({ data, cols, rows, n, spacing: [1, 1, 1], description: "x" }) as unknown as Volume;
    return { a: vol(A), v: vol(V), weak, artery, sac };
  }
  const at = (m: Parameters<typeof maskAt>[0], list: number[]) =>
    list.filter((i) => maskAt(m, i % 100, Math.floor(i / 100) % 100, Math.floor(i / 10000))).length / list.length;

  it("a weak thin sinus is a vein, the artery and a slow aneurysm stay arteries", () => {
    const { a, v, weak, artery, sac } = head();
    const r = classifyPhases(a, v, null);
    expect(at(r.veins, weak)).toBeGreaterThan(0.6);
    expect(at(r.arteries, artery)).toBeGreaterThan(0.9);
    expect(at(r.arteries, sac)).toBeGreaterThan(0.9);
  });
});

describe("a patient who moved between the phases", () => {
  /** The venous series with everything 3 voxels further along x. */
  function moved() {
    const { a, v } = phantom();
    const { cols, rows, n } = v;
    const data = new Int16Array(v.data.length);
    for (let z = 0; z < n; z++) for (let y = 0; y < rows; y++) for (let x = 0; x < cols; x++) {
      data[(z * rows + y) * cols + Math.min(cols - 1, x + 3)] = (v.data as Int16Array)[(z * rows + y) * cols + x];
    }
    return { a, v: { ...v, data } as unknown as Volume, i: (x: number) => (30 * rows + 30) * cols + x };
  }

  it("on the venous window, each vessel lands where the venous series has it", () => {
    const { a, v, i } = moved();
    const r = classifyPhases(a, v, null, 500, true);
    expect(r.shiftMm).toBeCloseTo(3);
    expect(maskAt(r.arteries, 23, 30, 30)).toBe(true);
    expect(maskAt(r.veins, 43, 30, 30)).toBe(true);
    expect(encodeWith(r, 0, true, false, i(23))).toBe(450);
    expect(encodeWith(r, 0, false, true, i(43))).toBe(VEIN_OFFSET + 250);
  });

  it("on the arterial window, likewise", () => {
    const { a, v, i } = moved();
    const r = classifyPhases(a, v, null, 500, false);
    expect(maskAt(r.arteries, 20, 30, 30)).toBe(true);
    expect(maskAt(r.veins, 40, 30, 30)).toBe(true);
    expect(encodeWith(r, 0, true, false, i(20))).toBe(450);
    expect(encodeWith(r, 0, false, true, i(40))).toBe(VEIN_OFFSET + 250);
  });
});

describe("a native scan that does not quite match", () => {
  /** Skull, and a thick sinus against it that is full on the arterial series and falls by the venous one. */
  function head() {
    const cols = 100, rows = 100, n = 150, N = cols * rows * n;
    const A = new Int16Array(N).fill(40), V = new Int16Array(N).fill(40), P = new Int16Array(N).fill(40);
    const sinus: number[] = [], outer: number[] = [];
    for (let z = 0; z < n; z++) for (let y = 0; y < rows; y++) for (let x = 0; x < cols; x++) {
      const i = (z * rows + y) * cols + x, r = Math.hypot(x - 50, y - 50);
      if (r > 44) A[i] = V[i] = P[i] = 1500;
      else if (Math.hypot(x - 88, y - 50) < 6) {
        A[i] = 350;
        V[i] = 170;
        sinus.push(i);
        if (x >= 92) outer.push(i);
      }
    }
    const vol = (data: Int16Array) => ({ data, cols, rows, n, spacing: [1, 1, 1], description: "x" }) as unknown as Volume;
    return { a: vol(A), v: vol(V), p: vol(P), P, sinus, outer };
  }
  const veinShare = (r: ReturnType<typeof classifyPhases>, list: number[]) => {
    const on = list.filter((i) => maskAt(r.veins, i % 100, Math.floor(i / 100) % 100, Math.floor(i / 10000)));
    return on.length / list.length;
  };

  it("keeps a sinus when the native inner table falls on part of it", () => {
    const { a, v, p, P, sinus, outer } = head();
    for (const i of outer) P[i] = 400;
    expect(veinShare(classifyPhases(a, v, p), sinus)).toBeGreaterThan(0.6);
  });

  it("but a piece that was bright before contrast all along is no sinus", () => {
    const { a, v, p, P, sinus } = head();
    for (const i of sinus) P[i] = 400;
    expect(veinShare(classifyPhases(a, v, p), sinus)).toBeLessThan(0.2);
  });
});

describe("bone outside the bone area, with both phases", () => {
  it("stays bone (mirrored), not a hidden vessel value; an artery keeps its HU", () => {
    const { a, v } = phantom();
    const r = classifyPhases(a, v, null);
    expect(encodeWith(r, 1500, false, false, 0, true)).toBe(-BONE_OFFSET - 1500);
    expect(encodeWith(r, 1500, false, true, 0, true)).toBe(-BONE_OFFSET - 1500);
    expect(encodeWith(r, 1500, false, false, 0, false)).toBe(VEIN_MIN_HU - 1);
  });
});

describe("open polygon", () => {
  it("adds up its segments in mm", () => {
    const pts: [number, number][] = [[0, 0], [0, 10], [10, 10]];
    expect(polylineMm(pts, 0.5, 0.5)).toBeCloseTo(10);
    expect(measureLines({ kind: "polyline", pts }, 0.5, 0.5, true, "", null)).toEqual(["1.00 cm"]);
    expect(measureLines({ kind: "polyline", pts: [[0, 0]] }, 0.5, 0.5, true, "", null)).toEqual([]);
  });
});

describe("the bone area's edge, voxel by voxel (both phases)", () => {
  // one row: bone, its partial-volume edge, brain, a lumen; the arterial series as given, the venous one is the row
  const row = [1200, 300, 40, 40, 250, 40, 40, 40];
  const masks = (art: number[]): VesselMasks => ({
    arteries: { bits: new Uint32Array(1), nx: 8, ny: 1, nz: 1, f: 1 }, veins: { bits: new Uint32Array(1), nx: 8, ny: 1, nz: 1, f: 1 },
    coreHu: 300, artData: art, venData: row, boneHu: 500,
  });
  const at = (m: VesselMasks, i: number, inArea: boolean, isArtery: boolean, isVein: boolean, onEdge: boolean) =>
    encodeCT(m, row, 8, 8, i, row[i], true, inArea, isArtery, isVein, onEdge);

  it("keeps the bone's edge with the bone, even when a move between the scans changed it", () => {
    expect(at(masks(row), 1, false, false, true, false)).toBe(-BONE_OFFSET - 300);
    expect(at(masks([1200, 500, 40, 40, 250, 40, 40, 40]), 1, false, false, true, false)).toBe(-BONE_OFFSET - 300);
  });

  it("takes a lumen out of a bone-area cell beside a vessel, by its own time course", () => {
    const venous = masks([1200, 300, 40, 40, 120, 40, 40, 40]);
    expect(at(venous, 4, true, false, false, true)).toBe(VEIN_OFFSET + 250);
    expect(at(masks([1200, 300, 40, 40, 400, 40, 40, 40]), 4, true, false, false, true)).toBe(400);   // artery: its arterial HU
    expect(at(venous, 4, true, false, false, false)).toBe(-BONE_OFFSET - 250);    // not beside a vessel: the cell
  });

  it("codes only a vein's own voxels as vein, and bone in an artery's cell as bone", () => {
    expect(at(masks(row), 3, false, false, true, false)).toBe(40);
    expect(at(masks(row), 3, false, false, true, false)).toBeLessThan(VEIN_MIN_HU);
    expect(at(masks(row), 0, false, true, false, false)).toBe(-BONE_OFFSET - 1200);
  });
});
