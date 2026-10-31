import { describe, expect, it } from "vitest";
import type { Volume } from "../src/dicom/volume";
import { ARTERY_TOP_HU, boneAndVessels, boneRegion, encodeVessel, gridStep, unpack, VEIN_MIN_HU, VEIN_OFFSET, vesselMasks, vesselsAwayFromBone, vesselsByPieces } from "../src/vr/bone";
import { buildTable, builtinPresets, fromDict, usesVeins } from "../src/vr/tf";

const NZ = 100, NY = 120, NX = 120;

/** tests/test_cta_pan_seed_collage.py synthetic_cta: brain, skull, artery from x = start to an aneurysm, vein, thin plate. */
function syntheticCta(start = 16) {
  const d = new Int16Array(NZ * NY * NX).fill(-1000);
  const sphere = new Uint8Array(d.length), artery = new Uint8Array(d.length), vein = new Uint8Array(d.length);
  for (let z = 0; z < NZ; z++) for (let y = 0; y < NY; y++) for (let x = 0; x < NX; x++) {
    const i = (z * NY + y) * NX + x;
    const r = Math.hypot(z - 50, y - 60, x - 60);
    if (r < 48) d[i] = 40;
    if (r >= 44 && r < 48) d[i] = 1400;
    if (Math.hypot(z - 50, y - 60) < 2.5 && x > start && x <= 60) {
      d[i] = 600;
      artery[i] = 1;
    }
    if (Math.hypot(z - 50, y - 60, x - 62) < 5) {
      d[i] = 650;
      sphere[i] = 1;
    }
    if (Math.hypot(z - 50, x - 90) < 2.5 && y > 40 && y < 80) {
      d[i] = 150;
      vein[i] = 1;
    }
    if (Math.abs(x - 62) < 1 && y >= 12 && y <= 18 && Math.abs(z - 50) < 6 && r < 48 && r >= 42) d[i] = 600;
  }
  return { d, sphere, artery, vein };
}

function fakeVol(d: Int16Array, spacing: [number, number, number] = [1, 1, 1]): Volume {
  return { data: d, rows: NY, cols: NX, n: NZ, spacing } as unknown as Volume;
}

/** Same digest as the reference script: how many voxels, and the sum of their indices modulo a prime. */
function digest(m: Uint8Array): [number, number] {
  let n = 0, s = 0;
  for (let i = 0; i < m.length; i++) {
    if (m[i]) {
      n++;
      s += i % 1000003;
    }
  }
  return [n, s];
}

// produced by the desktop's dcv/segment.py (bone_and_vessels, bone_region, artery_mask, vein_mask) on the same volumes
const DESKTOP = {
  synthetic: { bone: [106256, 46050530944], vessels: [1304, 948329992], region: [267225, 114146931202], art: [2749, 1999200287], vein: [819, 595650510], core: 390, pieces: [0, 0], away: [1304, 948329992] },
  free: { bone: [106214, 46019987809], vessels: [1178, 856700083], region: [267125, 114074209246], art: [2500, 1818122330], vein: [819, 595650510], core: 390, pieces: [1178, 856700083], away: [1178, 856700083] },
  aniso: { bone: [106277, 46065802543], vessels: [1283, 933058393], region: [293925, 131711321136], art: [2704, 1966475456], vein: [691, 502557284], core: 390, pieces: [0, 0], away: [1283, 933058393] },
};

describe("bone and vessels: voxel for voxel the desktop's", () => {
  const cases: [keyof typeof DESKTOP, number, [number, number, number]][] = [["synthetic", 16, [1, 1, 1]], ["free", 24, [1, 1, 1]], ["aniso", 16, [2, 0.5, 0.5]]];
  for (const [name, start, spacing] of cases) {
    it(name, () => {
      const ref = DESKTOP[name];
      const { d } = syntheticCta(start);
      const sp: [number, number, number] = [spacing[2], spacing[1], spacing[0]];
      const dims: [number, number, number] = [NX, NY, NZ];
      const high = new Uint8Array(d.length), core = new Uint8Array(d.length);
      for (let i = 0; i < d.length; i++) {
        high[i] = d[i] >= 500 ? 1 : 0;
        core[i] = d[i] >= 1000 ? 1 : 0;
      }
      expect(digest(vesselsByPieces(high, core, dims, sp))).toEqual(ref.pieces);
      expect(digest(vesselsAwayFromBone(high, core, dims, sp))).toEqual(ref.away);
      const { bone, vessels } = boneAndVessels(d, dims, sp, 500);
      expect(digest(bone)).toEqual(ref.bone);
      expect(digest(vessels)).toEqual(ref.vessels);
      const vol = fakeVol(d, spacing);
      const region = boneRegion(vol, 500, 3, 1.5, 1);
      expect(digest(unpack(region.bits, d.length))).toEqual(ref.region);
      const m = vesselMasks(vol, region);
      expect(m.coreHu).toBeCloseTo(ref.core, 5);
      expect(digest(unpack(m.arteries.bits, d.length))).toEqual(ref.art);
      expect(digest(unpack(m.veins.bits, d.length))).toEqual(ref.vein);
    });
  }
});

describe("what the owner saw on real CTA", () => {
  const { d, sphere, artery, vein } = syntheticCta();
  const dims: [number, number, number] = [NX, NY, NZ];

  it("an aneurysm above the bone threshold is a vessel, out of the bone area; the skull stays bone", () => {
    const { bone, vessels } = boneAndVessels(d, dims, [1, 1, 1], 500);
    const area = unpack(boneRegion(fakeVol(d), 500, 3, 1.5, 1).bits, d.length);
    let inBone = 0, isVessel = 0, n = 0, shell = 0, shellBone = 0;
    for (let i = 0; i < d.length; i++) {
      if (sphere[i]) {
        n++;
        inBone += bone[i] + area[i];
        isVessel += vessels[i];
      }
      if (d[i] === 1400) {
        shell++;
        shellBone += bone[i];
      }
    }
    expect(inBone).toBe(0);
    expect(isVessel).toBe(n);
    expect(shellBone / shell).toBeGreaterThan(0.99);
  });

  it("the half grid of phones finds them too", () => {
    const area = boneRegion(fakeVol(d), 500, 3, 1.5, 2);
    expect(area.f).toBe(2);
    const m = vesselMasks(fakeVol(d), area);
    const at = (r: typeof area, x: number, y: number, z: number) => {
      const i = (z * r.ny + Math.floor(y / 2)) * r.nx + Math.floor(x / 2);
      return (r.bits[i >>> 5] & (1 << (i & 31))) !== 0;
    };
    expect(at(area, 62, 60, 50)).toBe(false);          // aneurysm centre
    expect(at(area, 60, 60, 4)).toBe(true);            // skull
    expect(at(m.arteries, 62, 60, 50)).toBe(true);
    expect(at(m.veins, 90, 60, 50)).toBe(true);
    expect(at(m.arteries, 90, 60, 50)).toBe(false);
  });

  it("arteries and veins are told apart", () => {
    const m = vesselMasks(fakeVol(d), boneRegion(fakeVol(d), 500, 3, 1.5, 1));
    const art = unpack(m.arteries.bits, d.length), vn = unpack(m.veins.bits, d.length);
    let sa = 0, ss = 0, va = 0, vv = 0, nv = 0, aa = 0, na = 0;
    for (let i = 0; i < d.length; i++) {
      if (sphere[i]) {
        ss++;
        sa += art[i];
      }
      if (vein[i]) {
        nv++;
        va += art[i];
        vv += vn[i];
      }
      if (artery[i] && i % NX > 22) {
        na++;
        aa += art[i];
      }
    }
    expect(sa).toBe(ss);
    expect(va).toBe(0);
    expect(vv / nv).toBeGreaterThan(0.9);
    expect(aa / na).toBeGreaterThan(0.9);
  });

  it("another core can be chosen", () => {
    expect(vesselMasks(fakeVol(d), null, 300).coreHu).toBe(300);
  });

  it("uses the smaller grid earlier on phones and tablets", () => {
    expect(gridStep(78e6, true)).toBe(2);
    expect(gridStep(78e6, false)).toBe(1);
    expect(gridStep(157e6, false)).toBe(2);
    expect(gridStep(10e6, true)).toBe(1);
  });
});

describe("stored values and curves", () => {
  it("veins in their own domain, other bright voxels transparent, arteries clamped", () => {
    expect(encodeVessel(150, false, true)).toBe(VEIN_OFFSET + 150);
    expect(encodeVessel(5000, false, true)).toBe(VEIN_OFFSET + 3000);
    expect(encodeVessel(300, false, false)).toBe(VEIN_MIN_HU - 1);
    expect(encodeVessel(700, true, false)).toBe(ARTERY_TOP_HU);
    expect(encodeVessel(300, true, false)).toBe(300);
    expect(encodeVessel(40, false, false)).toBe(40);
  });

  it("a curve named Vein… or Vena… is a vein curve, and JSON keeps the layer", () => {
    const pts = [{ x: 0, y: 0 }, { x: 1, y: 1 }];
    expect(fromDict({ name: "t", curves: [{ name: "Veins", points: pts }] }).curves[0].layer).toBe("vein");
    expect(fromDict({ name: "t", curves: [{ name: "Vena cava", points: pts }] }).curves[0].layer).toBe("vein");
    expect(fromDict({ name: "t", curves: [{ name: "x", layer: "vein", points: pts }] }).curves[0].layer).toBe("vein");
  });

  it("presets: only Arteries / Veins uses the layer; points as on the desktop", () => {
    const P = builtinPresets();
    expect(P.filter(usesVeins).map((t) => t.name)).toEqual(["CT Angio - Arteries / Veins + Bone"]);
    const av = P.find(usesVeins)!;
    const xy = (name: string) => av.curves.find((c) => c.name === name)!.points.map((p) => [p.x, p.y]);
    expect(xy("Veins")).toEqual([[100, 0], [140, 0.2], [200, 0.45], [500, 0.45], [600, 0]]);
    expect(xy("Arteries")).toEqual([[110, 0], [170, 0.2], [280, 0.45], [450, 0.45], [500, 0]]);
    const vb = P.find((t) => t.name === "CT Angio - Vessels + Bone")!;
    expect(vb.curves[0].points.map((p) => [p.x, p.y])).toEqual([[130, 0], [190, 0.18], [280, 0.45], [800, 0.45], [1000, 0]]);
  });

  it("table: each layer only in its domain; no vein colour between bone and normal", () => {
    const tf = builtinPresets().find(usesVeins)!;
    const off = 4096, lo = -1024, hi = 3071, x0 = -off - hi, x1 = VEIN_OFFSET + 3000, W = 8192;
    const t = buildTable(tf, { x0, x1, width: W, lo, bone: true, boneOffset: off, vein: true, veinOffset: VEIN_OFFSET, gain: 1 });
    const cell = (v: number) => {
      const c = Math.round(((v - x0) / (x1 - x0)) * (W - 1)) * 4;
      return [t.data[c], t.data[c + 1], t.data[c + 2], t.data[c + 3]];
    };
    const vein = cell(VEIN_OFFSET + 300), art = cell(300), bone = cell(-off - 900);
    expect(vein[3]).toBeGreaterThan(0.3);
    expect(vein[2]).toBeGreaterThan(vein[0]);           // blue
    expect(art[3]).toBeGreaterThan(0.3);
    expect(art[0]).toBeGreaterThan(art[2]);             // red
    expect(bone[3]).toBeGreaterThan(0.5);
    expect(cell(1500)[3]).toBe(0);                      // above the arteries: nothing is stored there, and no colour
    expect(cell(10000)[3]).toBe(0);
    expect(cell(-2000 + (lo - 1 - -2000) / 2)[3]).toBe(0);    // the gap below the normal domain
  });
});
