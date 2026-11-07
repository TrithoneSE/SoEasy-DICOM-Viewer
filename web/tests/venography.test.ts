import { describe, expect, it } from "vitest";
import { BONE_OFFSET, boneRegion, maskAt, VEIN_OFFSET } from "../src/vr/bone";
import { cavity, encodeVeno, HIDDEN_HU, linesUp, MATCH_MIN, plainFor, venoIndex, venoMasks } from "../src/vr/venography";
import type { Volume } from "../src/dicom/volume";

// a head as tests/test_venography.py: skull with an open fontanelle at the top, brain, scalp outside, a sinus along the
// inner table under the fontanelle and a vessel outside the skull (160 HU with contrast, 45 without)
const NZ = 140, NY = 120, NX = 120;
function head(contrast: boolean, shift = 0): { vol: Volume; sinus: number[]; outside: number[]; skull: number[] } {
  const data = new Int16Array(NZ * NY * NX).fill(-1000);
  const sinus: number[] = [], outside: number[] = [], skull: number[] = [];
  for (let z = 0; z < NZ; z++) for (let y = 0; y < NY; y++) for (let x = 0; x < NX; x++) {
    const i = (z * NY + y) * NX + x, Z = z - shift;
    const r = Math.hypot(y - 60, x - 60, Z - 60);
    let v = -1000;
    if (r < 56) v = 40;
    if (r < 52) v = 1500;
    if (r < 48) v = 35;
    if (r >= 48 && r < 52 && Z > 100 && Math.abs(x - 60) < 9 && y < 60) v = 40;      // the fontanelle
    if (v === 1500) skull.push(i);
    if (r >= 44 && r < 48 && Math.abs(x - 60) < 3 && Z > 70) { v = contrast ? 160 : 45; sinus.push(i); }
    if (v < 0 && Math.hypot(y - 60, x - 60) < 12 && Z < 14 && Z >= 0) v = 40;            // the neck, around the vessel outside
    if (Math.hypot(y - 60, x - 60) < 3 && Z < 6 && Z >= 0) { v = contrast ? 160 : 45; outside.push(i); }
    data[i] = v;
  }
  const vol = { data, n: NZ, rows: NY, cols: NX, spacing: [1, 1, 1], description: "Head + CM", modality: "CT", space: "f",
    direction: [1, 0, 0, 0, 1, 0, 0, 0, 1], positions: Array.from({ length: NZ }, (_, z) => [0, 0, z]),
    valueRange: () => [-1000, 1500] } as unknown as Volume;
  return { vol, sinus, outside, skull };
}
const at = (i: number): [number, number, number] => [i % NX, Math.floor(i / NX) % NY, Math.floor(i / (NX * NY))];

describe("venography (contrast head CT)", () => {
  const cm = head(true), plain = head(false);
  const { cav, plates } = cavity(cm.vol);

  it("finds the skull's cavity with its gaps closed: the sinus under an open fontanelle is inside, the neck vessel not", () => {
    const inside = cm.sinus.filter((i) => maskAt(cav, ...at(i))).length;
    expect(inside / cm.sinus.length).toBeGreaterThan(0.9);
    expect(cm.outside.some((i) => maskAt(cav, ...at(i)))).toBe(false);
  });

  it("puts the sinus against the skull in the cavity's domain, the skull in the bone domain, and keeps the vessel outside", () => {
    const bone = boneRegion(cm.vol, 500, 3, 1.5, 1);
    const m = venoMasks(cm.vol, cav, bone, plain.vol.data, 1, 1, plates);
    const value = (i: number) => encodeVeno(m, Math.max(-1024, cm.vol.data[i]), venoIndex(m, ...at(i)), true);
    // the voxel touching the skull has bone beside it on the plain scan too: it stays bone (1 of the 4 here)
    expect(cm.sinus.filter((i) => value(i) >= VEIN_OFFSET + 60).length / cm.sinus.length).toBeGreaterThan(0.65);
    expect(cm.skull.filter((i) => value(i) <= -BONE_OFFSET - 500).length / cm.skull.length).toBeGreaterThan(0.95);
    expect(cm.outside.every((i) => value(i) === 160)).toBe(true);
    const brain = (NZ / 2 * NY + 60) * NX + 60;
    expect(value(brain)).toBe(VEIN_OFFSET + 35);
  });

  it("hides what is bright but no vessel outside the skull, and wraps the cavity's domain in its 0 HU", () => {
    const m = venoMasks(cm.vol, cav, null, plain.vol.data, 1);
    const skullOutside = cm.skull.find((i) => !maskAt(cav, ...at(i)) && !(m.layer[venoIndex(m, ...at(i)) >>> 5] & (1 << (venoIndex(m, ...at(i)) & 31))))!;
    expect(encodeVeno(m, 1500, venoIndex(m, ...at(skullOutside)), false)).toBe(HIDDEN_HU);
    let layered = 0, apart = 0;
    for (let k = 0; k < m.nx * m.ny * m.nz; k++) {
      if (!(m.layer[k >>> 5] & (1 << (k & 31)))) continue;
      layered++;
      const near = [1, -1, m.nx, -m.nx, m.nx * m.ny, -m.nx * m.ny].some((s) => { const j = k + s; return j >= 0 && (m.inner[j >>> 5] & (1 << (j & 31))) !== 0; });
      if (!near || (m.inner[k >>> 5] & (1 << (k & 31)))) apart++;
    }
    expect(layered).toBeGreaterThan(0);
    expect(apart).toBe(0);
  });

  it("lines up a plain scan of a patient who moved (10° and a few mm): register finds the motion", () => {
    // a bone ridge on each scan makes the skull asymmetric; the plain scan's geometry turned and shifted
    const ridge = (v: Volume) => {
      const d = v.data as Int16Array;
      for (let z = 20; z < 45; z++) for (let y = 100; y < 110; y++) for (let x = 35; x < 60; x++) d[(z * NY + y) * NX + x] = 1500;
      for (let z = 90; z < 100; z++) for (let y = 15; y < 22; y++) for (let x = 80; x < 95; x++) d[(z * NY + y) * NX + x] = 1500;
    };
    const a = head(true), b = head(false);
    ridge(a.vol);
    ridge(b.vol);
    const t = (10 * Math.PI) / 180, c = Math.cos(t), s = Math.sin(t);
    const D = [c, -s, 0, s, c, 0, 0, 0, 1];
    const moved = { ...b.vol, direction: D, positions: Array.from({ length: NZ }, (_, z) => [6, -4, 5 + z]) } as unknown as Volume;
    const r = plainFor(moved, a.vol);
    expect(r.moved).not.toBeNull();
    expect(r.match!).toBeGreaterThanOrEqual(MATCH_MIN);
    expect(r.data).not.toBeNull();
  }, 60000);

  it("uses a plain scan only where it lines up: one that moved is left out", () => {
    expect(linesUp(plain.vol.data, cm.vol)).toBeGreaterThan(MATCH_MIN);
    expect(linesUp(head(false, 12).vol.data, cm.vol)).toBeLessThan(MATCH_MIN);
  });
});
