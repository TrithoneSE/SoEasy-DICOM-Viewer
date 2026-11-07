import { describe, expect, it } from "vitest";
import { vesselTrees } from "../src/vr/phases";

/** As tests/test_vessel_trees.py: a thick sinus and an artery along x, 14 cells apart, one thin contact between them. */
const nx = 80, ny = 40, nz = 40, N = nx * ny * nz;
const at = (f: (x: number, y: number, z: number) => boolean) => {
  const m = new Uint8Array(N);
  for (let z = 0, i = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++, i++) if (f(x, y, z)) m[i] = 1;
  return m;
};
const sinusF = (x: number, y: number, z: number) => Math.hypot(z - 20, y - 12) <= 3 && x >= 5 && x < 75;
const arteryF = (x: number, y: number, z: number) => Math.hypot(z - 20, y - 26) <= 2.5 && x >= 5 && x < 75;
const bridgeF = (x: number, y: number, z: number) => z === 20 && y > 12 && y < 26 && x === 40;
const share = (m: Uint8Array, sel: Uint8Array) => {
  let a = 0, b = 0;
  for (let i = 0; i < N; i++) if (sel[i]) { b++; a += m[i]; }
  return a / b;
};
const priorOf = (sure: Uint8Array) => (mask: Uint8Array) => {
  const out = new Float32Array(N).fill(0.5);
  for (let i = 0; i < N; i++) if (mask[i] && sure[i]) out[i] = 0.97;
  return out;
};

describe("arteries and veins of one series as two trees (dcv/phases.py vessel_trees)", () => {
  it("a stretch the atlas got wrong takes the label of the vessel it belongs to", () => {
    const traced = at((x, y, z) => sinusF(x, y, z) && (x < 30 || x >= 50));
    const a = at((x, y, z) => (arteryF(x, y, z) && !(x >= 15 && x < 30)) || (sinusF(x, y, z) && x >= 30 && x < 50) || bridgeF(x, y, z));
    const v = at((x, y, z) => (sinusF(x, y, z) || arteryF(x, y, z) || bridgeF(x, y, z)) && !a[(z * ny + y) * nx + x]);
    vesselTrees(a, v, traced, priorOf(at((x, y, z) => arteryF(x, y, z) && x > 60)), [nx, ny, nz], [1, 1, 1]);
    expect(share(v, at(sinusF))).toBe(1);
    expect(share(a, at((x, y, z) => arteryF(x, y, z) && x >= 45))).toBe(1);
    expect(share(a, at(arteryF))).toBeGreaterThan(0.9);
    for (let i = 0; i < N; i++) expect(a[i] & v[i]).toBe(0);
  });
  it("a thin vessel the atlas is sure of seeds nothing: the labels stay", () => {
    const thin = at((x, y, z) => z === 20 && y === 5 && x >= 5 && x < 75);
    const a = new Uint8Array(N), v = thin.slice();
    vesselTrees(a, v, new Uint8Array(N), priorOf(thin), [nx, ny, nz], [1, 1, 1]);
    expect(share(v, thin)).toBe(1);
    expect(a.some((k) => k)).toBe(false);
  });
});
