import { describe, expect, it } from "vitest";
import { dilateMask, erodeMask } from "../src/vr/bone";

/** The byte masks' box dilation as it was (a running count per axis): the reference for the bit version. */
function dilateRef(m: Uint8Array, [nx, ny, nz]: [number, number, number], [rx, ry, rz]: [number, number, number]): Uint8Array {
  let cur = m;
  const pass = (r: number, stride: number, len: number, lines: (l: number) => number, count: number) => {
    const out = new Uint8Array(m.length);
    for (let l = 0; l < count; l++) {
      const start = lines(l);
      let c = 0;
      for (let i = 0; i < Math.min(r, len); i++) c += cur[start + i * stride] ? 1 : 0;
      for (let i = 0; i < len; i++) {
        if (i + r < len) c += cur[start + (i + r) * stride] ? 1 : 0;
        if (i - r - 1 >= 0) c -= cur[start + (i - r - 1) * stride] ? 1 : 0;
        out[start + i * stride] = c > 0 ? 1 : 0;
      }
    }
    cur = out;
  };
  if (rx > 0) pass(rx, 1, nx, (l) => l * nx, ny * nz);
  if (ry > 0) pass(ry, nx, ny, (l) => Math.floor(l / nx) * nx * ny + (l % nx), nx * nz);
  if (rz > 0) pass(rz, nx * ny, nz, (l) => l, nx * ny);
  return cur === m ? Uint8Array.from(m, (v) => (v ? 1 : 0)) : cur;
}

function erodeRef(m: Uint8Array, d: [number, number, number], r: [number, number, number]): Uint8Array {
  const inv = Uint8Array.from(m, (v) => (v ? 0 : 1));
  return Uint8Array.from(dilateRef(inv, d, r), (v) => (v ? 0 : 1));
}

describe("bit masks (bone area, sinus tracing)", () => {
  it("dilate and erode the same cells as the byte masks did, at any width and radius", () => {
    let s = 11;
    const rnd = () => ((s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
    for (const [nx, ny, nz] of [[64, 20, 9], [37, 13, 7], [100, 3, 4], [5, 31, 6], [33, 33, 33]] as [number, number, number][]) {
      for (const density of [0.02, 0.3, 0.9]) {
        const m = Uint8Array.from({ length: nx * ny * nz }, () => (rnd() < density ? 1 : 0));
        for (const r of [[1, 1, 1], [3, 0, 2], [0, 4, 0], [9, 2, 5], [40, 1, 1], [0, 0, 0]] as [number, number, number][]) {
          const diff = (a: Uint8Array, b: Uint8Array) => a.reduce((n, v, i) => n + (v !== b[i] ? 1 : 0), 0);
          expect(diff(dilateMask(m, [nx, ny, nz], r), dilateRef(m, [nx, ny, nz], r)), `dilate ${nx}×${ny}×${nz} r ${r}`).toBe(0);
          expect(diff(erodeMask(m, [nx, ny, nz], r), erodeRef(m, [nx, ny, nz], r)), `erode ${nx}×${ny}×${nz} r ${r}`).toBe(0);
        }
      }
    }
  }, 30000);
});
