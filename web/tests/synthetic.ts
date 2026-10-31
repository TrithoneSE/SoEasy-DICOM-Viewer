/** The synthetic XA run of tests/test_dsa.py, in TypeScript: 10-bit LOG data, skull-like rings, a vessel filling
 *  from frame 5, the patient moving by MOVE from frame 3. */
import { shiftImage } from "../src/dsa";
import { rleEncode16, writeDicom, type Elem } from "./dicom-writer";

export const MOVE: [number, number] = [1.5, -2.3];

function rng(seed: number) {
  let s = seed >>> 0;
  const u = () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) + 0.5) / 2 ** 32;
  return () => Math.sqrt(-2 * Math.log(u())) * Math.cos(2 * Math.PI * u());      // standard normal
}

export function xaFrames(n = 12, size = 256): Uint16Array[] {
  const bone = new Float32Array(size * size);
  const vessel = new Float32Array(size * size);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = y * size + x;
      bone[i] = 500 + 250 * (Math.floor(Math.hypot(y - 128, x - 128) / 20) % 2) + (Math.hypot(y - 100, x - 150) < 25 ? 150 : 0);
      vessel[i] = Math.abs(x - 0.6 * y - 50) < 3 ? -200 : 0;
    }
  }
  const moved = shiftImage(bone, size, size, ...MOVE);
  const noise = rng(3);
  const frames: Uint16Array[] = [];
  for (let k = 0; k < n; k++) {
    const src = k >= 2 ? moved : bone;
    const f = new Uint16Array(size * size);
    for (let i = 0; i < f.length; i++) {
      const v = src[i] + (k >= 4 ? vessel[i] : 0) + noise() * 2;
      f[i] = Math.max(0, Math.min(1023, Math.round(v)));
    }
    frames.push(f);
  }
  return frames;
}

export interface XAOptions {
  n?: number;
  size?: number;
  maskSeq?: boolean;
  rle?: boolean;
  plane?: "A" | "B";
  seriesUid?: string;
  sopUid?: string;
  lin?: boolean;
}

export function makeXA(o: XAOptions = {}): Uint8Array {
  const n = o.n ?? 12, size = o.size ?? 256;
  const frames = xaFrames(n, size);
  const e: Elem[] = [
    { tag: [0x0008, 0x0008], vr: "CS", value: `ORIGINAL\\PRIMARY\\BIPLANE ${o.plane ?? "A"}` },
    { tag: [0x0008, 0x0016], vr: "UI", value: "1.2.840.10008.5.1.4.1.1.12.1" },
    { tag: [0x0008, 0x0018], vr: "UI", value: o.sopUid ?? `1.2.3.9.${o.plane ?? "A"}` },
    { tag: [0x0008, 0x0060], vr: "CS", value: "XA" },
    { tag: [0x0008, 0x0070], vr: "LO", value: "Siemens" },
    { tag: [0x0008, 0x103e], vr: "LO", value: "DSA run" },
    { tag: [0x0010, 0x0010], vr: "PN", value: "DSA^PHANTOM" },
    { tag: [0x0010, 0x0020], vr: "LO", value: "DSA1" },
    { tag: [0x0018, 0x1063], vr: "DS", value: "66.7" },
    { tag: [0x0018, 0x1164], vr: "DS", value: [0.3, 0.3] },
    { tag: [0x0020, 0x000d], vr: "UI", value: "1.2.3.4" },
    { tag: [0x0020, 0x000e], vr: "UI", value: o.seriesUid ?? "1.2.3.4.7" },
    { tag: [0x0020, 0x0011], vr: "IS", value: "1" },
    { tag: [0x0020, 0x0013], vr: "IS", value: o.plane === "B" ? "2" : "1" },
    { tag: [0x0028, 0x0002], vr: "US", value: 1 },
    { tag: [0x0028, 0x0004], vr: "CS", value: "MONOCHROME2" },
    { tag: [0x0028, 0x0008], vr: "IS", value: String(n) },
    { tag: [0x0028, 0x0010], vr: "US", value: size },
    { tag: [0x0028, 0x0011], vr: "US", value: size },
    { tag: [0x0028, 0x0100], vr: "US", value: 16 },
    { tag: [0x0028, 0x0101], vr: "US", value: 10 },
    { tag: [0x0028, 0x0102], vr: "US", value: 9 },
    { tag: [0x0028, 0x0103], vr: "US", value: 0 },
    { tag: [0x0028, 0x1040], vr: "CS", value: o.lin ? "LIN" : "LOG" },
    { tag: [0x0028, 0x1050], vr: "DS", value: "512" },
    { tag: [0x0028, 0x1051], vr: "DS", value: "1024" },
  ];
  if (o.maskSeq) {
    e.push({ tag: [0x0028, 0x1090], vr: "CS", value: "SUB" });
    e.push({
      tag: [0x0028, 0x6100], vr: "SQ", value: [[
        { tag: [0x0028, 0x6101], vr: "CS", value: "AVG_SUB" },
        { tag: [0x0028, 0x6110], vr: "US", value: [3, 4] },          // 1-based, already moved: averaged
        { tag: [0x0028, 0x6114], vr: "FL", value: [0, 0] },
      ]],
    });
  }
  if (o.rle) return writeDicom(e, "1.2.840.10008.1.2.5", { fragments: frames.map(rleEncode16) });
  const px = new Uint8Array(n * size * size * 2);
  frames.forEach((f, k) => px.set(new Uint8Array(f.buffer), k * size * size * 2));
  return writeDicom(e, "1.2.840.10008.1.2.1", px);
}
