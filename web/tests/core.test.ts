import { deflateRawSync } from "node:zlib";
import { describe, expect, it } from "vitest";
import { parse, readHeader } from "../src/dicom/header";
import { decodeFrame, frameInfo, frameRanges, gather, NO_CODECS, zeroPadBeforeEOI } from "../src/dicom/pixels";
import { scan, seriesCount } from "../src/dicom/scan";
import type { Source } from "../src/dicom/source";
import { openSeries, stackOrder } from "../src/dicom/volume";
import { openFrames } from "../src/dicom/frames";
import { openZip } from "../src/dicom/zip";
import { DSA, DSA_BACKGROUND, shiftImage, type Frames } from "../src/dsa";
import { fmtArea, fmtDims, fmtLen } from "../src/view/measure";
import { concat, writeDicom, type Elem } from "./dicom-writer";
import { makeXA, MOVE, xaFrames } from "./synthetic";

const mem = (name: string, bytes: Uint8Array): Source => ({
  name,
  size: bytes.length,
  random: true,
  read: async (s = 0, e = bytes.length) => bytes.slice(s, e).buffer as ArrayBuffer,
});

function framesOf(file: Uint8Array): Frames & { data: Float32Array[] } {
  const ds = parse(file.buffer as ArrayBuffer, false);
  const h = readHeader(ds);
  const data: Float32Array[] = [];
  return {
    n: h.frames, rows: h.rows, cols: h.cols, data,
    frame(k) {
      if (!data[k]) throw new Error("not decoded");
      return data[k];
    },
  };
}

async function decodeAll(file: Uint8Array) {
  const ds = parse(file.buffer as ArrayBuffer, false);
  const h = readHeader(ds);
  const out: Float32Array[] = [];
  for (let k = 0; k < h.frames; k++) {
    const d = await decodeFrame(gather(file.buffer as ArrayBuffer, frameRanges(ds, h, k)), frameInfo(h), NO_CODECS);
    out.push(Float32Array.from(d.gray!, (v) => v * h.slope + h.intercept));
  }
  return { h, frames: out };
}

describe("measurement labels (dcv/measure.py)", () => {
  it("mm below 1 cm, cm from 1 cm up", () => {
    expect(fmtLen(8.44)).toBe("8.4 mm");
    expect(fmtLen(9.96)).toBe("1.00 cm");      // rounds to 10.0 mm: shown in cm, like Python round(…, 1)
    expect(fmtLen(32.7)).toBe("3.27 cm");
    expect(fmtDims(8.4, 6.1)).toBe("8.4 × 6.1 mm");
    expect(fmtDims(45.2, 31)).toBe("4.52 × 3.10 cm");
    expect(fmtArea(55)).toBe("55.0 mm²");
    expect(fmtArea(250)).toBe("2.50 cm²");
  });
});

describe("XA header (dcv/dsa.py hint_from_dataset)", () => {
  it("frame time, imager spacing, mask subtraction sequence", () => {
    const h = readHeader(parse(makeXA({ maskSeq: true }).buffer as ArrayBuffer, true));
    expect(h.frames).toBe(12);
    expect(h.frameTimeMs).toBeCloseTo(66.7);
    expect(h.spacing).toEqual([0.3, 0.3]);
    expect(h.spacingSource).toBe("imager");
    expect(h.dsa).toEqual({ mask: [2, 3], shift: [0, 0], log: false, recommended: true });
    expect([h.ww, h.wl]).toEqual([1024, 512]);
    const lin = readHeader(parse(makeXA({ lin: true }).buffer as ArrayBuffer, true));
    expect(lin.dsa.log).toBe(true);
    expect(lin.dsa.mask).toBeNull();
  });
});

describe("pixel data", () => {
  it("native and RLE frames decode to the same values", async () => {
    const ref = xaFrames(12, 256);
    const nat = await decodeAll(makeXA());
    const rle = await decodeAll(makeXA({ rle: true }));
    for (const k of [0, 5, 11]) {
      expect(Array.from(nat.frames[k])).toEqual(Array.from(ref[k]));
      expect(Array.from(rle.frames[k])).toEqual(Array.from(ref[k]));
    }
  });

  it("JPEG data get zero bits before EOI (the last pixel of a real Siemens run was 0 without them)", () => {
    expect(Array.from(zeroPadBeforeEOI(new Uint8Array([0xff, 0xd8, 7, 0, 0, 0xff, 0xd9, 0]))))
      .toEqual([0xff, 0xd8, 7, 0, 0, 0, 0, 0, 0, 0xff, 0xd9]);
  });

  it("signed 12-bit data are sign-extended and overlay bits masked", async () => {
    const vals = new Int16Array([-1000, -1, 0, 2047, -2048, 5]);
    const raw = new Uint16Array(vals.length);
    vals.forEach((v, i) => (raw[i] = (v & 0x0fff) | 0x3000));     // junk in the 4 high bits
    const e: Elem[] = [
      { tag: [0x0008, 0x0060], vr: "CS", value: "CT" },
      { tag: [0x0028, 0x0002], vr: "US", value: 1 },
      { tag: [0x0028, 0x0004], vr: "CS", value: "MONOCHROME2" },
      { tag: [0x0028, 0x0010], vr: "US", value: 2 },
      { tag: [0x0028, 0x0011], vr: "US", value: 3 },
      { tag: [0x0028, 0x0100], vr: "US", value: 16 },
      { tag: [0x0028, 0x0101], vr: "US", value: 12 },
      { tag: [0x0028, 0x0102], vr: "US", value: 11 },
      { tag: [0x0028, 0x0103], vr: "US", value: 1 },
    ];
    const { frames } = await decodeAll(writeDicom(e, "1.2.840.10008.1.2.1", new Uint8Array(raw.buffer)));
    expect(Array.from(frames[0])).toEqual(Array.from(vals));
  });
});

describe("series grouping (dcv/loader.py scan_folder)", () => {
  it("a biplane run is two entries; stacks group by series", async () => {
    const a = makeXA({ n: 40, rle: true, plane: "A", seriesUid: "9.9", sopUid: "9.9.1" });
    const b = makeXA({ n: 40, rle: true, plane: "B", seriesUid: "9.9", sopUid: "9.9.2" });
    const list = await scan([mem("bi/A.dcm", a), mem("bi/B.dcm", b), mem("bi/readme.txt", new Uint8Array(500))]);
    expect(list.map((s) => [s.description, seriesCount(s), s.multiframe])).toEqual([
      ["DSA run A", 40, true], ["DSA run B", 40, true]]);
  });

  it("stacks: majority orientation, sorted along the normal, duplicates dropped", () => {
    const ax = [1, 0, 0, 0, 1, 0];
    const inst = (z: number, iop = ax, n = 1) => ({
      src: mem(`s${z}`, new Uint8Array()), key: iop === ax ? z : 0,
      h: { rows: 4, cols: 4, iop, ipp: [0, 0, z], instanceNumber: n } as never,
    });
    const items = [inst(5), inst(1), inst(3), inst(3), inst(0, [0, 1, 0, 0, 0, -1]), inst(7)];
    expect(stackOrder(items).map((i) => i.key)).toEqual([1, 3, 5, 7]);
  });
});

describe("zip archives", () => {
  it("stored and deflated entries read in place", async () => {
    const f1 = makeXA({ n: 3, size: 64 });
    const f2 = new TextEncoder().encode("hello zip ".repeat(1000));
    const entries: Uint8Array[] = [];
    const central: Uint8Array[] = [];
    let off = 0;
    for (const [name, data, method] of [["a/one.dcm", f1, 8], ["a/two.txt", f2, 0]] as [string, Uint8Array, number][]) {
      const nb = new TextEncoder().encode(name);
      const body = method === 8 ? new Uint8Array(deflateRawSync(data)) : data;
      const lh = new DataView(new ArrayBuffer(30));
      lh.setUint32(0, 0x04034b50, true);
      lh.setUint16(8, method, true);
      lh.setUint32(18, body.length, true);
      lh.setUint32(22, data.length, true);
      lh.setUint16(26, nb.length, true);
      const ch = new DataView(new ArrayBuffer(46));
      ch.setUint32(0, 0x02014b50, true);
      ch.setUint16(10, method, true);
      ch.setUint32(20, body.length, true);
      ch.setUint32(24, data.length, true);
      ch.setUint16(28, nb.length, true);
      ch.setUint32(42, off, true);
      const local = concat([new Uint8Array(lh.buffer), nb, body]);
      entries.push(local);
      central.push(concat([new Uint8Array(ch.buffer), nb]));
      off += local.length;
    }
    const cd = concat(central);
    const end = new DataView(new ArrayBuffer(22));
    end.setUint32(0, 0x06054b50, true);
    end.setUint16(8, 2, true);
    end.setUint16(10, 2, true);
    end.setUint32(12, cd.length, true);
    end.setUint32(16, off, true);
    const zip = new Blob([concat([...entries, cd, new Uint8Array(end.buffer)]) as Uint8Array<ArrayBuffer>]);
    const src = await openZip(zip, "x.zip");
    expect(src.map((s) => [s.name, s.size])).toEqual([["x.zip/a/one.dcm", f1.length], ["x.zip/a/two.txt", f2.length]]);
    expect(new Uint8Array(await src[0].read())).toEqual(f1);
    expect(new Uint8Array(await src[0].read(0, 200))).toEqual(f1.slice(0, 200));
    expect(new Uint8Array(await src[1].read(10, 30))).toEqual(f2.slice(10, 30));
    const list = await scan(src);
    expect(list.length).toBe(1);
    expect(seriesCount(list[0])).toBe(3);
  });
});

describe("DSA (dcv/dsa.py)", () => {
  it("shift_image: integer and bilinear shifts, edges repeated", () => {
    const img = Float32Array.from({ length: 12 }, (_, i) => i);      // 3 × 4
    expect(Array.from(shiftImage(img, 3, 4, 1, 0))).toEqual([0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7]);
    expect(Array.from(shiftImage(img, 3, 4, 0, 0.5)).slice(0, 4)).toEqual([0, 0.5, 1.5, 2.5]);
  });

  it("window, default mask, scanner mask, auto pixel-shift", async () => {
    const { frames } = await decodeAll(makeXA());
    const vol = framesOf(makeXA());
    vol.data.push(...frames);
    const dsa = new DSA(vol, readHeader(parse(makeXA().buffer as ArrayBuffer, true)).dsa);
    expect(dsa.mask).toEqual([1]);                                  // the second frame
    const [ww, wl] = dsa.suggestWindow(1024);
    expect(ww).toBeCloseTo(0.6 * 1024);
    expect((0 - (wl - ww / 2)) / ww).toBeCloseTo(DSA_BACKGROUND);
    const sub = dsa.subtract(6, true);
    expect(sub[100 * 256 + 110]).toBeLessThan(-100);                // the vessel (col = 0.6·row + 50)
    // frame 2 is before the movement: the bone does not cancel; auto pixel-shift finds the movement
    const tv = (a: Float32Array) => {
      let s = 0;
      for (let y = 150; y < 250; y++) for (let x = 10; x < 80; x++) s += Math.abs(a[y * 256 + x]);   // no vessel here
      return s / (100 * 70);
    };
    const before = tv(dsa.subtract(6, true));
    const shift = dsa.autoShift(6, [200, 45], dsa.square());
    expect(Math.abs(shift[0] - MOVE[0])).toBeLessThan(0.15);
    expect(Math.abs(shift[1] - MOVE[1])).toBeLessThan(0.15);
    const after = tv(dsa.subtract(6, true));
    expect(after).toBeLessThan(4);
    expect(after).toBeLessThan(before);
    dsa.reset();
    expect([dsa.mask, dsa.shift]).toEqual([[1], [0, 0]]);
    dsa.setMask([3]);
    expect(dsa.label()[0]).toBe("Mask: 4");
    dsa.move(-0.1, 1);
    expect(dsa.label()[1]).toBe("Shift: x +1.0  y -0.1 px");

    const tagged = new DSA(vol, readHeader(parse(makeXA({ maskSeq: true }).buffer as ArrayBuffer, true)).dsa);
    expect(tagged.mask).toEqual([2, 3]);
    expect(tv(tagged.subtract(6, true))).toBeLessThan(4);            // scanner mask frames 3 + 4, already moved
  });
});

describe("lazy frame reading (multiframe runs without the whole file in memory)", () => {
  it("native and RLE runs: the same bytes as a whole-file parse, with only small reads", async () => {
    for (const file of [makeXA({ n: 12 }), makeXA({ n: 12, rle: true })]) {
      const reads: number[] = [];
      const src: Source = { ...mem("run.dcm", file), read: async (s = 0, e = file.length) => {
        reads.push(e - s);
        return file.slice(s, e).buffer as ArrayBuffer;
      } };
      const r = await openFrames(src);
      expect(r.whole).toBe(false);
      const ds = parse(file.buffer as ArrayBuffer, false);
      for (const k of [0, 7, 11]) {
        expect(await r.frame(k)).toEqual(gather(file.buffer as ArrayBuffer, frameRanges(ds, r.h, k)));
      }
      expect(Math.max(...reads)).toBeLessThan(file.length / 4);
    }
  });
});

describe("volume storage (dcv/loader.py: Int16 when the values fit)", () => {
  const ct = (z: number, raw: number[], bits = 16) => writeDicom([
    { tag: [0x0008, 0x0060], vr: "CS", value: "CT" },
    { tag: [0x0020, 0x000e], vr: "UI", value: "5.5" },
    { tag: [0x0020, 0x0032], vr: "DS", value: [0, 0, z] },
    { tag: [0x0020, 0x0037], vr: "DS", value: [1, 0, 0, 0, 1, 0] },
    { tag: [0x0028, 0x0002], vr: "US", value: 1 },
    { tag: [0x0028, 0x0004], vr: "CS", value: "MONOCHROME2" },
    { tag: [0x0028, 0x0010], vr: "US", value: 1 },
    { tag: [0x0028, 0x0011], vr: "US", value: 2 },
    { tag: [0x0028, 0x0100], vr: "US", value: 16 },
    { tag: [0x0028, 0x0101], vr: "US", value: bits },
    { tag: [0x0028, 0x0102], vr: "US", value: bits - 1 },
    { tag: [0x0028, 0x0103], vr: "US", value: 0 },
    { tag: [0x0028, 0x1052], vr: "DS", value: "-1024" },
    { tag: [0x0028, 0x1053], vr: "DS", value: "1" },
  ], "1.2.840.10008.1.2.1", new Uint8Array(new Uint16Array(raw).buffer));

  it("16-bit unsigned with intercept -1024 stays Int16; a value beyond 32767 turns it into Float32", async () => {
    const small = await scan([mem("a", ct(2, [0, 3000])), mem("b", ct(1, [24, 1024]))]);
    const v1 = (await openSeries(small[0])).vol;
    await v1.want([0, 1]);
    expect(v1.data.constructor.name).toBe("Int16Array");
    expect(Array.from(v1.data)).toEqual([-1000, 0, -1024, 1976]);       // sorted along the normal: z = 1, 2
    const big = await scan([mem("a", ct(1, [0, 1000])), mem("b", ct(2, [0, 60000]))]);
    const v2 = (await openSeries(big[0])).vol;
    await v2.want([0, 1]);
    expect(v2.data.constructor.name).toBe("Float32Array");
    expect(Array.from(v2.data)).toEqual([-1024, -24, -1024, 58976]);
  });
});
