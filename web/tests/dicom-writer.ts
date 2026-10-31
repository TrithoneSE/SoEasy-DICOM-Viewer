/** Minimal DICOM writer (explicit VR little endian, native or encapsulated pixel data) for the unit tests. */

export type Value = string | number | number[] | Uint8Array | Elem[][];
export interface Elem {
  tag: [number, number];
  vr: string;
  value: Value;
}

const LONG = new Set(["OB", "OW", "OF", "SQ", "UT", "UN"]);

class Out {
  parts: Uint8Array[] = [];
  len = 0;
  push(b: Uint8Array) {
    this.parts.push(b);
    this.len += b.length;
  }
  u16(v: number) {
    const b = new Uint8Array(2);
    new DataView(b.buffer).setUint16(0, v, true);
    this.push(b);
  }
  u32(v: number) {
    const b = new Uint8Array(4);
    new DataView(b.buffer).setUint32(0, v, true);
    this.push(b);
  }
  bytes(): Uint8Array {
    const o = new Uint8Array(this.len);
    let p = 0;
    for (const b of this.parts) {
      o.set(b, p);
      p += b.length;
    }
    return o;
  }
}

function valueBytes(e: Elem): Uint8Array {
  const v = e.value;
  if (v instanceof Uint8Array) return v.length % 2 ? concat([v, new Uint8Array(1)]) : v;
  if (e.vr === "US" || e.vr === "SS") {
    const a = Array.isArray(v) ? (v as number[]) : [v as number];
    const b = new Uint8Array(a.length * 2);
    a.forEach((x, i) => new DataView(b.buffer).setUint16(i * 2, x & 0xffff, true));
    return b;
  }
  if (e.vr === "UL") {
    const a = Array.isArray(v) ? (v as number[]) : [v as number];
    const b = new Uint8Array(a.length * 4);
    a.forEach((x, i) => new DataView(b.buffer).setUint32(i * 4, x, true));
    return b;
  }
  if (e.vr === "FL") {
    const a = Array.isArray(v) ? (v as number[]) : [v as number];
    const b = new Uint8Array(a.length * 4);
    a.forEach((x, i) => new DataView(b.buffer).setFloat32(i * 4, x, true));
    return b;
  }
  let s = Array.isArray(v) ? (v as number[]).join("\\") : String(v);
  if (s.length % 2) s += e.vr === "UI" ? "\0" : " ";
  return new TextEncoder().encode(s);
}

export function concat(parts: Uint8Array[]): Uint8Array {
  const o = new Uint8Array(parts.reduce((a, b) => a + b.length, 0));
  let p = 0;
  for (const b of parts) {
    o.set(b, p);
    p += b.length;
  }
  return o;
}

function writeElems(out: Out, elems: Elem[]) {
  for (const e of [...elems].sort((a, b) => a.tag[0] - b.tag[0] || a.tag[1] - b.tag[1])) {
    out.u16(e.tag[0]);
    out.u16(e.tag[1]);
    out.push(new TextEncoder().encode(e.vr));
    if (e.vr === "SQ") {
      out.u16(0);
      out.u32(0xffffffff);
      for (const it of e.value as Elem[][]) {
        out.u16(0xfffe); out.u16(0xe000); out.u32(0xffffffff);
        writeElems(out, it);
        out.u16(0xfffe); out.u16(0xe00d); out.u32(0);
      }
      out.u16(0xfffe); out.u16(0xe0dd); out.u32(0);
      continue;
    }
    const b = valueBytes(e);
    if (LONG.has(e.vr)) {
      out.u16(0);
      out.u32(b.length);
    } else out.u16(b.length);
    out.push(b);
  }
}

/** A Part 10 file: preamble, DICM, meta header, then the data set. `fragments` = encapsulated frames. */
export function writeDicom(elems: Elem[], ts: string, pixels: Uint8Array | { fragments: Uint8Array[] }): Uint8Array {
  const meta = new Out();
  const metaElems: Elem[] = [
    { tag: [2, 1], vr: "OB", value: new Uint8Array([0, 1]) },
    { tag: [2, 2], vr: "UI", value: "1.2.840.10008.5.1.4.1.1.12.1" },
    { tag: [2, 3], vr: "UI", value: "1.2.3.4.5" },
    { tag: [2, 0x10], vr: "UI", value: ts },
  ];
  writeElems(meta, metaElems);
  const mb = meta.bytes();
  const out = new Out();
  out.push(new Uint8Array(128));
  out.push(new TextEncoder().encode("DICM"));
  writeElems(out, [{ tag: [2, 0], vr: "UL", value: mb.length }]);
  out.push(mb);
  writeElems(out, elems);
  out.u16(0x7fe0); out.u16(0x0010);
  if (pixels instanceof Uint8Array) {
    out.push(new TextEncoder().encode("OW"));
    out.u16(0);
    out.u32(pixels.length);
    out.push(pixels);
  } else {
    out.push(new TextEncoder().encode("OB"));
    out.u16(0);
    out.u32(0xffffffff);
    out.u16(0xfffe); out.u16(0xe000); out.u32(0);                   // empty basic offset table
    for (const f of pixels.fragments) {
      const b = f.length % 2 ? concat([f, new Uint8Array(1)]) : f;
      out.u16(0xfffe); out.u16(0xe000); out.u32(b.length);
      out.push(b);
    }
    out.u16(0xfffe); out.u16(0xe0dd); out.u32(0);
  }
  return out.bytes();
}

/** DICOM RLE of one frame (16-bit grey): two PackBits segments, high bytes first (literal runs only). */
export function rleEncode16(frame: Uint16Array): Uint8Array {
  const seg = (byte: number) => {
    const raw = new Uint8Array(frame.length);
    for (let i = 0; i < frame.length; i++) raw[i] = byte ? frame[i] >> 8 : frame[i] & 0xff;
    const parts: Uint8Array[] = [];
    for (let i = 0; i < raw.length; i += 128) {
      const chunk = raw.subarray(i, i + 128);
      parts.push(new Uint8Array([chunk.length - 1]), chunk);
    }
    const s = concat(parts);
    return s.length % 2 ? concat([s, new Uint8Array([0x80])]) : s;     // -128 = no-op padding
  };
  const hi = seg(1), lo = seg(0);
  const head = new Uint8Array(64);
  const dv = new DataView(head.buffer);
  dv.setUint32(0, 2, true);
  dv.setUint32(4, 64, true);
  dv.setUint32(8, 64 + hi.length, true);
  return concat([head, hi, lo]);
}
