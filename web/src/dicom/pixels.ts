/**
 * Pixel data of one frame: finding its bytes in the file (native or encapsulated) and decoding them. `decodeFrame`
 * runs in the decode workers (and in the unit tests under Node); the WASM codecs come in through `Codecs` so each
 * environment can say where its .wasm files are.
 */
import * as dicomParser from "dicom-parser";
import type { DataSet } from "dicom-parser";
import { Decoder as LosslessDecoder } from "jpeg-lossless-decoder-js";
import type { Header } from "./header";
import { PIXEL_DATA } from "./header";

export const TS = {
  implicit: "1.2.840.10008.1.2",
  explicit: "1.2.840.10008.1.2.1",
  deflate: "1.2.840.10008.1.2.1.99",
  bigEndian: "1.2.840.10008.1.2.2",
  jpegBaseline: "1.2.840.10008.1.2.4.50",
  jpegExtended: "1.2.840.10008.1.2.4.51",
  jpegLossless: "1.2.840.10008.1.2.4.57",
  jpegLosslessSV1: "1.2.840.10008.1.2.4.70",
  jpegLS: "1.2.840.10008.1.2.4.80",
  jpegLSNear: "1.2.840.10008.1.2.4.81",
  j2kLossless: "1.2.840.10008.1.2.4.90",
  j2k: "1.2.840.10008.1.2.4.91",
  htj2kLossless: "1.2.840.10008.1.2.4.201",
  htj2kRpcl: "1.2.840.10008.1.2.4.202",
  htj2k: "1.2.840.10008.1.2.4.203",
  rle: "1.2.840.10008.1.2.5",
};

const NATIVE = new Set([TS.implicit, TS.explicit, TS.bigEndian]);
const SUPPORTED = new Set([...NATIVE, TS.jpegBaseline, TS.jpegLossless, TS.jpegLosslessSV1, TS.jpegLS, TS.jpegLSNear,
  TS.j2kLossless, TS.j2k, TS.htj2kLossless, TS.htj2kRpcl, TS.htj2k, TS.rle]);

export function isNative(ts: string): boolean {
  return NATIVE.has(ts);
}

export function unsupportedReason(h: Header): string | null {
  if (!SUPPORTED.has(h.transferSyntax)) {
    if (h.transferSyntax === TS.jpegExtended) return "12-bit JPEG (extended) is not supported yet";
    if (h.transferSyntax === TS.deflate) return "deflated DICOM is not supported yet";
    return `transfer syntax ${h.transferSyntax} is not supported`;
  }
  if (h.bitsAllocated === 1) return "1-bit images are not supported";
  if (h.samples !== 1 && h.samples !== 3) return `${h.samples} samples per pixel are not supported`;
  return null;
}

export interface FrameInfo {
  ts: string;
  rows: number;
  cols: number;
  samples: number;
  bitsAllocated: number;
  bitsStored: number;
  pixelRep: number;
  planarConfig: number;
  photometric: string;
}

export function frameInfo(h: Header): FrameInfo {
  return {
    ts: h.transferSyntax, rows: h.rows, cols: h.cols, samples: h.samples, bitsAllocated: h.bitsAllocated,
    bitsStored: h.bitsStored, pixelRep: h.pixelRep, planarConfig: h.planarConfig, photometric: h.photometric,
  };
}

export type Gray = Uint8Array | Int8Array | Uint16Array | Int16Array | Uint32Array | Int32Array;

/** A decoded frame: grey values as stored (before Rescale Slope / Intercept), or interleaved 8-bit RGB. */
export interface Decoded {
  gray?: Gray;
  rgb?: Uint8Array;
}

/** Byte ranges [position, length] of frame k inside the file, so a frame can be read without keeping the file. */
export function frameRanges(ds: DataSet, h: Header, k: number): [number, number][] {
  const el = ds.elements[PIXEL_DATA];
  if (!el) throw new Error("no pixel data");
  const base = ds.byteArray.byteOffset;
  if (!el.encapsulatedPixelData) {
    const size = (h.rows * h.cols * h.samples * h.bitsAllocated) / 8;
    if (el.dataOffset + (k + 1) * size > ds.byteArray.length) throw new Error("pixel data are cut short");
    return [[base + el.dataOffset + k * size, size]];
  }
  const frags = (el.fragments ?? []).map((f) => ({ ...f, position: base + f.position }));
  const table = framesFromFragments(h.frames, el.basicOffsetTable ?? [], frags);
  if (table) return table[k];
  if (!h.transferSyntax.startsWith("1.2.840.10008.1.2.4.")) throw new Error("cannot find the frames: no offset table");
  return framesFromFragments(h.frames, dicomParser.createJPEGBasicOffsetTable(ds, el), frags)![k];
}

interface Fragment {
  offset: number;      // from the first fragment's item tag (the Basic Offset Table's origin)
  position: number;    // absolute byte position of the fragment's data
  length: number;
}

/** Fragments of each frame: from the Basic Offset Table, else one fragment per frame, else (one frame) all of
 *  them. null when that cannot be decided without scanning the JPEG data (several fragments per frame, no table). */
export function framesFromFragments(n: number, bot: number[], frags: Fragment[]): [number, number][][] | null {
  if (bot.length === n) {
    return bot.map((start, k) => {
      const end = k + 1 < n ? bot[k + 1] : Infinity;
      return frags.filter((f) => f.offset >= start && f.offset < end).map((f) => [f.position, f.length] as [number, number]);
    });
  }
  if (frags.length === n) return frags.map((f) => [[f.position, f.length]]);
  if (n === 1) return [frags.map((f) => [f.position, f.length] as [number, number])];
  return null;
}

/**
 * Where each frame of a multiframe file is, read with a few small reads instead of loading the file: the header,
 * then the item headers of the encapsulated pixel data. A 121-frame run is then decoded frame by frame without the
 * whole file in memory (iPad tabs are killed above ~1–1.5 GB). null = the caller reads the file whole.
 */
export async function lazyFrameTable(read: (s: number, e: number) => Promise<ArrayBuffer>, size: number,
  h: Header, pixelDataOffset: number, pixelDataLength: number): Promise<[number, number][][] | null> {
  if (pixelDataLength !== 0xffffffff) {
    const fs = (h.rows * h.cols * h.samples * h.bitsAllocated) / 8;
    if (pixelDataOffset + h.frames * fs > size) return null;
    return Array.from({ length: h.frames }, (_, k) => [[pixelDataOffset + k * fs, fs]]);
  }
  const item = async (pos: number) => {
    const v = new DataView(await read(pos, pos + 8));
    return { group: v.getUint16(0, true), elem: v.getUint16(2, true), len: v.getUint32(4, true) };
  };
  let pos = pixelDataOffset;
  const first = await item(pos);
  if (first.group !== 0xfffe || first.elem !== 0xe000) return null;
  const bot: number[] = [];
  if (first.len) {
    const v = new DataView(await read(pos + 8, pos + 8 + first.len));
    for (let i = 0; i + 4 <= first.len; i += 4) bot.push(v.getUint32(i, true));
  }
  pos += 8 + first.len;
  const origin = pos;
  const frags: Fragment[] = [];
  while (pos + 8 <= size) {
    const it = await item(pos);
    if (it.group !== 0xfffe || it.elem !== 0xe000) break;          // sequence delimiter (E0DD) or the end
    frags.push({ offset: pos - origin, position: pos + 8, length: it.len });
    pos += 8 + it.len;
  }
  return framesFromFragments(h.frames, bot, frags);
}

export function gather(buf: ArrayBuffer, ranges: [number, number][]): Uint8Array {
  const total = ranges.reduce((a, r) => a + r[1], 0);
  const out = new Uint8Array(total);
  let p = 0;
  for (const [pos, len] of ranges) {
    out.set(new Uint8Array(buf, pos, len), p);
    p += len;
  }
  return out;
}

// ------------------------------------------------------------------ decoding
/** Emscripten codec modules (cornerstone's builds of CharLS, OpenJPEG and libjpeg-turbo). */
export interface Codecs {
  jpegLS(): Promise<any>;      // eslint-disable-line @typescript-eslint/no-explicit-any
  j2k(): Promise<any>;         // eslint-disable-line @typescript-eslint/no-explicit-any
  jpeg8(): Promise<any>;       // eslint-disable-line @typescript-eslint/no-explicit-any
}

/** For native / RLE data, decoded on the main thread without loading any WASM. */
export const NO_CODECS: Codecs = {
  jpegLS: () => Promise.reject(new Error("no codec")),
  j2k: () => Promise.reject(new Error("no codec")),
  jpeg8: () => Promise.reject(new Error("no codec")),
};

export async function decodeFrame(bytes: Uint8Array, info: FrameInfo, codecs: Codecs): Promise<Decoded> {
  const ts = info.ts;
  if (NATIVE.has(ts)) return finish(native(bytes, info), info);
  if (ts === TS.rle) return finish(rle(bytes, info), info);
  if (ts === TS.jpegLossless || ts === TS.jpegLosslessSV1) {
    const bps = info.bitsAllocated <= 8 ? 1 : 2;
    const data = zeroPadBeforeEOI(bytes);
    const out = new LosslessDecoder().decode(data.buffer as ArrayBuffer, data.byteOffset, data.length, bps);
    return finish(asSamples(out as Uint8Array | Uint16Array, info), info);
  }
  let factory: Promise<any>;   // eslint-disable-line @typescript-eslint/no-explicit-any
  let cls: string;
  if (ts === TS.jpegLS || ts === TS.jpegLSNear) [factory, cls] = [codecs.jpegLS(), "JpegLSDecoder"];
  else if (ts === TS.jpegBaseline) [factory, cls] = [codecs.jpeg8(), "JPEGDecoder"];
  else [factory, cls] = [codecs.j2k(), "J2KDecoder"];
  const mod = await factory;
  const dec = new mod[cls]();
  try {
    dec.getEncodedBuffer(bytes.length).set(bytes);
    dec.decode();
    const fi = dec.getFrameInfo();
    const raw: Uint8Array = dec.getDecodedBuffer();
    const copy = raw.slice();                              // the WASM heap is reused by the next frame
    if (fi.componentCount === 3 || info.samples === 3) return { rgb: copy.subarray(0, info.rows * info.cols * 3) };
    const n = info.rows * info.cols;
    const wide = fi.bitsPerSample > 8;
    let gray: Gray;
    if (wide) gray = info.pixelRep ? new Int16Array(copy.buffer, 0, n) : new Uint16Array(copy.buffer, 0, n);
    else gray = info.pixelRep ? new Int8Array(copy.buffer, 0, n) : copy.subarray(0, n);
    return finish({ gray }, info);
  } finally {
    dec.delete?.();
  }
}

/**
 * JPEG decoders read zero bits once the entropy-coded data reach a marker (libjpeg does). jpeg-lossless-decoder-js
 * instead stops, and left the last pixel of a real Siemens run 0 (129 in pydicom / libjpeg) when its final code
 * ran into the EOI marker. Four zero bytes before EOI give it those bits.
 */
export function zeroPadBeforeEOI(b: Uint8Array): Uint8Array {
  let eoi = -1;
  for (let i = b.length - 2; i >= Math.max(0, b.length - 16); i--) {
    if (b[i] === 0xff && b[i + 1] === 0xd9) {
      eoi = i;
      break;
    }
  }
  if (eoi < 0) eoi = b.length;
  const out = new Uint8Array(eoi + 6);
  out.set(b.subarray(0, eoi));
  out.set([0, 0, 0, 0, 0xff, 0xd9], eoi);
  return out;
}

function asSamples(out: Uint8Array | Uint16Array, info: FrameInfo): Decoded {
  if (info.samples === 3) return { rgb: new Uint8Array(out.buffer, out.byteOffset, out.length) };
  if (out instanceof Uint16Array && info.pixelRep) return { gray: new Int16Array(out.buffer, out.byteOffset, out.length) };
  if (out instanceof Uint8Array && info.pixelRep) return { gray: new Int8Array(out.buffer, out.byteOffset, out.length) };
  return { gray: out };
}

function native(bytes: Uint8Array, info: FrameInfo): Decoded {
  const n = info.rows * info.cols;
  const copy = bytes.slice();                              // aligned, and owned by us
  if (info.samples === 3) {
    const rgb = info.bitsAllocated === 8 ? copy : downTo8(new Uint16Array(copy.buffer), info.bitsStored);
    return { rgb: info.planarConfig === 1 ? interleave(rgb, n) : rgb };
  }
  if (info.ts === TS.bigEndian && info.bitsAllocated > 8) {
    const step = info.bitsAllocated / 8;
    for (let i = 0; i < copy.length; i += step) copy.subarray(i, i + step).reverse();
  }
  switch (info.bitsAllocated) {
    case 8: return { gray: info.pixelRep ? new Int8Array(copy.buffer, 0, n) : copy.subarray(0, n) };
    case 16: return { gray: info.pixelRep ? new Int16Array(copy.buffer, 0, n) : new Uint16Array(copy.buffer, 0, n) };
    case 32: return { gray: info.pixelRep ? new Int32Array(copy.buffer, 0, n) : new Uint32Array(copy.buffer, 0, n) };
  }
  throw new Error(`${info.bitsAllocated} bits allocated are not supported`);
}

function downTo8(a: Uint16Array, bits: number): Uint8Array {
  const out = new Uint8Array(a.length);
  const sh = Math.max(0, bits - 8);
  for (let i = 0; i < a.length; i++) out[i] = a[i] >> sh;
  return out;
}

function interleave(planar: Uint8Array, n: number): Uint8Array {
  const out = new Uint8Array(n * 3);
  for (let i = 0; i < n; i++) {
    out[i * 3] = planar[i];
    out[i * 3 + 1] = planar[n + i];
    out[i * 3 + 2] = planar[2 * n + i];
  }
  return out;
}

/** DICOM RLE Lossless (PS3.5 annex G): one PackBits segment per byte plane, most significant byte first. */
export function rle(bytes: Uint8Array, info: FrameInfo): Decoded {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const nseg = dv.getUint32(0, true);
  const offsets: number[] = [];
  for (let i = 0; i < nseg; i++) offsets.push(dv.getUint32(4 + 4 * i, true));
  const n = info.rows * info.cols;
  const bps = info.bitsAllocated / 8;
  if (nseg !== bps * info.samples) throw new Error(`RLE: ${nseg} segments for ${info.samples}×${bps} bytes`);
  const out = new Uint8Array(n * bps * info.samples);
  for (let s = 0; s < nseg; s++) {
    const sample = Math.floor(s / bps);
    const byte = bps - 1 - (s % bps);                     // segment 0 holds the most significant bytes
    const stride = bps * info.samples;
    let o = sample * bps + byte;                          // interleaved output position
    const end = s + 1 < nseg ? offsets[s + 1] : bytes.length;
    let p = offsets[s];
    let written = 0;
    while (p < end && written < n) {
      const c = (bytes[p++] << 24) >> 24;
      if (c >= 0) {
        for (let j = 0; j <= c && written < n; j++, written++, o += stride) out[o] = bytes[p++];
      } else if (c !== -128) {
        const v = bytes[p++];
        for (let j = 0; j <= -c && written < n; j++, written++, o += stride) out[o] = v;
      }
    }
  }
  if (info.samples === 3) return { rgb: info.bitsAllocated === 8 ? out : downTo8(new Uint16Array(out.buffer), info.bitsStored) };
  switch (info.bitsAllocated) {
    case 8: return { gray: info.pixelRep ? new Int8Array(out.buffer) : out };
    case 16: return { gray: info.pixelRep ? new Int16Array(out.buffer) : new Uint16Array(out.buffer) };
    case 32: return { gray: info.pixelRep ? new Int32Array(out.buffer) : new Uint32Array(out.buffer) };
  }
  throw new Error(`${info.bitsAllocated} bits allocated are not supported`);
}

/** Only Bits Stored count: mask the bits above them (overlays) and sign-extend signed data; YBR -> RGB. */
function finish(d: Decoded, info: FrameInfo): Decoded {
  const g = d.gray;
  if (g && info.bitsStored < info.bitsAllocated && info.bitsStored < 32) {
    const bits = info.bitsStored;
    const mask = 2 ** bits - 1;
    if (info.pixelRep) {
      const sh = 32 - bits;
      for (let i = 0; i < g.length; i++) g[i] = ((g[i] & mask) << sh) >> sh;
    } else {
      for (let i = 0; i < g.length; i++) g[i] = g[i] & mask;
    }
  }
  const ybr = info.photometric === "YBR_FULL" && NATIVE.has(info.ts) || info.photometric === "YBR_FULL" && info.ts === TS.rle;
  if (d.rgb && ybr) {
    const a = d.rgb;
    for (let i = 0; i < a.length; i += 3) {
      const y = a[i], cb = a[i + 1] - 128, cr = a[i + 2] - 128;
      a[i] = clamp8(y + 1.402 * cr);
      a[i + 1] = clamp8(y - 0.344136 * cb - 0.714136 * cr);
      a[i + 2] = clamp8(y + 1.772 * cb);
    }
  }
  return d;
}

function clamp8(v: number): number {
  return v < 0 ? 0 : v > 255 ? 255 : Math.round(v);
}
