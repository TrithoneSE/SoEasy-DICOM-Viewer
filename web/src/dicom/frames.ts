/** Frame bytes of a DICOM file, read lazily where the source allows it (see pixels.lazyFrameTable). */
import { parse, PIXEL_DATA, readHeader, TruncatedHeader, type Header } from "./header";
import { frameRanges, gather, lazyFrameTable } from "./pixels";
import type { Source } from "./source";

export interface FrameReader {
  h: Header;
  /** Compressed (or native) bytes of frame k, ready for decodeFrame. */
  frame(k: number): Promise<Uint8Array>;
  /** Whether the whole file is held in memory (until the reader is dropped). */
  whole: boolean;
}

export async function openFrames(src: Source): Promise<FrameReader> {
  if (src.random) {
    let ds = null;
    for (let n = 64 * 1024; ; n *= 4) {
      try {
        ds = parse(await src.read(0, Math.min(n, src.size)), true);
        break;
      } catch (e) {
        if (!(e instanceof TruncatedHeader) || n >= src.size) break;
      }
    }
    const el = ds?.elements[PIXEL_DATA];
    if (ds && el) {
      const h = readHeader(ds);
      const table = await lazyFrameTable((s, e) => src.read(s, e), src.size, h, el.dataOffset, el.length);
      if (table) {
        return {
          h,
          whole: false,
          async frame(k) {
            const parts = await Promise.all(table[k].map(([p, l]) => src.read(p, p + l)));
            if (parts.length === 1) return new Uint8Array(parts[0]);
            const out = new Uint8Array(parts.reduce((a, b) => a + b.byteLength, 0));
            let o = 0;
            for (const b of parts) {
              out.set(new Uint8Array(b), o);
              o += b.byteLength;
            }
            return out;
          },
        };
      }
    }
  }
  const buf = await src.read();
  const ds = parse(buf, false);
  const h = readHeader(ds);
  return { h, whole: true, frame: async (k) => gather(buf, frameRanges(ds, h, k)) };
}
