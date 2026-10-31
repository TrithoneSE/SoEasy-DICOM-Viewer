/**
 * .zip archives read in place: only the central directory is read when the archive is opened; an entry's bytes are
 * read (and inflated with the browser's DecompressionStream) when the viewer asks for them. A 1.3 GB CD image of a
 * DSA study therefore never sits in memory as a whole — the iPad tab would be killed. Stored and deflated entries,
 * ZIP64; encrypted entries are skipped.
 */
import type { Source } from "./source";

const EOCD = 0x06054b50;
const EOCD64 = 0x06064b50;
const LOC64 = 0x07064b50;
const CEN = 0x02014b50;
const LOC = 0x04034b50;

export class ZipError extends Error {}

export function isZipName(name: string): boolean {
  return /\.zip$/i.test(name);
}

function u64(v: DataView, o: number): number {
  return v.getUint32(o, true) + v.getUint32(o + 4, true) * 2 ** 32;
}

async function blobBytes(blob: Blob, start: number, end: number): Promise<DataView> {
  return new DataView(await blob.slice(start, end).arrayBuffer());
}

/** Inflate a raw-deflate stream, stopping once `limit` output bytes exist (header reads need only the start). */
async function inflate(data: Blob, limit: number): Promise<Uint8Array> {
  const reader = data.stream().pipeThrough(new DecompressionStream("deflate-raw")).getReader();
  const parts: Uint8Array[] = [];
  let total = 0;
  try {
    while (total < limit) {
      const { done, value } = await reader.read();
      if (done) break;
      parts.push(value);
      total += value.length;
    }
  } finally {
    reader.cancel().catch(() => undefined);
  }
  const out = new Uint8Array(Math.min(total, limit));
  let pos = 0;
  for (const p of parts) {
    const n = Math.min(p.length, out.length - pos);
    out.set(p.subarray(0, n), pos);
    pos += n;
    if (pos >= out.length) break;
  }
  return out;
}

/** Entries of a .zip (files only; folders, macOS resource forks and encrypted entries left out). */
export async function openZip(file: Blob, zipName: string): Promise<Source[]> {
  const size = file.size;
  const tailLen = Math.min(size, 65557);
  const tail = await blobBytes(file, size - tailLen, size);
  let eocd = -1;
  for (let i = tailLen - 22; i >= 0; i--) {
    if (tail.getUint32(i, true) === EOCD) {
      eocd = i;
      break;
    }
  }
  if (eocd < 0) throw new ZipError(`${zipName}: not a zip archive (or a damaged one)`);
  let count = tail.getUint16(eocd + 10, true);
  let cdSize = tail.getUint32(eocd + 12, true);
  let cdOffset = tail.getUint32(eocd + 16, true);
  if (eocd >= 20 && tail.getUint32(eocd - 20, true) === LOC64) {
    const z64 = u64(tail, eocd - 20 + 8);
    const rec = await blobBytes(file, z64, z64 + 56);
    if (rec.getUint32(0, true) !== EOCD64) throw new ZipError(`${zipName}: damaged ZIP64 directory`);
    count = u64(rec, 32);
    cdSize = u64(rec, 40);
    cdOffset = u64(rec, 48);
  }
  const cd = await blobBytes(file, cdOffset, cdOffset + cdSize);
  const utf8 = new TextDecoder("utf-8");
  const latin = new TextDecoder("latin1");
  const out: Source[] = [];
  let p = 0;
  for (let i = 0; i < count && p + 46 <= cd.byteLength; i++) {
    if (cd.getUint32(p, true) !== CEN) throw new ZipError(`${zipName}: damaged central directory`);
    const flags = cd.getUint16(p + 8, true);
    const method = cd.getUint16(p + 10, true);
    let csize = cd.getUint32(p + 20, true);
    let usize = cd.getUint32(p + 24, true);
    const nameLen = cd.getUint16(p + 28, true);
    const extraLen = cd.getUint16(p + 30, true);
    const commentLen = cd.getUint16(p + 32, true);
    let local = cd.getUint32(p + 42, true);
    const nameBytes = new Uint8Array(cd.buffer, cd.byteOffset + p + 46, nameLen);
    const name = (flags & 0x800 ? utf8 : latin).decode(nameBytes);
    // ZIP64 extra field: the 64-bit sizes / offset follow in this order, only for fields that overflowed
    let e = p + 46 + nameLen;
    const eEnd = e + extraLen;
    while (e + 4 <= eEnd) {
      const id = cd.getUint16(e, true);
      const len = cd.getUint16(e + 2, true);
      if (id === 1) {
        let q = e + 4;
        if (usize === 0xffffffff) { usize = u64(cd, q); q += 8; }
        if (csize === 0xffffffff) { csize = u64(cd, q); q += 8; }
        if (local === 0xffffffff) { local = u64(cd, q); q += 8; }
      }
      e += 4 + len;
    }
    p = eEnd + commentLen;
    const base = name.split("/").pop() || "";
    if (name.endsWith("/") || name.startsWith("__MACOSX/") || base.startsWith("._") || flags & 1) continue;
    if (method !== 0 && method !== 8) continue;            // other compression methods: not DICOM in practice
    out.push(zipEntry(file, `${zipName}/${name}`, local, csize, usize, method));
  }
  return out;
}

function zipEntry(file: Blob, name: string, local: number, csize: number, usize: number, method: number): Source {
  let dataStart = -1;
  const start = async () => {
    if (dataStart < 0) {
      const h = await blobBytes(file, local, local + 30);
      if (h.getUint32(0, true) !== LOC) throw new ZipError(`${name}: damaged zip entry`);
      dataStart = local + 30 + h.getUint16(26, true) + h.getUint16(28, true);
    }
    return dataStart;
  };
  return {
    name,
    size: usize,
    random: method === 0,
    async read(from = 0, to = usize) {
      to = Math.min(to, usize);
      const s = await start();
      if (method === 0) return file.slice(s + from, s + to).arrayBuffer();
      const bytes = await inflate(file.slice(s, s + csize), to);
      return bytes.buffer.slice(bytes.byteOffset + from, bytes.byteOffset + bytes.length) as ArrayBuffer;
    },
  };
}
