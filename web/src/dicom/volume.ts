/**
 * A loaded series: every frame / slice in one typed array (Int16 when the rescaled values fit, like the desktop
 * app; else Float32), filled progressively. The first frame shown (and the DSA mask frames) are decoded first, the
 * rest in the background; a frame asked for jumps the queue (dcv/loader.py FrameLoader). Compressed file bytes are
 * dropped once decoded, so memory ≈ the decoded pixels.
 */
import { directionFromIop, type M3, type V3 } from "../geometry";
import { openFrames, type FrameReader } from "./frames";
import { normal, parse, readHeader, type DsaHint, type Header } from "./header";
import { decodeFrame, frameInfo, frameRanges, gather, isNative, NO_CODECS, unsupportedReason, type Decoded } from "./pixels";
import { decodePool } from "./pool";
import type { Instance, Series } from "./scan";

export type Pixels = Int16Array | Float32Array;

export class Volume {
  readonly n: number;
  readonly rows: number;
  readonly cols: number;
  data: Pixels;
  rgb: Uint8Array | null;
  /** (slice, row, column) spacing in mm; row / column null-safe: 1 when the header has none. */
  readonly spacing: [number, number, number];
  readonly spacingKnown: boolean;
  readonly spacingSource: Header["spacingSource"];
  readonly frameTimeMs: number | null;
  readonly ww: number | null;
  readonly wl: number | null;
  readonly modality: string;
  readonly description: string;
  readonly patient: string;
  readonly monochrome1: boolean;
  readonly dsaHint: DsaHint | null;
  readonly multiframe: boolean;
  /** Image axes (column index, row index, slice index) → patient LPS, row-major 3×3. */
  readonly direction: M3;
  /** Image Position (Patient) of each slice in display order (null: no geometry, or a multiframe run). */
  readonly positions: V3[] | null;
  /** The series it was loaded from: segments stay with it when the decoded pixels are dropped for memory. */
  readonly series: Series;
  /** Patient coordinate system: the Frame of Reference UID (else the study): reference lines only within one. */
  readonly space: string;
  /** 0 = not yet, 1 = decoded, 2 = failed (shown black). */
  readonly ready: Uint8Array;
  readyCount = 0;
  errors: string[] = [];
  onFrame: ((k: number) => void) | null = null;
  private loader: Loader | null;

  constructor(s: Series, items: Instance[], first: number) {
    const h = items[Math.min(items.length - 1, Math.max(0, s.multiframe ? 0 : first))].h;
    this.series = s;
    this.multiframe = s.multiframe;
    this.n = s.multiframe ? h.frames : items.length;
    this.rows = h.rows;
    this.cols = h.cols;
    const N = this.rows * this.cols;
    this.rgb = h.samples === 3 ? new Uint8Array(this.n * N * 3) : null;
    this.data = this.rgb ? new Int16Array(0) : storageFor(items).alloc(this.n * N);
    this.spacingKnown = !!h.spacing;
    this.spacingSource = h.spacingSource;
    this.spacing = [sliceSpacing(s, items), h.spacing?.[0] ?? 1, h.spacing?.[1] ?? 1];
    this.frameTimeMs = h.frameTimeMs;
    this.ww = h.ww && h.ww > 1 ? h.ww : null;
    this.wl = this.ww !== null ? h.wl : null;
    this.modality = h.modality;
    this.description = s.description;
    this.patient = h.patientName;
    this.monochrome1 = h.photometric === "MONOCHROME1";
    this.dsaHint = s.multiframe ? h.dsa : null;
    this.direction = directionFromIop(items[0].h.iop);
    this.positions = !s.multiframe && items.every((i) => i.h.iop && i.h.ipp) ? items.map((i) => i.h.ipp!.slice(0, 3) as V3) : null;
    this.space = h.frameOfRef || h.studyUid;
    this.ready = new Uint8Array(this.n);
    this.loader = new Loader(this, s, items, first);
  }

  frame(k: number): Pixels {
    const N = this.rows * this.cols;
    return this.data.subarray(k * N, (k + 1) * N);
  }

  rgbFrame(k: number): Uint8Array | null {
    const N = this.rows * this.cols * 3;
    return this.rgb ? this.rgb.subarray(k * N, (k + 1) * N) : null;
  }

  isReady(k: number): boolean {
    return this.ready[k] !== 0;
  }

  get complete(): boolean {
    return this.readyCount >= this.n;
  }

  /** Decode these frames before any other; resolves when all of them are there. */
  want(ks: number[]): Promise<void> {
    return this.loader ? this.loader.want(ks) : Promise.resolve();
  }

  /** [min, max] of the decoded data (cached once every frame is in). */
  valueRange(): [number, number] {
    if (this.range && this.complete) return this.range;
    let lo = Infinity, hi = -Infinity;
    const d = this.data;
    for (let i = 0; i < d.length; i++) {
      const v = d[i];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    if (!d.length) [lo, hi] = [0, 255];
    this.range = [lo, hi];
    return this.range;
  }
  private range: [number, number] | null = null;

  /** Bytes of decoded pixels held. */
  get bytes(): number {
    return this.data.byteLength + (this.rgb?.byteLength ?? 0);
  }

  dispose(): void {
    this.loader?.stop();
    this.loader = null;
    this.data = new Int16Array(0);
    this.rgb = null;
    this.onFrame = null;
  }

  /** Rescaled grey values into frame k. Int16 storage turns into Float32 (once, for the whole series) when a
   *  value does not fit — like dcv/loader.py, which decides on the actual values, not on Bits Stored. */
  private storeGray(k: number, g: ArrayLike<number>, slope: number, intercept: number): void {
    const N = this.rows * this.cols;
    const out = this.frame(k);
    if (out instanceof Int16Array && Number.isInteger(slope) && Number.isInteger(intercept)) {
      let lo = 0, hi = 0;
      for (let i = 0; i < N; i++) {
        const v = g[i] * slope + intercept;
        out[i] = v;
        if (v < lo) lo = v;
        else if (v > hi) hi = v;
      }
      if (lo >= -32768 && hi <= 32767) return;
    }
    if (this.data instanceof Int16Array) this.data = Float32Array.from(this.data);
    const f = this.frame(k);
    for (let i = 0; i < N; i++) f[i] = g[i] * slope + intercept;
  }

  /** @internal called by the loader */
  store(k: number, d: Decoded | null, h: Header | null, err?: string): void {
    if (this.ready[k]) return;
    const N = this.rows * this.cols;
    if (d && h) {
      if (this.rgb && d.rgb) this.rgb.set(d.rgb.subarray(0, N * 3), k * N * 3);
      else if (d.gray && !this.rgb) this.storeGray(k, d.gray, h.slope, h.intercept);
    }
    if (err) this.errors.push(`Image ${k + 1}: ${err}`);
    this.ready[k] = d ? 1 : 2;
    this.readyCount++;
    if (this.complete) this.loader = null;          // drops the file bytes of a multiframe run
    this.onFrame?.(k);
  }
}

/** Int16 for integer Rescale Slope / Intercept (what CT, MR and XA use), else Float32. */
function storageFor(items: Instance[]): { alloc: (n: number) => Pixels } {
  const small = items.every(({ h }) => Number.isInteger(h.slope) && Number.isInteger(h.intercept));
  return { alloc: (n) => (small ? new Int16Array(n) : new Float32Array(n)) };
}

function sliceSpacing(s: Series, items: Instance[]): number {
  const h = items[0].h;
  const fallback = h.spacingBetweenSlices ?? h.sliceThickness ?? 1;
  if (s.multiframe || items.length < 2 || !items.every((i) => i.h.iop && i.h.ipp)) return fallback || 1;
  const d = items.slice(1).map((it, i) => it.key - items[i].key).sort((a, b) => a - b);
  return Math.abs(d[Math.floor(d.length / 2)]) || fallback || 1;
}

/**
 * Slices of a stack in display order: the most common image size, the most common orientation (a console reformat
 * starts with a reference image of another plane), sorted along the normal without duplicate positions — or by
 * instance number when there is no geometry.
 */
export function stackOrder(instances: Instance[]): Instance[] {
  const sizes = new Map<string, number>();
  for (const i of instances) sizes.set(`${i.h.rows}x${i.h.cols}`, (sizes.get(`${i.h.rows}x${i.h.cols}`) ?? 0) + 1);
  const size = [...sizes.entries()].sort((a, b) => b[1] - a[1])[0][0];
  let items = instances.filter((i) => `${i.h.rows}x${i.h.cols}` === size);
  if (items.every((i) => i.h.iop)) {
    const ns = items.map((i) => normal(i.h.iop!));
    let bestKeep: boolean[] | null = null;
    let bestCount = -1;
    const seen = new Set<string>();
    for (const n of ns) {
      const sig = n.map((v) => v.toFixed(2)).join(",");
      if (seen.has(sig)) continue;
      seen.add(sig);
      const keep = ns.map((m) => Math.abs(n[0] * m[0] + n[1] * m[1] + n[2] * m[2]) > 0.999);   // within ~2.5°
      const count = keep.filter(Boolean).length;
      if (count > bestCount) [bestKeep, bestCount] = [keep, count];
    }
    if (bestKeep) items = items.filter((_, i) => bestKeep![i]);
  }
  if (items.every((i) => i.h.iop && i.h.ipp)) {
    items = [...items].sort((a, b) => a.key - b.key);
    const uniq = [items[0]];
    for (const it of items.slice(1)) if (Math.abs(it.key - uniq[uniq.length - 1].key) > 1e-3) uniq.push(it);
    return uniq;
  }
  return [...items].sort((a, b) => a.h.instanceNumber - b.h.instanceNumber);
}

/** Open a series: resolves once frame `first` (default: 1st frame of a run, middle slice of a stack) is shown. */
export async function openSeries(s: Series): Promise<{ vol: Volume; first: number }> {
  const items = s.multiframe ? s.instances.slice(0, 1) : stackOrder(s.instances);
  const h = items[0].h;
  const why = unsupportedReason(h);
  if (why) throw new Error(why);
  const n = s.multiframe ? h.frames : items.length;
  const dynamic = s.multiframe && h.frameTimeMs !== null;
  const first = dynamic || s.multiframe ? 0 : Math.floor(n / 2);   // a run starts at its first frame
  const vol = new Volume(s, items, first);
  const firstNeeded = [first, ...(vol.dsaHint?.mask ?? (s.multiframe && n > 1 ? [1] : []))];
  await vol.want(firstNeeded);
  return { vol, first };
}

class Loader {
  private todo: number[];
  private urgent: number[] = [];
  private busy = new Set<number>();
  private waiters: { ks: number[]; resolve: () => void }[] = [];
  private stopped = false;
  private file: Promise<FrameReader> | null = null;     // a multiframe run (its bytes, when held whole)
  private readonly parallel: number;

  constructor(private vol: Volume, private s: Series, private items: Instance[], first: number) {
    const n = vol.n;
    this.todo = Array.from({ length: n }, (_, k) => k);
    if (!s.multiframe) this.todo.sort((a, b) => Math.abs(a - first) - Math.abs(b - first));   // outward from the middle
    const compressed = !isNative(items[0].h.transferSyntax);
    this.parallel = compressed ? decodePool().size + 1 : 6;
  }

  want(ks: number[]): Promise<void> {
    const missing = ks.filter((k) => k >= 0 && k < this.vol.n && !this.vol.isReady(k));
    if (!missing.length) {
      this.pump();
      return Promise.resolve();
    }
    this.urgent = [...missing, ...this.urgent.filter((k) => !missing.includes(k))];
    const p = new Promise<void>((resolve) => this.waiters.push({ ks: missing, resolve }));
    this.pump();
    return p;
  }

  stop(): void {
    this.stopped = true;
    this.file = null;
    for (const w of this.waiters) w.resolve();
    this.waiters = [];
  }

  private next(): number | null {
    for (const list of [this.urgent, this.todo]) {
      while (list.length) {
        const k = list.shift()!;
        if (!this.vol.isReady(k) && !this.busy.has(k)) return k;
      }
    }
    return null;
  }

  private pump(): void {
    while (!this.stopped && this.busy.size < this.parallel) {
      const k = this.next();
      if (k === null) return;
      const urgent = this.waiters.some((w) => w.ks.includes(k));
      this.busy.add(k);
      this.load(k, urgent).then(
        ([d, h]) => this.done(k, d, h),
        (e) => this.done(k, null, null, String((e as Error)?.message ?? e)),
      );
    }
  }

  private done(k: number, d: Decoded | null, h: Header | null, err?: string) {
    this.busy.delete(k);
    if (this.stopped) return;
    this.vol.store(k, d, h, err);
    this.waiters = this.waiters.filter((w) => {
      if (w.ks.every((j) => this.vol.isReady(j))) {
        w.resolve();
        return false;
      }
      return true;
    });
    if (this.vol.complete) this.file = null;
    this.pump();
  }

  private multiframe() {
    return (this.file ??= openFrames(this.items[0].src));
  }

  private async load(k: number, urgent: boolean): Promise<[Decoded, Header]> {
    let bytes: Uint8Array;
    let h: Header;
    if (this.s.multiframe) {
      const f = await this.multiframe();
      h = f.h;
      bytes = await f.frame(k);
    } else {
      const buf = await this.items[k].src.read();
      const ds = parse(buf, false);
      h = readHeader(ds);
      if (h.rows !== this.vol.rows || h.cols !== this.vol.cols) throw new Error("image size differs from the series");
      bytes = gather(buf, frameRanges(ds, h, 0));
    }
    if (this.stopped) throw new Error("stopped");
    const info = frameInfo(h);
    const d = isNative(h.transferSyntax) ? await decodeFrame(bytes, info, NO_CODECS)
      : await decodePool().decode(bytes, info, urgent);
    return [d, h];
  }
}
