/**
 * The segments of each series: seed, range and mask, kept with the series so they survive when its decoded pixels are
 * dropped for memory. Also paints them over a slice or any plane (2D, MPR, collage).
 */
import type { Volume } from "../dicom/volume";
import type { Series } from "../dicom/scan";
import { RegionGrower, volumeMl, type Grown } from "./segment";

export const SEG_COLORS: [number, number, number][] = [[255, 90, 70], [70, 190, 255], [120, 220, 90], [255, 200, 60], [200, 120, 255], [255, 130, 190]];

export interface Segment {
  id: number;
  name: string;
  color: [number, number, number];
  visible: boolean;
  seed: [number, number, number];     // (z, y, x)
  seedValue: number;
  lo: number;
  hi: number;
  grown: Grown | null;
  ml: number;
  /** Bumped when the mask or the colour changed (the 3D view uploads it again). */
  rev: number;
  hist: { counts: Float32Array; min: number; max: number };
}

let nextId = 1;

export class Segments {
  private bySeries = new WeakMap<Series, Segment[]>();
  private grower: { seg: Segment; g: RegionGrower } | null = null;
  /** Bumped by every change; the painted layers follow it. */
  version = 0;
  onChanged: () => void = () => undefined;

  list(vol: Volume | null): Segment[] {
    return vol ? this.listOf(vol.series) : [];
  }

  private listOf(s: Series): Segment[] {
    let l = this.bySeries.get(s);
    if (!l) this.bySeries.set(s, (l = []));
    return l;
  }

  private growerFor(vol: Volume, seg: Segment): RegionGrower {
    if (this.grower?.seg === seg && this.grower.g.data === vol.data) return this.grower.g;
    const [sz, sy, sx] = vol.spacing;
    const g = new RegionGrower(vol.data, vol.cols, vol.rows, vol.n, [sz, sy, sx], seg.seed);
    this.grower = { seg, g };
    return g;
  }

  /** A new segment from a seed (z, y, x), range from the voxels around it. Null when nothing grows there. */
  seed(vol: Volume, seed: [number, number, number]): Segment | null {
    const [sz, sy, sx] = vol.spacing;
    const g = new RegionGrower(vol.data, vol.cols, vol.rows, vol.n, [sz, sy, sx], seed);
    const r = g.autoGrow();
    if (!r.grown) return null;
    const list = this.list(vol);
    const n = list.length;
    const seg: Segment = {
      id: nextId++, name: `Segment ${n + 1}`, color: SEG_COLORS[n % SEG_COLORS.length], visible: true, seed, seedValue: vol.data[(seed[0] * vol.rows + seed[1]) * vol.cols + seed[2]], lo: r.lo, hi: r.hi,
      grown: r.grown, ml: volumeMl(r.grown.count, [sz, sy, sx]), rev: 1, hist: g.histogram(),
    };
    list.push(seg);
    this.grower = { seg, g };
    this.touch();
    return seg;
  }

  /** Grow again with another range; the seed stays. */
  setRange(vol: Volume, seg: Segment, lo: number, hi: number): void {
    seg.lo = lo;
    seg.hi = hi;
    seg.grown = this.growerFor(vol, seg).grow(lo, hi);
    seg.ml = seg.grown ? volumeMl(seg.grown.count, [vol.spacing[0], vol.spacing[1], vol.spacing[2]]) : 0;
    seg.rev++;
    this.touch();
  }

  remove(vol: Volume, seg: Segment): void {
    const l = this.list(vol);
    const i = l.indexOf(seg);
    if (i >= 0) l.splice(i, 1);
    if (this.grower?.seg === seg) this.grower = null;
    this.touch();
  }

  setVisible(seg: Segment, on: boolean): void {
    seg.visible = on;
    this.touch();
  }

  /** Editing finished: the search block (up to tens of MB) is not needed any more. */
  release(): void {
    this.grower = null;
  }

  private touch(): void {
    this.version++;
    this.onChanged();
  }

  /** RGBA of the segments on image k, with stronger edges, in `img` (cols × rows). False when none is on it. */
  paintSlice(vol: Volume, k: number, img: ImageData): boolean {
    const px = new Uint32Array(img.data.buffer);
    const W = img.width;
    let any = false;
    for (const seg of this.list(vol)) {
      const g = seg.grown;
      if (!g || !seg.visible || k < g.off[0] || k >= g.off[0] + g.dims[0]) continue;
      if (!any) px.fill(0);
      any = true;
      const [, h, w] = g.dims;
      const [, oy, ox] = g.off;
      const base = (k - g.off[0]) * h;
      const fill = rgba(seg.color, 0.38), edge = rgba(seg.color, 0.95);
      for (let y = 0; y < h; y++) {
        const row = (base + y) * w, out = (oy + y) * W + ox;
        for (let x = 0; x < w; x++) {
          if (!g.sub[row + x]) continue;
          const inner = x > 0 && x < w - 1 && y > 0 && y < h - 1 && g.sub[row + x - 1] && g.sub[row + x + 1] && g.sub[row + x - w] && g.sub[row + x + w];
          px[out + x] = inner ? fill : edge;
        }
      }
    }
    return any;
  }

  /**
   * The segments on a plane: pixel (r, c) is at voxel coordinates o + c·du + r·dv (x, y, z of the volume, voxel i
   * centred at i). `img` = the plane's size. False when none is on it.
   */
  paintPlane(vol: Volume, o: [number, number, number], du: [number, number, number], dv: [number, number, number], img: ImageData): boolean {
    const segs = this.list(vol).filter((s) => s.grown && s.visible);
    if (!segs.length) return false;
    const px = new Uint32Array(img.data.buffer);
    px.fill(0);
    let any = false;
    const W = img.width, H = img.height;
    for (const seg of segs) {
      const g = seg.grown!;
      const [d, h, w] = g.dims, [oz, oy, ox] = g.off;
      const color = rgba(seg.color, 0.45);
      for (let r = 0; r < H; r++) {
        let X = o[0] + dv[0] * r, Y = o[1] + dv[1] * r, Z = o[2] + dv[2] * r;
        const row = r * W;
        for (let c = 0; c < W; c++, X += du[0], Y += du[1], Z += du[2]) {
          const x = Math.round(X) - ox, y = Math.round(Y) - oy, z = Math.round(Z) - oz;
          if (x < 0 || y < 0 || z < 0 || x >= w || y >= h || z >= d) continue;
          if (g.sub[(z * h + y) * w + x]) {
            px[row + c] = color;
            any = true;
          }
        }
      }
    }
    return any;
  }
}

function rgba(c: [number, number, number], a: number): number {
  return ((Math.round(a * 255) << 24) | (c[2] << 16) | (c[1] << 8) | c[0]) >>> 0;
}

/** A canvas layer over a view: repainted only when the key changes. */
export class Layer {
  private canvas = document.createElement("canvas");
  private img: ImageData | null = null;
  private key = "";
  private shown = false;

  /** `paint` fills the image and says whether anything is on it. */
  get(key: string, w: number, h: number, paint: (img: ImageData) => boolean): HTMLCanvasElement | null {
    if (key !== this.key || !this.img || this.img.width !== w || this.img.height !== h) {
      this.key = key;
      if (!this.img || this.img.width !== w || this.img.height !== h) {
        this.img = new ImageData(w, h);
        this.canvas.width = w;
        this.canvas.height = h;
      }
      this.shown = paint(this.img);
      if (this.shown) this.canvas.getContext("2d")!.putImageData(this.img, 0, 0);
    }
    return this.shown ? this.canvas : null;
  }
}
