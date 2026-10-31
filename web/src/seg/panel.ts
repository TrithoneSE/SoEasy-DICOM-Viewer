/**
 * The range editor of a segment: a histogram of the voxels around the seed with two handles (min / max). Dragging one
 * grows the segment again from the same seed, as often as the device keeps up.
 */
import type { Volume } from "../dicom/volume";
import { dragRange, fmtMl, histColumns } from "./segment";
import type { Segment, Segments } from "./segments";

export class SegmentPanel {
  seg: Segment | null = null;
  private vol: Volume | null = null;
  private view0 = 0;
  private view1 = 1;
  private drag: "lo" | "hi" | null = null;
  private pending: [number, number] | null = null;
  private busy = false;
  lastGrowMs = 0;
  /** After a drag ended: the final mask (the 3D view may polish its surface). */
  onSettled: () => void = () => undefined;

  private readonly root: HTMLElement;
  private readonly canvas: HTMLCanvasElement;

  constructor(private readonly segs: Segments) {
    const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
    this.root = $("seg");
    this.canvas = $<HTMLCanvasElement>("seg-hist");
    this.canvas.addEventListener("pointerdown", (e) => this.down(e));
    this.canvas.addEventListener("pointermove", (e) => this.move(e));
    this.canvas.addEventListener("pointerup", () => this.up());
    this.canvas.addEventListener("pointercancel", () => this.up());
    $("seg-done").addEventListener("click", () => this.close());
    $("seg-hide").addEventListener("click", () => {
      if (!this.seg) return;
      this.segs.setVisible(this.seg, !this.seg.visible);
      this.sync();
    });
    $("seg-del").addEventListener("click", () => {
      if (this.seg && this.vol) this.segs.remove(this.vol, this.seg);
      this.close();
    });
    new ResizeObserver(() => this.draw()).observe(this.canvas);
  }

  /** A range is being dragged, or its last grows are still coming. */
  get dragging(): boolean {
    return !!this.drag || !!this.pending || this.busy;
  }

  get isOpen(): boolean {
    return !this.root.hidden;
  }

  open(vol: Volume, seg: Segment): void {
    this.vol = vol;
    this.seg = seg;
    this.fitView();
    this.root.hidden = false;
    this.sync();
  }

  close(): void {
    this.root.hidden = true;
    this.seg = null;
    this.vol = null;
    this.segs.release();
  }

  /** Around the range: wide enough to see what lies beside it, inside what the histogram covers. */
  private fitView(): void {
    const s = this.seg!;
    const c = (s.lo + s.hi) / 2;
    const half = Math.max(100, (s.hi - s.lo) * 2);
    this.view0 = Math.max(s.hist.min, c - half);
    this.view1 = Math.min(s.hist.max, c + half);
    if (this.view1 - this.view0 < 1) [this.view0, this.view1] = [s.hist.min, s.hist.max];
  }

  private xOf(v: number, w: number): number {
    return ((v - this.view0) / (this.view1 - this.view0)) * w;
  }

  private down(e: PointerEvent): void {
    const s = this.seg;
    if (!s) return;
    e.preventDefault();
    try {
      this.canvas.setPointerCapture(e.pointerId);
    } catch { /* synthetic pointer */ }
    const b = this.canvas.getBoundingClientRect();
    const x = e.clientX - b.left;
    const a = this.xOf(s.lo, b.width), c = this.xOf(s.hi, b.width);
    this.drag = Math.abs(x - a) <= Math.abs(x - c) ? "lo" : "hi";
    this.apply(x, b.width);
  }

  private move(e: PointerEvent): void {
    if (!this.drag) return;
    const b = this.canvas.getBoundingClientRect();
    this.apply(e.clientX - b.left, b.width);
  }

  private up(): void {
    if (!this.drag) return;
    this.drag = null;
    const s = this.seg;
    if (s) {
      // a handle at the edge of the view: show more of the histogram for the next drag
      const margin = (this.view1 - this.view0) * 0.04;
      if (s.lo < this.view0 + margin || s.hi > this.view1 - margin) {
        const grow = (this.view1 - this.view0) * 0.5;
        if (s.lo < this.view0 + margin) this.view0 = Math.max(s.hist.min, this.view0 - grow);
        if (s.hi > this.view1 - margin) this.view1 = Math.min(s.hist.max, this.view1 + grow);
      }
      this.draw();
    }
    if (!this.busy) this.onSettled();
  }

  private apply(x: number, w: number): void {
    const s = this.seg;
    if (!s) return;
    const v = Math.round(this.view0 + (Math.max(0, Math.min(w, x)) / w) * (this.view1 - this.view0));
    const [lo, hi] = dragRange(this.drag!, v, s.lo, s.hi, s.seedValue);
    this.pending = [lo, hi];
    this.draw(lo, hi);
    this.pump();
  }

  /** One grow at a time, with a pause as long as the last one took: the page stays responsive on a phone. */
  private pump(): void {
    if (this.busy || !this.pending) return;
    this.busy = true;
    requestAnimationFrame(() => {
      const p = this.pending, s = this.seg, vol = this.vol;
      this.pending = null;
      if (p && s && vol) {
        const t0 = performance.now();
        this.segs.setRange(vol, s, p[0], p[1]);
        this.lastGrowMs = performance.now() - t0;
        this.sync();
      }
      setTimeout(() => {
        this.busy = false;
        if (this.pending) this.pump();
        else if (!this.drag) this.onSettled();
      }, Math.min(300, this.lastGrowMs));
    });
  }

  private sync(): void {
    const s = this.seg;
    if (!s) return;
    const unit = this.vol?.modality === "CT" ? " HU" : "";
    (document.getElementById("seg-name") as HTMLElement).textContent = s.name;
    (document.getElementById("seg-dot") as HTMLElement).style.background = `rgb(${s.color.join(",")})`;
    (document.getElementById("seg-ml") as HTMLElement).textContent = s.grown ? fmtMl(s.ml) : "The seed is outside this range";
    (document.getElementById("seg-range") as HTMLElement).textContent = `${Math.round(s.lo)} to ${Math.round(s.hi)}${unit}`;
    (document.getElementById("seg-hide") as HTMLElement).textContent = s.visible ? "Hide" : "Show";
    this.draw();
  }

  private draw(lo = this.seg?.lo, hi = this.seg?.hi): void {
    const s = this.seg;
    const c = this.canvas;
    if (!s || lo === undefined || hi === undefined || this.root.hidden) return;
    const dpr = window.devicePixelRatio || 1;
    const W = c.clientWidth, H = c.clientHeight;
    if (!W || !H) return;
    if (c.width !== Math.round(W * dpr) || c.height !== Math.round(H * dpr)) {
      c.width = Math.round(W * dpr);
      c.height = Math.round(H * dpr);
    }
    const ctx = c.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#101216";
    ctx.fillRect(0, 0, W, H);
    const cols = histColumns(s.hist, this.view0, this.view1, Math.max(8, Math.round(W / 2)));
    const a = this.xOf(lo, W), b = this.xOf(hi, W);
    const bw = W / cols.length;
    for (let i = 0; i < cols.length; i++) {
      const x = i * bw;
      ctx.fillStyle = x + bw > a && x < b ? `rgb(${s.color.join(",")})` : "#59606c";
      const h = cols[i] * (H - 6);
      ctx.fillRect(x, H - h, bw + 0.5, h);
    }
    for (const x of [a, b]) {
      ctx.fillStyle = "#fff";
      ctx.fillRect(x - 1, 0, 2, H);
      ctx.beginPath();
      ctx.arc(x, H / 2, 7, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "#000";
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }
}
