/**
 * Several 2D viewports side by side (layouts 1×2, 1×3, 2×2), linked like the desktop app:
 *   reference lines  — in every view, where the slice shown in each other view cuts it (same patient space only),
 *                      clipped to the other image, in that view's colour;
 *   3D cursor        — the Locate tool (or Shift + press with any tool): the point touched is found in the other
 *                      series, which jump to the slice through it and mark it with a cross.
 * Everything uses the DICOM geometry of each slice (Image Position / Orientation), so it works between separately
 * acquired series (MRI axial / sagittal / coronal), not only reconstructions of one stack.
 */
import { Volume } from "../dicom/volume";
import { add, col, dot, scale, sub, type V3 } from "../geometry";
import type { ViewHooks, Viewport } from "./viewport";

export const VIEW_COLORS = ["rgb(40,170,255)", "rgb(255,60,150)", "rgb(90,220,120)", "rgb(255,150,40)"];
const CURSOR_COLOR = "rgb(0,255,255)";

/** One displayed slice in patient space: pixel (row, col) centre = o + col·sx·r + row·sy·c (mm, LPS). */
export interface Plane {
  o: V3;
  r: V3;
  c: V3;
  n: V3;
  sx: number;
  sy: number;
  rows: number;
  cols: number;
}

export function planeOf(vol: Volume, k: number): Plane | null {
  const p = vol.positions?.[k];
  if (!p) return null;
  const D = vol.direction;
  return { o: p, r: col(D, 0), c: col(D, 1), n: col(D, 2), sx: vol.spacing[2], sy: vol.spacing[1], rows: vol.rows, cols: vol.cols };
}

/** Patient point of the image coordinates a viewport uses (pixel i spans [i, i + 1)). */
export function worldOf(P: Plane, row: number, column: number): V3 {
  return add(P.o, add(scale(P.r, (column - 0.5) * P.sx), scale(P.c, (row - 0.5) * P.sy)));
}

/** Image coordinates (row, col) of a patient point projected onto the plane, and its distance from it (mm). */
export function imageOf(P: Plane, X: V3): [number, number, number] {
  const d = sub(X, P.o);
  return [dot(d, P.c) / P.sy + 0.5, dot(d, P.r) / P.sx + 0.5, dot(d, P.n)];
}

/**
 * Where plane B cuts the image of plane A: a segment in A's image coordinates [row, col] × 2, clipped to both
 * images; null when they are (nearly) parallel or do not meet.
 */
export function referenceLine(A: Plane, B: Plane): [[number, number], [number, number]] | null {
  // f(col, row) = nB · (X(col, row) − oB) = a·col + b·row + d
  const a = A.sx * dot(B.n, A.r), b = A.sy * dot(B.n, A.c);
  if (Math.hypot(dot(B.n, A.r), dot(B.n, A.c)) < 0.1) return null;          // within ~6° of parallel
  const d = dot(B.n, sub(A.o, B.o)) - 0.5 * a - 0.5 * b;
  // the line across A's image rectangle [0, cols] × [0, rows]
  const pts: [number, number][] = [];
  if (Math.abs(b) > 1e-9) for (const c of [0, A.cols]) pts.push([-(a * c + d) / b, c]);
  if (Math.abs(a) > 1e-9) for (const r of [0, A.rows]) pts.push([r, -(b * r + d) / a]);
  const inside = pts.filter(([r, c]) => r >= -1e-6 && r <= A.rows + 1e-6 && c >= -1e-6 && c <= A.cols + 1e-6);
  if (inside.length < 2) return null;
  let p0 = inside[0], p1 = inside[1], best = -1;
  for (const p of inside) for (const q of inside) {
    const l = Math.hypot(p[0] - q[0], p[1] - q[1]);
    if (l > best) [best, p0, p1] = [l, p, q];
  }
  // then to B's own image: B's coordinates are linear along the segment (Liang–Barsky)
  const [r0, c0] = imageOf(B, worldOf(A, ...p0)), [r1, c1] = imageOf(B, worldOf(A, ...p1));
  let t0 = 0, t1 = 1;
  for (const [v0, v1, hi] of [[r0, r1, B.rows], [c0, c1, B.cols]] as const) {
    for (const [p, q] of [[v0 - v1, v0], [v1 - v0, hi - v0]]) {
      if (Math.abs(p) < 1e-12) {
        if (q < 0) return null;
      } else {
        const t = q / p;
        if (p < 0) t0 = Math.max(t0, t);
        else t1 = Math.min(t1, t);
      }
    }
  }
  if (t1 - t0 < 1e-6) return null;
  const at = (t: number): [number, number] => [p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t];
  return [at(t0), at(t1)];
}

/** Index of the slice of `vol` nearest a patient point, or -1 when the point is outside the stack. */
export function sliceThrough(vol: Volume, X: V3): number {
  if (!vol.positions) return -1;
  const n = col(vol.direction, 2);
  const x = dot(n, X);
  let best = -1, dist = Infinity;
  vol.positions.forEach((p, k) => {
    const e = Math.abs(dot(n, p) - x);
    if (e < dist) [best, dist] = [k, e];
  });
  return dist <= Math.max(vol.spacing[0], 1) ? best : -1;
}

export class Linker {
  showLines = true;
  /** The 3D cursor (patient point) and the space it belongs to. */
  cursor: { X: V3; space: string } | null = null;

  /** `visible`: how many of `views` the layout shows (the first ones). */
  constructor(readonly views: Viewport[], readonly visible: () => number) {}

  private plane(v: Viewport): Plane | null {
    return v.vol instanceof Volume ? planeOf(v.vol, v.index) : null;
  }

  private shown(): Viewport[] {
    return this.views.slice(0, this.visible());
  }

  /** Re-draw the other views after one changed slice / series. */
  changed(from: Viewport): void {
    if (this.visible() < 2) return;
    for (const v of this.shown()) if (v !== from) v.requestRender();
  }

  locate(view: Viewport, x: number, y: number): void {
    const vol = view.vol;
    const P = this.plane(view);
    if (!(vol instanceof Volume) || !P) {
      view.onStatus("This image has no position in the patient (no DICOM geometry)");
      return;
    }
    const X = worldOf(P, ...view.toImage(x, y));
    this.cursor = { X, space: vol.space };
    for (const v of this.shown()) {
      if (v !== view && v.vol instanceof Volume && v.vol.space === vol.space) {
        const k = sliceThrough(v.vol, X);
        if (k >= 0) {
          v.stopCine();
          v.setIndex(k);
        }
      }
      v.requestRender();
    }
  }

  clearCursor(): void {
    if (!this.cursor) return;
    this.cursor = null;
    for (const v of this.views) v.requestRender();
  }

  hooks(view: Viewport): ViewHooks {
    return {
      down: (x, y, e) => {
        if (view.tool !== "locate" && !e.shiftKey) return false;
        this.locate(view, x, y);
        return true;
      },
      drag: (x, y) => this.locate(view, x, y),
      overlay: (ctx) => this.overlay(view, ctx),
    };
  }

  private overlay(view: Viewport, ctx: CanvasRenderingContext2D): void {
    const shown = this.shown();
    if (shown.length < 2) return;
    const me = shown.indexOf(view);
    const vol = view.vol;
    // the view's own colour: a small tab at the top, so a line can be matched with its view
    const W = view.canvas.clientWidth;
    ctx.fillStyle = VIEW_COLORS[me];
    ctx.fillRect(W / 2 - 14, 0, 28, 4);
    const A = this.plane(view);
    if (!A || !(vol instanceof Volume)) return;
    if (this.showLines) {
      shown.forEach((o, i) => {
        if (o === view || !(o.vol instanceof Volume) || o.vol.space !== vol.space) return;
        const B = this.plane(o);
        const seg = B && referenceLine(A, B);
        if (!seg) return;
        const [x0, y0] = view.toScreen(...seg[0]), [x1, y1] = view.toScreen(...seg[1]);
        ctx.strokeStyle = VIEW_COLORS[i];
        ctx.lineWidth = o.active ? 1.6 : 1;
        ctx.setLineDash(o.active ? [] : [6, 4]);
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
        ctx.stroke();
      });
      ctx.setLineDash([]);
    }
    const cur = this.cursor;
    if (cur && cur.space === vol.space) {
      const [r, c, dist] = imageOf(A, cur.X);
      if (Math.abs(dist) <= Math.max(vol.spacing[0], 1) * 0.75) {
        const [x, y] = view.toScreen(r, c);
        ctx.strokeStyle = CURSOR_COLOR;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
          ctx.moveTo(x + dx * 4, y + dy * 4);
          ctx.lineTo(x + dx * 14, y + dy * 14);
        }
        ctx.stroke();
      }
    }
  }
}
