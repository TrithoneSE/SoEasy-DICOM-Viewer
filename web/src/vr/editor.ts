/**
 * Colour-curve editor (dcv/tf_editor.py), for mouse and touch:
 *   drag a point = move it (HU left / right, opacity up / down) · drag a curve line = move the whole curve
 *   (Shift / two fingers vertical: its height) · double tap a line = add a point · double tap a point = its colour ·
 *   double tap empty space = a new curve · drag empty space = pan the HU axis · wheel / pinch = zoom the HU axis ·
 *   the dashed bone-lock line can be dragged (applied on release: the bone area is recomputed).
 * The opacity axis is square-root scaled (opacity = y²): small opacities are easy to set.
 */
import { combine, curveSamples, type RGB, type TFCurve, type TransferFunction } from "./tf";

const NEW_COLORS: RGB[] = [[40, 220, 60], [255, 220, 0], [60, 140, 255], [200, 80, 255], [0, 220, 220], [255, 130, 0], [255, 80, 160]];
const ML = 40, MR = 10, MT = 8, MB = 34;

const css = (c: RGB, a = 1) => `rgba(${Math.round(c[0] * 255)},${Math.round(c[1] * 255)},${Math.round(c[2] * 255)},${a})`;
const hex = (c: RGB) => "#" + c.map((v) => Math.round(v * 255).toString(16).padStart(2, "0")).join("");
const fromHex = (h: string): RGB => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16) / 255) as RGB;

function niceStep(raw: number): number {
  const p = 10 ** Math.floor(Math.log10(raw));
  for (const m of [1, 2, 5, 10]) if (m * p >= raw) return m * p;
  return 10 * p;
}

type Drag = { what: "point" | "curve" | "pan" | "bone" | "height"; x: number; y: number } | null;

export class CurveEditor {
  readonly canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  tf!: TransferFunction;
  gain = 1;
  /** null = no bone-lock marker. */
  boneHu: number | null = null;
  private hist: Float32Array | null = null;
  private dmin = -1024;
  private dmax = 3071;
  private vx0 = -1024;
  private vx1 = 3071;
  sel: { c: number; p: number | null } | null = null;
  private drag: Drag = null;
  private pointers = new Map<number, { x: number; y: number }>();
  private pinch: { d0: number; w0: number; cx: number; my: number } | null = null;
  private lastTap = { t: 0, x: 0, y: 0 };
  private colorIdx = 0;
  private colorInput: HTMLInputElement;
  private hover = "";
  /** fast = a drag is going on (the 3D view renders a lighter table); false = settled. */
  onChange: (fast: boolean) => void = () => undefined;
  onBone: (hu: number) => void = () => undefined;
  onSelect: () => void = () => undefined;

  constructor(readonly root: HTMLElement) {
    this.canvas = document.createElement("canvas");
    this.canvas.className = "ed-canvas";
    root.append(this.canvas);
    this.ctx = this.canvas.getContext("2d")!;
    this.colorInput = document.createElement("input");
    this.colorInput.type = "color";
    this.colorInput.hidden = true;
    root.append(this.colorInput);
    this.colorInput.addEventListener("input", () => {
      const s = this.sel;
      if (!s) return;
      const c = this.tf.curves[s.c];
      const col = fromHex(this.colorInput.value);
      if (s.p === null) for (const q of c.points) q.color = [...col] as RGB;    // a whole curve selected: all points
      else c.points[s.p].color = col;
      this.emit(false);
    });
    new ResizeObserver(() => this.draw()).observe(root);
    const cv = this.canvas;
    cv.addEventListener("pointerdown", (e) => this.down(e));
    cv.addEventListener("pointermove", (e) => this.move(e));
    cv.addEventListener("pointerup", (e) => this.up(e));
    cv.addEventListener("pointercancel", (e) => this.up(e));
    cv.addEventListener("dblclick", (e) => this.double(...this.pos(e)));
    cv.addEventListener("wheel", (e) => {
      e.preventDefault();
      this.zoomAt(this.pos(e)[0], 0.85 ** (-e.deltaY / 100));
    }, { passive: false });
    cv.addEventListener("pointerleave", () => {
      this.hover = "";
      this.draw();
    });
  }

  setData(tf: TransferFunction, lo: number, hi: number, hist: Float32Array | null): void {
    this.tf = tf;
    this.dmin = lo;
    this.dmax = hi;
    this.hist = hist;
    this.sel = null;
    this.fitToCurves();
  }

  setTF(tf: TransferFunction): void {
    this.tf = tf;
    this.sel = null;
    this.fitToCurves();
  }

  /** HU axis around the curves (the whole data range when there are none). */
  fitToCurves(): void {
    const xs = this.tf.curves.flatMap((c) => c.points.map((p) => p.x)).filter((x) => x < 2500);
    let a = xs.length ? Math.min(...xs) : this.dmin, b = xs.length ? Math.max(...xs) : this.dmax;
    if (b - a < 50) [a, b] = [a - 100, b + 100];
    const pad = (b - a) * 0.15;
    this.vx0 = Math.max(this.dmin - 50, a - pad);
    this.vx1 = Math.min(this.dmax + 50, b + pad);
    if (this.vx1 - this.vx0 < 50) [this.vx0, this.vx1] = [this.dmin, this.dmax];
    this.draw();
  }

  // ------------------------------------------------------------ coordinates
  private rect() {
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    return { x: ML, y: MT, w: Math.max(10, w - ML - MR), h: Math.max(10, h - MT - MB) };
  }
  private xToPx(x: number) {
    const r = this.rect();
    return r.x + ((x - this.vx0) / (this.vx1 - this.vx0)) * r.w;
  }
  private pxToX(px: number) {
    const r = this.rect();
    return this.vx0 + ((px - r.x) / r.w) * (this.vx1 - this.vx0);
  }
  private yToPy(y: number) {
    const r = this.rect();
    return r.y + r.h - Math.sqrt(Math.max(0, y)) * r.h;          // square-root axis
  }
  private pyToY(py: number) {
    const r = this.rect();
    const t = Math.min(1, Math.max(0, (r.y + r.h - py) / r.h));
    return t * t;
  }
  private pos(e: MouseEvent): [number, number] {
    const b = this.canvas.getBoundingClientRect();
    return [e.clientX - b.left, e.clientY - b.top];
  }

  // ------------------------------------------------------------ hit tests
  private hitPoint(x: number, y: number, tol: number): { c: number; p: number } | null {
    for (let ci = this.tf.curves.length - 1; ci >= 0; ci--) {
      const c = this.tf.curves[ci];
      for (let pi = 0; pi < c.points.length; pi++) {
        const q = c.points[pi];
        if (Math.hypot(this.xToPx(q.x) - x, this.yToPy(q.y) - y) < tol) return { c: ci, p: pi };
      }
    }
    return null;
  }
  private hitSegment(x: number, y: number, tol: number): { c: number; i: number } | null {
    for (let ci = this.tf.curves.length - 1; ci >= 0; ci--) {
      const pts = this.tf.curves[ci].points;
      for (let i = 0; i + 1 < pts.length; i++) {
        const ax = this.xToPx(pts[i].x), ay = this.yToPy(pts[i].y), bx = this.xToPx(pts[i + 1].x), by = this.yToPy(pts[i + 1].y);
        const dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy;
        const t = l2 ? Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / l2)) : 0;
        if (Math.hypot(x - ax - t * dx, y - ay - t * dy) < tol) return { c: ci, i };
      }
    }
    return null;
  }
  private hitBone(x: number, y: number, tol: number): boolean {
    if (this.boneHu === null) return false;
    const r = this.rect();
    return Math.abs(x - this.xToPx(this.boneHu)) < tol && y >= r.y && y <= r.y + 22;
  }

  // ------------------------------------------------------------ actions
  private emit(fast: boolean) {
    this.draw();
    this.onChange(fast);
  }

  addCurve(x: number, height = 0.3): void {
    const w = (this.vx1 - this.vx0) * 0.08;
    const col = NEW_COLORS[this.colorIdx++ % NEW_COLORS.length].map((v) => v / 255) as RGB;
    this.tf.curves.push({ name: `Curve ${this.tf.curves.length + 1}`, layer: "normal", hide_edges: false, points: [
      { x: x - w, y: 0, color: [...col] as RGB }, { x: x - w / 3, y: height, color: [...col] as RGB },
      { x: x + w / 3, y: height, color: [...col] as RGB }, { x: x + w, y: 0, color: [...col] as RGB }] });
    this.sel = { c: this.tf.curves.length - 1, p: null };
    this.onSelect();
    this.emit(false);
  }

  private addPoint(ci: number, x: number) {
    const c = this.tf.curves[ci];
    let i = c.points.findIndex((p) => p.x > x);
    if (i < 0) i = c.points.length;
    const a = c.points[Math.max(0, i - 1)], b = c.points[Math.min(c.points.length - 1, i)];
    const t = b.x > a.x ? (x - a.x) / (b.x - a.x) : 0;
    c.points.splice(i, 0, { x, y: a.y + (b.y - a.y) * t, color: a.color.map((v, k) => v + (b.color[k] - v) * t) as RGB });
    this.sel = { c: ci, p: i };
    this.onSelect();
    this.emit(false);
  }

  /** Delete the selected point (a curve keeps at least 2), or the selected curve. */
  deleteSelected(): void {
    const s = this.sel;
    if (!s) return;
    const c = this.tf.curves[s.c];
    if (s.p !== null && c.points.length > 2) c.points.splice(s.p, 1);
    else this.tf.curves.splice(s.c, 1);
    this.sel = null;
    this.onSelect();
    this.emit(false);
  }

  pickColor(): void {
    const s = this.sel;
    if (!s) return;
    const c = this.tf.curves[s.c];
    this.colorInput.value = hex((s.p === null ? c.points[Math.floor(c.points.length / 2)] : c.points[s.p]).color);
    this.colorInput.click();
  }

  selectedCurve(): TFCurve | null {
    return this.sel ? this.tf.curves[this.sel.c] ?? null : null;
  }

  private zoomAt(px: number, f: number) {
    const cx = this.pxToX(px);
    const w = Math.min(Math.max((this.vx1 - this.vx0) * f, 10), (this.dmax - this.dmin) * 3);
    const t = (cx - this.vx0) / (this.vx1 - this.vx0);
    this.vx0 = cx - t * w;
    this.vx1 = this.vx0 + w;
    this.draw();
  }

  private double(x: number, y: number) {
    const hp = this.hitPoint(x, y, 14);
    if (hp) {
      this.sel = hp;
      this.onSelect();
      this.pickColor();
      return;
    }
    const hs = this.hitSegment(x, y, 9);
    if (hs) {
      this.addPoint(hs.c, this.pxToX(x));
      return;
    }
    const r = this.rect();
    if (x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h) this.addCurve(this.pxToX(x), Math.max(0.15, this.pyToY(y)));
  }

  // ------------------------------------------------------------ pointer
  private down(e: PointerEvent) {
    e.preventDefault();
    try {
      this.canvas.setPointerCapture(e.pointerId);
    } catch { /* synthetic pointer */ }
    const [x, y] = this.pos(e);
    this.pointers.set(e.pointerId, { x, y });
    if (this.pointers.size === 2) {
      const [p, q] = [...this.pointers.values()];
      this.pinch = { d0: Math.max(10, Math.abs(p.x - q.x)), w0: this.vx1 - this.vx0, cx: this.pxToX((p.x + q.x) / 2), my: (p.y + q.y) / 2 };
      if (this.drag?.what === "curve") this.drag = { what: "height", x, y: (p.y + q.y) / 2 };   // two fingers on a curve: its height
      else this.drag = null;
      return;
    }
    if (this.pointers.size > 2) return;
    const touch = e.pointerType !== "mouse";
    const tol = touch ? 22 : 10;
    // double tap (touch / pen); the mouse has dblclick
    if (touch) {
      const now = performance.now();
      if (now - this.lastTap.t < 320 && Math.hypot(x - this.lastTap.x, y - this.lastTap.y) < 30) {
        this.lastTap.t = 0;
        this.double(x, y);
        return;
      }
      this.lastTap = { t: now, x, y };
    }
    if (e.button === 1) {
      this.drag = { what: "pan", x, y };
      return;
    }
    const hp = this.hitPoint(x, y, tol);
    if (hp) {
      this.sel = hp;
      this.drag = { what: "point", x, y };
    } else if (this.hitBone(x, y, tol)) {
      this.drag = { what: "bone", x, y };
    } else {
      const hs = this.hitSegment(x, y, tol * 0.7);
      if (hs) {
        this.sel = { c: hs.c, p: null };
        this.drag = { what: e.shiftKey ? "height" : "curve", x, y };
      } else {
        this.sel = null;
        this.drag = { what: "pan", x, y };
      }
    }
    this.onSelect();
    this.draw();
  }

  private move(e: PointerEvent) {
    const [x, y] = this.pos(e);
    if (!this.pointers.has(e.pointerId)) {
      if (e.pointerType === "mouse") {
        const hp = this.hitPoint(x, y, 10);
        if (hp) {
          const c = this.tf.curves[hp.c], q = c.points[hp.p];
          this.hover = `${c.name}: HU ${q.x.toFixed(0)}, opacity ${q.y.toFixed(3)}`;
        } else this.hover = `HU ${this.pxToX(x).toFixed(0)}`;
        this.canvas.style.cursor = hp ? "move" : this.hitBone(x, y, 10) ? "ew-resize" : this.hitSegment(x, y, 7) ? "grab" : "";
        this.draw();
      }
      return;
    }
    this.pointers.set(e.pointerId, { x, y });
    if (this.pinch && this.pointers.size >= 2) {
      const [p, q] = [...this.pointers.values()];
      const my = (p.y + q.y) / 2;
      if (this.drag?.what === "height" && this.sel) {
        const f = Math.exp(-(my - this.drag.y) / 80);
        for (const pt of this.tf.curves[this.sel.c].points) pt.y = Math.min(1, Math.max(0, pt.y * f));
        this.drag.y = my;
        this.emit(true);
        return;
      }
      const w = Math.min(Math.max((this.pinch.w0 * this.pinch.d0) / Math.max(10, Math.abs(p.x - q.x)), 10), (this.dmax - this.dmin) * 3);
      const r = this.rect();
      const t = ((p.x + q.x) / 2 - r.x) / r.w;
      this.vx0 = this.pinch.cx - t * w;
      this.vx1 = this.vx0 + w;
      this.draw();
      return;
    }
    const d = this.drag;
    if (!d) return;
    const dx = ((x - d.x) / this.rect().w) * (this.vx1 - this.vx0);
    const s = this.sel;
    if (d.what === "pan") {
      this.vx0 -= dx;
      this.vx1 -= dx;
      this.draw();
    } else if (d.what === "point" && s && s.p !== null) {
      const c = this.tf.curves[s.c];
      const q = c.points[s.p];
      const lo = s.p > 0 ? c.points[s.p - 1].x + 0.5 : -1e9;
      const hi = s.p < c.points.length - 1 ? c.points[s.p + 1].x - 0.5 : 1e9;
      q.x = Math.min(hi, Math.max(lo, this.pxToX(x)));
      q.y = this.pyToY(y);
      this.hover = `${c.name}: HU ${q.x.toFixed(0)}, opacity ${q.y.toFixed(3)}`;
      this.emit(true);
    } else if (d.what === "curve" && s) {
      for (const q of this.tf.curves[s.c].points) q.x += dx;
      this.emit(true);
    } else if (d.what === "height" && s) {
      const f = Math.exp(-(y - d.y) / 80);
      for (const q of this.tf.curves[s.c].points) q.y = Math.min(1, Math.max(0, q.y * f));
      this.emit(true);
    } else if (d.what === "bone") {
      this.boneHu = Math.min(3000, Math.max(100, Math.round(this.pxToX(x) / 10) * 10));
      this.hover = `Bone lock from ${this.boneHu} HU (applied on release)`;
      this.draw();
    }
    d.x = x;
    d.y = y;
  }

  private up(e: PointerEvent) {
    if (!this.pointers.has(e.pointerId)) return;
    this.pointers.delete(e.pointerId);
    if (this.pointers.size < 2) this.pinch = null;
    if (this.pointers.size === 0) {
      const d = this.drag;
      this.drag = null;
      if (d?.what === "bone" && this.boneHu !== null) this.onBone(this.boneHu);
      else if (d && d.what !== "pan") this.onChange(false);
    }
  }

  // ------------------------------------------------------------ drawing
  draw(): void {
    if (!this.tf) return;
    const cv = this.canvas, dpr = window.devicePixelRatio || 1;
    const W = cv.clientWidth, H = cv.clientHeight;
    if (!W || !H) return;
    if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) {
      cv.width = Math.round(W * dpr);
      cv.height = Math.round(H * dpr);
    }
    const ctx = this.ctx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#1c1c20";
    ctx.fillRect(0, 0, W, H);
    const r = this.rect();
    ctx.fillStyle = "#0c0c0e";
    ctx.fillRect(r.x, r.y, r.w, r.h);
    ctx.save();
    ctx.beginPath();
    ctx.rect(r.x, r.y, r.w, r.h);
    ctx.clip();
    // histogram (log counts) over the data range
    if (this.hist) {
      const n = this.hist.length;
      let mx = 0;
      for (const v of this.hist) mx = Math.max(mx, v);
      ctx.fillStyle = "rgba(90,90,100,0.47)";
      ctx.beginPath();
      ctx.moveTo(this.xToPx(this.dmin), r.y + r.h);
      for (let i = 0; i < n; i++) {
        const x = this.dmin + ((i + 0.5) / n) * (this.dmax - this.dmin);
        ctx.lineTo(this.xToPx(x), r.y + r.h - (this.hist[i] / (mx || 1)) * r.h * 0.9);
      }
      ctx.lineTo(this.xToPx(this.dmax), r.y + r.h);
      ctx.fill();
    }
    ctx.restore();
    // grid
    ctx.font = "10px system-ui, -apple-system, sans-serif";
    const step = niceStep((this.vx1 - this.vx0) / Math.max(3, Math.floor(r.w / 70)));
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let x = Math.ceil(this.vx0 / step) * step; x <= this.vx1; x += step) {
      const px = this.xToPx(x);
      ctx.strokeStyle = "#37373e";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(px, r.y);
      ctx.lineTo(px, r.y + r.h);
      ctx.stroke();
      ctx.fillStyle = "#aaaab4";
      ctx.fillText(`${Math.round(x)}`, px, r.y + r.h + 2);
    }
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.setLineDash([2, 3]);
    for (const yv of [0.05, 0.25, 0.5, 1]) {
      const py = this.yToPy(yv);
      ctx.strokeStyle = "#37373e";
      ctx.beginPath();
      ctx.moveTo(r.x, py);
      ctx.lineTo(r.x + r.w, py);
      ctx.stroke();
      ctx.fillStyle = "#aaaab4";
      ctx.fillText(`${yv}`, r.x - 4, py);
    }
    ctx.setLineDash([]);
    // bone-lock threshold: right of the line (plus its closed cavities and rim) is the bone area
    if (this.boneHu !== null && this.boneHu > this.vx0 && this.boneHu < this.vx1) {
      const bx = this.xToPx(this.boneHu);
      ctx.fillStyle = "rgba(230,220,190,0.07)";
      ctx.fillRect(bx, r.y, r.x + r.w - bx, r.h);
      ctx.strokeStyle = "rgb(240,225,170)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([6, 3, 2, 3]);
      ctx.beginPath();
      ctx.moveTo(bx, r.y);
      ctx.lineTo(bx, r.y + r.h);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "rgb(240,225,170)";
      ctx.beginPath();
      ctx.moveTo(bx - 7, r.y);
      ctx.lineTo(bx + 7, r.y);
      ctx.lineTo(bx, r.y + 10);
      ctx.fill();
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      ctx.fillText(`bone area ≥ ${this.boneHu}`, bx + 6, r.y + 12);
    }
    // curves
    ctx.save();
    ctx.beginPath();
    ctx.rect(r.x - 8, r.y - 8, r.w + 16, r.h + 16);
    ctx.clip();
    this.tf.curves.forEach((c, ci) => {
      if (c.points.length < 2) return;
      const pts = c.points.map((q) => [this.xToPx(q.x), this.yToPy(q.y)] as [number, number]);
      const x0 = pts[0][0], x1 = pts[pts.length - 1][0];
      const grad = ctx.createLinearGradient(x0, 0, x1 > x0 ? x1 : x0 + 1, 0);
      c.points.forEach((q, i) => grad.addColorStop(Math.min(1, Math.max(0, x1 === x0 ? 0 : (pts[i][0] - x0) / (x1 - x0))), css(q.color, 0.43)));
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.moveTo(pts[0][0], r.y + r.h);
      for (const [px, py] of pts) ctx.lineTo(px, py);
      ctx.lineTo(pts[pts.length - 1][0], r.y + r.h);
      ctx.fill();
      const selected = this.sel?.c === ci;
      const bone = c.layer === "bone";
      ctx.strokeStyle = selected ? "rgb(255,220,0)" : "rgb(235,235,235)";
      ctx.lineWidth = selected ? 2.2 : 1.3;
      ctx.setLineDash(bone ? [6, 4] : c.layer === "vein" ? [2, 3] : []);
      ctx.beginPath();
      pts.forEach(([px, py], i) => (i ? ctx.lineTo(px, py) : ctx.moveTo(px, py)));
      ctx.stroke();
      ctx.setLineDash([]);
      const top = pts.reduce((a, b) => (b[1] < a[1] ? b : a));
      ctx.fillStyle = selected ? "rgb(255,220,0)" : "rgb(210,210,210)";
      ctx.textAlign = "left";
      ctx.textBaseline = "alphabetic";
      ctx.font = "11px system-ui, -apple-system, sans-serif";
      ctx.fillText((bone ? "bone · " : "") + c.name + (c.hide_edges ? "  · rims hidden" : ""), top[0] + 6, Math.max(r.y + 12, top[1] - 6));
      c.points.forEach((q, pi) => {
        const isSel = selected && this.sel?.p === pi;
        ctx.fillStyle = css(q.color);
        ctx.strokeStyle = isSel ? "rgb(255,220,0)" : "#fff";
        ctx.lineWidth = isSel ? 2 : 1;
        ctx.beginPath();
        ctx.arc(pts[pi][0], pts[pi][1], isSel ? 6.5 : 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      });
    });
    ctx.restore();
    // strip of the merged result
    const n = Math.max(16, Math.floor(r.w));
    const xs = new Float64Array(n);
    for (let i = 0; i < n; i++) xs[i] = this.vx0 + ((i + 0.5) / n) * (this.vx1 - this.vx0);
    const m = combine(curveSamples(this.tf, xs, null), n, true);
    const by = r.y + r.h + 17;
    for (let i = 0; i < n; i++) {
      const k = Math.min(1, Math.sqrt(Math.min(1, m.a[i] * this.gain)) * 1.5);
      ctx.fillStyle = css([m.rgb[i * 3] * k, m.rgb[i * 3 + 1] * k, m.rgb[i * 3 + 2] * k]);
      ctx.fillRect(r.x + (i * r.w) / n, by, r.w / n + 1, 9);
    }
    ctx.strokeStyle = "#50505a";
    ctx.lineWidth = 1;
    ctx.strokeRect(r.x, by, r.w, 9);
    ctx.strokeRect(r.x, r.y, r.w, r.h);
    if (this.hover) {
      ctx.fillStyle = "#fff";
      ctx.textAlign = "right";
      ctx.textBaseline = "top";
      ctx.fillText(this.hover, r.x + r.w - 6, r.y + 4);
    }
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
  }
}
