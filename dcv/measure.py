"""2D measurement tools: length, ellipse, rectangle, polygon, open polygon, angle, Cobb, deviation, arrow, pencil, text."""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPolygonF

# number of points needed (None = free, finished by double-click / mouse release)
REQUIRED = {"length": 2, "arrow": 2, "ellipse": 2, "rect": 2, "angle": 3, "cobb": 4, "deviation": 3,
            "polygon": None, "openpoly": None, "pencil": None, "text": 1}
TOOL_NAMES = {
    "length": ("Length", "L"), "ellipse": ("Ellipse (ROI)", "E"), "rect": ("Rectangle (ROI)", "U"),
    "polygon": ("Closed polygon (ROI)", "G"), "openpoly": ("Open polygon", "O"),
    "angle": ("Angle", "A"), "cobb": ("Cobb angle", "C"), "deviation": ("Deviation (e.g. midline shift)", "D"),
    "arrow": ("Arrow", "Y"), "pencil": ("Pencil / freehand (ROI)", "P"), "text": ("Text", "T"),
}
ROI_KINDS = ("ellipse", "rect", "polygon", "pencil")
COLOR = QColor(255, 225, 0)
COLOR_SEL = QColor(0, 230, 255)


def fmt_len(mm):
    """Length label: mm below 1 cm ("8.4 mm"), cm from 1 cm up ("3.27 cm")."""
    return f"{mm:.1f} mm" if round(abs(mm), 1) < 10 else f"{mm / 10:.2f} cm"


def fmt_dims(a_mm, b_mm):
    """Two sizes in one unit: "8.4 × 6.1 mm" or "4.52 × 3.10 cm"."""
    if round(max(abs(a_mm), abs(b_mm)), 1) < 10:
        return f"{a_mm:.1f} × {b_mm:.1f} mm"
    return f"{a_mm / 10:.2f} × {b_mm / 10:.2f} cm"


def fmt_area(mm2):
    """Area label: mm² below 1 cm², cm² from 1 cm² up."""
    return f"{mm2:.1f} mm²" if round(mm2, 1) < 100 else f"{mm2 / 100:.2f} cm²"


class Measurement:
    def __init__(self, kind, pts, text=""):
        self.kind = kind
        self.pts = [tuple(p) for p in pts]   # (r, c) continuous image coordinates
        self.text = text
        self._stats_key = None
        self._stats = None
        self.label_offset = (0.0, 0.0)     # screen pixels the user dragged the label away from its default spot
        self._label_box = None             # last painted label rectangle (widget coordinates), for hit tests

    # ------------------------------------------------------------ geometri
    def shape_path(self):
        """QPainterPath in image coordinates (x=c, y=r) for ROIs."""
        path = QPainterPath()
        if self.kind == "ellipse":
            (r0, c0), (r1, c1) = self.pts[:2]
            path.addEllipse(QRectF(QPointF(min(c0, c1), min(r0, r1)), QPointF(max(c0, c1), max(r0, r1))))
        elif self.kind == "rect":
            (r0, c0), (r1, c1) = self.pts[:2]
            path.addRect(QRectF(QPointF(min(c0, c1), min(r0, r1)), QPointF(max(c0, c1), max(r0, r1))))
        else:
            path.addPolygon(QPolygonF([QPointF(c, r) for r, c in self.pts]))
            path.closeSubpath()
        return path

    def roi_stats(self, plane, sr, sc):
        key = (tuple(self.pts), plane.shape, id(plane))
        if self._stats_key == key:
            return self._stats
        h, w = plane.shape
        img = QImage(w, h, QImage.Format_Grayscale8)
        img.fill(0)
        p = QPainter(img)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(255, 255, 255))
        p.drawPath(self.shape_path())
        p.end()
        buf = np.frombuffer(img.constBits(), np.uint8).reshape(h, img.bytesPerLine())[:, :w]
        m = buf > 127
        if m.sum() == 0:
            self._stats = None
        else:
            v = plane[m].astype(np.float64)
            self._stats = dict(mean=v.mean(), sd=v.std(), min=v.min(), max=v.max(), n=int(m.sum()),
                               area_cm2=self.area_mm2(sr, sc) / 100.0)
        self._stats_key = key
        return self._stats

    def area_mm2(self, sr, sc):
        if self.kind == "ellipse":
            (r0, c0), (r1, c1) = self.pts[:2]
            return math.pi * abs(r1 - r0) * sr / 2 * abs(c1 - c0) * sc / 2
        if self.kind == "rect":
            (r0, c0), (r1, c1) = self.pts[:2]
            return abs(r1 - r0) * sr * abs(c1 - c0) * sc
        ys = np.array([p[0] * sr for p in self.pts])
        xs = np.array([p[1] * sc for p in self.pts])
        return 0.5 * abs(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))

    @staticmethod
    def _len(a, b, sr, sc):
        return math.hypot((a[0] - b[0]) * sr, (a[1] - b[1]) * sc)

    @staticmethod
    def _angle(v1, v2):
        a = math.degrees(math.atan2(v1[0], v1[1]) - math.atan2(v2[0], v2[1]))
        a = abs((a + 180) % 360 - 180)
        return a

    def label(self, plane, sr, sc, unit):
        k, P = self.kind, self.pts
        if k == "length" and len(P) >= 2:
            return [fmt_len(self._len(P[0], P[1], sr, sc))]
        if k == "openpoly" and len(P) >= 2:
            return [fmt_len(sum(self._len(a, b, sr, sc) for a, b in zip(P, P[1:])))]
        if k in ROI_KINDS and len(P) >= 2:
            s = self.roi_stats(plane, sr, sc)
            if not s:
                return []
            out = [f"Area: {fmt_area(s['area_cm2'] * 100)}",
                   f"Mean: {s['mean']:.1f}{unit}  SD: {s['sd']:.1f}",
                   f"Min: {s['min']:.0f}  Max: {s['max']:.0f}"]
            if k == "ellipse":
                (r0, c0), (r1, c1) = P[:2]
                out.insert(1, f"Ø {fmt_dims(abs(c1 - c0) * sc, abs(r1 - r0) * sr)}")
            if k == "rect":
                (r0, c0), (r1, c1) = P[:2]
                out.insert(1, fmt_dims(abs(c1 - c0) * sc, abs(r1 - r0) * sr))
            return out
        if k == "angle" and len(P) >= 3:
            v1 = ((P[0][0] - P[1][0]) * sr, (P[0][1] - P[1][1]) * sc)
            v2 = ((P[2][0] - P[1][0]) * sr, (P[2][1] - P[1][1]) * sc)
            return [f"{self._angle(v1, v2):.1f}°"]
        if k == "cobb" and len(P) >= 4:
            v1 = ((P[1][0] - P[0][0]) * sr, (P[1][1] - P[0][1]) * sc)
            v2 = ((P[3][0] - P[2][0]) * sr, (P[3][1] - P[2][1]) * sc)
            a = self._angle(v1, v2)
            return [f"Cobb: {min(a, 180 - a):.1f}°"]
        if k == "deviation" and len(P) >= 3:
            d, _ = self.deviation_foot(sr, sc)
            return [f"Deviation: {fmt_len(abs(d))}"]
        if k in ("arrow", "text") and self.text:
            return [self.text]
        return []

    def deviation_foot(self, sr, sc):
        (r0, c0), (r1, c1), (r2, c2) = self.pts[:3]
        ax, ay = c0 * sc, r0 * sr
        bx, by = c1 * sc, r1 * sr
        px, py = c2 * sc, r2 * sr
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy or 1e-9
        t = ((px - ax) * dx + (py - ay) * dy) / L2
        fx, fy = ax + t * dx, ay + t * dy
        d = ((px - ax) * dy - (py - ay) * dx) / math.sqrt(L2)
        return d, (fy / sr, fx / sc)

    # ---------------------------------------------------------------- draw
    def paint(self, p: QPainter, to_w, plane, sr, sc, unit, selected=False):
        col = COLOR_SEL if selected else COLOR
        pen = QPen(col, 1.6)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        W = [to_w(*q) for q in self.pts]
        k = self.kind
        if k in ("length", "openpoly", "angle"):
            for a, b in zip(W, W[1:]):
                p.drawLine(a, b)
        elif k == "cobb":
            if len(W) >= 2:
                p.drawLine(W[0], W[1])
            if len(W) >= 4:
                p.drawLine(W[2], W[3])
        elif k == "deviation":
            if len(W) >= 2:
                p.drawLine(W[0], W[1])
            if len(W) >= 3:
                _, foot = self.deviation_foot(sr, sc)
                p.setPen(QPen(col, 1.2, Qt.DashLine))
                p.drawLine(W[2], to_w(*foot))
                p.setPen(pen)
        elif k == "arrow" and len(W) >= 2:
            a, b = W[0], W[1]
            p.drawLine(a, b)
            ang = math.atan2(b.y() - a.y(), b.x() - a.x())
            for s in (0.45, -0.45):
                p.drawLine(b, b - QPointF(math.cos(ang + s) * 12, math.sin(ang + s) * 12))
        elif k in ROI_KINDS and len(W) >= 2:
            if k in ("ellipse", "rect"):
                r = QRectF(W[0], W[1]).normalized()
                p.drawEllipse(r) if k == "ellipse" else p.drawRect(r)
            else:
                p.drawPolygon(QPolygonF(W))
        if k not in ("pencil", "text"):
            p.setBrush(col)
            for q in W:
                p.drawEllipse(q, 2.5, 2.5)
        lines = self.label(plane, sr, sc, unit)
        self._label_box = None
        if lines and W:
            anchor = self.label_anchor(W)
            fm = p.fontMetrics()
            h = fm.height()
            wmax = max(fm.horizontalAdvance(t) for t in lines)
            box = QRectF(anchor.x(), anchor.y() - h * len(lines), wmax + 8, h * len(lines) + 4)
            self._label_box = box
            if k != "text" and math.hypot(*self.label_offset) > 4:
                # moved label: dashed leader line from the box to the nearest point
                near = min(W, key=lambda q: math.hypot(q.x() - box.center().x(), q.y() - box.center().y()))
                edge = QPointF(min(max(near.x(), box.left()), box.right()),
                               min(max(near.y(), box.top()), box.bottom()))
                p.setPen(QPen(col, 1.0, Qt.DashLine))
                p.drawLine(near, edge)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(0, 0, 0, 150))
            p.drawRect(box)
            p.setPen(col)
            for i, t in enumerate(lines):
                p.drawText(QPointF(box.x() + 4, box.y() + fm.ascent() + 2 + i * h), t)

    def label_anchor(self, W):
        ox, oy = self.label_offset
        if self.kind == "text":
            return QPointF(W[0].x() + ox, W[0].y() + oy)
        xs = [q.x() for q in W]
        ys = [q.y() for q in W]
        return QPointF(max(xs) + 12 + ox, min(ys) + 2 + oy)

    def move_label(self, dx, dy):
        self.label_offset = (self.label_offset[0] + dx, self.label_offset[1] + dy)

    # ------------------------------------------------------------- hit test
    def hit_label(self, pos):
        """True when pos (widget coordinates) is on the label box painted last."""
        b = self._label_box
        return b is not None and b.adjusted(-3, -3, 3, 3).contains(pos)

    def hit_handle(self, pos, to_w, tol=7):
        if self.kind == "pencil":
            return None
        for i, q in enumerate(self.pts):
            w = to_w(*q)
            if math.hypot(w.x() - pos.x(), w.y() - pos.y()) <= tol:
                return i
        return None

    def hit_body(self, pos, to_w, tol=6):
        W = [to_w(*q) for q in self.pts]
        if self.kind == "text" and W:
            return math.hypot(W[0].x() - pos.x(), W[0].y() - pos.y()) < 40
        segs = []
        if self.kind in ("ellipse", "rect"):
            r = QRectF(W[0], W[1]).normalized()
            return r.adjusted(-tol, -tol, tol, tol).contains(pos)
        if self.kind in ("polygon", "pencil"):
            segs = list(zip(W, W[1:] + W[:1]))
        elif self.kind == "cobb":
            segs = [(W[0], W[1])] + ([(W[2], W[3])] if len(W) >= 4 else [])
        else:
            segs = list(zip(W, W[1:]))
        return any(_dist_seg(pos, a, b) <= tol for a, b in segs)

    def translate(self, dr, dc):
        self.pts = [(r + dr, c + dc) for r, c in self.pts]


def _dist_seg(p, a, b):
    ax, ay, bx, by = a.x(), a.y(), b.x(), b.y()
    dx, dy = bx - ax, by - ay
    L = dx * dx + dy * dy
    t = 0 if L == 0 else max(0, min(1, ((p.x() - ax) * dx + (p.y() - ay) * dy) / L))
    return math.hypot(p.x() - (ax + t * dx), p.y() - (ay + t * dy))
