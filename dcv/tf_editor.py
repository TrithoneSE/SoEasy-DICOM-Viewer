"""Interactive color / opacity curve editor for 3D volume rendering.

Mouse:
  - drag a point            : move it (HU left / right, opacity up / down)
  - drag a curve line       : move the whole curve (Shift = change its height / opacity)
  - double-click empty space: new curve
  - double-click a line     : new point on that curve
  - double-click a point    : change its color
  - Ctrl + click            : add a point to the selected curve
  - Alt + drag              : move ALL curves together (W/L of the 3D colors)
  - right-click             : menu (color, delete, duplicate, rename)
  - wheel                   : zoom the HU axis; middle-drag: pan
  - Delete                  : delete the selected point
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QColorDialog, QInputDialog, QMenu, QWidget

from .tf_model import TFCurve, TFPoint, TransferFunction

NEW_CURVE_COLORS = [(40, 220, 60), (255, 220, 0), (60, 140, 255), (200, 80, 255),
                    (0, 220, 220), (255, 130, 0), (255, 80, 160)]


def _qc(rgb, a=255):
    return QColor(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255), a)


class TFEditor(QWidget):
    changed = Signal()
    boneLockMoved = Signal(float)   # bone-lock threshold dragged in the editor (emitted on release)

    ML, MR, MT, MB = 44, 12, 10, 40

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 220)      # must fit the 3D panel at its narrowest (with a vertical scrollbar)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.tf = TransferFunction()
        self.dmin, self.dmax = -1024.0, 3071.0
        self.vx0, self.vx1 = self.dmin, self.dmax
        self.hist = None
        self.sel_curve = None
        self.sel_point = None
        self._drag = None
        self._last = None
        self._hover_text = ""
        self._color_idx = 0
        self.gain = 1.0
        self.bone_lock_hu = None        # None = no bone-lock marker; otherwise a draggable threshold line

    # ------------------------------------------------------------ public
    def set_data_range(self, dmin, dmax, hist_data=None):
        self.dmin, self.dmax = float(dmin), float(dmax)
        self.vx0, self.vx1 = self.dmin, self.dmax
        if hist_data is not None:
            sample = hist_data.ravel()
            if sample.size > 4_000_000:
                sample = sample[:: sample.size // 4_000_000 + 1]
            counts, edges = np.histogram(sample, bins=512, range=(self.dmin, self.dmax))
            self.hist = (np.log1p(counts.astype(float)), edges)
        self.update()

    def set_tf(self, tf: TransferFunction):
        self.tf = tf
        self.sel_curve = self.sel_point = None
        self.update()
        self.changed.emit()

    def fit_view_to_curves(self):
        xs = [p.x for c in self.tf.curves for p in c.points]
        if xs:
            lo, hi = min(xs), max(xs)
            pad = max(50.0, (hi - lo) * 0.15)
            self.vx0 = max(self.dmin, lo - pad)
            self.vx1 = min(self.dmax, hi + pad)
            if self.vx1 - self.vx0 < 20:
                self.vx0, self.vx1 = self.dmin, self.dmax
        self.update()

    def reset_zoom(self):
        self.vx0, self.vx1 = self.dmin, self.dmax
        self.update()

    # -------------------------------------------------------- koordinat
    def plot_rect(self):
        return QRectF(self.ML, self.MT, self.width() - self.ML - self.MR, self.height() - self.MT - self.MB)

    def x_to_px(self, x):
        r = self.plot_rect()
        return r.left() + (x - self.vx0) / (self.vx1 - self.vx0) * r.width()

    def px_to_x(self, px):
        r = self.plot_rect()
        return self.vx0 + (px - r.left()) / r.width() * (self.vx1 - self.vx0)

    def y_to_py(self, y):
        r = self.plot_rect()
        return r.bottom() - math.sqrt(max(0.0, y)) * r.height()   # square-root scale: low opacities are easy to set

    def py_to_y(self, py):
        r = self.plot_rect()
        t = (r.bottom() - py) / r.height()
        return float(np.clip(t, 0, 1)) ** 2

    def pt(self, p: TFPoint):
        return QPointF(self.x_to_px(p.x), self.y_to_py(p.y))

    # ---------------------------------------------------------- hit test
    def hit_point(self, pos, radius=7):
        best = None
        for ci, c in enumerate(self.tf.curves):
            for pi, p in enumerate(c.points):
                d = math.hypot(self.pt(p).x() - pos.x(), self.pt(p).y() - pos.y())
                if d <= radius and (best is None or d < best[0]):
                    best = (d, ci, pi)
        return (best[1], best[2]) if best else None

    def set_bone_lock(self, hu):
        self.bone_lock_hu = None if hu is None else float(hu)
        self.update()

    def hit_bone_marker(self, pos, tol=6):
        if self.bone_lock_hu is None:
            return False
        r = self.plot_rect()
        return abs(self.x_to_px(self.bone_lock_hu) - pos.x()) <= tol and r.top() - 4 <= pos.y() <= r.bottom()

    def hit_segment(self, pos, tol=5):
        best = None
        for ci, c in enumerate(self.tf.curves):
            for pi in range(len(c.points) - 1):
                a, b = self.pt(c.points[pi]), self.pt(c.points[pi + 1])
                d = _dist_seg(pos, a, b)
                if d <= tol and (best is None or d < best[0]):
                    best = (d, ci, pi)
        return (best[1], best[2]) if best else None

    # -------------------------------------------------------------- draw
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(28, 28, 32))
        r = self.plot_rect()
        p.fillRect(r, QColor(12, 12, 14))

        # histogram
        if self.hist is not None:
            counts, edges = self.hist
            mx = counts.max() or 1
            path = QPainterPath()
            path.moveTo(self.x_to_px(edges[0]), r.bottom())
            for i, cnt in enumerate(counts):
                x = self.x_to_px((edges[i] + edges[i + 1]) / 2)
                path.lineTo(x, r.bottom() - cnt / mx * r.height() * 0.9)
            path.lineTo(self.x_to_px(edges[-1]), r.bottom())
            path.closeSubpath()
            p.save()
            p.setClipRect(r)
            p.fillPath(path, QColor(90, 90, 100, 120))
            p.restore()

        # grid
        p.setFont(QFont("Segoe UI", 8))
        step = _nice_step((self.vx1 - self.vx0) / 8)
        x = math.ceil(self.vx0 / step) * step
        while x <= self.vx1:
            px = self.x_to_px(x)
            p.setPen(QPen(QColor(55, 55, 62), 1))
            p.drawLine(QPointF(px, r.top()), QPointF(px, r.bottom()))
            p.setPen(QColor(170, 170, 180))
            p.drawText(QRectF(px - 30, r.bottom() + 2, 60, 14), Qt.AlignCenter, f"{x:g}")
            x += step
        for yv in (0.05, 0.25, 0.5, 1.0):
            py = self.y_to_py(yv)
            p.setPen(QPen(QColor(55, 55, 62), 1, Qt.DotLine))
            p.drawLine(QPointF(r.left(), py), QPointF(r.right(), py))
            p.setPen(QColor(170, 170, 180))
            p.drawText(QRectF(0, py - 7, self.ML - 4, 14), Qt.AlignRight | Qt.AlignVCenter, f"{yv:g}")

        # bone-lock threshold: everything right of the line (plus its filled cavities / edge) is the bone area
        if self.bone_lock_hu is not None and self.vx0 < self.bone_lock_hu < self.vx1:
            bx = self.x_to_px(self.bone_lock_hu)
            p.fillRect(QRectF(bx, r.top(), r.right() - bx, r.height()), QColor(230, 220, 190, 18))
            p.setPen(QPen(QColor(240, 225, 170), 1.5, Qt.DashDotLine))
            p.drawLine(QPointF(bx, r.top()), QPointF(bx, r.bottom()))
            p.setBrush(QColor(240, 225, 170))
            p.drawPolygon(QPolygonF([QPointF(bx - 6, r.top()), QPointF(bx + 6, r.top()), QPointF(bx, r.top() + 8)]))
            p.setPen(QColor(240, 225, 170))
            p.drawText(QPointF(bx + 5, r.top() + 20), f"🦴 bone area ≥ {self.bone_lock_hu:.0f}")

        p.save()
        p.setClipRect(r.adjusted(-8, -8, 8, 8))
        # curves
        for ci, c in enumerate(self.tf.curves):
            if len(c.points) < 2:
                continue
            pts = [self.pt(q) for q in c.points]
            poly = QPolygonF([QPointF(pts[0].x(), r.bottom())] + pts + [QPointF(pts[-1].x(), r.bottom())])
            x0, x1 = pts[0].x(), pts[-1].x()
            grad = QLinearGradient(x0, 0, x1 if x1 > x0 else x0 + 1, 0)
            for q, qp in zip(c.points, pts):
                t = 0 if x1 == x0 else (qp.x() - x0) / (x1 - x0)
                grad.setColorAt(float(np.clip(t, 0, 1)), _qc(q.color, 110))
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(grad))
            p.drawPolygon(poly)
            selected = ci == self.sel_curve
            bone = c.layer == "bone"
            p.setPen(QPen(QColor(255, 220, 0) if selected else QColor(235, 235, 235), 2.2 if selected else 1.3,
                          Qt.DashLine if bone else Qt.SolidLine))
            p.setBrush(Qt.NoBrush)
            for a, b in zip(pts, pts[1:]):
                p.drawLine(a, b)
            # curve name label
            top = min(pts, key=lambda q: q.y())
            p.setPen(QColor(255, 220, 0) if selected else QColor(210, 210, 210))
            p.drawText(top + QPointF(6, -6), ("🦴 " if bone else "") + c.name + ("  · rims hidden" if c.hide_edges else ""))
            for pi, (q, qp) in enumerate(zip(c.points, pts)):
                is_sel = selected and pi == self.sel_point
                p.setPen(QPen(QColor(255, 220, 0) if is_sel else QColor(255, 255, 255), 2 if is_sel else 1))
                p.setBrush(_qc(q.color))
                p.drawEllipse(qp, 6 if is_sel else 4.5, 6 if is_sel else 4.5)
        p.restore()

        # strip hasil gabungan
        xs, a, rgb = self.tf.sample(self.vx0, self.vx1, max(16, int(r.width())), gain=self.gain)
        bar = QRectF(r.left(), r.bottom() + 17, r.width(), 10)
        for i in range(len(xs)):
            col = rgb[i] * min(1.0, math.sqrt(a[i]) * 1.5)
            p.fillRect(QRectF(bar.left() + i * bar.width() / len(xs), bar.top(), bar.width() / len(xs) + 1,
                              bar.height()), _qc(col))
        p.setPen(QColor(80, 80, 90))
        p.setBrush(Qt.NoBrush)
        p.drawRect(bar)
        p.drawRect(r)

        if self._hover_text:
            p.setPen(QColor(255, 255, 255))
            p.drawText(QRectF(r.left() + 6, r.top() + 4, r.width() - 12, 16), Qt.AlignRight, self._hover_text)

    # ------------------------------------------------------------- mouse
    def _emit(self):
        self.update()
        self.changed.emit()

    def mousePressEvent(self, e):
        pos = e.position()
        self._last = pos
        if e.button() == Qt.MiddleButton:
            self._drag = "pan"
            return
        if e.button() == Qt.RightButton:
            self._context_menu(pos, e.globalPosition().toPoint())
            return
        if e.button() != Qt.LeftButton:
            return
        mods = e.modifiers()
        if mods & Qt.AltModifier:
            self._drag = "all"
            return
        if mods & Qt.ControlModifier and self.sel_curve is not None:
            self._add_point(self.sel_curve, self.px_to_x(pos.x()), self.py_to_y(pos.y()))
            return
        hp = self.hit_point(pos)
        if hp:
            self.sel_curve, self.sel_point = hp
            self._drag = "point"
            self.update()
            return
        if self.hit_bone_marker(pos):
            self._drag = "bone"
            return
        hs = self.hit_segment(pos)
        if hs:
            self.sel_curve, self.sel_point = hs[0], None
            self._drag = "curve"
            self.update()
            return
        self.sel_curve = self.sel_point = None
        self._drag = None
        self.update()

    def mouseMoveEvent(self, e):
        pos = e.position()
        hp = self.hit_point(pos)
        if self._drag is None:
            if hp:
                q = self.tf.curves[hp[0]].points[hp[1]]
                self._hover_text = f"{self.tf.curves[hp[0]].name}: HU {q.x:.0f}, opacity {q.y:.3f}"
                self.setCursor(Qt.SizeAllCursor)
            elif self.hit_bone_marker(pos):
                self.setCursor(Qt.SizeHorCursor)
                self._hover_text = (f"🦴 Bone lock starts at {self.bone_lock_hu:.0f} — drag left/right "
                                    "(left = more bone, fewer gaps)")
            elif self.hit_segment(pos):
                self.setCursor(Qt.OpenHandCursor)
                self._hover_text = f"HU {self.px_to_x(pos.x()):.0f}"
            else:
                self.unsetCursor()
                self._hover_text = f"HU {self.px_to_x(pos.x()):.0f}"
            self.update()
            return
        d = pos - self._last
        self._last = pos
        dx = d.x() / self.plot_rect().width() * (self.vx1 - self.vx0)
        if self._drag == "pan":
            self.vx0 -= dx
            self.vx1 -= dx
            self.update()
        elif self._drag == "point":
            c = self.tf.curves[self.sel_curve]
            q = c.points[self.sel_point]
            lo = c.points[self.sel_point - 1].x + 0.5 if self.sel_point > 0 else -1e9
            hi = c.points[self.sel_point + 1].x - 0.5 if self.sel_point < len(c.points) - 1 else 1e9
            q.x = float(np.clip(self.px_to_x(pos.x()), lo, hi))
            q.y = self.py_to_y(pos.y())
            self._hover_text = f"{c.name}: HU {q.x:.0f}, opacity {q.y:.3f}"
            self._emit()
        elif self._drag == "curve":
            c = self.tf.curves[self.sel_curve]
            if e.modifiers() & Qt.ShiftModifier:
                f = math.exp(-d.y() / 80.0)
                for q in c.points:
                    q.y = float(np.clip(q.y * f, 0, 1))
            else:
                for q in c.points:
                    q.x += dx
            self._emit()
        elif self._drag == "all":
            for c in self.tf.curves:
                for q in c.points:
                    q.x += dx
            self._emit()
        elif self._drag == "bone":
            self.bone_lock_hu = float(np.clip(round(self.px_to_x(pos.x()) / 10) * 10, 100, 3000))
            self._hover_text = f"🦴 Bone lock starts at {self.bone_lock_hu:.0f} (applied when you release)"
            self.update()

    def mouseReleaseEvent(self, _):
        if self._drag == "bone":
            self.boneLockMoved.emit(self.bone_lock_hu)   # recomputing the bone area takes ~1 s: only on release
        self._drag = None

    def mouseDoubleClickEvent(self, e):
        if e.button() != Qt.LeftButton:
            return
        pos = e.position()
        hp = self.hit_point(pos)
        if hp:
            self._pick_point_color(*hp)
            return
        hs = self.hit_segment(pos, tol=7)
        if hs:
            self._add_point(hs[0], self.px_to_x(pos.x()), None)
            return
        if self.plot_rect().contains(pos):
            self._add_curve(self.px_to_x(pos.x()), max(0.15, self.py_to_y(pos.y())))

    def wheelEvent(self, e):
        f = 0.85 ** (e.angleDelta().y() / 120)
        cx = self.px_to_x(e.position().x())
        w = float(np.clip((self.vx1 - self.vx0) * f, 10, (self.dmax - self.dmin) * 3))
        t = (cx - self.vx0) / (self.vx1 - self.vx0)
        self.vx0 = cx - t * w
        self.vx1 = self.vx0 + w
        self.update()

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            if self.sel_curve is not None and self.sel_point is not None:
                self._delete_point(self.sel_curve, self.sel_point)
            elif self.sel_curve is not None:
                self._delete_curve(self.sel_curve)
        else:
            super().keyPressEvent(e)

    # ---------------------------------------------------------- aksi
    def _add_curve(self, x, height):
        w = (self.vx1 - self.vx0) * 0.08
        rgb = NEW_CURVE_COLORS[self._color_idx % len(NEW_CURVE_COLORS)]
        self._color_idx += 1
        col = tuple(v / 255 for v in rgb)
        c = TFCurve(f"Curve {len(self.tf.curves) + 1}", [
            TFPoint(x - w, 0.0, col), TFPoint(x - w / 3, height, col),
            TFPoint(x + w / 3, height, col), TFPoint(x + w, 0.0, col)])
        self.tf.curves.append(c)
        self.sel_curve, self.sel_point = len(self.tf.curves) - 1, None
        self._emit()

    def add_curve_range(self, lo, hi, color, name, height=0.3):
        """Add a trapezoid curve over [lo, hi] (used by the color picker)."""
        ramp = max(2.0, (hi - lo) * 0.2)
        c = TFCurve(name, [TFPoint(lo - ramp, 0.0, color), TFPoint(lo, height, color),
                           TFPoint(hi, height, color), TFPoint(hi + ramp, 0.0, color)])
        self.tf.curves.append(c)
        self.sel_curve, self.sel_point = len(self.tf.curves) - 1, None
        if lo - ramp < self.vx0 or hi + ramp > self.vx1:
            self.fit_view_to_curves()
        self._emit()

    def _add_point(self, ci, x, y):
        c = self.tf.curves[ci]
        a, rgb = c.sample(np.array([x]))
        if y is None:
            y = float(a[0])
        c.points.append(TFPoint(float(x), float(y), tuple(float(v) for v in rgb[0])))
        c.sort()
        self.sel_curve = ci
        self.sel_point = next(i for i, q in enumerate(c.points) if q.x == float(x))
        self._emit()

    def _delete_point(self, ci, pi):
        c = self.tf.curves[ci]
        if len(c.points) <= 2:
            self._delete_curve(ci)
            return
        del c.points[pi]
        self.sel_point = None
        self._emit()

    def _delete_curve(self, ci):
        del self.tf.curves[ci]
        self.sel_curve = self.sel_point = None
        self._emit()

    def _pick_point_color(self, ci, pi):
        q = self.tf.curves[ci].points[pi]
        col = QColorDialog.getColor(_qc(q.color), self, "Point color")
        if col.isValid():
            q.color = (col.redF(), col.greenF(), col.blueF())
            self._emit()

    def _pick_curve_color(self, ci):
        c = self.tf.curves[ci]
        col = QColorDialog.getColor(_qc(c.points[len(c.points) // 2].color), self, f"Curve color {c.name}")
        if col.isValid():
            for q in c.points:
                q.color = (col.redF(), col.greenF(), col.blueF())
            self._emit()

    def _rename_curve(self, ci):
        c = self.tf.curves[ci]
        name, ok = QInputDialog.getText(self, "Rename curve", "Name:", text=c.name)
        if ok and name:
            c.name = name
            self._emit()

    def _duplicate_curve(self, ci):
        import copy
        c = copy.deepcopy(self.tf.curves[ci])
        c.name += " (copy)"
        shift = (self.vx1 - self.vx0) * 0.05
        for q in c.points:
            q.x += shift
        self.tf.curves.append(c)
        self.sel_curve = len(self.tf.curves) - 1
        self._emit()

    def _context_menu(self, pos, gpos):
        menu = QMenu(self)
        hp = self.hit_point(pos)
        hs = self.hit_segment(pos, tol=7)
        if hp:
            ci, pi = hp
            menu.addAction("Change point color...", lambda: self._pick_point_color(ci, pi))
            menu.addAction("Delete point", lambda: self._delete_point(ci, pi))
            menu.addSeparator()
        if hp or hs:
            ci = (hp or hs)[0]
            menu.addAction("Change color of the whole curve...", lambda: self._pick_curve_color(ci))
            act = menu.addAction("🦴 Bone curve (color locked to the bone area)")
            act.setCheckable(True)
            act.setChecked(self.tf.curves[ci].layer == "bone")
            act.toggled.connect(lambda on: (setattr(self.tf.curves[ci], "layer", "bone" if on else "normal"),
                                            self._emit()))
            act = menu.addAction("Hide partial-volume rims (tumor / soft tissue: not on vessel or bone walls)")
            act.setCheckable(True)
            act.setChecked(self.tf.curves[ci].hide_edges)
            act.toggled.connect(lambda on: (setattr(self.tf.curves[ci], "hide_edges", on), self._emit()))
            menu.addAction("Rename curve...", lambda: self._rename_curve(ci))
            menu.addAction("Duplicate curve", lambda: self._duplicate_curve(ci))
            menu.addAction("Delete curve", lambda: self._delete_curve(ci))
            menu.addSeparator()
        x = self.px_to_x(pos.x())
        y = max(0.15, self.py_to_y(pos.y()))
        menu.addAction("Add a new curve here", lambda: self._add_curve(x, y))
        menu.addAction("Zoom to curves", self.fit_view_to_curves)
        menu.addAction("Reset zoom (full range)", self.reset_zoom)
        menu.exec(gpos)


def _dist_seg(p, a, b):
    ax, ay, bx, by = a.x(), a.y(), b.x(), b.y()
    dx, dy = bx - ax, by - ay
    L = dx * dx + dy * dy
    t = 0 if L == 0 else max(0, min(1, ((p.x() - ax) * dx + (p.y() - ay) * dy) / L))
    return math.hypot(p.x() - (ax + t * dx), p.y() - (ay + t * dy))


def _nice_step(raw):
    if raw <= 0:
        return 1
    e = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 5, 10):
        if raw <= m * e:
            return m * e
    return 10 * e
