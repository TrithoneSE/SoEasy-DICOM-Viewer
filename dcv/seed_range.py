"""Range of a seed segment as a small curve editor: the histogram around the click with two handles (min / max). Dragging
a handle, or the band between them, re-grows the segment from its own click, so what enters it is tuned by eye."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from . import theme

HANDLE_PX = 7
RADIUS_MM = 40.0
SPAN_HU = 400.0


def seed_histogram(data, spacing, seed, radius_mm=RADIUS_MM, span=SPAN_HU):
    """(bin edges, log counts) of the voxels within radius_mm of the seed, ±span HU around its value."""
    r = [max(1, int(radius_mm / s)) for s in spacing]
    blk = data[tuple(slice(max(0, c - k), c + k + 1) for c, k in zip(seed, r))]
    val = float(data[tuple(seed)])
    lo, hi = max(float(blk.min()), val - span), min(float(blk.max()), val + span)
    if hi <= lo:
        lo, hi = val - 1.0, val + 1.0
    counts, edges = np.histogram(blk, bins=min(200, max(8, int(hi - lo) // 2 + 1)), range=(lo, hi))
    return edges, np.log1p(counts.astype(float))


class SeedRangeEditor(QWidget):
    rangeChanged = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(84)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMouseTracking(True)
        self.edges = self.counts = None
        self.lo = self.hi = self.value = 0.0
        self.color = QColor(40, 220, 60)
        self._key = None
        self._drag = None
        self._grab = 0.0
        self.setToolTip("Drag a handle to change what enters the segment, or the band to move both.\n"
                        "The segment re-grows from its click as you drag.")

    def dragging(self):
        return self._drag is not None

    def set_segment(self, data, spacing, seg):
        """Show a seed segment's range; None clears it."""
        if seg is None or seg.seed is None or seg.range is None:
            self._key = None
            self.setVisible(False)
            return
        key = (id(data), tuple(seg.seed))
        if key != self._key:
            self.edges, self.counts = seed_histogram(data, spacing, seg.seed)
            self.value = float(data[tuple(seg.seed)])
            self._key = key
        if not self.dragging():
            self.lo, self.hi = float(seg.range[0]), float(seg.range[1])
        self.color = QColor.fromRgbF(*seg.color)
        self.setVisible(True)
        self.update()

    # ------------------------------------------------------------ geometry
    def _plot(self):
        return QRectF(HANDLE_PX, 4, max(10, self.width() - 2 * HANDLE_PX), max(10, self.height() - 22))

    def _x(self, hu):
        r = self._plot()
        a, b = float(self.edges[0]), float(self.edges[-1])
        return r.left() + (min(max(hu, a), b) - a) / max(b - a, 1e-6) * r.width()

    def _hu(self, x):
        r = self._plot()
        a, b = float(self.edges[0]), float(self.edges[-1])
        return a + (x - r.left()) / r.width() * (b - a)

    # --------------------------------------------------------------- paint
    def paintEvent(self, _):
        if self.edges is None:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self._plot()
        p.fillRect(r, QColor(theme.BG))
        peak = max(float(self.counts.max()), 1e-6)
        path = QPainterPath(QPointF(r.left(), r.bottom()))
        for k, c in enumerate(self.counts):
            path.lineTo(self._x((self.edges[k] + self.edges[k + 1]) / 2), r.bottom() - c / peak * r.height())
        path.lineTo(r.right(), r.bottom())
        path.closeSubpath()
        p.fillPath(path, QColor(120, 125, 135, 150))
        x0, x1 = self._x(self.lo), self._x(self.hi)
        band = QColor(self.color)
        band.setAlpha(90)
        p.fillRect(QRectF(x0, r.top(), max(1.0, x1 - x0), r.height()), band)
        p.setPen(QPen(QColor(255, 255, 255, 170), 1, Qt.DashLine))
        xv = self._x(self.value)
        p.drawLine(QPointF(xv, r.top()), QPointF(xv, r.bottom()))
        p.setPen(QPen(self.color, 2))
        p.setBrush(QColor(255, 255, 255))
        for x in (x0, x1):
            p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
            p.drawEllipse(QPointF(x, r.center().y()), 5, 5)
        p.setPen(QColor(220, 222, 228))
        p.drawText(QRectF(0, r.bottom() + 2, self.width(), 16), Qt.AlignCenter,
                   f"{self.lo:.0f} … {self.hi:.0f} HU   (click: {self.value:.0f})")

    # --------------------------------------------------------------- mouse
    def _hit(self, x):
        x0, x1 = self._x(self.lo), self._x(self.hi)
        if abs(x - x0) <= HANDLE_PX and abs(x - x0) <= abs(x - x1):
            return "lo"
        if abs(x - x1) <= HANDLE_PX:
            return "hi"
        return "band" if x0 < x < x1 else None

    def mouseMoveEvent(self, e):
        x = e.position().x()
        if self._drag is None:
            hit = self._hit(x) if self.edges is not None else None
            self.setCursor({"lo": Qt.SizeHorCursor, "hi": Qt.SizeHorCursor, "band": Qt.OpenHandCursor}.get(hit, Qt.ArrowCursor))
            return
        hu = self._hu(x)
        step = float(self.edges[1] - self.edges[0])
        if self._drag == "lo":                  # the clicked voxel always stays inside the range
            self.lo = min(hu, self.value, self.hi - step)
        elif self._drag == "hi":
            self.hi = max(hu, self.value, self.lo + step)
        else:
            w = self.hi - self.lo
            lo = min(hu - self._grab, self.value)
            lo = max(lo, self.value - w)
            self.lo, self.hi = lo, lo + w
        self.update()
        self.rangeChanged.emit(self.lo, self.hi)

    def mousePressEvent(self, e):
        if self.edges is not None and e.button() == Qt.LeftButton:
            self._drag = self._hit(e.position().x())
            self._grab = self._hu(e.position().x()) - self.lo

    def mouseReleaseEvent(self, _):
        self._drag = None
