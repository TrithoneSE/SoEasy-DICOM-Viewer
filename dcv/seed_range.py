"""Range of a seed segment as a small curve editor: the histogram around the click with two handles (min / max). Dragging
a handle, or the band between them, re-grows the segment from its own click, so what enters it is tuned by eye."""

from __future__ import annotations

import threading
import time

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QColorDialog, QDialog, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from . import theme
from .segment import ANTI_LEAK_MM, Segment, region_grow

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


class SeedGrower(QObject):
    """Re-grows the active seed segment of one series in a thread, newest request wins, so dragging stays smooth. One per
    series, shared by the 2D panel and the 3D window: whichever moved the range, both follow `grown`."""
    grown = Signal(object, object)      # old segment, new segment
    busy = Signal(bool)                 # a grow has taken long enough to say so (a large region: seconds), or ended

    BUSY_AFTER_S = 0.15

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self.params = lambda: (ANTI_LEAK_MM, None, 0.0)     # anti-leak, crop, smoothing: the 3D window's once open
        self._want = self._job = None
        self._told = False
        self._timer = QTimer(self)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._poll)

    def request(self, lo, hi):
        self._want = (lo, hi)
        if self._job is None:
            self._start()

    def _start(self):
        st = self.state
        seg = st.active_segment
        lo, hi = self._want
        self._want = None
        if seg is None or seg.seed is None:
            return
        leak, bounds, smooth = self.params()
        job = {"seg": seg, "lo": lo, "hi": hi, "new": None, "t0": time.perf_counter()}

        def work():
            mask = region_grow(st.vol.data, st.vol.spacing, seg.seed, lo, hi, leak, bounds, 70.0, smooth)
            if mask is not None:
                job["new"] = Segment(seg.name, mask[0], mask[1], st.vol.shape, seg.color, st.vol.spacing)

        job["thread"] = threading.Thread(target=work, daemon=True)
        self._job = job
        job["thread"].start()
        self._timer.start()

    def _poll(self):
        job = self._job
        if job is not None and job["thread"].is_alive():
            if not self._told and time.perf_counter() - job["t0"] > self.BUSY_AFTER_S:
                self._told = True
                self.busy.emit(True)
            return
        if job is None:
            return
        self._timer.stop()
        self._job = None
        st, old, new = self.state, job["seg"], job["new"]
        if new is not None and old in st.segments:
            new.range = (job["lo"], job["hi"])
            new.seed = old.seed
            new.visible, new.opacity = old.visible, old.opacity
            st.segments[st.segments.index(old)] = new
            st.active_segment = new
            self.grown.emit(old, new)
        if self._want is not None:
            self._start()
        if self._job is None and self._told:
            self._told = False
            self.busy.emit(False)


def grower_of(state):
    """The series' one grower."""
    g = getattr(state, "grower", None)
    if g is None:
        g = state.grower = SeedGrower(state)
    return g


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
        self.growing = False
        self.setToolTip("Drag a handle to change what enters the segment, or the band to move both.\n"
                        "The segment re-grows from its click as you drag.")

    def dragging(self):
        return self._drag is not None

    def set_growing(self, on):
        """A slow re-grow (a large region) is running: the label says so, so a pause is not taken for a hang."""
        self.growing = bool(on)
        self.update()

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
        text = f"{self.lo:.0f} … {self.hi:.0f} HU   (click: {self.value:.0f})"
        if self.growing:
            p.setPen(QColor(255, 200, 80))
            text += "   ·   growing…"
        else:
            p.setPen(QColor(220, 222, 228))
        p.drawText(QRectF(0, r.bottom() + 2, self.width(), 16), Qt.AlignCenter, text)

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


class SeedPanel(QDialog):
    """The range of a seed in 2D: floats over the main and the MPR window, and the segment follows on the slices as the
    handles move. No 3D until *Show in 3D*: that is what makes a seed instant on a large CT."""
    show3d = Signal(object)         # state
    changed = Signal(object)        # state: colour, visibility or the list of segments changed

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Tool)
        self.setWindowTitle("Segment range")
        self.setStyleSheet(theme.main_stylesheet())
        self.state = self.grower = None
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 10, 10, 10)
        self.title = QLabel()
        v.addWidget(self.title)
        self.editor = SeedRangeEditor()
        self.editor.setMinimumHeight(110)
        self.editor.rangeChanged.connect(lambda lo, hi: self.grower and self.grower.request(lo, hi))
        v.addWidget(self.editor)
        row = QHBoxLayout()
        self.color_btn = QPushButton("Color")
        self.color_btn.clicked.connect(self._pick_color)
        self.hide_btn = QPushButton("Hide")
        self.hide_btn.clicked.connect(self._toggle)
        self.del_btn = QPushButton("Delete")
        self.del_btn.clicked.connect(self._delete)
        self.d3_btn = QPushButton("Show in 3D")
        self.d3_btn.setDefault(True)
        self.d3_btn.clicked.connect(lambda: self.state and self.show3d.emit(self.state))
        for b in (self.color_btn, self.hide_btn, self.del_btn, self.d3_btn):
            row.addWidget(b)
        v.addLayout(row)
        self.resize(420, 190)

    def follow(self, state, grower):
        """Show the active segment of this series."""
        if self.grower is not grower:
            if self.grower is not None:
                self.grower.grown.disconnect(self._on_grown)
                self.grower.busy.disconnect(self._on_busy)
            grower.grown.connect(self._on_grown)
            grower.busy.connect(self._on_busy)
        self.state, self.grower = state, grower
        self.sync()

    def _on_grown(self, *_):
        self.sync()
        self.changed.emit(self.state)

    def _on_busy(self, on):
        self.editor.set_growing(on)
        self.setCursor(Qt.BusyCursor) if on else self.unsetCursor()

    def sync(self):
        st = self.state
        seg = st.active_segment if st is not None else None
        if seg is None or seg.seed is None or seg.range is None:
            self.hide()
            return
        self.editor.set_segment(st.vol.data, st.vol.spacing, seg)
        self.title.setText(f"<b>{seg.name}</b> — {seg.volume_ml:.2f} ml")
        self.hide_btn.setText("Hide" if seg.visible else "Show")

    def _pick_color(self):
        seg = self.state.active_segment if self.state else None
        if seg is None:
            return
        c = QColorDialog.getColor(QColor.fromRgbF(*seg.color), self, "Segment color")
        if c.isValid():
            seg.color = (c.redF(), c.greenF(), c.blueF())
            self.sync()
            self.changed.emit(self.state)

    def _toggle(self):
        seg = self.state.active_segment if self.state else None
        if seg is not None:
            seg.visible = not seg.visible
            self.sync()
            self.changed.emit(self.state)

    def _delete(self):
        st = self.state
        seg = st.active_segment if st else None
        if seg is not None:
            st.segments.remove(seg)
            st.active_segment = st.segments[-1] if st.segments else None
            self.sync()
            self.changed.emit(st)
