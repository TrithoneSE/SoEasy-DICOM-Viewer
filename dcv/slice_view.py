"""2D viewer (axial / coronal / sagittal): free W/L, measurements, rotate / flip, MIP slab, filters, segment
brush, DSA and cine."""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QCursor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QApplication, QHBoxLayout, QInputDialog, QMenu, QSpinBox, QToolButton, QWidget

from . import privacy
from .icons import icon
from .measure import REQUIRED, Measurement

DRAG_TOOLS = ("wl", "zoom", "pan", "browse")


def lps_letter(v):
    """Patient orientation letters for a direction in LPS coordinates (+x = L, +y = P, +z = S)."""
    names = (("R", "L"), ("A", "P"), ("I", "S"))
    out = ""
    for i in np.argsort(-np.abs(v))[:2]:
        if abs(v[i]) > 0.25:
            out += names[i][1 if v[i] > 0 else 0]
    return out


def _ramp(points):
    xs = np.array([p[0] for p in points]) * 255
    cols = np.array([p[1] for p in points], float)
    t = np.arange(256)
    return np.stack([np.interp(t, xs, cols[:, i]) for i in range(3)], axis=1).astype(np.uint8)


COLORMAPS = {
    "Gray": _ramp([(0, (0, 0, 0)), (1, (255, 255, 255))]),
    "Hot": _ramp([(0, (0, 0, 0)), (0.4, (230, 0, 0)), (0.75, (255, 220, 0)), (1, (255, 255, 255))]),
    "Bone": _ramp([(0, (0, 0, 0)), (0.375, (81, 81, 112)), (0.75, (166, 198, 198)), (1, (255, 255, 255))]),
    "Rainbow": _ramp([(0, (0, 0, 128)), (0.25, (0, 128, 255)), (0.5, (0, 255, 0)),
                      (0.75, (255, 255, 0)), (1, (255, 0, 0))]),
    "Perfusion": _ramp([(0, (0, 0, 0)), (0.2, (0, 0, 255)), (0.4, (0, 255, 255)), (0.6, (0, 255, 0)),
                        (0.8, (255, 255, 0)), (1, (255, 0, 0))]),
}
TF_COLORMAP = "3D color curves"


def _blur3(a):
    p = np.pad(a, 1, mode="edge")
    return (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:] + 2 * p[1:-1, :-2] + 4 * p[1:-1, 1:-1] + 2 * p[1:-1, 2:]
            + p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) / 16.0


FILTERS = {
    "none": ("No filter", lambda a: a),
    "smooth": ("Smooth", _blur3),
    "smooth2": ("Smooth (strong)", lambda a: _blur3(_blur3(_blur3(a)))),
    "sharpen": ("Sharpen", lambda a: a + 1.0 * (a - _blur3(a))),
    "sharpen2": ("Sharpen (strong)", lambda a: a + 2.5 * (a - _blur3(a))),
}


class CineBar(QWidget):
    """Cine bar over the bottom of the image while the mouse is on a multi-frame viewport: previous /
    next series, previous / next frame, play / pause and the frame rate."""

    def __init__(self, view):
        super().__init__(view)
        self.view = view
        self.setObjectName("cinebar")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(
            "QWidget#cinebar { background: rgba(20, 20, 24, 210); border: 1px solid #4a4a52; border-radius: 6px; }"
            " QToolButton { border: none; padding: 3px; border-radius: 4px; }"
            " QToolButton:hover { background: rgba(255, 255, 255, 45); }"
            " QSpinBox { background: #1c1c20; color: #e8e8e8; border: 1px solid #4a4a52; border-radius: 3px; }")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 3, 6, 3)
        lay.setSpacing(2)

        def button(name, tip, fn):
            b = QToolButton()
            b.setIcon(icon(name))
            b.setIconSize(QSize(18, 18))
            b.setToolTip(tip)
            b.setFocusPolicy(Qt.NoFocus)
            b.clicked.connect(lambda: (view.activated.emit(view), fn()))
            lay.addWidget(b)
            return b

        button("c_prev_series", "Previous series", lambda: view.seriesStep.emit(view, -1))
        button("c_prev", "Previous frame", lambda: view.set_index(view.index - 1))
        self.play = button("c_play", "Play / pause  [Space]", view.toggle_cine)
        button("c_next", "Next frame", lambda: view.set_index(view.index + 1))
        button("c_next_series", "Next series", lambda: view.seriesStep.emit(view, 1))
        self.fps = QSpinBox()
        self.fps.setRange(1, 60)
        self.fps.setSuffix(" fps")
        self.fps.setFixedWidth(72)
        self.fps.setFocusPolicy(Qt.ClickFocus)
        self.fps.setToolTip("Frame rate (the series' own rate when it was opened)")
        self.fps.valueChanged.connect(view.set_cine_fps)
        lay.addWidget(self.fps)
        self.hide()

    def sync(self):
        self.play.setIcon(icon("c_pause" if self.view.cine_playing else "c_play"))
        self.fps.blockSignals(True)
        self.fps.setValue(int(round(self.view.cine_fps_value())))
        self.fps.blockSignals(False)


class SliceView(QWidget):
    # shared by every viewport (set by the main window)
    tools = {"left": "wl", "right": "zoom", "middle": "pan"}
    wl_sensitivity = 1.0
    brush_radius_mm = 5.0

    windowChanged = Signal(float, float)
    crossChanged = Signal(object, int, int, int)
    indexChanged = Signal(object)
    eyedropper = Signal(object, int, int, int)
    seedPicked = Signal(object, int, int, int)
    requestSegment = Signal(object)
    brushDone = Signal(object)
    annotChanged = Signal(object)
    activated = Signal(object)
    maximizeRequested = Signal(object)
    status = Signal(str)
    cineChanged = Signal(object)         # play / pause of this viewport changed
    seriesStep = Signal(object, int)     # cine bar: previous (-1) / next (+1) series in this viewport

    def __init__(self, orientation="axial", parent=None):
        super().__init__(parent)
        self.orientation = orientation
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(120, 120)
        self.state = None
        self.index = 0
        self.ww, self.wl = 400.0, 40.0
        self.zoom = 1.0
        self.pan = QPointF(0, 0)
        self.rot = 0
        self.flip_h = self.flip_v = False
        self.invert = False
        self.colormap = "Gray"
        self.tf_table = None
        self.slab_mm = 0.0
        self.slab_mode = "mip"
        self.filter = "none"
        self.show_annot = True
        self.show_cross = True
        self.cross = None
        self.fusion = None
        self._img = None
        self._raw = None
        self._hover = None
        self._hover_pos = None
        self._press = None
        self._creating = None
        self._sel = None
        self._handle = None
        self._move = None
        self._label_drag = None          # measurement whose caption is being dragged
        self._wlroi = None
        self._moved = False
        self.dsa = None                  # dsa.DSA while subtraction is on (dynamic XA series, frames = slices)
        self._native_window = None       # W/L before DSA was switched on
        # cine per viewport: own timer and frame rate; None = the series' Frame Time, else 10 fps
        self.cine_playing = False
        self.cine_fps = None
        self._cine = QTimer(self)
        self._cine.timeout.connect(lambda: self.set_index((self.index + 1) % max(1, self.count())))
        self.cine_bar = CineBar(self)

    # ================================================================ data
    @property
    def vol(self):
        return self.state.vol if self.state else None

    def set_state(self, state, keep_window=False):
        self._cine.stop()
        self.cine_playing, self.cine_fps = False, None
        self.state = state
        self.dsa = self._native_window = None
        dynamic = state.vol.frame_time is not None and self.orientation == "axial"
        self.index = 0 if dynamic else self.count() // 2         # a run / loop starts at its first frame
        self.cross = None
        self._creating = self._sel = None
        if not keep_window:
            v = state.vol
            if v.default_ww and v.default_wl is not None:
                self.ww, self.wl = float(v.default_ww), float(v.default_wl)
            else:
                lo, hi = np.percentile(v.frame(self.index) if dynamic else v.data[v.shape[0] // 2], (0.5, 99.5))
                self.ww, self.wl = max(1.0, float(hi - lo)), float(hi + lo) / 2
        self.reset_view()

    def set_orientation(self, o):
        if o != "axial":
            self.set_dsa(False)              # subtraction works on the acquired frames only
        self.orientation = o
        if self.state:
            self.index = self.count() // 2
        self.rot = 0
        self.flip_h = self.flip_v = False
        self.reset_view()

    def count(self):
        if self.vol is None:
            return 0
        z, y, x = self.vol.shape
        return {"axial": z, "coronal": y, "sagittal": x}[self.orientation]

    def axis_spacing(self):
        sz, sy, sx = self.vol.spacing
        return {"axial": sz, "coronal": sy, "sagittal": sx}[self.orientation]

    def _spacing_rc(self):
        sz, sy, sx = self.vol.spacing
        return {"axial": (sy, sx), "coronal": (sz, sx), "sagittal": (sz, sy)}[self.orientation]

    def _plane(self, arr, k):
        if self.orientation == "axial":
            return arr[k]
        if self.orientation == "coronal":
            return arr[::-1, k, :]
        return arr[::-1, :, k]

    def _slab(self, arr, k):
        n = int(round(self.slab_mm / self.axis_spacing()))
        if n <= 1:
            return self._plane(arr, k).astype(np.float32)
        k0, k1 = max(0, k - n // 2), min(self.count(), k - n // 2 + n)
        ax = {"axial": 0, "coronal": 1, "sagittal": 2}[self.orientation]
        sl = [slice(None)] * 3
        sl[ax] = slice(k0, k1)
        block = arr[tuple(sl)]
        red = {"mip": np.max, "minip": np.min, "avg": np.mean}[self.slab_mode](block, axis=ax).astype(np.float32)
        return red if self.orientation == "axial" else red[::-1]

    def raw_plane(self):
        if self.vol.loader is not None and self.orientation == "axial":
            self.vol.frame(self.index)          # compressed run still decoding in the background
        if self._raw is None or self._raw[0] != (self.orientation, self.index, id(self.state)):
            self._raw = ((self.orientation, self.index, id(self.state)), self._plane(self.vol.data, self.index))
        return self._raw[1]

    def image_to_voxel(self, r, c):
        nz = self.vol.shape[0]
        k = self.index
        r, c = int(r), int(c)
        if self.orientation == "axial":
            return k, r, c
        if self.orientation == "coronal":
            return nz - 1 - r, k, c
        return nz - 1 - r, c, k

    def voxel_to_image(self, z, y, x):
        nz = self.vol.shape[0]
        if self.orientation == "axial":
            return y + 0.5, x + 0.5
        if self.orientation == "coronal":
            return nz - 1 - z + 0.5, x + 0.5
        return nz - 1 - z + 0.5, y + 0.5

    def index_of_voxel(self, z, y, x):
        return {"axial": z, "coronal": y, "sagittal": x}[self.orientation]

    # ---------------------------------------------- patient geometry (LPS)
    def plane_axes_lps(self):
        """(right, down, normal) of the displayed raw plane in patient LPS coordinates."""
        D = self.vol.direction
        if self.orientation == "axial":
            return D[:, 0], D[:, 1], D[:, 2]
        if self.orientation == "coronal":
            return D[:, 0], -D[:, 2], D[:, 1]
        return D[:, 1], -D[:, 2], D[:, 0]

    def plane_center_world(self):
        nz, ny, nx = self.vol.shape
        c = {"axial": (self.index, ny / 2, nx / 2), "coronal": (nz / 2, self.index, nx / 2),
             "sagittal": (nz / 2, ny / 2, self.index)}[self.orientation]
        return self.vol.index_to_world(np.array(c, float))

    def world_pos(self, k=None):
        """Slice position (mm) along the plane normal in patient space."""
        return float(np.dot(self.plane_center_world(), self.plane_axes_lps()[2]))

    def set_from_world(self, p):
        """Show the slice that contains (or is nearest to) patient point p."""
        z, y, x = self.vol.world_to_index(np.asarray(p, float))
        self.set_index(int(round({"axial": z, "coronal": y, "sagittal": x}[self.orientation])), emit=False)

    def _plane_voxel_grid(self):
        """(h, w, 3) float (z, y, x) voxel indices of every pixel of the raw plane."""
        h, w = self.raw_plane().shape
        nz = self.vol.shape[0]
        r, c = np.mgrid[0:h, 0:w].astype(np.float32)
        k = np.full_like(r, self.index)
        if self.orientation == "axial":
            return np.stack([k, r, c], -1)
        if self.orientation == "coronal":
            return np.stack([nz - 1 - r, k, c], -1)
        return np.stack([nz - 1 - r, c, k], -1)

    # ------------------------------------------------- display transform
    def _shapes(self):
        h, w = self.raw_plane().shape
        shapes = [(h, w)]
        for _ in range(self.rot):
            h, w = w, h
            shapes.append((h, w))
        return shapes

    def img_to_disp(self, r, c):
        shapes = self._shapes()
        for t in range(self.rot):
            h, w = shapes[t]
            r, c = w - c, r
        H, W = shapes[-1]
        if self.flip_h:
            c = W - c
        if self.flip_v:
            r = H - r
        return r, c

    def disp_to_img(self, r, c):
        shapes = self._shapes()
        H, W = shapes[-1]
        if self.flip_v:
            r = H - r
        if self.flip_h:
            c = W - c
        for t in range(self.rot - 1, -1, -1):
            h, w = shapes[t]
            r, c = c, w - r
        return r, c

    def _disp_spacing(self):
        sr, sc = self._spacing_rc()
        return (sc, sr) if self.rot % 2 else (sr, sc)

    def _target_rect(self):
        H, W = self._shapes()[-1]
        sr, sc = self._disp_spacing()
        pw, ph = W * sc, H * sr
        scale = min(self.width() / pw, self.height() / ph) * self.zoom
        dw, dh = pw * scale, ph * scale
        x0 = (self.width() - dw) / 2 + self.pan.x()
        y0 = (self.height() - dh) / 2 + self.pan.y()
        return QRectF(x0, y0, dw, dh), W, H

    def widget_to_image(self, pos):
        rect, W, H = self._target_rect()
        c = (pos.x() - rect.x()) / rect.width() * W
        r = (pos.y() - rect.y()) / rect.height() * H
        return self.disp_to_img(r, c)

    def image_to_widget(self, r, c):
        r, c = self.img_to_disp(r, c)
        rect, W, H = self._target_rect()
        return QPointF(rect.x() + c / W * rect.width(), rect.y() + r / H * rect.height())

    def _in_image(self, r, c):
        h, w = self.raw_plane().shape
        return 0 <= r < h and 0 <= c < w

    # ============================================================ settings
    def invalidate(self):
        self._img = None
        self.update()

    def set_window(self, ww, wl, emit=False):
        self.ww, self.wl = max(1.0, float(ww)), float(wl)
        self.invalidate()
        if emit:
            self.windowChanged.emit(self.ww, self.wl)

    def set_colormap(self, name):
        self.colormap = name
        self.invalidate()

    def set_tf_table(self, table):
        self.tf_table = table
        if self.colormap == TF_COLORMAP:
            self.invalidate()

    def set_index(self, k, emit=True):
        if self.vol is None:
            return
        k = int(np.clip(k, 0, self.count() - 1))
        if k != self.index:
            self.index = k
            self._creating = None
            self.invalidate()
            if emit:
                self.indexChanged.emit(self)

    def set_cross(self, z, y, x):
        self.cross = (z, y, x)
        self.set_index(self.index_of_voxel(z, y, x), emit=False)
        self.update()

    def reset_view(self):
        self.zoom = 1.0
        self.pan = QPointF(0, 0)
        self.invalidate()

    def rotate(self, cw=True):
        self.rot = (self.rot + (3 if cw else 1)) % 4
        self.invalidate()

    def flip(self, horizontal=True):
        if horizontal:
            self.flip_h = not self.flip_h
        else:
            self.flip_v = not self.flip_v
        self.invalidate()

    # ================================================================ cine
    def is_movie(self):
        """A dynamic run (XA / US loop with a frame time) shown in its own plane."""
        return self.vol is not None and self.vol.frame_time is not None and self.orientation == "axial"

    def cine_fps_value(self):
        if self.cine_fps:
            return self.cine_fps
        return 1000.0 / self.vol.frame_time if self.is_movie() else 10.0

    def set_cine(self, on):
        self.cine_playing = bool(on) and self.vol is not None and self.count() > 1
        if self.cine_playing:
            self._cine.start(int(round(1000.0 / self.cine_fps_value())))
        else:
            self._cine.stop()
        self.cine_bar.sync()
        self.update()
        self.cineChanged.emit(self)

    def toggle_cine(self):
        self.set_cine(not self.cine_playing)

    def set_cine_fps(self, fps):
        self.cine_fps = float(fps)
        if self.cine_playing:
            self._cine.setInterval(int(round(1000.0 / self.cine_fps)))
        self.update()

    def _show_cine_bar(self, on):
        on = on and self.vol is not None and self.count() > 1 and (self.is_movie() or self.cine_playing)
        if on and not self.cine_bar.isVisible():
            self.cine_bar.sync()
            self._place_cine_bar()
            self.cine_bar.show()
        elif not on:
            self.cine_bar.hide()

    def _place_cine_bar(self):
        b = self.cine_bar
        b.adjustSize()
        b.move((self.width() - b.width()) // 2, self.height() - b.height() - 48)

    def resizeEvent(self, e):
        self._place_cine_bar()
        super().resizeEvent(e)

    def enterEvent(self, e):
        self._show_cine_bar(True)
        super().enterEvent(e)

    # ================================================================= DSA
    def dsa_available(self):
        return self.vol is not None and self.orientation == "axial" and self.vol.shape[0] > 1 and self.vol.rgb is None

    def set_dsa(self, on):
        """Subtraction on / off (key /). On: frames minus the mask with the DSA window
        (dsa.suggest_window); off: the window from before. Returns whether DSA is now on."""
        if on and self.dsa is None and self.dsa_available():
            from .dsa import DSA
            self.dsa = DSA(self.vol)
            self._native_window = (self.ww, self.wl)
            self.set_window(*self.dsa.suggest_window(self.ww, self.index), emit=True)
        elif not on and self.dsa is not None:
            self.dsa = None
            if self._native_window:
                self.set_window(*self._native_window, emit=True)
        self.invalidate()
        return self.dsa is not None

    def dsa_mask_current(self):
        if self.set_dsa(True):
            self.dsa.set_mask([self.index])
            self.status.emit(f"DSA mask = image {self.index + 1}")
            self.invalidate()

    def dsa_reset(self):
        if self.dsa is not None:
            self.dsa.reset()
            self.status.emit("DSA mask reset (default mask, no shift)")
            self.invalidate()

    def dsa_move(self, dy, dx):
        if self.set_dsa(True):
            self.dsa.move(dy, dx)
            self.status.emit(self.dsa.label()[1])
            self.invalidate()

    def _dsa_square(self):
        """Half side (image px) of the auto pixel-shift area."""
        h, w = self.vol.shape[1:]
        return max(16, min(h, w) // 10)

    def dsa_auto_at(self, r, c):
        if not self.set_dsa(True):
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.dsa.auto_shift(self.index, (r, c), self._dsa_square())
        finally:
            QApplication.restoreOverrideCursor()
        self.status.emit("Auto pixel-shift: " + self.dsa.label()[1])
        self.invalidate()

    def measurements(self, create=False):
        if self.state is None:
            return []
        key = (self.orientation, self.index)
        if create:
            return self.state.measurements.setdefault(key, [])
        return self.state.measurements.get(key, [])

    def clear_measurements(self, all_slices=False):
        if self.state is None:
            return
        if all_slices:
            self.state.measurements.clear()
        else:
            self.state.measurements.pop((self.orientation, self.index), None)
        self._sel = None
        self.annotChanged.emit(self.state)
        self.update()

    def auto_window(self):
        pl = self.raw_plane()
        lo, hi = np.percentile(pl, (1, 99))
        self.set_window(max(1.0, hi - lo), (hi + lo) / 2, emit=True)

    # =============================================================== render
    def _compose(self):
        if self.vol.loader is not None and self.orientation == "axial":
            self.vol.frame(self.index)
        sl = self.dsa.subtract(self.index) if self.dsa is not None else self._slab(self.vol.data, self.index)
        sl = FILTERS.get(self.filter, FILTERS["none"])[1](sl)
        lo = self.wl - self.ww / 2
        norm = np.clip((sl - lo) / self.ww, 0, 1)
        if self.invert:
            norm = 1 - norm
        if self.vol.rgb is not None:
            rgb = self._plane(self.vol.rgb, self.index).astype(np.float32)
        elif self.colormap == TF_COLORMAP and self.tf_table is not None:
            gray = (norm * 255).astype(np.float32)
            rgb = np.repeat(gray[..., None], 3, axis=2)
            vmin, vmax, tf_rgb, tf_a = self.tf_table[:4]
            src = self._slab(self.state.color_source(), self.index)
            n = len(tf_a)
            idx = np.clip(((src - vmin) / max(1e-6, vmax - vmin) * (n - 1)).astype(np.int32), 0, n - 1)
            a = np.clip(tf_a[idx] * 3.0, 0, 1)
            col = tf_rgb[idx].astype(np.float32)
            region = self.state.bone_region() if len(self.tf_table) == 6 else None
            if region is not None:
                bone = self._plane(region, self.index)
                bone_rgb, bone_a = self.tf_table[4], self.tf_table[5]
                a = np.where(bone, np.clip(bone_a[idx] * 3.0, 0, 1), a)
                col = np.where(bone[..., None], bone_rgb[idx].astype(np.float32), col)
            a = a[..., None]
            rgb = rgb * (1 - a) + col * a
        else:
            lut = COLORMAPS.get(self.colormap, COLORMAPS["Gray"])
            rgb = lut[(norm * 255).astype(np.uint8)].astype(np.float32)
        if self.fusion is not None:
            rgb = self._apply_fusion(rgb)
        for seg in self.state.segments:
            if not seg.visible:
                continue
            m = seg.plane(self.orientation, self.index, rgb.shape[:2])
            if m.any():
                col = np.array(seg.color, np.float32) * 255
                rgb[m] = rgb[m] * 0.5 + col * 0.5
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        if self.rot:
            rgb = np.rot90(rgb, self.rot)
        if self.flip_h:
            rgb = rgb[:, ::-1]
        if self.flip_v:
            rgb = rgb[::-1]
        rgb = np.ascontiguousarray(rgb)
        h, w, _ = rgb.shape
        return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()

    def _apply_fusion(self, rgb):
        """Blend another series (any orientation / resolution) on top, matched in patient coordinates."""
        f = self.fusion
        ov = f["state"].vol
        key = (self.orientation, self.index, id(self.state), id(ov))
        if getattr(self, "_fus_key", None) != key:
            world = self.vol.index_to_world(self._plane_voxel_grid())
            idx = np.rint(ov.world_to_index(world)).astype(np.int64)
            nz, ny, nx = ov.shape
            valid = ((idx[..., 0] >= 0) & (idx[..., 0] < nz) & (idx[..., 1] >= 0) & (idx[..., 1] < ny)
                     & (idx[..., 2] >= 0) & (idx[..., 2] < nx))
            iz, iy, ix = (np.clip(idx[..., i], 0, n - 1) for i, n in enumerate((nz, ny, nx)))
            vals = ov.data[iz, iy, ix].astype(np.float32)
            col = ov.rgb[iz, iy, ix].astype(np.float32) if ov.rgb is not None else None
            self._fus_key, self._fus = key, (vals, col, valid)
        vals, col, valid = self._fus
        if col is not None and f["cmap"] == "Native RGB":
            a = (col.max(axis=-1) > 15) & valid
            color = col
        else:
            lo = f["wl"] - f["ww"] / 2
            norm = np.clip((vals - lo) / max(1.0, f["ww"]), 0, 1)
            color = COLORMAPS.get(f["cmap"], COLORMAPS["Hot"])[(norm * 255).astype(np.uint8)].astype(np.float32)
            a = valid & ((norm > 0) if f["hide_below"] else True)
        a = a.astype(np.float32)[..., None] * f["opacity"]
        return rgb * (1 - a) + color * a

    def set_fusion(self, fusion):
        self.fusion = fusion
        self._fus_key = None
        self.invalidate()

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.black)
        if self.vol is None:
            p.setPen(QColor(110, 110, 110))
            p.drawText(self.rect(), Qt.AlignCenter, "Drag a series here\nor click a series in the list")
            return
        if self._img is None:
            self._img = self._compose()
        rect, W, H = self._target_rect()
        p.setRenderHint(QPainter.SmoothPixmapTransform, self.zoom < 3)
        p.drawImage(rect, self._img)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setFont(QFont("Segoe UI", 9))

        if self.show_cross and self.cross is not None:
            pt = self.image_to_widget(*self.voxel_to_image(*self.cross))
            p.setPen(QPen(QColor(255, 210, 0, 140), 1, Qt.DashLine))
            p.drawLine(QPointF(rect.left(), pt.y()), QPointF(rect.right(), pt.y()))
            p.drawLine(QPointF(pt.x(), rect.top()), QPointF(pt.x(), rect.bottom()))

        sr, sc = self._spacing_rc()
        unit = " HU" if self.vol.modality == "CT" else ""
        plane = self.raw_plane()
        for m in self.measurements():
            m.paint(p, self.image_to_widget, plane, sr, sc, unit, selected=m is self._sel)
        if self._creating is not None:
            self._creating.paint(p, self.image_to_widget, plane, sr, sc, unit, selected=True)
        if self._wlroi:
            a, b = self._wlroi
            p.setPen(QPen(QColor(0, 200, 255), 1, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawRect(QRectF(self.image_to_widget(*a), self.image_to_widget(*b)).normalized())
        if self._uses_brush() and self._hover_pos is not None:
            px_per_mm = rect.width() / (W * self._disp_spacing()[1])
            rad = self.brush_radius_mm * px_per_mm
            p.setPen(QPen(QColor(255, 255, 255, 200), 1))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(self._hover_pos, rad, rad)
        if self.tools.get("left") == "dsa_auto" and self._hover_pos is not None and self.dsa_available():
            r, c = self.widget_to_image(self._hover_pos)       # the area auto pixel-shift will align
            hs = self._dsa_square()
            p.setPen(QPen(QColor(255, 220, 0), 1.5))
            p.setBrush(Qt.NoBrush)
            p.drawRect(QRectF(self.image_to_widget(r - hs, c - hs), self.image_to_widget(r + hs, c + hs)).normalized())

        if self.show_annot:
            self._paint_annotations(p)

    def _paint_annotations(self, p):
        fm = p.fontMetrics()
        m = 6
        p.setPen(QColor(235, 235, 235))
        tl = [privacy.name(self.vol.patient), self.vol.description]
        if self.cine_playing:
            tl.append(f"Cine: {self.cine_fps_value():.0f} fps")
        for i, t in enumerate(tl):
            p.drawText(m, m + fm.ascent() + i * fm.height(), t)
        # a dynamic run (XA / US movie) has no patient orientation: no plane name, edge letters or position
        movie = self.vol.frame_time is not None and self.orientation == "axial"
        n = np.abs(self.plane_axes_lps()[2])
        tr = [] if movie else [("SAGITTAL", "CORONAL", "AXIAL")[int(np.argmax(n))] if n.max() > 0.8 else "OBLIQUE"]
        if self.fusion is not None:
            tr.append(f"+ {self.fusion['state'].vol.description} ({self.fusion['opacity'] * 100:.0f}%)")
        if self.dsa is not None:
            tr.append("DSA (subtraction)")
        elif self.slab_mm > 0:
            tr.append(f"{self.slab_mode.upper()} slab {self.slab_mm:.0f} mm")
        if self.colormap != "Gray":
            tr.append(self.colormap)
        if self.filter != "none":
            tr.append(FILTERS[self.filter][0])
        for i, t in enumerate(tr):
            p.drawText(self.width() - fm.horizontalAdvance(t) - m, m + fm.ascent() + i * fm.height(), t)
        bl = [f"WW: {self.ww:.0f}  WL: {self.wl:.0f}" + ("  (invert)" if self.invert else "")]
        if self.dsa is not None:
            bl += list(self.dsa.label())
        if self._hover:
            bl.append(self._hover)
        for i, t in enumerate(reversed(bl)):
            p.drawText(m, self.height() - m - i * fm.height(), t)
        pos = "" if movie else f"   Pos: {self.world_pos():.1f} mm"
        br = [f"Zoom: {self.zoom * 100:.0f}%{pos}", f"Img: {self.index + 1}/{self.count()}"]
        for i, t in enumerate(reversed(br)):
            p.drawText(self.width() - fm.horizontalAdvance(t) - m, self.height() - m - i * fm.height(), t)
        if movie:
            return
        # label orientasi pasien di tepi
        rgt, dwn, _ = self.plane_axes_lps()
        top, right, bottom, left = lps_letter(-dwn), lps_letter(rgt), lps_letter(dwn), lps_letter(-rgt)
        for _ in range(self.rot):
            top, right, bottom, left = right, bottom, left, top
        if self.flip_h:
            left, right = right, left
        if self.flip_v:
            top, bottom = bottom, top
        p.setPen(QColor(255, 170, 60))
        cx, cy = self.width() / 2, self.height() / 2
        p.drawText(QPointF(cx - 4, m + fm.ascent() + fm.height() * 2), top)
        p.drawText(QPointF(cx - 4, self.height() - m - fm.height() * 2), bottom)
        p.drawText(QPointF(m, cy), left)
        p.drawText(QPointF(self.width() - m - 10, cy), right)

    # ================================================================ mouse
    def _tool_for(self, button):
        name = {Qt.LeftButton: "left", Qt.RightButton: "right", Qt.MiddleButton: "middle"}.get(button)
        return self.tools.get(name) if name else None

    def _uses_brush(self):
        return self.tools.get("left") == "brush"

    def _hit_measurement(self, pos):
        for m in reversed(self.measurements()):
            h = m.hit_handle(pos, self.image_to_widget)
            if h is not None:
                return m, h
        for m in reversed(self.measurements()):
            if m.hit_body(pos, self.image_to_widget):
                return m, None
        return None, None

    def _hit_label(self, pos):
        for m in reversed(self.measurements()):
            if m.hit_label(pos):
                return m
        return None

    def mousePressEvent(self, e):
        self.activated.emit(self)
        if self.vol is None:
            return
        pos = e.position()
        self._press = (pos, e.button())
        self._last = pos
        self._moved = False
        r, c = self.widget_to_image(pos)
        mods = e.modifiers()
        tool = self._tool_for(e.button())

        if e.button() == Qt.LeftButton and mods & Qt.ControlModifier:
            if self._in_image(r, c):
                self.seedPicked.emit(self.state, *self.image_to_voxel(r, c))
            self._press = None
            return
        # a polygon being drawn
        if self._creating is not None:
            if e.button() == Qt.RightButton and REQUIRED[self._creating.kind] is None:
                self._finish_creating()
                return
            if e.button() == Qt.LeftButton:
                self._creating_click(r, c)
                return
        # a measurement's caption (or a text annotation) can be dragged out of the way with any tool
        if e.button() == Qt.LeftButton:
            m = self._hit_label(pos)
            if m is not None:
                self._sel = m
                self._label_drag = m
                self.update()
                return
        if tool is None:
            return
        # edit an existing measurement
        if tool not in ("brush", "eyedropper", "cursor3d", "seed"):
            m, h = self._hit_measurement(pos)
            if m is not None and (h is not None or tool in REQUIRED or tool == "edit"):
                self._sel = m
                if h is not None:
                    self._handle = (m, h)
                else:
                    self._move = (m, (r, c))
                self.update()
                return
        if tool in REQUIRED:
            self._sel = None
            self._start_creating(tool, r, c)
        elif tool == "edit":
            self._sel = None
            self.update()
        elif tool == "cursor3d":
            self._set_cross_at(r, c)
        elif tool == "eyedropper":
            if self._in_image(r, c):
                self.eyedropper.emit(self.state, *self.image_to_voxel(r, c))
        elif tool == "seed":
            if self._in_image(r, c):
                self.seedPicked.emit(self.state, *self.image_to_voxel(r, c))
        elif tool == "brush":
            if self.state.active_segment is None:
                self.requestSegment.emit(self.state)
            self._paint_brush(r, c, erase=bool(mods & Qt.ShiftModifier))
        elif tool == "wlroi":
            self._wlroi = ((r, c), (r, c))
        elif tool == "dsa_auto":
            if self._in_image(r, c):
                self.dsa_auto_at(r, c)
        elif tool == "dsa_shift":
            self.set_dsa(True)

    def mouseMoveEvent(self, e):
        if self.vol is None:
            return
        pos = e.position()
        self._hover_pos = pos
        r, c = self.widget_to_image(pos)
        if self._in_image(r, c):
            z, y, x = self.image_to_voxel(r, c)
            v = self.vol.data[z, y, x]
            unit = " HU" if self.vol.modality == "CT" else ""
            self._hover = f"Value: {v:.0f}{unit}   (x {x}, y {y}, z {z})"
        else:
            self._hover = None

        if self._creating is not None and (self._press is None or self._creating.kind != "pencil"):
            self._creating.pts[-1] = (r, c)
            self._creating._stats_key = None
            self.update()
            if self._press is None:
                return
        if self._press is None:
            on_label = self._creating is None and self._hit_label(pos) is not None
            if on_label:
                self.setCursor(Qt.OpenHandCursor)
            elif getattr(self, "_hand_cursor", False):
                self.unsetCursor()
            self._hand_cursor = on_label
            self.update()
            return
        d = pos - self._last
        self._last = pos
        if abs(d.x()) + abs(d.y()) > 0:
            self._moved = True
        button = self._press[1]
        tool = self._tool_for(button)
        if self._label_drag is not None:
            self._label_drag.move_label(d.x(), d.y())
        elif self._handle:
            m, h = self._handle
            m.pts[h] = (r, c)
            m._stats_key = None
        elif self._move:
            m, (r0, c0) = self._move
            m.translate(r - r0, c - c0)
            m._stats_key = None
            self._move = (m, (r, c))
        elif self._creating is not None and self._creating.kind == "pencil":
            if math.dist(self._creating.pts[-1], (r, c)) > 0.7:
                self._creating.pts.append((r, c))
        elif tool == "wl":
            sens = max(0.5, self.ww / 250.0) * self.wl_sensitivity
            self.set_window(self.ww + d.x() * sens, self.wl + d.y() * sens, emit=True)
        elif tool == "zoom":
            self.zoom = float(np.clip(self.zoom * math.exp(-d.y() / 150.0), 0.05, 60))
            self.invalidate()
        elif tool == "pan":
            self.pan += d
        elif tool == "browse":
            self._browse_acc = getattr(self, "_browse_acc", 0) + d.y()
            step = int(self._browse_acc / 4)
            if step:
                self._browse_acc -= step * 4
                self.set_index(self.index + step)
        elif tool == "cursor3d":
            self._set_cross_at(r, c)
        elif tool == "brush":
            self._paint_brush(r, c, erase=bool(e.modifiers() & Qt.ShiftModifier))
        elif tool == "wlroi" and self._wlroi:
            self._wlroi = (self._wlroi[0], (r, c))
        elif tool == "dsa_shift" and self.dsa is not None:
            r0, c0 = self.widget_to_image(pos - d)          # the mask follows the mouse (zoom / rotation aware)
            self.dsa.move(r - r0, c - c0)
            self.invalidate()
        self.update()

    def mouseReleaseEvent(self, e):
        if self._press is None:
            return
        button = self._press[1]
        tool = self._tool_for(button)
        if self._label_drag is not None or self._handle or self._move:
            self._label_drag = self._handle = self._move = None
            self.annotChanged.emit(self.state)
        elif self._creating is not None:
            k = self._creating.kind
            n = len(self._creating.pts)
            req = REQUIRED[k]
            if k == "pencil":
                self._finish_creating()
            elif req is not None and self._moved:
                if n >= req:
                    self._finish_creating()
                elif n % 2 == 0:  # one line dragged, go on with the next point
                    self._creating.pts.append(self._creating.pts[-1])
        elif tool == "brush":
            self.brushDone.emit(self.state)
        elif tool == "wlroi" and self._wlroi:
            self._apply_wlroi()
        elif button == Qt.RightButton and not self._moved and tool in DRAG_TOOLS:
            self._context_menu(e.globalPosition().toPoint(), e.position())
        self._press = None
        self.update()

    def mouseDoubleClickEvent(self, e):
        if self._creating is not None and REQUIRED[self._creating.kind] is None:
            self._finish_creating()
            return
        if e.button() == Qt.LeftButton and self._tool_for(Qt.LeftButton) in DRAG_TOOLS + ("edit",):
            self.maximizeRequested.emit(self)

    def wheelEvent(self, e):
        if self.vol is None:
            return
        steps = e.angleDelta().y() / 120
        if e.modifiers() & Qt.ControlModifier:
            old = self.widget_to_image(e.position())
            self.zoom = float(np.clip(self.zoom * (1.15 ** steps), 0.05, 60))
            new_w = self.image_to_widget(*old)
            self.pan += e.position() - new_w
            self.invalidate()
        elif e.modifiers() & Qt.ShiftModifier and self._uses_brush():
            SliceView.brush_radius_mm = float(np.clip(self.brush_radius_mm * (1.15 ** steps), 0.5, 100))
            self.status.emit(f"Brush radius: {self.brush_radius_mm:.1f} mm")
            self.update()
        else:
            self.set_index(self.index - int(np.sign(steps)) * max(1, int(abs(steps))))

    def keyPressEvent(self, e):
        if self.vol is None:
            return
        k = e.key()
        arrows = {Qt.Key_Up: (-1, 0), Qt.Key_Down: (1, 0), Qt.Key_Left: (0, -1), Qt.Key_Right: (0, 1)}
        if self.dsa is not None and k in arrows and e.modifiers() & Qt.ShiftModifier:
            step = 0.1 if e.modifiers() & Qt.ControlModifier else 1.0     # Shift 1 px, Ctrl+Shift 0.1 px
            self.dsa_move(arrows[k][0] * step, arrows[k][1] * step)
        elif k in (Qt.Key_Up, Qt.Key_PageUp):
            self.set_index(self.index - (10 if k == Qt.Key_PageUp else 1))
        elif k in (Qt.Key_Down, Qt.Key_PageDown):
            self.set_index(self.index + (10 if k == Qt.Key_PageDown else 1))
        elif k == Qt.Key_Space and self.count() > 1:      # Space starts / stops cine
            self.toggle_cine()
        elif k == Qt.Key_Home:
            self.set_index(0)
        elif k == Qt.Key_End:
            self.set_index(self.count() - 1)
        elif k in (Qt.Key_Return, Qt.Key_Enter) and self._creating is not None:
            self._finish_creating()
        elif k == Qt.Key_Escape:
            if self._creating is not None:
                self._creating = None
            else:
                self._sel = None
            self.update()
        elif k in (Qt.Key_Delete, Qt.Key_Backspace) and self._sel is not None:
            self._delete_measurement(self._sel)
        else:
            super().keyPressEvent(e)

    def leaveEvent(self, _):
        self._hover_pos = None
        # moving onto the cine bar (a child) also leaves this widget: keep the bar while the mouse is over the view
        self._show_cine_bar(self.rect().contains(self.mapFromGlobal(QCursor.pos())))
        self.update()

    # ========================================================= tool actions
    def _set_cross_at(self, r, c):
        if self._in_image(r, c):
            z, y, x = self.image_to_voxel(r, c)
            self.cross = (z, y, x)
            self.crossChanged.emit(self.state, z, y, x)
            self.update()

    def _start_creating(self, kind, r, c):
        if kind == "text":
            txt, ok = QInputDialog.getText(self, "Text annotation", "Text:")
            if ok and txt:
                self.measurements(create=True).append(Measurement("text", [(r, c)], txt))
                self.annotChanged.emit(self.state)
            return
        pts = [(r, c)] if kind == "pencil" else [(r, c), (r, c)]
        self._creating = Measurement(kind, pts)
        self.update()

    def _creating_click(self, r, c):
        m = self._creating
        req = REQUIRED[m.kind]
        m.pts[-1] = (r, c)
        if req is not None and len(m.pts) >= req:
            self._finish_creating()
        else:  # fix this point, add one that follows the mouse
            m.pts.append((r, c))
            self._moved = False
        self.update()

    def _finish_creating(self):
        m = self._creating
        self._creating = None
        if m is None:
            return
        if REQUIRED[m.kind] is None and m.kind != "pencil":
            # drop the moving point / duplicates
            pts = []
            for q in m.pts:
                if not pts or math.dist(pts[-1], q) > 0.5:
                    pts.append(q)
            m.pts = pts
        need = {"polygon": 3, "pencil": 3, "openpoly": 2}.get(m.kind, REQUIRED[m.kind])
        if len(m.pts) >= need:
            if m.kind == "pencil" and len(m.pts) > 400:
                m.pts = m.pts[:: len(m.pts) // 400 + 1]
            self.measurements(create=True).append(m)
            self._sel = m
            self.annotChanged.emit(self.state)
        self.update()

    def _delete_measurement(self, m):
        lst = self.measurements()
        if m in lst:
            lst.remove(m)
        self._sel = None
        self.annotChanged.emit(self.state)
        self.update()

    def _paint_brush(self, r, c, erase=False):
        seg = self.state.active_segment
        if seg is None:
            return
        sr, sc = self._spacing_rc()
        rr, rc = self.brush_radius_mm / sr, self.brush_radius_mm / sc
        h, w = self.raw_plane().shape
        r0, r1 = int(max(0, r - rr)), int(min(h, r + rr + 1))
        c0, c1 = int(max(0, c - rc)), int(min(w, c + rc + 1))
        if r0 >= r1 or c0 >= c1:
            return
        yy, xx = np.mgrid[r0:r1, c0:c1]
        disk = ((yy + 0.5 - r) / rr) ** 2 + ((xx + 0.5 - c) / rc) ** 2 <= 1
        seg.paint(self.orientation, self.index, r0, r1, c0, c1, disk, 0 if erase else 1)
        self.invalidate()

    def _apply_wlroi(self):
        (r0, c0), (r1, c1) = self._wlroi
        self._wlroi = None
        pl = self.raw_plane()
        h, w = pl.shape
        ra, rb = sorted((int(np.clip(r0, 0, h)), int(np.clip(r1, 0, h))))
        ca, cb = sorted((int(np.clip(c0, 0, w)), int(np.clip(c1, 0, w))))
        if rb - ra < 2 or cb - ca < 2:
            return
        blk = pl[ra:rb, ca:cb].astype(np.float32)
        lo, hi = float(blk.min()), float(blk.max())
        self.set_window(max(1.0, hi - lo), (hi + lo) / 2, emit=True)

    # =============================================================== menu
    def _context_menu(self, gpos, pos):
        menu = QMenu(self)
        m, _ = self._hit_measurement(pos)
        if m is not None:
            self._sel = m
            menu.addAction("Delete this measurement", lambda: self._delete_measurement(m))
            if m.kind in ("arrow", "text"):
                menu.addAction("Edit text...", lambda: self._edit_text(m))
            menu.addSeparator()
        menu.addAction("Reset zoom/pan", self.reset_view)
        menu.addAction("Auto window (this slice)", self.auto_window)
        menu.addSeparator()
        menu.addAction("Rotate 90° right", lambda: self.rotate(True))
        menu.addAction("Rotate 90° left", lambda: self.rotate(False))
        menu.addAction("Flip horizontal", lambda: self.flip(True))
        menu.addAction("Flip vertical", lambda: self.flip(False))
        act = menu.addAction("Invert")
        act.setCheckable(True)
        act.setChecked(self.invert)
        act.toggled.connect(lambda v: (setattr(self, "invert", v), self.invalidate()))
        menu.addSeparator()
        menu.addAction("Delete measurements on this image", lambda: self.clear_measurements(False))
        menu.addAction("Delete all measurements of the series", lambda: self.clear_measurements(True))
        menu.addSeparator()
        menu.addAction("Copy image to clipboard", lambda: QApplication.clipboard().setImage(self.grab().toImage()))
        act = menu.addAction("Show crosshair")
        act.setCheckable(True)
        act.setChecked(self.show_cross)
        act.toggled.connect(lambda v: (setattr(self, "show_cross", v), self.update()))
        act = menu.addAction("Show text annotations")
        act.setCheckable(True)
        act.setChecked(self.show_annot)
        act.toggled.connect(lambda v: (setattr(self, "show_annot", v), self.update()))
        menu.exec(gpos)

    def _edit_text(self, m):
        txt, ok = QInputDialog.getText(self, "Text", "Text:", text=m.text)
        if ok:
            m.text = txt
            self.annotChanged.emit(self.state)
            self.update()
