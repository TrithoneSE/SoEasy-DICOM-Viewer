"""MPR window: sagittal / coronal / axial with draggable and rotatable axes.

Each MPR view is a SliceView, so every 2D tool works here too (W/L, pan, zoom, all measurements, W/L ROI...).
On top of that, in every view:
  - drag the line intersection      : move the centre
  - drag a line                     : shift the plane that line represents
  - drag a small circle on a line   : rotate the axes (oblique MPR)
  - wheel / arrow keys              : move this plane along its normal
"""

from __future__ import annotations

import math

import numpy as np
from . import vtk_lite as vtk
from vtkmodules.util import numpy_support
from PySide6.QtCore import QObject, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QAction, QActionGroup, QColor, QImage, QKeySequence, QPen
from PySide6.QtWidgets import (
    QDoubleSpinBox, QGridLayout, QLabel, QMainWindow, QMenu, QMessageBox, QSizePolicy, QToolBar, QToolButton,
    QWidget, QWidgetAction,
)

from . import privacy, theme
from .icons import icon
from .measure import REQUIRED
from .segment import np_to_vtk_image
from .slice_view import COLORMAPS, FILTERS, TF_COLORMAP, SliceView
from .tools import ToolBank
from .ui import cmap_icon

SLABS = [("Off", (0, "max")), ("MIP 5 mm", (5, "max")), ("MIP 10 mm", (10, "max")), ("MIP 20 mm", (20, "max")),
         ("MIP 40 mm", (40, "max")), ("MinIP 10 mm", (10, "min")), ("Average 5 mm", (5, "mean")),
         ("Average 10 mm", (10, "mean"))]
HINT = ("Drag a line = shift plane • drag a circle = tilt (oblique) • drag the centre = move • Ctrl+click = "
        "segmentation seed • double-click = maximise • F1 = guide")
PLANE_COLORS = {"sagittal": QColor(40, 170, 255), "coronal": QColor(255, 60, 150), "axial": QColor(255, 215, 0)}
TITLES = {"sagittal": "Sagittal", "coronal": "Coronal", "axial": "Axial"}


def _rot(axis, theta):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * K @ K


def _letter(v):
    """Patient orientation letters for direction v (VTK coords: +x = L, +y = P, +z = S)."""
    names = (("R", "L"), ("A", "P"), ("I", "S"))
    out = ""
    for i in np.argsort(-np.abs(v))[:2]:
        if abs(v[i]) > 0.25:
            out += names[i][1 if v[i] > 0 else 0]
    return out


class MPRModel(QObject):
    changed = Signal()

    # reference "up" & "right" for each view (VTK coords: +x = L, +y = P, +z = S)
    _REF = {"sagittal": ((0, 0, 1), (0, 1, 0)), "coronal": ((0, 0, 1), (1, 0, 0)), "axial": ((0, -1, 0), (1, 0, 0))}

    def __init__(self, state):
        super().__init__()
        self.state = state
        vol = state.vol
        self.D = np.asarray(vol.direction, float)        # image axes -> patient LPS
        self.image = np_to_vtk_image(vol.data, vol.spacing)
        sz, sy, sx = vol.spacing
        nz, ny, nx = vol.shape
        self.extent_mm = np.array([nx * sx, ny * sy, nz * sz])
        self.vcenter = (self.extent_mm - np.array([sx, sy, sz])) / 2
        self.step = min(vol.spacing)
        self.lo = float(vol.value_range[0])
        self.slab_mm = 0.0
        self.slab_mode = "max"
        self.show_lines = True
        self.tf_table = None
        self.measurements = {}
        self.roll = {}                  # per view: in-plane rotation (radians) from the Roll tool
        self.reset()

    def reset(self):
        self.center = self.vcenter.copy()
        self.R = np.eye(3)
        self.roll = {k: 0.0 for k in ("sagittal", "coronal", "axial")}
        self.changed.emit()

    def axes(self, kind):
        """(right, up, normal). The 'up' direction follows the patient reference, so rotating the
        axes in one view does not rotate that view's own image; it only makes the other views oblique."""
        R = self.R
        n = {"sagittal": R[:, 0], "coronal": R[:, 1], "axial": R[:, 2]}[kind]
        # reference directions are in patient space; convert them to the image (voxel) frame
        up_ref, right_ref = (self.D.T @ np.array(v, float) for v in self._REF[kind])
        b = up_ref - np.dot(up_ref, n) * n
        if np.linalg.norm(b) < 1e-3:
            a = right_ref - np.dot(right_ref, n) * n
            a /= np.linalg.norm(a)
            b = np.cross(n, a)
        else:
            b /= np.linalg.norm(b)
            a = np.cross(b, n)
        if np.dot(a, right_ref) < 0:
            a = -a
        th = self.roll.get(kind, 0.0)
        if th:                          # Roll tool: turn this view's own image in its plane
            c, s = math.cos(th), math.sin(th)
            a, b = c * a + s * b, -s * a + c * b
        return a, b, n

    def normal(self, kind):
        return self.axes(kind)[2]

    def clamp_center(self):
        self.center = np.clip(self.center, 0, self.extent_mm)

    def world_to_voxel(self, p):
        sz, sy, sx = self.state.vol.spacing
        nz, ny, nx = self.state.vol.shape
        i, j, k = int(round(p[0] / sx)), int(round(p[1] / sy)), int(round(p[2] / sz))
        inside = 0 <= i < nx and 0 <= j < ny and 0 <= k < nz
        return (int(np.clip(k, 0, nz - 1)), int(np.clip(j, 0, ny - 1)), int(np.clip(i, 0, nx - 1))), inside


class ObliqueView(SliceView):
    is_active = False
    tools = {"left": "wl", "right": "zoom", "middle": "pan"}   # separate from the main window

    def __init__(self, model: MPRModel, kind, parent=None):
        self._ready = False
        super().__init__("axial", parent)
        self.m = model
        self.kind = kind
        self.state = model.state
        self.show_cross = False
        v = model.state.vol
        if v.default_ww:
            self.ww, self.wl = float(v.default_ww), float(v.default_wl)
        self.reslice = vtk.vtkImageReslice()
        self.reslice.SetInputData(model.image)
        self.reslice.SetInterpolationModeToLinear()
        self.reslice.SetOutputDimensionality(2)
        self.reslice.SetBackgroundLevel(model.lo)
        self.res = model.step
        self.N = int(math.ceil(float(np.linalg.norm(model.extent_mm)) / 2 / self.res))
        self.reslice.SetOutputSpacing(self.res, self.res, self.res)
        self.reslice.SetOutputOrigin(0, 0, 0)
        self.reslice.SetOutputExtent(-self.N, self.N, -self.N, self.N, 0, 0)
        self._plane_cache = None
        self._mdrag = None
        self._mlast = None
        self._ready = True
        model.changed.connect(self._on_model_changed)

    # ----------------------------------------------------- plane & index
    def _origin(self):
        a, b, n = self.m.axes(self.kind)
        return self.m.vcenter + np.dot(self.m.center - self.m.vcenter, n) * n

    def _on_model_changed(self):
        self._plane_cache = None
        self._creating = None
        self.invalidate()

    def plane_key(self):
        a, b, n = self.m.axes(self.kind)
        return (self.kind, tuple(np.round(n, 3)), tuple(np.round(a, 3)), round(float(np.dot(self._origin(), n)), 1))

    @property
    def index(self):
        if not getattr(self, "_ready", False):
            return 0
        n = self.m.normal(self.kind)
        return int(round(float(np.dot(self.m.center - self.m.vcenter, n)) / self.m.step)) + self._half_count()

    @index.setter
    def index(self, k):
        if not getattr(self, "_ready", False):
            return
        n = self.m.normal(self.kind)
        self.m.center = self.m.center + n * (k - self.index) * self.m.step
        self.m.clamp_center()
        self.m.changed.emit()

    def _half_count(self):
        return int(math.ceil(float(np.linalg.norm(self.m.extent_mm)) / 2 / self.m.step))

    def count(self):
        return 2 * self._half_count() + 1 if getattr(self, "_ready", False) else 0

    def axis_spacing(self):
        return self.m.step

    def _spacing_rc(self):
        return self.res, self.res

    def raw_plane(self):
        if self._plane_cache is None:
            a, b, n = self.m.axes(self.kind)
            self.reslice.SetResliceAxesDirectionCosines(*a, *b, *n)
            self.reslice.SetResliceAxesOrigin(*self._origin())
            k = int(round(self.m.slab_mm / self.res))
            self.reslice.SetSlabNumberOfSlices(max(1, k))
            {"max": self.reslice.SetSlabModeToMax, "min": self.reslice.SetSlabModeToMin,
             "mean": self.reslice.SetSlabModeToMean}[self.m.slab_mode]()
            self.reslice.Update()
            arr = numpy_support.vtk_to_numpy(self.reslice.GetOutput().GetPointData().GetScalars())
            n2 = 2 * self.N + 1
            self._plane_cache = np.ascontiguousarray(arr.reshape(n2, n2)[::-1]).astype(np.float32)
        return self._plane_cache

    # ------------------------------------------- image <-> world coordinates
    def image_to_uv(self, r, c):
        return (c - 0.5 - self.N) * self.res, (self.N - (r - 0.5)) * self.res

    def uv_to_image(self, u, v):
        return self.N - v / self.res + 0.5, u / self.res + self.N + 0.5

    def image_to_world(self, r, c):
        a, b, _ = self.m.axes(self.kind)
        u, v = self.image_to_uv(r, c)
        return self._origin() + u * a + v * b

    def world_to_image(self, w):
        a, b, _ = self.m.axes(self.kind)
        d = np.asarray(w, float) - self._origin()
        return self.uv_to_image(float(np.dot(d, a)), float(np.dot(d, b)))

    def image_to_voxel(self, r, c):
        return self.m.world_to_voxel(self.image_to_world(r, c))[0]

    def voxel_to_image(self, z, y, x):
        sz, sy, sx = self.vol.spacing
        return self.world_to_image((x * sx, y * sy, z * sz))

    def _in_image(self, r, c):
        h, w = self.raw_plane().shape
        return 0 <= r < h and 0 <= c < w and self.m.world_to_voxel(self.image_to_world(r, c))[1]

    def world_pos(self, k=None):
        return float(np.dot(self.m.center, self.m.normal(self.kind)))

    def _target_rect(self):
        n2 = 2 * self.N + 1
        scale = min(self.width(), self.height()) / (float(self.m.extent_mm.max()) * 1.05) * self.zoom
        size = n2 * self.res * scale
        x0 = (self.width() - size) / 2 + self.pan.x()
        y0 = (self.height() - size) / 2 + self.pan.y()
        return QRectF(x0, y0, size, size), n2, n2

    # --------------------------------------------------------- measurements
    def measurements(self, create=False):
        key = self.plane_key()
        if create:
            return self.m.measurements.setdefault(key, [])
        return self.m.measurements.get(key, [])

    def clear_measurements(self, all_slices=False):
        if all_slices:
            for k in [k for k in self.m.measurements if k[0] == self.kind]:
                del self.m.measurements[k]
        else:
            self.m.measurements.pop(self.plane_key(), None)
        self._sel = None
        self.update()

    def _paint_brush(self, r, c, erase=False):
        self.status.emit("The segment brush works in the main window's 2D views.")

    def _set_cross_at(self, r, c):
        if self._in_image(r, c):
            self.m.center = self.image_to_world(r, c)
            self.m.changed.emit()

    # --------------------------------------------------------------- render
    def _compose(self):
        sl = FILTERS.get(self.filter, FILTERS["none"])[1](self.raw_plane())
        lo = self.wl - self.ww / 2
        norm = np.clip((sl - lo) / self.ww, 0, 1)
        if self.invert:
            norm = 1 - norm
        if self.colormap == TF_COLORMAP and self.m.tf_table is not None:
            gray = (norm * 255).astype(np.float32)
            rgb = np.repeat(gray[..., None], 3, axis=2)
            vmin, vmax, tf_rgb, tf_a = self.m.tf_table
            nn = len(tf_a)
            idx = np.clip(((self.raw_plane() - vmin) / max(1e-6, vmax - vmin) * (nn - 1)).astype(np.int32), 0, nn - 1)
            a = np.clip(tf_a[idx] * 3.0, 0, 1)[..., None]
            rgb = rgb * (1 - a) + tf_rgb[idx].astype(np.float32) * a
        else:
            rgb = COLORMAPS.get(self.colormap, COLORMAPS["Gray"])[(norm * 255).astype(np.uint8)].astype(np.float32)
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

    def _line_points(self):
        """[(other view, centre widget point, unit direction in widget space)]"""
        a, b, n = self.m.axes(self.kind)
        c_img = self.world_to_image(self.m.center)
        c = self.image_to_widget(*c_img)
        out = []
        for other in ("sagittal", "coronal", "axial"):
            if other == self.kind:
                continue
            d = np.cross(n, self.m.normal(other))
            d /= max(1e-9, np.linalg.norm(d))
            p2 = self.image_to_widget(*self.world_to_image(self.m.center + d * 20))
            dx, dy = p2.x() - c.x(), p2.y() - c.y()
            L = math.hypot(dx, dy) or 1
            out.append((other, c, (dx / L, dy / L)))
        return out

    def _handle_dist(self):
        return min(self.width(), self.height()) * 0.33

    def paintEvent(self, e):
        super().paintEvent(e)
        from PySide6.QtGui import QPainter
        if self.is_active:      # yellow frame of the active view, as in the main window
            p = QPainter(self)
            p.setPen(QPen(QColor(theme.ACTIVE_VIEWPORT), 2))
            p.drawRect(1, 1, self.width() - 2, self.height() - 2)
            p.end()
        if not self.m.show_lines:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        L = max(self.width(), self.height()) * 2
        hd = self._handle_dist()
        c = None
        for other, c, (dx, dy) in self._line_points():
            p.setPen(QPen(PLANE_COLORS[other], 1.4))
            p.drawLine(QPointF(c.x() - dx * L, c.y() - dy * L), QPointF(c.x() + dx * L, c.y() + dy * L))
            p.setBrush(Qt.NoBrush)
            for sgn in (1, -1):
                p.drawEllipse(QPointF(c.x() + sgn * dx * hd, c.y() + sgn * dy * hd), 5, 5)
        if c is not None:
            p.setBrush(PLANE_COLORS[self.kind])
            p.setPen(Qt.NoPen)
            p.drawEllipse(c, 4, 4)

    def _paint_annotations(self, p):
        fm = p.fontMetrics()
        mg = 6
        p.fillRect(self.width() - 14, 4, 10, 10, PLANE_COLORS[self.kind])
        p.setPen(QColor(235, 235, 235))
        title = "MPR " + TITLES[self.kind]
        if self.m.slab_mm > 0:
            title += f"   {self.m.slab_mode.upper()} {self.m.slab_mm:.0f} mm"
        if self.colormap != "Gray":
            title += f"   {self.colormap}"
        p.drawText(mg, mg + fm.ascent(), title)
        bl = [f"WL: {self.wl:.0f} WW: {self.ww:.0f}" + ("  (invert)" if self.invert else "")]
        if self._hover:
            bl.append(self._hover)
        for i, t in enumerate(reversed(bl)):
            p.drawText(mg, self.height() - mg - i * fm.height(), t)
        t = f"Zoom: {self.zoom * 100:.0f}%"
        p.drawText(self.width() - fm.horizontalAdvance(t) - mg, self.height() - mg, t)
        a, b, _ = self.m.axes(self.kind)
        p.setPen(QColor(255, 170, 60))
        cx, cy = self.width() / 2, self.height() / 2
        D = self.m.D
        p.drawText(QPointF(cx - 6, mg + fm.ascent() + fm.height()), _letter(D @ b))
        p.drawText(QPointF(cx - 6, self.height() - mg - fm.height() * 2), _letter(D @ -b))
        p.drawText(QPointF(mg, cy), _letter(D @ -a))
        p.drawText(QPointF(self.width() - mg - 18, cy), _letter(D @ a))

    # ---------------------------------------------------- axis interaction
    def _hit_axes(self, pos):
        if not self.m.show_lines:
            return None
        lines = self._line_points()
        if not lines:
            return None
        c = lines[0][1]
        if math.hypot(pos.x() - c.x(), pos.y() - c.y()) < 10:
            return ("center", None)
        hd = self._handle_dist()
        for other, c, (dx, dy) in lines:
            for sgn in (1, -1):
                if math.hypot(pos.x() - (c.x() + sgn * dx * hd), pos.y() - (c.y() + sgn * dy * hd)) < 9:
                    return ("rotate", other)
        for other, c, (dx, dy) in lines:
            vx, vy = pos.x() - c.x(), pos.y() - c.y()
            if abs(vx * dy - vy * dx) < 5:
                return ("move", other)
        return None

    def _roll_center(self):
        return self.image_to_widget(*self.world_to_image(self.m.center))

    def mousePressEvent(self, e):
        mods = e.modifiers()
        if self._tool_for(e.button()) == "roll" and self._creating is None and self._hit_label(e.position()) is None:
            self.activated.emit(self)
            self._rolling = e.position()
            return
        if e.button() == Qt.LeftButton and not (mods & (Qt.ControlModifier | Qt.ShiftModifier)) \
                and self._creating is None and self._hit_label(e.position()) is None:
            h = self._hit_axes(e.position())
            # with a measurement tool, a plain line hit starts a measurement instead of shifting the plane
            # (the centre point and the rotate circles still work)
            if h is not None and h[0] == "move" and self.tools.get("left") in REQUIRED:
                h = None
            if h is not None:
                self.activated.emit(self)
                self._mdrag, self._mlast = h, e.position()
                return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if getattr(self, "_rolling", None) is not None:
            c, prev, pos = self._roll_center(), self._rolling, e.position()
            self._rolling = pos
            a0 = math.atan2(prev.y() - c.y(), prev.x() - c.x())
            a1 = math.atan2(pos.y() - c.y(), pos.x() - c.x())
            flip = -1 if (self.flip_h != self.flip_v) else 1
            self.m.roll[self.kind] = (self.m.roll.get(self.kind, 0.0) + (a1 - a0) * flip) % (2 * math.pi)
            self.m.changed.emit()
            return
        if self._mdrag is None:
            if self._press is None and self._creating is None:
                h = self._hit_axes(e.position())
                self.setCursor({"center": Qt.SizeAllCursor, "rotate": Qt.CrossCursor,
                                "move": Qt.SizeAllCursor}.get(h[0]) if h else Qt.ArrowCursor)
            super().mouseMoveEvent(e)
            return
        pos, prev = e.position(), self._mlast
        self._mlast = pos
        kind, other = self._mdrag
        w0 = self.image_to_world(*self.widget_to_image(prev))
        w1 = self.image_to_world(*self.widget_to_image(pos))
        delta = w1 - w0
        if kind == "center":
            self.m.center = self.m.center + delta
        elif kind == "move":
            no = self.m.normal(other)
            self.m.center = self.m.center + np.dot(delta, no) * no
        elif kind == "rotate":
            c = self.image_to_widget(*self.world_to_image(self.m.center))
            a0 = math.atan2(-(prev.y() - c.y()), prev.x() - c.x())
            a1 = math.atan2(-(pos.y() - c.y()), pos.x() - c.x())
            a, b, _ = self.m.axes(self.kind)
            flip = -1 if (self.flip_h != self.flip_v) else 1
            self.m.R = _rot(np.cross(a, b), (a1 - a0) * flip) @ self.m.R
        self.m.clamp_center()
        self.m.changed.emit()

    def mouseReleaseEvent(self, e):
        if getattr(self, "_rolling", None) is not None:
            self._rolling = None
            return
        if self._mdrag is not None:
            self._mdrag = None
            return
        super().mouseReleaseEvent(e)


class MPRWindow(QMainWindow):
    seedPicked = Signal(object, int, int, int)
    eyedropper = Signal(object, int, int, int)

    def update_title(self):
        v = self.model.state.vol
        self.setWindowTitle(f"3D MPR - {privacy.name(v.patient)} - {v.description}")

    def __init__(self, state, wl_presets, parent=None):
        super().__init__(parent)
        self.resize(1600, 850)
        self.model = MPRModel(state)
        self.update_title()
        self.views = {}
        self._max = None
        central = QWidget()
        central.setStyleSheet("background:black")
        self.grid = QGridLayout(central)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(2)
        for i, kind in enumerate(("sagittal", "coronal", "axial")):
            view = ObliqueView(self.model, kind)
            view.maximizeRequested.connect(self._toggle_max)
            view.seedPicked.connect(self.seedPicked)
            view.eyedropper.connect(self.eyedropper)
            view.windowChanged.connect(lambda ww, wl, src=view: self._sync_window(src, ww, wl))
            view.status.connect(lambda m: self.statusBar().showMessage(m, 5000))
            view.activated.connect(self._set_active)
            self.views[kind] = view
            self.grid.addWidget(view, 0, i)
        self.setCentralWidget(central)
        self.active = self.views["axial"]

        self.setStyleSheet(theme.main_stylesheet())
        self.wl_presets = wl_presets
        self._build_toolbar()
        self.mouse_lbl = QLabel()
        self.statusBar().addPermanentWidget(self.mouse_lbl)
        self.bank.assign("left", "wl")
        self.update_tf_table()
        self._sync_spins(self.active.ww, self.active.wl)
        self._set_active(self.active)
        self.statusBar().showMessage(HINT, 15000)

    # ================================================================ toolbar
    def _act(self, text, fn, shortcut=None, tip=None, icon_name=None, label=None, checkable=False):
        """Action whose shortcut works anywhere in the window (also when it only sits in a ▾ menu)."""
        a = QAction(text, self)
        a.setCheckable(checkable)
        (a.toggled if checkable else a.triggered).connect(fn)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        a.setToolTip((tip or text) + (f"  [{shortcut}]" if shortcut else ""))
        if icon_name:
            a.setIcon(icon(icon_name))
        if label:
            a.setIconText(label)
        self.addAction(a)
        return a

    def _menu_button(self, icon_name, label, tip, menu):
        b = QToolButton()
        b.setIcon(icon(icon_name))
        b.setText(label)
        b.setToolTip(tip)
        b.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        b.setIconSize(QSize(22, 22))
        b.setAutoRaise(True)
        b.setMenu(menu)
        b.setPopupMode(QToolButton.InstantPopup)
        return b

    def _build_toolbar(self):
        """One row like the main window: mouse tools, then layout / axes / image / rotate / save in ▾ menus."""
        tb = QToolBar("Tools")
        tb.setObjectName("mpr_tools")
        tb.setMovable(False)
        tb.setIconSize(QSize(22, 22))
        tb.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.addToolBar(tb)
        self.bank = ToolBank(self, ObliqueView.tools, extra=("roll", "cursor3d", "eyedropper", "seed", "wlroi"))
        self.bank.changed.connect(lambda _: (self.mouse_lbl.setText(self.bank.mouse_html())
                                             if hasattr(self, "mouse_lbl") else None, self._update_all()))
        self.bank.populate(tb)
        menu = self.bank.measure_menu
        menu.addAction("Delete measurements on this plane", lambda: self.active.clear_measurements(False))
        menu.addAction("Delete all measurements in this view", lambda: self.active.clear_measurements(True))

        # W/L ▾: exact values + presets (keys 1–9 like the main window) + auto
        w = QWidget()
        g = QGridLayout(w)
        g.setContentsMargins(12, 8, 12, 6)
        self.ww_spin = QDoubleSpinBox()
        self.ww_spin.setRange(1, 100000)
        self.wl_spin = QDoubleSpinBox()
        self.wl_spin.setRange(-100000, 100000)
        for r, (text, sp) in enumerate((("Width (WW)", self.ww_spin), ("Level (WL)", self.wl_spin))):
            sp.setDecimals(0)
            sp.setKeyboardTracking(False)
            sp.valueChanged.connect(lambda _: self.set_window(self.ww_spin.value(), self.wl_spin.value()))
            g.addWidget(QLabel(text), r, 0)
            g.addWidget(sp, r, 1)
        editor = QWidgetAction(self)
        editor.setDefaultWidget(w)
        wm = QMenu(self)
        wm.addAction(editor)
        wm.addSeparator()
        for i, (name, ww, wl) in enumerate(self.wl_presets):
            wm.addAction(self._act(f"{name}  ({ww:g} / {wl:g})", lambda _=False, a=ww, b=wl: self.set_window(a, b),
                                   str(i + 1) if i < 9 else None))
        wm.addSeparator()
        wm.addAction(self._act("Auto window (active view)", lambda: self.active.auto_window(), "0"))
        wlb = self.bank.buttons["wl"]
        wlb.setMenu(wm)
        wlb.setPopupMode(QToolButton.MenuButtonPopup)
        wlb.setToolTip("Window / Level  [W]   ▾ presets (1–9), exact WW / WL, auto (0)\n"
                       "Left-click = assign to LEFT mouse button, right-click = assign to RIGHT mouse button")
        tb.addSeparator()

        lm = QMenu(self)
        for key, label in (("1x3", "Side by side (1 × 3)"), ("3x1", "Stacked (3 × 1)"),
                           ("big_axial", "Big axial + 2 small"), ("big_sagittal", "Big sagittal + 2 small"),
                           ("big_coronal", "Big coronal + 2 small")):
            lm.addAction(label, lambda k=key: self.set_layout(k))
        tb.addWidget(self._menu_button("layout", "Layout", "Arrangement of the three views", lm))
        self.lines_act = self._act("Show axes", lambda on: (setattr(self.model, "show_lines", on), self._update_all()),
                                   None, "Show / hide the plane lines and their rotation handles", "axes", "Axes", True)
        self.lines_act.setChecked(True)
        tb.addAction(self.lines_act)
        tb.addAction(self._act("Text overlay", self._toggle_annot, "Ctrl+Shift+A", "Show / hide the text overlay",
                               "annot", "Overlay"))
        tb.addSeparator()

        tb.addWidget(self._menu_button("image", "Image", "Color map, slab (MIP / MinIP / average), filter, invert",
                                       self._build_image_menu()))
        rm = QMenu(self)
        for text, fn, sc, ic in (("Rotate 90° right", lambda: self.active.rotate(True), "Ctrl+Right", "rotate_r"),
                                 ("Rotate 90° left", lambda: self.active.rotate(False), "Ctrl+Left", "rotate_l"),
                                 ("Flip horizontal", lambda: self.active.flip(True), "Ctrl+H", "flip_h"),
                                 ("Flip vertical", lambda: self.active.flip(False), "Ctrl+J", "flip_v")):
            rm.addAction(self._act(text, fn, sc, f"{text} (active view)", ic))
        rm.addSeparator()
        rm.addAction(self._act("Reset zoom / pan of the active view", lambda: self.active.reset_view()))
        tb.addWidget(self._menu_button("transform", "Rotate", "Rotate / flip the active view", rm))
        tb.addAction(self._act("Reset axes", self._reset, "Ctrl+R", "Reset axes, position and zoom of all views",
                               "reset", "Reset"))
        tb.addSeparator()
        sm = QMenu(self)
        sm.addAction(self._act("Save active view...", self._save_image, "Ctrl+S"))
        sm.addAction(self._act("Copy active view", self._copy_image, "Ctrl+Shift+C"))
        sm.addAction(self._act("Save all three views...", self._save_all))
        tb.addWidget(self._menu_button("camera", "Save", "Save or copy the active view, or all three views", sm))
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tb.addWidget(spacer)
        tb.addAction(self._act("Quick guide", self._help, "F1", "How to move and tilt the planes", "help", "Help"))

    def _build_image_menu(self):
        m = QMenu(self)

        def group(title, items, on):
            sub = m.addMenu(title)
            grp = QActionGroup(self)
            out = []
            for label, data in items:
                a = sub.addAction(label)
                a.setCheckable(True)
                grp.addAction(a)
                a.triggered.connect(lambda _=False, d=data: on(d))
                out.append((a, data))
            out[0][0].setChecked(True)
            return out

        acts = group("Color map", [(n, n) for n in list(COLORMAPS) + [TF_COLORMAP]], self._set_cmap)
        self.cmap_actions = {d: a for a, d in acts}
        for name, a in self.cmap_actions.items():
            if name in COLORMAPS:
                a.setIcon(cmap_icon(COLORMAPS[name]))
        self.slab_actions = group("Slab (MIP / MinIP / average)", SLABS + [("Custom...", "custom")], self._set_slab)
        acts = group("Filter", [(label, k) for k, (label, _) in FILTERS.items()], self._set_filter)
        self.filter_actions = {d: a for a, d in acts}
        m.addSeparator()
        m.addAction(self._act("Invert", lambda: self._all(lambda vw: (setattr(vw, "invert", not vw.invert),
                                                                       vw.invalidate())), "I", None, "invert"))
        return m

    def _set_cmap(self, name):
        self._all(lambda vw: vw.set_colormap(name))

    def _set_filter(self, k):
        self._all(lambda vw: (setattr(vw, "filter", k), vw.invalidate()))

    def _help(self):
        QMessageBox.information(self, "MPR — quick guide", (
            "PLANES\n"
            "  Drag a colored line = shift that plane\n"
            "  Drag a small circle on a line = tilt the plane (oblique MPR)\n"
            "  Drag the centre = move all planes • Ctrl+R = reset axes\n"
            "  Double-click a view = maximise it • Axes button = hide the lines\n\n"
            "MOUSE TOOLS (same as the main window)\n"
            "  Left-click a tool = left mouse button, right-click = right mouse button\n"
            "  Wheel = slice • Middle = pan • Ctrl+click = segmentation seed\n"
            "  W/L ▾ = presets (1–9), exact WW / WL, auto (0)\n\n"
            "Image ▾ = color map, slab (MIP / MinIP / average, also custom thickness), filter, invert (I)\n"
            "Save ▾ = save / copy the active view (Ctrl+S / Ctrl+Shift+C) or all three views"))

    def _all(self, fn):
        for vw in self.views.values():
            fn(vw)

    def _update_all(self):
        self._all(lambda vw: vw.update())

    def _set_active(self, view):
        self.active = view
        for vw in self.views.values():
            vw.is_active = vw is view
            vw.update()

    def _sync_window(self, src, ww, wl):
        self._all(lambda vw: vw.set_window(ww, wl) if vw is not src else None)
        self._sync_spins(ww, wl)

    def set_window(self, ww, wl):
        self._all(lambda vw: vw.set_window(ww, wl))
        self._sync_spins(ww, wl)

    def _sync_spins(self, ww, wl):
        if not hasattr(self, "ww_spin"):
            return
        for sp, val in ((self.ww_spin, ww), (self.wl_spin, wl)):
            sp.blockSignals(True)
            sp.setValue(val)
            sp.blockSignals(False)

    def set_layout(self, key):
        self._max = None
        order = ("sagittal", "coronal", "axial")
        for vw in self.views.values():
            self.grid.removeWidget(vw)
            vw.show()
        for i in range(3):
            self.grid.setRowStretch(i, 0)
            self.grid.setColumnStretch(i, 0)
        if key == "1x3":
            for i, k in enumerate(order):
                self.grid.addWidget(self.views[k], 0, i)
                self.grid.setColumnStretch(i, 1)
            self.grid.setRowStretch(0, 1)
        elif key == "3x1":
            for i, k in enumerate(order):
                self.grid.addWidget(self.views[k], i, 0)
                self.grid.setRowStretch(i, 1)
            self.grid.setColumnStretch(0, 1)
        else:
            big = key.split("_")[1]
            small = [k for k in order if k != big]
            self.grid.addWidget(self.views[big], 0, 0, 2, 1)
            self.grid.addWidget(self.views[small[0]], 0, 1)
            self.grid.addWidget(self.views[small[1]], 1, 1)
            self.grid.setColumnStretch(0, 2)
            self.grid.setColumnStretch(1, 1)
            self.grid.setRowStretch(0, 1)
            self.grid.setRowStretch(1, 1)

    def _toggle_annot(self):
        on = not self.active.show_annot
        self._all(lambda vw: (setattr(vw, "show_annot", on), vw.update()))

    def _save_image(self):
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(self, "Save image", "mpr.png", "PNG (*.png);;JPEG (*.jpg)")
        if path:
            self.active.grab().save(path)

    def _copy_image(self):
        from PySide6.QtWidgets import QApplication
        QApplication.clipboard().setImage(self.active.grab().toImage())
        self.statusBar().showMessage("Image copied to the clipboard.", 3000)

    def _save_all(self):
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(self, "Save all views", "mpr_all.png", "PNG (*.png);;JPEG (*.jpg)")
        if path:
            self.centralWidget().grab().save(path)

    def _reset(self):
        self._all(lambda vw: vw.reset_view())
        self.model.reset()

    def update_tf_table(self):
        st = self.model.state
        lo, hi = st.vol.value_range
        _, a, rgb = st.tf.sample(lo, hi, 2048, gain=st.gain)
        self.model.tf_table = (lo, hi, (rgb * 255).astype(np.uint8), a)
        self._all(lambda vw: vw.invalidate() if vw.colormap == TF_COLORMAP else None)

    def _set_slab(self, d):
        if d == "custom":
            from PySide6.QtWidgets import QInputDialog
            mm, ok = QInputDialog.getDouble(self, "Slab thickness", "Thickness (mm):", 15, 1, 200, 1)
            if not ok:
                return
            mode, ok = QInputDialog.getItem(self, "Slab mode", "Mode:", ["MIP", "MinIP", "Average"], 0, False)
            if not ok:
                return
            d = (mm, {"MIP": "max", "MinIP": "min", "Average": "mean"}[mode])
        self.model.slab_mm, self.model.slab_mode = d
        self.model.changed.emit()

    def _toggle_max(self, view):
        if self._max is view:
            self._all(lambda vw: vw.show())
            self._max = None
        else:
            self._all(lambda vw: vw.setVisible(vw is view))
            self._max = view
