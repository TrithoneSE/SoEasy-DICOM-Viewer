"""Tractography window (DTI): FA / color-FA slices, ROI spheres, deterministic streamlines in 3D.

ROIs are placed by clicking on the slices:
  Shift+click = INCLUDE sphere (green)   — streamlines must pass through every include sphere
  Ctrl+click  = EXCLUDE sphere (red)     — streamlines passing through it are removed
  Alt+click   = SEED sphere (yellow)     — tracking starts only inside seed spheres (when "Seed spheres" is selected)
or with the Include / Exclude / Seed toolbar buttons and a plain click.
"""

from __future__ import annotations

import numpy as np
import vtkmodules.qt

vtkmodules.qt.PyQtImpl = "PySide6"

from . import vtk_lite as vtk  # noqa: E402
from vtkmodules.util import numpy_support  # noqa: E402
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor  # noqa: E402
from PySide6.QtCore import QSize, Qt  # noqa: E402
from PySide6.QtGui import QAction, QActionGroup, QIcon, QKeySequence  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QCheckBox, QComboBox, QDockWidget, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMainWindow, QMenu, QMessageBox, QPushButton, QScrollArea,
    QSizePolicy, QSlider, QSpinBox, QToolBar, QToolButton, QVBoxLayout, QWidget,
)

from . import privacy, theme  # noqa: E402
from .dti import TrackParams, fit_tensor, make_seeds, track  # noqa: E402
from .icons import icon  # noqa: E402
from .segment import np_to_vtk_image  # noqa: E402
from .ui import collapsible, color_dot  # noqa: E402

HINT = ("Include / Exclude / Seed tool or Shift / Ctrl / Alt + click on a slice = sphere • left drag = rotate • "
        "right = zoom • middle = pan • F1 = guide")
ROI_COLORS = {"include": (0.1, 0.9, 0.2), "exclude": (0.95, 0.15, 0.15), "seed": (1.0, 0.85, 0.0)}


def lines_polydata(lines, colors=None):
    pts = np.concatenate(lines).astype(np.float32)
    pd = vtk.vtkPolyData()
    vp = vtk.vtkPoints()
    vp.SetData(numpy_support.numpy_to_vtk(pts, deep=True))
    pd.SetPoints(vp)
    counts = np.array([len(l) for l in lines], np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    ca = vtk.vtkCellArray()
    ca.SetData(numpy_support.numpy_to_vtkIdTypeArray(offsets, deep=True),
               numpy_support.numpy_to_vtkIdTypeArray(np.arange(len(pts), dtype=np.int64), deep=True))
    pd.SetLines(ca)
    if colors is not None:
        c = numpy_support.numpy_to_vtk(colors, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        c.SetName("RGB")
        pd.GetPointData().SetScalars(c)
    return pd


def direction_colors(lines, D):
    """Per-point RGB from the local fibre direction in patient space (red L-R, green A-P, blue S-I)."""
    out = []
    for l in lines:
        t = np.gradient(l, axis=0) if len(l) > 1 else np.zeros_like(l)
        t = t @ D.T
        n = np.linalg.norm(t, axis=1, keepdims=True)
        out.append(np.abs(t / np.maximum(n, 1e-9)))
    return (np.clip(np.concatenate(out) * 255, 0, 255)).astype(np.uint8)


class TractWindow(QMainWindow):
    def update_title(self):
        v = self.tm.dwi
        self.setWindowTitle(f"Tractography (DTI) - {privacy.name(v.patient)} - {v.description}")

    def __init__(self, tm, targets_cb=None, parent=None):
        super().__init__(parent)
        self.tm = tm
        self.targets_cb = targets_cb or (lambda: [])
        self.update_title()
        self.resize(1500, 900)
        self.rois = []          # [(kind, center_mm, radius, actor)]
        self.lines = []
        self.tract_actor = None

        self.vtk_widget = QVTKRenderWindowInteractor(self)
        self.setCentralWidget(self.vtk_widget)
        self.ren = vtk.vtkRenderer()
        self.ren.SetBackground(0.05, 0.05, 0.07)
        self.rw = self.vtk_widget.GetRenderWindow()
        self.rw.AddRenderer(self.ren)
        self.iren = self.rw.GetInteractor()
        self.iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self._char = self.iren.AddObserver("CharEvent", lambda *_: self.iren.GetCommand(self._char).SetAbortFlag(1),
                                           100.0)
        self._click = self.iren.AddObserver("LeftButtonPressEvent", self._on_click, 10.0)
        self.vtk_widget.Initialize()

        self.setStyleSheet(theme.main_stylesheet())
        self._build_toolbar()
        self._build_panel()
        self._build_slices()
        self._set_camera()
        self.hint_lbl = QLabel(HINT)
        self.statusBar().addPermanentWidget(self.hint_lbl)

    # ============================================================== toolbar
    def _act(self, text, fn, shortcut=None, tip=None, icon_name=None, label=None, checkable=False):
        a = QAction(text, self)
        a.setCheckable(checkable)
        a.triggered.connect(fn)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        a.setToolTip((tip or text) + (f"  [{shortcut}]" if shortcut else ""))
        if icon_name:
            a.setIcon(icon(icon_name))
        if label:
            a.setIconText(label)
        self.addAction(a)
        return a

    def _build_toolbar(self):
        """Run, the three sphere tools, camera, output — one row like the other windows."""
        tb = QToolBar("Tractography")
        tb.setObjectName("tract_tools")
        tb.setMovable(False)
        tb.setIconSize(QSize(22, 22))
        tb.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.addToolBar(tb)
        tb.addAction(self._act("Run tractography", self.run, "Ctrl+Return", "Track streamlines with the settings "
                               "on the right", "run", "Run"))
        tb.addSeparator()
        self.roi_group = QActionGroup(self)
        self.roi_group.setExclusionPolicy(QActionGroup.ExclusionPolicy.ExclusiveOptional)
        self.roi_acts = {}
        for kind, label, ic, mod in (("include", "Include", "roi_include", "Shift"),
                                     ("exclude", "Exclude", "roi_exclude", "Ctrl"),
                                     ("seed", "Seed", "seed", "Alt")):
            a = self._act(f"{label} sphere", lambda _=False: self._update_hint(), None,
                          f"Click a slice to place a {kind.upper()} sphere ({mod}+click works too). "
                          f"Click the button again to rotate with the mouse.", ic, label, True)
            self.roi_group.addAction(a)
            tb.addAction(a)
            self.roi_acts[kind] = a
        tb.addSeparator()
        tb.addAction(self._act("Reset view", self._set_camera, "Ctrl+R", "Camera back to the start position",
                               "reset", "Reset"))
        tb.addAction(self._act("Show in 3D VR", self._send, None, "Show the streamlines in the 3D VR window chosen "
                               "on the right", "3d", "To 3D VR"))
        sm = QMenu(self)
        sm.addAction(self._act("Save image (PNG)...", self._snapshot, "Ctrl+S", None, "camera"))
        sm.addAction(self._act("Export tracts (.vtk, LPS — opens in 3D Slicer)...", self._export, None, None,
                               "export"))
        b = QToolButton()
        b.setIcon(icon("camera"))
        b.setText("Save")
        b.setToolTip("Save an image of the view or export the tracts")
        b.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        b.setIconSize(QSize(22, 22))
        b.setAutoRaise(True)
        b.setMenu(sm)
        b.setPopupMode(QToolButton.InstantPopup)
        tb.addWidget(b)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tb.addWidget(spacer)
        tb.addAction(self._act("Quick guide", self._help, "F1", "How to place spheres and run tracking", "help",
                               "Help"))

    def roi_mode(self):
        a = self.roi_group.checkedAction()
        return next((k for k, x in self.roi_acts.items() if x is a), None)

    def _update_hint(self):
        mode = self.roi_mode()
        if mode:
            self.hint_lbl.setText(f"Click a slice to place a <b>{mode.upper()}</b> sphere • drag elsewhere = rotate")
        else:
            self.hint_lbl.setText(HINT)

    def _help(self):
        QMessageBox.information(self, "Tractography — quick guide", (
            "1. Choose Seeding: whole brain, or seed spheres only.\n"
            "2. Place spheres on the slices: pick Include / Exclude / Seed in the toolbar and click a slice\n"
            "   (or Shift / Ctrl / Alt + click). Include = streamlines must pass through every include sphere,\n"
            "   Exclude = streamlines touching it are removed, Seed = tracking starts there.\n"
            "3. Run (Ctrl+Enter). After changing include / exclude spheres, Apply filters is enough (no re-run).\n\n"
            "Mouse: left drag = rotate • right drag / wheel = zoom • middle = pan • Ctrl+R = reset view\n"
            "Colors: red = left-right, green = front-back, blue = up-down.\n"
            "Tracts look wrong (e.g. no corpus callosum)? Try the gradient flips at the bottom of the panel.\n\n"
            "DTI has not been validated with real scanner data yet — a visualisation aid, not a diagnostic device."))

    def _snapshot(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save image", "tracts.png", "PNG (*.png)")
        if not path:
            return
        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(self.rw)
        w2i.ReadFrontBufferOff()
        w2i.Update()
        wr = vtk.vtkPNGWriter()
        wr.SetFileName(path)
        wr.SetInputConnection(w2i.GetOutputPort())
        wr.Write()
        self.statusBar().showMessage(f"Saved {path}", 6000)

    # ================================================================ panel
    def _dspin(self, lo, hi, val, step, suffix=""):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setValue(val)
        s.setSuffix(suffix)
        return s

    def _build_panel(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(8)
        p = TrackParams()

        box = QGroupBox("Tracking")
        f = QFormLayout(box)
        self.seed_mode = QComboBox()
        self.seed_mode.addItem("Whole brain", "brain")
        self.seed_mode.addItem("Seed spheres only", "spheres")
        f.addRow("Seeding:", self.seed_mode)
        self.p_seed_fa = self._dspin(0.05, 0.9, p.seed_fa, 0.05)
        f.addRow("Seed FA ≥", self.p_seed_fa)
        self.p_stop_fa = self._dspin(0.05, 0.9, p.stop_fa, 0.05)
        f.addRow("Stop FA <", self.p_stop_fa)
        self.p_angle = self._dspin(10, 90, p.max_angle, 5, "°")
        f.addRow("Max angle:", self.p_angle)
        adv = QWidget()
        fa = QFormLayout(adv)
        fa.setContentsMargins(0, 0, 0, 0)
        self.p_step = self._dspin(0.1, 3, p.step_mm, 0.1, " mm")
        fa.addRow("Step:", self.p_step)
        self.p_min = self._dspin(0, 300, p.min_len, 5, " mm")
        fa.addRow("Min length:", self.p_min)
        self.p_max = self._dspin(20, 500, p.max_len, 10, " mm")
        fa.addRow("Max length:", self.p_max)
        self.p_stride = QSpinBox()
        self.p_stride.setRange(1, 6)
        self.p_stride.setValue(p.stride)
        self.p_stride.setToolTip("Whole-brain seeding: one seed every N voxels")
        fa.addRow("Seed spacing:", self.p_stride)
        self.p_spv = QSpinBox()
        self.p_spv.setRange(1, 20)
        self.p_spv.setValue(4)
        self.p_spv.setToolTip("Seeds per voxel inside seed spheres")
        fa.addRow("Seeds per voxel:", self.p_spv)
        f.addRow(collapsible("More parameters", adv))
        b = QPushButton(icon("run"), "  Run tractography")
        b.setObjectName("primary")
        b.setToolTip("Ctrl+Enter")
        b.clicked.connect(self.run)
        f.addRow(b)
        self.result_lbl = QLabel("No streamlines yet.")
        self.result_lbl.setStyleSheet(f"color: {theme.MUTED};")
        f.addRow(self.result_lbl)
        v.addWidget(box)

        box = QGroupBox("ROI spheres")
        bl = QVBoxLayout(box)
        legend = QHBoxLayout()
        for kind, text in (("include", "Include (Shift)"), ("exclude", "Exclude (Ctrl)"), ("seed", "Seed (Alt)")):
            lb = QLabel()
            lb.setPixmap(color_dot(ROI_COLORS[kind]).pixmap(12, 12))
            legend.addWidget(lb)
            legend.addWidget(QLabel(text))
            legend.addSpacing(6)
        legend.addStretch(1)
        bl.addLayout(legend)
        row = QHBoxLayout()
        row.addWidget(QLabel("New sphere radius:"))
        self.roi_radius = self._dspin(1, 40, 6, 1, " mm")
        row.addWidget(self.roi_radius)
        bl.addLayout(row)
        self.roi_list = QListWidget()
        self.roi_list.setMaximumHeight(120)
        bl.addWidget(self.roi_list)
        row = QHBoxLayout()
        for text, ic, fn, tip in (("Delete", "trash", self._delete_roi, "Delete the selected sphere"),
                                  ("Clear", None, self._clear_rois, "Delete all spheres"),
                                  ("Apply filters", None, self._refilter,
                                   "Re-apply include / exclude spheres to the tracked streamlines (no re-run)")):
            bt = QPushButton(icon(ic) if ic else QIcon(), text)
            bt.setToolTip(tip)
            bt.clicked.connect(fn)
            row.addWidget(bt)
        bl.addLayout(row)
        v.addWidget(box)

        box = QGroupBox("Background slices")
        f = QFormLayout(box)
        self.bg_combo = QComboBox()
        self.bg_combo.addItems(["Color FA", "FA", "b0", "MD"])
        self.bg_combo.currentIndexChanged.connect(lambda _: self._build_slices())
        f.addRow("Image:", self.bg_combo)
        self.slice_sliders = {}
        nz, ny, nx = self.tm.fa.shape
        for name, n in (("Axial", nz), ("Coronal", ny), ("Sagittal", nx)):
            row = QHBoxLayout()
            cb = QCheckBox()
            cb.setChecked(True)
            cb.setToolTip(f"Show the {name.lower()} slice")
            sl = QSlider(Qt.Horizontal)
            sl.setRange(0, n - 1)
            sl.setValue(n // 2)
            sl.valueChanged.connect(lambda _, nm=name: self._update_slice(nm))
            cb.toggled.connect(lambda _, nm=name: self._update_slice(nm))
            row.addWidget(cb)
            row.addWidget(sl)
            self.slice_sliders[name] = (cb, sl)
            f.addRow(name + ":", row)
        self.slice_opacity = QSlider(Qt.Horizontal)
        self.slice_opacity.setRange(10, 100)
        self.slice_opacity.setValue(100)
        self.slice_opacity.valueChanged.connect(lambda _: [self._update_slice(n) for n in self.slice_sliders])
        f.addRow("Slice opacity:", self.slice_opacity)
        v.addWidget(box)

        box = QGroupBox("Streamline display")
        f = QFormLayout(box)
        self.color_mode = QComboBox()
        self.color_mode.addItems(["Direction (RGB)", "Single color"])
        self.color_mode.currentIndexChanged.connect(lambda _: self._show_lines())
        f.addRow("Color:", self.color_mode)
        self.line_width = self._dspin(0.5, 6, 1.5, 0.5)
        self.line_width.valueChanged.connect(lambda _: self._style_actor())
        f.addRow("Line width:", self.line_width)
        self.tubes = QCheckBox("Tubes (slower)")
        self.tubes.toggled.connect(lambda _: self._show_lines())
        f.addRow(self.tubes)
        self.tract_opacity = QSlider(Qt.Horizontal)
        self.tract_opacity.setRange(5, 100)
        self.tract_opacity.setValue(100)
        self.tract_opacity.valueChanged.connect(lambda _: self._style_actor())
        f.addRow("Opacity:", self.tract_opacity)
        v.addWidget(box)

        box = QGroupBox("Show in a 3D VR window")
        f = QVBoxLayout(box)
        row = QHBoxLayout()
        self.target_combo = QComboBox()
        row.addWidget(self.target_combo, 1)
        bt = QToolButton()
        bt.setIcon(icon("reload"))
        bt.setAutoRaise(True)
        bt.setToolTip("Refresh the list of open 3D VR windows")
        bt.clicked.connect(self._refresh_targets)
        row.addWidget(bt)
        f.addLayout(row)
        bt = QPushButton(icon("3d"), "Show tracts in this 3D VR window")
        bt.clicked.connect(self._send)
        f.addWidget(bt)
        bt = QPushButton(icon("export"), "Export tracts (.vtk, LPS — opens in 3D Slicer)")
        bt.clicked.connect(self._export)
        f.addWidget(bt)
        v.addWidget(box)

        flips = QWidget()
        fv = QVBoxLayout(flips)
        fv.setContentsMargins(0, 0, 0, 0)
        f = QHBoxLayout()
        self.flips = []
        for ax, val in zip("XYZ", self.tm.flip):
            cb = QCheckBox(f"Flip {ax}")
            cb.setChecked(val < 0)
            self.flips.append(cb)
            f.addWidget(cb)
        f.addStretch(1)
        fv.addLayout(f)
        bt = QPushButton("Refit the tensor with these flips")     # own row: beside the flips it overflowed
        bt.clicked.connect(self._refit)
        fv.addWidget(bt)
        v.addWidget(collapsible("Gradient orientation (if tracts look wrong)", flips))
        v.addStretch(1)
        self._refresh_targets()

        for cb in w.findChildren(QComboBox):         # do not size to the longest item: must fit the dock width
            cb.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            cb.setMinimumContentsLength(8)
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sa.setWidget(w)
        dock = QDockWidget("DTI tractography", self)
        dock.setObjectName("tract_panel")
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        dock.setWidget(sa)
        dock.setMinimumWidth(360)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

    # =============================================================== slices
    def _background(self):
        tm = self.tm
        name = self.bg_combo.currentText()
        if name == "Color FA":
            return np.ascontiguousarray(tm.color_fa()), None
        arr = {"FA": tm.fa, "b0": tm.b0, "MD": tm.md}[name].astype(np.float32)
        lo, hi = np.percentile(arr[tm.mask] if tm.mask.any() else arr, (1, 99.5))
        return np.ascontiguousarray(np.clip((arr - lo) / max(1e-9, hi - lo) * 255, 0, 255).astype(np.uint8)), 1

    def _build_slices(self):
        for a in getattr(self, "_slice_actors", {}).values():
            self.ren.RemoveActor(a)
        data, gray = self._background()
        d = self.tm.dwi
        if gray is None:
            nz, ny, nx, _ = data.shape
            img = vtk.vtkImageData()
            img.SetDimensions(nx, ny, nz)
            img.SetSpacing(d.spacing[2], d.spacing[1], d.spacing[0])
            arr = numpy_support.numpy_to_vtk(data.reshape(-1, 3), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
            img.GetPointData().SetScalars(arr)
        else:
            img = np_to_vtk_image(data, d.spacing)
        self._img = img
        self._slice_actors = {}
        for name in ("Axial", "Coronal", "Sagittal"):
            a = vtk.vtkImageActor()
            a.GetMapper().SetInputData(img)
            a.InterpolateOn()
            self._slice_actors[name] = a
            self.ren.AddActor(a)
            self._update_slice(name, render=False)
        self.rw.Render()

    def _update_slice(self, name, render=True):
        a = self._slice_actors.get(name)
        if a is None:
            return
        cb, sl = self.slice_sliders[name]
        k = sl.value()
        nz, ny, nx = self.tm.fa.shape
        ext = {"Axial": (0, nx - 1, 0, ny - 1, k, k), "Coronal": (0, nx - 1, k, k, 0, nz - 1),
               "Sagittal": (k, k, 0, ny - 1, 0, nz - 1)}[name]
        a.SetDisplayExtent(*ext)
        a.SetVisibility(cb.isChecked())
        a.SetOpacity(self.slice_opacity.value() / 100)
        if render:
            self.rw.Render()

    def _set_camera(self):
        cam = self.ren.GetActiveCamera()
        b = self._img.GetBounds()
        c = [(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2]
        D = np.asarray(self.tm.dwi.direction, float)
        d, up = D.T @ np.array([0.6, -1.0, 0.5]), D.T @ np.array([0, 0, 1.0])
        dist = max(b[1] - b[0], b[3] - b[2], b[5] - b[4]) * 2.5
        cam.SetFocalPoint(*c)
        cam.SetPosition(*(np.array(c) + d / np.linalg.norm(d) * dist))
        cam.SetViewUp(*up)
        self.ren.ResetCamera()
        self.rw.Render()

    # ================================================================= ROIs
    def _on_click(self, *_):
        shift, ctrl, alt = self.iren.GetShiftKey(), self.iren.GetControlKey(), self.iren.GetAltKey()
        mode = self.roi_mode()
        if not (shift or ctrl or alt or mode):
            return
        self.iren.GetCommand(self._click).SetAbortFlag(1)
        x, y = self.iren.GetEventPosition()
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.002)
        for a in self._slice_actors.values():
            if a.GetVisibility():
                picker.AddPickList(a)
        picker.PickFromListOn()
        if not picker.Pick(x, y, 0, self.ren):
            if mode and not (shift or ctrl or alt):
                self.iren.GetCommand(self._click).SetAbortFlag(0)     # tool on, missed the slices: rotate as usual
            else:
                self.statusBar().showMessage("Click on one of the slices to place a sphere.", 4000)
            return
        kind = "include" if shift else ("exclude" if ctrl else ("seed" if alt else mode))
        self._add_roi(kind, np.array(picker.GetPickPosition()), self.roi_radius.value())

    def _add_roi(self, kind, center, r):
        s = vtk.vtkSphereSource()
        s.SetCenter(*center)
        s.SetRadius(r)
        s.SetThetaResolution(24)
        s.SetPhiResolution(16)
        m = vtk.vtkPolyDataMapper()
        m.SetInputConnection(s.GetOutputPort())
        a = vtk.vtkActor()
        a.SetMapper(m)
        a.GetProperty().SetColor(*ROI_COLORS[kind])
        a.GetProperty().SetOpacity(0.35)
        self.ren.AddActor(a)
        self.rois.append((kind, center, r, a))
        self.roi_list.addItem(QListWidgetItem(color_dot(ROI_COLORS[kind]), f"{kind.capitalize()}  r = {r:.0f} mm  at "
                                              f"({center[0]:.0f}, {center[1]:.0f}, {center[2]:.0f})"))
        if kind == "seed":
            self.seed_mode.setCurrentIndex(1)
        self.rw.Render()

    def _delete_roi(self):
        i = self.roi_list.currentRow()
        if 0 <= i < len(self.rois):
            self.ren.RemoveActor(self.rois.pop(i)[3])
            self.roi_list.takeItem(i)
            self.rw.Render()

    def _clear_rois(self):
        for r in self.rois:
            self.ren.RemoveActor(r[3])
        self.rois = []
        self.roi_list.clear()
        self.rw.Render()

    # ============================================================ tracking
    def _params(self):
        return TrackParams(seed_fa=self.p_seed_fa.value(), stop_fa=self.p_stop_fa.value(),
                           max_angle=self.p_angle.value(), step_mm=self.p_step.value(), min_len=self.p_min.value(),
                           max_len=self.p_max.value(), stride=self.p_stride.value(),
                           seeds_per_voxel=self.p_spv.value() if self.seed_mode.currentData() == "spheres" else 1)

    def _progress(self, i, n, msg):
        self.statusBar().showMessage(f"{msg} {i}/{n}")
        QApplication.processEvents()

    def run(self):
        p = self._params()
        seeds_rois = [(c, r) for k, c, r, _ in self.rois if k == "seed"]
        if self.seed_mode.currentData() == "spheres":
            if not seeds_rois:
                QMessageBox.information(self, "No seed sphere", "Alt+click on a slice to place a seed sphere first.")
                return
            seeds = np.concatenate([make_seeds(self.tm, p, roi=sr) for sr in seeds_rois])
        else:
            seeds = make_seeds(self.tm, p)
        if len(seeds) == 0:
            QMessageBox.information(self, "No seeds", "No voxels above the seed FA threshold in the seed region.")
            return
        self.setCursor(Qt.WaitCursor)
        try:
            self.all_lines = track(self.tm, seeds, p, progress=self._progress)
        finally:
            self.unsetCursor()
        self._refilter()

    def _refilter(self):
        lines = getattr(self, "all_lines", [])
        inc = [(c, r) for k, c, r, _ in self.rois if k == "include"]
        exc = [(c, r) for k, c, r, _ in self.rois if k == "exclude"]
        out = []
        for l in lines:
            if any(np.min(np.linalg.norm(l - c, axis=1)) > r for c, r in inc):
                continue
            if any(np.min(np.linalg.norm(l - c, axis=1)) <= r for c, r in exc):
                continue
            out.append(l)
        self.lines = out
        self.result_lbl.setText(f"{len(out)} streamlines shown ({len(lines)} tracked)")
        self.statusBar().showMessage(f"{len(out)} streamlines.", 6000)
        self._show_lines()

    def _make_actor(self, lines, D_target_points=None):
        D = np.asarray(self.tm.dwi.direction, float)
        colors = direction_colors(lines, D) if self.color_mode.currentIndex() == 0 else None
        pd = lines_polydata(lines if D_target_points is None else D_target_points, colors)
        src = pd
        if self.tubes.isChecked() and len(lines) < 4000:
            tf = vtk.vtkTubeFilter()
            tf.SetInputData(pd)
            tf.SetRadius(0.35)
            tf.SetNumberOfSides(6)
            tf.Update()
            src = tf.GetOutput()
        m = vtk.vtkPolyDataMapper()
        m.SetInputData(src)
        if colors is not None:
            m.SetScalarModeToUsePointData()
            m.SetColorModeToDirectScalars()
        else:
            m.ScalarVisibilityOff()
        a = vtk.vtkActor()
        a.SetMapper(m)
        a.GetProperty().SetColor(1.0, 0.85, 0.2)
        a.GetProperty().SetLineWidth(self.line_width.value())
        a.GetProperty().SetOpacity(self.tract_opacity.value() / 100)
        return a

    def _show_lines(self):
        if self.tract_actor is not None:
            self.ren.RemoveActor(self.tract_actor)
            self.tract_actor = None
        if self.lines:
            self.tract_actor = self._make_actor(self.lines)
            self.ren.AddActor(self.tract_actor)
        self.rw.Render()

    def _style_actor(self):
        if self.tract_actor is not None:
            self.tract_actor.GetProperty().SetLineWidth(self.line_width.value())
            self.tract_actor.GetProperty().SetOpacity(self.tract_opacity.value() / 100)
            self.rw.Render()

    # ============================================================== output
    def _lines_lps(self):
        d = self.tm.dwi
        D = np.asarray(d.direction, float)
        o = np.asarray(d.origin, float)
        return [o + l[:, 0:1] * D[:, 0] + l[:, 1:2] * D[:, 1] + l[:, 2:3] * D[:, 2] for l in self.lines]

    def _refresh_targets(self):
        self.target_combo.clear()
        for label, win in self.targets_cb():
            self.target_combo.addItem(label, win)
        if self.target_combo.count() == 0:
            self.target_combo.addItem("(open a 3D VR window first)", None)

    def _send(self):
        win = self.target_combo.currentData()
        if win is None or not self.lines:
            QMessageBox.information(self, "Nothing to show", "Run tractography and open a 3D VR window first.")
            return
        vol = win.state.vol
        sz, sy, sx = vol.spacing
        mapped = []
        for l in self._lines_lps():
            zyx = vol.world_to_index(l)
            mapped.append(np.stack([zyx[:, 2] * sx, zyx[:, 1] * sy, zyx[:, 0] * sz], 1).astype(np.float32))
        win.vr.set_extra("tracts", self._make_actor(self.lines, mapped))
        win.raise_()
        self.statusBar().showMessage(f"{len(mapped)} streamlines sent to '{vol.description}'.", 6000)

    def _export(self):
        if not self.lines:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export tracts", "tracts.vtk", "VTK polydata (*.vtk)")
        if not path:
            return
        pd = lines_polydata(self._lines_lps(), direction_colors(self.lines, np.asarray(self.tm.dwi.direction)))
        w = vtk.vtkPolyDataWriter()
        w.SetFileName(path)
        w.SetHeader("SPACE=LPS")
        w.SetInputData(pd)
        w.Write()
        self.statusBar().showMessage(f"Saved {path}", 6000)

    def _refit(self):
        flip = tuple(-1 if cb.isChecked() else 1 for cb in self.flips)
        self.setCursor(Qt.WaitCursor)
        try:
            self.tm = fit_tensor(self.tm.dwi, flip=flip, progress=self._progress)
        finally:
            self.unsetCursor()
        self._build_slices()
        self.all_lines, self.lines = [], []
        self._show_lines()
        self.result_lbl.setText("Tensor refitted — run tractography again.")
