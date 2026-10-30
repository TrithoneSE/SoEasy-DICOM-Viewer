"""Separate 3D window: volume rendering + color curve editor + segmentation + 3D measurements."""

from __future__ import annotations

import os
import re

import numpy as np
from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QActionGroup, QColor, QIcon, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QColorDialog, QComboBox, QDockWidget, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QMenu, QPushButton, QScrollArea, QSizePolicy, QSlider, QSpinBox, QTabWidget, QToolBar, QToolButton,
    QVBoxLayout, QWidget,
)

from . import privacy, theme, tf_model
from .busy import BusyDialog
from .icons import icon
from .snapshots import SnapshotPanel
from .ui import collapsible, hint
from .segment import Segment, auto_grow, local_stats, region_grow
from .tf_editor import NEW_CURVE_COLORS, TFEditor
from .tools import abc2_dialog
from .volume_view import MEASURE3D, VolumeView


def _swatch(rgb, size=16):
    pm = QPixmap(size, size)
    pm.fill(QColor(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))
    return QIcon(pm)


# Rim threshold slider: logarithmic 5..200 HU. On real CT the useful part is ~5–60 HU (homogeneous tissue has a
# gradient of a few HU; the rims around contrast vessels ~20–80); a linear 10–300 slider spent most of its travel
# where nothing changes any more.
EDGE_MIN, EDGE_MAX = 5.0, 200.0


def _edge_hu(pos):
    return EDGE_MIN * (EDGE_MAX / EDGE_MIN) ** (pos / 100)


def _edge_pos(hu):
    return int(round(100 * np.log(np.clip(hu, EDGE_MIN, EDGE_MAX) / EDGE_MIN) / np.log(EDGE_MAX / EDGE_MIN)))


def _scroll(page):
    """Vertical scrolling only: the panel content must fit the dock's width. A combo box otherwise asks for the
    width of its longest item and pushed the Segments tab wider than the dock (clipped spin-box arrows)."""
    for cb in page.findChildren(QComboBox):
        cb.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        cb.setMinimumContentsLength(8)
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    sa.setWidget(page)
    return sa


class VRWindow(QMainWindow):
    tfChanged = Signal(object)          # state
    segmentsChanged = Signal(object)    # state
    closed = Signal(object)             # state

    def update_title(self):
        v = self.state.vol
        self.setWindowTitle(f"3D VR - {privacy.name(v.patient)} - {v.description}")

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        v = state.vol
        self.update_title()
        self.resize(1500, 900)
        self.setStyleSheet(theme.main_stylesheet())
        self.current_preset_path = None

        self.vr = VolumeView()
        self.vr.cutDone.connect(lambda _: self._after_cut())
        self.vr.inspected.connect(self._on_inspected)
        self.vr.status.connect(lambda m: self.statusBar().showMessage(m, 8000))
        self.vr.measurementsChanged.connect(self._refresh_meas)
        self.vr.measureSelected.connect(self._on_meas_selected)
        self.vr.toolFinished.connect(self._release_tool)
        self.setCentralWidget(self.vr)

        self._build_toolbar()
        dock = QDockWidget("3D settings", self)
        dock.setObjectName("panel3d")
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        tabs = QTabWidget()
        tabs.addTab(_scroll(self._build_color_tab()), icon("palette"), "Colors")
        tabs.addTab(_scroll(self._build_seg_tab()), icon("segment"), "Segments")
        tabs.addTab(self._build_meas_tab(), icon("ruler"), "Measurements")
        dock.setWidget(tabs)
        dock.setMinimumWidth(380)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self.tabs = tabs

        self._tf_timer = QTimer(self)
        self._tf_timer.setSingleShot(True)
        self._tf_timer.setInterval(25)
        self._tf_timer.timeout.connect(lambda: self.apply_tf(fast=True))
        self._fine_timer = QTimer(self)
        self._fine_timer.setSingleShot(True)
        self._fine_timer.setInterval(400)
        self._fine_timer.timeout.connect(self.apply_tf)

        self._load_preset_list()
        lo, hi = v.value_range
        self.editor.set_data_range(lo, hi, v.data)
        self._sync_controls_from_tf()
        self.editor.set_tf(state.tf)
        self.editor.fit_view_to_curves()
        self._update_encoding()
        self._show_bone_marker()
        self.vr.set_state(state)
        self.apply_tf()
        self._refresh_seg_list()
        self._show_res()

    # ================================================================ UI
    def _act(self, text, fn, shortcut=None, tip=None, checkable=False, label=None, icon_name=None):
        """Action with a shortcut that works anywhere in the window (also when it only sits in a ▾ menu)."""
        a = QAction(text, self)
        a.setCheckable(checkable)
        (a.toggled if checkable else a.triggered).connect(fn)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        a.setToolTip((tip or text) + (f"  [{shortcut}]" if shortcut else ""))
        if label:
            a.setIconText(label)            # short label under the toolbar icon
        if icon_name:
            a.setIcon(icon(icon_name))
        self.addAction(a)
        return a

    def _tb_button(self, icon_name, label, tip, action=None, menu=None):
        """Toolbar button: icon + label. With `action` the button runs it and ▾ opens `menu`;
        without, a click opens the menu."""
        b = QToolButton()
        b.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        b.setIconSize(QSize(22, 22))
        b.setAutoRaise(True)
        if action:
            b.setDefaultAction(action)
        else:
            b.setIcon(icon(icon_name))
            b.setText(label)
            b.setToolTip(tip)
        if menu:
            b.setMenu(menu)
            b.setPopupMode(QToolButton.MenuButtonPopup if action else QToolButton.InstantPopup)
        return b

    def _build_toolbar(self):
        """One row like the main window: render mode, views, crop, measure, scalpel, inspect, undo, save, quality."""
        tb = QToolBar("3D")
        tb.setObjectName("tools3d")
        tb.setMovable(False)
        tb.setIconSize(QSize(22, 22))
        tb.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.addToolBar(tb)

        self._blend = "composite"
        bm = QMenu(self)
        group = QActionGroup(self)
        self.blend_acts = {}
        for label, mode, tip in (("VR", "composite", "Volume rendering with the color curves"),
                                 ("MIP", "mip", "Maximum intensity projection"),
                                 ("MinIP", "minip", "Minimum intensity projection")):
            a = bm.addAction(f"{label}  —  {tip}")
            a.setCheckable(True)
            group.addAction(a)
            a.triggered.connect(lambda _=False, m=mode: self.set_blend(m))
            self.blend_acts[mode] = (a, label)
        self.blend_acts["composite"][0].setChecked(True)
        self.blend_btn = self._tb_button("3d", "VR", "Render mode: VR / MIP / MinIP", menu=bm)
        tb.addWidget(self.blend_btn)
        tb.addSeparator()

        vm = QMenu(self)
        for name, label, sc in [("anterior", "Front", "1"), ("posterior", "Back", "2"), ("right", "Right", "3"),
                                ("left", "Left", "4"), ("superior", "Top", "5"), ("inferior", "Bottom", "6")]:
            vm.addAction(self._act(label, lambda _=False, n=name: self.vr.view(n), sc))
        vm.addSeparator()
        for a in (self._act("Zoom in", lambda: self.vr.zoom(1.25), "+", icon_name="zoom"),
                  self._act("Zoom out", lambda: self.vr.zoom(0.8), "-", icon_name="zoom_out"),
                  self._act("Fit to window", self.vr.fit, "0", icon_name="fullscreen"),
                  self._act("Rotate 90°", lambda: self.vr.rotate(90), icon_name="rotate_r")):
            vm.addAction(a)
        tb.addWidget(self._tb_button("eye", "Views", "Standard views (keys 1–6), zoom (+ / −), fit (0), rotate 90°",
                                     menu=vm))
        # what a left-drag does: three side-by-side buttons (one is always on), not a ▾ menu
        tb.addSeparator()
        group = QActionGroup(self)
        group.setExclusive(True)
        self.rot_acts = {}
        for mode, label, tip, ic in (
                ("free", "Rotate", "Left-drag = free 3D rotate (trackball)", "orbit"),
                ("roll", "Roll", "Left-drag around the centre = turn the picture in the screen plane only", "roll"),
                ("turn", "Turn", "Left-drag left / right = turn around the head's long axis only", "turn")):
            a = self._act(label, lambda on, m=mode: on and self.set_rotate_mode(m), None,
                          tip + ". Right-drag / wheel = zoom, middle = pan.", True, label, ic)
            group.addAction(a)
            tb.addAction(a)
            self.rot_acts[mode] = (a, label, ic)
        self.rot_acts["free"][0].setChecked(True)
        tb.addSeparator()

        # crop: on/off, show/hide the box (used often) and reset, all as their own buttons
        self.crop_act = self._act("Crop box", self._on_crop, "Shift+C", "Crop box on/off", True, "Crop", "crop")
        self.handles_act = self._act("Show crop box", lambda on: self.vr.set_crop_handles(on), "H",
                                     "Show / hide the crop box (the volume stays cropped)", True, "Box", "eye")
        self.handles_act.setChecked(True)
        tb.addAction(self.crop_act)
        tb.addAction(self.handles_act)
        tb.addAction(self._act("Reset crop", self.vr.reset_crop, None, "Back to the full volume", False,
                               "Reset crop", "reset"))
        tb.addSeparator()

        # measurement menu, same layout as the 2D one
        self.measure_kind = "length"
        mb = QToolButton()
        mb.setCheckable(True)
        mb.setPopupMode(QToolButton.MenuButtonPopup)
        mb.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        mb.setIconSize(QSize(22, 22))
        mb.setAutoRaise(True)
        mb.setToolTip("3D measurements: press on the surface and drag (like 2D). After one measurement the mouse "
                      "rotates again.\nClick a measurement (cyan) + Delete = delete it; right-click it = menu.")
        menu = QMenu(mb)
        for k, (name, sc) in MEASURE3D.items():
            menu.addAction(icon(k), f"{name}\t{sc}").triggered.connect(lambda _=False, kk=k: self.select_measure(kk))
            sa = QAction(self)
            sa.setShortcut(QKeySequence(sc))
            sa.triggered.connect(lambda _=False, kk=k: self.select_measure(kk))
            self.addAction(sa)
        menu.addSeparator()
        menu.addAction("Delete selected measurement\tDel").triggered.connect(self._delete_selected_meas)
        menu.addAction("Delete all 3D measurements\tCtrl+Del").triggered.connect(self.vr.clear_measurements3d)
        menu.addSeparator()
        menu.addAction("ABC/2 volume calculator...").triggered.connect(self._abc2)
        mb.setMenu(menu)
        mb.toggled.connect(lambda on: self._set_mode(self.measure_kind, on))
        self.measure_btn = mb
        self._label_measure_btn()
        tb.addWidget(mb)
        a = QAction(self)
        a.setShortcut(QKeySequence("Ctrl+Delete"))
        a.triggered.connect(self.vr.clear_measurements3d)
        self.addAction(a)

        self.cut_act = self._act("Scalpel", lambda on: self._set_mode("scalpel", on), "X",
                                 "Draw an area (click points or drag), then Cut inside (Del) / Cut outside (Enter) / "
                                 "Clear area (Esc)", True, "Scalpel", "scalpel")
        self.undo_act = self._act("Undo cut", self._undo_cut, "Ctrl+Z", "Undo the last cut", label="Undo",
                                  icon_name="undo")
        km = QMenu(self)
        km.addAction(self.undo_act)
        km.addAction(self._act("Reset all cuts", self._reset_cut, None, "Bring back everything that was cut"))
        tb.addWidget(self._tb_button("scalpel", "Scalpel", "", self.cut_act, km))
        self.inspect_act = self._act("Inspect", lambda on: self._set_mode("inspect", on), "I",
                                     "Click the 3D surface → show HU & the curve that colors it", True, "Inspect",
                                     "inspect")
        tb.addAction(self.inspect_act)
        tb.addAction(self.undo_act)
        tb.addSeparator()

        tb.addAction(self._act("Save PNG...", self._snapshot, "Ctrl+S", "Save the 3D view as a PNG image", False,
                               "Save PNG", "camera"))
        # 3D snapshots: saved views to jump between, save / load, and a movie through them
        self.snap_panel = None
        tb.addAction(self._act("3D snapshots", self.show_snapshots, None,
                               "3D snapshots: save views (camera, crop, colors), jump between them, make a movie",
                               False, "Snapshots", "image"))
        self._act("Add 3D snapshot", lambda: self.show_snapshots().add(), "Insert")
        self._act("Previous 3D snapshot", lambda: self.show_snapshots().step(-1), "PgUp")
        self._act("Next 3D snapshot", lambda: self.show_snapshots().step(1), "PgDown")
        qm = QMenu(self)
        group = QActionGroup(self)
        self.res_acts = {}
        for text, factor in (("Resolution: auto", None), ("Resolution: full", 1), ("Resolution: half (¼ memory)", 2)):
            a = qm.addAction(text)
            a.setCheckable(True)
            group.addAction(a)
            a.triggered.connect(lambda _=False, f=factor: self._on_res(f))
            self.res_acts[factor] = a
        (self.res_acts.get(self.state.render_factor) or self.res_acts[None]).setChecked(True)
        qm.addSeparator()
        qm.addAction(self._act("Redraw", self._redraw, "F5", "Re-upload the volume and redraw the 3D view"))
        tb.addWidget(self._tb_button("quality", "Quality",
                                     "3D resolution (half = 4× less memory, faster; auto uses half for large volumes "
                                     "when free RAM is low) and Redraw [F5]", menu=qm))
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tb.addWidget(spacer)
        tb.addAction(self._act("Quick guide", self._help, "F1", "Mouse, tools and shortcuts of the 3D window", False,
                               "Help", "help"))
        self.mode_acts = {"scalpel": self.cut_act, "inspect": self.inspect_act}

    def _build_color_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(8, 10, 8, 8)
        v.setSpacing(8)
        row = QHBoxLayout()
        row.addWidget(QLabel("Preset"))
        self.preset_combo = QComboBox()
        self.preset_combo.activated.connect(self._on_preset_selected)
        row.addWidget(self.preset_combo, 1)
        sb = QToolButton()
        sb.setIcon(icon("save"))
        sb.setAutoRaise(True)
        sb.setToolTip("Save the curves into this preset")
        sb.clicked.connect(self._save_preset)
        row.addWidget(sb)
        more = QToolButton()
        more.setIcon(icon("more"))
        more.setAutoRaise(True)
        more.setToolTip("More: save as, import, export, delete")
        more.setPopupMode(QToolButton.InstantPopup)
        more.setStyleSheet("QToolButton::menu-indicator { image: none; width: 0; }")
        m = QMenu(more)
        m.addAction(icon("save"), "Save", self._save_preset)
        m.addAction("Save as...", self._save_preset_as)
        m.addSeparator()
        m.addAction("Import...", self._import_preset)
        m.addAction("Export...", self._export_preset)
        m.addSeparator()
        m.addAction(icon("trash"), "Delete preset", self._delete_preset)
        more.setMenu(m)
        row.addWidget(more)
        v.addLayout(row)
        self.editor = TFEditor()
        self.editor.setMinimumHeight(260)
        self.editor.changed.connect(self.schedule_tf)
        self.editor.boneLockMoved.connect(lambda hu: self.bone_hu_spin.setValue(hu))
        v.addWidget(self.editor, 3)
        v.addWidget(collapsible("Curve editor tips", hint(
            "• Drag points or lines. Double-click empty space = new curve, double-click a line = new point, "
            "double-click a point = its color.<br>"
            "• Right-click = menu (🦴 bone curve, color, delete). Alt+drag = shift all. Wheel = zoom HU.<br>"
            "• Color picker (K in the main window): click a tissue in 2D → automatic curve.<br>"
            "• The dashed 🦴 line marks where the bone area starts: drag it left / right. The bone look itself is "
            "the dashed 🦴 curve (raise it = more solid bone, move it right = less haze).<br>"
            "• <b>Tumor vs contrast in vessels</b>: the wall of every bright vessel or bone passes through the HU of an "
            "enhancing tumor. Right-click the tumor curve → <b>Hide partial-volume rims</b> (its label then says <i>rims hidden</i>): it colors only "
            "homogeneous tissue, so vessels keep their own color (fine-tune with Rim threshold below).")))

        box = QGroupBox("Bone color lock")
        f = QFormLayout(box)
        self.bone_chk = QCheckBox("Bone area: only 🦴 bone curves")
        self.bone_chk.setToolTip(
            "Without this, the bone edge (300–700 HU) is colored by other curves (e.g. vascular), so bone\n"
            "looks red. With this, the bone area (+ its edge) only uses curves marked 🦴.")
        self.bone_chk.setChecked(self.state.bone_lock)
        self.bone_chk.toggled.connect(self._on_bone_changed)
        f.addRow(self.bone_chk)
        self.bone_hu_spin = self._dspin(100, 3000, self.state.bone_lock_hu, " HU", self._on_bone_changed)
        f.addRow("Bone starts at:", self.bone_hu_spin)
        self.bone_close_spin = self._dspin(0, 10, self.state.bone_close_mm, " mm", self._on_bone_changed, 0.5)
        f.addRow("Fill bone cavities (diploe):", self.bone_close_spin)
        self.bone_rim_spin = self._dspin(0, 5, self.state.bone_rim_mm, " mm", self._on_bone_changed, 0.5)
        f.addRow("Bone edge width:", self.bone_rim_spin)
        v.addWidget(box)

        box = QGroupBox("Rendering")
        f = QFormLayout(box)
        self.shade_chk = QCheckBox("Shading (3D lighting)")
        self.shade_chk.toggled.connect(self.schedule_tf)
        f.addRow(self.shade_chk)
        self.gain_slider = self._slider(1, 400, int(self.state.gain * 100), self.schedule_tf)
        f.addRow("Global opacity:", self.gain_slider)
        row = QHBoxLayout()
        self.edge_slider = self._slider(0, 100, _edge_pos(self.state.tf.edge_threshold), self._on_edge_changed)
        self.edge_lbl = QLabel()
        self.edge_lbl.setMinimumWidth(52)
        row.addWidget(self.edge_slider, 1)
        row.addWidget(self.edge_lbl)
        f.addRow("Rim threshold:", row)
        self.edge_hint = hint("Right-click the tumor curve → Hide partial-volume rims to use this.")
        f.addRow(self.edge_hint)
        adv = QWidget()
        fa = QFormLayout(adv)
        fa.setContentsMargins(0, 0, 0, 0)
        self.clean_spin = QSpinBox()
        self.clean_spin.setRange(0, 4)
        self.clean_spin.setSuffix(" voxel")
        self.clean_spin.setValue(self.state.edge_clean)
        self.clean_spin.setToolTip("Neutralise bone-edge voxels below 250 HU (partial volume).")
        self.clean_spin.valueChanged.connect(self._on_clean_changed)
        fa.addRow("Clean bone edge:", self.clean_spin)
        self.amb_slider = self._slider(0, 100, 25, self.schedule_tf)
        fa.addRow("Ambient:", self.amb_slider)
        self.dif_slider = self._slider(0, 100, 75, self.schedule_tf)
        fa.addRow("Diffuse:", self.dif_slider)
        self.spec_slider = self._slider(0, 100, 30, self.schedule_tf)
        fa.addRow("Specular:", self.spec_slider)
        self.quality_slider = self._slider(25, 400, 100, self.schedule_tf)
        fa.addRow("Quality:", self.quality_slider)
        bg = QPushButton("Background color...")
        bg.clicked.connect(self._pick_bg)
        fa.addRow(bg)
        f.addRow(collapsible("Lighting, quality, background", adv))
        v.addWidget(box)
        return w

    def _build_seg_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(8, 10, 8, 8)
        v.setSpacing(8)
        v.addWidget(hint("<b>Ctrl+click</b> inside the lesion in a 2D image (main window or MPR) to grow a segment. "
                          "Refine it with the <b>Brush</b> (B) in 2D; Shift = erase.", muted=False))
        v.addWidget(collapsible("How the range is chosen", hint(
            "<b>Automatic</b> reads the values around the click (mean ± k·SD). <b>Anti-leak</b> cuts thin "
            "connections to other structures. <b>From the selected curve</b> uses the HU range of the curve "
            "selected in the Colors tab.")))
        box = QGroupBox("New segment")
        f = QFormLayout(box)
        self.seg_mode = QComboBox()
        self.seg_mode.addItem("Automatic (statistics around the click)", "auto")
        self.seg_mode.addItem("Clicked value ± tolerance", "tol")
        self.seg_mode.addItem("Manual range (min–max)", "manual")
        self.seg_mode.addItem("From the selected curve in the editor", "curve")
        f.addRow("Range:", self.seg_mode)
        self.seg_k = self._dspin(0.5, 6, 2.5, "", None, 0.25)
        self.seg_k.setToolTip("Automatic range = mean ± k × SD of the values around the click")
        f.addRow("k × SD:", self.seg_k)
        self.seg_tol = self._dspin(1, 5000, 25, "", None)
        f.addRow("Tolerance ±:", self.seg_tol)
        self.seg_lo = self._dspin(-100000, 100000, 55, "", None)
        self.seg_hi = self._dspin(-100000, 100000, 100, "", None)
        f.addRow("Manual min:", self.seg_lo)        # one per row: side by side they were wider than the panel
        f.addRow("Manual max:", self.seg_hi)
        self.seg_leak = QSpinBox()
        self.seg_leak.setRange(0, 4)
        self.seg_leak.setValue(1)
        self.seg_leak.setSuffix(" voxel")
        f.addRow("Anti-leak:", self.seg_leak)
        self.seg_crop = QCheckBox("Limit to the crop box (when active)")
        self.seg_crop.setChecked(True)
        f.addRow(self.seg_crop)
        self.seg_smooth = QCheckBox("Smooth noise first (1 mm)")
        self.seg_smooth.setChecked(False)
        self.seg_smooth.setToolTip("Grow on a lightly smoothed copy: on noisy CT the lesion stays one solid region\n"
                                   "instead of speckles and holes. Leave off for thin structures (small vessels).\n"
                                   "Toggling re-grows the selected segment from its click at once.")
        self.seg_smooth.toggled.connect(self._on_seg_smooth_toggled)
        f.addRow(self.seg_smooth)
        self.seg_name = QLineEdit("Tumor")
        f.addRow("Name:", self.seg_name)
        self.seg_color = (0.1, 0.9, 0.2)
        self.seg_color_btn = QPushButton("Choose color")
        self.seg_color_btn.setIcon(_swatch(self.seg_color))
        self.seg_color_btn.clicked.connect(self._pick_seg_color)
        f.addRow("Color:", self.seg_color_btn)
        v.addWidget(box)

        box = QGroupBox("Segments")
        bv = QVBoxLayout(box)
        self.seg_list = QListWidget()
        self.seg_list.setMinimumHeight(110)
        self.seg_list.itemChanged.connect(self._on_seg_item_changed)
        self.seg_list.currentItemChanged.connect(self._on_seg_selected)
        bv.addWidget(self.seg_list, 1)
        row = QHBoxLayout()
        for text, ic, fn, tip in [("Empty", "brush", self.new_empty_segment, "New empty segment to paint with the brush"),
                                  ("Color", "palette", self._seg_recolor, "Color of the selected segment"),
                                  ("Delete", "trash", self._seg_delete, "Delete the selected segment"),
                                  ("STL", "save", self._seg_stl, "Export the selected segment as STL")]:
            b = QPushButton(icon(ic), text)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            row.addWidget(b)
        bv.addLayout(row)
        self.hide_vol_chk = QCheckBox("Hide volume (show segments only)")
        self.hide_vol_chk.toggled.connect(lambda h: self.vr.set_volume_visible(not h))
        bv.addWidget(self.hide_vol_chk)
        row = QHBoxLayout()
        row.addWidget(QLabel("Opacity:"))
        self.seg_opacity = self._slider(5, 100, 100, self._seg_opacity_changed)
        self.seg_opacity.setToolTip("Opacity of the selected segment")
        row.addWidget(self.seg_opacity)
        bv.addLayout(row)
        v.addWidget(box, 1)
        return w

    def _build_meas_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(8, 10, 8, 8)
        v.setSpacing(8)
        v.addWidget(hint("Choose a tool in <b>Measure ▾</b> (Length L, Angle A, Cobb C, Open polygon O, Arrow Y, "
                          "Text T), then press on the surface and drag.", muted=False))
        v.addWidget(collapsible("Editing and deleting", hint(
            "Click a measurement in 3D or in this list (it turns cyan), then press <b>Delete</b>, or right-click it "
            "in 3D. Drag its points to adjust it. Lengths switch from mm to cm at 1 cm.")))
        self.meas_list = QListWidget()
        self.meas_list.currentRowChanged.connect(self.vr.highlight_measurement3d)
        v.addWidget(self.meas_list, 1)
        row = QHBoxLayout()
        for text, ic, fn in (("Delete", "trash", self._delete_selected_meas),
                             ("Delete all", None, self.vr.clear_measurements3d),
                             ("Copy values", None, self._copy_meas)):
            b = QPushButton(icon(ic) if ic else QIcon(), text)
            b.clicked.connect(fn)
            row.addWidget(b)
        v.addLayout(row)
        return w

    def _refresh_meas(self):
        self.meas_list.blockSignals(True)
        self.meas_list.clear()
        for i, m in enumerate(self.vr.measurements3d):
            self.meas_list.addItem(f"{i + 1}.  {m.name}:  {m.label}")
        sel = self.vr._sel
        self.meas_list.setCurrentRow(self.vr.measurements3d.index(sel) if sel in self.vr.measurements3d else -1)
        self.meas_list.blockSignals(False)

    def _on_meas_selected(self, i):
        self.meas_list.blockSignals(True)
        self.meas_list.setCurrentRow(i)
        self.meas_list.blockSignals(False)

    def _copy_meas(self):
        from PySide6.QtWidgets import QApplication
        QApplication.clipboard().setText("\n".join(f"{m.name}: {m.label}" for m in self.vr.measurements3d))

    def _slider(self, lo, hi, val, slot):
        s = QSlider(Qt.Horizontal)
        s.setRange(lo, hi)
        s.setValue(val)
        s.valueChanged.connect(slot)
        return s

    def _dspin(self, lo, hi, val, suffix, slot, step=1.0):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setValue(val)
        s.setSuffix(suffix)
        s.setKeyboardTracking(False)
        if slot:
            s.valueChanged.connect(slot)
        return s

    # ============================================================ render
    def schedule_tf(self, *_):
        """Every curve / slider change: a quick update now, the full-resolution one once the drag pauses (low
        resolution while dragging, high resolution 0.4 s later)."""
        self._tf_timer.start()
        self._fine_timer.start()

    def apply_tf(self, fast=False):
        tf = self.state.tf
        tf.shading = self.shade_chk.isChecked()
        tf.ambient = self.amb_slider.value() / 100
        tf.diffuse = self.dif_slider.value() / 100
        tf.specular = self.spec_slider.value() / 100
        self.state.gain = self.gain_slider.value() / 100
        self.editor.gain = self.state.gain
        self.editor.update()
        self.vr.apply_tf(tf, gain=self.state.gain, blend=self._blend,
                         quality=self.quality_slider.value() / 100, fast=fast)
        self._update_edge_ui()
        if not fast:
            self.tfChanged.emit(self.state)     # 2D slices with the 3D colors: once the drag pauses

    def _sync_controls_from_tf(self):
        tf = self.state.tf
        for s, val in ((self.amb_slider, tf.ambient), (self.dif_slider, tf.diffuse), (self.spec_slider, tf.specular)):
            s.blockSignals(True)
            s.setValue(int(val * 100))
            s.blockSignals(False)
        self.shade_chk.blockSignals(True)
        self.shade_chk.setChecked(tf.shading)
        self.shade_chk.blockSignals(False)
        self.edge_slider.blockSignals(True)
        self.edge_slider.setValue(_edge_pos(tf.edge_threshold))
        self.edge_slider.blockSignals(False)
        self._update_edge_ui()

    def _on_edge_changed(self, pos):
        self.state.tf.edge_threshold = _edge_hu(pos)
        self.schedule_tf()

    def _update_edge_ui(self):
        """The threshold only acts on curves with hidden rims: grey it out (and say how) while there are none."""
        on = any(c.hide_edges for c in self.state.tf.curves)
        self.edge_slider.setEnabled(on)
        self.edge_lbl.setText(f"{self.state.tf.edge_threshold:.0f} HU")
        self.edge_hint.setVisible(not on)
        tip = ("Curves with hidden rims skip voxels whose local change is above this: the rim of a brighter structure\n"
               "(vessel / bone wall). Lower = stricter (only very homogeneous tissue), higher = like a normal curve.\n"
               "Most of the effect is between ~10 and 60 HU (the slider is logarithmic).")
        self.edge_slider.setToolTip(tip if on else tip + "\n\nNo curve hides its rims yet: right-click the tumor "
                                    "curve → Hide partial-volume rims.")

    def set_tf(self, tf, path=None):
        self.state.tf = tf.copy()
        self.current_preset_path = path
        self._sync_controls_from_tf()
        self.editor.set_tf(self.state.tf)
        self.editor.fit_view_to_curves()

    def _update_encoding(self):
        want = self.bone_chk.isChecked() and self._blend == "composite"
        st = self.state
        if want != st.encode_bone or st.render_data is None:
            st.encode_bone = want
            self.setCursor(Qt.WaitCursor)
            try:
                st.update_render_data()
            finally:
                self.unsetCursor()
            self.vr.refresh_data(render=False)      # every caller re-applies the transfer function right after

    def _on_bone_changed(self, *_):
        st = self.state
        st.bone_lock = self.bone_chk.isChecked()
        st.bone_lock_hu = self.bone_hu_spin.value()
        st.bone_close_mm = self.bone_close_spin.value()
        st.bone_rim_mm = self.bone_rim_spin.value()
        st.encode_bone = None   # force recompute
        self._update_encoding()
        self._show_bone_marker()
        self.apply_tf()

    def _show_bone_marker(self):
        st = self.state
        self.editor.set_bone_lock(st.bone_lock_hu if st.bone_encoded else None)

    def _on_clean_changed(self, val):
        self.state.edge_clean = val
        self.setCursor(Qt.WaitCursor)
        try:
            self.state.update_render_data()
        finally:
            self.unsetCursor()
        self.vr.refresh_data()
        self.tfChanged.emit(self.state)

    def _pick_bg(self):
        tf = self.state.tf
        c = QColorDialog.getColor(QColor(*[int(v * 255) for v in tf.background]), self, "3D background color")
        if c.isValid():
            tf.background = (c.redF(), c.greenF(), c.blueF())
            self.schedule_tf()

    def _on_crop(self, on):
        self.vr.set_crop(on)

    def _uncheck_tools(self, keep=None):
        for w in list(self.mode_acts.values()) + [self.measure_btn]:
            if w is not keep and w.isChecked():
                w.blockSignals(True)
                w.setChecked(False)
                w.blockSignals(False)

    def _set_mode(self, mode, on):
        if on:
            self._uncheck_tools(keep=self.measure_btn if mode in MEASURE3D else self.mode_acts.get(mode))
            self.vr.set_scalpel(mode)
        elif self.vr.scalpel == mode:
            self.vr.set_scalpel(None)

    def _release_tool(self):
        """After one measurement / cut: no tool active, the left mouse button rotates again."""
        self._uncheck_tools()
        self.vr.set_scalpel(None)

    def _label_measure_btn(self):
        self.measure_btn.setText(MEASURE3D[self.measure_kind][0])
        self.measure_btn.setIcon(icon(self.measure_kind))

    def select_measure(self, kind):
        self.measure_kind = kind
        self._label_measure_btn()
        if self.measure_btn.isChecked():
            self._set_mode(kind, True)
        else:
            self.measure_btn.setChecked(True)

    def _delete_selected_meas(self):
        if self.vr._sel is not None:
            self.vr.delete_measurement3d(self.vr._sel)
        else:
            self.statusBar().showMessage("Click a measurement in the 3D view (it turns cyan) or in the list first.",
                                         6000)

    def _abc2(self):
        # from the points, not the label text (labels switch between mm and cm)
        abc2_dialog(self, [float(np.linalg.norm(m.pts[1] - m.pts[0])) / 10
                           for m in self.vr.measurements3d if m.kind == "length" and len(m.pts) >= 2])

    def _after_cut(self):
        self.vr.refresh_data()
        self.statusBar().showMessage("Area cut. Ctrl+Z to undo.", 4000)

    def _undo_cut(self):
        if self.state.undo_cut():
            self.vr.refresh_data()

    def _reset_cut(self):
        self.state.reset_cut()
        self.vr.refresh_data()

    def _on_res(self, factor):
        self.state.render_factor = factor
        self.setCursor(Qt.WaitCursor)
        try:
            self.state.update_render_data()
            self.vr.refresh_data()
            self.vr.render()
        finally:
            self.unsetCursor()
        self._show_res()

    def _show_res(self):
        f = self.state._render_f
        z, y, x = self.state.render_data.shape
        self.statusBar().showMessage(f"3D resolution: {'full' if f == 1 else 'half'} ({x}×{y}×{z}, "
                                     f"{self.state.render_data.nbytes / 2 ** 20:.0f} MB)", 8000)

    def set_rotate_mode(self, mode):
        """free / roll / turn: what a left-drag does when no tool is on."""
        self.vr.set_rotate_mode(mode)
        a, label, ic = self.rot_acts[mode]
        a.blockSignals(True)
        a.setChecked(True)
        a.blockSignals(False)
        self.statusBar().showMessage({"free": "Left-drag = free 3D rotate.",
                                      "roll": "Left-drag around the centre = roll the picture (one axis).",
                                      "turn": "Left-drag left / right = turn around the head's long axis."}[mode], 6000)

    def show_snapshots(self):
        if self.snap_panel is None:
            self.snap_panel = SnapshotPanel(self)
            g = self.geometry()
            self.snap_panel.move(g.x() + 60, g.y() + 140)
        self.snap_panel.show()
        self.snap_panel.raise_()
        return self.snap_panel

    def set_blend(self, mode):
        """VR / MIP / MinIP."""
        self._blend = mode
        a, label = self.blend_acts[mode]
        a.setChecked(True)
        self.blend_btn.setText(label)
        self._update_encoding()
        self._show_bone_marker()
        self.apply_tf()             # now, not via the timer: the data may have just been re-encoded

    def _help(self):
        QMessageBox.information(self, "3D window — quick guide", (
            "MOUSE\n"
            "  Left drag = rotate • Right drag / wheel = zoom • Middle drag = pan\n"
            "  Rotate / Roll / Turn buttons: free 3D rotate, Roll (screen plane only) or Turn (head's long axis)\n"
            "  Keys 1–6 = front, back, right, left, top, bottom • + / − = zoom • 0 = fit\n\n"
            "TOOLS (one use, then the mouse rotates again)\n"
            "  Measure ▾ (L, A, C, O, Y, T): press on the surface and drag\n"
            "  Scalpel (X): draw an area, then Del = cut inside, Enter = cut outside, Esc = clear • Ctrl+Z = undo\n"
            "  Inspect (I): click the surface → HU value and the curve that colors it\n"
            "  Crop (Shift+C): drag the box faces • Box (H) = show / hide the box • Reset crop\n"
            "  Measurement captions: drag them out of the way\n"
            "  Snapshots: save views (Insert = add, PgUp / PgDn = previous / next), Save / Load, Create movie\n\n"
            "COLORS TAB\n"
            "  Each curve colors one HU range. Double-click empty space = new curve, right-click = menu.\n"
            "  Bone color lock: the bone area is colored only by 🦴 curves, so vessels next to bone stay red.\n\n"
            "SEGMENTS TAB\n"
            "  Ctrl+click a lesion in 2D to grow a segment; refine with the brush (B) in 2D.\n\n"
            "Ctrl+S = save PNG • F5 = redraw\n\n"
            "Note: a personal visualisation aid, not a certified diagnostic device."))

    def _redraw(self):
        self.vr.image.Modified()
        self.apply_tf()
        self.vr.sync_segments()
        self.vr.render()

    def showEvent(self, e):
        super().showEvent(e)
        if self.vr.image is not None:
            self.vr.render()

    def _snapshot(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save 3D image", "render3d.png", "PNG (*.png)")
        if path:
            self.vr.snapshot(path)

    def _on_inspected(self, state, vox):
        region = state.bone_region() if state.bone_encoded else None
        # look for the most opaque voxel around the pick (the pick can be 1–2 voxels off the surface)
        nz, ny, nx = state.vol.shape
        best, best_a, hu, in_bone = None, 0.0, float(state.vol.data[vox]), False
        z0, y0, x0 = vox
        for z in range(max(0, z0 - 2), min(nz, z0 + 3)):
            for y in range(max(0, y0 - 2), min(ny, y0 + 3)):
                for x in range(max(0, x0 - 2), min(nx, x0 + 3)):
                    v = float(state.vol.data[z, y, x])
                    ib = region is not None and bool(region[z, y, x])
                    for i, c in enumerate(state.tf.curves):
                        if region is not None and (c.layer == "bone") != ib:
                            continue
                        a, _ = c.sample(np.array([v]))
                        if a[0] > best_a:
                            best, best_a, hu, in_bone = i, float(a[0]), v, ib
        unit = " HU" if state.vol.modality == "CT" else ""
        where = " (bone area)" if in_bone else ""
        if best is None:
            msg = f"This point: {hu:.0f}{unit}{where} — no curve is active at this value."
        else:
            self.editor.sel_curve, self.editor.sel_point = best, None
            self.editor.update()
            msg = (f"This point: {hu:.0f}{unit}{where} → colored by curve '{state.tf.curves[best].name}' "
                   f"(opacity {best_a:.2f}). That curve is now selected in the editor.")
        self.statusBar().showMessage(msg, 15000)

    # ------------------------------------------------------------ presets
    def _load_preset_list(self, select_path=None):
        self.preset_combo.clear()
        for tf in tf_model.builtin_presets():
            self.preset_combo.addItem("★ " + tf.name, ("builtin", tf))
        for path, tf in tf_model.list_user_presets():
            self.preset_combo.addItem(tf.name, ("user", path))
        self.preset_combo.setCurrentIndex(-1)
        for i in range(self.preset_combo.count()):
            kind, val = self.preset_combo.itemData(i)
            if (select_path and val == select_path) or (not select_path and kind == "builtin"
                                                         and val.name == self.state.tf.name):
                self.preset_combo.setCurrentIndex(i)

    def _on_preset_selected(self, idx):
        kind, val = self.preset_combo.itemData(idx)
        if kind == "builtin":
            self.set_tf(val)
        else:
            try:
                self.set_tf(tf_model.TransferFunction.load(val), val)
            except Exception as ex:  # noqa: BLE001
                QMessageBox.warning(self, "Could not load preset", str(ex))

    def _save_preset(self):
        if self.current_preset_path:
            self.state.tf.save(self.current_preset_path)
            self.statusBar().showMessage(f"Preset '{self.state.tf.name}' saved.", 4000)
        else:
            self._save_preset_as()

    def _save_preset_as(self):
        name, ok = QInputDialog.getText(self, "Save preset", "Preset name:", text=self.state.tf.name)
        if not ok or not name:
            return
        self.state.tf.name = name
        fn = re.sub(r"[^\w\- ]", "_", name).strip() or "preset"
        path = os.path.join(tf_model.user_preset_dir(), fn + ".json")
        self.state.tf.save(path)
        self.current_preset_path = path
        self._load_preset_list(select_path=path)
        self.statusBar().showMessage(f"Preset saved: {path}", 5000)

    def _delete_preset(self):
        data = self.preset_combo.currentData()
        if not data or data[0] != "user":
            QMessageBox.information(self, "Built-in preset", "Built-in presets (★) cannot be deleted.")
            return
        if QMessageBox.question(self, "Delete preset", f"Delete preset '{self.preset_combo.currentText()}'?") \
                == QMessageBox.Yes:
            os.remove(data[1])
            self.current_preset_path = None
            self._load_preset_list()

    def _import_preset(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import preset", "", "Preset (*.json)")
        if path:
            tf = tf_model.TransferFunction.load(path)
            dest = os.path.join(tf_model.user_preset_dir(), os.path.basename(path))
            tf.save(dest)
            self._load_preset_list(select_path=dest)
            self.set_tf(tf, dest)

    def _export_preset(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export preset", self.state.tf.name + ".json", "Preset (*.json)")
        if path:
            self.state.tf.save(path)

    # ---------------------------------------------- from the 2D windows
    def add_eyedropper_curve(self, z, y, x):
        st = self.state
        m, s = local_stats(st.vol.data, (z, y, x))
        s = max(s, 4.0)
        lo, hi = m - 2 * s, m + 2 * s
        default = NEW_CURVE_COLORS[len(st.tf.curves) % len(NEW_CURVE_COLORS)]
        c = QColorDialog.getColor(QColor(*default), self, f"Color for tissue ≈ {m:.0f} (±{2 * s:.0f})")
        if not c.isValid():
            return
        unit = " HU" if st.vol.modality == "CT" else ""
        self.editor.add_curve_range(lo, hi, (c.redF(), c.greenF(), c.blueF()), f"Picked {m:.0f}{unit}")
        self.statusBar().showMessage(f"New curve {lo:.0f}…{hi:.0f}{unit}.", 8000)

    # ======================================================= segmentation
    def _pick_seg_color(self):
        c = QColorDialog.getColor(QColor(*[int(v * 255) for v in self.seg_color]), self, "Segment color")
        if c.isValid():
            self.seg_color = (c.redF(), c.greenF(), c.blueF())
            self.seg_color_btn.setIcon(_swatch(self.seg_color))

    def seed(self, z, y, x, busy=None, replace=None):
        """Grow a segment from a clicked voxel. busy: a BusyDialog already on screen (from the main window);
        without one a panel is shown here, so the click never just freezes the window. replace: a segment
        that the new one takes the place of (same name, colour and position in the list)."""
        own = busy is None
        if own:
            busy = BusyDialog(self, "Segmenting")
        try:
            self._seed(z, y, x, busy, replace)
        finally:
            if own:
                busy.close()

    def _on_seg_smooth_toggled(self, _):
        """The checkbox acts on the selected segment at once: re-grow it from its own click."""
        seg = self.state.active_segment
        if seg is not None and seg.seed is not None and seg in self.state.segments:
            self.seed(*seg.seed, replace=seg)

    def _seed(self, z, y, x, busy, replace=None):
        st = self.state
        val = float(st.vol.data[z, y, x])
        mode = self.seg_mode.currentData()
        leak = self.seg_leak.value()
        bounds = self.vr.crop_bounds_ijk() if self.seg_crop.isChecked() else None
        smooth = 1.0 if self.seg_smooth.isChecked() else 0.0
        grow_msg = "Growing the region from the clicked point..."
        self.setCursor(Qt.WaitCursor)
        try:
            if mode == "auto":
                mask, (lo, hi) = busy.run(grow_msg, auto_grow, st.vol.data, st.vol.spacing, (z, y, x),
                                          self.seg_k.value(), leak, bounds, smooth)
            else:
                if mode == "tol":
                    lo, hi = val - self.seg_tol.value(), val + self.seg_tol.value()
                elif mode == "manual":
                    lo, hi = self.seg_lo.value(), self.seg_hi.value()
                else:
                    ci = self.editor.sel_curve
                    if ci is None:
                        QMessageBox.information(self, "Select a curve", "First click one of the curves in the editor.")
                        return
                    pts = st.tf.curves[ci].points
                    xs = [p.x for p in pts if p.y > 0.001] or [p.x for p in pts]
                    lo, hi = min(xs), max(xs)
                if not (lo <= val <= hi):
                    busy.hide()
                    r = QMessageBox.question(
                        self, "Clicked value outside the range",
                        f"The value at the clicked point = {val:.0f}, outside the range {lo:.0f}…{hi:.0f}.\n\n"
                        "Use Automatic mode (statistics around the click) for this point?")
                    if r != QMessageBox.Yes:
                        return
                    mask, (lo, hi) = busy.run(grow_msg, auto_grow, st.vol.data, st.vol.spacing, (z, y, x),
                                              self.seg_k.value(), leak, bounds, smooth)
                else:
                    mask = busy.run(grow_msg, region_grow, st.vol.data, st.vol.spacing, (z, y, x), lo, hi, leak,
                                    bounds, 70.0, smooth)
            seg = None
            if mask is not None:
                sub, off = mask
                name, color = (replace.name, replace.color) if replace else (self.seg_name.text() or "Segment",
                                                                             self.seg_color)
                seg = busy.run("Building the 3D surface...", Segment, name, sub, off,
                               st.vol.shape, color, st.vol.spacing)
        finally:
            self.unsetCursor()
        busy.hide()
        if seg is None:
            QMessageBox.information(self, "Empty", "No voxels were segmented.")
            return
        seg.range = (lo, hi)
        seg.seed = (z, y, x)
        if replace is not None and replace in st.segments:
            seg.visible, seg.opacity = replace.visible, replace.opacity
            st.segments[st.segments.index(replace)] = seg
            st.active_segment = seg
            self._refresh_seg_list(select=seg)
            self.vr.sync_segments()
            self.segmentsChanged.emit(st)
            self.statusBar().showMessage(f"Segment '{seg.name}' re-grown: {seg.volume_ml:.2f} ml "
                                         f"({lo:.0f}…{hi:.0f})", 8000)
            return
        self._add_segment(seg)
        self.tabs.setCurrentIndex(1)
        if seg.volume_ml > 300:
            QMessageBox.warning(self, "Very large segment",
                                "The segment is very large and probably leaked into other structures.\n"
                                "Increase Anti-leak, lower k, or enable the crop box.")

    def _add_segment(self, seg):
        st = self.state
        st.segments.append(seg)
        st.active_segment = seg
        self._refresh_seg_list(select=seg)
        self.vr.sync_segments()
        self.segmentsChanged.emit(st)
        rng = f" ({seg.range[0]:.0f}…{seg.range[1]:.0f})" if seg.range else ""
        self.statusBar().showMessage(f"Segment '{seg.name}': {seg.volume_ml:.2f} ml{rng}", 8000)

    def new_empty_segment(self):
        st = self.state
        seg = Segment.empty(self.seg_name.text() or "Manual", st.vol.shape, self.seg_color, st.vol.spacing)
        self._add_segment(seg)

    def brush_done(self):
        seg = self.state.active_segment
        if seg is None:
            return
        seg.seed = None             # re-growing would throw the brush strokes away
        seg.rebuild()
        self.vr.sync_segments()
        self._refresh_seg_list(select=seg)

    def _refresh_seg_list(self, select=None):
        st = self.state
        self.seg_list.blockSignals(True)
        self.seg_list.clear()
        for seg in st.segments:
            rng = f"  ({seg.range[0]:.0f}…{seg.range[1]:.0f})" if seg.range else ""
            it = QListWidgetItem(_swatch(seg.color), f"{seg.name} — {seg.volume_ml:.2f} ml{rng}")
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if seg.visible else Qt.Unchecked)
            it.setData(Qt.UserRole, seg)
            self.seg_list.addItem(it)
            if seg is (select or st.active_segment):
                self.seg_list.setCurrentItem(it)
        self.seg_list.blockSignals(False)

    def _current_seg(self):
        it = self.seg_list.currentItem()
        return (it, it.data(Qt.UserRole)) if it else (None, None)

    def _on_seg_selected(self, cur, _):
        self.state.active_segment = cur.data(Qt.UserRole) if cur else None
        if self.state.active_segment:
            self.seg_opacity.blockSignals(True)
            self.seg_opacity.setValue(int(self.state.active_segment.opacity * 100))
            self.seg_opacity.blockSignals(False)

    def _on_seg_item_changed(self, it):
        it.data(Qt.UserRole).visible = it.checkState() == Qt.Checked
        self.vr.sync_segments()
        self.segmentsChanged.emit(self.state)

    def _seg_recolor(self):
        it, seg = self._current_seg()
        if not seg:
            return
        c = QColorDialog.getColor(QColor(*[int(v * 255) for v in seg.color]), self, "Segment color")
        if c.isValid():
            seg.color = (c.redF(), c.greenF(), c.blueF())
            it.setIcon(_swatch(seg.color))
            self.vr.sync_segments()
            self.segmentsChanged.emit(self.state)

    def _seg_delete(self):
        it, seg = self._current_seg()
        if not seg:
            return
        self.state.segments.remove(seg)
        if self.state.active_segment is seg:
            self.state.active_segment = None
        self._refresh_seg_list()
        self.vr.sync_segments()
        self.segmentsChanged.emit(self.state)

    def _seg_stl(self):
        _, seg = self._current_seg()
        if not seg:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export STL", seg.name + ".stl", "STL (*.stl)")
        if path:
            self.vr.export_segment_stl(seg, path)

    def _seg_opacity_changed(self, val):
        _, seg = self._current_seg()
        if seg:
            seg.opacity = val / 100
            self.vr.sync_segments()

    def closeEvent(self, e):
        self.vr.set_scalpel(None)
        self.vr.shutdown()
        self.state.release_3d()
        self.closed.emit(self.state)
        super().closeEvent(e)
