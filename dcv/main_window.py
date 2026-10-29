"""Main window: series thumbnails on the left, one big image, tool toolbar.

MPR and 3D VR open in separate windows (mpr_window.py, vr_window.py).
"""

from __future__ import annotations

import datetime
import html
import json
import os

import numpy as np
import pydicom
from PySide6.QtCore import QEvent, QMimeData, QObject, QPoint, QRectF, QSize, Qt, QThread, Signal
from PySide6.QtGui import (
    QAction, QActionGroup, QColor, QDrag, QFont, QIcon, QImage, QKeySequence, QPainter, QPixmap,
)
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDialog, QDialogButtonBox, QDockWidget, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGridLayout, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit, QListView, QListWidget, QListWidgetItem,
    QMainWindow, QMenu, QMessageBox, QProgressBar, QPushButton, QSizePolicy, QTableWidget, QTableWidgetItem,
    QToolBar,
    QToolButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget, QWidgetAction,
)

from . import privacy, theme
from .archive import ARCHIVE_EXT, is_archive
from .busy import BusyDialog
from .dti import fit_tensor, load_dwi
from .icons import icon
from .loader import SeriesInfo, load_series, scan_folder
from .measure import TOOL_NAMES
from .mpr_window import MPRWindow
from .segment import VolumeState
from .slice_view import COLORMAPS, FILTERS, TF_COLORMAP, SliceView
from .tools import BASIC_TOOLS, EXTRA_TOOLS, MEASURE_ORDER, ToolBank, abc2_dialog
from .tract_window import TractWindow
from .ui import cmap_icon
from .viewport import SERIES_MIME, ViewportFrame
from .vr_window import VRWindow

# data folder keeps the old internal name: the user's presets live there
APPDIR = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), "DicomColorViewer")
DEFAULT_WL = [
    ("Brain", 80, 40), ("Subdural", 250, 80), ("Stroke", 40, 40), ("Abdomen / Mediastinum", 400, 40),
    ("Liver", 150, 60), ("Lung", 1500, -600), ("Bone", 1800, 400), ("Angio", 600, 170), ("Temporal bone", 4000, 700),
]
LAYOUTS = [(1, 1), (1, 2), (2, 1), (2, 2), (1, 3), (2, 3), (3, 3), (3, 4), (4, 4), (4, 5), (5, 5)]
SLABS = [("Off", 0, "mip"), ("MIP 5 mm", 5, "mip"), ("MIP 10 mm", 10, "mip"), ("MIP 20 mm", 20, "mip"),
         ("MIP 40 mm", 40, "mip"), ("MinIP 10 mm", 10, "minip"), ("MinIP 20 mm", 20, "minip"),
         ("Average 5 mm", 5, "avg"), ("Average 10 mm", 10, "avg")]


def _fmt_date(d):
    """DICOM date 20240115 -> 15 Jan 2024 (other strings are returned unchanged)."""
    try:
        return datetime.datetime.strptime(d, "%Y%m%d").strftime("%d %b %Y")
    except (TypeError, ValueError):
        return d or ""


def _fmt_time(t):
    """DICOM time 075817.123 -> 07:58:17 (empty when missing or malformed)."""
    t = (t or "").split(".")[0]
    if len(t) >= 4 and t[:4].isdigit():
        return f"{t[:2]}:{t[2:4]}" + (f":{t[4:6]}" if len(t) >= 6 and t[4:6].isdigit() else "")
    return ""


class LoadingOverlay(QWidget):
    """Panel in the middle of the image area while DICOM files are scanned / loaded."""

    def __init__(self, host):
        super().__init__(host)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(
            "LoadingOverlay { background: rgb(30, 32, 38); border: 1px solid #3a8ee6; border-radius: 8px; }"
            " QLabel { color: #e8e8e8; background: transparent; }"
            " QProgressBar { background: #15161a; border: 1px solid #444; border-radius: 4px; color: white;"
            " text-align: center; } QProgressBar::chunk { background: #2f7fd8; border-radius: 3px; }")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 18, 24, 18)
        self.title = QLabel()
        self.title.setStyleSheet("font-size: 15px; font-weight: bold;")
        self.title.setAlignment(Qt.AlignCenter)
        self.bar = QProgressBar()
        self.bar.setTextVisible(True)
        self.bar.setFixedHeight(22)
        self.detail = QLabel()
        self.detail.setAlignment(Qt.AlignCenter)
        self.detail.setStyleSheet("color: #aab;")
        for wdg in (self.title, self.bar, self.detail):
            lay.addWidget(wdg)
        self.setFixedSize(460, 120)
        host.installEventFilter(self)
        self.hide()

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Resize:
            self._center()
        return False

    def _center(self):
        p = self.parentWidget()
        self.move((p.width() - self.width()) // 2, (p.height() - self.height()) // 2)

    def start(self, title):
        self.title.setText(title)
        self.bar.setRange(0, 0)
        self.detail.setText("Starting...")
        self._center()
        self.show()
        self.raise_()

    def update_progress(self, i, n, text):
        self.bar.setRange(0, n)
        if n:
            self.bar.setValue(i)
        self.detail.setText(text)
        self.raise_()


class Worker(QObject):
    progress = Signal(int, int, str)
    done = Signal(object, object)     # (worker, result) -> runs on the GUI thread
    failed = Signal(object, str)

    def __init__(self, fn, args, callback):
        super().__init__()
        self.fn, self.args, self.callback = fn, args, callback

    def run(self):
        try:
            res = self.fn(*self.args, progress=lambda i, n, m: self.progress.emit(i, n, m))
            self.done.emit(self, res)
        except Exception as ex:  # noqa: BLE001
            self.failed.emit(self, str(ex))


THUMB = 112          # series thumbnail size (px)
HEADER_ROLE = Qt.UserRole + 1   # "patient" / "study" on the header rows of the series list


def series_icon(info, rgb=None):
    """Thumbnail on black with a modality badge (top-left) and the image count (bottom-right), like a film tile."""
    pm = QPixmap(THUMB, THUMB)
    pm.fill(Qt.black)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setRenderHint(QPainter.SmoothPixmapTransform)
    img = None
    if rgb is not None:
        img = QImage(rgb.data, rgb.shape[1], rgb.shape[0], 3 * rgb.shape[1], QImage.Format_RGB888).copy()
    elif info.thumbnail is not None:
        t = info.thumbnail
        img = QImage(t.data, t.shape[1], t.shape[0], t.shape[1], QImage.Format_Grayscale8).copy()
    if img is not None:
        img = img.scaled(THUMB, THUMB, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        p.drawImage((THUMB - img.width()) // 2, (THUMB - img.height()) // 2, img)
    f = QFont()
    f.setPixelSize(10)
    f.setBold(True)
    p.setFont(f)
    fm = p.fontMetrics()

    def badge(text, right, bottom, bg):
        w, h = fm.horizontalAdvance(text) + 8, fm.height() + 2
        x = THUMB - w - 3 if right else 3
        y = THUMB - h - 3 if bottom else 3
        p.setPen(Qt.NoPen)
        p.setBrush(bg)
        p.drawRoundedRect(QRectF(x, y, w, h), 3, 3)
        p.setPen(QColor(theme.TEXT))
        p.drawText(QRectF(x, y, w, h), Qt.AlignCenter, text)

    if info.modality:
        badge(info.modality, False, False, QColor(theme.ACCENT_WEAK))
    if info.is_dwi or info.derived is not None:
        badge("DWI" if info.is_dwi else "MAP", True, False, QColor(90, 60, 140, 220))
    badge(str(info.count), True, True, QColor(0, 0, 0, 170))
    p.end()
    ic = QIcon(pm)
    ic.addPixmap(pm, QIcon.Selected)        # no selection tint over the image
    return ic


class WelcomePanel(QWidget):
    """Shown in the image area until a study is open: what to do first, in one place."""

    def __init__(self, host, open_folder, open_files):
        super().__init__(host)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(
            f"WelcomePanel {{ background: {theme.SURFACE}; border: 1px solid {theme.BORDER}; border-radius: 10px; }}"
            f" QLabel {{ background: transparent; color: {theme.TEXT}; }}"
            f" QPushButton {{ background: {theme.SURFACE_2}; color: {theme.TEXT}; border: 1px solid {theme.BORDER};"
            f" border-radius: 6px; padding: 9px 14px; font-size: 13px; text-align: left; }}"
            f" QPushButton:hover {{ background: {theme.HOVER}; border-color: {theme.ACCENT}; }}"
            f" QPushButton#primary {{ background: {theme.ACCENT}; border-color: {theme.ACCENT}; color: white; }}"
            f" QPushButton#primary:hover {{ background: #529BEE; }}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(32, 26, 32, 22)
        lay.setSpacing(10)
        pm = QApplication.windowIcon().pixmap(72, 72)
        if not pm.isNull():
            logo = QLabel()
            logo.setPixmap(pm)
            logo.setAlignment(Qt.AlignCenter)
            lay.addWidget(logo)
        title = QLabel("SoEasy DICOM Viewer")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 19px; font-weight: 600;")
        lay.addWidget(title)
        sub = QLabel("Open a study to start")
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet(f"color: {theme.MUTED}; font-size: 13px; margin-bottom: 6px;")
        lay.addWidget(sub)
        for text, ic, fn, primary in (("Open DICOM folder…        Ctrl+O", "open", open_folder, True),
                                      ("Open files or archive (.zip .rar .7z)…        Ctrl+Shift+O", "archive",
                                       open_files, False)):
            b = QPushButton(icon(ic), "  " + text)
            b.setIconSize(QSize(20, 20))
            b.setCursor(Qt.PointingHandCursor)
            if primary:
                b.setObjectName("primary")
            b.clicked.connect(fn)
            lay.addWidget(b)
        for text, size in (("…or drag a folder, DICOM files or an archive onto this window.", 12),
                           ("Personal visualisation aid — not a certified diagnostic device.", 11)):
            lb = QLabel(text)
            lb.setAlignment(Qt.AlignCenter)
            lb.setWordWrap(True)
            lb.setStyleSheet(f"color: {theme.MUTED}; font-size: {size}px; margin-top: 4px;")
            lay.addWidget(lb)
        self.setFixedWidth(460)
        self.adjustSize()
        host.installEventFilter(self)

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Resize:
            self._center()
        return False

    def _center(self):
        p = self.parentWidget()
        self.move((p.width() - self.width()) // 2, (p.height() - self.height()) // 2)

    def showEvent(self, e):
        self._center()
        self.raise_()
        super().showEvent(e)


class SeriesList(QListWidget):
    def __init__(self):
        super().__init__()
        self.setDragEnabled(False)          # own drag, see mouseMoveEvent
        self.setViewMode(QListView.IconMode)
        self.setFlow(QListView.TopToBottom)
        self.setWrapping(False)
        self.setMovement(QListView.Static)
        self.setResizeMode(QListView.Adjust)
        self.setIconSize(QSize(THUMB, THUMB))
        self.setSpacing(3)                  # no fixed grid: header rows are wider and lower than thumbnails
        self.setWordWrap(True)
        self.setTextElideMode(Qt.ElideRight)
        self.setStyleSheet(
            f"QListWidget {{ background: {theme.BG}; border: none; outline: none; padding-top: 4px; }}"
            f"QListWidget::item {{ color: {theme.TEXT}; border: 1px solid transparent; border-radius: 6px;"
            f" padding: 3px; }}"
            f"QListWidget::item:hover {{ background: {theme.HOVER}; }}"
            f"QListWidget::item:selected {{ background: {theme.ACCENT_WEAK}; border-color: {theme.ACCENT};"
            f" color: {theme.TEXT}; }}")

    def mimeData(self, items):
        md = QMimeData()
        if items and items[0].data(Qt.UserRole) is not None:
            md.setData(SERIES_MIME, items[0].data(Qt.UserRole).uid.encode())
        return md

    # Patient / study headers between the thumbnails, so mixed studies stay apart when scrolling.
    def add_header(self, kind, rich_text):
        it = QListWidgetItem()
        it.setFlags(Qt.NoItemFlags)                 # not selectable, not clickable, not draggable
        it.setData(HEADER_ROLE, kind)
        self.addItem(it)
        lab = QLabel(rich_text)
        lab.setWordWrap(True)
        lab.setAlignment(Qt.AlignCenter)
        bg = theme.ACCENT_WEAK if kind == "patient" else theme.SURFACE_2
        lab.setStyleSheet(f"background: {bg}; color: {theme.TEXT}; border-radius: 5px; padding: 5px 6px;"
                          f" border: 1px solid {theme.BORDER};")
        self.setItemWidget(it, lab)
        self._size_header(it)
        return it

    def _size_header(self, it):
        lab = self.itemWidget(it)
        if lab is None:     # addItem() can resize the list (scroll bar appears) before setItemWidget()
            return
        w = max(60, self.viewport().width() - 8)
        it.setSizeHint(QSize(w, lab.heightForWidth(w) + 6 if lab.hasHeightForWidth() else 48))

    def series_items(self):
        """Thumbnail items only (headers skipped)."""
        return [self.item(i) for i in range(self.count()) if self.item(i).data(Qt.UserRole) is not None]

    def resizeEvent(self, e):
        super().resizeEvent(e)
        for i in range(self.count()):
            if self.item(i).data(HEADER_ROLE):
                self._size_header(self.item(i))

    # Drag a thumbnail onto any viewport. Started here, because the built-in drag of a static icon-mode list does
    # not start reliably on Windows.
    def mousePressEvent(self, e):
        self._press = e.position().toPoint() if e.button() == Qt.LeftButton else None
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        p = getattr(self, "_press", None)
        if p is not None and e.buttons() & Qt.LeftButton and \
                (e.position().toPoint() - p).manhattanLength() >= QApplication.startDragDistance():
            self._press = None
            it = self.itemAt(p)
            if it is not None and it.data(Qt.UserRole) is not None:
                self.start_series_drag(it)
                return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if getattr(self, "_dragged", False):     # the release that ends a drag is not a click on the series
            self._dragged = False
            e.accept()
            return
        super().mouseReleaseEvent(e)

    def start_series_drag(self, it):
        drag = QDrag(self)
        drag.setMimeData(self.mimeData([it]))
        pm = it.icon().pixmap(72, 72)
        drag.setPixmap(pm)
        drag.setHotSpot(QPoint(pm.width() // 2, pm.height() // 2))
        self._dragged = True
        drag.exec(Qt.CopyAction)
        if not QApplication.mouseButtons() & Qt.LeftButton:
            self._dragged = False                # the drag loop already consumed the release
            self.setState(QListWidget.NoState)

    def paintEvent(self, e):
        super().paintEvent(e)
        if self.count() == 0:       # empty state: say how to open a study
            p = QPainter(self.viewport())
            p.setPen(QColor(theme.MUTED))
            p.drawText(self.viewport().rect().adjusted(14, 0, -14, 0), Qt.AlignCenter | Qt.TextWordWrap,
                       "No study open")
            p.end()


class GridPicker(QWidget):
    """Layout picker: hover over the grid, click."""
    picked = Signal(int, int)

    def __init__(self, n=5):
        super().__init__()
        self.n = n
        self.setMouseTracking(True)
        self.setFixedSize(n * 30 + 6, n * 30 + 26)
        self.hover = (1, 1)

    def paintEvent(self, _):
        from PySide6.QtGui import QPainter
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(45, 45, 50))
        for r in range(self.n):
            for c in range(self.n):
                on = r < self.hover[0] and c < self.hover[1]
                p.fillRect(4 + c * 30, 4 + r * 30, 27, 27, QColor(232, 180, 0) if on else QColor(95, 95, 100))
        p.setPen(Qt.white)
        p.drawText(6, self.height() - 6, f"{self.hover[0]} × {self.hover[1]}")

    def mouseMoveEvent(self, e):
        r = int(min(self.n, max(1, e.position().y() // 30 + 1)))
        c = int(min(self.n, max(1, e.position().x() // 30 + 1)))
        if (r, c) != self.hover:
            self.hover = (r, c)
            self.update()

    def mouseReleaseEvent(self, _):
        self.picked.emit(*self.hover)


class MainWindow(QMainWindow):
    windows = []        # every open main window (keeps the extra ones alive)

    def __init__(self):
        super().__init__()
        MainWindow.windows.append(self)
        self.setWindowTitle("SoEasy DICOM Viewer")
        self.resize(1600, 950)
        self.setAcceptDrops(True)
        self.setStyleSheet(theme.main_stylesheet())
        os.makedirs(APPDIR, exist_ok=True)
        self.series = {}
        self.states = {}
        self._threads = []
        self.viewports = []
        self.active = None
        self.rows, self.cols = 1, 1
        self.maximized = None
        self.link = True
        self.vr_windows = {}
        self.mpr_windows = {}
        self.tract_windows = []
        self.fusion_dlg = None
        self.wl_presets = self._load_wl_presets()

        self.center = QWidget()
        self.center.setStyleSheet("background:black")
        self.grid = QGridLayout(self.center)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(1)
        self.setCentralWidget(self.center)

        self._build_series_dock()
        self._build_toolbars()
        self._build_menus()
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(260)
        self.progress.hide()
        self.statusBar().addPermanentWidget(self.progress)
        self.mouse_lbl = QLabel()
        self.statusBar().addPermanentWidget(self.mouse_lbl)
        self.welcome = WelcomePanel(self.center, self.open_folder, self.open_files)
        self.loading = LoadingOverlay(self.center)
        self._loading_title = "Loading..."

        self.set_grid(1, 1)
        self.bank.assign("right", "zoom")
        self.bank.assign("left", "wl")

    # ================================================================ build
    def _build_series_dock(self):
        dock = QDockWidget("Series", self)
        dock.setObjectName("series")
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.study_header = QLabel()
        self.study_header.setWordWrap(True)
        self.study_header.setStyleSheet(f"background: {theme.SURFACE}; color: {theme.TEXT}; padding: 8px 10px;"
                                        f" border-bottom: 1px solid {theme.BORDER};")
        self.study_header.hide()
        lay.addWidget(self.study_header)
        self.series_list = SeriesList()
        self.series_list.itemClicked.connect(lambda it: self._open_series_in(self.active, it.data(Qt.UserRole)))
        lay.addWidget(self.series_list, 1)
        dock.setWidget(panel)
        dock.setMinimumWidth(THUMB + 46)
        dock.setMaximumWidth(260)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        self.resizeDocks([dock], [THUMB + 58], Qt.Horizontal)
        self.series_dock = dock

    def _build_actions(self):
        """Actions shared by the toolbar dropdowns and the menu bar (one object each, so one shortcut each)."""
        def act(label, text, fn, shortcut, tip, icon_name, checkable=False):
            a = self._act(text, fn, shortcut, tip)
            a.setIconText(label)            # short label under the toolbar icon; menus show `text`
            a.setIcon(icon(icon_name))
            a.setCheckable(checkable)
            self.addAction(a)               # shortcut works even when the action only sits in a dropdown
            return a

        self.act_open_folder = act("Open", "Open folder...", self.open_folder, "Ctrl+O",
                                   "Open a DICOM folder (CD/DVD/USB also work)", "open")
        self.act_open_files = act("Open files", "Open files / archive (.zip .rar .7z)...", self.open_files,
                                  "Ctrl+Shift+O", "Open DICOM files or an archive (.zip / .rar / .7z)", "archive")
        self.act_new_window = act("New window", "New window", self.new_window, "Ctrl+N",
                                  "An empty window for another study / patient", "layout")
        self.link_act = act("Sync", "Synchronise slices", self._toggle_link, "Ctrl+L",
                            "Synchronise slice position between series", "link", True)
        self.link_act.setChecked(True)
        self.privacy_act = act("Anonymous", "Hide patient identity", self._toggle_privacy, "Ctrl+Shift+H",
                               "Hide names, birth dates and study dates on screen (screenshots, teaching)", None,
                               True)
        self.privacy_act.setChecked(privacy.enabled)
        self.annot_act = act("Overlay", "Text overlay", lambda: self._all_slices(
            lambda v: (setattr(v, "show_annot", not v.show_annot), v.update())), "Ctrl+Shift+A",
            "Show / hide the text overlay (patient, W/L, zoom...)", "annot")
        self.act_rot_r = act("Right", "Rotate 90° right", lambda: self._slice_do(lambda v: v.rotate(True)),
                             "Ctrl+Right", None, "rotate_r")
        self.act_rot_l = act("Left", "Rotate 90° left", lambda: self._slice_do(lambda v: v.rotate(False)),
                             "Ctrl+Left", None, "rotate_l")
        self.act_flip_h = act("Flip H", "Flip horizontal", lambda: self._slice_do(lambda v: v.flip(True)),
                              "Ctrl+H", None, "flip_h")
        self.act_flip_v = act("Flip V", "Flip vertical", lambda: self._slice_do(lambda v: v.flip(False)),
                              "Ctrl+J", None, "flip_v")
        self.act_invert = act("Invert", "Invert", lambda: self._slice_do(
            lambda v: (setattr(v, "invert", not v.invert), v.invalidate())), "I", None, "invert")
        self.act_reset = act("Reset", "Reset viewport", self._reset_active, "Ctrl+R",
                             "Reset rotation, flip, zoom and pan of the active viewport", "reset")
        self.act_auto_wl = act("Auto", "Auto window (this slice)", self._auto_wl, "0", None, None)
        self.cine_act = act("Cine", "Cine", self._toggle_cine, "Ctrl+Space", "Cine: play slices automatically",
                            "cine", True)
        # DSA keys: / on-off, Shift+/ displayed image = mask, Ctrl+Shift+/ reset, Shift+M manual shift,
        # Ctrl+M auto shift while DSA is on (otherwise Ctrl+M stays MPR, see open_mpr)
        self.act_dsa = act("DSA", "Subtraction (DSA)", self._toggle_dsa, "/",
                           "DSA: subtract the mask frame from every frame of an XA / fluoroscopy series", "dsa", True)
        self.act_dsa_mask = act("Mask", "Use displayed image as mask", lambda: self._dsa_do(
            lambda v: v.dsa_mask_current()), "Shift+/", None, None)
        self.act_dsa_mask.setShortcuts([QKeySequence("Shift+/"), QKeySequence("?")])   # Qt sees Shift+/ as '?'
        self.act_dsa_reset = act("Reset mask", "Reset mask (default mask, no shift)", lambda: self._dsa_do(
            lambda v: v.dsa_reset()), "Ctrl+Shift+/", None, None)
        self.act_dsa_reset.setShortcuts([QKeySequence("Ctrl+Shift+/"), QKeySequence("Ctrl+?")])
        self.act_dsa_shift = act("Pixel-shift", "Manual pixel-shift (drag the mask)",
                                 lambda: self._dsa_tool("dsa_shift"), "Shift+M", None, "dsa_shift")
        self.act_dsa_auto = act("Auto shift", "Auto pixel-shift (click in the yellow square)\tCtrl+M",
                                lambda: self._dsa_tool("dsa_auto"), None, None, "dsa_auto")
        self.act_mpr = act("MPR", "MPR...", self.open_mpr, "Ctrl+M",
                           "Open the MPR window (sagittal / coronal / axial, oblique capable)", "mpr")
        self.act_3d = act("3D", "3D VR...", self.open_3d, "Ctrl+3",
                          "Open the 3D VR window (color curves, crop, scalpel, segmentation, 3D measurements)", "3d")
        self.act_fusion = act("Fusion", "Fusion (overlay series)...", self.open_fusion, "Ctrl+U",
                              "Overlay another series (e.g. FLAIR / FA / color FA on T1) on the active viewport",
                              "fusion")
        self.act_dti = act("DTI", "DTI / tractography...", self.open_dti, "Ctrl+D",
                           "Diffusion tensor: FA / MD / color FA maps and tractography from a DWI series", "dti")
        self.act_tags = act("Tags", "DICOM tags...", self._dicom_tags, "Ctrl+T", None, "tags")
        self.act_help = act("Help", "Quick guide", self._help, "F1", "Mouse buttons, tools and shortcuts", "help")

    def _menu_button(self, icon_name, label, tip, menu, default=None):
        """Toolbar button with a ▾ menu. With `default` the button itself runs it and only ▾ opens the menu."""
        b = QToolButton()
        b.setIcon(icon(icon_name))
        b.setText(label)
        b.setToolTip(tip)
        b.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        b.setIconSize(QSize(22, 22))
        b.setAutoRaise(True)
        b.setMenu(menu)
        if default:
            b.clicked.connect(default)
            b.setPopupMode(QToolButton.MenuButtonPopup)
        else:
            b.setPopupMode(QToolButton.InstantPopup)
        return b

    def _build_toolbars(self):
        """One toolbar row: icon + short label; settings live in each button's ▾ menu."""
        self._build_actions()
        tb = QToolBar("Tools")
        tb.setObjectName("tools")
        tb.setMovable(False)
        tb.setIconSize(QSize(22, 22))
        tb.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.addToolBar(tb)

        om = QMenu(self)
        om.addAction(self.act_open_folder)
        om.addAction(self.act_open_files)
        om.addSeparator()
        om.addAction(self.act_new_window)
        tb.addWidget(self._menu_button(
            "open", "Open", "Open a DICOM folder  [Ctrl+O]\n▾ files or an archive (.zip / .rar / .7z)  "
            "[Ctrl+Shift+O]\nYou can also drag folders, files or archives from Explorer onto the window.",
            om, self.open_folder))
        tb.addSeparator()
        lmenu = QMenu(self)
        picker = GridPicker()
        picker.picked.connect(lambda r, c: (self.set_grid(r, c), lmenu.close()))
        wa = QWidgetAction(lmenu)
        wa.setDefaultWidget(picker)
        lmenu.addAction(wa)
        tb.addWidget(self._menu_button("layout", "Layout", "Viewport layout (rows × columns)", lmenu))
        tb.addAction(self.link_act)
        tb.addAction(self.annot_act)
        tb.addSeparator()

        self.bank = ToolBank(self, SliceView.tools)
        self.bank.changed.connect(lambda _: (self._update_mouse_label(),      # shown permanently in the status bar
                                             self._all_slices(lambda v: v.update())))
        self.bank.populate(tb)
        mm = self.bank.measure_menu
        mm.addAction("Delete measurements on this image", lambda: self._slice_do(lambda v: v.clear_measurements(False)))
        mm.addAction("Delete all measurements of the series",
                     lambda: self._slice_do(lambda v: v.clear_measurements(True)))
        mm.addAction("ABC/2 volume calculator...", self._abc2)
        # W/L button: ▾ = presets, exact WW / WL values and drag sensitivity
        self._build_wl_editor()
        self.wl_tool_menu = QMenu(self)
        wlb = self.bank.buttons["wl"]
        wlb.setMenu(self.wl_tool_menu)
        wlb.setPopupMode(QToolButton.MenuButtonPopup)
        wlb.setToolTip("Window / Level  [W]   ▾ presets, exact WW / WL, sensitivity\n"
                       "Left-click = assign to LEFT mouse button, right-click = assign to RIGHT mouse button")
        tb.addSeparator()

        tb.addWidget(self._menu_button("image", "Image", "Color map, slab (MIP / MinIP / average) and filter",
                                       self._build_image_menu()))
        tm = QMenu(self)
        for a in (self.act_rot_r, self.act_rot_l, self.act_flip_h, self.act_flip_v, None, self.act_invert, None,
                  self.act_reset):
            tm.addAction(a) if a else tm.addSeparator()
        tb.addWidget(self._menu_button("transform", "Rotate", "Rotate, flip, invert, reset the active viewport",
                                       tm))
        tb.addAction(self.cine_act)
        dm = QMenu(self)
        onoff = dm.addAction(icon("dsa"), "Subtraction on / off\t/")   # not act_dsa itself: it owns this menu
        onoff.triggered.connect(self.act_dsa.trigger)
        for a in (None, self.act_dsa_mask, self.act_dsa_reset, None, self.act_dsa_shift,
                  self.act_dsa_auto):
            dm.addAction(a) if a else dm.addSeparator()
        self.act_dsa.setMenu(dm)
        tb.addAction(self.act_dsa)
        tb.widgetForAction(self.act_dsa).setPopupMode(QToolButton.MenuButtonPopup)   # click = on / off, ▾ = tools
        tb.addSeparator()
        for a in (self.act_mpr, self.act_3d, self.act_fusion, self.act_dti):
            tb.addAction(a)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tb.addWidget(spacer)
        tb.addAction(self.act_help)

    def _build_wl_editor(self):
        w = QWidget()
        g = QGridLayout(w)
        g.setContentsMargins(12, 8, 12, 6)
        g.setHorizontalSpacing(10)
        self.ww_spin = QDoubleSpinBox()
        self.ww_spin.setRange(1, 100000)
        self.wl_spin = QDoubleSpinBox()
        self.wl_spin.setRange(-100000, 100000)
        for sp in (self.ww_spin, self.wl_spin):
            sp.setDecimals(0)
            sp.setKeyboardTracking(False)
            sp.valueChanged.connect(self._on_wl_spin)
        self.sens = QDoubleSpinBox()
        self.sens.setRange(0.05, 10)
        self.sens.setSingleStep(0.1)
        self.sens.setValue(1.0)
        self.sens.setToolTip("Speed of W/L changes while dragging (smaller = finer)")
        self.sens.valueChanged.connect(lambda v: setattr(SliceView, "wl_sensitivity", v))
        for r, (text, sp) in enumerate((("Width (WW)", self.ww_spin), ("Level (WL)", self.wl_spin),
                                        ("Drag sensitivity", self.sens))):
            g.addWidget(QLabel(text), r, 0)
            g.addWidget(sp, r, 1)
        self.wl_editor = QWidgetAction(self)
        self.wl_editor.setDefaultWidget(w)

    def _build_image_menu(self):
        m = QMenu(self)
        self.cmap_actions, self.filter_actions, self.slab_actions = {}, {}, []

        def group(title, items, on):
            sub = m.addMenu(title)
            g = QActionGroup(self)
            out = []
            for label, data in items:
                a = sub.addAction(label)
                a.setCheckable(True)
                g.addAction(a)
                a.triggered.connect(lambda _=False, d=data: on(d))
                out.append((a, data))
            return sub, out

        self.cmap_menu, acts = group("Color map", [(n, n) for n in list(COLORMAPS) + [TF_COLORMAP]], self._set_cmap)
        self.cmap_actions = {d: a for a, d in acts}
        for name, a in self.cmap_actions.items():
            if name in COLORMAPS:
                a.setIcon(cmap_icon(COLORMAPS[name]))
        self.slab_menu, self.slab_actions = group("Slab (MIP / MinIP / average)",
                                                  [(label, (mm, mode)) for label, mm, mode in SLABS], self._set_slab)
        self.filter_menu, acts = group("Filter", [(label, k) for k, (label, _) in FILTERS.items()], self._set_filter)
        self.filter_actions = {d: a for a, d in acts}
        return m

    def _act(self, text, fn, shortcut=None, tip=None):
        a = QAction(text, self)
        a.triggered.connect(fn)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        if tip:
            a.setToolTip(f"{tip}  [{shortcut}]" if shortcut else tip)
        return a

    def _build_menus(self):
        mb = self.menuBar()
        m = mb.addMenu("&File")
        m.addAction(self.act_open_folder)
        m.addAction(self.act_open_files)
        m.addAction(self.act_new_window)
        m.addSeparator()
        m.addAction(self._act("Save viewport image...", self._save_image, "Ctrl+S"))
        m.addAction(self._act("Copy viewport image", self._copy_image, "Ctrl+Shift+C"))
        m.addAction(self.act_tags)
        m.addSeparator()
        m.addAction(self._act("Exit", self.close))

        m = mb.addMenu("&View")
        lm = m.addMenu("Layout")
        for r, c in LAYOUTS:
            lm.addAction(f"{r} × {c}", lambda rr=r, cc=c: self.set_grid(rr, cc))
        om = m.addMenu("Active viewport orientation (reconstruction)")
        for o, label in (("axial", "Axial"), ("coronal", "Coronal"), ("sagittal", "Sagittal")):
            om.addAction(label, lambda oo=o: self._slice_do(lambda v: v.set_orientation(oo)))
        m.addAction(self._act("Maximise active viewport", lambda: self._toggle_maximize(self.active), "F"))
        m.addAction(self._act("Full screen", self._toggle_fullscreen, "F11"))
        m.addSeparator()
        m.addAction(self._act("Crosshair on/off", lambda: self._all_slices(
            lambda v: (setattr(v, "show_cross", not v.show_cross), v.update())), "X"))
        m.addAction(self.link_act)
        m.addAction(self.annot_act)
        m.addAction(self.privacy_act)
        m.addAction(self.series_dock.toggleViewAction())
        m.addSeparator()
        for a in (self.act_mpr, self.act_3d, self.act_fusion, self.act_dti):
            m.addAction(a)

        m = mb.addMenu("&Image")
        for a in (self.act_rot_r, self.act_rot_l, self.act_flip_h, self.act_flip_v, None, self.act_invert,
                  self.act_reset, None):
            m.addAction(a) if a else m.addSeparator()
        for sub in (self.cmap_menu, self.slab_menu, self.filter_menu):
            m.addMenu(sub)

        m = mb.addMenu("&Measure")
        for k in MEASURE_ORDER:
            name, sc = TOOL_NAMES[k]
            m.addAction(icon(k), f"{name}\t{sc}", lambda kk=k: self.bank.select_measure(kk))
        m.addSeparator()
        for key, _, tip, _ in BASIC_TOOLS:
            m.addAction(icon(key), tip, lambda kk=key: self.bank.assign("left", kk))
        for key, (_, tip, _) in EXTRA_TOOLS.items():
            m.addAction(icon(key), tip, lambda kk=key: self.bank.assign("left", kk))
        m.addSeparator()
        m.addAction(self._act("ABC/2 volume calculator...", self._abc2))

        self.wl_menu = mb.addMenu("&Window")
        m = mb.addMenu("&Tools")
        m.addAction(self.cine_act)
        m.addMenu(self.act_dsa.menu()).setText("DSA (subtraction)")
        m.addAction(self.act_tags)
        m.addAction(self._act("ABC/2 volume calculator...", self._abc2))
        m = mb.addMenu("&Help")
        m.addAction(self.act_help)
        self._refresh_wl_ui()

    def _update_mouse_label(self):
        if not hasattr(self, "mouse_lbl"):
            return
        self.mouse_lbl.setText(self.bank.mouse_html())

    # ============================================================ viewports
    def _new_viewport(self):
        vp = ViewportFrame()
        vp.activated.connect(self._set_active)
        vp.dropped.connect(lambda f, uid: self._open_series_in(f, self.series.get(uid)))
        vp.maximizeRequested.connect(self._toggle_maximize)
        s = vp.slice
        s.windowChanged.connect(lambda ww, wl, sv=s: self._on_window_changed(sv, ww, wl))
        s.crossChanged.connect(self._on_cross)
        s.indexChanged.connect(self._on_index_changed)
        s.eyedropper.connect(self._on_eyedropper)
        s.seedPicked.connect(self._on_seed)
        s.requestSegment.connect(lambda st: self.vr_window(st).new_empty_segment())
        s.brushDone.connect(lambda st: self.vr_window(st).brush_done())
        s.annotChanged.connect(lambda st: self._all_slices(lambda v: v.update()))
        s.status.connect(lambda m: self.statusBar().showMessage(m, 3000))
        s.cineChanged.connect(self._on_cine_changed)
        s.seriesStep.connect(self._series_step)
        self.viewports.append(vp)
        return vp

    def set_grid(self, rows, cols):
        self.maximized = None
        n = rows * cols
        while len(self.viewports) < n:
            self._new_viewport()
        for vp in self.viewports:
            self.grid.removeWidget(vp)
            vp.hide()
        for i in range(5):
            self.grid.setRowStretch(i, 0)
            self.grid.setColumnStretch(i, 0)
        loaded = [vp.state for vp in self.viewports[:n] if vp.state is not None]
        for i in range(n):
            vp = self.viewports[i]
            self.grid.addWidget(vp, i // cols, i % cols)
            self.grid.setRowStretch(i // cols, 1)
            self.grid.setColumnStretch(i % cols, 1)
            vp.show()
        self.rows, self.cols = rows, cols
        if self.active not in self.viewports[:n]:
            self._set_active(self.viewports[0])
        if loaded:
            self._fill_empty_viewports()

    def _fill_empty_viewports(self):
        """Empty viewports get the next series."""
        used = {vp.state.vol.uid for vp in self.visible_viewports() if vp.state is not None}
        queue = [s for s in self.series.values() if s.uid not in used]
        for vp in self.visible_viewports():
            if vp.state is None and queue:
                info = queue.pop(0)
                self._load(info, lambda st, f=vp: (f.set_state(st), self._update_tf_tables(st)))

    def visible_viewports(self):
        return [vp for vp in self.viewports[: self.rows * self.cols] if vp.isVisible()]

    def _toggle_maximize(self, vp):
        if vp is None:
            return
        if self.maximized is vp:
            self.set_grid(self.rows, self.cols)
            return
        for other in self.viewports:
            other.setVisible(other is vp)
        self.maximized = vp
        self._set_active(vp)

    def _set_active(self, vp):
        if vp is self.active:
            return
        if self.active:
            self.active.set_active(False)
        self.active = vp
        vp.set_active(True)
        s = vp.slice
        self._sync_wl_widgets(s.ww, s.wl)
        self._sync_image_menu(s)
        self.act_dsa.setChecked(s.dsa is not None)
        self.cine_act.setChecked(s.cine_playing)
        if vp.state is not None:
            self._select_series_item(vp.state.vol.uid)

    def _select_series_item(self, uid):
        for it in self.series_list.series_items():
            if it.data(Qt.UserRole).uid == uid:
                self.series_list.blockSignals(True)
                self.series_list.setCurrentItem(it)
                self.series_list.blockSignals(False)

    def _active_slice(self):
        return self.active.slice if self.active and self.active.state else None

    def _slice_do(self, fn):
        s = self._active_slice()
        if s:
            fn(s)

    def _all_slices(self, fn, state=None):
        for vp in self.viewports:
            if vp.slice.state is not None and (state is None or vp.slice.state is state):
                fn(vp.slice)

    def _reset_active(self):
        s = self._active_slice()
        if s:
            s.rot, s.flip_h, s.flip_v, s.invert = 0, False, False, False
            s.reset_view()

    def _toggle_fullscreen(self):
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    # =================================================== 3D / MPR windows
    def _active_state(self):
        return self.active.state if self.active else None

    def vr_window(self, state, show=True, busy=None):
        w = self.vr_windows.get(id(state))
        if w is None:
            own = busy is None
            if own:
                busy = BusyDialog(self, "Opening 3D")
            try:
                def prepare():                    # the slow part of opening 3D, off the GUI thread
                    state.encode_bone = state.bone_lock
                    if state.encode_bone:
                        state.bone_region()
                    state.update_render_data()
                busy.run("Preparing the 3D volume (bone area, render copy)...", prepare)
                busy.stage("Uploading the volume to the graphics card...")
                w = VRWindow(state)
            finally:
                if own:
                    busy.close()
            w.tfChanged.connect(self._on_tf_changed)
            w.segmentsChanged.connect(lambda st: self._all_slices(lambda v: v.invalidate(), st))
            w.closed.connect(self._on_vr_closed)
            self.vr_windows[id(state)] = w
        if show and not w.isVisible():
            w.show()
        return w

    def closeEvent(self, e):
        if self in MainWindow.windows:
            MainWindow.windows.remove(self)
        # close the OpenGL windows first so their contexts are released cleanly
        for w in list(self.vr_windows.values()):
            w.close()
        for tw in self.tract_windows:
            tw.rw.Finalize()
            tw.close()
        for w in list(self.mpr_windows.values()):
            w.close()
        super().closeEvent(e)

    def _on_vr_closed(self, state):
        w = self.vr_windows.pop(id(state), None)
        if w is not None:
            w.deleteLater()
        self._trim_memory()

    def _states_in_use(self):
        used = {id(vp.state) for vp in self.viewports if vp.state is not None}
        used |= {id(w.state) for w in self.vr_windows.values()}
        used |= {id(w.model.state) for w in self.mpr_windows.values() if w.isVisible()}
        for vp in self.viewports:
            if vp.slice.fusion is not None:
                used.add(id(vp.slice.fusion["state"]))
        if self.fusion_dlg is not None and self.fusion_dlg.ov_state is not None:
            used.add(id(self.fusion_dlg.ov_state))
        return used

    def _trim_memory(self):
        """Drop series that are not shown anywhere (they reload from disk on demand). Series with segments,
        measurements or cuts and derived series (FA, MD...) are kept."""
        import gc
        used = self._states_in_use()
        for uid, st in list(self.states.items()):
            if id(st) in used or st.has_user_work() or self.series.get(uid) is not None and \
                    self.series[uid].derived is not None:
                continue
            del self.states[uid]
            w = self.mpr_windows.pop(id(st), None)
            if w is not None:
                w.deleteLater()
        for st in self.states.values():
            if id(st) not in {id(w.state) for w in self.vr_windows.values()} and st.render_data is not None:
                st.release_3d()
        gc.collect()

    def open_3d(self):
        st = self._active_state()
        if st is None:
            return
        self.setCursor(Qt.WaitCursor)
        try:
            w = self.vr_window(st)
        finally:
            self.unsetCursor()
        w.raise_()
        w.activateWindow()

    def open_mpr(self):
        s = self._active_slice()
        if s is not None and s.dsa is not None:     # Ctrl+M in DSA mode = auto pixel-shift
            self._dsa_tool("dsa_auto")
            return
        st = self._active_state()
        if st is None:
            return
        w = self.mpr_windows.get(id(st))
        if w is None:
            w = MPRWindow(st, self.wl_presets)
            w.seedPicked.connect(self._on_seed)
            w.eyedropper.connect(self._on_eyedropper)
            s = self._active_slice()
            if s:
                w.set_window(s.ww, s.wl)
            self.mpr_windows[id(st)] = w
        w.show()
        w.raise_()
        w.activateWindow()

    def _on_tf_changed(self, state):
        self._update_tf_tables(state)
        w = self.mpr_windows.get(id(state))
        if w is not None:
            w.update_tf_table()

    def _update_tf_tables(self, state):
        lo, hi = state.vol.value_range
        if state.bone_lock and state.vol.modality == "CT":
            # two tables: outside the bone area (normal curves) and inside it (🦴 curves)
            _, a, rgb = state.tf.sample(lo, hi, 2048, gain=state.gain, layer="normal")
            _, ab, rgbb = state.tf.sample(lo, hi, 2048, gain=state.gain, layer="bone")
            table = (lo, hi, (rgb * 255).astype(np.uint8), a, (rgbb * 255).astype(np.uint8), ab)
        else:
            _, a, rgb = state.tf.sample(lo, hi, 2048, gain=state.gain)
            table = (lo, hi, (rgb * 255).astype(np.uint8), a)
        self._all_slices(lambda v: v.set_tf_table(table), state)

    def _on_seed(self, state, z, y, x):
        busy = BusyDialog(self, "Segmenting")
        try:
            w = self.vr_window(state, busy=busy)
            w.seed(z, y, x, busy=busy)
        finally:
            busy.close()

    def _on_eyedropper(self, state, z, y, x):
        self.vr_window(state).add_eyedropper_curve(z, y, x)

    # ============================================================= loading
    def _run(self, fn, args, on_done):
        # connect only to MainWindow methods so the callbacks run on the GUI thread
        th = QThread(self)
        wk = Worker(fn, args, on_done)
        wk.moveToThread(th)
        th.started.connect(wk.run)
        wk.progress.connect(self._on_progress)
        wk.done.connect(self._on_worker_done)
        wk.failed.connect(self._on_worker_failed)
        self._threads.append((th, wk))
        self.progress.setMaximum(0)
        self.progress.show()
        self.loading.start(self._loading_title)
        th.start()

    def _finish_worker(self, wk):
        for pair in list(self._threads):
            if pair[1] is wk:
                pair[0].quit()
                pair[0].wait()
                self._threads.remove(pair)
        if not self._threads:
            self.progress.hide()
            self.loading.hide()

    def _on_worker_done(self, wk, res):
        self._finish_worker(wk)
        wk.callback(res)

    def _on_worker_failed(self, wk, msg):
        self._finish_worker(wk)
        QMessageBox.warning(self, "Failed", msg)
        self.welcome.setVisible(not self.series)

    def _on_progress(self, i, n, msg):
        self.progress.setMaximum(n)          # n == 0 -> busy (moving) bar
        self.progress.setValue(i)
        text = f"{msg} {i}/{n}" if n else msg
        self.statusBar().showMessage(text)
        self.loading.update_progress(i, n, text)

    def open_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Choose a DICOM folder (CD/DVD/USB also work)")
        if d:
            self.open_paths([d])

    def open_files(self):
        arc = " ".join(f"*{e}" for e in ARCHIVE_EXT)
        files, _ = QFileDialog.getOpenFileNames(
            self, "Choose DICOM files or an archive", "",
            f"DICOM files and archives (*);;Archives ({arc});;DICOMDIR (DICOMDIR)")
        if files:
            self.open_paths(files)

    # files / folders / archives dragged from Explorer
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls() and any(u.isLocalFile() for u in e.mimeData().urls()):
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if paths:
            e.acceptProposedAction()
            self.open_paths(paths)

    def open_paths(self, paths):
        self.statusBar().showMessage("Scanning...")
        self._loading_title = ("Opening archive" if any(is_archive(p) for p in paths) else "Reading DICOM folder")
        self.welcome.hide()
        self._run(scan_folder, (paths,), self._on_scanned)

    def _on_scanned(self, series):
        if not series:
            QMessageBox.information(self, "Nothing found", "No DICOM images were found at that location.")
            self.welcome.setVisible(not self.series)
            return
        new = [s for s in series if s.uid not in self.series]
        had_study = bool(self.series)
        if had_study and new:
            where = self._ask_open_target(new)
            if where is None:
                return
            if where == "new":
                self.new_window()._on_scanned(series)
                return
        for s in new:
            self.series[s.uid] = s
        self._rebuild_series_list()
        self.welcome.hide()         # also in a new window, which gets its series here without open_paths()
        self._update_study_header()
        self._update_title()
        if had_study:       # added to this window: keep what is on screen, the new series wait in the list
            self.statusBar().showMessage(f"{len(new)} series added. Drag one onto a viewport, or use Fusion to "
                                         "overlay it.", 10000)
            return
        self.statusBar().showMessage(f"{len(series)} series. Click a series to open it, or drag it to a viewport.")
        best = max(series, key=lambda s: s.count)
        target = self.active if self.active.state is None else self.viewports[0]
        self._open_series_in(target, best)

    def _update_title(self):
        patients = {s.patient for s in self.series.values() if s.derived is None}
        first = next((s for s in self.series.values() if s.derived is None), None)
        if first is None:
            self.setWindowTitle("SoEasy DICOM Viewer")
            return
        who = "Several patients" if len(patients) > 1 else " - ".join(
            x for x in (privacy.name(first.patient), privacy.date(first.study_date)) if x)
        self.setWindowTitle(f"{who} - SoEasy DICOM Viewer")

    def _toggle_privacy(self):
        """Hide / show the patient identity in every window (screenshots, teaching)."""
        privacy.enabled = self.privacy_act.isChecked()
        for w in MainWindow.windows:
            w.privacy_act.setChecked(privacy.enabled)
            w._update_title()
            w._rebuild_series_list()
            w._update_study_header()
            w._all_slices(lambda v: v.update())
            for sub in list(w.vr_windows.values()) + list(w.mpr_windows.values()) + list(w.tract_windows):
                sub.update_title()
        self.statusBar().showMessage("Patient identity hidden on screen." if privacy.enabled
                                     else "Patient identity shown.", 5000)

    def _ask_open_target(self, new):
        """This window already shows a study: open the new one in a new window (default; always for
        another patient) or add it here (e.g. MRI + CT of the same patient for Fusion). Returns 'new' / 'here' / None."""
        def names(infos):
            return sorted({(s.patient or "(no name)").replace("^", " ").strip() for s in infos})

        cur, nxt = names(self.series.values()), names(new)
        same = cur == nxt
        box = QMessageBox(self)
        box.setWindowTitle("Open study")
        box.setIcon(QMessageBox.Question if same else QMessageBox.Warning)
        if same:
            box.setText(f"<b>{html.escape(', '.join(nxt))}</b> is already open in this window.")
            box.setInformativeText(f"Add the {len(new)} new series here (e.g. to fuse MRI and CT), or open them in "
                                   "a new window?")
        else:
            box.setText("<b>Different patient.</b>")
            box.setInformativeText(f"This window shows <b>{html.escape(', '.join(cur))}</b>.<br>The new study is "
                                   f"<b>{html.escape(', '.join(nxt))}</b>.<br><br>A separate window keeps the two "
                                   "patients apart.")
        new_btn = box.addButton("Open in new window", QMessageBox.AcceptRole)
        here_btn = box.addButton("Add to this window", QMessageBox.ActionRole)
        box.addButton(QMessageBox.Cancel)
        box.setDefaultButton(here_btn if same else new_btn)
        box.exec()
        return {new_btn: "new", here_btn: "here"}.get(box.clickedButton())

    def new_window(self):
        """Another main window (own series, viewports, 3D / MPR windows), slightly offset from this one."""
        w = MainWindow()
        w.resize(self.size())
        w.move(self.pos() + QPoint(30, 30))
        w.show()
        return w

    @staticmethod
    def _patient_key(s):
        return ((s.patient_id or s.patient or "").strip(), s.birth_date)

    @staticmethod
    def _study_key(s):
        return s.study_uid or f"{s.study_date}|{s.study_desc}"

    def _rebuild_series_list(self):
        """Thumbnails grouped: a patient header (name, birth date) whenever the patient changes and a
        study header (date and time, description, modalities, number of series) whenever the study changes.
        Groups keep the order in which they were opened; inside a study, series are sorted by series number."""
        lst = self.series_list
        cur = lst.currentItem()
        cur_uid = cur.data(Qt.UserRole).uid if cur is not None and cur.data(Qt.UserRole) is not None else None
        infos = list(self.series.values())
        pat, study = {}, {}
        for s in infos:
            pat.setdefault(self._patient_key(s), len(pat))
            study.setdefault((self._patient_key(s), self._study_key(s)), len(study))
        infos.sort(key=lambda s: (pat[self._patient_key(s)], study[(self._patient_key(s), self._study_key(s))],
                                  s.derived is not None, s.series_number))
        lst.blockSignals(True)
        try:
            self._fill_series_list(infos)
        finally:
            lst.blockSignals(False)
        if cur_uid:
            self._select_series_item(cur_uid)

    def _fill_series_list(self, infos):
        lst = self.series_list
        lst.clear()
        last_p = last_s = None
        for s in infos:
            pk, sk = self._patient_key(s), self._study_key(s)
            if pk != last_p:
                name = privacy.name(s.patient)
                born = privacy.date(_fmt_date(s.birth_date))
                lst.add_header("patient", f"<b style='font-size:12px'>{html.escape(name.upper())}</b>"
                                          + (f"<br>{html.escape(born)}" if born else ""))
                last_p, last_s = pk, None
            if sk != last_s:
                members = [x for x in infos if self._patient_key(x) == pk and self._study_key(x) == sk]
                mods = "/".join(sorted({x.modality for x in members if x.modality})) or "?"
                when = privacy.date(" ".join(x for x in (_fmt_date(s.study_date), _fmt_time(s.study_time)) if x))
                desc = s.study_desc or ("Derived maps" if s.derived is not None else "")
                lines = [f"<b>{html.escape(when or ('Study' if privacy.enabled else 'Unknown date'))}</b>"]
                if desc:
                    lines.append(html.escape(desc))
                lines.append(f"<span style='color:{theme.MUTED}'>{html.escape(mods)}: {len(members)} series</span>")
                lst.add_header("study", "<br>".join(lines))
                last_s = sk
            self._add_series_item(s, getattr(s, "_thumb_rgb", None))

    def _add_series_item(self, s, rgb=None):
        s._thumb_rgb = rgb
        text = s.description or "(no description)"
        if s.series_number:
            text += f"\n#{s.series_number}"
        it = QListWidgetItem(text)
        it.setTextAlignment(Qt.AlignHCenter | Qt.AlignTop)
        it.setIcon(series_icon(s, rgb))
        it.setData(Qt.UserRole, s)
        who = "\n".join(x for x in (privacy.name(s.patient), privacy.date(_fmt_date(s.study_date))) if x)
        it.setToolTip(f"{who}\n{s.modality} #{s.series_number} {s.description}\n"
                      f"{s.count} images\n\nClick = show in the active viewport\n"
                      "Drag onto a viewport = show it there")
        self.series_list.addItem(it)

    def _update_study_header(self):
        """Patient, date, modalities and number of series above the thumbnails. Only used when no list headers
        exist; the grouped list (_rebuild_series_list) shows the same per patient / study."""
        infos = list(self.series.values())
        if not infos or self.series_list.count() > len(self.series_list.series_items()):
            self.study_header.hide()
            return
        patients = sorted({i.patient for i in infos if i.patient})
        mods = sorted({i.modality for i in infos if i.modality})
        dates = sorted({i.study_date for i in infos if i.study_date})
        name = privacy.name(patients[0]) if len(patients) == 1 else f"{len(patients)} patients"
        date = privacy.date(_fmt_date(dates[0])) if len(dates) == 1 else (f"{len(dates)} studies" if dates else "")
        details = " · ".join(x for x in (date, "/".join(mods), f"{len(infos)} series") if x)
        self.study_header.setText(f"<span style='font-size:13px; font-weight:600'>{html.escape(name)}</span><br>"
                                  f"<span style='color:{theme.MUTED}'>{html.escape(details)}</span>")
        self.study_header.show()

    def _open_series_in(self, vp, info):
        if info is None or vp is None:
            return
        orient = vp.slice.orientation if vp.state is not None else "axial"

        def done(st):
            vp.set_state(st, orient)
            self._update_tf_tables(st)
            self._set_active(vp)
            self.act_dsa.setChecked(vp.slice.dsa is not None)       # a new series starts unsubtracted
            if vp.slice.is_movie() and vp.slice.count() > 1:
                vp.slice.set_cine(True)                             # XA / US runs play on opening
            self._sync_wl_widgets(vp.slice.ww, vp.slice.wl)
            self._select_series_item(st.vol.uid)
        self._load(info, done)

    def _load(self, info, then):
        if info.uid in self.states:
            then(self.states[info.uid])
            return
        self.statusBar().showMessage(f"Loading {info.description}...")

        def done(vol):
            st = VolumeState(vol)
            self.states[info.uid] = st
            then(st)
            self._trim_memory()
            lo, hi = vol.value_range
            z, y, x = vol.shape
            self.statusBar().showMessage(
                f"{privacy.name(vol.patient)} | {vol.description} | {x}×{y}×{z} | spacing {vol.spacing[2]:.2f}×"
                f"{vol.spacing[1]:.2f}×{vol.spacing[0]:.2f} mm | values {lo:.0f}…{hi:.0f}")

        self._loading_title = f"Opening {info.description or 'series'} ({info.count} images)"
        self._run(load_series, (info,), done)

    # =============================================================== window
    def _on_window_changed(self, src, ww, wl):
        # same series elsewhere follows, unless one shows it subtracted (DSA values are live − mask)
        self._all_slices(lambda v: v.set_window(ww, wl) if v is not src and (v.dsa is None) == (src.dsa is None)
                         else None, src.state)
        if self.active and self.active.slice is src:
            self._sync_wl_widgets(ww, wl)

    def _sync_wl_widgets(self, ww, wl):
        for s, val in ((self.ww_spin, ww), (self.wl_spin, wl)):
            s.blockSignals(True)
            s.setValue(val)
            s.blockSignals(False)

    def _on_wl_spin(self):
        s = self._active_slice()
        if s:
            self._all_slices(lambda v: v.set_window(self.ww_spin.value(), self.wl_spin.value()), s.state)

    def _apply_wl(self, ww, wl):
        s = self._active_slice()
        if not s:
            return
        self._all_slices(lambda v: v.set_window(ww, wl), s.state)
        self._sync_wl_widgets(ww, wl)

    def _apply_wl_choice(self, choice):
        s = self._active_slice()
        if not s:
            return
        if choice == "default":
            v = s.vol
            if v.default_ww:
                self._apply_wl(v.default_ww, v.default_wl)
            else:
                s.auto_window()
        elif choice == "full":
            lo, hi = s.vol.value_range
            self._apply_wl(hi - lo, (hi + lo) / 2)

    def _auto_wl(self):
        s = self._active_slice()
        if s:
            s.auto_window()

    def _load_wl_presets(self):
        path = os.path.join(APPDIR, "wl_presets.json")
        try:
            with open(path, encoding="utf-8") as f:
                return [tuple(x) for x in json.load(f)]
        except Exception:
            return list(DEFAULT_WL)

    def _save_wl_presets(self):
        with open(os.path.join(APPDIR, "wl_presets.json"), "w", encoding="utf-8") as f:
            json.dump(self.wl_presets, f, indent=1)

    def _refresh_wl_ui(self):
        """Fill the W/L button's ▾ menu and the Window menu with the presets (keys 1–9 = first nine)."""
        if not hasattr(self, "wl_menu") or not hasattr(self, "wl_tool_menu"):
            return
        self.wl_tool_menu.clear()
        self.wl_menu.clear()
        for a in getattr(self, "_wl_actions", []):
            a.deleteLater()
        presets = [self._act(f"{name}  ({ww:g} / {wl:g})", lambda _=False, a=ww, b=wl: self._apply_wl(a, b),
                             str(i + 1) if i < 9 else None) for i, (name, ww, wl) in enumerate(self.wl_presets)]
        default = self._act("DICOM default", lambda: self._apply_wl_choice("default"))
        full = self._act("Full range", lambda: self._apply_wl_choice("full"))
        add = self._act("Add preset from current WW/WL...", self._add_wl_preset)
        manage = self._act("Manage window presets...", self._manage_wl)
        self._wl_actions = presets + [default, full, add, manage]
        for menu in (self.wl_tool_menu, self.wl_menu):
            if menu is self.wl_tool_menu:
                menu.addAction(self.wl_editor)
                menu.addSeparator()
            menu.addAction(default)
            for a in presets:
                menu.addAction(a)
            menu.addAction(full)
            menu.addSeparator()
            menu.addAction(self.act_auto_wl)
            menu.addAction(add)
            menu.addAction(manage)

    def _add_wl_preset(self):
        name, ok = QInputDialog.getText(self, "New window preset",
                                        f"Name (WW {self.ww_spin.value():.0f} / WL {self.wl_spin.value():.0f}):")
        if ok and name:
            self.wl_presets.append((name, self.ww_spin.value(), self.wl_spin.value()))
            self._save_wl_presets()
            self._refresh_wl_ui()

    def _manage_wl(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Manage window presets (keys 1–9 = first 9 rows)")
        dlg.resize(460, 420)
        v = QVBoxLayout(dlg)
        tbl = QTableWidget(len(self.wl_presets), 3)
        tbl.setHorizontalHeaderLabels(["Name", "WW", "WL"])
        tbl.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for r, (n, ww, wl) in enumerate(self.wl_presets):
            for c, val in enumerate((n, f"{ww:g}", f"{wl:g}")):
                tbl.setItem(r, c, QTableWidgetItem(val))
        v.addWidget(tbl)
        row = QHBoxLayout()

        def add():
            r = tbl.rowCount()
            tbl.insertRow(r)
            for c, val in enumerate(("New", f"{self.ww_spin.value():g}", f"{self.wl_spin.value():g}")):
                tbl.setItem(r, c, QTableWidgetItem(val))

        def move(d):
            r = tbl.currentRow()
            t = r + d
            if r < 0 or not (0 <= t < tbl.rowCount()):
                return
            for c in range(3):
                a, b = tbl.takeItem(r, c), tbl.takeItem(t, c)
                tbl.setItem(r, c, b)
                tbl.setItem(t, c, a)
            tbl.setCurrentCell(t, 0)

        for text, fn in [("+ From current WW/WL", add), ("Delete", lambda: tbl.removeRow(tbl.currentRow())),
                         ("▲", lambda: move(-1)), ("▼", lambda: move(1))]:
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        v.addLayout(row)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.RestoreDefaults)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        bb.button(QDialogButtonBox.RestoreDefaults).clicked.connect(
            lambda: (setattr(self, "wl_presets", list(DEFAULT_WL)), dlg.reject()))
        v.addWidget(bb)
        if dlg.exec() == QDialog.Accepted:
            out = []
            for r in range(tbl.rowCount()):
                try:
                    out.append((tbl.item(r, 0).text(), float(tbl.item(r, 1).text()), float(tbl.item(r, 2).text())))
                except (AttributeError, ValueError):
                    pass
            self.wl_presets = out
        self._save_wl_presets()
        self._refresh_wl_ui()

    def _set_cmap(self, name):
        s = self._active_slice()
        if s:
            self._update_tf_tables(s.state)
        self._all_slices(lambda v: v.set_colormap(name), s.state if s else None)

    def _set_slab(self, data):
        mm, mode = data
        self._slice_do(lambda v: (setattr(v, "slab_mm", mm), setattr(v, "slab_mode", mode), v.invalidate()))

    def _set_filter(self, k):
        self._slice_do(lambda v: (setattr(v, "filter", k), v.invalidate()))

    def _sync_image_menu(self, s):
        """Tick the color map / slab / filter of the active viewport in the Image menu."""
        a = self.cmap_actions.get(s.colormap)
        if a:
            a.setChecked(True)
        a = self.filter_actions.get(s.filter)
        if a:
            a.setChecked(True)
        for a, (mm, mode) in self.slab_actions:
            if mm == s.slab_mm and (mm == 0 or mode == s.slab_mode):
                a.setChecked(True)
                break

    # ============================================================= sync
    def _toggle_link(self):
        self.link = self.link_act.isChecked()

    def _on_index_changed(self, src):
        if not self.link:
            return
        p = src.plane_center_world()
        n_src = src.plane_axes_lps()[2]
        for vp in self.visible_viewports():
            v = vp.slice
            if v is src or v.state is None or v.state is src.state:
                continue
            if abs(float(np.dot(n_src, v.plane_axes_lps()[2]))) > 0.8:   # same plane direction in the patient
                v.set_from_world(p)

    def _on_cross(self, state, z, y, x):
        world = state.vol.index_to_world(np.array([z, y, x], float))
        for vp in self.viewports:
            v = vp.slice
            if v.state is None:
                continue
            if v.state is state:
                v.set_cross(z, y, x)
            elif self.link:
                idx = np.rint(v.state.vol.world_to_index(world)).astype(int)
                idx = np.clip(idx, 0, np.array(v.state.vol.shape) - 1)
                v.set_cross(*(int(i) for i in idx))

    # cine runs per viewport (SliceView): the toolbar button / Ctrl+Space play the active one
    def _toggle_cine(self):
        s = self._active_slice()
        if s is None:
            self.cine_act.setChecked(False)
            return
        s.set_cine(self.cine_act.isChecked())
        if s.cine_playing:
            self.statusBar().showMessage(f"Cine: {s.cine_fps_value():.0f} frames/s", 4000)

    def _on_cine_changed(self, s):
        if self.active is not None and self.active.slice is s:
            self.cine_act.setChecked(s.cine_playing)

    def _series_step(self, s, step):
        """Cine bar: open the previous / next series of the list in this viewport (a run plays on opening)."""
        vp = next((v for v in self.viewports if v.slice is s), None)
        items = [it.data(Qt.UserRole) for it in self.series_list.series_items()]
        uids = [i.uid for i in items]
        if vp is None or s.state is None or s.vol.uid not in uids:
            return
        k = uids.index(s.vol.uid) + step
        if 0 <= k < len(items):
            self._open_series_in(vp, items[k])

    # ================================================================= DSA
    def _toggle_dsa(self):
        s = self._active_slice()
        if s is None or not s.dsa_available():
            self.act_dsa.setChecked(False)
            self.statusBar().showMessage("DSA needs a dynamic series with several frames (XA / fluoroscopy), "
                                         "shown in its own (axial) plane.", 6000)
            return
        on = s.set_dsa(self.act_dsa.isChecked())
        self.act_dsa.setChecked(on)
        if on:
            self.statusBar().showMessage(f"DSA on — {s.dsa.label()[0]}. Shift+/ = displayed image as mask, "
                                         "Shift+M = manual pixel-shift, Ctrl+M = auto pixel-shift.", 8000)

    def _dsa_do(self, fn):
        s = self._active_slice()
        if s is not None and s.dsa_available():
            fn(s)
            self.act_dsa.setChecked(s.dsa is not None)

    def _dsa_tool(self, key):
        """Manual / auto pixel-shift on the left mouse button; switches DSA on."""
        s = self._active_slice()
        if s is None or not s.dsa_available():
            self._toggle_dsa()          # shows why not
            return
        s.set_dsa(True)
        self.act_dsa.setChecked(True)
        self.bank.assign("left", key)

    # ============================================================= fusion
    def open_fusion(self):
        if self.fusion_dlg is None:
            self.fusion_dlg = FusionDialog(self)
        self.fusion_dlg.refresh()
        self.fusion_dlg.show()
        self.fusion_dlg.raise_()

    # =============================================================== DTI
    def _add_derived(self, vol, ww, wl):
        st = VolumeState(vol)
        vol.default_ww, vol.default_wl = ww, wl
        self.states[vol.uid] = st
        info = SeriesInfo(uid=vol.uid, files=[""] * vol.shape[0], description=vol.description,
                          modality=vol.modality, patient=vol.patient, derived=st)
        mid = vol.shape[0] // 2
        if vol.rgb is not None:
            thumb = vol.rgb[mid].mean(axis=-1)
        else:
            thumb = vol.data[mid].astype(np.float32)
        lo, hi = np.percentile(thumb, (1, 99.5))
        t = np.clip((thumb - lo) / max(1e-9, hi - lo) * 255, 0, 255).astype(np.uint8)
        info.thumbnail = np.ascontiguousarray(t[:: max(1, t.shape[0] // 96), :: max(1, t.shape[1] // 96)])
        src = next((s for s in self.series.values() if s.patient == vol.patient and s.derived is None), None)
        if src is not None:          # same patient group; the maps get their own "Derived maps" study header
            info.patient_id, info.birth_date, info.study_date = src.patient_id, src.birth_date, src.study_date
        info.study_uid = f"derived:{src.study_uid if src else ''}"
        self.series[vol.uid] = info
        if vol.rgb is not None:
            info._thumb_rgb = np.ascontiguousarray(
                vol.rgb[mid][:: max(1, vol.shape[1] // 96), :: max(1, vol.shape[2] // 96)])
        self._rebuild_series_list()
        self._update_study_header()
        return st

    def open_dti(self):
        dwi = [s for s in self.series.values() if s.is_dwi]
        st = self._active_state()
        info = next((s for s in dwi if st is not None and s.uid == st.vol.uid), None)
        if info is None and dwi:
            if len(dwi) == 1:
                info = dwi[0]
            else:
                names = [f"{s.description} ({s.count} img)" for s in dwi]
                most = max(range(len(dwi)), key=lambda i: dwi[i].count)    # the tensor series, not TRACEW / ADC
                name, ok = QInputDialog.getItem(self, "DTI", "Diffusion (DWI) series:", names, most, False)
                if not ok:
                    return
                info = dwi[names.index(name)]
        if info is None:
            QMessageBox.information(self, "No diffusion series",
                                    "No diffusion-weighted (DWI/DTI) series with b-value tags was found.\n"
                                    "Open the folder that contains the DTI acquisition (usually 'DTI', 'DWI' or "
                                    "'ep2d_diff' with b-values and ≥6 directions).")
            return

        def compute(info, progress=None):
            dwi_data = load_dwi(info, progress)
            return fit_tensor(dwi_data, progress=progress)

        def done(tm):
            d = tm.dwi
            self._add_derived(d.volume((tm.fa * 1000).astype(np.int16), "FA", f"DTI FA ({d.description})"), 1000, 500)
            self._add_derived(d.volume(np.clip(tm.md * 1e6, 0, 32000).astype(np.int16), "MD",
                                       f"DTI MD ×10⁻⁶ mm²/s ({d.description})"), 2500, 1250)
            self._add_derived(d.volume((tm.fa * 1000).astype(np.int16), "CFA", f"DTI color FA ({d.description})",
                                       rgb=tm.color_fa()), 1000, 500)
            b0 = tm.b0
            hi = float(np.percentile(b0, 99.5)) or 1.0
            self._add_derived(d.volume(np.clip(b0 / hi * 2000, 0, 32000).astype(np.int16), "B0",
                                       f"DWI b0 ({d.description})"), 2000, 1000)
            tw = TractWindow(tm, targets_cb=self._vr_targets)
            self.tract_windows.append(tw)
            tw.show()
            self.statusBar().showMessage(f"DTI: {len(d.bvals)} volumes, b = {sorted(set(np.round(d.bvals)))}. "
                                         "FA / MD / color FA / b0 were added to the series list.", 10000)

        self.statusBar().showMessage("Computing diffusion tensor...")
        self._loading_title = "Computing diffusion tensor (DTI)"
        self._run(compute, (info,), done)

    def _vr_targets(self):
        return [(f"3D VR: {w.state.vol.description}", w) for w in self.vr_windows.values() if w.isVisible()]

    # ============================================================== misc
    def _save_image(self):
        s = self._active_slice()
        if s is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save image", "image.png", "PNG (*.png);;JPEG (*.jpg)")
        if path:
            s.grab().save(path)
            self.statusBar().showMessage(f"Saved: {path}", 5000)

    def _copy_image(self):
        s = self._active_slice()
        if s:
            QApplication.clipboard().setImage(s.grab().toImage())
            self.statusBar().showMessage("Image copied to the clipboard.", 3000)

    def _dicom_tags(self):
        s = self._active_slice()
        st = self._active_state()
        if st is None or not st.vol.files:
            return
        z = s.index if s and s.orientation == "axial" else (s.cross[0] if s and s.cross else st.vol.shape[0] // 2)
        path = st.vol.files[min(z, len(st.vol.files) - 1)]
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
        except Exception as ex:  # noqa: BLE001
            QMessageBox.warning(self, "Failed", str(ex))
            return
        dlg = QDialog(self)
        dlg.setWindowTitle(f"DICOM tags — {os.path.basename(path)}")
        dlg.resize(760, 700)
        v = QVBoxLayout(dlg)
        flt = QLineEdit()
        flt.setPlaceholderText("Search tag / name / value...")
        v.addWidget(flt)
        tree = QTreeWidget()
        tree.setHeaderLabels(["Tag", "Name", "VR", "Value"])
        tree.header().setSectionResizeMode(3, QHeaderView.Stretch)

        def add(parent, dataset):
            for el in dataset:
                val = "" if el.VR == "SQ" else str(el.value)[:300]
                it = QTreeWidgetItem([f"({el.tag.group:04X},{el.tag.element:04X})", el.name, el.VR, val])
                (parent.addChild(it) if parent else tree.addTopLevelItem(it))
                if el.VR == "SQ":
                    for i, sub in enumerate(el.value):
                        si = QTreeWidgetItem([f"Item {i + 1}", "", "", ""])
                        it.addChild(si)
                        add(si, sub)

        add(None, ds)
        v.addWidget(tree)

        def do_filter(text):
            t = text.lower()
            for i in range(tree.topLevelItemCount()):
                it = tree.topLevelItem(i)
                it.setHidden(bool(t) and not any(t in it.text(c).lower() for c in range(4)))

        flt.textChanged.connect(do_filter)
        dlg.exec()

    def _abc2(self):
        s = self._active_slice()
        lengths = []
        if s:
            sr, sc = s._spacing_rc()
            for m in s.measurements():
                if m.kind == "length":
                    lengths.append(m._len(m.pts[0], m.pts[1], sr, sc) / 10)
        abc2_dialog(self, lengths)

    def _help(self):
        QMessageBox.information(self, "Quick guide", (
            "TOOLS & MOUSE BUTTONS\n"
            "  LEFT-click a tool button = assign it to the left mouse button (blue line)\n"
            "  RIGHT-click a tool button = assign it to the right mouse button (pink line)\n"
            "  Middle button = pan • Wheel = slice • Ctrl+wheel = zoom at cursor\n"
            "  Double-click / F = maximise viewport • F11 = full screen\n\n"
            "MPR (Ctrl+M) and 3D (Ctrl+3) open in separate windows.\n"
            "  MPR: all 2D tools + drag a line = shift plane, drag a circle = rotate (oblique), drag centre = move.\n"
            "  3D: color curves, 🦴 bone color lock, crop, scalpel, inspect, segmentation, 3D length/angle.\n\n"
            "DSA (XA / fluoroscopy runs):\n"
            "  / subtraction on/off • Shift+/ displayed image = mask • Ctrl+Shift+/ reset mask\n"
            "  Shift+M manual pixel-shift (drag; Shift+arrows 1 px, Ctrl+Shift+arrows 0.1 px)\n"
            "  Ctrl+M (in DSA mode) auto pixel-shift: click inside the yellow square • Ctrl+Space cine at the "
            "series' frame rate\n"
            "  Runs play on opening. Cine bar (mouse over the run): previous / next series, previous / next frame,\n"
            "  play / pause [Space], frames per second.\n\n"
            "SHORTCUTS: W W/L • M pan • Z zoom • S browse • L length • E ellipse • U rectangle • G polygon\n"
            "  O open polygon • A angle • C Cobb • D deviation • Y arrow • P pencil • T text\n"
            "  Q 3D cursor • K color picker • B brush • J seed • R W/L ROI • I invert • 0 auto W/L\n"
            "  1–9 window presets • X crosshair • Ctrl+L link • Ctrl+T DICOM tags\n\n"
            "Note: a personal visualisation aid, not a certified diagnostic device."))


class FusionDialog(QDialog):
    """Overlay another series on the active viewport (multi-sequence fusion in patient coordinates)."""

    def __init__(self, main):
        super().__init__(main)
        from PySide6.QtWidgets import QCheckBox, QGroupBox, QSlider
        self.main = main
        self.setWindowTitle("Fusion — overlay series")
        self.setModal(False)
        self.setMinimumWidth(460)
        v = QVBoxLayout(self)
        v.setContentsMargins(14, 12, 14, 12)
        v.setSpacing(10)
        head = QHBoxLayout()
        ic = QLabel()
        ic.setPixmap(icon("fusion").pixmap(30, 30))
        head.addWidget(ic, 0, Qt.AlignTop)
        title = QLabel("<span style='font-size:14px; font-weight:600'>Overlay another series</span><br>"
                       f"<span style='color:{theme.MUTED}'>Shown on the active viewport, aligned in patient "
                       "coordinates — e.g. FLAIR, FA or color FA on T1.</span>")
        title.setWordWrap(True)
        head.addWidget(title, 1)
        v.addLayout(head)

        box = QGroupBox("Overlay")
        f = QFormLayout(box)
        self.series = QComboBox()
        self.series.setIconSize(QSize(36, 36))
        self.series.activated.connect(lambda _: self._on_series())
        f.addRow("Series:", self.series)
        self.cmap = QComboBox()
        self.cmap.setIconSize(QSize(44, 12))
        for name in ["Hot", "Rainbow", "Perfusion", "Bone", "Gray", "Native RGB"]:
            self.cmap.addItem(cmap_icon(COLORMAPS[name]) if name in COLORMAPS else QIcon(), name)
        self.cmap.currentIndexChanged.connect(lambda _: self.apply())
        f.addRow("Color map:", self.cmap)
        v.addWidget(box)

        box = QGroupBox("Display")
        f = QFormLayout(box)
        row = QHBoxLayout()
        self.opacity = QSlider(Qt.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.setValue(50)
        self.opacity_lbl = QLabel("50 %")
        self.opacity_lbl.setMinimumWidth(40)
        self.opacity.valueChanged.connect(lambda val: (self.opacity_lbl.setText(f"{val} %"), self.apply()))
        row.addWidget(self.opacity, 1)
        row.addWidget(self.opacity_lbl)
        f.addRow("Opacity:", row)
        row = QHBoxLayout()
        self.ww = QDoubleSpinBox()
        self.ww.setRange(1, 100000)
        self.ww.setDecimals(0)
        self.wl = QDoubleSpinBox()
        self.wl.setRange(-100000, 100000)
        self.wl.setDecimals(0)
        for sp in (self.ww, self.wl):
            sp.setKeyboardTracking(False)
            sp.valueChanged.connect(lambda _: self.apply())
        row.addWidget(QLabel("WW"))
        row.addWidget(self.ww, 1)
        row.addWidget(QLabel("WL"))
        row.addWidget(self.wl, 1)
        bt = QPushButton("Auto")
        bt.setToolTip("Window from the overlay's own values")
        bt.clicked.connect(self._auto)
        row.addWidget(bt)
        f.addRow("Window:", row)
        self.hide_below = QCheckBox("Transparent below the window (show only bright parts)")
        self.hide_below.setChecked(True)
        self.hide_below.toggled.connect(lambda _: self.apply())
        f.addRow(self.hide_below)
        self.all_vp = QCheckBox("Apply to every viewport of the active series")
        self.all_vp.setChecked(True)
        self.all_vp.toggled.connect(lambda _: self.apply())
        f.addRow(self.all_vp)
        v.addWidget(box)

        row = QHBoxLayout()
        bt = QPushButton(icon("trash"), "Remove overlay")
        bt.clicked.connect(self.remove)
        row.addWidget(bt)
        row.addStretch(1)
        bt = QPushButton("Close")
        bt.clicked.connect(self.close)
        row.addWidget(bt)
        v.addLayout(row)
        self.ov_state = None

    def remove(self):
        self.series.setCurrentIndex(0)
        self._on_series()

    def refresh(self):
        cur = self.series.currentData()
        self.series.blockSignals(True)
        self.series.clear()
        self.series.addItem("(none)", None)
        for uid, info in self.main.series.items():
            self.series.addItem(series_icon(info), f"{info.modality}  {info.description}  ({info.count} img)", uid)
        if cur:
            i = self.series.findData(cur)
            if i >= 0:
                self.series.setCurrentIndex(i)
        self.series.blockSignals(False)

    def _on_series(self):
        uid = self.series.currentData()
        if uid is None:
            self.ov_state = None
            self.apply()
            return

        def loaded(st):
            self.ov_state = st
            if st.vol.rgb is not None:
                self.cmap.setCurrentText("Native RGB")
            self._auto()
        self.main._load(self.main.series[uid], loaded)

    def _auto(self):
        if self.ov_state is None:
            return
        d = self.ov_state.vol.data
        lo, hi = np.percentile(d[d.shape[0] // 2], (40, 99.7))
        for sp, val in ((self.ww, max(1.0, hi - lo)), (self.wl, (hi + lo) / 2)):
            sp.blockSignals(True)
            sp.setValue(float(val))
            sp.blockSignals(False)
        self.apply()

    def apply(self):
        s = self.main._active_slice()
        if s is None:
            return
        fus = None
        if self.ov_state is not None and self.ov_state is not s.state:
            fus = dict(state=self.ov_state, cmap=self.cmap.currentText(), opacity=self.opacity.value() / 100,
                       ww=self.ww.value(), wl=self.wl.value(), hide_below=self.hide_below.isChecked())
        if self.all_vp.isChecked():
            self.main._all_slices(lambda v: v.set_fusion(fus), s.state)
        else:
            s.set_fusion(fus)
