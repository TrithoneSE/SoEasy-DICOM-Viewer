"""Collage (toolbar Collage, File → Collage…, Ctrl+G): one image of 5 × 5, 4 × 4, 4 × 3… tiles from the active
viewport, one slice every N mm (5 mm by default). The run of slices starts centred in the series — the middle of a
head, not the first slices below the skull base — and a slider moves it. A series whose slices are already that far
apart (a thick CT of 29 slices, an MRI of 20) keeps every slice. Tiles are rendered by the viewport itself, so window /
level, colour map, filters, slabs, rotation and segments look exactly as on screen."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QPushButton, QSlider, QSpinBox, QVBoxLayout,
)

from . import privacy

# (columns, rows)
GRIDS = [(5, 5), (4, 4), (4, 3), (3, 3), (3, 2), (6, 5)]
DEFAULT_INTERVAL_MM = 5.0
HEADER_PX = 30


def step_for(spacing_mm, interval_mm):
    """Slices between two tiles: 1 when the slices are already interval_mm or more apart."""
    if spacing_mm <= 0 or interval_mm <= 0:
        return 1
    return max(1, int(round(interval_mm / spacing_mm)))


def slice_indices(count, spacing_mm, interval_mm, cells, first=None):
    """At most `cells` indices, one every interval_mm, starting at `first` (None = centred in the series)."""
    step = step_for(spacing_mm, interval_mm)
    n = min(cells, (count - 1) // step + 1)
    max_first = count - 1 - (n - 1) * step
    first = max_first // 2 if first is None else int(np.clip(first, 0, max_first))
    return [first + i * step for i in range(n)]


def frame_indices(count, cells):
    """A movie (XA / US run) has no slice spacing: spread the frames evenly over the image."""
    if count <= cells:
        return list(range(count))
    return sorted(set(int(round(v)) for v in np.linspace(0, count - 1, cells)))


def is_movie(view):
    return view.vol.frame_time is not None and view.orientation == "axial"


def render_collage(view, indices, cols, rows, tile_w=384, labels=True, title=""):
    """One QImage of cols × rows tiles (the first cols × rows of `indices`)."""
    indices = list(indices)[:cols * rows]
    if not indices:
        return None
    old = view.index
    tiles = []
    try:
        for k in indices:
            view.index = k
            tiles.append((k, view._compose(), None if is_movie(view) else view.world_pos()))
    finally:
        view.index = old
    img0 = tiles[0][1]
    sr, sc = view._disp_spacing()
    tile_h = max(16, int(round(tile_w * (img0.height() * sr) / max(1e-6, img0.width() * sc))))
    top = HEADER_PX if labels else 0
    used_rows = (len(tiles) + cols - 1) // cols        # no empty rows when the series has fewer slices
    page = QImage(cols * tile_w, top + used_rows * tile_h, QImage.Format_RGB888)
    page.fill(Qt.black)
    qp = QPainter(page)
    qp.setRenderHint(QPainter.SmoothPixmapTransform, True)
    qp.setRenderHint(QPainter.Antialiasing, True)
    for i, (k, img, pos) in enumerate(tiles):
        r, c = divmod(i, cols)
        rect = QRectF(c * tile_w, top + r * tile_h, tile_w, tile_h)
        qp.drawImage(rect, img)
        if labels:
            qp.setFont(QFont("Segoe UI", max(8, tile_w // 32)))
            fm = qp.fontMetrics()
            qp.setPen(QColor(235, 235, 235))
            qp.drawText(int(rect.x()) + 5, int(rect.y()) + 3 + fm.ascent(), f"{k + 1}/{view.count()}")
            if pos is not None:
                t = f"{pos:.1f} mm"
                qp.drawText(int(rect.right() - fm.horizontalAdvance(t) - 5), int(rect.bottom()) - 5, t)
        qp.setPen(QColor(60, 60, 60))
        qp.drawRect(rect)
    if labels:
        qp.setFont(QFont("Segoe UI", 11))
        fm = qp.fontMetrics()
        qp.setPen(QColor(235, 235, 235))
        qp.drawText(8, (HEADER_PX + fm.ascent() - fm.descent()) // 2, title)
    qp.end()
    return page


class CollageDialog(QDialog):
    """Settings and a live preview; Save writes the image, Copy puts it on the clipboard."""

    def __init__(self, parent, view):
        super().__init__(parent)
        self.view = view
        self.setWindowTitle("Collage")
        self.movie = is_movie(view)
        self.spacing = view.axis_spacing()
        v = QVBoxLayout(self)
        f = QFormLayout()
        self.grid = QComboBox()
        for c, r in GRIDS:
            self.grid.addItem(f"{c} × {r}  ({c} columns, {r} rows)", (c, r))
        self.grid.addItem("Custom…", None)
        f.addRow("Tiles:", self.grid)
        row = QHBoxLayout()
        self.cols = QSpinBox()
        self.cols.setRange(1, 12)
        self.cols.setValue(5)
        self.rows = QSpinBox()
        self.rows.setRange(1, 12)
        self.rows.setValue(5)
        row.addWidget(self.cols)
        row.addWidget(QLabel("columns ×"))
        row.addWidget(self.rows)
        row.addWidget(QLabel("rows"))
        self.custom_row = row
        f.addRow("", row)
        self.interval = QDoubleSpinBox()
        self.interval.setRange(0.1, 100.0)
        self.interval.setDecimals(1)
        self.interval.setSingleStep(0.5)
        self.interval.setSuffix(" mm")
        self.interval.setValue(DEFAULT_INTERVAL_MM)
        self.interval.setToolTip("One image every … mm. Slices already this far apart are all kept.")
        f.addRow("One image every:", self.interval)
        row = QHBoxLayout()
        self.start = QSlider(Qt.Horizontal)
        self.start.setToolTip("Move the run of slices through the series (it starts centred)")
        self.first = QSpinBox()
        self.first.setPrefix("from ")
        self.first.setToolTip("First image of the collage")
        center = QPushButton("Centre")
        center.setToolTip("Put the run of slices back in the middle of the series")
        center.clicked.connect(lambda: self.set_first(None))
        row.addWidget(self.start, 1)
        row.addWidget(self.first)
        row.addWidget(center)
        f.addRow("Position:", row)
        self.size = QComboBox()
        for px in (256, 384, 512):
            self.size.addItem(f"{px} px", px)
        self.size.setCurrentIndex(1)
        f.addRow("Tile width:", self.size)
        self.labels = QCheckBox("Header, image number and position")
        self.labels.setChecked(True)
        f.addRow(self.labels)
        v.addLayout(f)
        self.info = QLabel()
        self.info.setWordWrap(True)
        v.addWidget(self.info)
        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(420, 320)
        self.preview.setStyleSheet("background: black;")
        v.addWidget(self.preview, 1)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        self.save_btn = QPushButton("Save…")
        self.copy_btn = QPushButton("Copy")
        bb.addButton(self.save_btn, QDialogButtonBox.ActionRole)
        bb.addButton(self.copy_btn, QDialogButtonBox.ActionRole)
        bb.rejected.connect(self.reject)
        self.save_btn.clicked.connect(lambda: self.save())
        self.copy_btn.clicked.connect(self.copy)
        v.addWidget(bb)
        if self.movie:
            self.interval.setEnabled(False)
            self.interval.setToolTip("A movie run has no slice spacing: its frames are spread over the image.")
            for w in (self.start, self.first, center):
                w.setEnabled(False)
        self._first = None                  # None = centred
        self._image = None
        for w in (self.cols, self.rows, self.interval):
            w.valueChanged.connect(self._layout_changed)
        self.start.valueChanged.connect(self.set_first)
        self.first.valueChanged.connect(lambda k: self.set_first(k - 1))
        self.grid.currentIndexChanged.connect(self._grid_changed)
        self.size.currentIndexChanged.connect(self.refresh)
        self.labels.toggled.connect(self.refresh)
        self._grid_changed()

    def cells(self):
        return self.cols.value() * self.rows.value()

    def _grid_changed(self):
        cr = self.grid.currentData()
        custom = cr is None
        for i in range(self.custom_row.count()):
            w = self.custom_row.itemAt(i).widget()
            if w:
                w.setVisible(custom)
        if not custom:
            for sp, val in ((self.cols, cr[0]), (self.rows, cr[1])):
                sp.blockSignals(True)
                sp.setValue(val)
                sp.blockSignals(False)
        self._layout_changed()

    def _layout_changed(self):
        """Grid or interval changed: keep the run centred on the same slice."""
        if self._first is not None:
            step = step_for(self.spacing, self.interval.value())
            n = min(self.cells(), (self.view.count() - 1) // step + 1)
            self._first = int(round(self._mid - (n - 1) * step / 2))
        self.refresh()

    def set_first(self, k):
        self._first = k
        self.refresh()

    def indices(self):
        if self.movie:
            return frame_indices(self.view.count(), self.cells())
        return slice_indices(self.view.count(), self.spacing, self.interval.value(), self.cells(), self._first)

    def title(self):
        vol = self.view.vol
        parts = [privacy.name(vol.patient), vol.description, self.view.orientation.upper()]
        if not self.movie:
            step = step_for(self.spacing, self.interval.value()) * self.spacing
            parts.append(f"every {step:.1f} mm")
        parts.append(f"WW {self.view.ww:.0f} / WL {self.view.wl:.0f}")
        return "   ·   ".join(p for p in parts if p)

    def image(self):
        if self._image is None:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                self._image = render_collage(self.view, self.indices(), self.cols.value(), self.rows.value(),
                                             self.size.currentData(), self.labels.isChecked(), self.title())
            finally:
                QApplication.restoreOverrideCursor()
        return self._image

    def refresh(self):
        self._image = None
        idx = self.indices()
        self._mid = (idx[0] + idx[-1]) / 2      # centre kept when the grid or the interval changes
        n = self.view.count()
        step = step_for(self.spacing, self.interval.value())
        if not self.movie:
            max_first = n - 1 - (len(idx) - 1) * step
            for w in (self.start, self.first):
                w.blockSignals(True)
            self.start.setRange(0, max(0, max_first))
            self.start.setValue(idx[0])
            self.first.setRange(1, max(1, max_first + 1))
            self.first.setValue(idx[0] + 1)
            self.first.setSuffix(f" to {idx[-1] + 1} of {n}")
            for w in (self.start, self.first):
                w.blockSignals(False)
        if self.movie:
            how = f"{len(idx)} frames spread over the run of {n}"
        elif len(idx) == n and step == 1 and self.spacing >= self.interval.value() - 1e-3:
            how = f"all {n} slices — they are {self.spacing:.1f} mm apart already"
        elif len(idx) < self.cells():
            how = f"the whole series fits: {len(idx)} images, one every {step * self.spacing:.1f} mm"
        else:
            how = (f"{len(idx)} images, one every {step * self.spacing:.1f} mm — images {idx[0] + 1}–{idx[-1] + 1} "
                   f"of {n}; move the slider to shift them")
        self.info.setText(how + ".")
        img = self.image()
        if img is not None:
            self.preview.setPixmap(QPixmap.fromImage(img).scaled(self.preview.size(), Qt.KeepAspectRatio,
                                                                 Qt.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._image is not None:
            self.preview.setPixmap(QPixmap.fromImage(self._image).scaled(self.preview.size(), Qt.KeepAspectRatio,
                                                                         Qt.SmoothTransformation))

    def save(self, path=None):
        """Returns the file written, or None."""
        if not path:
            path, _ = QFileDialog.getSaveFileName(self, "Save collage", "collage.png", "PNG (*.png);;JPEG (*.jpg)")
        if not path:
            return None
        self.image().save(path)
        if self.parent() is not None and hasattr(self.parent(), "statusBar"):
            self.parent().statusBar().showMessage(f"Collage saved: {path}", 8000)
        return path

    def copy(self):
        img = self.image()
        if img is not None:
            QApplication.clipboard().setImage(img)
