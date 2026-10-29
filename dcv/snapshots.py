"""3D snapshots and movies.

A snapshot stores the camera (orientation, position, zoom), the crop box, the color preset, the render mode /
global opacity and a thumbnail.
Clicking a snapshot restores it; a movie flies through the snapshots in list order (camera on a spline, crop box
interpolated, colors switching half way) and is written as an animated GIF or a numbered PNG sequence.
Snapshot files: *.se3d (JSON, thumbnails as base64 PNG).
"""

from __future__ import annotations

import base64
import json
import os

import numpy as np
from . import vtk_lite as vtk
from vtkmodules.util import numpy_support
from PySide6.QtCore import QByteArray, QBuffer, QIODevice, QSize, Qt, Signal
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QGridLayout, QListView, QListWidget,
    QListWidgetItem, QMessageBox, QProgressDialog, QPushButton, QSpinBox, QVBoxLayout,
)

from . import theme
from .tf_model import TransferFunction

THUMB = 96


# ----------------------------------------------------------------- capture / apply
def _camera(cam):
    return dict(position=list(cam.GetPosition()), focal=list(cam.GetFocalPoint()), up=list(cam.GetViewUp()),
                angle=cam.GetViewAngle(), parallel_scale=cam.GetParallelScale())


def _set_camera(cam, c):
    cam.SetPosition(*c["position"])
    cam.SetFocalPoint(*c["focal"])
    cam.SetViewUp(*c["up"])
    cam.SetViewAngle(c["angle"])
    cam.SetParallelScale(c["parallel_scale"])


def _crop(vr):
    if vr.box_widget is None:
        return dict(on=False, matrix=None)
    t = vtk.vtkTransform()
    vr.box_widget.GetRepresentation().GetTransform(t)
    m = t.GetMatrix()
    return dict(on=bool(vr.crop_on), matrix=[m.GetElement(i, j) for i in range(4) for j in range(4)])


def _set_box_matrix(vr, matrix):
    vr._ensure_box()
    t = vtk.vtkTransform()
    t.SetMatrix(matrix)
    vr.box_widget.GetRepresentation().SetTransform(t)


def grab_rgb(vr, max_w=None):
    """Current 3D view as an RGB numpy array (crop box handles hidden), optionally scaled to max_w wide."""
    handles = vr.box_widget is not None and vr.box_widget.GetEnabled()
    if handles:
        vr.box_widget.Off()
    vr.render()
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(vr.rw)
    w2i.SetInputBufferTypeToRGB()
    w2i.ReadFrontBufferOff()
    w2i.Update()
    img = w2i.GetOutput()
    w, h, _ = img.GetDimensions()
    arr = numpy_support.vtk_to_numpy(img.GetPointData().GetScalars()).reshape(h, w, 3)[::-1].copy()
    if handles:
        vr.box_widget.On()
        vr.render()
    if max_w and w > max_w:
        q = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888).scaledToWidth(max_w, Qt.SmoothTransformation)
        q = q.convertToFormat(QImage.Format_RGB888)
        arr = np.frombuffer(q.constBits(), np.uint8).reshape(q.height(), q.bytesPerLine())[:, :q.width() * 3]
        arr = arr.reshape(q.height(), q.width(), 3).copy()
    return arr


def _thumb_png(rgb):
    h, w, _ = rgb.shape
    q = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).scaled(THUMB, THUMB, Qt.KeepAspectRatio,
                                                                     Qt.SmoothTransformation)
    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QIODevice.WriteOnly)
    q.save(buf, "PNG")
    return bytes(ba)


def capture(vw):
    """Snapshot of a VRWindow."""
    vr = vw.vr
    return dict(camera=_camera(vr.renderer.GetActiveCamera()), crop=_crop(vr), tf=vw.state.tf.to_dict(),
                gain=vw.state.gain, blend=vw._blend, thumb=_thumb_png(grab_rgb(vr)))


def apply(vw, snap, colors=True, crop=True):
    vr = vw.vr
    if colors:
        if snap.get("blend") and snap["blend"] != vw._blend:
            vw.set_blend(snap["blend"])
        vw.set_tf(TransferFunction.from_dict(snap["tf"]))
        vw.gain_slider.setValue(int(round(snap.get("gain", 1.0) * 100)))
        vw.apply_tf()
    if crop:
        _apply_crop(vw, snap.get("crop") or {})
    _set_camera(vr.renderer.GetActiveCamera(), snap["camera"])
    vr.renderer.ResetCameraClippingRange()
    vr.render()


def _apply_crop(vw, crop):
    vr = vw.vr
    if crop.get("matrix"):
        _set_box_matrix(vr, crop["matrix"])
    if bool(crop.get("on")) != vw.crop_act.isChecked():
        vw.crop_act.setChecked(bool(crop.get("on")))      # toggles VolumeView.set_crop
    elif vr.crop_on and vr.box_widget is not None:
        vr._on_box(vr.box_widget, None)


def _movie_box(snap):
    """Crop box matrix a movie flies through: a snapshot without crop counts as the whole volume (the box placed
    on the volume bounds = identity), so the cut opens and closes smoothly instead of switching mid-flight."""
    crop = snap.get("crop") or {}
    if crop.get("on") and crop.get("matrix"):
        return crop["matrix"]
    return [1.0 if i == j else 0.0 for i in range(4) for j in range(4)]


# ----------------------------------------------------------------- files
def save_file(path, snaps, movie):
    data = dict(format="SoEasy 3D snapshots", version=1, movie=movie,
                snapshots=[{**s, "thumb": base64.b64encode(s["thumb"]).decode("ascii")} for s in snaps])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def load_file(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("format") != "SoEasy 3D snapshots":
        raise ValueError("Not a SoEasy 3D snapshot file.")
    snaps = [{**s, "thumb": base64.b64decode(s["thumb"])} for s in data.get("snapshots", [])]
    return snaps, data.get("movie") or {}


# ----------------------------------------------------------------- movie
DEFAULT_MOVIE = dict(frames=40, fps=20, hold=10, loop=False, width=960)


def movie_frames(snaps, movie):
    """(snapshot index for the colors, camera / crop interpolation time) for every movie frame."""
    order = list(range(len(snaps))) + ([0] if movie.get("loop") and len(snaps) > 1 else [])
    out = []
    for k, i in enumerate(order):
        out += [(i, float(k))] * max(0, int(movie.get("hold", 0)))
        if k + 1 < len(order):
            n = max(1, int(movie.get("frames", 40)))
            for f in range(n):
                t = k + f / n
                out.append((order[k + 1] if f >= n / 2 else i, t))
    if not movie.get("hold"):                     # with a pause the last snapshot is already on screen
        out.append((order[-1], float(len(order) - 1)))
    return out, order


def make_movie(vw, snaps, movie, path, progress=None):
    """Render the fly-through and write it. Returns the number of frames written (0 when cancelled)."""
    vr = vw.vr
    frames, order = movie_frames(snaps, movie)
    ci = vtk.vtkCameraInterpolator()
    ci.SetInterpolationTypeToSpline()
    have_box = any((snaps[i].get("crop") or {}).get("on") for i in order)
    for k, i in enumerate(order):
        cam = vtk.vtkCamera()
        _set_camera(cam, snaps[i]["camera"])
        ci.AddCamera(float(k), cam)
    # crop box: element-wise blend of the matrices. vtkTransformInterpolator splits a matrix into rotation and
    # scale, which a box that was turned and then shortened (rotation x non-uniform scale) does not survive: the
    # box jumped to another shape right after the first snapshot. A blend is exact at every snapshot.
    boxes = [np.array(_movie_box(snaps[i]), float) for i in order] if have_box else []
    saved = capture(vw)
    if have_box and not vw.crop_act.isChecked():
        vw.crop_act.setChecked(True)                     # the crop stays on for the whole flight
    gif = path.lower().endswith(".gif")
    base, _ = os.path.splitext(path)
    images, color_of, written = [], None, 0
    try:
        for n, (ci_idx, t) in enumerate(frames):
            if progress is not None:
                progress(n, len(frames))
                if progress.cancelled():
                    return 0
            if ci_idx != color_of:
                apply(vw, snaps[ci_idx], colors=True, crop=not have_box)
                color_of = ci_idx
            ci.InterpolateCamera(t, vr.renderer.GetActiveCamera())
            if have_box:
                k = min(int(t), len(boxes) - 1)
                f = t - k
                m = boxes[k] if f <= 0 or k + 1 >= len(boxes) else (1 - f) * boxes[k] + f * boxes[k + 1]
                tt = vtk.vtkTransform()
                tt.SetMatrix(m.tolist())
                vr.box_widget.GetRepresentation().SetTransform(tt)
                if vr.crop_on:
                    vr._on_box(vr.box_widget, None)
            vr.renderer.ResetCameraClippingRange()
            rgb = grab_rgb(vr, movie.get("width") or None)
            if gif:
                from PIL import Image
                images.append(Image.fromarray(rgb).convert("P", palette=Image.ADAPTIVE, colors=255))
            else:
                from PIL import Image
                Image.fromarray(rgb).save(f"{base}_{n + 1:04d}.png")
            written += 1
        if gif and images:
            images[0].save(path, save_all=True, append_images=images[1:], loop=0,
                           duration=int(round(1000 / max(1, int(movie.get("fps", 20))))), optimize=False)
    finally:
        if have_box and not saved["crop"].get("matrix"):
            saved["crop"]["matrix"] = _movie_box({})      # the box was made for the movie: back to the whole volume
        apply(vw, saved, colors=True)
    return written


# ----------------------------------------------------------------- panel
class MovieSettings(QDialog):
    def __init__(self, parent, movie):
        super().__init__(parent)
        self.setWindowTitle("Movie settings")
        f = QFormLayout(self)
        self.frames = self._spin(2, 600, movie["frames"], " frames")
        f.addRow("Between two snapshots:", self.frames)
        self.hold = self._spin(0, 600, movie["hold"], " frames")
        f.addRow("Pause on each snapshot:", self.hold)
        self.fps = self._spin(1, 60, movie["fps"], " frames/s")
        f.addRow("Speed:", self.fps)
        self.width = self._spin(160, 3840, movie["width"], " px")
        f.addRow("Width:", self.width)
        self.loop = QCheckBox("Return to the first snapshot at the end")
        self.loop.setChecked(bool(movie["loop"]))
        f.addRow(self.loop)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        f.addRow(bb)

    @staticmethod
    def _spin(lo, hi, val, suffix):
        s = QSpinBox()
        s.setRange(lo, hi)
        s.setValue(int(val))
        s.setSuffix(suffix)
        return s

    def value(self):
        return dict(frames=self.frames.value(), hold=self.hold.value(), fps=self.fps.value(),
                    width=self.width.value(), loop=self.loop.isChecked())


class SnapshotPanel(QDialog):
    """Floating '3D snapshots' tool window of one VRWindow: list + Add / Delete / Load / Save / Movie."""
    changed = Signal()

    def __init__(self, vw):
        super().__init__(vw, Qt.Tool)
        self.vw = vw
        self.snaps = []
        self.movie = dict(DEFAULT_MOVIE)
        self.setWindowTitle("3D snapshots")
        self.setStyleSheet(theme.main_stylesheet())
        v = QVBoxLayout(self)
        self.list = QListWidget()
        self.list.setViewMode(QListView.IconMode)
        self.list.setFlow(QListView.TopToBottom)
        self.list.setWrapping(False)
        self.list.setMovement(QListView.Static)
        self.list.setIconSize(QSize(THUMB, THUMB))
        self.list.setSpacing(4)
        self.list.setMinimumWidth(THUMB + 60)
        self.list.itemClicked.connect(lambda it: self.go(self.list.row(it)))
        v.addWidget(self.list, 1)
        g = QGridLayout()
        self.buttons = {}
        for i, (key, text, fn, tip) in enumerate((
                ("movie", "Movie settings", self.edit_movie, "Frames between snapshots, pause, speed, width"),
                ("make", "Create movie...", self.create_movie, "Fly through the snapshots → animated GIF or PNGs"),
                ("add", "Add", self.add, "Add the current 3D view  [Insert]"),
                ("delete", "Delete", self.delete, "Delete the selected snapshot"),
                ("load", "Load", self.load, "Open a snapshot file (*.se3d)"),
                ("save", "Save", self.save, "Save all snapshots to a file (*.se3d)"))):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            g.addWidget(b, i // 2, i % 2)
            self.buttons[key] = b
        v.addLayout(g)
        self.resize(250, 560)
        self._sync_buttons()

    # ------------------------------------------------------------ list
    def _sync_buttons(self):
        has = bool(self.snaps)
        for k in ("delete", "save", "make"):
            self.buttons[k].setEnabled(has if k != "make" else len(self.snaps) >= 2)
        self.buttons["delete"].setEnabled(self.list.currentRow() >= 0 and has)

    def _refresh(self, select=None):
        self.list.clear()
        for i, s in enumerate(self.snaps):
            pm = QPixmap()
            pm.loadFromData(s["thumb"], "PNG")
            it = QListWidgetItem(QIcon(pm), f"{i + 1}")
            it.setTextAlignment(Qt.AlignHCenter)
            self.list.addItem(it)
        if select is not None and 0 <= select < len(self.snaps):
            self.list.setCurrentRow(select)
        self._sync_buttons()
        self.changed.emit()

    def add(self):
        self.snaps.append(capture(self.vw))
        self._refresh(select=len(self.snaps) - 1)
        self.vw.statusBar().showMessage(f"Snapshot {len(self.snaps)} added.", 4000)

    def delete(self):
        i = self.list.currentRow()
        if 0 <= i < len(self.snaps):
            del self.snaps[i]
            self._refresh(select=min(i, len(self.snaps) - 1))

    def go(self, i):
        if 0 <= i < len(self.snaps):
            self.list.setCurrentRow(i)
            apply(self.vw, self.snaps[i])
            self._sync_buttons()

    def step(self, d):
        if self.snaps:
            i = self.list.currentRow()
            self.go((i + d) % len(self.snaps) if i >= 0 else 0)

    # ------------------------------------------------------------ files
    def save(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save 3D snapshots", "snapshots.se3d", "3D snapshots (*.se3d)")
        if path:
            save_file(path, self.snaps, self.movie)
            self.vw.statusBar().showMessage(f"{len(self.snaps)} snapshots saved: {path}", 6000)

    def load(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load 3D snapshots", "", "3D snapshots (*.se3d)")
        if not path:
            return
        try:
            self.snaps, movie = load_file(path)
        except (OSError, ValueError, KeyError) as ex:
            QMessageBox.warning(self, "Could not load snapshots", str(ex))
            return
        self.movie.update(movie)
        self._refresh(select=0 if self.snaps else None)
        if self.snaps:
            self.go(0)

    # ------------------------------------------------------------ movie
    def edit_movie(self):
        d = MovieSettings(self, self.movie)
        if d.exec():
            self.movie = d.value()

    def create_movie(self):
        if len(self.snaps) < 2:
            QMessageBox.information(self, "Create movie", "Add at least two snapshots first.")
            return
        path, flt = QFileDialog.getSaveFileName(self, "Create movie from 3D snapshots", "movie3d.gif",
                                                "Animated GIF (*.gif);;PNG image sequence (*.png)")
        if not path:
            return
        frames, _ = movie_frames(self.snaps, self.movie)
        dlg = QProgressDialog("Rendering the movie...", "Cancel", 0, len(frames), self)
        dlg.setWindowTitle("Create movie")
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)

        def progress(i, n):
            dlg.setValue(i)
            QApplication.processEvents()
        progress.cancelled = dlg.wasCanceled
        n = make_movie(self.vw, self.snaps, self.movie, path, progress)
        dlg.close()
        if n:
            self.vw.statusBar().showMessage(f"Movie written: {path} ({n} frames)", 10000)
