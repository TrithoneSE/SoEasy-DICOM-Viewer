"""3D viewer: GPU volume rendering (VTK), crop box, scalpel cut/keep, segment surfaces, 3D measurements."""

from __future__ import annotations

import math

import numpy as np
import vtkmodules.qt

vtkmodules.qt.PyQtImpl = "PySide6"

from . import vtk_lite as vtk  # noqa: E402
from vtkmodules.util import numpy_support  # noqa: E402
from PySide6.QtCore import Qt, Signal  # noqa: E402
from PySide6.QtGui import QColor, QCursor, QImage, QPainter, QPixmap, QPolygonF  # noqa: E402
from PySide6.QtCore import QPointF  # noqa: E402
from PySide6.QtWidgets import QVBoxLayout, QWidget  # noqa: E402
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor  # noqa: E402

from .measure import fmt_len  # noqa: E402
from .segment import BONE_OFFSET, np_to_vtk_image  # noqa: E402
from .tf_model import combine  # noqa: E402

# 3D measurement tools (same names / shortcuts as the 2D tools; ROI tools need a flat plane, so they are 2D only)
MEASURE3D = {"length": ("Length", "L"), "angle": ("Angle", "A"), "cobb": ("Cobb angle", "C"),
             "openpoly": ("Open polygon", "O"), "arrow": ("Arrow", "Y"), "text": ("Text", "T")}
COL_NORMAL, COL_SELECTED, COL_DRAWING = (1, 0.85, 0), (0.2, 0.9, 1.0), (1.0, 0.55, 0.1)
RIM_GRAD_MAX = 320.0      # top of the 2D transfer function's gradient axis (HU); rim thresholds go up to 200


def project_inside(lasso, renderer, shape, spacing):
    """Voxels (full resolution, bool z/y/x) whose screen projection falls inside the drawn area.

    lasso: bool image (h, w), row 0 = top of the view. Fast path: per slice a coarse grid first finds the part
    of the slice that projects near the area, only that part is projected voxel by voxel; slices run in parallel
    threads (NumPy releases the GIL).
    """
    import os
    from concurrent.futures import ThreadPoolExecutor
    h, w = lasso.shape
    rows, cols = np.where(lasso.any(1))[0], np.where(lasso.any(0))[0]
    nz, ny, nx = shape
    inside = np.zeros(shape, bool)
    if not len(rows):
        return inside
    bx0, bx1 = cols[0], cols[-1]
    by0, by1 = h - 1 - rows[-1], h - 1 - rows[0]           # display coordinates: y up
    lut = np.zeros((h + 2, w + 2), bool)                   # 1-pixel border: out-of-view = outside
    lut[1:-1, 1:-1] = lasso[::-1]
    cam = renderer.GetActiveCamera()
    vm = cam.GetCompositeProjectionTransformMatrix(renderer.GetTiledAspectRatio(), -1, 1)
    M = np.array([[vm.GetElement(i, j) for j in range(4)] for i in range(4)])
    A = M.astype(np.float32)
    sz, sy, sx = spacing
    xi = (np.arange(nx) * sx).astype(np.float32)
    yi = (np.arange(ny) * sy).astype(np.float32)
    s = 8
    cy, cx = np.arange(0, ny, s), np.arange(0, nx, s)

    def proj(ys, xs, z):
        X, Y = xi[xs][None, :], yi[ys][:, None]
        inv = 1.0 / (A[3, 0] * X + A[3, 1] * Y + np.float32(M[3, 2] * z + M[3, 3]))
        px = (A[0, 0] * X + A[0, 1] * Y + np.float32(M[0, 2] * z + M[0, 3])) * inv
        py = (A[1, 0] * X + A[1, 1] * Y + np.float32(M[1, 2] * z + M[1, 3])) * inv
        return (px + 1) * np.float32(0.5 * w), (py + 1) * np.float32(0.5 * h)

    def work(k):
        z = k * sz
        px, py = proj(cy, cx, z)
        # margin = largest screen gap between coarse samples, so no voxel near the area is skipped
        pad = 2 + max(np.abs(np.diff(px, axis=1)).max(initial=0), np.abs(np.diff(py, axis=0)).max(initial=0),
                      np.abs(np.diff(py, axis=1)).max(initial=0), np.abs(np.diff(px, axis=0)).max(initial=0))
        hit = (px >= bx0 - pad) & (px <= bx1 + pad) & (py >= by0 - pad) & (py <= by1 + pad)
        if not hit.any():
            return
        r, c = np.where(hit.any(1))[0], np.where(hit.any(0))[0]
        y0, y1 = max(0, cy[r[0]] - s), min(ny, cy[r[-1]] + s + 1)
        x0, x1 = max(0, cx[c[0]] - s), min(nx, cx[c[-1]] + s + 1)
        px, py = proj(np.arange(y0, y1), np.arange(x0, x1), z)
        np.clip(px, -1, w, out=px)
        np.clip(py, -1, h, out=py)
        inside[k, y0:y1, x0:x1] = lut[(py + 1).astype(np.int32), (px + 1).astype(np.int32)]

    with ThreadPoolExecutor(min(16, os.cpu_count() or 4)) as ex:
        list(ex.map(work, range(nz)))
    return inside


def _angle3(v1, v2):
    c = np.dot(v1, v2) / max(1e-9, np.linalg.norm(v1) * np.linalg.norm(v2))
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


class Measure3D:
    """One 3D measurement: points on the rendered surface (world mm), its VTK actors, an optional note."""

    def __init__(self, kind, pts, note=""):
        self.kind = kind
        self.pts = [np.asarray(p, float) for p in pts]
        self.note = note
        self.actors = []
        self.label_offset = [8.0, 8.0]   # screen pixels from the anchor; the user can drag the caption
        self.text_actor = None

    @property
    def name(self):
        return MEASURE3D[self.kind][0]

    @property
    def label(self):
        P, k = self.pts, self.kind
        if k == "length" and len(P) >= 2:
            return fmt_len(np.linalg.norm(P[1] - P[0]))
        if k == "openpoly" and len(P) >= 2:
            return fmt_len(sum(np.linalg.norm(b - a) for a, b in zip(P, P[1:])))
        if k == "angle" and len(P) >= 3:
            return f"{_angle3(P[0] - P[1], P[2] - P[1]):.1f}°"
        if k == "cobb" and len(P) >= 4:
            a = _angle3(P[1] - P[0], P[3] - P[2])
            return f"Cobb: {min(a, 180 - a):.1f}°"
        return self.note

    @property
    def anchor(self):
        P, k = self.pts, self.kind
        if k == "length":
            return (P[0] + P[1]) / 2
        if k == "angle":
            return P[1]
        if k == "cobb" and len(P) >= 4:
            return (P[2] + P[3]) / 2
        return P[-1]


class VolumeView(QWidget):
    activated = Signal(object)
    cutDone = Signal(object)          # state
    inspected = Signal(object, object)  # state, (z, y, x)
    measurementsChanged = Signal()
    measureSelected = Signal(int)     # index in measurements3d, -1 = none
    toolFinished = Signal()           # a measurement / cut is done: the window goes back to rotate
    status = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        lay.addWidget(self.vtk_widget)
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0, 0, 0)
        self.rw = self.vtk_widget.GetRenderWindow()
        self.rw.AddRenderer(self.renderer)
        self.iren = self.rw.GetInteractor()
        self.style = vtk.vtkInteractorStyleTrackballCamera()
        self.iren.SetInteractorStyle(self.style)
        # disable VTK's built-in keyboard shortcuts ('3' = stereo, 'w'/'s' = wireframe/surface, 'e'/'q' = exit,
        # 'f' = fly-to, 'p' = pick...). Typing app shortcuts while the 3D view has focus must not break rendering.
        self._char_obs = self.iren.AddObserver("CharEvent", self._swallow_char, 100.0)
        self.iren.AddObserver("LeftButtonPressEvent", lambda *_: self.activated.emit(self), 10.0)
        self.iren.AddObserver("RightButtonPressEvent", lambda *_: self.activated.emit(self), 10.0)

        self.state = None
        self.image = None
        self.mapper = None
        self.volume = None
        self.prop = None
        self.ctf = vtk.vtkColorTransferFunction()
        self.otf = vtk.vtkPiecewiseFunction()
        self.box_widget = None
        self.crop_on = False
        self.crop_handles = True
        self._planes = None
        self._seg_actors = {}
        self._render_version = -1
        self.scalpel = None
        self._lasso = []
        self._lasso_down = False      # left button held while drawing the scalpel area
        self._lasso_obs = []

        self._text = vtk.vtkTextActor()
        self._text.GetTextProperty().SetFontSize(14)
        self._text.GetTextProperty().SetColor(0.85, 0.85, 0.85)
        self._text.SetPosition(8, 8)
        self.renderer.AddViewProp(self._text)

        # garis lasso scalpel
        self._lasso_pts = vtk.vtkPoints()
        self._lasso_pd = vtk.vtkPolyData()
        self._lasso_pd.SetPoints(self._lasso_pts)
        self._lasso_lines = vtk.vtkCellArray()
        self._lasso_pd.SetLines(self._lasso_lines)
        m2 = vtk.vtkPolyDataMapper2D()
        m2.SetInputData(self._lasso_pd)
        self._lasso_actor = vtk.vtkActor2D()
        self._lasso_actor.SetMapper(m2)
        self._lasso_actor.GetProperty().SetColor(1, 0.2, 0.2)
        self._lasso_actor.GetProperty().SetLineWidth(2)
        self.renderer.AddViewProp(self._lasso_actor)

        # overlay layer for 3D measurements: always visible, never hidden by the volume
        self.rw.SetNumberOfLayers(2)
        self.overlay = vtk.vtkRenderer()
        self.overlay.SetLayer(1)
        self.overlay.InteractiveOff()
        self.overlay.SetActiveCamera(self.renderer.GetActiveCamera())
        self.rw.AddRenderer(self.overlay)
        self.measurements3d: list[Measure3D] = []
        self._sel = None              # selected measurement (cyan), Delete removes it
        self._drawing = None          # measurement being drawn
        self._dragging = False        # left button held while drawing
        self._edit = None             # (measurement, point index) whose handle is being dragged
        self._label_drag = None       # (measurement, last mouse position) whose caption is being dragged
        self._text_renderer = vtk.vtkTextRenderer()
        self._press_xy = None
        self._rpress_xy = None
        self._coord = vtk.vtkCoordinate()
        self._coord.SetCoordinateSystemToWorld()

        self.vtk_widget.Initialize()
        self._add_orientation_cube()
        # measurement mouse/keyboard handling runs before the trackball style (priority 9 > 0) and before the
        # scalpel / inspect observers (5); it only swallows events it uses, so rotate / zoom / pan keep working
        self._m_obs = {}
        for ev, fn in (("LeftButtonPressEvent", self._m_press), ("MouseMoveEvent", self._m_move),
                       ("LeftButtonReleaseEvent", self._m_release), ("RightButtonPressEvent", self._m_rpress),
                       ("KeyPressEvent", self._m_key)):
            self._m_obs[ev] = self.iren.AddObserver(ev, fn, 9.0)
        # while rotating / zooming the trackball style grabs the mouse focus and normal observers stop getting
        # the button release; passive observers still get it (used for click = select, right-click = menu)
        for ev, fn in (("LeftButtonReleaseEvent", self._m_click_select), ("RightButtonReleaseEvent", self._m_rrelease)):
            self.iren.GetCommand(self.iren.AddObserver(ev, fn, 9.5)).SetPassiveObserver(1)
        # left-drag rotation mode: "free" = trackball (VTK style), "roll" = around the viewing axis only,
        # "turn" = around the patient's head-foot axis only. Runs after the measurement handlers (9) and before
        # the trackball (0); only takes the drag when no tool is on, so measuring / cutting work as before.
        self.rotate_mode = "free"
        self._rot_last = None
        # the mouse pointer shows what a left-drag does (rotate / roll / turn icon, a cross for tools) and turns
        # into VTK's hand / resize arrows over the crop box; QVTK would otherwise reset it to a plain arrow
        self._cursors = {}
        self._hover_xy, self._held = None, None
        self.vtk_widget.ShowCursor = self._show_cursor
        for ev, fn in (("MouseMoveEvent", self._hover_move), ("LeftButtonPressEvent", self._hover_press),
                       ("LeftButtonReleaseEvent", self._hover_release)):
            self.iren.GetCommand(self.iren.AddObserver(ev, fn, 10.0)).SetPassiveObserver(1)
        self._r_obs = {}
        for ev, fn in (("LeftButtonPressEvent", self._r_press), ("MouseMoveEvent", self._r_move),
                       ("LeftButtonReleaseEvent", self._r_release)):
            self._r_obs[ev] = self.iren.AddObserver(ev, fn, 8.5)

    # ------------------------------------------------------ rotate modes
    def _box_state(self, x, y):
        """vtkBoxRepresentation interaction state under (x, y): 0 = outside, 1–6 = face handle, 7 = centre handle
        (move), 8 = on a face (rotate the box)."""
        bw = self.box_widget
        if bw is None or not bw.GetEnabled():
            return 0
        return bw.GetRepresentation().ComputeInteractionState(int(x), int(y), 0)

    def _on_crop_box(self, x, y):
        """True when (x, y) is on the visible crop box (a face, handle or its outline)."""
        return self._box_state(x, y) != 0

    def _r_press(self, *_):
        if (self.rotate_mode == "free" or self.scalpel is not None or self.state is None
                or self.iren.GetShiftKey() or self.iren.GetControlKey()):
            return                                  # Shift / Ctrl keep the trackball's pan / spin
        if self._on_crop_box(*self.iren.GetEventPosition()):
            return                                  # the crop box always gets its own drags, in every mode
        self._rot_last = self.iren.GetEventPosition()
        self.rw.SetDesiredUpdateRate(self.iren.GetDesiredUpdateRate())     # interactive quality while dragging
        self.iren.GetCommand(self._r_obs["LeftButtonPressEvent"]).SetAbortFlag(1)

    def _r_move(self, *_):
        if self._rot_last is None:
            return
        x, y = self.iren.GetEventPosition()
        x0, y0 = self._rot_last
        self._rot_last = (x, y)
        if self.rotate_mode == "roll":
            w, h = self.rw.GetSize()
            d = math.degrees(math.atan2(y - h / 2, x - w / 2) - math.atan2(y0 - h / 2, x0 - w / 2))
            self.roll_view((d + 180) % 360 - 180)
        else:
            self.turn_view((x - x0) * 0.5)
        self.iren.GetCommand(self._r_obs["MouseMoveEvent"]).SetAbortFlag(1)

    def _r_release(self, *_):
        if self._rot_last is None:
            return
        self._rot_last = None
        self.rw.SetDesiredUpdateRate(self.iren.GetStillUpdateRate())       # full quality again
        self.iren.GetCommand(self._r_obs["LeftButtonReleaseEvent"]).SetAbortFlag(1)
        self.render()

    def set_rotate_mode(self, mode):
        self.rotate_mode = mode
        self.update_cursor()

    def _mode_cursor(self):
        if self.scalpel is not None:
            return QCursor(Qt.CrossCursor)
        name = {"free": "orbit", "roll": "roll", "turn": "turn"}.get(self.rotate_mode, "orbit")
        if name not in self._cursors:
            from .icons import icon
            pm = QPixmap(30, 30)
            pm.fill(Qt.transparent)
            p = QPainter(pm)
            p.setRenderHint(QPainter.Antialiasing)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(0, 0, 0, 150))            # dark disc: visible on white bone and on black background
            p.drawEllipse(1, 1, 28, 28)
            p.drawPixmap(4, 4, icon(name).pixmap(22, 22))
            p.end()
            self._cursors[name] = QCursor(pm, 15, 15)
        return self._cursors[name]

    def _hover_cursor(self, x, y):
        """Qt cursor for what a left-press at (x, y) grabs instead of rotating, or None. VTK's box widget only sets
        a cursor while it is dragged, so hovering the crop box or a caption used to show the rotate icon."""
        if self.state is None or (self.scalpel is not None and not self._measuring()):
            return None
        if self._hit_label(x, y) is not None:
            return Qt.OpenHandCursor
        if self._hit_handle(x, y):
            return Qt.PointingHandCursor
        if self.scalpel is not None:                    # a measuring tool takes the press before the crop box
            return None
        s = self._box_state(x, y)
        if 1 <= s <= 6:                                 # face handle: resize arrows along that face's normal
            pd = vtk.vtkPolyData()
            self.box_widget.GetRepresentation().GetPolyData(pd)
            a, b = (self._px(np.array(pd.GetPoint(i))) for i in (14, 8 + s - 1))   # box centre, face centre
            if np.hypot(*(b - a)) < 10:                 # face turned to the viewer: no screen direction
                return Qt.SizeAllCursor
            ang = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 180          # display y points up
            return (Qt.SizeHorCursor if ang < 22.5 or ang >= 157.5 else Qt.SizeBDiagCursor if ang < 67.5
                    else Qt.SizeVerCursor if ang < 112.5 else Qt.SizeFDiagCursor)
        return {7: Qt.SizeAllCursor, 8: Qt.OpenHandCursor}.get(s)

    def _hover_move(self, *_):
        self._hover_xy = self.iren.GetEventPosition()
        if self._held is None:          # no picking while a drag is under way (rotating stays smooth)
            self._show_cursor()

    def _hover_press(self, *_):
        c = self._hover_cursor(*self.iren.GetEventPosition())
        self._held = Qt.ClosedHandCursor if c in (Qt.OpenHandCursor, Qt.PointingHandCursor) else c or "mode"
        self._show_cursor()

    def _hover_release(self, *_):
        self._held = None
        self._show_cursor()

    def _show_cursor(self):
        if self._held == "mode":
            c = None
        else:
            c = self._held or (self._hover_cursor(*self._hover_xy) if self._hover_xy else None)
        if c is not None:
            self.vtk_widget.setCursor(c)
            return
        vc = self.rw.GetCurrentCursor()
        if vc in (0, 1):                                # VTK default / arrow -> what a drag does here
            self.vtk_widget.setCursor(self._mode_cursor())
        else:                                           # VTK's own cursors (e.g. while the box is dragged)
            self.vtk_widget.setCursor(self.vtk_widget._CURSOR_MAP.get(vc, Qt.ArrowCursor))

    def update_cursor(self):
        self._show_cursor()

    def roll_view(self, deg):
        """Rotate the picture in the screen plane by `deg` (counter-clockwise on screen, follows the mouse)."""
        cam = self.renderer.GetActiveCamera()
        cam.Roll(-deg)
        self.renderer.ResetCameraClippingRange()
        self.rw.Render()

    def turn_view(self, deg):
        """Turn the volume about the patient's head-foot axis (through the focal point) by `deg`."""
        if self.vol is None:
            return
        cam = self.renderer.GetActiveCamera()
        axis = np.asarray(self.vol.direction, float).T @ np.array([0.0, 0.0, 1.0])    # patient S in image frame
        fp = np.array(cam.GetFocalPoint())
        t = vtk.vtkTransform()
        t.Translate(*fp)
        t.RotateWXYZ(-deg, *axis)
        t.Translate(*(-fp))
        cam.SetPosition(t.TransformPoint(cam.GetPosition()))
        cam.SetViewUp(t.TransformVector(cam.GetViewUp()))
        cam.OrthogonalizeViewUp()
        self.renderer.ResetCameraClippingRange()
        self.rw.Render()

    def _swallow_char(self, *_):
        self.iren.GetCommand(self._char_obs).SetAbortFlag(1)

    def _add_orientation_cube(self):
        cube = vtk.vtkAnnotatedCubeActor()
        cube.SetXPlusFaceText("L")
        cube.SetXMinusFaceText("R")
        cube.SetYPlusFaceText("P")
        cube.SetYMinusFaceText("A")
        cube.SetZPlusFaceText("S")
        cube.SetZMinusFaceText("I")
        cube.SetFaceTextScale(0.6)
        cube.GetCubeProperty().SetColor(0.25, 0.3, 0.4)
        cube.GetTextEdgesProperty().SetColor(0.6, 0.6, 0.6)
        for prop in (cube.GetXPlusFaceProperty(), cube.GetXMinusFaceProperty()):
            prop.SetColor(1, 0.6, 0.3)
        for prop in (cube.GetYPlusFaceProperty(), cube.GetYMinusFaceProperty()):
            prop.SetColor(0.5, 1, 0.5)
        for prop in (cube.GetZPlusFaceProperty(), cube.GetZMinusFaceProperty()):
            prop.SetColor(0.5, 0.8, 1)
        self.orient_widget = vtk.vtkOrientationMarkerWidget()
        self.orient_widget.SetOrientationMarker(cube)
        self.orient_widget.SetInteractor(self.iren)
        self.orient_widget.SetViewport(0.85, 0.0, 1.0, 0.15)
        self.orient_widget.SetEnabled(1)
        self.orient_widget.InteractiveOff()

    # =================================================================== data
    @property
    def vol(self):
        return self.state.vol if self.state else None

    def set_state(self, state):
        self.clear()
        self.state = state
        if state.render_data is None:
            state.update_render_data()
        vol = state.vol
        self.image = np_to_vtk_image(state.render_data, state.render_spacing)
        self._render_version = state.version
        self.mapper = vtk.vtkGPUVolumeRayCastMapper()
        self.mapper.SetInputData(self.image)
        self.mapper.SetUseJittering(True)        # hilangkan artefak garis-garis (wood grain)
        self.mapper.SetAutoAdjustSampleDistances(True)
        self.prop = vtk.vtkVolumeProperty()
        self.prop.SetInterpolationTypeToLinear()
        self.prop.SetScalarOpacityUnitDistance(min(vol.spacing))
        self.prop.SetColor(self.ctf)
        self.prop.SetScalarOpacity(self.otf)
        self.volume = vtk.vtkVolume()
        self.volume.SetMapper(self.mapper)
        self.volume.SetProperty(self.prop)
        # VTK puts the gradient on the 2D table's y axis as gradient / (0.25 * data range). With the bone lock the
        # range is ~13 600 HU, so one of 128 rows spanned ~27 HU and rim thresholds of 30–300 HU looked the same.
        # A fixed 0..RIM_GRAD_MAX HU axis (one line of VTK 9.7's computeGradient replaced) gives ~2.5 HU per row.
        sp = self.volume.GetShaderProperty()
        sp.AddFragmentShaderReplacement("grad_mag = grad_mag / (0.25 * range);", False,
                                        "grad_mag = grad_mag / in_rimGradMax;", False)
        sp.GetFragmentCustomUniforms().SetUniformf("in_rimGradMax", RIM_GRAD_MAX)
        self.renderer.AddVolume(self.volume)
        self._text.SetInput(vol.description)
        # orientation cube: patient (LPS) labels rotated into the image frame (D^T)
        D = np.asarray(vol.direction, float)
        M = vtk.vtkMatrix4x4()
        for i in range(3):
            for j in range(3):
                M.SetElement(i, j, D[j, i])
        self.orient_widget.GetOrientationMarker().SetUserMatrix(M)
        self.sync_segments()
        self.view("anterior")

    def clear(self):
        self.set_scalpel(None)
        if self.volume is not None:
            self.renderer.RemoveVolume(self.volume)
        for actor, _ in self._seg_actors.values():
            self.renderer.RemoveActor(actor)
        self._seg_actors = {}
        if self.box_widget is not None:
            self.box_widget.Off()
            self.box_widget = None
        self.crop_on = False
        self._planes = None
        self.volume = self.image = self.mapper = None
        self.state = None
        self.render()

    def refresh_data(self, render=True):
        """Called after the scalpel / bone-edge clean-up changed state.render_data.
        render=False when the transfer function changes too (bone lock): the caller renders after apply_tf, so no
        frame shows the new data with the old table (bone briefly vanished: a flash of the inside of the skull)."""
        if self.image is not None and self.state.version != self._render_version:
            if self.image.GetDimensions()[::-1] != self.state.render_data.shape:   # resolution changed
                self.image = np_to_vtk_image(self.state.render_data, self.state.render_spacing)
                self.mapper.SetInputData(self.image)
            # the buffer is rewritten in place: mark the array too, or VTK keeps its cached (finite) range, which the
            # GPU mapper uses to place values on the 2D transfer function (stale range = everything in one column)
            self.image.GetPointData().GetScalars().Modified()
            self.image.Modified()
            self._render_version = self.state.version
            if render:
                self.render()

    # ===================================================== transfer function
    def apply_tf(self, tf, gain=1.0, blend="composite", quality=1.0, fast=False):
        """fast: while a curve is being dragged — a narrower 2D table (its upload, not the maths, is what costs:
        ~80 ms per frame at full width vs ~25 ms); the caller re-applies at full resolution when the drag pauses."""
        if self.vol is None:
            return
        lo, hi = self.vol.value_range
        bone = self.state.bone_encoded
        vr = blend == "composite"
        # One regular ~1 HU grid over the whole stored range: [lo, hi] = non-bone curves; with the bone lock the bone
        # area is stored mirrored as -OFF - HU (bone curves only), with a transparent gap between the two domains.
        # Sent to VTK in one call each: adding thousands of points one by one cost ~0.25 s per curve drag step.
        x0, x1 = (-BONE_OFFSET - hi if bone else lo), hi
        n = int(np.clip(x1 - x0 + 1, 1024, 32768))
        xs = np.linspace(x0, x1, n)
        alpha = np.zeros(n)
        rgb = np.zeros((n, 3))
        domains = [(xs >= lo, xs, "normal" if bone else None)]
        if bone:
            domains.append((xs <= -BONE_OFFSET - lo, -BONE_OFFSET - xs, "bone"))
        for sel, hu, layer in domains:
            a, c = combine(tf.curve_samples(hu[sel], layer), int(sel.sum()), fill_black=not vr)
            alpha[sel] = np.clip(a * gain, 0, 1)
            rgb[sel] = c
        self.ctf.RemoveAllPoints()
        self.otf.RemoveAllPoints()
        self.ctf.BuildFunctionFromTable(float(x0), float(x1), n, np.ascontiguousarray(rgb.ravel()))
        self.otf.BuildFunctionFromTable(float(x0), float(x1), n, np.ascontiguousarray(alpha))
        if vr and any(c.hide_edges for c in tf.curves):
            self.prop.SetTransferFunction2D(self._tf2d(tf, gain, lo, bone, 2048 if fast else 8192))
            self.prop.SetTransferFunctionModeTo2D()
        else:
            self.prop.SetTransferFunctionModeTo1D()
        if tf.shading and blend == "composite":
            self.prop.ShadeOn()
        else:
            self.prop.ShadeOff()
        self.prop.SetAmbient(tf.ambient)
        self.prop.SetDiffuse(tf.diffuse)
        self.prop.SetSpecular(tf.specular)
        self.prop.SetSpecularPower(tf.specular_power)
        {"composite": self.mapper.SetBlendModeToComposite,
         "mip": self.mapper.SetBlendModeToMaximumIntensity,
         "minip": self.mapper.SetBlendModeToMinimumIntensity}[blend]()
        self.mapper.SetSampleDistance(min(self.vol.spacing) * 0.5 / max(0.1, quality))
        self.renderer.SetBackground(*tf.background)
        mode = {"composite": "VR", "mip": "MIP", "minip": "MinIP"}[blend]
        self._text.SetInput(f"{self.vol.description}   [{mode}]")
        self.render()

    def _tf2d(self, tf, gain, lo, bone, max_width=8192):
        """2D transfer function (x = voxel value, y = gradient magnitude) for curves with hide_edges: they fade out
        where the gradient is above tf.edge_threshold, i.e. on the partial-volume rim of brighter structures.
        Axes as in VTK 9.7's shader: x = (value - min) / range of the rendered data; y = central-difference gradient
        (aspect-corrected, in value units) / RIM_GRAD_MAX (our shader replacement), clamped to 0..1."""
        rmin, rmax = self.image.GetScalarRange()
        rng = max(rmax - rmin, 1.0)
        W, H = int(np.clip(rng / 2, 1024, max_width)), 128
        xs = rmin + (np.arange(W) + 0.5) / W * rng
        grad = (np.arange(H) + 0.5) / H * RIM_GRAD_MAX
        t = np.clip((grad - 0.6 * tf.edge_threshold) / (0.8 * tf.edge_threshold), 0, 1)
        keep = 1.0 - t * t * (3 - 2 * t)                          # smoothstep from 0.6·T to 1.4·T
        rgba = np.zeros((H, W, 4), np.float32)
        domains = [(xs >= lo, xs, "normal" if bone else None)]
        if bone:
            domains.append((xs <= -BONE_OFFSET - lo, -BONE_OFFSET - xs, "bone"))
        # only rows in the threshold's transition differ: below it every row keeps the curves, above it every row
        # hides them -> merge once per distinct factor and copy (keeps curve dragging smooth)
        levels, row_of = np.unique(np.round(keep, 3), return_inverse=True)
        for sel, hu, layer in domains:
            samples = tf.curve_samples(hu[sel], layer)
            m = int(sel.sum())
            if not m or not samples:
                continue
            _, base_rgb = combine(samples, m, fill_black=False)        # color where every curve has faded out
            block = np.zeros((len(levels), m, 4), np.float32)
            for k, factor in enumerate(levels):                       # same merge as combine(), per factor
                a, rgb = combine(samples, m, fill_black=True, keep=factor)
                rgb[a <= 1e-9] = base_rgb[a <= 1e-9]
                block[k, :, :3] = rgb
                block[k, :, 3] = np.clip(a * gain, 0, 1)
            rgba[:, sel, :] = block[row_of]
        img = vtk.vtkImageData()
        img.SetDimensions(W, H, 1)
        img.GetPointData().SetScalars(numpy_support.numpy_to_vtk(rgba.reshape(-1, 4), deep=True,
                                                                 array_type=vtk.VTK_FLOAT))
        self._tf2d_image = img
        return img

    def set_volume_visible(self, v):
        if self.volume is not None:
            self.volume.SetVisibility(bool(v))
            self.render()

    # ================================================================ kamera
    def view(self, name):
        if self.image is None:
            return
        cam = self.renderer.GetActiveCamera()
        b = self.image.GetBounds()
        c = [(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2]
        dirs = {
            "anterior": ((0, -1, 0), (0, 0, 1)), "posterior": ((0, 1, 0), (0, 0, 1)),
            "right": ((-1, 0, 0), (0, 0, 1)), "left": ((1, 0, 0), (0, 0, 1)),
            "superior": ((0, 0, 1), (0, -1, 0)), "inferior": ((0, 0, -1), (0, -1, 0)),
        }
        d, up = dirs[name]
        # directions are patient (LPS); convert to the image frame for non-axial acquisitions
        D = np.asarray(self.vol.direction, float)
        d, up = D.T @ np.array(d, float), D.T @ np.array(up, float)
        dist = max(b[1] - b[0], b[3] - b[2], b[5] - b[4]) * 2.5
        cam.SetFocalPoint(*c)
        cam.SetPosition(c[0] + d[0] * dist, c[1] + d[1] * dist, c[2] + d[2] * dist)
        cam.SetViewUp(*up)
        self.renderer.ResetCamera()
        self.renderer.ResetCameraClippingRange()
        self.render()

    def zoom(self, factor):
        cam = self.renderer.GetActiveCamera()
        cam.Dolly(factor)
        self.renderer.ResetCameraClippingRange()
        self.render()

    def fit(self):
        self.renderer.ResetCamera()
        self.render()

    def rotate(self, deg):
        self.renderer.GetActiveCamera().Azimuth(deg)
        self.renderer.ResetCameraClippingRange()
        self.render()

    # ================================================================== crop
    def _ensure_box(self):
        if self.box_widget is None and self.image is not None:
            rep = vtk.vtkBoxRepresentation()
            rep.SetPlaceFactor(1.0)
            rep.PlaceWidget(self.image.GetBounds())
            rep.InsideOutOn()
            rep.GetOutlineProperty().SetColor(1, 0.8, 0.2)
            self.box_widget = vtk.vtkBoxWidget2()
            self.box_widget.SetInteractor(self.iren)
            self.box_widget.SetRepresentation(rep)
            self.box_widget.AddObserver("InteractionEvent", self._on_box)
        return self.box_widget

    def set_crop(self, on):
        if self.image is None:
            return
        self.crop_on = bool(on)
        if on:
            self._ensure_box()
            self.box_widget.SetEnabled(1 if self.crop_handles else 0)
            self._on_box(self.box_widget, None)
        else:
            if self.box_widget is not None:
                self.box_widget.Off()
            self._planes = None
            self._apply_planes()
        self.render()

    def set_crop_handles(self, show):
        self.crop_handles = bool(show)
        if self.box_widget is not None:
            self.box_widget.SetEnabled(1 if (show and self.crop_on) else 0)
        self.render()

    def reset_crop(self):
        if self.box_widget is not None:
            self.box_widget.GetRepresentation().PlaceWidget(self.image.GetBounds())
            if self.crop_on:
                self._on_box(self.box_widget, None)
        self.fit()

    def crop_bounds_ijk(self):
        """Crop box in voxel indices (z0, z1, y0, y1, x0, x1), or None."""
        if not self.crop_on or self.box_widget is None:
            return None
        b = self.box_widget.GetRepresentation().GetBounds()
        sz, sy, sx = self.vol.spacing
        nz, ny, nx = self.vol.shape
        x0, x1 = int(max(0, b[0] / sx)), int(min(nx, b[1] / sx + 1))
        y0, y1 = int(max(0, b[2] / sy)), int(min(ny, b[3] / sy + 1))
        z0, z1 = int(max(0, b[4] / sz)), int(min(nz, b[5] / sz + 1))
        return z0, z1, y0, y1, x0, x1

    def _on_box(self, widget, _):
        planes = vtk.vtkPlanes()
        widget.GetRepresentation().GetPlanes(planes)
        self._planes = planes
        self._apply_planes()

    def _apply_planes(self):
        mappers = [self.mapper] + [a.GetMapper() for a, _ in self._seg_actors.values()]
        for m in mappers:
            if m is None:
                continue
            if self._planes is None:
                m.RemoveAllClippingPlanes()
            else:
                m.SetClippingPlanes(self._planes)

    # =============================================================== segmen
    def sync_segments(self):
        if self.state is None:
            return
        live = {s.id: s for s in self.state.segments}
        for sid in list(self._seg_actors):
            if sid not in live:
                self.renderer.RemoveActor(self._seg_actors.pop(sid)[0])
        for sid, seg in live.items():
            entry = self._seg_actors.get(sid)
            if entry is None or entry[1] != seg.version:
                if entry is not None:
                    self.renderer.RemoveActor(entry[0])
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(seg.polydata)
                mapper.ScalarVisibilityOff()
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetSpecular(0.3)
                actor.GetProperty().SetSpecularPower(20)
                self.renderer.AddActor(actor)
                self._seg_actors[sid] = (actor, seg.version)
            actor = self._seg_actors[sid][0]
            actor.SetVisibility(seg.visible)
            actor.GetProperty().SetColor(*seg.color)
            actor.GetProperty().SetOpacity(seg.opacity)
        self._apply_planes()
        self.render()

    # ========================================================= color inspect
    def pick_voxel(self, x, y):
        """First visible voxel (z, y, x) at screen position (x, y), or None."""
        picker = vtk.vtkVolumePicker()
        picker.SetTolerance(1e-6)
        picker.SetVolumeOpacityIsovalue(0.15)
        picker.PickCroppingPlanesOff()
        if not picker.Pick(x, y, 0, self.renderer) or picker.GetVolume() is None:
            return None
        px, py, pz = picker.GetPickPosition()
        sz, sy, sx = self.vol.spacing
        nz, ny, nx = self.vol.shape
        return (int(np.clip(round(pz / sz), 0, nz - 1)), int(np.clip(round(py / sy), 0, ny - 1)),
                int(np.clip(round(px / sx), 0, nx - 1)))

    # ======================================================= pengukuran 3D
    def pick_world(self, x, y):
        picker = vtk.vtkVolumePicker()
        picker.SetTolerance(1e-6)
        picker.SetVolumeOpacityIsovalue(0.15)
        picker.PickCroppingPlanesOff()
        for actor, _ in self._seg_actors.values():
            picker.AddPickList(actor)
        if self.volume is not None:
            picker.AddPickList(self.volume)
        picker.PickFromListOn()
        if not picker.Pick(x, y, 0, self.renderer):
            return None
        return np.array(picker.GetPickPosition())

    # ------------------------------------------------------- drawing actors
    def _sphere(self, p, r, col):
        s = vtk.vtkSphereSource()
        s.SetCenter(*p)
        s.SetRadius(r)
        m = vtk.vtkPolyDataMapper()
        m.SetInputConnection(s.GetOutputPort())
        a = vtk.vtkActor()
        a.SetMapper(m)
        a.GetProperty().SetColor(*col)
        return a

    def _polyline(self, pts, col):
        line = vtk.vtkPolyLineSource()
        line.SetNumberOfPoints(len(pts))
        for i, p in enumerate(pts):
            line.SetPoint(i, *p)
        m = vtk.vtkPolyDataMapper()
        m.SetInputConnection(line.GetOutputPort())
        a = vtk.vtkActor()
        a.SetMapper(m)
        a.GetProperty().SetColor(*col)
        a.GetProperty().SetLineWidth(3)
        return a

    def _build(self, m, col=None):
        """(Re)create the actors of measurement m."""
        for a in m.actors:
            self.overlay.RemoveActor(a)
        m.actors = []
        col = col or (COL_SELECTED if m is self._sel else COL_NORMAL)
        P = m.pts
        r = max(0.6, min(self.vol.spacing) * 1.5)

        def add(a):
            self.overlay.AddActor(a)
            m.actors.append(a)

        for i, p in enumerate(P):
            if not (m.kind == "arrow" and i == 0):          # the arrow tip gets a cone instead
                add(self._sphere(p, r, col))
        for seg in ([P[0:2], P[2:4]] if m.kind == "cobb" else [P]):
            if len(seg) >= 2:
                add(self._polyline(seg, col))
        if m.kind == "arrow" and len(P) >= 2 and np.linalg.norm(P[1] - P[0]) > 1e-3:
            u = (P[0] - P[1]) / np.linalg.norm(P[1] - P[0])
            h = min(0.35 * np.linalg.norm(P[1] - P[0]), max(4.0, 6 * r))
            cone = vtk.vtkConeSource()
            cone.SetResolution(24)
            cone.SetHeight(h)
            cone.SetRadius(h * 0.35)
            cone.SetDirection(*u)
            cone.SetCenter(*(P[0] - u * h / 2))
            cm = vtk.vtkPolyDataMapper()
            cm.SetInputConnection(cone.GetOutputPort())
            ca = vtk.vtkActor()
            ca.SetMapper(cm)
            ca.GetProperty().SetColor(*col)
            add(ca)
        text = m.label
        m.text_actor = None
        if text:
            ta = vtk.vtkBillboardTextActor3D()
            ta.SetInput(text)
            ta.SetPosition(*m.anchor)
            ta.SetDisplayOffset(int(m.label_offset[0]), int(m.label_offset[1]))
            m.text_actor = ta
            tp = ta.GetTextProperty()
            tp.SetFontSize(18)
            tp.SetColor(*col)
            tp.SetBackgroundColor(0, 0, 0)
            tp.SetBackgroundOpacity(0.6)
            add(ta)

    def _remove(self, m):
        for a in m.actors:
            self.overlay.RemoveActor(a)
        m.actors = []

    # ------------------------------------------------------- public API
    def select_measurement(self, m, notify=True):
        old, self._sel = self._sel, m
        for x in {old, m} - {None}:
            if x in self.measurements3d:
                self._build(x)
        if notify:
            self.measureSelected.emit(self.measurements3d.index(m) if m in self.measurements3d else -1)
        self.render()

    def highlight_measurement3d(self, i):
        """Select by list index (from the Measurements tab)."""
        m = self.measurements3d[i] if 0 <= i < len(self.measurements3d) else None
        self.select_measurement(m, notify=False)

    def delete_measurement3d(self, i):
        if isinstance(i, Measure3D):
            i = self.measurements3d.index(i) if i in self.measurements3d else -1
        if 0 <= i < len(self.measurements3d):
            m = self.measurements3d.pop(i)
            self._remove(m)
            if m is self._sel:
                self._sel = None
            self.status.emit(f"Deleted: {m.name} {m.label}".strip())
            self.measurementsChanged.emit()
            self.render()

    def clear_measurements3d(self):
        for m in self.measurements3d:
            self._remove(m)
        self.measurements3d = []
        self._sel = None
        self._cancel_drawing()
        self.measurementsChanged.emit()
        self.render()

    def add_measurement3d(self, kind, pts, note=""):
        m = Measure3D(kind, pts, note)
        self.measurements3d.append(m)
        self._build(m)
        self.measurementsChanged.emit()
        self.select_measurement(m)
        self.status.emit(f"3D {m.name}: {m.label}")
        return m

    # ------------------------------------------------------- hit tests (screen pixels)
    def _px(self, p):
        self._coord.SetValue(*p)
        return np.array(self._coord.GetComputedDoubleDisplayValue(self.renderer))

    def _hit_handle(self, x, y, tol=9):
        best, bd = None, tol
        for m in self.measurements3d:
            for i, p in enumerate(m.pts):
                d = np.hypot(*(self._px(p) - (x, y)))
                if d < bd:
                    best, bd = (m, i), d
        return best

    def _hit_measure(self, x, y, tol=7):
        h = self._hit_handle(x, y)
        if h:
            return h[0]
        q = np.array([x, y], float)
        best, bd = None, tol
        for m in self.measurements3d:
            S = [self._px(p) for p in m.pts]
            segs = [(S[0], S[1]), (S[2], S[3])] if m.kind == "cobb" and len(S) >= 4 else list(zip(S, S[1:]))
            for a, b in segs:
                ab = b - a
                t = np.clip(np.dot(q - a, ab) / max(1e-9, np.dot(ab, ab)), 0, 1)
                d = np.linalg.norm(q - (a + t * ab))
                if d < bd:
                    best, bd = m, d
        lab = self._hit_label(x, y)
        return lab if lab is not None else best

    def _label_rect(self, m):
        """Screen rectangle (x0, y0, x1, y1) of a measurement's caption, or None."""
        ta = m.text_actor
        if ta is None:
            return None
        bb = [0, 0, 0, 0]
        if not self._text_renderer.GetBoundingBox(ta.GetTextProperty(), ta.GetInput(), bb, self.rw.GetDPI()):
            return None
        ax, ay = self._px(m.anchor)
        x0, y0 = ax + m.label_offset[0], ay + m.label_offset[1]
        return x0 + bb[0], y0 + bb[2], x0 + bb[1], y0 + bb[3]

    def _hit_label(self, x, y, pad=4):
        for m in reversed(self.measurements3d):
            r = self._label_rect(m)
            if r and r[0] - pad <= x <= r[2] + pad and r[1] - pad <= y <= r[3] + pad:
                return m
        return None

    # ------------------------------------------------------- mouse / keyboard
    def _abort(self, ev):
        self.iren.GetCommand(self._m_obs[ev]).SetAbortFlag(1)

    def _measuring(self):
        return self.scalpel in MEASURE3D

    def _active(self):
        return self.state is not None and (self.scalpel is None or self._measuring())

    def _cancel_drawing(self):
        if self._drawing is not None:
            self._remove(self._drawing)
        self._drawing, self._dragging = None, False
        self.render()

    def _finish_drawing(self):
        d = self._drawing
        self._remove(d)
        self._drawing, self._dragging = None, False
        pts = d.pts
        if d.kind == "openpoly":
            pts = [p for i, p in enumerate(pts[:-1]) if i == 0 or np.linalg.norm(p - pts[i - 1]) > 0.5]
            if len(pts) < 2:
                self.render()
                return
        self.add_measurement3d(d.kind, pts, d.note)
        self.toolFinished.emit()

    def _m_press(self, *_):
        if not self._active():
            return
        x, y = self.iren.GetEventPosition()
        d = self._drawing
        if d is not None:                         # next step of an angle / Cobb / polygon
            self._abort("LeftButtonPressEvent")
            p = self.pick_world(x, y)
            if p is None:
                self.status.emit("No visible surface at that point.")
                return
            if d.kind == "openpoly":
                if self.iren.GetRepeatCount():    # double-click ends the polygon
                    self._finish_drawing()
                    return
                d.pts[-1] = p
                d.pts.append(p.copy())
            elif d.kind == "angle":
                d.pts[2] = p
                self._finish_drawing()
                return
            elif d.kind == "cobb":                # start dragging the second line
                d.pts[2:] = [p, p.copy()]
                self._dragging = True
            self._build(d, COL_DRAWING)
            self.render()
            return
        lab = self._hit_label(x, y)
        if lab is not None:                       # drag the caption out of the way (like 2D)
            self._abort("LeftButtonPressEvent")
            self._label_drag = (lab, (x, y))
            self.select_measurement(lab)
            return
        hit = self._hit_handle(x, y)
        if hit:                                   # drag an existing point (like 2D)
            self._abort("LeftButtonPressEvent")
            self._edit = hit
            self.select_measurement(hit[0])
            return
        if self._measuring():
            self._abort("LeftButtonPressEvent")
            p = self.pick_world(x, y)
            if p is None:
                self.status.emit("No visible surface at that point.")
                return
            if self.scalpel == "text":
                from PySide6.QtWidgets import QInputDialog
                txt, ok = QInputDialog.getText(self, "3D text", "Text:")
                if ok and txt.strip():
                    self.add_measurement3d("text", [p], txt.strip())
                    self.toolFinished.emit()
                return
            self._drawing = Measure3D(self.scalpel, [p, p.copy()])
            self._dragging = True
            self._build(self._drawing, COL_DRAWING)
            self.render()
            return
        self._press_xy = (x, y)                   # rotate mode: a click without dragging selects

    def _m_move(self, *_):
        if self._label_drag is not None:
            self._abort("MouseMoveEvent")
            m, (lx, ly) = self._label_drag
            x, y = self.iren.GetEventPosition()
            m.label_offset = [m.label_offset[0] + x - lx, m.label_offset[1] + y - ly]
            self._label_drag = (m, (x, y))
            if m.text_actor is not None:
                m.text_actor.SetDisplayOffset(int(m.label_offset[0]), int(m.label_offset[1]))
            self.render()
            return
        if self._edit is None and self._drawing is None:
            return
        x, y = self.iren.GetEventPosition()
        if self._edit is not None:
            self._abort("MouseMoveEvent")
            p = self.pick_world(x, y)
            if p is not None:
                m, i = self._edit
                m.pts[i] = p
                self._build(m)
                self.render()
            return
        d = self._drawing
        if self._dragging or d.kind == "openpoly" or (d.kind == "angle" and len(d.pts) == 3):
            self._abort("MouseMoveEvent")
            p = self.pick_world(x, y)
            if p is not None:
                d.pts[-1] = p
                self._build(d, COL_DRAWING)
                self.render()

    def _m_release(self, *_):
        if self._label_drag is not None:
            self._abort("LeftButtonReleaseEvent")
            self._label_drag = None
            return
        if self._edit is not None:
            self._abort("LeftButtonReleaseEvent")
            self._edit = None
            self.measurementsChanged.emit()
            self.select_measurement(self._sel)
            return
        d = self._drawing
        if d is not None and self._dragging:
            self._abort("LeftButtonReleaseEvent")
            self._dragging = False
            moved = np.linalg.norm(d.pts[-1] - d.pts[-2]) >= 0.5
            if d.kind in ("length", "arrow"):
                if moved:
                    self._finish_drawing()
                else:
                    self._cancel_drawing()
                    self.status.emit(f"{d.name}: press on the surface and drag, then release.")
            elif d.kind == "angle":
                if moved:
                    d.pts.append(d.pts[-1].copy())
                    self.status.emit("Angle: now click the third point.")
                else:
                    self._cancel_drawing()
                    self.status.emit("Angle: drag the first arm (start → vertex), then click the third point.")
            elif d.kind == "cobb":
                if len(d.pts) == 2:
                    if moved:
                        self.status.emit("Cobb: now drag the second line.")
                    else:
                        self._cancel_drawing()
                        self.status.emit("Cobb: drag the first line, then drag the second line.")
                elif moved:
                    self._finish_drawing()
            elif d.kind == "openpoly" and moved:
                d.pts.append(d.pts[-1].copy())
            return

    def _m_click_select(self, *_):
        if self._press_xy is None:
            return
        (px, py), (x, y) = self._press_xy, self.iren.GetEventPosition()
        self._press_xy = None
        if abs(px - x) + abs(py - y) <= 3:
            self.select_measurement(self._hit_measure(x, y))

    def _m_rpress(self, *_):
        if not self._active():
            return
        if self._drawing is not None:             # right-click ends a polygon, cancels anything else
            self._abort("RightButtonPressEvent")
            if self._drawing.kind == "openpoly":
                self._finish_drawing()
            else:
                self._cancel_drawing()
            return
        self._rpress_xy = self.iren.GetEventPosition()

    def _m_rrelease(self, *_):
        if self._rpress_xy is None:
            return
        (px, py), (x, y) = self._rpress_xy, self.iren.GetEventPosition()
        self._rpress_xy = None
        if abs(px - x) + abs(py - y) > 3:         # it was a right-drag (zoom)
            return
        m = self._hit_measure(x, y)
        if m is None:
            return
        self.select_measurement(m)
        from PySide6.QtGui import QCursor
        from PySide6.QtWidgets import QMenu
        menu = QMenu(self)
        a_del = menu.addAction(f"Delete this measurement ({m.name} {m.label})".replace(" )", ")"))
        a_all = menu.addAction("Delete all 3D measurements")
        act = menu.exec(QCursor.pos())
        if act is a_del:
            self.delete_measurement3d(m)
        elif act is a_all:
            self.clear_measurements3d()

    def _m_key(self, *_):
        if not self._active():
            return
        key = self.iren.GetKeySym()
        if key in ("Delete", "BackSpace"):
            self._abort("KeyPressEvent")
            if self._drawing is not None:
                self._cancel_drawing()
            elif self._sel is not None:
                self.delete_measurement3d(self._sel)
            else:
                self.status.emit("Click a measurement first (it turns cyan), then press Delete.")
        elif key == "Escape":
            self._abort("KeyPressEvent")
            if self._drawing is not None:
                self._cancel_drawing()
            else:
                self.select_measurement(None)
        elif key in ("Return", "KP_Enter") and self._drawing is not None and self._drawing.kind == "openpoly":
            self._abort("KeyPressEvent")
            self._finish_drawing()

    def set_extra(self, key, actor):
        """Add / replace / remove (actor=None) an extra 3D object, e.g. a tractography bundle."""
        extras = self.__dict__.setdefault("_extras", {})
        old = extras.pop(key, None)
        if old is not None:
            self.renderer.RemoveActor(old)
        if actor is not None:
            extras[key] = actor
            self.renderer.AddActor(actor)
        self.render()

    # =============================================================== scalpel
    def set_scalpel(self, mode):
        """mode: None, 'cut' (remove inside the lasso), 'keep' (keep only inside it), 'inspect' (click = info)."""
        for o in self._lasso_obs:
            self.iren.RemoveObserver(o)
        self._lasso_obs = []
        self.scalpel = mode
        self._clear_lasso()
        if hasattr(self, "_cursors"):
            self.update_cursor()
        if mode is None:
            self.iren.SetInteractorStyle(self.style)
            return
        if mode in MEASURE3D:
            self._cancel_drawing()
            self.iren.SetInteractorStyle(self.style)       # right-drag = zoom, middle = pan still work
            how = {"length": "press on the surface and drag, release = done",
                   "angle": "drag the first arm (start → vertex), then click the third point",
                   "cobb": "drag the first line, then drag the second line",
                   "openpoly": "click points; double-click, Enter or right-click = done",
                   "arrow": "drag from the tip to the tail", "text": "click where the text goes"}[mode]
            self.status.emit(f"3D {MEASURE3D[mode][0]}: {how}.  Click a measurement + Delete = delete it.")
            return
        if mode == "inspect":
            def on_click(*_):
                x, y = self.iren.GetEventPosition()
                vox = self.pick_voxel(x, y)
                if vox is not None:
                    self.inspected.emit(self.state, vox)
            self._lasso_obs = [self.iren.AddObserver("LeftButtonPressEvent", on_click, 5.0)]
            self.iren.SetInteractorStyle(vtk.vtkInteractorStyleUser())
            self.status.emit("Inspect: click the 3D surface to see which curve colors it.")
            return
        # scalpel: click / drag to draw an area, then Cut inside / Cut outside / Clear area
        self.iren.SetInteractorStyle(self.style)          # right = zoom, middle = pan still work
        self._lasso_obs = [
            self.iren.AddObserver("LeftButtonPressEvent", self._lasso_press, 5.0),
            self.iren.AddObserver("MouseMoveEvent", self._lasso_move, 5.0),
            self.iren.AddObserver("LeftButtonReleaseEvent", self._lasso_release, 5.0),
            self.iren.AddObserver("KeyPressEvent", self._lasso_key, 5.0),
            self.iren.AddObserver("ConfigureEvent", lambda *_: self._layout_bar(), 5.0),
        ]
        self.status.emit("Scalpel: click points or drag to draw an area, then Cut inside (Del) / "
                         "Cut outside (Enter) / Clear area (Esc). Right-drag = zoom, middle = pan.")

    # --------------------------------------------------- scalpel drawing
    def _ensure_scalpel_actors(self):
        if getattr(self, "_fill_actor", None) is not None:
            return
        self._fill_pd = vtk.vtkPolyData()
        tri = vtk.vtkTriangleFilter()
        tri.SetInputData(self._fill_pd)
        m = vtk.vtkPolyDataMapper2D()
        m.SetInputConnection(tri.GetOutputPort())
        self._fill_actor = vtk.vtkActor2D()
        self._fill_actor.SetMapper(m)
        self._fill_actor.GetProperty().SetColor(0.0, 0.75, 0.0)
        self._fill_actor.GetProperty().SetOpacity(0.35)
        self.renderer.AddViewProp(self._fill_actor)
        self._lasso_actor.GetProperty().SetColor(0.1, 1.0, 0.1)
        self._bar = []
        for key, title, sub in (("inside", "Cut inside", "Del"), ("outside", "Cut outside", "Enter"),
                                ("clear", "Clear area", "Esc")):
            t = vtk.vtkTextActor()
            t.SetInput(f"{title}\n{sub}")
            tp = t.GetTextProperty()
            tp.SetFontSize(18)
            tp.SetJustificationToCentered()
            tp.SetColor(0.85, 0.85, 0.85)
            tp.SetBackgroundColor(0.35, 0.35, 0.35)
            tp.SetBackgroundOpacity(0.95)
            tp.SetFrame(True)
            tp.SetFrameColor(0.6, 0.6, 0.6)
            tp.SetBold(True)
            t.SetVisibility(False)
            self.renderer.AddViewProp(t)
            self._bar.append((key, t))

    def _layout_bar(self):
        if not getattr(self, "_bar", None):
            return
        w, h = self.rw.GetSize()
        n = len(self._bar)
        bw = 170
        x0 = w / 2 - bw * (n - 1) / 2
        for i, (_, t) in enumerate(self._bar):
            t.SetPosition(x0 + i * bw, 40)

    def _clear_lasso(self):
        self._lasso = []
        self._lasso_pts.Reset()
        self._lasso_lines.Reset()
        self._lasso_pd.Modified()
        if getattr(self, "_fill_actor", None) is not None:
            self._fill_pd.Initialize()
            self._fill_pd.Modified()
            for _, t in self._bar:
                t.SetVisibility(False)

    def _redraw_lasso(self):
        pts = self._lasso
        self._lasso_pts.Reset()
        for x, y in pts + pts[:1]:
            self._lasso_pts.InsertNextPoint(x, y, 0)
        self._lasso_lines.Reset()
        if len(pts) >= 2:
            self._lasso_lines.InsertNextCell(len(pts) + 1)
            for i in range(len(pts) + 1):
                self._lasso_lines.InsertCellPoint(i)
        self._lasso_pd.Modified()
        self._ensure_scalpel_actors()
        fp = vtk.vtkPoints()
        poly = vtk.vtkCellArray()
        if len(pts) >= 3:
            for x, y in pts:
                fp.InsertNextPoint(x, y, 0)
            poly.InsertNextCell(len(pts))
            for i in range(len(pts)):
                poly.InsertCellPoint(i)
        self._fill_pd.SetPoints(fp)
        self._fill_pd.SetPolys(poly)
        self._fill_pd.Modified()
        self._layout_bar()
        for _, t in self._bar:
            t.SetVisibility(len(pts) >= 3)
        self.render()

    def _bar_hit(self, x, y):
        for key, t in getattr(self, "_bar", []):
            if not t.GetVisibility():
                continue
            bb = [0.0] * 4
            t.GetBoundingBox(self.renderer, bb)          # relative to the actor's position
            px, py = t.GetPosition()
            if px + bb[0] - 4 <= x <= px + bb[1] + 4 and py + bb[2] - 4 <= y <= py + bb[3] + 4:
                return key
        return None

    def _abort_left(self):
        self.iren.GetCommand(self._lasso_obs[0]).SetAbortFlag(1)

    def _lasso_press(self, *_):
        self._abort_left()
        x, y = self.iren.GetEventPosition()
        hit = self._bar_hit(x, y)
        if hit:
            self.scalpel_action(hit)
            return
        self._lasso_down = True
        self._lasso.append((x, y))
        self._redraw_lasso()

    def _lasso_move(self, *_):
        if not self._lasso_down:
            return
        self.iren.GetCommand(self._lasso_obs[1]).SetAbortFlag(1)
        p = self.iren.GetEventPosition()
        if not self._lasso or abs(p[0] - self._lasso[-1][0]) + abs(p[1] - self._lasso[-1][1]) > 2:
            self._lasso.append(p)
            self._redraw_lasso()

    def _lasso_release(self, *_):
        if self._lasso_down:
            self.iren.GetCommand(self._lasso_obs[2]).SetAbortFlag(1)
        self._lasso_down = False

    def _lasso_key(self, *_):
        key = self.iren.GetKeySym()
        action = {"Delete": "inside", "Return": "outside", "KP_Enter": "outside", "Escape": "clear"}.get(key)
        if action:
            self.iren.GetCommand(self._lasso_obs[3]).SetAbortFlag(1)
            self.scalpel_action(action)

    def scalpel_action(self, action):
        poly = list(self._lasso)
        self._clear_lasso()
        if action != "clear" and len(poly) >= 3 and self.state is not None:
            self.status.emit("Cutting volume...")
            inside = self._project_inside(poly)
            self.state.apply_cut(inside if action == "inside" else ~inside)
            self.cutDone.emit(self.state)
            self.render()
            self.toolFinished.emit()      # after a cut the mouse rotates again
            return
        self.render()

    def _project_inside(self, poly):
        w, h = self.rw.GetSize()
        img = QImage(w, h, QImage.Format_Grayscale8)
        img.fill(0)
        p = QPainter(img)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(255, 255, 255))
        p.drawPolygon(QPolygonF([QPointF(x, h - 1 - y) for x, y in poly]))
        p.end()
        lasso = np.frombuffer(img.constBits(), np.uint8).reshape(h, img.bytesPerLine())[:, :w] > 127

        return project_inside(lasso, self.renderer, self.vol.shape, self.vol.spacing)

    # ================================================================== util
    def render(self):
        self.rw.Render()

    def shutdown(self):
        """Release the OpenGL context (driver memory) when the window closes."""
        self.clear()
        self.rw.Finalize()
        self.vtk_widget.close()

    def snapshot(self, path, scale=2):
        handles = self.box_widget is not None and self.box_widget.GetEnabled()
        if handles:
            self.box_widget.Off()
        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(self.rw)
        w2i.SetScale(scale)
        w2i.SetInputBufferTypeToRGB()
        w2i.ReadFrontBufferOff()
        w2i.Update()
        wr = vtk.vtkPNGWriter()
        wr.SetFileName(path)
        wr.SetInputConnection(w2i.GetOutputPort())
        wr.Write()
        if handles:
            self.box_widget.On()
        self.render()

    def export_segment_stl(self, seg, path):
        w = vtk.vtkSTLWriter()
        w.SetFileName(path)
        w.SetInputData(seg.polydata)
        w.Write()
