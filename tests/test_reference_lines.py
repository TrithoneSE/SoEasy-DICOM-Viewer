"""GUI test: reference lines between viewports (1 × 3 layout, MRI T1 axial / FLAIR sagittal / T1 coronal).
Each viewport shows where the slices of the others cut it, only where that slice has image (a partial slab gives a
short line); scrolling one viewport moves the lines in the others; dragging a line moves that viewport's slice;
series of another frame of reference, parallel planes and the View menu switch give no lines.
Run via tests/run_all.py (args: app_dir data_dir out_dir)."""
import os
import sys
import time
import traceback

app_dir, data_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, app_dir)
import builtins  # noqa: E402
_print = builtins.print
builtins.print = lambda *a, **k: _print(*a, **{**k, "flush": True})
import numpy as np  # noqa: E402
from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402
from dcv.main_window import MainWindow  # noqa: E402

errors = []
sys.excepthook = lambda *a: (errors.append(a), traceback.print_exception(*a))
for _n in ("information", "warning", "question"):
    setattr(QMessageBox, _n, staticmethod(lambda *a, _n=_n, **k: (print("MSGBOX", _n, a[1:3]), QMessageBox.Yes)[1]))


def check(ok, what):
    print(("OK   " if ok else "FAIL ") + what)
    if not ok:
        errors.append(what)


def wait_for(cond, timeout=60):
    t = time.time()
    while not cond() and time.time() - t < timeout:      # no QTest.qWait: it holds the GIL from the loader thread
        app.processEvents()
        time.sleep(0.05)
    return cond()


mri = os.path.join(app_dir, "samples", "mri")
app = QApplication(sys.argv)
w = MainWindow()
w.resize(1500, 700)
w.show()
w.open_paths([os.path.join(mri, "T1"), os.path.join(mri, "FLAIR")])
try:
    check(wait_for(lambda: len(w.series) == 2 and w.viewports[0].state is not None), "T1 + FLAIR opened")
    infos = {i.description: i for i in w.series.values()}
    t1 = next(i for d, i in infos.items() if "T1" in d.upper())
    flair = next(i for d, i in infos.items() if "FLAIR" in d.upper())
    w.set_grid(1, 3)
    ax, sag, cor = (w.viewports[k] for k in range(3))
    w._open_series_in(ax, t1)
    w._open_series_in(sag, flair)
    w._open_series_in(cor, t1)
    check(wait_for(lambda: all(v.state is not None for v in (ax, sag, cor))
                   and sag.state.vol.uid == flair.uid and cor.state.vol.uid == t1.uid), "three viewports loaded")
    cor.slice.set_orientation("coronal")
    app.processEvents()
    A, S, C = ax.slice, sag.slice, cor.slice
    check(A.plane_name() == "axial" and S.plane_name() == "sagittal" and C.plane_name() == "coronal",
          f"planes {A.plane_name()} / {S.plane_name()} / {C.plane_name()}")
    check(A.vol.frame_uid and A.vol.frame_uid == S.vol.frame_uid, "T1 and FLAIR share the frame of reference")

    def seg(view, src):
        return next(((a, b) for s, a, b in view.ref_segments() if s is src), None)

    def world(view, uv):
        return view.image_to_world(*uv)

    # every view shows the two others
    for v, name in ((A, "axial"), (S, "sagittal"), (C, "coronal")):
        srcs = {s.plane_name() for s, _, _ in v.ref_segments()}
        check(srcs == {"axial", "sagittal", "coronal"} - {name}, f"{name} view shows lines of {sorted(srcs)}")

    # the line lies on the source plane and inside the source slice
    ok = True
    for v in (A, S, C):
        for src, a, b in v.ref_segments():
            n = src.plane_axes_lps()[2]
            for uv in (a, b):
                p = world(v, uv)
                if abs(float(np.dot(p - src.plane_center_world(), n))) > 0.5:
                    ok = False
                idx = src.vol.world_to_index(p)
                if np.any(idx < -0.6) or np.any(idx > np.array(src.vol.shape) - 0.4):
                    ok = False
    check(ok, "every line lies on its source slice and within its image")

    # partial slab: the T1 coronal reformat covers only the T1 slab (60 x 1.5 mm) of the taller FLAIR image
    a, b = seg(S, C)
    length = float(np.linalg.norm(world(S, a) - world(S, b)))
    h_mm = S.raw_plane().shape[0] * S._spacing_rc()[0]
    check(abs(length - 90) < 3 and length < h_mm - 20,
          f"coronal T1 line in the FLAIR is cut to the T1 slab ({length:.0f} mm of {h_mm:.0f} mm)")

    # scrolling the axial view moves its line in the sagittal view by the same distance
    a0, _ = seg(S, A)
    z0 = A.world_pos()
    QTest.mouseMove(A, QPoint(A.width() // 2, A.height() // 2))
    for _ in range(10):
        A.wheelEvent(type("E", (), {"angleDelta": lambda self: QPoint(0, -120), "modifiers": lambda self: Qt.NoModifier,
                                    "position": lambda self: A.rect().center().toPointF()})())
    app.processEvents()
    dz = A.world_pos() - z0
    a1, _ = seg(S, A)
    moved = float(np.dot(world(S, a1) - world(S, a0), A.plane_axes_lps()[2]))
    check(abs(dz) > 10 and abs(moved - dz) < 0.5, f"scroll moves the line: slice {dz:+.1f} mm, line {moved:+.1f} mm")

    # drag the axial line in the sagittal view: the axial viewport follows the mouse
    a, b = seg(S, A)
    mid = S.image_to_widget((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    start = QPoint(int(mid.x()), int(mid.y()))
    end = start + QPoint(0, 60)
    S.mouseMoveEvent(type("E", (), {"position": lambda self: start.toPointF(), "modifiers": lambda self: Qt.NoModifier,
                                    "buttons": lambda self: Qt.NoButton})())
    check(S.cursor().shape() == Qt.SizeAllCursor, "pointer shows the move cursor over a line")
    idx0 = A.index
    QTest.mousePress(S, Qt.LeftButton, Qt.NoModifier, start)
    for t in np.linspace(0, 1, 6)[1:]:
        QTest.mouseMove(S, start + QPoint(0, int(60 * t)))
    QTest.mouseRelease(S, Qt.LeftButton, Qt.NoModifier, end)
    app.processEvents()
    r, c = S.widget_to_image(end.toPointF())
    want = float(np.dot(S.image_to_world(r, c), A.plane_axes_lps()[2]))
    check(A.index != idx0 and abs(A.world_pos() - want) <= A.axis_spacing(),
          f"dragging the line moves the axial slice ({idx0} -> {A.index}, at {A.world_pos():.1f} / {want:.1f} mm)")
    check(S._ref_drag is None, "drag released")

    # another frame of reference (other study) and parallel planes: no line
    keep = S.vol.frame_uid
    S.vol.frame_uid = "other"
    check(seg(A, S) is None and seg(S, A) is None, "no lines between different frames of reference")
    S.vol.frame_uid = keep
    C.set_orientation("axial")
    check(seg(A, C) is None, "no line between parallel (both axial) views")
    C.set_orientation("coronal")

    # View menu switch
    w.ref_act.trigger()
    check(not any(v.ref_segments() for v in (A, S, C)), "Reference lines off: none drawn")
    w.ref_act.trigger()
    check(all(v.ref_segments() for v in (A, S, C)), "Reference lines on again")
    check(w.ref_act.shortcut().toString() == "Ctrl+Shift+L", "shortcut Ctrl+Shift+L")
    w.grab().save(os.path.join(out_dir, "reference_lines.png"))
except Exception:
    traceback.print_exc()
    errors.append("exception")
print("errors:", len(errors))
