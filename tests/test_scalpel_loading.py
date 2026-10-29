"""GUI test: loading overlay, scalpel by drag + Delete and by the on-screen Cut outside button, rotate/select
after a cut, undo. Run via tests\\run_tests.bat (args: app_dir data_dir out_dir)."""
import sys
import time
import traceback

app_dir, data_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, app_dir)
import builtins  # noqa: E402
_print = builtins.print
builtins.print = lambda *a, **k: _print(*a, **{**k, "flush": True})
import numpy as np  # noqa: E402
from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402
from dcv.main_window import MainWindow  # noqa: E402

errors = []
sys.excepthook = lambda *a: (errors.append(a), traceback.print_exception(*a))
for _n in ("information", "warning", "question"):
    setattr(QMessageBox, _n, staticmethod(lambda *a, _n=_n, **k: (print("MSGBOX", _n, a[1:3]), QMessageBox.Yes)[1]))
app = QApplication(sys.argv)
w = MainWindow()
w.resize(1400, 900)
w.show()
w.open_paths([data_dir])
shots = {"n": 0}


def snap_loading():
    if w.loading.isVisible() and shots["n"] < 1:
        shots["n"] += 1
        w.grab().save(out_dir + "\\v8_loading.png")
        print("loading overlay visible:", w.loading.title.text(), "|", w.loading.detail.text())
    if w.viewports[0].state is None:
        QTimer.singleShot(30, snap_loading)


QTimer.singleShot(5, snap_loading)


def run():
    if w.viewports[0].state is None:
        QTimer.singleShot(300, run)
        return
    try:
        print("overlay hidden after load:", not w.loading.isVisible())
        st = w.viewports[0].state
        vw = w.vr_window(st)
        vw.resize(1300, 850)
        app.processEvents()
        vr, iren = vw.vr, vw.vr.iren
        W, H = vr.rw.GetSize()
        cx, cy = W // 2, H // 2

        def ev(name, x, y, repeat=0):
            iren.SetEventInformation(int(x), int(y), 0, 0, "\0", repeat)
            iren.InvokeEvent(name)

        def key(sym):
            iren.SetKeySym(sym)
            iren.InvokeEvent("KeyPressEvent")

        def cam():
            return np.array(vr.renderer.GetActiveCamera().GetPosition())

        # a measurement to click on later
        vw.select_measure("length")
        ev("LeftButtonPressEvent", cx - 100, cy)
        for t in np.linspace(0, 1, 5)[1:]:
            ev("MouseMoveEvent", cx - 100 + 200 * t, cy)
        ev("LeftButtonReleaseEvent", cx + 100, cy)

        # scalpel exactly like the user: toolbar button, drag an area, Delete
        vw.cut_act.trigger()
        print("scalpel on:", vr.scalpel)
        pts = [(cx + 150 + 120 * np.cos(a), cy + 150 + 120 * np.sin(a)) for a in np.linspace(0, 6.2, 30)]
        ev("LeftButtonPressEvent", *pts[0])
        for p in pts[1:]:
            ev("MouseMoveEvent", *p)
        ev("LeftButtonReleaseEvent", *pts[-1])
        t0 = time.perf_counter()
        key("Delete")
        print(f"cut (Delete) took {time.perf_counter() - t0:.2f}s; removed voxels:",
              int(st.removed.sum()) if st.removed is not None else 0,
              "| tool:", vr.scalpel, vw.cut_act.isChecked())

        # after the cut: dragging must rotate, clicking must select
        c0 = cam()
        ev("LeftButtonPressEvent", cx, cy)
        for i in range(1, 8):
            ev("MouseMoveEvent", cx + 15 * i, cy + 5 * i)
        ev("LeftButtonReleaseEvent", cx + 105, cy + 35)
        print("rotates after cut:", not np.allclose(c0, cam()), "| errors so far:", len(errors))
        vr.select_measurement(None)
        m = vr.measurements3d[0]
        mx, my = vr._px((m.pts[0] + m.pts[1]) / 2)
        ev("LeftButtonPressEvent", mx, my)
        ev("LeftButtonReleaseEvent", mx, my)
        print("click selects after cut:", vr._sel is m)

        # scalpel by clicking the "Cut outside" bar button, then rotate again
        vw.cut_act.trigger()
        for p in [(cx - 300, cy - 300), (cx + 300, cy - 300), (cx + 300, cy + 300), (cx - 300, cy + 300)]:
            ev("LeftButtonPressEvent", *p)
            ev("LeftButtonReleaseEvent", *p)
        vr.render()
        bb = [0.0] * 4
        [t for k, t in vr._bar if k == "outside"][0].GetBoundingBox(vr.renderer, bb)
        ox, oy = [t for k, t in vr._bar if k == "outside"][0].GetPosition()
        bx, by = ox + (bb[0] + bb[1]) / 2, oy + (bb[2] + bb[3]) / 2
        n_before = int(st.removed.sum())
        print("bar hit test (expect outside):", vr._bar_hit(bx, by))
        t0 = time.perf_counter()
        ev("LeftButtonPressEvent", bx, by)
        ev("LeftButtonReleaseEvent", bx, by)
        print(f"cut outside via button took {time.perf_counter() - t0:.2f}s | tool:", vr.scalpel,
              "| removed before/after:", n_before, int(st.removed.sum()), "of", st.removed.size)
        c0 = cam()
        ev("LeftButtonPressEvent", cx, cy)
        for i in range(1, 8):
            ev("MouseMoveEvent", cx - 15 * i, cy)
        ev("LeftButtonReleaseEvent", cx - 105, cy)
        print("rotates after 2nd cut:", not np.allclose(c0, cam()))
        vw._undo_cut()
        vw._undo_cut()
        print("undo x2 -> removed voxels:", int(st.removed.sum()))
        vr.snapshot(out_dir + "\\v8_after.png", scale=1)
    except Exception:
        traceback.print_exc()
        errors.append(1)
    QTimer.singleShot(300, app.quit)


QTimer.singleShot(400, run)
app.exec()
print("errors:", len(errors))
