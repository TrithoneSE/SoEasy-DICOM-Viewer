"""GUI test: 3D measurements (drag length, angle, Cobb, open polygon, arrow, text), click-select + Delete,
dragging a point, tool auto-release, bone-lock marker. Run via tests\\run_tests.bat (args: app_dir data_dir out_dir)."""
import sys
import traceback

app_dir, data_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, app_dir)
import builtins  # noqa: E402
_print = builtins.print
builtins.print = lambda *a, **k: _print(*a, **{**k, "flush": True})
import numpy as np  # noqa: E402
from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QInputDialog, QMessageBox  # noqa: E402
from dcv.main_window import MainWindow  # noqa: E402

errors = []
sys.excepthook = lambda *a: (errors.append(a), traceback.print_exception(*a))
for _n in ("information", "warning", "question"):
    setattr(QMessageBox, _n, staticmethod(lambda *a, _n=_n, **k: (print("MSGBOX", _n, a[1:3]), QMessageBox.Yes)[1]))
QInputDialog.getText = staticmethod(lambda *a, **k: ("Lesion", True))
app = QApplication(sys.argv)
w = MainWindow()
w.show()
w.open_paths([data_dir])


def run():
    if w.viewports[0].state is None:
        QTimer.singleShot(300, run)
        return
    try:
        st = w.viewports[0].state
        before = st.vol.data.copy()
        vw = w.vr_window(st)
        vw.resize(1300, 850)
        app.processEvents()
        vr = vw.vr
        iren = vr.iren
        W, H = vr.rw.GetSize()
        print("3D size", W, H, "data unchanged after 3D:", np.array_equal(before, st.vol.data))

        def ev(name, x, y, repeat=0):
            iren.SetEventInformation(int(x), int(y), 0, 0, "\0", repeat)
            iren.InvokeEvent(name)

        def drag(x0, y0, x1, y1):
            ev("LeftButtonPressEvent", x0, y0)
            for t in np.linspace(0, 1, 6)[1:]:
                ev("MouseMoveEvent", x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
            ev("LeftButtonReleaseEvent", x1, y1)

        def click(x, y, repeat=0):
            ev("LeftButtonPressEvent", x, y, repeat)
            ev("LeftButtonReleaseEvent", x, y)

        def key(sym):
            iren.SetKeySym(sym)
            iren.InvokeEvent("KeyPressEvent")

        cx, cy = W // 2, H // 2
        # length by drag
        vw.select_measure("length")
        print("tool on:", vr.scalpel, vw.measure_btn.isChecked())
        drag(cx - 120, cy, cx + 120, cy + 20)
        print("length:", [(m.name, m.label) for m in vr.measurements3d],
              "| tool released:", vr.scalpel, vw.measure_btn.isChecked(), "| list:", vw.meas_list.count())
        # click without drag does not create anything
        vw.select_measure("length")
        click(cx, cy - 50)
        print("click-only measurements:", len(vr.measurements3d), "tool still on:", vr.scalpel)
        vw._release_tool()
        # select by clicking on the line in rotate mode, then Delete
        vr.select_measurement(None)
        mx, my = vr._px((vr.measurements3d[0].pts[0] + vr.measurements3d[0].pts[1]) / 2)
        click(mx, my)
        print("selected after click on line:", vr._sel is not None, "list row:", vw.meas_list.currentRow())
        key("Delete")
        print("after Delete:", len(vr.measurements3d), vw.meas_list.count())
        # angle: drag first arm, click third point
        vw.select_measure("angle")
        drag(cx - 120, cy + 60, cx, cy)
        ev("MouseMoveEvent", cx + 100, cy + 70)
        click(cx + 100, cy + 70)
        # cobb: two drags
        vw.select_measure("cobb")
        drag(cx - 130, cy - 80, cx - 20, cy - 60)
        drag(cx + 20, cy - 60, cx + 130, cy - 100)
        # open polygon: clicks + double-click
        vw.select_measure("openpoly")
        click(cx - 100, cy + 120)
        ev("MouseMoveEvent", cx - 20, cy + 140)
        click(cx - 20, cy + 140)
        click(cx + 60, cy + 120)
        click(cx + 60, cy + 120, repeat=1)
        # arrow + text
        vw.select_measure("arrow")
        drag(cx + 40, cy + 40, cx + 140, cy + 140)
        vw.select_measure("text")
        click(cx - 60, cy - 120)
        print("all:", [(m.name, m.label, len(m.pts)) for m in vr.measurements3d])
        print("tool after text:", vr.scalpel, vw.measure_btn.isChecked())
        # drag a point of the first measurement (edit like 2D)
        m0 = vr.measurements3d[0]
        lab0 = m0.label
        sx, sy = vr._px(m0.pts[0])
        drag(sx, sy, sx - 40, sy - 30)
        print("edited angle:", lab0, "->", m0.label)
        # right-click menu path: just hit-test
        print("hit at label of text:", vr._hit_measure(*vr._px(vr.measurements3d[-1].pts[0])) is not None)
        app.processEvents()
        vr.snapshot(out_dir + "\\v7_meas.png", scale=1)
        # scalpel releases after a cut
        vw.cut_act.setChecked(True)
        vr._lasso = [(cx - 300, cy - 300), (cx - 200, cy - 300), (cx - 200, cy - 200)]
        vr.scalpel_action("inside")
        print("scalpel after cut:", vr.scalpel, vw.cut_act.isChecked(),
              "| data unchanged after cut:", np.array_equal(before, st.vol.data))
        # bone marker in the editor
        print("bone marker:", vw.editor.bone_lock_hu)
        vw.editor.boneLockMoved.emit(400.0)
        app.processEvents()
        print("bone lock after drag:", st.bone_lock_hu, vw.editor.bone_lock_hu,
              "min render value:", int(st.render_data.min()))
        vw.grab().save(out_dir + "\\v7_window.png")
    except Exception:
        traceback.print_exc()
        errors.append(1)
    QTimer.singleShot(300, app.quit)


QTimer.singleShot(400, run)
app.exec()
print("errors:", len(errors))
