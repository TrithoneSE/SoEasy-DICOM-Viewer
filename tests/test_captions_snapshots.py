"""GUI test: series thumbnails (middle slice + DICOM window), draggable
measurement / text captions in 2D, MPR and 3D, busy panel for seed / opening 3D, inline Rotate / Roll / Turn and
crop buttons, MPR Roll tool, patient / study headers in the series list, 3D snapshots + movie.
Run via tests\\run_tests.bat (args: app_dir data_dir out_dir)."""
import math
import os
import sys
import traceback

app_dir, data_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, app_dir)
import builtins  # noqa: E402
_print = builtins.print
builtins.print = lambda *a, **k: _print(*a, **{**k, "flush": True})
import numpy as np  # noqa: E402
import pydicom  # noqa: E402
from PySide6.QtCore import QPointF, Qt, QTimer  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QInputDialog, QMessageBox  # noqa: E402
from dcv import busy as busy_mod  # noqa: E402
from dcv import snapshots  # noqa: E402
from dcv.loader import scan_folder  # noqa: E402
from dcv.main_window import MainWindow  # noqa: E402
from dcv.measure import Measurement  # noqa: E402

errors = []
sys.excepthook = lambda *a: (errors.append(a), traceback.print_exception(*a))
for _n in ("information", "warning", "question"):
    setattr(QMessageBox, _n, staticmethod(lambda *a, _n=_n, **k: (print("MSGBOX", _n, str(a[1:2])), QMessageBox.Yes)[1]))
QInputDialog.getText = staticmethod(lambda *a, **k: ("Lesion", True))


def check(ok, what):
    print(("OK   " if ok else "FAIL ") + what)
    if not ok:
        errors.append(what)


# busy panel: record what it showed
shown = []
_stage = busy_mod.BusyDialog.stage
busy_mod.BusyDialog.stage = lambda self, text: (shown.append((self.title.text(), text, self.isVisible())),
                                                 _stage(self, text))[1]

# 1) thumbnails: spatially middle slice, DICOM window
s0 = scan_folder(data_dir)[0]
order = sorted(s0.slice_keys)
ds = pydicom.dcmread(s0.middle_file(), stop_before_pixels=True)
from dcv.loader import _slice_key  # noqa: E402
check(abs(_slice_key(ds) - order[len(order) // 2]) < 1e-6, "thumbnail uses the spatially middle slice")
check(ds.get("WindowWidth") is not None, "sample has a stored DICOM window")

app = QApplication(sys.argv)
w = MainWindow()
w.resize(1500, 950)
w.show()
w.open_paths([data_dir])


def drag_widget(widget, a, b, steps=6):
    QTest.mousePress(widget, Qt.LeftButton, Qt.NoModifier, a.toPoint())
    for t in np.linspace(0, 1, steps)[1:]:
        p = QPointF(a.x() + (b.x() - a.x()) * t, a.y() + (b.y() - a.y()) * t).toPoint()
        QTest.mouseMove(widget, p)
    QTest.mouseRelease(widget, Qt.LeftButton, Qt.NoModifier, b.toPoint())


def label_drag_2d(view, what):
    st = view.state
    h, wdt = view.plane_shape() if hasattr(view, "plane_shape") else (100, 100)
    m = Measurement("length", [(h * 0.4, wdt * 0.4), (h * 0.4, wdt * 0.6)])
    view.measurements(create=True).append(m)
    view.repaint()
    app.processEvents()
    check(m._label_box is not None, f"{what}: caption painted")
    pts = list(m.pts)
    c = m._label_box.center()
    drag_widget(view, c, c + QPointF(60, 45))
    app.processEvents()
    check(m.pts == pts and math.hypot(*m.label_offset) > 40, f"{what}: caption dragged {m.label_offset}, "
                                                             "measurement unchanged")
    t = Measurement("text", [(h * 0.6, wdt * 0.5)], "Tumor")
    view.measurements(create=True).append(t)
    view.repaint()
    app.processEvents()
    c = t._label_box.center()
    drag_widget(view, c, c + QPointF(-50, 30))
    app.processEvents()
    check(math.hypot(*t.label_offset) > 40, f"{what}: text annotation moved {t.label_offset}")
    del st


def run():
    if w.viewports[0].state is None:
        QTimer.singleShot(300, run)
        return
    try:
        # 7) series list: patient + study header rows, thumbnails not headers
        lst = w.series_list
        kinds = [lst.item(i).data(Qt.UserRole + 1) for i in range(lst.count())]
        check(kinds[:2] == ["patient", "study"] and len(lst.series_items()) == len(w.series),
              f"series list grouped: {kinds}")
        # add a second study (MRI sample) to this window -> a new patient / study header
        w._ask_open_target = lambda new: "here"
        w.open_paths([os.path.join(app_dir, "samples", "mri")])
        n0 = len(w.series)
        import time
        for _ in range(1200):            # not QTest.qWait: PySide's qWait keeps the GIL and starves the scan thread
            app.processEvents()
            if len(w.series) > n0:
                break
            time.sleep(0.05)
        for _ in range(6):
            app.processEvents()
            time.sleep(0.05)
        kinds = [lst.item(i).data(Qt.UserRole + 1) for i in range(lst.count())]
        check(kinds.count("study") >= 2, f"second study gets its own header: {kinds}")
        lst.grab().save(os.path.join(out_dir, "series_list.png"))

        # 1/2) 2D captions
        sv = w.viewports[0].slice
        w.bank.assign("left", "wl")
        label_drag_2d(sv, "2D")

        # 6) MPR: caption drag + Roll tool
        w._set_active(w.viewports[0])
        w.open_mpr()
        app.processEvents()
        mpr = next(iter(w.mpr_windows.values()))
        mpr.resize(1400, 820)
        app.processEvents()
        ax = mpr.views["axial"]
        label_drag_2d(ax, "MPR")
        check("roll" in mpr.bank.buttons, "MPR has a Roll tool button")
        mpr.bank.assign("left", "roll")
        others = {k: [np.round(a, 4).tolist() for a in ax.m.axes(k)] for k in ("sagittal", "coronal")}
        c = ax._roll_center()
        r = 150
        p0 = QPointF(c.x() + r, c.y())
        p1 = QPointF(c.x(), c.y() + r)            # +90 deg on screen (y down = clockwise visually)
        # a world point to the right of the centre: where is it on screen before / after?
        a_vec = ax.m.axes("axial")[0]
        wp = ax.m.center + a_vec * 30
        s_before = ax.image_to_widget(*ax.world_to_image(wp))
        QTest.mousePress(ax, Qt.LeftButton, Qt.NoModifier, p0.toPoint())
        for t in np.linspace(0, math.pi / 2, 10)[1:]:
            QTest.mouseMove(ax, QPointF(c.x() + r * math.cos(t), c.y() + r * math.sin(t)).toPoint())
        QTest.mouseRelease(ax, Qt.LeftButton, Qt.NoModifier, p1.toPoint())
        app.processEvents()
        roll = math.degrees(ax.m.roll["axial"])
        check(abs((roll % 360) - 90) < 6 or abs((roll % 360) - 270) < 6, f"axial view rolled {roll:.0f} deg")
        s_after = ax.image_to_widget(*ax.world_to_image(wp))
        cc = ax._roll_center()
        ang = lambda q: math.degrees(math.atan2(q.y() - cc.y(), q.x() - cc.x()))  # noqa: E731
        moved = (ang(s_after) - ang(s_before) + 180) % 360 - 180
        check(abs(moved - 90) < 8, f"image turns with the mouse ({moved:.0f} deg, mouse +90)")
        same = all([np.round(a, 4).tolist() for a in ax.m.axes(k)] == others[k] for k in others)
        check(same, "Roll leaves the other two views unchanged")
        mpr.grab().save(os.path.join(out_dir, "mpr_roll.png"))
        mpr.bank.assign("left", "wl")

        # 3) seed with busy panel (3D opens + segment)
        st = w.viewports[0].state
        for vw0 in list(w.vr_windows.values()):
            vw0.close()
        app.processEvents()
        shown.clear()
        nz, ny, nx = st.vol.shape
        w._on_seed(st, nz // 2, ny // 2, nx // 2)
        app.processEvents()
        stages = [t for _, t, _ in shown]
        check(any("Preparing" in t for t in stages) and any("Growing" in t for t in stages),
              f"busy panel during seed: {stages}")
        vw = w.vr_windows.get(id(st))
        check(vw is not None, "seed opened the 3D window")

        # 4/5) toolbar: Rotate / Roll / Turn and crop buttons visible side by side
        from PySide6.QtWidgets import QToolBar
        tb = next(t for t in vw.findChildren(QToolBar) if t.objectName() == "tools3d")
        acts = tb.actions()
        check(all(vw.rot_acts[k][0] in acts for k in ("free", "roll", "turn")), "Rotate / Roll / Turn are buttons")
        check(vw.crop_act in acts and vw.handles_act in acts, "Crop and Box are buttons")
        vw.rot_acts["turn"][0].trigger()
        check(vw.vr.rotate_mode == "turn", "Turn button selects turn")
        vw.rot_acts["free"][0].trigger()
        vw.handles_act.trigger()
        check(not vw.handles_act.isChecked() and not vw.vr.crop_handles, "Box button hides the crop box")
        vw.handles_act.trigger()

        # 1) 3D caption drag
        vw.resize(1300, 850)
        app.processEvents()
        vr, iren = vw.vr, vw.vr.iren
        W, H = vr.rw.GetSize()
        cx, cy = W // 2, H // 2

        def ev(name, x, y):
            iren.SetEventInformation(int(x), int(y), 0, 0, "\0", 0)
            iren.InvokeEvent(name)

        # Roll / Turn must not steal drags from the crop box; the pointer shows the mode
        vr.view("anterior")
        vw.crop_act.setChecked(True)
        app.processEvents()
        rep = vr.box_widget.GetRepresentation()
        cam = vr.renderer.GetActiveCamera()
        for mode in ("roll", "turn"):
            vw.rot_acts[mode][0].trigger()
            app.processEvents()
            check(vr.vtk_widget.cursor().shape() == Qt.BitmapCursor, f"{mode}: pointer shows the mode icon")
            vr.reset_crop()
            vr.render()
            b0 = list(rep.GetBounds())
            from dcv import vtk_lite as vtk
            pd = vtk.vtkPolyData()
            rep.GetPolyData(pd)
            face = np.array(pd.GetPoint(9))          # +x face handle
            fx, fy = vr._px(face)
            check(vr._on_crop_box(fx, fy), f"{mode}: face handle detected under the mouse")
            up0, pos0 = cam.GetViewUp(), cam.GetPosition()
            ev("MouseMoveEvent", fx, fy)
            ev("LeftButtonPressEvent", fx, fy)
            for t in np.linspace(0, 1, 6)[1:]:
                ev("MouseMoveEvent", fx - 120 * t, fy)
            ev("LeftButtonReleaseEvent", fx - 120, fy)
            b1 = list(rep.GetBounds())
            check(np.allclose(up0, cam.GetViewUp()) and np.allclose(pos0, cam.GetPosition()),
                  f"{mode}: dragging the crop box does not move the camera")
            check(not np.allclose(b0, b1), f"{mode}: crop box face moved {np.round(b0[:2], 1)} -> "
                                           f"{np.round(b1[:2], 1)}")
            # away from the box the drag still rolls / turns
            ev("LeftButtonPressEvent", 40, 40)
            for t in np.linspace(0, 1, 6)[1:]:
                ev("MouseMoveEvent", 40 + 150 * t, 40 + 60 * t)
            ev("LeftButtonReleaseEvent", 190, 100)
            moved = not (np.allclose(up0, cam.GetViewUp()) and np.allclose(pos0, cam.GetPosition()))
            check(moved, f"{mode}: drag outside the box still {mode}s")
            vr.view("anterior")
        vw.rot_acts["free"][0].trigger()
        vr.reset_crop()
        vw.crop_act.setChecked(False)

        vw.select_measure("length")
        ev("LeftButtonPressEvent", cx - 100, cy)
        for t in np.linspace(0, 1, 5)[1:]:
            ev("MouseMoveEvent", cx - 100 + 200 * t, cy)
        ev("LeftButtonReleaseEvent", cx + 100, cy)
        m3 = vr.measurements3d[-1]
        r = vr._label_rect(m3)
        check(r is not None, "3D caption has a screen rectangle")
        pts = [p.copy() for p in m3.pts]
        lx, ly = (r[0] + r[2]) / 2, (r[1] + r[3]) / 2
        ev("LeftButtonPressEvent", lx, ly)
        for t in np.linspace(0, 1, 5)[1:]:
            ev("MouseMoveEvent", lx + 80 * t, ly + 60 * t)
        ev("LeftButtonReleaseEvent", lx + 80, ly + 60)
        check(all(np.allclose(a, b) for a, b in zip(pts, m3.pts)) and m3.label_offset[0] > 70,
              f"3D caption dragged {m3.label_offset}, measurement unchanged")
        check(tuple(m3.text_actor.GetDisplayOffset()) == (int(m3.label_offset[0]), int(m3.label_offset[1])),
              "3D caption drawn at the new offset")

        # 8) snapshots
        panel = vw.show_snapshots()
        vr.view("anterior")
        panel.add()
        vr.view("left")
        vw.crop_act.setChecked(True)
        panel.add()
        cam = vr.renderer.GetActiveCamera()
        pos_left = np.array(cam.GetPosition())
        panel.go(0)
        check(not np.allclose(pos_left, cam.GetPosition()) and not vr.crop_on, "snapshot 1 restores camera + crop off")
        panel.go(1)
        check(np.allclose(pos_left, cam.GetPosition(), atol=1e-3) and vr.crop_on, "snapshot 2 restores camera + crop")
        path = os.path.join(out_dir, "snaps.se3d")
        snapshots.save_file(path, panel.snaps, panel.movie)
        snaps, movie = snapshots.load_file(path)
        check(len(snaps) == 2 and snaps[1]["camera"] == panel.snaps[1]["camera"], "snapshot file round trip")
        gif = os.path.join(out_dir, "movie.gif")
        n = snapshots.make_movie(vw, panel.snaps, dict(frames=6, hold=1, fps=10, width=320, loop=False), gif)
        from PIL import Image
        with Image.open(gif) as im:            # Pillow merges identical consecutive frames (the pauses)
            total = 0
            for i in range(im.n_frames):
                im.seek(i)
                total += im.info.get("duration", 0)
        check(n == 8 and abs(total - n * 100) <= n * 2, f"movie GIF: {n} frames rendered, {total} ms long")
        vw.grab().save(os.path.join(out_dir, "vr.png"))
    except Exception:
        traceback.print_exc()
        errors.append("exception")
    QTimer.singleShot(300, app.quit)


QTimer.singleShot(400, run)
app.exec()
print("errors:", len(errors))
