"""GUI test: (1) opening another study when one is open: new window vs add here (different-patient warning, default
button, cancel, added series do not replace the viewports, Ctrl+N); (2) curves with "hide partial-volume edges":
2D transfer function on/off, MIP stays 1D, edge threshold control, stale VTK range cache after bone lock (the bug
that made every voxel use the last table column), 2D without suppression renders like 1D.
Run via tests\\run_tests.bat (args: app_dir data_dir out_dir; samples/mri next to data_dir)."""
import os
import sys
import traceback

app_dir, data_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
mri_dir = os.path.join(os.path.dirname(data_dir.rstrip("\\/")), "mri")
sys.path.insert(0, app_dir)
import builtins  # noqa: E402
_print = builtins.print
builtins.print = lambda *a, **k: _print(*a, **{**k, "flush": True})
from dcv import vtk_lite as vtk  # noqa: E402
from vtkmodules.util import numpy_support  # noqa: E402
from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402
from dcv import tf_model  # noqa: E402
from dcv.main_window import MainWindow  # noqa: E402

errors = []
sys.excepthook = lambda *a: (errors.append(a), traceback.print_exception(*a))
for _n in ("information", "warning", "question"):
    setattr(QMessageBox, _n, staticmethod(lambda *a, _n=_n, **k: (print("MSGBOX", _n, str(a[1:2])), QMessageBox.Yes)[1]))

dialogs = []        # (text, informative, default button) of the real "Open study" box
answer = ["Open in new window"]


def fake_exec(box):
    dialogs.append((box.text(), box.informativeText(), box.defaultButton().text()))
    for b in box.buttons():
        if b.text() == answer[0] or (answer[0] == "Cancel" and box.buttonRole(b) == QMessageBox.RejectRole):
            b.click()
            break
    return 0


QMessageBox.exec = fake_exec


def check(ok, what):
    print(("OK   " if ok else "FAIL ") + what)
    if not ok:
        errors.append(what)


app = QApplication(sys.argv)
w = MainWindow()
w.show()
w.open_paths([data_dir])
steps = []


def wait(cond, then, tries=[0]):
    if not cond() and tries[0] < 80:
        tries[0] += 1
        QTimer.singleShot(250, lambda: wait(cond, then))
        return
    tries[0] = 0
    try:
        then()
    except Exception:
        traceback.print_exc()
        errors.append("exception")
        finish()


def start():
    wait(lambda: w.viewports[0].state is not None, step_new_window)


def step_new_window():
    head = w.viewports[0].state.vol.uid
    answer[0] = "Open in new window"
    w.open_paths([mri_dir])
    wait(lambda: len(MainWindow.windows) == 2 and MainWindow.windows[1].viewports[0].state is not None,
         lambda: after_new(head))


def after_new(head):
    text, info, default = dialogs[-1]
    check("Different patient" in text and "HEAD PHANTOM" in info and "BRAIN PHANTOM" in info,
          "different patient: warning names both patients")
    check(default == "Open in new window", f"default button for another patient: {default}")
    w2 = MainWindow.windows[1]
    check(len(w.series) == 1 and w.viewports[0].state.vol.uid == head, "first window unchanged")
    check(len(w2.series) == 3 and "BRAIN" in w2.windowTitle(), f"new window has the MRI ({len(w2.series)} series)")
    check(not w2.welcome.isVisible(), "new window: welcome panel closed")
    # cancel: nothing happens
    answer[0] = "Cancel"
    n = len(dialogs)
    w.open_paths([mri_dir])
    wait(lambda: len(dialogs) > n, lambda: after_cancel(head))


def after_cancel(head):
    check(len(w.series) == 1 and len(MainWindow.windows) == 2, "Cancel opens nothing")
    answer[0] = "Add to this window"
    n = len(dialogs)
    w.open_paths([mri_dir])
    wait(lambda: len(dialogs) > n and len(w.series) == 4, lambda: after_add(head))


def after_add(head):
    check(len(w.series) == 4 and len(w.series_list.series_items()) == 4, "Add to this window: series added to the list")
    check(w.viewports[0].state.vol.uid == head, "added series do not replace the viewport on screen")
    check("Several patients" in w.windowTitle(), f"title warns: {w.windowTitle()}")
    # same patient again -> default is 'add here'
    w2 = MainWindow.windows[1]
    answer[0] = "Cancel"
    w2.open_paths([mri_dir])                     # all series already there: no question
    n = len(dialogs)
    QTimer.singleShot(1500, lambda: after_same(w2, n))


def after_same(w2, n):
    check(len(dialogs) == n, "re-opening the same series asks nothing")
    k = len(MainWindow.windows)
    w.act_new_window.trigger()
    check(len(MainWindow.windows) == k + 1 and w.act_new_window.shortcut().toString() == "Ctrl+N",
          "Ctrl+N opens an empty window")
    MainWindow.windows[-1].close()
    w2.close()
    step_tf()


def grab(vr):
    vr.rw.Render()
    f = vtk.vtkWindowToImageFilter()
    f.SetInput(vr.rw)
    f.ReadFrontBufferOff()
    f.Update()
    return numpy_support.vtk_to_numpy(f.GetOutput().GetPointData().GetScalars()).astype(float)


def step_tf():
    st = next(s for s in w.states.values() if "HEAD" in s.vol.patient)
    vw = w.vr_window(st)
    vw.resize(1100, 800)
    app.processEvents()
    vr = vw.vr
    vw.bone_chk.setChecked(True)
    app.processEvents()
    vw.bone_chk.setChecked(False)                # the in-place rewrite that left a stale range before the fix
    app.processEvents()
    arr = vr.image.GetPointData().GetScalars()
    data = numpy_support.vtk_to_numpy(arr)
    check(tuple(arr.GetFiniteRange()) == (float(data.min()), float(data.max())),
          f"VTK range cache refreshed after bone lock: {arr.GetFiniteRange()}")
    tf = next(t for t in tf_model.builtin_presets() if t.name.startswith("CT - Enhancing"))
    check(any(c.hide_edges for c in tf.curves), "tumor preset marks its lesion curve (hide edges)")
    vw.set_tf(tf)
    vw.apply_tf()
    check(vr.prop.GetTransferFunctionMode() == 1, "curve with hide edges -> 2D transfer function")
    vw.blend_acts["mip"][0].trigger()
    vw.apply_tf()                                # set_blend applies through a 25 ms timer
    check(vr.prop.GetTransferFunctionMode() == 0, "MIP uses the 1D transfer function")
    vw.blend_acts["composite"][0].trigger()
    check(vw.edge_slider.isEnabled() and not vw.edge_hint.isVisible(), "edge slider enabled with a hide-edges curve")
    vw.edge_slider.setValue(vw.edge_slider.maximum())
    vw.apply_tf()
    check(abs(vw.state.tf.edge_threshold - 200) < 0.5 and vw.edge_lbl.text() == "200 HU",
          f"edge threshold slider + label ({vw.edge_lbl.text()})")
    vw.edge_slider.setValue(50)                  # logarithmic: the middle is ~32 HU, where the useful range is
    check(25 < vw.state.tf.edge_threshold < 40, f"rim slider is logarithmic ({vw.state.tf.edge_threshold:.0f} HU)")
    # while dragging: narrow 2D table, full width once the drag pauses
    vw.apply_tf(fast=True)
    fast_w = vr._tf2d_image.GetDimensions()[0]
    vw.apply_tf()
    full_w = vr._tf2d_image.GetDimensions()[0]
    check(fast_w <= 2048 and full_w >= fast_w, f"2D table width while dragging {fast_w}, after {full_w}")
    # a curve drag step must stay quick (adding points one by one took ~250 ms with the bone lock)
    import time
    vw.bone_chk.setChecked(True)
    app.processEvents()
    plain = next(t for t in tf_model.builtin_presets() if t.name.startswith("CT Angio - Vessels + Bone"))
    vw.set_tf(plain)
    vw.apply_tf()
    check(not vw.edge_slider.isEnabled() and not vw.edge_hint.isHidden(), "no hide-edges curve: slider off, hint shown")
    t0 = time.perf_counter()
    for i in range(5):
        vw.state.tf.curves[0].points[1].x += 1 if i % 2 else -1
        vw.apply_tf(fast=True)
    step = (time.perf_counter() - t0) / 5 * 1000
    check(step < 120, f"curve drag step with bone lock: {step:.0f} ms")
    vw.bone_chk.setChecked(False)
    app.processEvents()
    vw.set_tf(tf)
    vw.apply_tf()
    # 2D without suppression must look like 1D (catches value-axis / opacity mismatches)
    for c in vw.state.tf.curves:
        c.hide_edges = False
    vw.apply_tf()
    one = grab(vr)
    vw.state.tf.curves[1].hide_edges = True
    vw.state.tf.edge_threshold = 1e9
    vw.apply_tf()
    two = grab(vr)
    check(vr.prop.GetTransferFunctionMode() == 1 and abs(one - two).mean() < 1.0,
          f"2D without suppression == 1D (mean diff {abs(one - two).mean():.2f})")
    vw.state.tf.edge_threshold = 60
    vw.apply_tf()
    check(abs(grab(vr) - two).mean() > 0.01, "suppression changes the picture")
    vw.close()
    finish()


def finish():
    print("errors:", len(errors))
    app.quit()


QTimer.singleShot(500, start)
app.exec()
