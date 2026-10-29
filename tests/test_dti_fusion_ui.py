"""GUI test: the redesigned tractography window and Fusion dialog on the synthetic MRI (samples/mri, 30-dir DTI).
Tract window: one toolbar row with icons, sphere tools (Include / Exclude / Seed + plain click on a slice, modifier
clicks still work, no tool = rotate), run, delete / clear, PNG snapshot, shortcuts. Fusion: series list with
thumbnails, overlay on / off, opacity label. Run via tests\\run_tests.bat (args: app_dir data_dir out_dir; data_dir
is the head CT, the MRI sample is found next to it)."""
import os
import sys
import traceback

app_dir, data_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
mri_dir = os.path.join(os.path.dirname(data_dir.rstrip("\\/")), "mri")
sys.path.insert(0, app_dir)
import builtins  # noqa: E402
_print = builtins.print
builtins.print = lambda *a, **k: _print(*a, **{**k, "flush": True})
from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QToolBar, QToolButton  # noqa: E402
from dcv.main_window import MainWindow  # noqa: E402

errors = []
sys.excepthook = lambda *a: (errors.append(a), traceback.print_exception(*a))
for _n in ("information", "warning", "question"):
    setattr(QMessageBox, _n, staticmethod(lambda *a, _n=_n, **k: (print("MSGBOX", _n, str(a[1:2])), QMessageBox.Yes)[1]))
snap_path = os.path.join(out_dir, "tract_snapshot.png")
QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (snap_path, "PNG (*.png)"))


def check(ok, what):
    print(("OK   " if ok else "FAIL ") + what)
    if not ok:
        errors.append(what)


app = QApplication(sys.argv)
w = MainWindow()
w.show()
w.open_paths([mri_dir])


def run():
    if w.viewports[0].state is None:
        QTimer.singleShot(300, run)
        return
    w.open_dti()
    QTimer.singleShot(500, wait_tract)


def wait_tract():
    if not w.tract_windows:
        QTimer.singleShot(300, wait_tract)
        return
    try:
        tw = w.tract_windows[0]
        tw.resize(1300, 850)
        app.processEvents()
        bars = [t for t in tw.findChildren(QToolBar) if t.isVisible()]
        check(len(bars) == 1, f"tract window: one toolbar row ({len(bars)})")
        buttons = [b for b in bars[0].findChildren(QToolButton) if b.isVisible() and b.text()]
        check(all(not b.icon().isNull() for b in buttons), f"every tract toolbar button has an icon ({len(buttons)})")
        keys = {a.shortcut().toString() for a in tw.actions()}
        check({"Ctrl+Return", "Ctrl+R", "Ctrl+S", "F1"} <= keys, "tract window shortcuts")

        iren = tw.iren
        W, H = tw.rw.GetSize()

        def click(x, y, ctrl=0, shift=0):
            iren.SetEventInformation(int(x), int(y), ctrl, shift, "\0", 0)
            iren.InvokeEvent("LeftButtonPressEvent")
            iren.InvokeEvent("LeftButtonReleaseEvent")

        tw._set_camera()
        tw.rw.Render()
        cx, cy = W // 2, H // 2
        click(cx, cy)
        check(len(tw.rois) == 0, "no tool: plain click places nothing (rotates)")
        tw.roi_acts["include"].setChecked(True)
        click(cx, cy)
        check(len(tw.rois) == 1 and tw.rois[0][0] == "include", "Include tool + click on a slice -> include sphere")
        click(cx + 5, cy + 5, ctrl=1)
        check(len(tw.rois) == 2 and tw.rois[1][0] == "exclude", "Ctrl+click still -> exclude sphere")
        n = len(tw.rois)
        click(3, 3)
        check(len(tw.rois) == n, "tool on, click outside the slices -> nothing placed")
        tw.roi_acts["include"].setChecked(False)
        check(tw.roi_mode() is None, "tool off again")
        check(tw.roi_list.count() == 2 and not tw.roi_list.item(0).icon().isNull(), "sphere list with color dots")
        tw._clear_rois()
        check(len(tw.rois) == 0 and tw.roi_list.count() == 0, "Clear removes all spheres")

        tw.seed_mode.setCurrentIndex(0)
        tw.p_stride.setValue(3)
        next(a for a in tw.actions() if a.shortcut().toString() == "Ctrl+Return").trigger()
        check(len(getattr(tw, "all_lines", [])) > 0 and len(tw.lines) > 0,
              f"Run -> {len(tw.lines)} streamlines")
        if os.path.exists(snap_path):
            os.remove(snap_path)
        next(a for a in tw.actions() if a.shortcut().toString() == "Ctrl+S").trigger()
        check(os.path.exists(snap_path) and os.path.getsize(snap_path) > 1000, "PNG snapshot saved")
        from panel_fit import clipped_controls
        from PySide6.QtWidgets import QDockWidget
        panel = tw.findChild(QDockWidget).widget()
        cut = clipped_controls(tw, [("tract panel", panel, lambda: None)])
        check(not cut, f"tract panel: no control cut off at minimum width {cut[:4]}")
        tw.close()

        # Fusion dialog
        w.open_fusion()
        dlg = w.fusion_dlg
        app.processEvents()
        check(dlg.series.count() > 2 and not dlg.series.itemIcon(1).isNull(), "fusion: series list with thumbnails")
        check(not dlg.cmap.itemIcon(0).isNull(), "fusion: color maps with swatches")
        i = next(i for i in range(dlg.series.count()) if "color FA" in dlg.series.itemText(i))
        dlg.series.setCurrentIndex(i)
        dlg._on_series()
        QTimer.singleShot(1500, lambda: fusion_on(dlg))
    except Exception:
        traceback.print_exc()
        errors.append("exception")
        finish()


def fusion_on(dlg):
    try:
        s = w._active_slice()
        check(s.fusion is not None and s.fusion["cmap"] == "Native RGB", "fusion overlay on (color FA, native RGB)")
        dlg.opacity.setValue(70)
        check(dlg.opacity_lbl.text() == "70 %" and abs(s.fusion["opacity"] - 0.7) < 1e-6, "opacity 70 %")
        dlg.remove()
        check(s.fusion is None, "Remove overlay")
        dlg.close()
    except Exception:
        traceback.print_exc()
        errors.append("exception")
    finish()


def finish():
    print("errors:", len(errors))
    app.quit()


QTimer.singleShot(500, run)
app.exec()
