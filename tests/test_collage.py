"""GUI test: collage (toolbar Collage, File → Collage…, Ctrl+G). One image of cols × rows tiles, one slice every N mm
(all slices when they are already that far apart), centred in the series by default and moved with the slider;
tiles rendered with the viewport's window.
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
from PySide6.QtWidgets import QApplication, QMessageBox, QToolBar  # noqa: E402
from dcv.collage import CollageDialog, frame_indices, slice_indices  # noqa: E402
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
    while not cond() and time.time() - t < timeout:
        app.processEvents()
        time.sleep(0.05)
    return cond()


# slice selection: one image, centred
idx = slice_indices(292, 0.625, 5.0, 25)
check(len(idx) == 25 and idx[1] - idx[0] == 8, "292 × 0.625 mm, 5 × 5: 25 images, every 8th slice (5 mm)")
check(idx[0] + 1 == 50 and idx[-1] + 1 == 242, f"centred: images {idx[0] + 1}–{idx[-1] + 1} of 292")
check(slice_indices(29, 5.0, 5.0, 25) == list(range(2, 27)), "29 thick slices, 5 × 5: the middle 25")
check(slice_indices(29, 5.0, 5.0, 36) == list(range(29)), "29 thick slices, 6 × 6: all 29")
check(slice_indices(20, 6.0, 5.0, 25) == list(range(20)), "20 MRI slices 6 mm apart: all 20")
check(slice_indices(292, 0.625, 5.0, 25, 0)[0] == 0 and slice_indices(292, 0.625, 5.0, 25, 999)[-1] == 291,
      "start clamped to the series")
check(frame_indices(10, 25) == list(range(10)) and len(frame_indices(200, 25)) == 25, "movie frames fill the image")

app = QApplication(sys.argv)
w = MainWindow()
w.resize(1400, 800)
w.show()
w.open_paths([data_dir])
try:
    check(wait_for(lambda: w.viewports[0].state is not None), "head CT opened")
    tb = next(t for t in w.findChildren(QToolBar) if w.act_collage in t.actions())
    btn = tb.widgetForAction(w.act_collage)
    check(btn is not None and not w.act_collage.icon().isNull(), "Collage button with icon in the toolbar")
    check(w.act_collage.shortcut().toString() == "Ctrl+G", "Ctrl+G")
    file_menu = w.menuBar().actions()[0].menu()
    check(w.act_collage in file_menu.actions(), "File → Collage…")

    s = w._active_slice()
    n, sp = s.count(), s.axis_spacing()
    old_index = s.index
    dlg = CollageDialog(w, s)
    dlg.resize(760, 760)
    dlg.show()
    app.processEvents()
    check(dlg.grid.currentData() == (5, 5) and dlg.interval.value() == 5.0, "defaults: 5 × 5, every 5 mm")
    idx = dlg.indices()
    check(idx == slice_indices(n, sp, 5.0, 25), f"{len(idx)} images from {n} slices of {sp:.2f} mm")
    check(len(idx) == 25 and abs((idx[0] + idx[-1]) / 2 - (n - 1) / 2) <= 1, "centred in the series")
    img = dlg.image()
    tile_w = dlg.size.currentData()
    check(img.width() == 5 * tile_w and img.height() > 5 * 0.5 * tile_w, "one image of 5 × 5 tiles")
    check(s.index == old_index, "the viewport keeps its slice")
    check("move the slider" in dlg.info.text() and "page" not in dlg.info.text(), "info: no pages, slider hint")

    # slider moves the run; grid / interval changes keep its centre
    dlg.start.setValue(0)
    app.processEvents()
    check(dlg.indices()[0] == 0 and dlg.first.value() == 1, "slider to the start: images from 1")
    dlg.start.setValue(dlg.start.maximum())
    app.processEvents()
    check(dlg.indices()[-1] in (n - 1, n - 2), "slider to the end reaches the last slices")
    dlg.start.setValue(10)
    mid = (dlg.indices()[0] + dlg.indices()[-1]) / 2
    dlg.grid.setCurrentIndex(next(i for i in range(dlg.grid.count()) if dlg.grid.itemData(i) == (4, 3)))
    app.processEvents()
    check(len(dlg.indices()) == 12 and abs((dlg.indices()[0] + dlg.indices()[-1]) / 2 - mid) <= 2,
          "4 × 3 keeps the same centre")
    next(b for b in dlg.findChildren(type(dlg.save_btn)) if b.text() == "Centre").click()
    app.processEvents()
    check(abs((dlg.indices()[0] + dlg.indices()[-1]) / 2 - (n - 1) / 2) <= 2, "Centre puts it back in the middle")

    # every slice when the interval is at most the spacing and the series fits
    dlg.grid.setCurrentIndex(dlg.grid.count() - 1)
    dlg.cols.setValue(10)
    dlg.rows.setValue(8)
    dlg.interval.setValue(sp)
    app.processEvents()
    check(dlg.indices() == list(range(n)) and "all" in dlg.info.text(), "10 × 8 at the slice spacing: every slice")

    # window / level is the viewport's
    dlg.grid.setCurrentIndex(0)
    dlg.interval.setValue(5.0)
    a = dlg.image().copy()
    s.set_window(80, 40)
    dlg.refresh()
    check(a != dlg.image(), "tiles follow the viewport's window / level")

    path = dlg.save(os.path.join(out_dir, "collage.png"))
    check(path and os.path.isfile(path), "saved one image")
    dlg.copy()
    check(not QApplication.clipboard().image().isNull(), "Copy puts the image on the clipboard")
    dlg.close()
except Exception:
    traceback.print_exc()
    errors.append("exception")
print("errors:", len(errors))
