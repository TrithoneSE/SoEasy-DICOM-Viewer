"""GUI test: the redesigned main window. One toolbar row with vector icons, W/L presets + exact WW/WL in the W/L ▾
menu, color map / slab / filter in the Image ▾ menu, left / right mouse assignment marks, welcome panel, series header,
shortcuts of the shared actions (the MPR window: test_mpr_ui.py). Run via tests\\run_tests.bat (args: app_dir data_dir out_dir)."""
import os
import sys
import traceback

app_dir, data_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, app_dir)
import builtins  # noqa: E402
_print = builtins.print
builtins.print = lambda *a, **k: _print(*a, **{**k, "flush": True})
from PySide6.QtCore import QPoint, Qt, QTimer  # noqa: E402
from PySide6.QtGui import QIcon  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox, QToolBar, QToolButton  # noqa: E402
from dcv import theme  # noqa: E402
from dcv.main_window import MainWindow  # noqa: E402

errors = []
sys.excepthook = lambda *a: (errors.append(a), traceback.print_exception(*a))
for _n in ("information", "warning", "question"):
    setattr(QMessageBox, _n, staticmethod(lambda *a, _n=_n, **k: (print("MSGBOX", _n, a[1:3]), QMessageBox.Yes)[1]))


def check(ok, what):
    print(("OK   " if ok else "FAIL ") + what)
    if not ok:
        errors.append(what)


app = QApplication(sys.argv)
w = MainWindow()
w.resize(1280, 860)
w.show()
app.processEvents()

bars = [t for t in w.findChildren(QToolBar) if t.isVisible()]
check(len(bars) == 1, f"one toolbar row ({len(bars)})")
buttons = [b for b in bars[0].findChildren(QToolButton) if b.isVisible() and b.text()]
no_icon = [b.text() for b in buttons if b.icon().isNull()]
check(not no_icon, f"every toolbar button has an icon ({len(buttons)} buttons, without: {no_icon})")
emoji = [b.text() for b in buttons if any(ord(ch) > 0x2000 for ch in b.text())]
check(not emoji, f"no emoji / symbols in toolbar labels {emoji}")
check(w.welcome.isVisible(), "welcome panel shown before a study is open")
w.open_paths([data_dir])


def run():
    if w.viewports[0].state is None:
        QTimer.singleShot(300, run)
        return
    try:
        sv = w.viewports[0].slice
        check(not w.welcome.isVisible(), "welcome panel hidden after opening")
        heads = [w.series_list.item(i).data(Qt.UserRole + 1) for i in range(w.series_list.count())]
        check(heads[:2] == ["patient", "study"] and "1 series" in
              w.series_list.itemWidget(w.series_list.item(1)).text(), "patient + study header above the thumbnails")
        ic = w.series_list.series_items()[0].icon()
        check(not ic.pixmap(64, 64, QIcon.Selected).isNull(), "thumbnail has its own selected pixmap (no tint)")

        # W/L ▾: preset, exact values
        brain = next(a for a in w.wl_tool_menu.actions() if a.text().startswith("Brain"))
        check(brain.shortcut().toString() == "1", "first preset has key 1")
        brain.trigger()
        check((sv.ww, sv.wl) == (80, 40), f"preset Brain -> {sv.ww:g}/{sv.wl:g}")
        check(w.ww_spin.value() == 80, "WW box follows the preset")
        w.ww_spin.setValue(350)
        w.wl_spin.setValue(45)
        check((sv.ww, sv.wl) == (350, 45), f"typed WW/WL -> {sv.ww:g}/{sv.wl:g}")
        check(w.wl_editor in w.wl_tool_menu.actions(), "WW/WL editor sits in the W/L menu")

        # Image ▾: color map, slab, filter
        w.cmap_actions["Hot"].trigger()
        check(sv.colormap == "Hot", f"color map -> {sv.colormap}")
        mip10 = next(a for a, d in w.slab_actions if d == (10, "mip"))
        mip10.trigger()
        check(sv.slab_mm == 10 and sv.slab_mode == "mip", f"slab -> {sv.slab_mm} {sv.slab_mode}")
        k, a = next((k, a) for k, a in w.filter_actions.items() if k != next(iter(w.filter_actions)))
        a.trigger()
        check(sv.filter == k, f"filter -> {sv.filter}")
        # a second viewport with defaults: switching back re-ticks the first viewport's choices
        w.set_grid(1, 2)
        app.processEvents()
        w._set_active(w.viewports[1])
        w._set_active(w.viewports[0])
        check(w.cmap_actions["Hot"].isChecked() and mip10.isChecked() and a.isChecked(),
              "Image menu shows the active viewport's settings")
        w.set_grid(1, 1)

        # mouse assignment marks
        pan = w.bank.buttons["pan"]
        pan.customContextMenuRequested.emit(QPoint(2, 2))
        check(w.bank.tools["right"] == "pan" and theme.RIGHT in pan.styleSheet(), "right-click assigns Pan (pink)")
        check("Right: Pan" in w.mouse_lbl.text(), "status bar shows the mouse assignment")
        w.bank.select_measure("ellipse")
        check(w.bank.measure_btn.text() == "Ellipse" and w.bank.tools["left"] == "ellipse", "measure menu -> Ellipse")
        check(theme.ACCENT in w.bank.measure_btn.styleSheet(), "measure button marked as left tool (blue)")
        w.bank.assign("left", "wl")

        # shared actions keep their shortcuts and work
        check(w.act_rot_r.shortcut().toString() == "Ctrl+Right", "rotate shortcut kept")
        rot = sv.rot
        w.act_rot_r.trigger()
        check(sv.rot == (rot + 3) % 4, "Rotate right works")       # slice_view counts clockwise as +3
        w.act_reset.trigger()
        check(sv.rot == 0, "Reset works")
        inv = sv.invert
        w.act_invert.trigger()
        check(sv.invert != inv, "Invert works")
        w.act_invert.trigger()
        check(w.act_open_files.shortcut().toString() == "Ctrl+Shift+O", "open archive shortcut")
        w.grab().save(os.path.join(out_dir, "main_ui.png"))

    except Exception:
        traceback.print_exc()
        errors.append("exception")
    print("errors:", len(errors))
    app.quit()


QTimer.singleShot(500, run)
app.exec()
