"""GUI test: the redesigned MPR window. One toolbar row with icons + labels, W/L presets (keys 1–9) and exact WW/WL in
the W/L ▾ menu, color map / slab / filter in Image ▾, axes toggle, active-view frame, shortcuts of actions that only
sit in ▾ menus, mouse assignment in the status bar. Run via tests\\run_tests.bat (args: app_dir data_dir out_dir)."""
import os
import sys
import traceback

app_dir, data_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, app_dir)
import builtins  # noqa: E402
_print = builtins.print
builtins.print = lambda *a, **k: _print(*a, **{**k, "flush": True})
from PySide6.QtCore import QPoint, Qt, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox, QToolBar, QToolButton  # noqa: E402
from dcv import theme  # noqa: E402
from dcv.main_window import MainWindow  # noqa: E402

errors = []
sys.excepthook = lambda *a: (errors.append(a), traceback.print_exception(*a))
for _n in ("information", "warning", "question"):
    setattr(QMessageBox, _n, staticmethod(lambda *a, _n=_n, **k: (print("MSGBOX", _n, str(a[1:2])), QMessageBox.Yes)[1]))


def check(ok, what):
    print(("OK   " if ok else "FAIL ") + what)
    if not ok:
        errors.append(what)


app = QApplication(sys.argv)
w = MainWindow()
w.show()
w.open_paths([data_dir])


def run():
    if w.viewports[0].state is None:
        QTimer.singleShot(300, run)
        return
    try:
        w.open_mpr()
        app.processEvents()
        mpr = next(iter(w.mpr_windows.values()))
        mpr.resize(1400, 820)
        app.processEvents()
        bars = [t for t in mpr.findChildren(QToolBar) if t.isVisible()]
        check(len(bars) == 1, f"one toolbar row ({len(bars)})")
        buttons = [b for b in bars[0].findChildren(QToolButton) if b.isVisible() and b.text()]
        check(all(not b.icon().isNull() for b in buttons), f"every MPR toolbar button has an icon ({len(buttons)})")
        check(all(b.toolButtonStyle() == Qt.ToolButtonTextUnderIcon for b in mpr.bank.buttons.values()),
              "tool buttons show a label under the icon")
        emoji = [b.text() for b in buttons if any(ord(ch) > 0x2000 for ch in b.text())]
        check(not emoji, f"no emoji in MPR toolbar labels {emoji}")

        views = list(mpr.views.values())
        # W/L presets and exact values apply to all three views
        brain = next(a for a in mpr.bank.buttons["wl"].menu().actions() if a.text().startswith("Brain"))
        check(brain.shortcut().toString() == "1", "first preset has key 1")
        brain.trigger()
        check(all((v.ww, v.wl) == (80, 40) for v in views), "preset Brain -> all views 80/40")
        mpr.ww_spin.setValue(300)
        check(all(v.ww == 300 for v in views), "typed WW -> all views")

        # Image menu
        mpr.cmap_actions["Hot"].trigger()
        check(all(v.colormap == "Hot" for v in views), "color map Hot -> all views")
        a10 = next(a for a, d in mpr.slab_actions if d == (10, "max"))
        a10.trigger()
        check(mpr.model.slab_mm == 10 and mpr.model.slab_mode == "max", "slab MIP 10 mm")
        k = list(mpr.filter_actions)[1]
        mpr.filter_actions[k].trigger()
        check(all(v.filter == k for v in views), f"filter {k} -> all views")

        # axes toggle, active frame
        mpr.lines_act.trigger()
        check(not mpr.model.show_lines, "Axes off hides the plane lines")
        mpr.lines_act.trigger()
        mpr._set_active(mpr.views["sagittal"])
        check(mpr.views["sagittal"].is_active and not mpr.views["axial"].is_active, "active view flagged (frame)")

        # mouse assignment
        mpr.bank.buttons["pan"].customContextMenuRequested.emit(QPoint(2, 2))
        check("Right: Pan" in mpr.mouse_lbl.text() and theme.RIGHT in mpr.bank.buttons["pan"].styleSheet(),
              "right-click assigns Pan (status bar + pink mark)")

        # shortcuts of actions that only sit in dropdown menus
        keys = {a.shortcut().toString() for a in mpr.actions()}
        missing = [s for s in ("1", "0", "I", "Ctrl+R", "Ctrl+Right", "Ctrl+H", "Ctrl+S", "Ctrl+Shift+C", "F1",
                               "Ctrl+Shift+A") if s not in keys]
        check(not missing, f"window shortcuts present (missing: {missing})")
        rot = mpr.active.rot
        next(a for a in mpr.actions() if a.shortcut().toString() == "Ctrl+Right").trigger()
        check(mpr.active.rot != rot, "rotate (active view) works")
        next(a for a in mpr.actions() if a.shortcut().toString() == "Ctrl+R").trigger()
        mpr.grab().save(os.path.join(out_dir, "mpr_ui.png"))
        mpr.close()
    except Exception:
        traceback.print_exc()
        errors.append("exception")
    print("errors:", len(errors))
    app.quit()


QTimer.singleShot(500, run)
app.exec()
