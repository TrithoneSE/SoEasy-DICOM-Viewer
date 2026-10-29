"""GUI test: drag a series thumbnail onto a viewport in a 1×3 layout (samples/mri: T1, FLAIR, DTI). The drag starts
from a real mouse press + move on the list; QDrag.exec is intercepted (the OS drag loop blocks) and the drag-enter /
drop events are then sent to the target viewport. Also: highlight while hovering, plain click still opens in the
active viewport. Run via tests\\run_tests.bat (args: app_dir data_dir out_dir; samples/mri next to data_dir)."""
import os
import sys
import traceback

app_dir, data_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
mri_dir = os.path.join(os.path.dirname(data_dir.rstrip("\\/")), "mri")
sys.path.insert(0, app_dir)
import builtins  # noqa: E402
_print = builtins.print
builtins.print = lambda *a, **k: _print(*a, **{**k, "flush": True})
from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, QTimer, QUrl  # noqa: E402
from PySide6.QtGui import QDrag, QDragEnterEvent, QDragLeaveEvent, QDropEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402
from dcv import theme  # noqa: E402
from dcv.main_window import MainWindow  # noqa: E402

errors = []
sys.excepthook = lambda *a: (errors.append(a), traceback.print_exception(*a))
for _n in ("information", "warning", "question"):
    setattr(QMessageBox, _n, staticmethod(lambda *a, _n=_n, **k: (print("MSGBOX", _n, str(a[1:2])), QMessageBox.Yes)[1]))
started = []


def fake_exec(self, *a, **k):
    started.append((self.mimeData(), self.pixmap()))
    return Qt.CopyAction


QDrag.exec = fake_exec


def check(ok, what):
    print(("OK   " if ok else "FAIL ") + what)
    if not ok:
        errors.append(what)


app = QApplication(sys.argv)
w = MainWindow()
w.resize(1500, 850)
w.show()
w.open_paths([mri_dir])


def uid_of(vp):
    return vp.state.vol.uid if vp.state is not None else None


def run():
    if w.viewports[0].state is None:
        QTimer.singleShot(300, run)
        return
    try:
        w.set_grid(1, 3)
        app.processEvents()
        lst = w.series_list
        first = uid_of(w.viewports[0])
        item = next(it for it in lst.series_items() if it.data(Qt.UserRole).uid != first)
        target_uid = item.data(Qt.UserRole).uid
        rect = lst.visualItemRect(item)
        lst.scrollToItem(item)
        app.processEvents()
        rect = lst.visualItemRect(item)
        a = rect.center()
        QTest.mousePress(lst.viewport(), Qt.LeftButton, Qt.NoModifier, a)
        QTest.mouseMove(lst.viewport(), a + QPoint(40, 5))
        QTest.mouseRelease(lst.viewport(), Qt.LeftButton, Qt.NoModifier, a + QPoint(40, 5))
        check(len(started) == 1, f"press + move on a thumbnail starts a drag ({len(started)})")
        md, pm = started[-1]
        check(bytes(md.data("application/x-dcv-series")).decode() == target_uid, "drag carries the series")
        check(not pm.isNull(), "drag shows the thumbnail")

        vp = w.viewports[2]
        pos = QPointF(vp.width() / 2, vp.height() / 2)
        enter = QDragEnterEvent(pos.toPoint(), Qt.CopyAction, md, Qt.LeftButton, Qt.NoModifier)
        QApplication.sendEvent(vp, enter)
        check(enter.isAccepted() and theme.ACCENT in vp.styleSheet(), "target viewport accepts and is highlighted")
        QApplication.sendEvent(vp, QDragLeaveEvent())
        check(theme.ACCENT not in vp.styleSheet(), "highlight removed when the drag leaves")
        files = QMimeData()
        files.setUrls([QUrl.fromLocalFile(mri_dir)])
        ev = QDragEnterEvent(pos.toPoint(), Qt.CopyAction, files, Qt.LeftButton, Qt.NoModifier)
        QApplication.sendEvent(vp, ev)       # propagates to the main window, which opens dropped files
        check(ev.isAccepted() and theme.ACCENT not in vp.styleSheet(),
              "files from Explorer: not a series drop, the main window takes them")
        QApplication.sendEvent(vp, QDragEnterEvent(pos.toPoint(), Qt.CopyAction, md, Qt.LeftButton, Qt.NoModifier))
        QApplication.sendEvent(vp, QDropEvent(pos, Qt.CopyAction, md, Qt.LeftButton, Qt.NoModifier))
        QTimer.singleShot(200, lambda: after_drop(vp, target_uid, first))
    except Exception:
        traceback.print_exc()
        errors.append("exception")
        finish()


def after_drop(vp, target_uid, first, tries=[0]):
    if uid_of(vp) != target_uid and tries[0] < 40:
        tries[0] += 1
        QTimer.singleShot(250, lambda: after_drop(vp, target_uid, first))
        return
    try:
        check(uid_of(vp) == target_uid, "dropped series opens in that viewport (3rd of 1x3)")
        check(uid_of(w.viewports[0]) == first, "the other viewports keep their series")
        check(w.active is vp and theme.ACTIVE_VIEWPORT in vp.styleSheet(), "target becomes the active viewport")
        w.grab().save(os.path.join(out_dir, "drag_1x3.png"))
        # plain click (no drag) still opens in the active viewport
        w._set_active(w.viewports[1])
        n = len(started)
        lst = w.series_list
        it = lst.series_items()[0]
        QTest.mouseClick(lst.viewport(), Qt.LeftButton, Qt.NoModifier, lst.visualItemRect(it).center())
        check(len(started) == n, "a click does not start a drag")
        QTimer.singleShot(200, lambda: after_click(it.data(Qt.UserRole).uid))
    except Exception:
        traceback.print_exc()
        errors.append("exception")
        finish()


def after_click(uid, tries=[0]):
    if uid_of(w.viewports[1]) != uid and tries[0] < 40:
        tries[0] += 1
        QTimer.singleShot(250, lambda: after_click(uid))
        return
    check(uid_of(w.viewports[1]) == uid, "click opens the series in the active viewport")
    finish()


def finish():
    print("errors:", len(errors))
    app.quit()


QTimer.singleShot(500, run)
app.exec()
