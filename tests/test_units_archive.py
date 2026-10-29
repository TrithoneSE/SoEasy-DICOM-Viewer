"""GUI test: measurement units switch mm -> cm (2D drag with the mouse + 3D labels), and opening a study
packed in .zip / .rar / nested archives. Run via tests\\run_tests.bat (args: app_dir data_dir out_dir)."""
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
import zipfile

app_dir, data_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, app_dir)
import builtins  # noqa: E402
_print = builtins.print
builtins.print = lambda *a, **k: _print(*a, **{**k, "flush": True})
import numpy as np  # noqa: E402
from PySide6.QtCore import QPoint, Qt, QTimer  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402
from dcv import archive  # noqa: E402
from dcv.loader import scan_folder  # noqa: E402
from dcv.main_window import MainWindow  # noqa: E402
from dcv.measure import fmt_area, fmt_dims, fmt_len  # noqa: E402
from dcv.volume_view import Measure3D  # noqa: E402

errors = []
sys.excepthook = lambda *a: (errors.append(a), traceback.print_exception(*a))


def check(ok, what):
    print(("OK   " if ok else "FAIL ") + what)
    if not ok:
        errors.append(what)


for _n in ("information", "warning", "question"):
    setattr(QMessageBox, _n, staticmethod(lambda *a, _n=_n, **k: (print("MSGBOX", _n, a[1:3]), QMessageBox.Yes)[1]))

# ---------------------------------------------------------------- formatting
check(fmt_len(8.44) == "8.4 mm", f"8.44 mm -> {fmt_len(8.44)}")
check(fmt_len(9.96) == "1.00 cm", f"9.96 mm -> {fmt_len(9.96)}")
check(fmt_len(32.7) == "3.27 cm", f"32.7 mm -> {fmt_len(32.7)}")
check(fmt_len(123.4) == "12.34 cm", f"123.4 mm -> {fmt_len(123.4)}")
check(fmt_dims(8, 6.1) == "8.0 × 6.1 mm", f"dims small -> {fmt_dims(8, 6.1)}")
check(fmt_dims(45.2, 6.1) == "4.52 × 0.61 cm", f"dims mixed -> {fmt_dims(45.2, 6.1)}")
check(fmt_area(55) == "55.0 mm²" and fmt_area(250) == "2.50 cm²", f"area -> {fmt_area(55)}, {fmt_area(250)}")
check(Measure3D("length", [(0, 0, 0), (0, 0, 7)]).label == "7.0 mm", "3D length 7 mm")
check(Measure3D("length", [(0, 0, 0), (30, 40, 0)]).label == "5.00 cm", "3D length 50 mm -> 5.00 cm")

# ---------------------------------------------------------------- archives
tmp = tempfile.mkdtemp(prefix="dcv_test_")
n_files = sum(len(f) for _, _, f in os.walk(data_dir))
zpath = os.path.join(tmp, "kepala.zip")
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, names in os.walk(data_dir):
        for n in names:
            z.write(os.path.join(root, n), os.path.join("CT Kepala", os.path.relpath(os.path.join(root, n), data_dir)))
s = scan_folder([zpath])
check(len(s) >= 1 and sum(x.count for x in s) == n_files, f"zip: {len(s)} series, {sum(x.count for x in s)} images")
# a zip inside a folder inside a zip
outer = os.path.join(tmp, "outer.zip")
with zipfile.ZipFile(outer, "w") as z:
    z.write(zpath, "patient/kepala.zip")
s = scan_folder([outer])
check(sum(x.count for x in s) == n_files, f"nested zip: {sum(x.count for x in s)} images")
# a damaged archive and nothing else -> clear error
bad = os.path.join(tmp, "bad.zip")
with open(bad, "wb") as f:
    f.write(b"PK\x03\x04 not really a zip")
try:
    scan_folder([bad])
    check(False, "damaged zip raises an error")
except RuntimeError as ex:
    check("bad.zip" in str(ex), f"damaged zip -> error: {str(ex).splitlines()[0]}")
rar_exe = next((p for p in (r"C:\Program Files\WinRAR\Rar.exe", r"C:\Program Files (x86)\WinRAR\Rar.exe")
                if os.path.isfile(p)), None)
rpath = os.path.join(tmp, "kepala.rar")
if rar_exe:
    subprocess.run([rar_exe, "a", "-r", "-ep1", "-idq", rpath, data_dir], check=True)
    tools, archive._tools = archive._tools, lambda: tools()[:1]    # only Windows tar.exe, as on a PC without WinRAR
    s = scan_folder([rpath])
    archive._tools = tools
    check(sum(x.count for x in s) == n_files, f"rar via tar.exe: {sum(x.count for x in s)} images")
else:
    print("SKIP rar (WinRAR not installed to create a test archive)")
    rpath = zpath

# ---------------------------------------------------------------- GUI: open the archive, measure with the mouse
app = QApplication(sys.argv)
w = MainWindow()
w.resize(1300, 900)
w.show()
w.open_paths([rpath])


def run():
    if w.viewports[0].state is None:
        QTimer.singleShot(300, run)
        return
    try:
        sv = w.viewports[0].slice
        check(archive.BASE in sv.state.vol.files[0], "series loaded from the extracted archive")
        sr, sc = sv._spacing_rc()
        r0, c0 = sv.state.vol.shape[1] / 2, sv.state.vol.shape[2] / 2

        def drag(dc_mm, dr_mm=0.0):        # start away from earlier handles (pressing one moves it)
            a = sv.image_to_widget(r0 + dr_mm / sr, c0)
            b = sv.image_to_widget(r0 + dr_mm / sr, c0 + dc_mm / sc)
            a, b = QPoint(round(a.x()), round(a.y())), QPoint(round(b.x()), round(b.y()))
            w.bank.select_measure("length")
            QTest.mousePress(sv, Qt.LeftButton, Qt.NoModifier, a)
            for t in np.linspace(0, 1, 6)[1:]:
                QTest.mouseMove(sv, a + (b - a) * t)
            QTest.mouseRelease(sv, Qt.LeftButton, Qt.NoModifier, b)
            app.processEvents()
            m = sv.measurements()[-1]
            return m, m.label(None, sr, sc, "")[0]       # length labels do not need the pixel plane

        m, lab = drag(6)
        check(lab.endswith(" mm"), f"2D drag ~6 mm -> {lab}")
        m, lab = drag(45, 30)
        check(len(sv.measurements()) == 2 and lab.endswith(" cm") and 4.3 < float(lab.split()[0]) < 4.7, f"2D drag ~45 mm -> {lab}")
        sv.grab().save(os.path.join(out_dir, "units_cm.png"))
    except Exception:
        traceback.print_exc()
        errors.append("exception")
    shutil.rmtree(tmp, ignore_errors=True)
    print("errors:", len(errors))
    app.quit()


QTimer.singleShot(500, run)
app.exec()
