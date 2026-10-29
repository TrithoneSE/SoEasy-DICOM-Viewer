"""GUI test for DSA (digital subtraction angiography) on a synthetic XA multiframe: header hints (Frame Time,
Imager Pixel Spacing, Mask Subtraction Sequence), subtraction on/off with its own window, displayed image as mask,
reset, manual pixel-shift (drag, Shift / Ctrl+Shift + arrows), Ctrl+M = auto pixel-shift in DSA mode (recovers a
known patient movement), cine at the series' frame rate. No real DSA file was available when this was written.
Run via tests\\run_tests.bat (args: app_dir data_dir out_dir)."""
import os
import shutil
import sys
import tempfile
import time
import traceback

app_dir, data_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, app_dir)
import builtins  # noqa: E402
_print = builtins.print
builtins.print = lambda *a, **k: _print(*a, **{**k, "flush": True})
import numpy as np  # noqa: E402
import pydicom  # noqa: E402
from pydicom.dataset import Dataset, FileMetaDataset  # noqa: E402
from pydicom.uid import ExplicitVRLittleEndian, generate_uid  # noqa: E402
from PySide6.QtCore import QPointF, Qt, QTimer  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402
from dcv import privacy  # noqa: E402
from dcv.dsa import shift_image  # noqa: E402
from dcv.loader import load_series, scan_folder  # noqa: E402
from dcv.main_window import MainWindow  # noqa: E402

errors = []
sys.excepthook = lambda *a: (errors.append(a), traceback.print_exception(*a))
for _n in ("information", "warning", "question"):
    setattr(QMessageBox, _n, staticmethod(lambda *a, _n=_n, **k: (print("MSGBOX", _n, str(a[1:2])), QMessageBox.Yes)[1]))

MOVE = (1.5, -2.3)          # patient movement (rows, columns) from frame 3 on


def check(ok, what):
    print(("OK   " if ok else "FAIL ") + what)
    if not ok:
        errors.append(what)


def make_xa(folder, mask_seq=False, n=12, size=256, rle=False, plane="A", series_uid=None, name="XA0001.dcm"):
    """10-bit log-scale XA run: skull-like rings, a vessel filling from frame 5, the patient moves at frame 3."""
    yy, xx = np.mgrid[:size, :size]
    bone = (500 + 250 * ((np.hypot(yy - 128, xx - 128) // 20) % 2)
            + 150 * (np.hypot(yy - 100, xx - 150) < 25)).astype(np.float32)
    vessel = -200.0 * (np.abs(xx - 0.6 * yy - 50) < 3)
    rng = np.random.default_rng(3)
    frames = []
    for k in range(n):
        f = shift_image(bone, *MOVE) if k >= 2 else bone.copy()
        if k >= 4:
            f = f + vessel
        frames.append(np.clip(f + rng.normal(0, 2, f.shape), 0, 1023))
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.12.1"          # X-Ray Angiographic Image
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
    ds.Modality, ds.PatientName, ds.PatientID = "XA", "DSA^PHANTOM", "DSA1"
    ds.StudyInstanceUID, ds.SeriesInstanceUID = "1.2.3.4", series_uid or generate_uid()
    ds.SeriesDescription, ds.SeriesNumber, ds.InstanceNumber = "DSA run", 1, 1 if plane == "A" else 2
    ds.Manufacturer = "Siemens"
    ds.ImageType = ["ORIGINAL", "PRIMARY", f"BIPLANE {plane}"]
    ds.add_new((0x0019, 0x0010), "LO", "SIEMENS AX")    # Siemens XA private data where MR keeps its b-value
    ds.add_new((0x0019, 0x100C), "IS", "1000")
    ds.Rows = ds.Columns = size
    ds.NumberOfFrames = n
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 10, 9, 0
    ds.ImagerPixelSpacing = [0.3, 0.3]
    ds.FrameTime = 66.7
    ds.PixelIntensityRelationship = "LOG"
    ds.WindowCenter, ds.WindowWidth = 512, 1024
    if mask_seq:
        it = Dataset()
        it.MaskOperation = "AVG_SUB"
        it.MaskFrameNumbers = [3, 4]                 # 1-based, already moved: averaged, no shift needed
        it.MaskSubPixelShift = [0.0, 0.0]
        ds.MaskSubtractionSequence = [it]
        ds.RecommendedViewingMode = "SUB"
    ds.PixelData = np.stack(frames).astype(np.uint16).tobytes()
    if rle:
        from pydicom.uid import RLELossless
        ds.compress(RLELossless)
    os.makedirs(folder, exist_ok=True)
    ds.save_as(os.path.join(folder, name), enforce_file_format=True)


tmp = tempfile.mkdtemp(prefix="dcv_dsa_")
plain, tagged, biplane = (os.path.join(tmp, n) for n in ("plain", "tagged", "biplane"))
make_xa(plain)
make_xa(tagged, mask_seq=True)
uid = generate_uid()
make_xa(biplane, n=40, rle=True, plane="A", series_uid=uid, name="A.dcm")
make_xa(biplane, n=40, rle=True, plane="B", series_uid=uid, name="B.dcm")

# 0) a biplane run (two multiframe files, one series) = two entries; Siemens XA private tags are not DWI
bi = scan_folder(biplane)
check([(s.description, s.count, s.is_dwi) for s in bi] == [("DSA run A", 40, False), ("DSA run B", 40, False)],
      f"biplane run: one entry per plane, frame counts, no DWI badge: "
      f"{[(s.description, s.count, s.is_dwi) for s in bi]}")
check(all(s.thumbnail is not None for s in bi), "thumbnails from one decoded frame")
t0 = time.perf_counter()
vb = load_series(bi[0])
check(vb.loader is not None and vb.loader.ready[0] and not vb.loader.ready.all(),
      f"compressed run opens before all frames are decoded ({(time.perf_counter() - t0) * 1000:.0f} ms)")
f30 = vb.frame(30).copy()
check(vb.loader is None or vb.loader.ready[30], "a frame asked for is decoded on the spot")
for _ in range(200):
    if vb.loader is None:
        break
    time.sleep(0.05)
ref = pydicom.dcmread(bi[0].files[0]).pixel_array
check(vb.loader is None and np.array_equal(vb.data, ref.astype(vb.data.dtype)) and np.array_equal(f30, ref[30]),
      "background decoding finished, every frame identical to pydicom")

# 1) header hints
v = load_series(scan_folder(tagged)[0])
check(v.shape == (12, 256, 256) and abs(v.frame_time - 66.7) < 1e-6, f"XA multiframe: {v.shape}, frame time "
                                                                       f"{v.frame_time} ms")
check(v.spacing[1:] == (0.3, 0.3), f"Imager Pixel Spacing used: {v.spacing}")
check(v.dsa_hint["mask"] == [2, 3] and not v.dsa_hint["log"] and v.dsa_hint["recommended"],
      f"Mask Subtraction Sequence read: {v.dsa_hint}")

app = QApplication(sys.argv)
w = MainWindow()
w.resize(1300, 900)
w.show()
w.open_paths([plain])


def pump(cond, timeout=60):
    t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        app.processEvents()
        time.sleep(0.05)
    for _ in range(5):
        app.processEvents()
    return cond()


def run():
    try:
        pump(lambda: w.viewports[0].state is not None)
        sv = w.viewports[0].slice
        check(sv.cine_playing and w.cine_act.isChecked() and sv._cine.interval() == 67,
              f"a run plays on opening at its own frame rate ({sv._cine.interval()} ms)")
        sv.set_cine(False)
        check(not w.cine_act.isChecked(), "pausing the viewport unchecks the Cine button")
        sv._show_cine_bar(True)
        check(sv.cine_bar.isVisible() and sv.cine_bar.fps.value() == 15, f"cine bar shown, {sv.cine_bar.fps.value()} fps")
        sv.cine_bar.play.click()
        check(sv.cine_playing, "cine bar: play")
        sv.cine_bar.fps.setValue(5)
        check(sv._cine.interval() == 200, f"cine bar: 5 fps -> {sv._cine.interval()} ms")
        sv.cine_bar.play.click()
        sv.setFocus()
        QTest.keyClick(sv, Qt.Key_Space)
        check(sv.cine_playing, "Space plays")
        QTest.keyClick(sv, Qt.Key_Space)
        check(not sv.cine_playing, "Space pauses")
        sv.cine_bar.grab().save(os.path.join(out_dir, "dsa_cine_bar.png"))
        sv.set_index(8)
        check(w.act_dsa.shortcut().toString() == "/" and w.act_dsa_shift.shortcut().toString() == "Shift+M",
              "DSA shortcuts / and Shift+M")
        native = (sv.ww, sv.wl)
        w.act_dsa.trigger()
        bg = (0 - (sv.wl - sv.ww / 2)) / sv.ww          # grey level of "no change"
        check(sv.dsa is not None and w.act_dsa.isChecked() and abs(sv.ww - 0.6 * native[0]) < 1 and abs(bg - 0.64) < 0.01,
              f"DSA on: default DSA window (WW {sv.ww:.0f} = 0.6 x {native[0]:.0f}, background {bg:.2f} grey)")
        check(sv.dsa.mask == [1], f"default mask = image 2 ({sv.dsa.mask})")
        sv.repaint()
        app.processEvents()
        sub = sv.dsa.subtract(8)
        check(sub[100, 110] < -100, f"vessel negative after subtraction ({sub[100, 110]:.0f})")   # col = 0.6·row + 50
        sv.grab().save(os.path.join(out_dir, "dsa_unaligned.png"))

        # displayed image as mask, reset
        sv.set_index(3)
        w.act_dsa_mask.trigger()
        check(sv.dsa.mask == [3], "Shift+/: displayed image (4) is the mask")
        before = float(np.abs(sv.dsa.subtract(8)[:, :100]).mean())
        w.act_dsa_reset.trigger()
        check(sv.dsa.mask == [1] and sv.dsa.shift == (0.0, 0.0), "Ctrl+Shift+/: default mask, no shift")
        after = float(np.abs(sv.dsa.subtract(8)[:, :100]).mean())
        check(before < after, f"a mask after the movement subtracts the bone better ({before:.1f} < {after:.1f})")

        # manual pixel-shift: drag (the mask follows the mouse) and keys
        w.act_dsa_shift.trigger()
        check(sv.tools["left"] == "dsa_shift", "Shift+M: manual pixel-shift on the left mouse button")
        a = sv.image_to_widget(128, 128)
        b = sv.image_to_widget(138, 123)
        QTest.mousePress(sv, Qt.LeftButton, Qt.NoModifier, a.toPoint())
        for t in np.linspace(0, 1, 6)[1:]:
            QTest.mouseMove(sv, QPointF(a.x() + (b.x() - a.x()) * t, a.y() + (b.y() - a.y()) * t).toPoint())
        QTest.mouseRelease(sv, Qt.LeftButton, Qt.NoModifier, b.toPoint())
        dy, dx = sv.dsa.shift
        check(abs(dy - 10) < 1 and abs(dx + 5) < 1, f"drag moved the mask by ({dy:.1f}, {dx:.1f}) px")
        sv.setFocus()
        QTest.keyClick(sv, Qt.Key_Right, Qt.ShiftModifier)
        QTest.keyClick(sv, Qt.Key_Up, Qt.ShiftModifier | Qt.ControlModifier)
        check(abs(sv.dsa.shift[1] - (dx + 1)) < 1e-6 and abs(sv.dsa.shift[0] - (dy - 0.1)) < 1e-6,
              f"Shift+arrow 1 px, Ctrl+Shift+arrow 0.1 px: {sv.dsa.shift}")
        check(sv.dsa.label()[1].startswith("Shift: x"), "overlay shows the shift")

        # Ctrl+M in DSA mode = auto pixel-shift (no MPR window), recovers the patient movement
        w.act_dsa_reset.trigger()
        n_mpr = len(w.mpr_windows)
        w.act_mpr.trigger()
        check(sv.tools["left"] == "dsa_auto" and len(w.mpr_windows) == n_mpr, "Ctrl+M in DSA mode: auto pixel-shift")
        sv.set_index(8)
        sv._hover_pos = sv.image_to_widget(110, 150)
        sv.repaint()
        sv.grab().save(os.path.join(out_dir, "dsa_auto_square.png"))
        p = sv.image_to_widget(110, 150)
        QTest.mouseClick(sv, Qt.LeftButton, Qt.NoModifier, p.toPoint())
        check(np.allclose(sv.dsa.shift, MOVE, atol=0.15), f"auto pixel-shift found the movement {sv.dsa.shift}")
        bg = float(np.abs(sv.dsa.subtract(8)[:, :60]).mean())
        check(bg < 4, f"background after auto-shift ~ noise ({bg:.1f})")
        sv.repaint()
        sv.grab().save(os.path.join(out_dir, "dsa_aligned.png"))

        # Ctrl+Space / Cine button: the active viewport
        w.cine_act.trigger()
        check(sv.cine_playing and sv._cine.interval() == 200, f"Cine button plays the active viewport "
                                                               f"({sv._cine.interval()} ms, the 5 fps chosen)")
        w.cine_act.trigger()
        check(not sv.cine_playing, "Cine button pauses")

        # off: original window back
        w.act_dsa.trigger()
        check(sv.dsa is None and not w.act_dsa.isChecked() and (sv.ww, sv.wl) == native,
              "DSA off: original window restored")
        w.bank.assign("left", "wl")

        # anonymous display: no patient name / dates in the title, series list, tooltips or image overlay
        def shown_texts():
            lst = w.series_list
            heads = [lst.itemWidget(lst.item(i)).text() for i in range(lst.count()) if lst.item(i).data(Qt.UserRole + 1)]
            tips = [it.toolTip() for it in lst.series_items()]
            return [w.windowTitle()] + heads + tips
        check(any("PHANTOM" in t for t in shown_texts()), "patient name shown normally")
        w.privacy_act.trigger()
        app.processEvents()
        leaked = [t for t in shown_texts() if "PHANTOM" in t or "2025" in t]
        check(privacy.enabled and not leaked and privacy.name(sv.vol.patient) == "Anonymous",
              f"Ctrl+Shift+H hides the identity (leaks: {leaked})")
        sv.grab().save(os.path.join(out_dir, "dsa_anonymous.png"))
        w.privacy_act.trigger()
        check(not privacy.enabled and "PHANTOM" in w.windowTitle(), "identity shown again")

        # the scanner's mask (Mask Subtraction Sequence): already aligned frames 3+4
        w._ask_open_target = lambda new: "here"
        n0 = len(w.series)
        w.open_paths([tagged])
        pump(lambda: len(w.series) > n0)
        info = next(s for s in w.series.values() if s.uid != sv.vol.uid and s.modality == "XA")
        uids = [it.data(Qt.UserRole).uid for it in w.series_list.series_items()]
        step = 1 if uids.index(info.uid) > uids.index(sv.vol.uid) else -1
        sv.seriesStep.emit(sv, step)                  # cine bar: next / previous series in this viewport
        pump(lambda: sv.vol.uid == info.uid)
        check(sv.vol.uid == info.uid and sv.cine_playing, "cine bar: next series opened, and it plays")
        sv.set_cine(False)
        check(w.act_dsa.isChecked() is False, "another series starts unsubtracted")
        w.act_dsa.trigger()
        bg = float(np.abs(sv.dsa.subtract(8)[:, :60]).mean())
        check(sv.dsa.mask == [2, 3] and bg < 4, f"scanner mask frames 3+4 used ({sv.dsa.label()[0]}, bg {bg:.1f})")
    except Exception:
        traceback.print_exc()
        errors.append("exception")
    QTimer.singleShot(200, app.quit)


QTimer.singleShot(300, run)
app.exec()
shutil.rmtree(tmp, ignore_errors=True)
print("errors:", len(errors))
