"""Bone lock vs bright vessels and artery / vein classes on a synthetic CTA; 3D Pan tool; seed range editor (handles
re-grow the segment); collage in another patient plane. Run via tests/run_all.py (args: app_dir data_dir out_dir)."""
import os
import sys
import time
import traceback

app_dir, data_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, app_dir)
import builtins  # noqa: E402
_print = builtins.print
builtins.print = lambda *a, **k: _print(*a, **{**k, "flush": True})
import numpy as np  # noqa: E402
from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402
from dcv import collage, segment  # noqa: E402
from dcv.loader import Volume  # noqa: E402
from dcv.main_window import MainWindow  # noqa: E402
from dcv.segment import VolumeState  # noqa: E402

errors = []
sys.excepthook = lambda *a: (errors.append(a), traceback.print_exception(*a))
for _n in ("information", "warning", "question"):
    setattr(QMessageBox, _n, staticmethod(lambda *a, _n=_n, **k: (print("MSGBOX", _n, a[1:3]), QMessageBox.Yes)[1]))


def check(ok, what):
    print(("OK   " if ok else "FAIL ") + what)
    if not ok:
        errors.append(what)


app = QApplication(sys.argv)


def pump(sec=0.3):
    t = time.time()
    while time.time() - t < sec:
        app.processEvents()
        time.sleep(0.02)


def wait_for(cond, timeout=60):
    t = time.time()
    while not cond() and time.time() - t < timeout:
        app.processEvents()
        time.sleep(0.03)
    return cond()


def synthetic_cta():
    """100 x 120 x 120 voxels of 1 mm: brain 40 HU; skull shell 1400 HU; an artery (600 HU) running from the shell to
    an aneurysm (650 HU, r = 5 mm) inside; a vein (150 HU) elsewhere; a thin 600 HU plate on the skull."""
    z, y, x = np.mgrid[0:100, 0:120, 0:120].astype(np.float32)
    r = np.sqrt((z - 50) ** 2 + (y - 60) ** 2 + (x - 60) ** 2)
    d = np.full(z.shape, -1000, np.int16)
    d[r < 48] = 40
    d[(r >= 44) & (r < 48)] = 1400
    artery = (np.hypot(z - 50, y - 60) < 2.5) & (x > 16) & (x <= 60)   # touches the skull at x = 16
    d[artery] = 600
    sphere = np.sqrt((z - 50) ** 2 + (y - 60) ** 2 + (x - 62) ** 2) < 5
    d[sphere] = 650
    vein = (np.hypot(z - 50, x - 90) < 2.5) & (y > 40) & (y < 80)
    d[vein] = 150
    plate = (np.abs(x - 62) < 1) & (y >= 12) & (y <= 18) & (np.abs(z - 50) < 6)
    d[plate & (r < 48) & (r >= 42)] = 600
    return Volume(data=d, spacing=(1.0, 1.0, 1.0), origin=(0, 0, 0), modality="CT", description="synthetic CTA",
                  patient="X", uid="synthetic", frame_uid="f"), sphere, artery, vein


try:
    vol, sphere, artery, vein = synthetic_cta()
    st = VolumeState(vol)
    bone, vessels = segment.bone_and_vessels(vol.data, vol.spacing, 500.0)
    check(not bone[sphere].any() and vessels[sphere].all(), "aneurysm at 650 HU (>= bone threshold) is not bone")
    away = artery & (np.mgrid[0:100, 0:120, 0:120][2] > 22)
    check(vessels[away].mean() > 0.95, f"artery touching the skull stays a vessel ({vessels[away].mean() * 100:.0f} %)")
    shell = vol.data == 1400
    check(bone[shell].mean() > 0.99, f"skull is bone ({bone[shell].mean() * 100:.1f} %)")
    region = st.bone_region()
    check(not region[sphere].any(), "aneurysm stays out of the bone area")

    core = segment.artery_threshold(vol.data, ~region)
    check(180 <= core <= 500, f"automatic arterial core {core:.0f} HU")
    st.encode_bone = st.encode_veins = True
    st.update_render_data()
    d = st.render_data
    check(st.artery_mask()[sphere].all() and not st.artery_mask()[vein].any(), "aneurysm = artery, the vein is not")
    check(st.vein_mask()[vein].mean() > 0.9, "vein classified")
    check((d[vein] >= segment.VEIN_OFFSET).mean() > 0.9, "veins stored in their own domain")
    check((d[sphere] > 0).all() and d[sphere].max() <= segment.ARTERY_TOP_HU, "aneurysm in the normal domain, clamped")
    check((vol.data == 650).sum() == sphere.sum(), "2D data untouched")

    # the real windows
    w = MainWindow()
    w.resize(1400, 900)
    w.show()
    w.open_paths([data_dir])
    check(wait_for(lambda: w.viewports[0].state is not None), "study opened")
    sv = w.viewports[0].slice
    vw = w.vr_window(sv.state)
    vw.resize(1400, 900)
    pump(0.5)
    vr = vw.vr
    cam = vr.renderer.GetActiveCamera()

    # Pan tool
    vw.rot_acts["pan"][0].trigger()
    check(vr.rotate_mode == "pan", "Pan button sets the mode")
    f0, up0 = np.array(cam.GetFocalPoint()), np.array(cam.GetViewUp())
    dir0 = np.array(cam.GetDirectionOfProjection())
    iren = vr.iren

    def ev(name, x, y):
        iren.SetEventInformation(int(x), int(y), 0, 0, chr(0), 0)
        iren.InvokeEvent(name)

    W, H = vr.rw.GetSize()
    ev("LeftButtonPressEvent", W // 2, H // 2)
    ev("MouseMoveEvent", W // 2 + 40, H // 2 + 20)
    ev("MouseMoveEvent", W // 2 + 80, H // 2 + 40)
    ev("LeftButtonReleaseEvent", W // 2 + 80, H // 2 + 40)
    moved = np.linalg.norm(np.array(cam.GetFocalPoint()) - f0)
    check(moved > 1 and np.allclose(dir0, cam.GetDirectionOfProjection(), atol=1e-6) and np.allclose(up0, cam.GetViewUp()),
          f"left drag pans {moved:.1f} mm without rotating")
    vw.rot_acts["free"][0].trigger()

    # preset with veins re-encodes; other presets go back
    from dcv import tf_model
    av = next(p for p in tf_model.builtin_presets() if "Veins" in p.name)
    vw.set_tf(av)
    vw.apply_tf()
    check(sv.state.encode_veins and vw.artery_spin.isVisibleTo(vw), "Arteries / Veins preset turns on the vein layer")
    plain = next(p for p in tf_model.builtin_presets() if p.name == "CT Angio - Vessels + Bone")
    vw.set_tf(plain)
    vw.apply_tf()
    check(not sv.state.encode_veins and sv.state.render_data.max() < segment.VEIN_OFFSET, "other presets turn it off")

    # seed range editor: the handles re-grow the segment from its click
    vol2 = sv.state.vol
    z, y, x = np.unravel_index(np.argmax((vol2.data > 45) & (vol2.data < 60)), vol2.shape)
    vw.seg_mode.setCurrentIndex(vw.seg_mode.findData("manual"))
    vw.seg_crop.setChecked(False)
    v0 = float(vol2.data[z, y, x])
    vw.seg_lo.setValue(v0 - 15)
    vw.seg_hi.setValue(v0 + 15)
    vw.seed(int(z), int(y), int(x))
    check(wait_for(lambda: sv.state.active_segment is not None), "seed made a segment")
    seg = sv.state.active_segment
    ed = vw.seed_range
    vw.tabs.setCurrentIndex(1)
    pump(0.3)
    check(ed.isVisible(), "range editor shown for a seed segment")
    check(abs(ed.lo - (v0 - 15)) < 1 and abs(ed.hi - (v0 + 15)) < 1, f"handles at the segment's range {ed.lo:.0f}…{ed.hi:.0f}")
    ml0 = seg.volume_ml
    yc = ed.height() // 2
    x1 = int(ed._x(ed.hi))
    QTest.mousePress(ed, Qt.LeftButton, Qt.NoModifier, QPoint(x1, yc))
    QTest.mouseMove(ed, QPoint(x1 + 8, yc))
    QTest.mouseMove(ed, QPoint(x1 + 16, yc))
    QTest.mouseMove(ed, QPoint(x1 + 24, yc))
    QTest.mouseRelease(ed, Qt.LeftButton, Qt.NoModifier, QPoint(x1 + 24, yc))
    check(ed.hi > v0 + 15, f"dragging the handle widens the range to {ed.hi:.0f}")
    ok = wait_for(lambda: vw._range_job is None and vw._range_want is None and sv.state.active_segment is not seg, 60)
    new = sv.state.active_segment
    check(ok and new.seed == seg.seed and new.range[1] > v0 + 15, f"segment re-grown, {ml0:.2f} -> {new.volume_ml:.2f} ml")
    check(new.volume_ml >= ml0, "a wider range keeps at least the same volume")
    check(len(sv.state.segments) == 1, "replaces the segment, no new one")
    # the clicked value cannot leave the range
    x0 = int(ed._x(ed.lo))
    QTest.mousePress(ed, Qt.LeftButton, Qt.NoModifier, QPoint(x0, yc))
    QTest.mouseMove(ed, QPoint(ed.width() - 2, yc))
    QTest.mouseRelease(ed, Qt.LeftButton, Qt.NoModifier, QPoint(ed.width() - 2, yc))
    check(ed.lo <= ed.value, "the clicked value stays inside the range")
    wait_for(lambda: vw._range_job is None and vw._range_want is None, 60)
    vw.close()

    # collage in another plane
    state0 = (sv.orientation, sv.index, sv.rot)
    choices = dict(collage.plane_choices(sv))
    check(set(choices) == {"Axial", "Coronal", "Sagittal"}, f"collage offers {sorted(choices)}")
    dlg = collage.CollageDialog(w, sv)
    check(dlg.plane.currentText() == "Axial", "default = the plane on screen")
    axial_n = dlg.n
    a0 = dlg.image()
    dlg.plane.setCurrentIndex(dlg.plane.findText("Coronal"))
    check(dlg.n == sv.vol.shape[1] and dlg.n != axial_n, f"coronal: {dlg.n} positions")
    a1 = dlg.image()
    check(a1 is not None and (a1.width(), a1.height()) != (a0.width(), a0.height()) or a1 != a0, "coronal tiles differ")
    check("CORONAL" in dlg.title(), "title names the plane")
    check((sv.orientation, sv.index, sv.rot) == state0, "the viewport itself did not change")
    a1.save(os.path.join(out_dir, "collage_coronal.png"))
    dlg.close()
except Exception:
    traceback.print_exc()
    errors.append("exception")
print("errors:", len(errors))
