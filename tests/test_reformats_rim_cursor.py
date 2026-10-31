"""GUI test: console reformats with a reference image of another plane (all parallel slices kept), series list
header rows added while the scroll bar appears (several patients, DTI maps and the tractography window), rim
threshold on a fixed gradient axis, 3D pointer over the crop box / captions, arteries / veins preset.
Run via tests\\run_tests.bat (args: app_dir data_dir out_dir; samples/mri and samples/phantom next to data_dir)."""
import os
import shutil
import sys
import tempfile
import time
import traceback

app_dir, data_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
root = os.path.dirname(data_dir.rstrip("\\/"))
sys.path.insert(0, app_dir)
import builtins  # noqa: E402
_print = builtins.print
builtins.print = lambda *a, **k: _print(*a, **{**k, "flush": True})
import numpy as np  # noqa: E402
import pydicom  # noqa: E402
from dcv import vtk_lite as vtk  # noqa: E402
from vtkmodules.util import numpy_support  # noqa: E402
from PySide6.QtCore import Qt, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QListWidgetItem, QMessageBox  # noqa: E402
from dcv import tf_model  # noqa: E402
from dcv.loader import load_series, scan_folder  # noqa: E402
from dcv.main_window import MainWindow, SeriesList  # noqa: E402
from dcv.volume_view import RIM_GRAD_MAX  # noqa: E402

errors = []
sys.excepthook = lambda *a: (errors.append(a), traceback.print_exception(*a))
for _n in ("information", "warning", "question"):
    setattr(QMessageBox, _n, staticmethod(lambda *a, _n=_n, **k: (print("MSGBOX", _n, str(a[1:2])), QMessageBox.Yes)[1]))


def check(ok, what):
    print(("OK   " if ok else "FAIL ") + what)
    if not ok:
        errors.append(what)


def pump(cond, timeout=120):
    """Wait for worker threads without QTest.qWait (it keeps the GIL and starves them)."""
    t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        app.processEvents()
        time.sleep(0.05)
    for _ in range(5):
        app.processEvents()
    return cond()


# 1) a reformat whose first image is a reference image of another plane: keep the 11 parallel slices
tmp = tempfile.mkdtemp(prefix="dcv_reformat_")
try:
    names = sorted(os.listdir(data_dir))[:12]
    for i, n in enumerate(names):
        ds = pydicom.dcmread(os.path.join(data_dir, n))
        if i == 0:                                  # sagittal reference image with the slice lines
            ds.ImageOrientationPatient = [0, 1, 0, 0, 0, -1]
            ds.ImagePositionPatient = [0.0, -120.0, 150.0]
        ds.save_as(os.path.join(tmp, n))
    s = scan_folder(tmp)[0]
    v = load_series(s)
    check(s.count == 12 and v.shape[0] == 11, f"reformat with a reference image: {s.count} files -> "
                                                f"{v.shape[0]} slices (was 2)")
    check(0.5 < v.spacing[0] < 10, f"slice spacing from the parallel slices: {v.spacing[0]:.2f} mm")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
head = load_series(scan_folder(data_dir)[0])
pos = [float(np.dot(head.direction[:, 2], pydicom.dcmread(f, stop_before_pixels=True).ImagePositionPatient))
       for f in head.files]
check(len(pos) == head.shape[0] and np.all(np.diff(pos) > 0), "parallel file reading keeps the slice order")

# 1b) segmentation on CT-like noise: a 75 HU lesion in 40 HU brain, noise SD 9 (as on real contrast CT)
from dcv.segment import auto_grow  # noqa: E402
rng = np.random.default_rng(5)
zz, yy, xx = np.mgrid[:60, :80, :80]
sphere = (zz - 30) ** 2 + (yy - 40) ** 2 + (xx - 40) ** 2 < 15 ** 2
noisy = (np.where(sphere, 75.0, 40.0) + rng.normal(0, 9, sphere.shape)).astype(np.int16)
true_n = int(sphere.sum())
for smooth, name in ((0.0, "raw"), (1.0, "smoothed")):
    (sub, off), (lo, hi) = auto_grow(noisy, (1.0, 1.0, 1.0), (30, 40, 40), 2.5, 2, None, smooth)
    frac = sub.sum() / true_n
    print(f"     {name}: {frac:.2f} of the lesion, range {lo:.0f}..{hi:.0f}")
    if smooth:
        check(0.85 < frac < 1.15, f"smoothed region growing keeps a noisy lesion whole ({frac:.2f} of its volume)")

app = QApplication(sys.argv)

# 2) header rows: addItem() can resize the list before the label widget exists (AttributeError, list half built)
lst = SeriesList()
lst.resize(140, 260)
lst.show()
for i in range(12):
    lst.add_header("patient" if i % 3 == 0 else "study", f"<b>HEADER {i}</b><br>line")
    lst.addItem(QListWidgetItem(f"series {i}"))
app.processEvents()
heads = [lst.item(i) for i in range(lst.count()) if lst.item(i).data(Qt.UserRole + 1)]
check(not errors and all(lst.itemWidget(h) is not None for h in heads) and lst.verticalScrollBar().isVisible(),
      f"header rows added while the scroll bar appears ({len(heads)} headers, {len(errors)} errors)")
lst.close()

w = MainWindow()
w.resize(1500, 950)
w.show()
w.open_paths([data_dir])
pump(lambda: w.viewports[0].state is not None)
w._ask_open_target = lambda new: "here"
for extra in ("contoh_phantom", "mri"):
    n0 = len(w.series)
    w.open_paths([os.path.join(root, extra)])
    pump(lambda: len(w.series) > n0)
kinds = [w.series_list.item(i).data(Qt.UserRole + 1) for i in range(w.series_list.count())]
check(len(w.series_list.series_items()) == len(w.series) and kinds.count("patient") >= 2,
      f"several patients in one window: {len(w.series_list.series_items())}/{len(w.series)} thumbnails, "
      f"{kinds.count('patient')} patient headers")
check(not w.series_list.signalsBlocked(), "series list signals unblocked after rebuilding")
w.series_list.grab().save(os.path.join(out_dir, "series_list.png"))

# 3) DTI with a full series list: maps added and the tractography window opens
n0 = len(w.series)
w.open_dti()
pump(lambda: w.tract_windows, 180)
added = [s.modality for s in w.series.values() if s.derived is not None]
check(len(w.tract_windows) == 1 and len(added) == 4, f"DTI: tract window {len(w.tract_windows)}, maps {added}")
check(len(w.series_list.series_items()) == len(w.series), "DTI maps listed")
for tw in w.tract_windows:
    tw.close()


def grab(vr):
    vr.rw.Render()
    f = vtk.vtkWindowToImageFilter()
    f.SetInput(vr.rw)
    f.ReadFrontBufferOff()
    f.Update()
    return numpy_support.vtk_to_numpy(f.GetOutput().GetPointData().GetScalars()).astype(float)


def run3d():
    st = next(s for s in w.states.values() if s.vol.uid == head.uid)
    vw = w.vr_window(st)
    vw.resize(1300, 850)
    app.processEvents()
    vr, iren = vw.vr, vw.vr.iren

    # 4) rim threshold: fixed gradient axis (VTK scaled it by a quarter of the data range: ~27 HU per row with
    #    the bone lock, so 30–300 HU all looked the same)
    sp = vr.volume.GetShaderProperty()
    check(sp.GetNumberOfShaderReplacements() >= 1, "gradient axis shader replacement installed")
    tf = next(t for t in tf_model.builtin_presets() if t.name.startswith("CT - Enhancing"))
    vw.bone_chk.setChecked(True)
    app.processEvents()
    vw.set_tf(tf)
    vw.state.tf.edge_threshold = 20
    vw.apply_tf()
    img = vr._tf2d_image
    W, H, _ = img.GetDimensions()
    rgba = numpy_support.vtk_to_numpy(img.GetPointData().GetScalars()).reshape(H, W, 4)
    rmin, rmax = vr.image.GetScalarRange()
    col = int((110 - rmin) / (rmax - rmin) * W)            # 110 HU: inside the lesion curve
    row = lambda g: int(g / RIM_GRAD_MAX * H)             # noqa: E731
    check(rgba[row(5), col, 3] > 0 and rgba[row(40), col, 3] == 0,
          f"T = 20 HU with bone lock: lesion kept at 5 HU gradient ({rgba[row(5), col, 3]:.3f}), "
          f"hidden at 40 ({rgba[row(40), col, 3]:.3f})")
    vw.state.tf.curves = [c for c in vw.state.tf.curves if c.layer != "bone"]     # see the lesion
    vr.view("left")
    shots = {}
    for t in (8, 40, 120, 200):
        vw.state.tf.edge_threshold = t
        vw.apply_tf()
        shots[t] = grab(vr)
    changed = lambda a, b: np.mean(np.abs(shots[a] - shots[b]).max(axis=1) > 10) * 100     # noqa: E731
    check(changed(8, 40) > 0.1, f"rim threshold 8 vs 40 HU changes the picture ({changed(8, 40):.2f} % pixels)")
    print(f"     (40 vs 120: {changed(40, 120):.2f} %, 120 vs 200: {changed(120, 200):.2f} %)")
    vw.bone_chk.setChecked(False)
    app.processEvents()

    # 5) arteries / veins preset: two separate vessel curves
    av = next(t for t in tf_model.builtin_presets() if "Arteries / Veins" in t.name)
    names = [c.name for c in av.curves if c.layer != "bone"]
    check(names == ["Veins", "Arteries"], f"arteries / veins preset curves: {names}")
    vw.set_tf(av)
    vw.apply_tf()

    # 6) pointer: crop box handles / faces, captions
    W, H = vr.rw.GetSize()
    cx, cy = W // 2, H // 2

    def ev(name, x, y):
        iren.SetEventInformation(int(x), int(y), 0, 0, "\0", 0)
        iren.InvokeEvent(name)
        app.processEvents()

    shape = lambda: vr.vtk_widget.cursor().shape()        # noqa: E731
    vr.view("anterior")
    vw.crop_act.setChecked(True)
    app.processEvents()
    vr.reset_crop()
    vr.render()
    rep = vr.box_widget.GetRepresentation()
    pd = vtk.vtkPolyData()
    rep.GetPolyData(pd)
    fx, fy = vr._px(np.array(pd.GetPoint(9)))             # +x face handle: left / right on screen from the front
    ev("MouseMoveEvent", fx, fy)
    check(shape() == Qt.SizeHorCursor, f"hover +x face handle (front view): resize arrows ({shape()})")
    zx, zy = vr._px(np.array(pd.GetPoint(13)))            # +z face handle: up / down
    ev("MouseMoveEvent", zx, zy)
    check(shape() == Qt.SizeVerCursor, f"hover +z face handle: vertical arrows ({shape()})")
    ox, oy = vr._px(np.array(pd.GetPoint(14)))            # front face handle over the centre: faces the viewer
    ev("MouseMoveEvent", ox, oy)
    check(shape() == Qt.SizeAllCursor, f"hover the handle facing the viewer: move cursor ({shape()})")
    cam = vr.renderer.GetActiveCamera()
    cam.Azimuth(35)
    cam.Elevation(25)
    cam.OrthogonalizeViewUp()
    vr.render()
    ox, oy = vr._px(np.array(pd.GetPoint(14)))
    ev("MouseMoveEvent", ox, oy)
    check(shape() == Qt.SizeAllCursor and vr._box_state(ox, oy) == 7, f"hover the centre handle: move cursor "
                                                                        f"({shape()}, state {vr._box_state(ox, oy)})")
    vr.view("anterior")
    ev("MouseMoveEvent", 30, 30)
    check(shape() == Qt.BitmapCursor, f"away from the box: rotate icon ({shape()})")
    ev("MouseMoveEvent", fx, fy)
    ev("LeftButtonPressEvent", fx, fy)
    ev("MouseMoveEvent", fx - 60, fy)
    check(shape() == Qt.SizeHorCursor, "dragging the face keeps the resize arrows")
    ev("LeftButtonReleaseEvent", fx - 60, fy)
    ev("MouseMoveEvent", 30, 30)
    vr.reset_crop()
    vw.crop_act.setChecked(False)
    app.processEvents()

    p = vr.pick_world(cx, cy)
    if p is None:
        p = np.array([(a + b) / 2 for a, b in zip(vr.image.GetBounds()[::2], vr.image.GetBounds()[1::2])])
    vr.add_measurement3d("length", [p, p + np.array([30.0, 0, 0])], "")
    vr.render()
    m = vr.measurements3d[-1]
    r = vr._label_rect(m)
    lx, ly = (r[0] + r[2]) / 2, (r[1] + r[3]) / 2
    ev("MouseMoveEvent", lx, ly)
    check(shape() == Qt.OpenHandCursor, f"hover a 3D caption: open hand ({shape()})")
    ev("LeftButtonPressEvent", lx, ly)
    ev("MouseMoveEvent", lx + 30, ly + 20)
    check(shape() == Qt.ClosedHandCursor, f"dragging the caption: closed hand ({shape()})")
    ev("LeftButtonReleaseEvent", lx + 30, ly + 20)
    hx, hy = vr._px(m.pts[1])
    ev("MouseMoveEvent", hx, hy)
    check(shape() == Qt.PointingHandCursor, f"hover a measurement point: hand ({shape()})")
    ev("MouseMoveEvent", 30, 30)
    vw.select_measure("length")
    app.processEvents()
    ev("MouseMoveEvent", 30, 30)
    check(shape() == Qt.BitmapCursor or shape() == Qt.CrossCursor, f"measuring tool: tool pointer ({shape()})")
    vw.grab().save(os.path.join(out_dir, "vr.png"))
    vw.close()


def finish():
    try:
        run3d()
    except Exception:
        traceback.print_exc()
        errors.append("exception")
    QTimer.singleShot(300, app.quit)


QTimer.singleShot(300, finish)
app.exec()
print("errors:", len(errors))
