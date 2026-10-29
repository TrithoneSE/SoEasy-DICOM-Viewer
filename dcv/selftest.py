"""`--selftest report.txt [data folder]`: check that an installed / frozen build has everything it needs.

Without a folder: pixel decoders, VTK off-screen volume rendering (pixels drawn), archive extraction, every GUI
module. With a folder of sample studies (the repository's `samples`): the same windows a user opens — series
scan and load, 2D, MPR, 3D VR, DSA on a synthetic run, DTI tensor fit and tractography, a snapshot movie (GIF).
Each line of the report is OK or FAILED; the build script refuses a build with a FAILED line.
"""

from __future__ import annotations

import os
import tempfile
import time
import traceback

import numpy as np


class Report:
    def __init__(self):
        self.lines = []
        self.failed = False

    def step(self, name, fn):
        t = time.perf_counter()
        try:
            detail = fn()
            self.lines.append(f"OK      {name}{': ' + str(detail) if detail else ''} ({time.perf_counter() - t:.1f} s)")
        except Exception as ex:  # noqa: BLE001 - every failure goes into the report
            self.failed = True
            self.lines.append(f"FAILED  {name}: {ex}\n" + "".join(traceback.format_exception(ex)))

    def write(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines) + "\n")


def _decoders():
    from pydicom.pixels import get_decoder
    from pydicom.uid import JPEG2000Lossless, JPEGBaseline8Bit, JPEGLosslessSV1, RLELossless
    out = []
    for uid in (JPEGBaseline8Bit, JPEGLosslessSV1, JPEG2000Lossless, RLELossless):
        dec = get_decoder(uid)
        if not dec.is_available:
            raise RuntimeError(f"no decoder for {uid.name}")
        out.append(f"{uid.name.split(' (')[0].split(',')[0]} ({', '.join(dec.available_plugins)})")
    return "; ".join(out)


def _opengl():
    from . import graphics
    from . import vtk_lite as vtk
    from vtkmodules.util import numpy_support
    rw = vtk.vtkRenderWindow()
    rw.SetOffScreenRendering(1)
    rw.SetSize(160, 160)
    ren = vtk.vtkRenderer()
    rw.AddRenderer(ren)
    z, y, x = np.mgrid[:48, :48, :48]
    a = (np.hypot(np.hypot(x - 24, y - 24), z - 24) < 18).astype(np.int16) * 1000
    img = vtk.vtkImageData()
    img.SetDimensions(48, 48, 48)
    img.GetPointData().SetScalars(numpy_support.numpy_to_vtk(a.ravel(), deep=True, array_type=vtk.VTK_SHORT))
    m = vtk.vtkGPUVolumeRayCastMapper()
    m.SetInputData(img)
    p = vtk.vtkVolumeProperty()
    c, o = vtk.vtkColorTransferFunction(), vtk.vtkPiecewiseFunction()
    c.AddRGBPoint(0, 0, 0, 0)
    c.AddRGBPoint(1000, 1, 0.4, 0.3)
    o.AddPoint(0, 0)
    o.AddPoint(1000, 0.3)
    p.SetColor(c)
    p.SetScalarOpacity(o)
    v = vtk.vtkVolume()
    v.SetMapper(m)
    v.SetProperty(p)
    ren.AddVolume(v)
    ren.ResetCamera()
    rw.Render()
    caps = [c.split(":", 1)[1].strip() for c in rw.ReportCapabilities().splitlines()[1:3]]
    f = vtk.vtkWindowToImageFilter()
    f.SetInput(rw)
    f.Update()
    lit = int((numpy_support.vtk_to_numpy(f.GetOutput().GetPointData().GetScalars()).max(axis=1) > 30).sum())
    rw.Finalize()
    if lit < 500:
        raise RuntimeError(f"volume rendering drew {lit} pixels ({' / '.join(caps)})")
    return f"{graphics.mode}, {' / '.join(caps)}, {lit} px"


def _archive(tmp):
    import zipfile
    from .archive import extract_archive
    src = os.path.join(tmp, "a.zip")
    with zipfile.ZipFile(src, "w") as z:
        z.writestr("folder/file.txt", "x")
    out = extract_archive(src)
    if not os.path.isfile(os.path.join(out, "folder", "file.txt")):
        raise RuntimeError("zip not extracted")
    tar = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "tar.exe")
    return "zip OK, .rar / .7z via " + ("Windows tar.exe" if os.path.isfile(tar) else "7-Zip / WinRAR if installed")


def _xa_run(folder, n=16, size=192):
    """Synthetic 10-bit RLE-compressed XA run with a vessel filling from frame 5 (DSA check)."""
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, RLELossless, generate_uid
    yy, xx = np.mgrid[:size, :size]
    bone = 500 + 250 * ((np.hypot(yy - size / 2, xx - size / 2) // 16) % 2)
    vessel = -200 * (np.abs(xx - 0.6 * yy - 30) < 3)
    frames = np.stack([bone + (vessel if k >= 4 else 0) for k in range(n)]).astype(np.uint16)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.12.1"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
    ds.Modality, ds.PatientName, ds.SeriesInstanceUID, ds.StudyInstanceUID = "XA", "SELFTEST", generate_uid(), "1.2"
    ds.Rows = ds.Columns = size
    ds.NumberOfFrames, ds.FrameTime, ds.PixelIntensityRelationship = n, 66.7, "LOG"
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 10, 9, 0
    ds.WindowCenter, ds.WindowWidth = 512, 1024
    ds.PixelData = frames.tobytes()
    ds.compress(RLELossless)
    os.makedirs(folder, exist_ok=True)
    ds.save_as(os.path.join(folder, "XA.dcm"), enforce_file_format=True)


def _with_data(rep, data, tmp, app):
    """The windows a user opens, on the sample studies."""
    from .loader import load_series, scan_folder
    from .main_window import MainWindow

    def pump(cond, timeout=120):
        t = time.time()
        while not cond() and time.time() - t < timeout:
            app.processEvents()
            time.sleep(0.03)
        return cond()

    head = os.path.join(data, "head_ct")
    w = MainWindow()
    w.resize(1200, 800)
    w.show()

    def open_study():
        w.open_paths([head])
        if not pump(lambda: w.viewports[0].state is not None):
            raise RuntimeError("study did not open")
        sv = w.viewports[0].slice
        sv.repaint()
        return f"{len(w.series)} series, {sv.vol.shape} {sv.vol.modality}"
    rep.step("open a study, 2D view", open_study)

    def mpr():
        w.open_mpr()
        app.processEvents()
        mw = next(iter(w.mpr_windows.values()))
        mw.repaint()
        app.processEvents()
        mw.close()
    rep.step("MPR window", mpr)

    def vr():
        st = w.viewports[0].state
        vw = w.vr_window(st)
        vw.resize(900, 700)
        app.processEvents()
        from . import snapshots
        rgb = snapshots.grab_rgb(vw.vr)
        lit = int((rgb.max(axis=-1) > 30).sum())
        if lit < 1000:
            raise RuntimeError(f"3D view drew {lit} pixels")
        snaps = [snapshots.capture(vw)]
        vw.vr.rotate(60)
        snaps.append(snapshots.capture(vw))
        gif = os.path.join(tmp, "movie.gif")
        n = snapshots.make_movie(vw, snaps, dict(frames=4, hold=0, fps=10, width=200, loop=False), gif)
        vw.close()
        return f"{lit} px lit, movie {n} frames"
    rep.step("3D VR window + snapshot movie", vr)

    def dsa():
        xa = os.path.join(tmp, "xa")
        _xa_run(xa)
        v = load_series(scan_folder(xa)[0])
        from .dsa import DSA
        d = DSA(v)
        sub = d.subtract(10)
        if not (sub[100, 90] < -150 and abs(float(np.median(sub))) < 2):
            raise RuntimeError(f"unexpected subtraction (vessel {sub[100, 90]}, median {np.median(sub)})")
        return f"{v.shape[0]} frames, frame time {v.frame_time:.0f} ms"
    rep.step("DSA on a compressed XA run", dsa)

    def dti():
        dwi_dir = os.path.join(data, "mri")
        if not os.path.isdir(dwi_dir):
            return "skipped (no samples/mri)"
        from .dti import TrackParams, fit_tensor, load_dwi, make_seeds, track
        info = next(s for s in scan_folder(dwi_dir) if s.is_dwi)
        tm = fit_tensor(load_dwi(info))
        p = TrackParams(max_seeds=2000)
        lines = track(tm, make_seeds(tm, p), p)
        if not lines:
            raise RuntimeError("no streamlines")
        return f"FA mean {float(tm.fa[tm.mask].mean()):.2f}, {len(lines)} streamlines"
    rep.step("DTI tensor fit + tractography", dti)
    w.close()


def run(path, data=None):
    rep = Report()
    rep.step("pixel decoders", _decoders)
    rep.step("OpenGL volume rendering", _opengl)
    with tempfile.TemporaryDirectory(prefix="soeasy_selftest_") as tmp:
        rep.step("archive extraction", lambda: _archive(tmp))
        rep.step("GUI modules", lambda: __import__("dcv.main_window") and "OK")
        if data:
            from PySide6.QtWidgets import QApplication
            app = QApplication.instance() or QApplication([])
            _with_data(rep, data, tmp, app)
    rep.write(path)
    return not rep.failed
