"""Scan DICOM folders, group the files into series and build numpy volumes."""

from __future__ import annotations

import os
import threading
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError
from pydicom.pixels import pixel_array


@dataclass
class SeriesInfo:
    uid: str
    files: list = field(default_factory=list)
    description: str = ""
    modality: str = ""
    patient: str = ""
    study_date: str = ""
    series_number: int = 0
    thumbnail: np.ndarray | None = None  # uint8 2D
    is_dwi: bool = False                 # diffusion-weighted (DTI) series
    derived: object = None               # VolumeState for derived series (FA, MD, ...)
    # study / patient identity for the grouped series list (patient header, then study header)
    study_uid: str = ""
    study_desc: str = ""
    study_time: str = ""
    patient_id: str = ""
    birth_date: str = ""
    slice_keys: list = field(default_factory=list)   # per file: position along the slice normal (or instance nr)
    frames: int = 0                      # multiframe file (XA run, US loop): its own entry with this many frames
    instance: int = 0

    def middle_file(self):
        """File of the spatially middle slice (file order is arbitrary), used for the thumbnail."""
        if len(self.slice_keys) == len(self.files) and self.files:
            order = sorted(range(len(self.files)), key=lambda i: self.slice_keys[i])
            return self.files[order[len(order) // 2]]
        return self.files[len(self.files) // 2]

    @property
    def count(self) -> int:
        return self.frames or len(self.files)

    def label(self) -> str:
        desc = self.description or "(no description)"
        return f"{self.modality} #{self.series_number}  {desc}\n{self.count} images"


def _direction_from_iop(iop):
    row = np.array(iop[:3], float)
    col = np.array(iop[3:], float)
    row /= np.linalg.norm(row)
    col /= np.linalg.norm(col)
    return np.stack([row, col, np.cross(row, col)], axis=1)


@dataclass
class Volume:
    data: np.ndarray            # (z, y, x)
    spacing: tuple              # (sz, sy, sx) mm
    origin: tuple               # (x, y, z) mm, posisi voxel pertama
    modality: str = ""
    description: str = ""
    patient: str = ""
    default_ww: float | None = None
    default_wl: float | None = None
    uid: str = ""
    files: list = field(default_factory=list)   # file per slice (z order), for DICOM tags
    # columns = patient (LPS) directions of +x (DICOM row direction), +y (column direction), +z (slice normal)
    direction: np.ndarray = field(default_factory=lambda: np.eye(3))
    rgb: np.ndarray | None = None               # optional (z, y, x, 3) uint8 color volume (e.g. color FA)
    frame_time: float | None = None             # ms between frames of a dynamic series (XA / US cine)
    dsa_hint: dict | None = None                # multiframe: what the header says about subtraction (dsa.py)
    loader: object = None                       # FrameLoader while compressed frames are still being decoded
    # patient coordinate system shared with other series (Frame of Reference UID, else "study:<uid>"); empty =
    # no usable geometry (XA run, US loop). Reference lines are drawn only between series with the same value.
    frame_uid: str = ""

    @property
    def shape(self):
        return self.data.shape

    def frame(self, k):
        """data[k], decoded first if the background decoder has not reached it yet."""
        if self.loader is not None:
            self.loader.wait(k)
        return self.data[k]

    def index_to_world(self, zyx):
        """(..., 3) array of (z, y, x) voxel indices -> (..., 3) patient LPS coordinates (mm)."""
        zyx = np.asarray(zyx, float)
        sz, sy, sx = self.spacing
        D = self.direction
        return (np.asarray(self.origin, float) + (zyx[..., 2:3] * sx) * D[:, 0] + (zyx[..., 1:2] * sy) * D[:, 1]
                + (zyx[..., 0:1] * sz) * D[:, 2])

    def world_to_index(self, p):
        """(..., 3) LPS coordinates -> (..., 3) float (z, y, x) voxel indices."""
        d = np.asarray(p, float) - np.asarray(self.origin, float)
        sz, sy, sx = self.spacing
        D = self.direction
        return np.stack([d @ D[:, 2] / sz, d @ D[:, 1] / sy, d @ D[:, 0] / sx], axis=-1)

    @property
    def value_range(self):
        if getattr(self, "_vr", None) is None:
            self._vr = (float(self.data.min()), float(self.data.max()))
        return self._vr


def _is_image(ds) -> bool:
    return "Rows" in ds and "Columns" in ds and "DirectoryRecordSequence" not in ds


def _first(v, default=None):
    if v is None or v == "":
        return default
    try:
        return float(v[0])
    except (TypeError, IndexError):
        try:
            return float(v)
        except (TypeError, ValueError):
            return default


def scan_folder(paths, progress=None, cancelled=lambda: False) -> list[SeriesInfo]:
    """Find every DICOM file (recursively) and group them per SeriesInstanceUID.
    Archives (.zip / .rar / .7z ...) given directly or found in the folders are extracted first (nested up to 3)."""
    from .archive import ArchiveError, extract_archive, is_archive
    if isinstance(paths, str):
        paths = [paths]
    files, arc_errors = [], []
    pending = [(p, 0) for p in paths]
    while pending:
        p, depth = pending.pop(0)
        if os.path.isfile(p):
            if is_archive(p) and depth < 3:
                try:
                    pending.append((extract_archive(p, progress, cancelled), depth + 1))
                except ArchiveError as ex:
                    if cancelled():
                        break
                    arc_errors.append(str(ex))
            else:
                files.append(p)
        else:
            for root, _, names in os.walk(p):
                for n in names:
                    f = os.path.join(root, n)
                    if is_archive(f) and depth < 3:
                        pending.append((f, depth))
                    else:
                        files.append(f)
                if progress:
                    progress(0, 0, f"Finding files... {len(files)}")     # total 0 = busy indicator

    series: dict[str, SeriesInfo] = {}
    total = len(files)
    for i, (f, ds) in enumerate(_read_parallel(files, _read_header, cancelled)):
        if progress and i % 20 == 0:
            progress(i, total, "Scanning files...")
        if ds is None or not _is_image(ds):
            continue
        from .dti import dwi_info
        uid = str(ds.get("SeriesInstanceUID", "")) or f"nouid-{os.path.dirname(f)}"
        frames = int(ds.get("NumberOfFrames", 1) or 1)
        if frames > 1:      # an XA run / US loop is a movie of its own (biplane A and B share one series)
            uid = f"{uid}#{ds.get('SOPInstanceUID', f)}"
        s = series.get(uid)
        if s is None:
            s = SeriesInfo(
                uid=uid,
                frames=frames if frames > 1 else 0,
                instance=int(ds.get("InstanceNumber", 0) or 0),
                description=str(ds.get("SeriesDescription", "")) + (_plane_suffix(ds) if frames > 1 else ""),
                modality=str(ds.get("Modality", "")),
                patient=str(ds.get("PatientName", "")),
                study_date=str(ds.get("StudyDate", "")),
                series_number=int(ds.get("SeriesNumber", 0) or 0),
                study_uid=str(ds.get("StudyInstanceUID", "")),
                study_desc=str(ds.get("StudyDescription", "")),
                study_time=str(ds.get("StudyTime", "")),
                patient_id=str(ds.get("PatientID", "")),
                birth_date=str(ds.get("PatientBirthDate", "")),
            )
            series[uid] = s
        s.files.append(f)
        s.slice_keys.append(_slice_key(ds))
        # MR only: Siemens XA files carry the same private tag numbers with another meaning
        if not s.is_dwi and len(s.files) <= 3 and s.modality == "MR" and dwi_info(ds) is not None:
            s.is_dwi = True

    result = sorted(series.values(), key=lambda s: (s.study_date, s.series_number, s.instance))
    if not result and arc_errors:
        raise RuntimeError("\n\n".join(arc_errors))
    for i, s in enumerate(result):
        if cancelled():
            break
        if progress:
            progress(i, len(result), "Creating thumbnails...")
        s.thumbnail = _make_thumbnail(s.middle_file())
    if progress:
        progress(1, 1, "Done")
    return result


def _plane_suffix(ds):
    """' A' / ' B' for the two planes of a biplane run, ' (ref)' for a reference image (Siemens ImageType)."""
    it = [str(t).upper() for t in ds.get("ImageType", [])]
    if any("REFIMAGE" in t for t in it):
        return " (ref)"
    for t in it:
        for p in ("A", "B"):
            if t in (f"BIPLANE {p}", f"SINGLE {p}"):
                return f" {p}"
    return ""


def _read_header(f):
    try:
        return pydicom.dcmread(f, stop_before_pixels=True)
    except (InvalidDicomError, OSError, Exception):
        return None


def _read_image(f):
    try:
        ds = pydicom.dcmread(f)
        arr = _to_gray(ds, ds.pixel_array)
        del ds.PixelData            # keep only the decoded array (halves peak memory)
        return ds, arr
    except Exception:
        return None


def _read_parallel(files, fn, cancelled=lambda: False, workers=8):
    """Yield (file, fn(file)) in file order, reading several files at once. The first read of a folder is
    dominated by the disk and the virus scanner opening each file (~7 ms per file, 10× a cached read), which
    threads overlap; parsing itself holds the GIL, so more threads do not help."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(workers) as pool:
        for start in range(0, len(files), 256):
            if cancelled():
                return
            chunk = files[start:start + 256]
            yield from zip(chunk, pool.map(fn, chunk))


def _slice_key(ds):
    """Position of a slice along its normal (mm); falls back to the instance number."""
    iop, ipp = ds.get("ImageOrientationPatient"), ds.get("ImagePositionPatient")
    try:
        if iop is not None and ipp is not None:
            n = np.cross(np.array(iop[:3], float), np.array(iop[3:], float))
            return float(np.dot(n, np.array(ipp, float)))
    except (TypeError, ValueError):
        pass
    return float(ds.get("InstanceNumber", 0) or 0)


def thumbnail_window(ds, arr):
    """(low, high) display range for a thumbnail: the window the scanner stored in the DICOM
    header (WindowCenter / WindowWidth, first value); without one, the 1–99 % percentile range of the image."""
    ww, wl = _window_from(ds)
    if ww and ww > 1 and wl is not None:
        return wl - ww / 2, wl + ww / 2
    lo, hi = np.percentile(arr, (1, 99))
    return float(lo), float(hi if hi > lo else lo + 1)


def _make_thumbnail(path, size=128):
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        n = int(ds.get("NumberOfFrames", 1) or 1)
        # one frame only: decoding a whole compressed XA run took ~8 s per thumbnail
        arr = pixel_array(path, index=n // 2) if n > 1 else pixel_array(path)
        if arr.ndim == 3 and arr.shape[-1] in (3, 4) and ds.get("SamplesPerPixel", 1) > 1:
            arr = arr[..., :3].mean(axis=-1)
        elif arr.ndim == 3:
            arr = arr[arr.shape[0] // 2]
        arr = arr.astype(np.float32)
        slope = float(ds.get("RescaleSlope", 1) or 1)       # the stored window is in rescaled units (HU)
        inter = float(ds.get("RescaleIntercept", 0) or 0)
        arr = arr * slope + inter
        lo, hi = thumbnail_window(ds, arr)
        img = np.clip((arr - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
        if str(ds.get("PhotometricInterpretation", "")) == "MONOCHROME1":
            img = 255 - img
        step = max(1, max(img.shape) // size)
        return np.ascontiguousarray(img[::step, ::step])
    except Exception:
        return None


def _rescale(ds, arr):
    slope = float(ds.get("RescaleSlope", 1) or 1)
    inter = float(ds.get("RescaleIntercept", 0) or 0)
    arr = arr.astype(np.float32)
    if slope != 1 or inter != 0:
        arr = arr * slope + inter
    if str(ds.get("PhotometricInterpretation", "")) == "MONOCHROME1":
        arr = arr.max() + arr.min() - arr
    return arr


def _to_gray(ds, arr):
    if ds.get("SamplesPerPixel", 1) > 1 and arr.shape[-1] in (3, 4):
        return arr[..., :3].mean(axis=-1)
    return arr


def _assemble(items):
    """Stack slices straight into the final array (int16 when the rescaled values are integral and fit),
    releasing each source slice as soon as it is copied. Peak memory ≈ raw + final instead of 4× the volume."""
    def lin(d):
        return float(d.get("RescaleSlope", 1) or 1), float(d.get("RescaleIntercept", 0) or 0)

    integral = all(sl == int(sl) and ic == int(ic) for sl, ic in (lin(d) for d, _ in items))
    if integral:
        lo = min(float(a.min()) * lin(d)[0] + lin(d)[1] for d, a in items)
        hi = max(float(a.max()) * lin(d)[0] + lin(d)[1] for d, a in items)
        mono1 = any(str(d.get("PhotometricInterpretation", "")) == "MONOCHROME1" for d, _ in items)
        integral = -32768 <= lo and hi <= 32767 and not mono1
    h, w = items[0][1].shape
    out = np.empty((len(items), h, w), np.int16 if integral else np.float32)
    for i, (d, a) in enumerate(items):
        out[i] = _rescale(d, a)
        items[i] = (d, None)
    return out


def _normal(ds):
    iop = ds.get("ImageOrientationPatient")
    try:
        n = np.cross(np.array(iop[:3], float), np.array(iop[3:], float))
        return n / np.linalg.norm(n)
    except (TypeError, ValueError, IndexError):
        return None


def _majority_orientation(items):
    """Keep the slices that share the most common orientation. Reformats made at the console (AX / SAG / COR,
    ~30 slices) often start with a reference image of another plane with the slice lines burnt in; sorting every
    slice along that image's normal collapsed the stack to 2 images."""
    normals = [_normal(d) for d, _ in items]
    if any(n is None for n in normals):
        return items
    N = np.array(normals)
    _, first = np.unique(np.round(N, 2), axis=0, return_index=True)
    if len(first) == 1:
        return items
    parallel = np.abs(N[first] @ N.T) > 0.999                      # within ~2.5°
    keep = parallel[int(np.argmax(parallel.sum(axis=1)))]
    return [it for it, k in zip(items, keep) if k]


def _finalize(vol: np.ndarray):
    lo, hi = float(vol.min()), float(vol.max())
    if lo >= -32768 and hi <= 32767 and np.allclose(vol, np.round(vol)):
        return vol.astype(np.int16)
    return vol.astype(np.float32)


def _window_from(ds):
    ww = _first(ds.get("WindowWidth"))
    wl = _first(ds.get("WindowCenter"))
    return ww, wl


def load_series(info: SeriesInfo, progress=None, cancelled=lambda: False) -> Volume:
    if len(info.files) == 1:
        return _load_multiframe(info)

    items = []
    n = len(info.files)
    for i, (f, r) in enumerate(_read_parallel(info.files, _read_image, cancelled)):
        if progress and i % 5 == 0:
            progress(i, n, "Loading images...")
        if r is not None and r[1].ndim == 2:
            items.append(r)
    if cancelled():
        raise RuntimeError("Cancelled")
    if not items:
        raise RuntimeError("No readable images in this series.")
    if progress:
        progress(0, 0, "Building volume...")

    # drop images of another size (e.g. a localizer)
    shape_mode = Counter(a.shape for _, a in items).most_common(1)[0][0]
    items = [(d, a) for d, a in items if a.shape == shape_mode]
    items = _majority_orientation(items)

    ds0 = items[0][0]
    iop = ds0.get("ImageOrientationPatient")
    # some derived maps (e.g. color FA) carry an all-zero orientation: no patient geometry
    geometric = (iop is not None and len(iop) == 6 and all("ImagePositionPatient" in d for d, _ in items)
                 and np.linalg.norm(np.cross(np.array(iop[:3], float), np.array(iop[3:], float))) > 0.5)
    if geometric:
        row = np.array(iop[:3], float)
        col = np.array(iop[3:], float)
        normal = np.cross(row, col)
        keyed = [(float(np.dot(normal, np.array(d.ImagePositionPatient, float))), d, a) for d, a in items]
        keyed.sort(key=lambda t: t[0])
        pos = np.array([k[0] for k in keyed])
        # drop duplicate positions
        uniq = [keyed[0]]
        for k in keyed[1:]:
            if abs(k[0] - uniq[-1][0]) > 1e-3:
                uniq.append(k)
        keyed = uniq
        pos = np.array([k[0] for k in keyed])
        sz = float(np.median(np.diff(pos))) if len(pos) > 1 else float(ds0.get("SliceThickness", 1) or 1)
        items = [(d, a) for _, d, a in keyed]
        origin = tuple(float(v) for v in items[0][0].ImagePositionPatient)
        direction = _direction_from_iop(iop)
    else:
        items.sort(key=lambda t: int(t[0].get("InstanceNumber", 0) or 0))
        sz = float(ds0.get("SpacingBetweenSlices", ds0.get("SliceThickness", 1)) or 1)
        origin = (0.0, 0.0, 0.0)
        direction = np.eye(3)

    ps = ds0.get("PixelSpacing", [1, 1])
    sy, sx = float(ps[0]), float(ps[1])
    vol = _assemble(items)
    ww, wl = _window_from(ds0)
    if progress:
        progress(n, n, "Done")
    return Volume(
        data=vol,
        spacing=(abs(sz) or 1.0, sy, sx),
        origin=origin,
        modality=info.modality,
        description=info.description,
        patient=info.patient,
        default_ww=ww,
        default_wl=wl,
        uid=info.uid,
        files=[str(getattr(d, "filename", "")) for d, _ in items],
        direction=direction,
        frame_uid=_frame_uid(ds0, info) if geometric else "",
    )


class FrameLoader:
    """Decodes the frames of a compressed multiframe file (XA run: ~80 ms per JPEG-lossless frame, 10 s for 120
    frames) in a background thread, so the run opens at once. A frame wanted before the thread reaches it is
    decoded on the spot (Volume.frame). The decoder holds the GIL, so more threads would not be faster."""

    def __init__(self, vol, path, slope, inter):
        self.vol, self.path, self.slope, self.inter = vol, path, slope, inter
        self.ready = np.zeros(vol.data.shape[0], bool)
        self.error = None

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _decode(self, k):
        a = pixel_array(self.path, index=int(k))
        if self.slope != 1 or self.inter != 0:
            a = a * self.slope + self.inter
        self.vol.data[k] = a
        self.ready[k] = True

    def wait(self, k):
        if not self.ready[k]:
            self._decode(k)

    def _run(self):
        try:
            for k in range(len(self.ready)):
                if self.vol.loader is not self:
                    return
                if not self.ready[k]:
                    self._decode(k)
        except Exception as ex:  # noqa: BLE001 - shown frames stay black; the error is kept for the log / tests
            self.error = ex
            return
        self.vol._vr = None
        self.vol.loader = None


def _geometry(ds):
    """(spacing (sz, sy, sx)) of a multiframe image."""
    sy = sx = 1.0
    ps = ds.get("PixelSpacing")
    try:
        pm = ds.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0]
        ps = pm.PixelSpacing
        sz = float(pm.get("SpacingBetweenSlices", pm.get("SliceThickness", 1)) or 1)
    except Exception:
        sz = float(ds.get("SpacingBetweenSlices", ds.get("SliceThickness", 1)) or 1)
    if ps is None:
        ps = ds.get("ImagerPixelSpacing")      # XA / RF: spacing at the detector (no magnification correction)
    if ps is not None:
        sy, sx = float(ps[0]), float(ps[1])
    return sz, sy, sx


def _load_multiframe(info: SeriesInfo) -> Volume:
    path = info.files[0]
    head = pydicom.dcmread(path, stop_before_pixels=True)
    n = int(head.get("NumberOfFrames", 1) or 1)
    progressive = (n > 4 and head.file_meta.TransferSyntaxUID.is_compressed and head.get("SamplesPerPixel", 1) == 1
                   and str(head.get("PhotometricInterpretation", "")) == "MONOCHROME2")
    if progressive:
        return _load_progressive(info, head, n)
    ds = pydicom.dcmread(path)
    arr = _to_gray(ds, ds.pixel_array)
    if arr.ndim == 2:
        arr = arr[None]
    vol = _rescale(ds, arr)
    ww, wl = _window_from(ds)
    from .dsa import hint_from_dataset
    return Volume(
        data=_finalize(vol),
        spacing=_geometry(ds),
        origin=(0.0, 0.0, 0.0),
        modality=info.modality,
        description=info.description,
        patient=info.patient,
        default_ww=ww,
        default_wl=wl,
        uid=info.uid,
        files=[info.files[0]] * vol.shape[0],
        frame_time=_frame_time(ds),
        dsa_hint=hint_from_dataset(ds, vol.shape[0]),
    )


def _load_progressive(info, head, n):
    """Compressed grayscale run: frame 1 and the mask frames now, the rest in the background (FrameLoader)."""
    from .dsa import hint_from_dataset
    slope = float(head.get("RescaleSlope", 1) or 1)
    inter = float(head.get("RescaleIntercept", 0) or 0)
    bits, signed = int(head.get("BitsStored", 16) or 16), int(head.get("PixelRepresentation", 0) or 0)
    lo = -(2 ** (bits - 1)) if signed else 0
    hi = 2 ** (bits - 1) - 1 if signed else 2 ** bits - 1
    small = slope == int(slope) and inter == int(inter) and -32768 <= min(lo * slope, hi * slope) + inter \
        and max(lo * slope, hi * slope) + inter <= 32767
    ww, wl = _window_from(head)
    hint = hint_from_dataset(head, n)
    vol = Volume(
        data=np.zeros((n, int(head.Rows), int(head.Columns)), np.int16 if small else np.float32),
        spacing=_geometry(head), origin=(0.0, 0.0, 0.0), modality=info.modality, description=info.description,
        patient=info.patient, default_ww=ww, default_wl=wl, uid=info.uid, files=[info.files[0]] * n,
        frame_time=_frame_time(head), dsa_hint=hint)
    vol.loader = FrameLoader(vol, info.files[0], slope, inter)
    for k in [0] + list(hint.get("mask") or []):
        vol.loader.wait(k)
    vol.loader.start()
    return vol


def _frame_uid(ds, info):
    """Frame of Reference UID (series that share it share patient coordinates), else the study."""
    uid = str(ds.get("FrameOfReferenceUID", "") or "")
    if uid:
        return uid
    return f"study:{info.study_uid}" if info.study_uid else ""


def _frame_time(ds):
    """Milliseconds per frame: Frame Time, else the mean of Frame Time Vector, else from Cine Rate /
    Recommended Display Frame Rate (frames per second)."""
    ft = _first(ds.get("FrameTime"))
    if ft and ft > 0:
        return ft
    vec = ds.get("FrameTimeVector")
    if vec is not None and len(vec) > 1:
        v = np.asarray(vec, float)[1:]               # the first entry is 0 by definition
        if (v > 0).any():
            return float(v[v > 0].mean())
    for tag in ("CineRate", "RecommendedDisplayFrameRate"):
        fps = _first(ds.get(tag))
        if fps and fps > 0:
            return 1000.0 / fps
    return None
