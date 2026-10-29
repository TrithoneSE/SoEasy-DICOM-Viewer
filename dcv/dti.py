"""Diffusion MRI: DWI DICOM loading (standard / Siemens / GE / Philips tags, Siemens mosaic), tensor fitting,
FA / MD / color-FA maps and deterministic streamline tractography.

Everything here is plain NumPy. Coordinates:
  - voxel arrays are (z, y, x)
  - "voxel mm space" = (x*sx, y*sy, z*sz), the space the 3D views work in
  - gradient directions and eigenvectors are expressed along the image axes (x, y, z) in mm space
  - patient space is DICOM LPS; Volume.direction maps image axes -> LPS
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import numpy as np
import pydicom

from .loader import Volume, _direction_from_iop, _rescale


# ============================================================ DICOM tags
def _num(v, count=1):
    """Read numbers from a DICOM value that may be a number, a list, a string or raw bytes (private tags)."""
    if v is None:
        return None
    if isinstance(v, (bytes, bytearray)):
        b = bytes(v)
        if len(b) == 8 * count:
            return list(struct.unpack(f"<{count}d", b))
        if len(b) == 4 * count:
            return list(struct.unpack(f"<{count}f", b))
        try:
            txt = b.decode("ascii", "ignore").replace("\x00", "").strip()
            parts = [p for p in txt.replace("\\", " ").split() if p]
            return [float(p) for p in parts[:count]]
        except ValueError:
            return None
    if isinstance(v, (list, tuple, pydicom.multival.MultiValue)):
        return [float(x) for x in list(v)[:count]]
    try:
        return [float(v)]
    except (TypeError, ValueError):
        return None


def _tag(ds, group, elem):
    el = ds.get((group, elem))
    return None if el is None else el.value


def dwi_info(ds):
    """(b-value, gradient [3] or None, gradient frame) for one DWI image, or None when it is not diffusion data.
    gradient frame: 'patient' (LPS) or 'image'."""
    # DICOM standard (classic or enhanced MR)
    b = _num(_tag(ds, 0x0018, 0x9087))
    if b is None:
        try:
            mr = ds.SharedFunctionalGroupsSequence[0].MRDiffusionSequence[0]
            b = [float(mr.DiffusionBValue)]
        except Exception:
            pass
    if b is not None:
        g = _num(_tag(ds, 0x0018, 0x9089), 3)
        if g is None:
            try:
                g = [float(x) for x in ds[0x0018, 0x9076][0][0x0018, 0x9089].value]
            except Exception:
                g = None
        return b[0], g, "patient"
    manu = str(ds.get("Manufacturer", "")).upper()
    # Siemens (CSA private fields exposed as tags)
    if "SIEMENS" in manu or (0x0019, 0x100C) in ds:
        b = _num(_tag(ds, 0x0019, 0x100C))
        if b is not None:
            return b[0], _num(_tag(ds, 0x0019, 0x100E), 3), "patient"
    # GE
    if "GE" in manu and (0x0043, 0x1039) in ds:
        b = _num(_tag(ds, 0x0043, 0x1039), 4)
        if b:
            bval = b[0] % 1e9 if b[0] >= 1e9 else b[0]
            g = [(_num(_tag(ds, 0x0019, t)) or [0.0])[0] for t in (0x10BB, 0x10BC, 0x10BD)]
            return bval, g, "image"
    # Philips
    if (0x2001, 0x1003) in ds:
        b = _num(_tag(ds, 0x2001, 0x1003))
        if b is not None:
            g = [(_num(_tag(ds, 0x2005, t)) or [0.0])[0] for t in (0x10B0, 0x10B1, 0x10B2)]
            return b[0], g, "patient"
    return None


@dataclass
class DWIData:
    data: np.ndarray        # (V, z, y, x) float32
    bvals: np.ndarray       # (V,)
    bvecs: np.ndarray       # (V, 3) unit vectors in the given frame (zeros for b0)
    frame: str              # 'patient' or 'image'
    spacing: tuple          # (sz, sy, sx)
    origin: tuple
    direction: np.ndarray
    description: str
    patient: str
    uid: str

    def volume(self, data, modality, description, rgb=None):
        v = Volume(data=data, spacing=self.spacing, origin=self.origin, modality=modality,
                   description=description, patient=self.patient, uid=f"{self.uid}-{modality}",
                   direction=self.direction, rgb=rgb)
        return v

    def bvecs_image(self, flip=(1, 1, 1)):
        """Gradient directions along the image axes (x, y, z)."""
        g = self.bvecs if self.frame == "image" else self.bvecs @ self.direction
        g = g * np.array(flip, float)
        n = np.linalg.norm(g, axis=1, keepdims=True)
        return np.where(n > 1e-6, g / np.maximum(n, 1e-6), 0.0)


def _unmosaic(ds, arr):
    n = int((_num(_tag(ds, 0x0019, 0x100A)) or [0])[0])
    rows, cols = arr.shape
    if n <= 1:
        return None
    tiles = int(math.ceil(math.sqrt(n)))
    th, tw = rows // tiles, cols // tiles
    sl = [arr[(i // tiles) * th:(i // tiles + 1) * th, (i % tiles) * tw:(i % tiles + 1) * tw] for i in range(n)]
    return np.stack(sl), th, tw


def load_dwi(info, progress=None):
    entries = []
    n = len(info.files)
    for i, f in enumerate(info.files):
        if progress and i % 10 == 0:
            progress(i, n, "Loading diffusion images...")
        try:
            ds = pydicom.dcmread(f)
            arr = ds.pixel_array
        except Exception:
            continue
        di = dwi_info(ds)
        if di is None or arr.ndim != 2:
            continue
        entries.append((ds, arr, di))
    if not entries:
        raise RuntimeError("No diffusion-weighted images with b-value tags were found in this series.")
    ds0 = entries[0][0]
    D = _direction_from_iop(ds0.ImageOrientationPatient)
    ps = ds0.get("PixelSpacing", [1, 1])
    sy, sx = float(ps[0]), float(ps[1])
    mosaic = "MOSAIC" in [str(t).upper() for t in ds0.get("ImageType", [])]

    vols, bvals, bvecs = [], [], []
    if mosaic:
        entries.sort(key=lambda e: (int(e[0].get("AcquisitionNumber", 0) or 0), int(e[0].get("InstanceNumber", 0) or 0)))
        sz = float(ds0.get("SpacingBetweenSlices", ds0.get("SliceThickness", 1)) or 1)
        for ds, arr, (b, g, frame) in entries:
            um = _unmosaic(ds, arr)
            if um is None:
                continue
            stack, th, tw = um
            vols.append(_rescale(ds, stack))
            bvals.append(b)
            bvecs.append(g if g is not None else [0, 0, 0])
        _, th, tw = _unmosaic(ds0, entries[0][1])
        rows, cols = entries[0][1].shape
        ipp = np.array(ds0.ImagePositionPatient, float)
        origin = ipp + D[:, 0] * sx * (cols - tw) / 2 + D[:, 1] * sy * (rows - th) / 2
    else:
        normal = D[:, 2]
        bypos = {}
        for ds, arr, di in entries:
            pos = round(float(np.dot(normal, np.array(ds.ImagePositionPatient, float))), 2)
            order = (int(ds.get("AcquisitionNumber", 0) or 0), int(ds.get("InstanceNumber", 0) or 0))
            bypos.setdefault(pos, []).append((order, ds, arr, di))
        positions = sorted(bypos)
        nvol = min(len(v) for v in bypos.values())
        for v in bypos.values():
            v.sort(key=lambda t: t[0])
        sz = float(np.median(np.diff(positions))) if len(positions) > 1 else float(ds0.get("SliceThickness", 1) or 1)
        for k in range(nvol):
            vols.append(np.stack([_rescale(bypos[p][k][1], bypos[p][k][2]) for p in positions]))
            b, g, frame = bypos[positions[0]][k][3]
            bvals.append(b)
            bvecs.append(g if g is not None else [0, 0, 0])
        first = bypos[positions[0]][0][1]
        origin = np.array(first.ImagePositionPatient, float)
    frame = entries[0][2][2]
    data = np.stack(vols).astype(np.float32)
    if progress:
        progress(n, n, "Done")
    return DWIData(data=data, bvals=np.array(bvals, float), bvecs=np.array(bvecs, float), frame=frame,
                   spacing=(abs(sz), sy, sx), origin=tuple(float(v) for v in origin), direction=D,
                   description=info.description, patient=info.patient, uid=info.uid)


# ======================================================== tensor fitting
@dataclass
class TensorMaps:
    fa: np.ndarray          # (z, y, x) float32
    md: np.ndarray          # mm²/s
    v1: np.ndarray          # (z, y, x, 3) principal eigenvector along image axes (x, y, z)
    b0: np.ndarray
    mask: np.ndarray
    dwi: DWIData
    flip: tuple

    def color_fa(self):
        """RGB (z, y, x, 3) uint8: red = left-right, green = anterior-posterior, blue = superior-inferior."""
        v_lps = self.v1 @ self.dwi.direction.T
        rgb = np.abs(v_lps) * np.clip(self.fa, 0, 1)[..., None]
        return np.clip(rgb * 255 * 1.4, 0, 255).astype(np.uint8)


def fit_tensor(dwi: DWIData, flip=(1, 1, 1), progress=None):
    b = dwi.bvals
    is_b0 = b < 50
    if not is_b0.any():
        raise RuntimeError("No b0 (b≈0) image in this diffusion series.")
    use = b <= 3500
    if (use & ~is_b0).sum() < 6:
        raise RuntimeError("At least 6 diffusion directions are needed for a tensor fit.")
    g = dwi.bvecs_image(flip)[use]
    bb = b[use]
    B = np.stack([np.ones_like(bb), -bb * g[:, 0] ** 2, -bb * g[:, 1] ** 2, -bb * g[:, 2] ** 2,
                  -2 * bb * g[:, 0] * g[:, 1], -2 * bb * g[:, 0] * g[:, 2], -2 * bb * g[:, 1] * g[:, 2]], axis=1)
    Binv = np.linalg.pinv(B)
    data = dwi.data[use]
    b0 = dwi.data[is_b0].mean(axis=0)
    thr = 0.12 * np.percentile(b0, 99)
    mask = b0 > thr
    nz = b0.shape[0]
    fa = np.zeros(b0.shape, np.float32)
    md = np.zeros(b0.shape, np.float32)
    v1 = np.zeros(b0.shape + (3,), np.float32)
    for z in range(nz):
        if progress:
            progress(z, nz, "Fitting tensors...")
        m = mask[z]
        if not m.any():
            continue
        S = np.maximum(data[:, z][:, m], 1.0)
        p = Binv @ np.log(S)                       # (7, M)
        Dxx, Dyy, Dzz, Dxy, Dxz, Dyz = p[1], p[2], p[3], p[4], p[5], p[6]
        T = np.stack([np.stack([Dxx, Dxy, Dxz], -1), np.stack([Dxy, Dyy, Dyz], -1),
                      np.stack([Dxz, Dyz, Dzz], -1)], -2)
        ev, evec = np.linalg.eigh(T)
        ev = np.clip(ev, 0, None)
        mean = ev.mean(axis=1)
        num = ((ev - mean[:, None]) ** 2).sum(axis=1)
        den = (ev ** 2).sum(axis=1)
        fa[z][m] = np.sqrt(1.5 * num / np.maximum(den, 1e-12))
        md[z][m] = mean
        v1[z][m] = evec[:, :, 2]
    if progress:
        progress(nz, nz, "Done")
    return TensorMaps(fa=np.clip(fa, 0, 1), md=md, v1=v1, b0=b0, mask=mask, dwi=dwi, flip=tuple(flip))


# ========================================================= tractography
@dataclass
class TrackParams:
    seed_fa: float = 0.25
    stop_fa: float = 0.15
    max_angle: float = 45.0
    step_mm: float = 0.5
    min_len: float = 20.0
    max_len: float = 250.0
    seeds_per_voxel: int = 1
    stride: int = 2               # whole-brain seeding: one seed every `stride` voxels
    max_seeds: int = 60000


def _sample(fa, v1, pos, d):
    """Trilinear FA and sign-aligned trilinear principal direction at float voxel positions pos (N, 3: x, y, z)."""
    nz, ny, nx = fa.shape
    x0 = np.floor(pos[:, 0]).astype(np.int64)
    y0 = np.floor(pos[:, 1]).astype(np.int64)
    z0 = np.floor(pos[:, 2]).astype(np.int64)
    fx, fy, fz = pos[:, 0] - x0, pos[:, 1] - y0, pos[:, 2] - z0
    f_out = np.zeros(len(pos), np.float32)
    v_out = np.zeros((len(pos), 3), np.float32)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                w = (fx if dx else 1 - fx) * (fy if dy else 1 - fy) * (fz if dz else 1 - fz)
                xi = np.clip(x0 + dx, 0, nx - 1)
                yi = np.clip(y0 + dy, 0, ny - 1)
                zi = np.clip(z0 + dz, 0, nz - 1)
                f_out += w * fa[zi, yi, xi]
                v = v1[zi, yi, xi]
                s = np.sign((v * d).sum(axis=1))
                s[s == 0] = 1
                v_out += (w * s)[:, None] * v
    n = np.linalg.norm(v_out, axis=1, keepdims=True)
    return f_out, v_out / np.maximum(n, 1e-9)


def _track_dir(tm, seeds, d0, p: TrackParams):
    """Track all seeds in one direction at once. Returns positions (steps+1, N, 3) and valid point counts (N,)."""
    sz, sy, sx = tm.dwi.spacing
    sp = np.array([sx, sy, sz], np.float32)
    nz, ny, nx = tm.fa.shape
    hi = np.array([nx - 1, ny - 1, nz - 1], np.float32)
    steps = int(p.max_len / 2 / p.step_mm) + 1
    cos_max = math.cos(math.radians(p.max_angle))
    pos = seeds.astype(np.float32).copy()
    d = d0.astype(np.float32).copy()
    out = np.empty((steps + 1, len(pos), 3), np.float32)
    out[0] = pos
    alive = np.ones(len(pos), bool)
    count = np.ones(len(pos), np.int32)
    for s in range(1, steps + 1):
        if alive.any():
            f, v = _sample(tm.fa, tm.v1, pos, d)
            ok = alive & (f >= p.stop_fa) & ((v * d).sum(axis=1) >= cos_max)
            nxt = pos + v * (p.step_mm / sp)
            ok &= (nxt >= 0).all(axis=1) & (nxt <= hi).all(axis=1)
            mi = np.clip(np.round(nxt).astype(np.int64), 0, hi.astype(np.int64))
            ok &= tm.mask[mi[:, 2], mi[:, 1], mi[:, 0]]
            pos = np.where(ok[:, None], nxt, pos)
            d = np.where(ok[:, None], v, d)
            alive = ok
            count += ok
        out[s] = pos
    return out, count


def make_seeds(tm, p: TrackParams, roi=None, rng=None):
    """Seed positions (N, 3: x, y, z float voxel). roi = (center_mm (x,y,z), radius_mm) or a boolean mask."""
    rng = rng or np.random.default_rng(0)
    fa = tm.fa
    if roi is None:
        st = max(1, p.stride)
        m = np.zeros_like(fa, bool)
        m[::st, ::st, ::st] = True
        m &= (fa >= p.seed_fa) & tm.mask
    elif isinstance(roi, np.ndarray):
        m = roi.astype(bool) & (fa >= p.seed_fa)
    else:
        (cx, cy, cz), r = roi
        sz, sy, sx = tm.dwi.spacing
        z, y, x = np.ogrid[:fa.shape[0], :fa.shape[1], :fa.shape[2]]
        m = ((x * sx - cx) ** 2 + (y * sy - cy) ** 2 + (z * sz - cz) ** 2 <= r * r) & (fa >= p.seed_fa)
    zz, yy, xx = np.nonzero(m)
    base = np.stack([xx, yy, zz], axis=1).astype(np.float32)
    seeds = np.repeat(base, p.seeds_per_voxel, axis=0)
    seeds += rng.uniform(-0.5, 0.5, seeds.shape).astype(np.float32)
    if len(seeds) > p.max_seeds:
        seeds = seeds[rng.choice(len(seeds), p.max_seeds, replace=False)]
    return seeds


def track(tm, seeds, p: TrackParams, include=(), exclude=(), progress=None, batch=15000):
    """Deterministic tensor tractography. include / exclude: [(center_mm (x,y,z), radius_mm), ...].
    Returns a list of (n_i, 3) float32 arrays in voxel-mm space (x*sx, y*sy, z*sz)."""
    sz, sy, sx = tm.dwi.spacing
    sp = np.array([sx, sy, sz], np.float32)
    lines = []
    for b0 in range(0, len(seeds), batch):
        if progress:
            progress(b0, len(seeds), "Tracking...")
        sd = seeds[b0:b0 + batch]
        _, v = _sample(tm.fa, tm.v1, sd, np.tile(np.array([[1, 0, 0]], np.float32), (len(sd), 1)))
        fwd, nf = _track_dir(tm, sd, v, p)
        bwd, nb = _track_dir(tm, sd, -v, p)
        for i in range(len(sd)):
            pts = np.concatenate([bwd[:nb[i], i][::-1], fwd[1:nf[i], i]]) * sp
            if len(pts) < 2:
                continue
            seg = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
            if seg < p.min_len or seg > p.max_len:
                continue
            if any(np.min(np.linalg.norm(pts - np.array(c), axis=1)) > r for c, r in include):
                continue
            if any(np.min(np.linalg.norm(pts - np.array(c), axis=1)) <= r for c, r in exclude):
                continue
            lines.append(pts.astype(np.float32))
    if progress:
        progress(len(seeds), len(seeds), "Done")
    return lines
