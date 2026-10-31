"""Per-series state (segments, measurements, scalpel cuts), segmentation and bone-edge clean-up."""

from __future__ import annotations

import numpy as np
from . import vtk_lite as vtk
from vtkmodules.util import numpy_support


BONE_OFFSET = 10000
VEIN_OFFSET = 20000     # veins are stored as VEIN_OFFSET + HU, far above the normal domain (see update_render_data)
BONE_CORE_HU = 1000.0   # cortical bone; contrast in vessels stays well below it (real CTA: arteries ~590, max ~800)
VEIN_MIN_HU = 110.0
ARTERY_TOP_HU = 450.0


def free_ram_mb():
    """Free physical memory (MB) on Windows; a large number elsewhere."""
    try:
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullAvailPhys / 2 ** 20
    except Exception:
        return 1e9


# ------------------------------------------------------------------ util VTK
def np_to_vtk_image(arr: np.ndarray, spacing):
    """arr (z, y, x) -> vtkImageData (x varies fastest). spacing = (sz, sy, sx)."""
    z, y, x = arr.shape
    img = vtk.vtkImageData()
    img.SetDimensions(x, y, z)
    img.SetSpacing(spacing[2], spacing[1], spacing[0])
    img.SetOrigin(0, 0, 0)
    flat = np.ascontiguousarray(arr).ravel()
    va = numpy_support.numpy_to_vtk(flat, deep=False)
    img.GetPointData().SetScalars(va)
    img._np_ref = flat  # keep the numpy buffer alive while VTK uses it
    return img


def vtk_image_to_np(img, shape):
    return numpy_support.vtk_to_numpy(img.GetPointData().GetScalars()).reshape(shape)


def _dilate_axis(m, axis, r):
    """OR of copies shifted by -r..r along one axis (slice operations: fast, little memory)."""
    out = m.copy()
    n = m.shape[axis]
    for k in range(1, min(r, n - 1) + 1):
        dst = [slice(None)] * 3
        src = [slice(None)] * 3
        dst[axis], src[axis] = slice(k, None), slice(None, -k)
        out[tuple(dst)] |= m[tuple(src)]
        dst[axis], src[axis] = slice(None, -k), slice(k, None)
        out[tuple(dst)] |= m[tuple(src)]
    return out


def box_dilate(mask, radii):
    """Binary dilation with a box (separable per axis, O(N)). radii = (rz, ry, rx) voxels."""
    out = mask.astype(bool)
    for axis, r in enumerate(radii):
        if r > 0:
            out = _dilate_axis(out, axis, int(r))
    return out


def box_erode(mask, radii):
    return ~box_dilate(~mask.astype(bool), radii)


def mm_to_radii(mm, spacing):
    return tuple(int(round(mm / s)) for s in spacing)


def _morph(mask_u8, spacing, radius, erode):
    """radius in in-plane (x / y) voxels; z is scaled so the distance in mm is the same."""
    if radius <= 0:
        return mask_u8
    radii = (int(round(radius * spacing[2] / spacing[0])), radius, radius)
    out = box_erode(mask_u8, radii) if erode else box_dilate(mask_u8, radii)
    return out.astype(np.uint8)


def label_regions(mask, spacing):
    """Connected regions of a boolean mask: uint16 labels (0 = background, 1 = largest, ...)."""
    conn = vtk.vtkImageConnectivityFilter()
    conn.SetInputData(np_to_vtk_image(mask.astype(np.uint8), spacing))
    conn.SetScalarRange(1, 1)
    conn.SetExtractionModeToAllRegions()
    conn.SetLabelModeToSizeRank()
    conn.SetLabelScalarTypeToUnsignedShort()
    conn.Update()
    return vtk_image_to_np(conn.GetOutput(), mask.shape).copy()


def _vessels_by_pieces(data, spacing, high, core, core_share=0.15, thin_mm3=20.0):
    """A 1 mm erosion cuts the thin contacts between arteries and the skull base; then a piece of bone has much of its
    volume above cortical density (real CTA: 80-99 %) and a vessel almost none (0 %). What the erosion removed goes back
    to its piece; leftovers are bone when dense, or when they are small slivers touching bone."""
    radii = tuple(max(1, r) for r in mm_to_radii(1.0, spacing))
    eroded = box_erode(high, radii)
    lab = label_regions(eroded, spacing)
    n = int(lab.max()) + 1
    size, dense = np.bincount(lab.ravel(), minlength=n), np.bincount(lab[core], minlength=n)
    is_bone = dense >= np.maximum(20, core_share * size)
    is_bone[0] = False
    bone = box_dilate(is_bone[lab], tuple(r + 1 for r in radii)) & high
    rest = high & ~bone
    lab = label_regions(rest, spacing)
    n = int(lab.max()) + 1
    size, dense = np.bincount(lab.ravel(), minlength=n), np.bincount(lab[core], minlength=n)
    touches = np.bincount(lab[box_dilate(bone, (1, 1, 1))], minlength=n) > 0
    thick = np.bincount(lab[eroded], minlength=n) > 0
    small = size * float(np.prod(spacing)) <= thin_mm3
    is_bone = (dense >= np.maximum(3, core_share * size)) | (touches & small & ~thick)
    is_bone[0] = False
    return rest & ~is_bone[lab]


def _vessels_away_from_bone(data, spacing, high, core, reach_mm=4.0, far_share=0.2):
    """For a vessel in broad contact with bone, which no erosion separates (a carotid entering its canal, an aneurysm on
    the clinoid): take away dense bone and its partial-volume edge; what remains is a vessel when it is thick (thin plates
    of bone are not) and runs away from dense bone (a diploe or spongy island lies within reach_mm of it)."""
    rc = mm_to_radii(1.5, spacing)
    solid = box_erode(box_dilate(core, rc), rc) | core
    rest = high & ~box_dilate(solid, tuple(max(1, r) + 1 for r in mm_to_radii(1.0, spacing)))
    lab = label_regions(rest, spacing)
    n = int(lab.max()) + 1
    size = np.bincount(lab.ravel(), minlength=n)
    thick = np.bincount(lab[box_erode(rest, (1, 1, 1))], minlength=n) >= 3
    far = np.bincount(lab[~box_dilate(solid, mm_to_radii(reach_mm, spacing))], minlength=n)
    is_vessel = thick & (far >= far_share * size)
    is_vessel[0] = False
    return is_vessel[lab]


def bone_and_vessels(data, spacing, hu):
    """(bone, bright vessels) among the voxels >= hu. One HU threshold cannot tell them apart on a CTA: contrast in a
    large artery or an aneurysm can pass 500 HU (real CTA: arteries ~590, max ~800; bone 1250-1650). Density can."""
    high = data >= hu
    core = data >= max(BONE_CORE_HU, hu + 200.0)
    vessels = _vessels_by_pieces(data, spacing, high, core) | _vessels_away_from_bone(data, spacing, high, core)
    return high & ~vessels, vessels


def artery_threshold(data, region_free):
    """Auto arterial core (HU): 60 % of the bright end of the soft-tissue side of the scan, within sane limits."""
    v = data[region_free & (data >= 150)]
    if v.size < 100:
        return 300.0
    return float(np.clip(0.6 * np.percentile(v, 90), 180, 500))


def classify_arteries(data, spacing, region_free, core_hu):
    """Boolean artery mask: voxels >= core_hu, grown through >= 0.55 core_hu voxels connected to them (arterial tree),
    then 3 mm further through >= 110 HU (small distal branches), then one voxel for the partial-volume wall. On a late
    phase veins are as bright as arteries and connected: HU and connectivity cannot separate them there."""
    low = (data >= 0.55 * core_hu) & region_free
    lab = label_regions(low, spacing)
    n = int(lab.max()) + 1
    keep = np.bincount(lab[(data >= core_hu) & region_free], minlength=n) > 0
    keep[0] = False
    art = keep[lab]
    faint = (data >= 110) & region_free
    for _ in range(max(1, int(round(3.0 / min(spacing))))):
        art = box_dilate(art, (1, 1, 1)) & (faint | art)
    return box_dilate(art, (1, 1, 1)) & region_free


def _connected(mask_u8, spacing, seed_zyx):
    z, y, x = seed_zyx
    pts = vtk.vtkPoints()
    pts.InsertNextPoint(x * spacing[2], y * spacing[1], z * spacing[0])
    seeds = vtk.vtkPolyData()
    seeds.SetPoints(pts)
    conn = vtk.vtkImageConnectivityFilter()
    conn.SetInputData(np_to_vtk_image(mask_u8, spacing))
    conn.SetSeedData(seeds)
    conn.SetScalarRange(1, 1)
    conn.SetExtractionModeToSeededRegions()
    conn.SetLabelModeToConstantValue()
    conn.SetLabelConstantValue(1)
    conn.SetLabelScalarTypeToUnsignedChar()
    conn.Update()
    return vtk_image_to_np(conn.GetOutput(), mask_u8.shape).copy()


# -------------------------------------------------------------- segmentasi
def local_stats(data, seed, radius=2):
    z, y, x = seed
    nz, ny, nx = data.shape
    blk = data[max(0, z - 1):min(nz, z + 2), max(0, y - radius):min(ny, y + radius + 1),
               max(0, x - radius):min(nx, x + radius + 1)].astype(np.float32)
    return float(blk.mean()), float(blk.std())


def _block(shape, spacing, seed, bounds, radius_mm=70.0):
    """Local search block around the seed (z0, z1, y0, y1, x0, x1), intersected with optional crop bounds.
    Working on a block instead of the whole volume keeps memory and time small."""
    z, y, x = seed
    rz, ry, rx = (int(np.ceil(radius_mm / s)) for s in spacing)
    b = [max(0, z - rz), min(shape[0], z + rz + 1), max(0, y - ry), min(shape[1], y + ry + 1),
         max(0, x - rx), min(shape[2], x + rx + 1)]
    if bounds is not None:
        b = [max(b[0], bounds[0]), min(b[1], bounds[1]), max(b[2], bounds[2]), min(b[3], bounds[3]),
             max(b[4], bounds[4]), min(b[5], bounds[5])]
    return b


def _trim(mask, off):
    """Crop a mask to its bounding box -> (sub, offset) or None when empty."""
    if not mask.any():
        return None
    idx = [np.flatnonzero(mask.any(axis=tuple(a for a in range(3) if a != ax))) for ax in range(3)]
    lo = [i[0] for i in idx]
    hi = [i[-1] + 1 for i in idx]
    sub = np.ascontiguousarray(mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).astype(np.uint8)
    return sub, (off[0] + lo[0], off[1] + lo[1], off[2] + lo[2])


def smooth_block(blk, spacing, sigma_mm=1.0):
    """Gaussian-like smoothing (repeated separable [1 2 1] / 4 passes, sigma ≈ sqrt(passes / 2) voxels per axis).
    On contrast CT the noise (SD ~8 HU) is as large as the tumour-to-brain difference, so thresholding raw voxels
    breaks a lesion into speckles and holes; ~1 mm of smoothing keeps it one solid region."""
    out = blk.astype(np.float32)
    for ax, s in enumerate(spacing):
        for _ in range(int(round(2 * (sigma_mm / s) ** 2))):
            p = np.concatenate([out.take([0], ax), out, out.take([-1], ax)], ax)
            cut = [slice(None)] * 3
            parts = []
            for a, b in ((0, -2), (1, -1), (2, None)):
                cut[ax] = slice(a, b)
                parts.append(p[tuple(cut)])
            out = (parts[0] + 2 * parts[1] + parts[2]) * 0.25
    return out


def _smoothed_block(data, spacing, seed, bounds, smooth_mm):
    """(smoothed search block, its offset, the seed inside it)."""
    z0, z1, y0, y1, x0, x1 = _block(data.shape, spacing, seed, bounds)
    blk = smooth_block(data[z0:z1, y0:y1, x0:x1], spacing, smooth_mm)
    return blk, (z0, y0, x0), (seed[0] - z0, seed[1] - y0, seed[2] - x0)


def _shift(res, off):
    return None if res is None else (res[0], (res[1][0] + off[0], res[1][1] + off[1], res[1][2] + off[2]))


def region_grow(data, spacing, seed, lo, hi, anti_leak=1, bounds=None, radius_mm=70.0, smooth_mm=0.0):
    """Voxels connected to the seed with values in [lo, hi], searched in a block of ±radius_mm around the seed.

    anti_leak: erosion radius (voxels) before the connectivity search, dilated back afterwards. This cuts thin
    bridges so the segment does not 'leak' into neighbouring structures. smooth_mm: smooth the block first.
    Returns (sub_mask uint8, offset (z0, y0, x0)) or None.
    """
    if smooth_mm > 0:
        blk, off, s = _smoothed_block(data, spacing, seed, bounds, smooth_mm)
        return _shift(region_grow(blk, spacing, s, lo, hi, anti_leak), off)
    z0, z1, y0, y1, x0, x1 = _block(data.shape, spacing, seed, bounds, radius_mm)
    if not (z0 <= seed[0] < z1 and y0 <= seed[1] < y1 and x0 <= seed[2] < x1):
        return None
    blk = data[z0:z1, y0:y1, x0:x1]
    s = (seed[0] - z0, seed[1] - y0, seed[2] - x0)
    thr = ((blk >= lo) & (blk <= hi)).astype(np.uint8)
    if not thr[s]:
        return None
    work = thr
    if anti_leak > 0:
        er = _morph(thr, spacing, anti_leak, erode=True)
        if er[s]:
            work = er
    mask = _connected(work, spacing, s)
    if work is not thr:
        mask = _morph(mask, spacing, anti_leak, erode=False) & thr
        mask = _connected(mask, spacing, s)
    return _trim(mask, (z0, y0, x0))


def auto_grow(data, spacing, seed, k=2.5, anti_leak=1, bounds=None, smooth_mm=0.0):
    """Automatic mode: range from the statistics around the seed (mean ± k·SD), refined once. With smooth_mm the
    statistics and the growing both use the smoothed block."""
    if smooth_mm > 0:
        blk, off, sd = _smoothed_block(data, spacing, seed, bounds, smooth_mm)
        res, rng = auto_grow(blk, spacing, sd, k, anti_leak)
        return _shift(res, off), rng
    m, s = local_stats(data, seed)
    s = max(s, 4.0)
    lo, hi = m - k * s, m + k * s
    res = region_grow(data, spacing, seed, lo, hi, anti_leak, bounds)
    if res is not None and res[0].sum() > 50:
        sub, (oz, oy, ox) = res
        vals = data[oz:oz + sub.shape[0], oy:oy + sub.shape[1], ox:ox + sub.shape[2]][sub.astype(bool)]
        vals = vals.astype(np.float32)
        m2, s2 = float(vals.mean()), max(float(vals.std()), 4.0)
        lo2, hi2 = m2 - k * s2, m2 + k * s2
        w0 = hi - lo
        if hi2 - lo2 > 1.5 * w0:   # do not let the range widen too much (prevents gradual leaking)
            c = (lo2 + hi2) / 2
            lo2, hi2 = c - 0.75 * w0, c + 0.75 * w0
        res2 = region_grow(data, spacing, seed, lo2, hi2, anti_leak, bounds)
        if res2 is not None:
            return res2, (lo2, hi2)
    return res, (lo, hi)


def build_surface(sub, offset, spacing, smooth=True):
    """Surface of a cropped binary mask placed at voxel offset (z0, y0, x0)."""
    m = 3
    padded = np.pad(sub, m)
    img = np_to_vtk_image(np.ascontiguousarray(padded), spacing)
    img.SetOrigin((offset[2] - m) * spacing[2], (offset[1] - m) * spacing[1], (offset[0] - m) * spacing[0])
    src_port = None
    if smooth:
        cast = vtk.vtkImageCast()
        cast.SetInputData(img)
        cast.SetOutputScalarTypeToFloat()
        g = vtk.vtkImageGaussianSmooth()
        g.SetInputConnection(cast.GetOutputPort())
        g.SetStandardDeviations(1.0, 1.0, 1.0)
        src_port = g.GetOutputPort()
    fe = vtk.vtkFlyingEdges3D()
    if src_port is not None:
        fe.SetInputConnection(src_port)
    else:
        fe.SetInputData(img)
    fe.SetValue(0, 0.5)
    fe.ComputeNormalsOn()
    fe.Update()
    pd = vtk.vtkPolyData()
    pd.DeepCopy(fe.GetOutput())
    return pd


class Segment:
    """A segment stores only its bounding box (sub) and offset, not a full-size mask: a 30 ml tumor in a
    512×512×300 CT costs a few MB instead of 76 MB."""
    _n = 0

    def __init__(self, name, sub, offset, shape, color, spacing):
        Segment._n += 1
        self.id = Segment._n
        self.name = name
        self.sub = sub if sub is not None else np.zeros((0, 0, 0), np.uint8)
        self.off = tuple(int(v) for v in offset)
        self.shape = tuple(shape)
        self.color = color
        self.visible = True
        self.opacity = 1.0
        self.spacing = spacing
        self.smooth = True
        self.range = None
        self.seed = None            # clicked voxel it was grown from; None once the brush has changed it
        self.polydata = None
        self.version = 0
        self.rebuild()

    @classmethod
    def empty(cls, name, shape, color, spacing):
        return cls(name, None, (0, 0, 0), shape, color, spacing)

    @property
    def volume_ml(self):
        sz, sy, sx = self.spacing
        return float(self.sub.sum()) * sz * sy * sx / 1000.0

    def rebuild(self):
        if self.sub.size and self.sub.any():
            trimmed = _trim(self.sub, self.off)
            self.sub, self.off = trimmed
            self.polydata = build_surface(self.sub, self.off, self.spacing, self.smooth)
        else:
            self.sub = np.zeros((0, 0, 0), np.uint8)
            self.polydata = vtk.vtkPolyData()
        self.version += 1

    # ---------------------------------------------- 2D access (SliceView orientation conventions)
    def plane(self, orientation, k, plane_shape):
        """Full-size boolean plane as SliceView shows it (coronal / sagittal are flipped vertically)."""
        out = np.zeros(plane_shape, bool)
        if not self.sub.size:
            return out
        z0, y0, x0 = self.off
        dz, dy, dx = self.sub.shape
        nz = self.shape[0]
        if orientation == "axial":
            if z0 <= k < z0 + dz:
                out[y0:y0 + dy, x0:x0 + dx] = self.sub[k - z0]
        elif orientation == "coronal":
            if y0 <= k < y0 + dy:
                out[nz - z0 - dz:nz - z0, x0:x0 + dx] = self.sub[::-1, k - y0, :]
        else:
            if x0 <= k < x0 + dx:
                out[nz - z0 - dz:nz - z0, y0:y0 + dy] = self.sub[::-1, :, k - x0]
        return out

    def _ensure(self, z0, z1, y0, y1, x0, x1):
        """Grow the stored box so it covers [z0:z1, y0:y1, x0:x1]."""
        if self.sub.size:
            oz, oy, ox = self.off
            dz, dy, dx = self.sub.shape
            z0, y0, x0 = min(z0, oz), min(y0, oy), min(x0, ox)
            z1, y1, x1 = max(z1, oz + dz), max(y1, oy + dy), max(x1, ox + dx)
        new = np.zeros((z1 - z0, y1 - y0, x1 - x0), np.uint8)
        if self.sub.size:
            new[oz - z0:oz - z0 + dz, oy - y0:oy - y0 + dy, ox - x0:ox - x0 + dx] = self.sub
        self.sub, self.off = new, (z0, y0, x0)

    def paint(self, orientation, k, r0, r1, c0, c1, disk, value):
        """Write a 2D disk (plane rows r0:r1, cols c0:c1) into the segment."""
        nz = self.shape[0]
        if orientation == "axial":
            box = (k, k + 1, r0, r1, c0, c1)
        elif orientation == "coronal":
            box = (nz - r1, nz - r0, k, k + 1, c0, c1)
        else:
            box = (nz - r1, nz - r0, c0, c1, k, k + 1)
        if value:
            self._ensure(*box)
        elif not self.sub.size:
            return
        oz, oy, ox = self.off
        dz, dy, dx = self.sub.shape
        # intersect with the stored box when erasing
        z0, z1 = max(box[0], oz), min(box[1], oz + dz)
        y0, y1 = max(box[2], oy), min(box[3], oy + dy)
        x0, x1 = max(box[4], ox), min(box[5], ox + dx)
        if z0 >= z1 or y0 >= y1 or x0 >= x1:
            return
        view = self.sub[z0 - oz:z1 - oz, y0 - oy:y1 - oy, x0 - ox:x1 - ox]
        if orientation == "axial":
            d = disk[y0 - box[2]:y1 - box[2], x0 - box[4]:x1 - box[4]]
            view[0][d] = value
        elif orientation == "coronal":
            # plane row r <-> z = nz - 1 - r
            d = disk[::-1][z0 - box[0]:z1 - box[0], x0 - box[4]:x1 - box[4]]
            view[:, 0, :][d] = value
        else:
            d = disk[::-1][z0 - box[0]:z1 - box[0], y0 - box[2]:y1 - box[2]]
            view[:, :, 0][d] = value


# --------------------------------------------------------- state per series
class VolumeState:
    """Everything attached to one loaded series."""

    def __init__(self, vol):
        from .tf_model import builtin_presets, generic_preset
        self.vol = vol
        self.segments: list[Segment] = []
        self.active_segment: Segment | None = None
        self.measurements = {}          # (orientation, index) -> list[Measurement]
        self.removed = None             # uint8 mask scalpel (3D)
        self._undo = []
        self.edge_clean = 0             # bone-edge clean-up radius (voxels), 0 = off
        self.bone_hu = 250.0
        self._clean_mask = None
        self._clean_key = None
        # bone color lock: the bone area (HU >= bone_lock_hu, closed + widened) is colored by bone curves only
        is_ct = vol.modality == "CT"
        self.bone_lock = is_ct
        self.bone_lock_hu = 500.0
        self.bone_close_mm = 3.0
        self.bone_rim_mm = 1.5
        self._bone_region = None
        self._bone_key = None
        self.encode_bone = False        # set by the 3D window (composite VR only)
        self.encode_veins = False       # set by the 3D window: the curves include a Veins layer
        self.artery_hu = None           # None = automatic (artery_threshold)
        self._artery_mask = None
        self._artery_key = None
        self._vein_mask = None
        self._vein_key = None
        self.artery_hu_used = None
        self._bright_vessels = None
        self.render_data = None         # 3D copy (scalpel + edge cleaning + bone lock), possibly half resolution
        self.render_factor = None       # None = automatic, 1 = full, 2 = half
        self._auto_factor = None
        self._render_f = 1
        self.version = 0
        lo, hi = vol.value_range
        self.tf = builtin_presets()[1].copy() if is_ct else generic_preset(lo, hi)
        self.gain = 1.0

    # bone color lock ----------------------------------------------------------
    def bone_region(self):
        if not self.bone_lock or self.vol.modality != "CT":
            return None
        key = (self.bone_lock_hu, self.bone_close_mm, self.bone_rim_mm)
        if self._bone_key != key:
            sp = self.vol.spacing
            bone, vessels = bone_and_vessels(self.vol.data, sp, self.bone_lock_hu)
            rc = mm_to_radii(self.bone_close_mm, sp)
            closed = box_erode(box_dilate(bone, rc), rc) | bone      # fill the diploe / spongy bone
            region = box_dilate(closed, mm_to_radii(self.bone_rim_mm, sp))
            # a bright vessel and its wall stay out of the bone area, even right next to bone
            self._bone_region = region & ~(box_dilate(vessels, tuple(max(1, r) for r in mm_to_radii(1.0, sp))) & ~bone)
            self._bone_key = key
        return self._bone_region

    # artery / vein separation -------------------------------------------------
    def artery_mask(self):
        region = self.bone_region()
        free = np.ones(self.vol.shape, bool) if region is None else ~region
        key = (self._bone_key if region is not None else None, self.artery_hu)
        if self._artery_key != key:
            core = self.artery_hu or artery_threshold(self.vol.data, free)
            self.artery_hu_used = core
            self._artery_mask = classify_arteries(self.vol.data, self.vol.spacing, free, core)
            self._artery_key = key
        return self._artery_mask

    def vein_mask(self):
        """Other vessels: >= VEIN_MIN_HU outside the arteries, away from bone (its partial-volume edge) and in pieces of
        at least 30 mm3 (brain noise and enhancing dura are specks)."""
        arteries = self.artery_mask()
        key = self._artery_key
        if self._vein_key != key:
            sp = self.vol.spacing
            region = self.bone_region()
            m = (self.vol.data >= VEIN_MIN_HU) & ~arteries
            if region is not None:
                m &= ~box_dilate(region, mm_to_radii(2.0, sp))
            lab = label_regions(m, sp)
            size = np.bincount(lab.ravel())
            big = size * float(np.prod(sp)) >= 30.0
            big[0] = False
            self._vein_mask, self._vein_key = big[lab], key
        return self._vein_mask

    # bone-edge clean-up (partial volume) --------------------------------------
    def clean_mask(self):
        if self.edge_clean <= 0 or self.vol.modality != "CT":
            return None
        key = (self.edge_clean, self.bone_hu)
        if self._clean_key != key:
            bone = (self.vol.data >= self.bone_hu).astype(np.uint8)
            near = _morph(bone, self.vol.spacing, self.edge_clean, erode=False)
            self._clean_mask = (near.astype(bool) & ~bone.astype(bool))
            self._clean_key = key
        return self._clean_mask

    def color_source(self):
        """Data for 2D coloring: bone edges (partial volume) neutralised."""
        cm = self.clean_mask()
        if cm is None:
            return self.vol.data
        if getattr(self, "_csrc_key", None) != self._clean_key:
            d = self.vol.data.copy()
            d[cm] = -1024
            self._csrc, self._csrc_key = d, self._clean_key
        return self._csrc

    # resolution of the 3D copy ------------------------------------------------
    def render_step(self):
        """1 = full resolution, 2 = half resolution in-plane (¼ memory) for the 3D view.
        render_factor None = automatic: half when the volume is large and free RAM is low, or when 3D runs on
        software OpenGL (no usable graphics driver), where every frame is computed by the CPU."""
        if self.render_factor in (1, 2):
            return self.render_factor
        if self._auto_factor is None:
            nvox = float(np.prod(self.vol.shape))
            from . import graphics
            low_ram = nvox > 30e6 and free_ram_mb() < 3500
            self._auto_factor = 2 if low_ram or (nvox > 8e6 and graphics.mode == "software") else 1
        return self._auto_factor

    @property
    def render_spacing(self):
        sz, sy, sx = self.vol.spacing
        f = self._render_f
        return (sz, sy * f, sx * f)

    def update_render_data(self):
        lo = float(self.vol.value_range[0])
        f = self.render_step()
        self._render_f = f
        src = self.vol.data[:, ::f, ::f]
        if self.render_data is None or self.render_data.shape != src.shape:
            self.render_data = None
            # always a real copy: with f == 1 ascontiguousarray would return vol.data itself,
            # and the bone encoding / scalpel below would then overwrite the original (2D) data
            self.render_data = np.array(src, order="C", copy=True)
        else:
            np.copyto(self.render_data, src)
        cm = self.clean_mask()
        if cm is not None:
            self.render_data[cm[:, ::f, ::f]] = lo
        br = self.bone_region() if self.encode_bone else None
        if br is not None:
            br = br[:, ::f, ::f]
            # bone-area voxels are stored mirrored as -BONE_OFFSET - HU, so only bone curves color them.
            # Mirrored (not shifted): interpolating from the area edge (low HU) to the tissue outside passes
            # only through bone-domain values BELOW the edge HU and air values (transparent). A plain shift would
            # pass through the dense-bone part of the bone curve and leave a speckled dark shell at the edge.
            self.render_data[br] = -BONE_OFFSET - np.minimum(self.render_data[br], 32767 - BONE_OFFSET - 1)
        if self.vein_encoded:
            # vessels that are not arteries go to their own domain, so only vein curves color them and the partial
            # volume wall of an artery (which passes through vein HU) is not painted as a vein
            # Stored above the normal domain, not between it and the bone domain: there every bone surface would
            # cross the vein values and get blue specks. A vein surface crosses the arteries' range instead; clamping
            # that range (ARTERY_TOP_HU) and the large offset make the crossing ~2 % of a voxel.
            arteries = self.artery_mask()[:, ::f, ::f]
            veins = self.vein_mask()[:, ::f, ::f]
            d = self.render_data
            other = (d >= VEIN_MIN_HU) & ~arteries
            d[d > ARTERY_TOP_HU] = ARTERY_TOP_HU
            d[other] = VEIN_MIN_HU - 1            # neither (brain noise, bone edge): transparent in the normal domain
            d[veins] = VEIN_OFFSET + np.minimum(src[veins], 3000)
        if self.removed is not None:
            self.render_data[self.removed[:, ::f, ::f].astype(bool)] = lo
        self.version += 1

    @property
    def vein_encoded(self):
        return bool(self.encode_veins) and self.vol.modality == "CT"

    def release_3d(self):
        """Free the 3D-only buffers (copy of the volume, bone area, edge mask) when no 3D window uses them."""
        self.render_data = None
        self._bone_region = self._bone_key = None
        self._artery_mask = self._artery_key = self._vein_mask = self._vein_key = None
        self._clean_mask = self._clean_key = None
        self._csrc = self._csrc_key = None
        self.encode_bone = self.encode_veins = False
        self.version += 1

    def has_user_work(self):
        return bool(self.segments) or any(self.measurements.values()) or self.removed is not None

    @property
    def bone_encoded(self):
        return self.encode_bone and self.bone_region() is not None

    # scalpel ------------------------------------------------------------------
    def apply_cut(self, cut_mask_bool):
        if self.removed is None:
            self.removed = np.zeros(self.vol.shape, np.uint8)
        self._undo.append(np.packbits(self.removed))
        self._undo = self._undo[-10:]
        self.removed[cut_mask_bool] = 1
        if self.render_data is None:
            self.update_render_data()
        else:   # incremental: only blank the newly cut voxels instead of rebuilding the whole 3D copy
            f = self._render_f
            self.render_data[cut_mask_bool[:, ::f, ::f]] = self.vol.value_range[0]
            self.version += 1

    def undo_cut(self):
        if not self._undo:
            return False
        bits = self._undo.pop()
        self.removed = np.unpackbits(bits)[: int(np.prod(self.vol.shape))].reshape(self.vol.shape)
        self.update_render_data()
        return True

    def reset_cut(self):
        if self.removed is not None:
            self._undo.append(np.packbits(self.removed))
        self.removed = None
        self.update_render_data()
