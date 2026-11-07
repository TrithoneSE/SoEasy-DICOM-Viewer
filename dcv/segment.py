"""Per-series state (segments, measurements, scalpel cuts), segmentation and bone-edge clean-up."""

from __future__ import annotations

import numpy as np
from . import vtk_lite as vtk
from vtkmodules.util import numpy_support

from .phases import (DIFF_MIN_HU, VESSEL_MIN_HU, _pool, classify_phases, coarse_steps, inside_skull, phase_of_name, sinus_tracing,
                     upsample)


BONE_OFFSET = 10000
VEIN_OFFSET = 20000     # veins are stored as VEIN_OFFSET + HU, far above the normal domain (see update_render_data)
BONE_CORE_HU = 1000.0   # cortical bone; contrast in vessels stays well below it (real CTA: arteries ~590, max ~800)
VEIN_MIN_HU = 110.0
THIN_SKULL_SHARE = 0.4  # VolumeState.thin_skull: dense cells / bright cells below this = a newborn's skull
THIN_BONE_HU = 150.0    # a newborn's skull plate is bone from this HU (real CT: 150-450)
ARTERY_TOP_HU = 450.0
# VolumeState.bone_source
PLAIN_BONE_HU = 150.0   # bone on the plain scan (blood without contrast ~40 HU; a child's facial bones 190-760 with it)
PLAIN_TOL_MM = 0.9      # the plain scan's bone counts within this: one voxel of shift between the scans moves a bone edge
                        # by hundreds of HU (real infant CT: 26 % of its plain bone looked enhanced voxel by voxel, 0.6 % so)
ENHANCE_HU = 50.0       # contrast: brighter than anything within PLAIN_TOL_MM on the plain scan by this
LUMEN_PLAIN_MAX_HU = 130.0  # and below this there: a lumen is blood before contrast (~40-60 HU, ~100 at its edge). Brighter
                            # is bone: the noise of dense bone passes ENHANCE_HU (rings on the skull), and a plain scan of
                            # thicker slices blurs thin bone (real CTA, 1.25 vs 0.625 mm: the maxilla's walls 150-280 HU)
ENHANCE_NEIGHBOURS = 9  # of the 27 voxels around one, this many enhanced too: alone it is noise (grain on the face)
LUMEN_EDGE_HU = 150.0   # a lumen against bone: brighter than the plain scan at the voxel itself by this,
LUMEN_EDGE_MM = 1.0     # within this of a lumen found above (real CTA: an aneurysm and its ICA on the clinoid took bone)
NOT_VESSEL_HU = 100.0   # what did not get brighter is stored no higher than this: under every vessel curve
CTA_TOP_HU = 140.0      # phases.intracranial_top from this: a CTA (real: 161-208); a routine '+ CM' head CT (74-112)
BONE_CURVE_FROM_HU = 200.0  # bone curves start here on a bone area from the plain scan (fit_bone_curves)


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


def box_count(mask):
    """How many of the 3 x 3 x 3 voxels around each voxel (itself included) are set: uint8."""
    out = mask.astype(np.uint8)
    for axis in range(3):
        src = out.copy()
        lo, hi = [slice(None)] * 3, [slice(None)] * 3
        lo[axis], hi[axis] = slice(1, None), slice(None, -1)
        out[tuple(lo)] += src[tuple(hi)]
        out[tuple(hi)] += src[tuple(lo)]
    return out


def box_max(a, radii):
    """Maximum over a box of (2 rz + 1) x (2 ry + 1) x (2 rx + 1) voxels: per axis, one voxel per pass."""
    out = a.copy()
    for axis, r in enumerate(radii):
        lo, hi = [slice(None)] * 3, [slice(None)] * 3
        lo[axis], hi[axis] = slice(1, None), slice(None, -1)
        lo, hi = tuple(lo), tuple(hi)
        for _ in range(int(r)):
            src = out.copy()
            np.maximum(out[lo], src[hi], out=out[lo])
            np.maximum(out[hi], src[lo], out=out[hi])
    return out


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
    breaks a lesion into speckles and holes; ~1 mm of smoothing keeps it one solid region. The passes of an axis are one
    binomial kernel, run by VTK's separable convolution (threaded: a ±70 mm block of a 0.43 mm CT in ~1.3 s instead of
    ~2.7 s, the same values)."""
    out = np.ascontiguousarray(blk, dtype=np.float32)
    passes = [int(round(2 * (sigma_mm / s) ** 2)) for s in spacing]
    if not any(passes):
        return out.copy()
    conv = vtk.vtkImageSeparableConvolution()
    conv.SetInputData(np_to_vtk_image(out, spacing))
    for set_kernel, p in zip((conv.SetZKernel, conv.SetYKernel, conv.SetXKernel), passes):
        w = np.array([1.0])
        for _ in range(2 * p):
            w = np.convolve(w, [0.5, 0.5])
        k = vtk.vtkFloatArray()
        for v in w:
            k.InsertNextValue(float(v))
        set_kernel(k)
    conv.Update()
    return vtk_image_to_np(conv.GetOutput(), out.shape).copy()


def _smoothed_block(data, spacing, seed, bounds, smooth_mm):
    """(smoothed search block, its offset, the seed inside it)."""
    z0, z1, y0, y1, x0, x1 = _block(data.shape, spacing, seed, bounds)
    blk = smooth_block(data[z0:z1, y0:y1, x0:x1], spacing, smooth_mm)
    return blk, (z0, y0, x0), (seed[0] - z0, seed[1] - y0, seed[2] - x0)


def _shift(res, off):
    return None if res is None else (res[0], (res[1][0] + off[0], res[1][1] + off[1], res[1][2] + off[2]))


ANTI_LEAK_MM = 1.6     # contacts narrower than ~3 mm are cut
CUT_SMOOTH_MM = 0.7    # the cut works on lightly smoothed values
_smooth_cache = {}     # (id(data), block, sigma) -> smoothed block: one per seed, so dragging a handle is quick


def _cut_block(data, spacing, block, sigma):
    key = (id(data), tuple(block), sigma)
    blk = _smooth_cache.get(key)
    if blk is None:
        z0, z1, y0, y1, x0, x1 = block
        blk = data[z0:z1, y0:y1, x0:x1]
        blk = smooth_block(blk, spacing, sigma) if sigma > 0 else blk
        _smooth_cache.clear()
        _smooth_cache[key] = blk
    return blk


def _nearest(mask, s, reach):
    """The voxel of mask nearest to s within reach (voxels per axis), or None."""
    if mask[s]:
        return s
    sl = tuple(slice(max(0, c - k), c + k + 1) for c, k in zip(s, reach))
    idx = np.argwhere(mask[sl])
    if not len(idx):
        return None
    pos = idx + np.array([q.start for q in sl])
    return tuple(int(v) for v in pos[((pos - np.array(s)) ** 2).sum(1).argmin()])


def region_grow(data, spacing, seed, lo, hi, anti_leak=ANTI_LEAK_MM, bounds=None, radius_mm=70.0, smooth_mm=0.0):
    """Voxels connected to the seed with values in [lo, hi], searched in a block of ±radius_mm around the seed.

    anti_leak (mm): contacts narrower than twice this are cut, so the segment does not leak into a neighbour of the same
    HU (real CT: a sinonasal tumour of ~45 HU touching muscle of ~55 grew to 209 ml from a ~20 ml mass). It works on
    values smoothed by CUT_SMOOTH_MM: on 0.4 mm voxels the noise (SD ~7 HU) otherwise breaks the core apart, and the click
    itself fell out of it. A click that is not in the core is moved to the nearest core voxel, a narrower cut is tried,
    and only a structure thinner than any cut grows without one. smooth_mm: smooth more (the checkbox).
    Returns (sub_mask uint8, offset (z0, y0, x0)) or None.
    """
    block = _block(data.shape, spacing, seed, bounds, radius_mm)
    z0, z1, y0, y1, x0, x1 = block
    if not (z0 <= seed[0] < z1 and y0 <= seed[1] < y1 and x0 <= seed[2] < x1):
        return None
    sigma = max(smooth_mm, CUT_SMOOTH_MM if anti_leak > 0 else 0.0)
    blk = _cut_block(data, spacing, block, sigma)
    s = (seed[0] - z0, seed[1] - y0, seed[2] - x0)
    own = data[z0:z1, y0:y1, x0:x1]
    # A small window around the click first, larger only when the segment reaches its side: the cost follows the
    # segment, not the ±radius_mm block (real 0.43 mm CT: ~3 s per handle move for every seed before, as the web grows
    # only what it reaches). What a window holds away from its sides is exactly what the whole block would give.
    margin = max(max(1, int(round(anti_leak / k))) for k in spacing) + 2
    sizes = [w for w in WINDOWS_MM if w < radius_mm] + [radius_mm]
    key = (id(data), tuple(seed), tuple(block))
    # dragging a handle: start from the window the last grow of this click needed (the side check keeps it exact)
    for i in range(min(_last_window.get(key, 0), len(sizes) - 1), len(sizes)):
        w_mm = sizes[i]
        win = [(max(0, c - int(np.ceil(w_mm / k))), min(n, c + int(np.ceil(w_mm / k)) + 1))
               for c, k, n in zip(s, spacing, blk.shape)]
        cut = tuple(slice(a, b) for a, b in win)
        res = _grow_window(blk[cut], own[cut], spacing, tuple(c - a for c, (a, _) in zip(s, win)), lo, hi, anti_leak)
        if res is None:
            return None
        inner = [(a > 0, b < n) for (a, b), n in zip(win, blk.shape)]
        if i == len(sizes) - 1 or not _reaches_side(res, inner, margin):
            out = _trim(res, (z0 + win[0][0], y0 + win[1][0], x0 + win[2][0]))
            if out is not None:      # the smallest window that would have held it starts the next grow
                reach = max(max(abs(o - c), abs(o + n - 1 - c)) * k for o, n, c, k in
                            zip(out[1], out[0].shape, seed, spacing)) + margin * max(spacing)
                _last_window.clear()
                _last_window[key] = next((j for j, w in enumerate(sizes) if w >= reach), len(sizes) - 1)
            return out
    return None


WINDOWS_MM = (10.0, 20.0, 35.0)     # region_grow: the windows tried before the whole block
_last_window = {}                   # (id(data), seed, block) -> index into the windows the last grow needed


def _reaches_side(mask, inner, margin):
    """mask has voxels within margin of a side of its window that is not the side of the whole block."""
    for ax, (lo_in, hi_in) in enumerate(inner):
        n = mask.shape[ax]
        if lo_in and mask.take(range(min(margin, n)), ax).any():
            return True
        if hi_in and mask.take(range(max(0, n - margin), n), ax).any():
            return True
    return False


def _grow_window(blk, own, spacing, s, lo, hi, anti_leak):
    """region_grow inside one window: blk the cut's (smoothed) values, own the HU, s the click. Boolean mask or None."""
    thr = (blk >= lo) & (blk <= hi)                     # smoothed: the core, which noise must not break apart
    edge = thr | ((own >= lo) & (own <= hi))            # the way back out to the edge, which smoothing blurs away
    if not edge[s]:
        return None
    mask = None
    cut = anti_leak
    while cut > 0 and mask is None:
        r = tuple(max(1, int(round(cut / k))) for k in spacing)
        core = box_erode(thr, r)
        start = _nearest(core, s, tuple(2 * k + 1 for k in r))
        if start is not None:
            mask = _connected(core.astype(np.uint8), spacing, start).astype(bool)
            # back out to the edge through the range, one voxel at a time: follows the shape, no box corners. √3 · r
            # steps: a box of ±r reaches only so far into a round surface, so r steps would leave a shell out
            back = tuple(int(np.ceil(np.sqrt(3) * k)) for k in r)
            # only around the core: it moves by at most `back` voxels
            box = tuple(slice(max(0, i[0] - b), i[-1] + b + 1) for i, b in
                        zip((np.flatnonzero(mask.any(axis=tuple(a for a in range(3) if a != ax))) for ax in range(3)), back))
            m, e = mask[box], edge[box]
            for i in range(max(back)):
                m = box_dilate(m, tuple(1 if i < k else 0 for k in back)) & e
            mask[box] = m
        elif max(r) > 1:
            cut /= 2
        else:
            cut = 0
    if mask is None:
        mask = _connected(edge.astype(np.uint8), spacing, s).astype(bool)
    return mask


GUARD_MM = 30.0     # tissue_guard: the neighbourhood whose tissues bound the automatic range


def tissue_guard(data, spacing, seed, bounds=None):
    """(lowest lo, highest hi) the automatic range may take on CT, so it stays out of the large tissue next to the
    seed's own. A heterogeneous enhancing tumour has a wide SD, so mean − 2.5·SD reached ~38 HU and the segment ran into
    the brain (~35 HU): real CT, a 0.625 mm frontal tumour grew to 100–700 ml in 5–8 s. On the smoothed values within
    GUARD_MM, the peak of the histogram nearest the seed's level is its tissue; the range stops at the valley between
    it and the largest peak below (above): ~56 HU there, the tumour alone (~76 ml) in ~1 s."""
    block = _block(data.shape, spacing, seed, bounds)
    sm = _cut_block(data, spacing, block, CUT_SMOOTH_MM)     # the smoothed block region_grow uses next (cached)
    s = (seed[0] - block[0], seed[1] - block[2], seed[2] - block[4])
    sl = tuple(slice(max(0, c - int(np.ceil(GUARD_MM / k))), c + int(np.ceil(GUARD_MM / k)) + 1) for c, k in zip(s, spacing))
    r = tuple(max(1, int(round(2.0 / k))) for k in spacing)
    m = float(np.median(sm[tuple(slice(max(0, c - b), c + b + 1) for c, b in zip(s, r))]))
    at_seed = float(sm[s])
    base = -1024
    h = np.bincount(np.clip(np.round(sm[sl]).astype(np.int32).ravel() - base, 0, 4095), minlength=4096).astype(np.float64)
    h[0] = h[-1] = 0                                          # out of the HU scale
    g = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2)
    h = np.convolve(h, g / g.sum(), "same")
    peaks = [i for i in range(3, len(h) - 3) if h[i] > 0.002 * h.sum() and h[i] == h[i - 3:i + 4].max()]
    lo, hi = -np.inf, np.inf
    if not peaks:
        return lo, hi
    own = min(peaks, key=lambda p: abs(p - (round(m) - base)))
    below, above = [p for p in peaks if p < own], [p for p in peaks if p > own]
    if below:
        p = max(below, key=lambda q: h[q])
        v = base + p + int(np.argmin(h[p:own + 1]))
        if v < min(m, at_seed):
            lo = float(v)
    if above:
        p = max(above, key=lambda q: h[q])
        v = base + own + int(np.argmin(h[own:p + 1]))
        if v > max(m, at_seed):
            hi = float(v)
    return lo, hi


def auto_grow(data, spacing, seed, k=2.5, anti_leak=ANTI_LEAK_MM, bounds=None, smooth_mm=0.0, guard=False):
    """Automatic mode: range from the statistics around the seed (mean ± k·SD), refined once. With smooth_mm the
    statistics use the smoothed block too. guard (CT): the range stays out of the neighbouring tissue (tissue_guard)."""
    g_lo, g_hi = tissue_guard(data, spacing, seed, bounds) if guard else (-np.inf, np.inf)
    if smooth_mm > 0:
        blk, off, sd = _smoothed_block(data, spacing, seed, bounds, smooth_mm)
        _, rng = auto_grow(blk, spacing, sd, k, anti_leak)
        rng = (max(rng[0], g_lo), min(rng[1], g_hi))
        res = region_grow(data, spacing, seed, rng[0], rng[1], anti_leak, bounds, 70.0, smooth_mm)
        return res, rng
    m, s = local_stats(data, seed)
    s = max(s, 4.0)
    lo, hi = max(m - k * s, g_lo), min(m + k * s, g_hi)
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
        lo2, hi2 = max(lo2, g_lo), min(hi2, g_hi)
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
        self._polydata = None
        self.version = 0
        self.rebuild()

    @classmethod
    def empty(cls, name, shape, color, spacing):
        return cls(name, None, (0, 0, 0), shape, color, spacing)

    @property
    def volume_ml(self):
        sz, sy, sx = self.spacing
        return float(self.sub.sum()) * sz * sy * sx / 1000.0

    @property
    def polydata(self):
        """The 3D surface, made when 3D first asks for it: a seed tuned in 2D never pays for one per handle drag."""
        if self._polydata is None:
            self._polydata = (build_surface(self.sub, self.off, self.spacing, self.smooth) if self.sub.size
                              else vtk.vtkPolyData())
        return self._polydata

    def rebuild(self):
        if self.sub.size and self.sub.any():
            self.sub, self.off = _trim(self.sub, self.off)
        else:
            self.sub = np.zeros((0, 0, 0), np.uint8)
        self._polydata = None
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
        self._bone_coarse = None        # the bone area on the mask grid (_coarse_region)
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
        self.phase_name = phase_of_name(vol.description) if is_ct else None     # from the series description
        self.phase_mode = "auto"        # the user's word: "auto" / "arterial" / "venous"
        self.partner = None             # Volume of the other phase of this CTA (main_window.find_phase_partner)
        self.native = None              # the pre-contrast series on the mask grid (phases.pooled_native)
        self.combine_phases = True      # arteries from the arterial series, veins from the venous one
        self._phase_masks = None
        self._phase_key = None
        self._coarse = None
        self.sinus_ml = None            # what the sinus tracing found, for the panel
        self.atlas_used = False         # the head's vessels labelled by the vessel atlas (one series)
        self._tracing = None            # sinus_tracing of this series (tracing())
        self.vessel_floor = None        # one contrast series: from this HU, what is no vessel is hidden (VEIN_MIN_HU)
        self.plain_loader = None        # () -> Volume of the study's plain scan or None (main_window, venography)
        self._plain = None              # its HU on this grid (plain_data)
        self._plain_tried = False
        self._thin = None               # thin_skull
        self._plates = None             # thin_plates
        self._source = None             # bone_source
        self._plain_any = None          # the plain scan on this grid even when it does not line up as a whole
        self._bone_plain = None         # bone_plain
        self._enhanced = None           # bone_source 'plain': the voxels brighter than on the plain scan (_plain_area)
        self._rule = None               # _rule_source
        self.plain_match = None         # venography.lines_up of the plain scan found (None: none on this patient's grid)
        self.plain_moved = None         # (R, t) when the patient moved and register() lined the plain scan up
        self.veno_built = False         # the 3D copy holds the venography domains (update_render_data)
        self.render_data = None         # 3D copy (scalpel + edge cleaning + bone lock), possibly half resolution
        self.render_factor = None       # None = automatic, 1 = full, 2 = half
        self._auto_factor = None
        self._render_f = 1
        self.version = 0
        lo, hi = vol.value_range
        self.tf = builtin_presets()[1].copy() if is_ct else generic_preset(lo, hi)
        self.gain = 1.0

    # the grid of the masks ------------------------------------------------------
    @property
    def mask_steps(self):
        """Masks are made on a grid of about 1 mm: on a 600-slice 0.625 mm CT that took opening 3D from ~40 s to ~5 s,
        and the surface still comes from the full-resolution HU."""
        return coarse_steps(self.vol.spacing, self.vol.shape)

    def _coarse_data(self):
        """(mean, max, spacing) of the mask grid."""
        if self._coarse is None:
            steps = self.mask_steps
            mean, top = _pool(self.vol.data, steps)
            self._coarse = mean, top, tuple(s * k for s, k in zip(self.vol.spacing, steps))
        return self._coarse

    def _full(self, mask):
        return upsample(mask, self.mask_steps, self.vol.shape)

    # phases of a CTA --------------------------------------------------------------
    @property
    def phase(self):
        """'arterial', 'venous', or None when nothing says so (treated as arterial, as a CTA is unless named)."""
        if self.vol.modality != "CT":
            return None
        return self.phase_mode if self.phase_mode != "auto" else self.phase_name

    @property
    def paired(self):
        return self.partner is not None and self.combine_phases and self.vol.modality == "CT"

    def phase_masks(self):
        """Arteries, veins and bone from both phases of the CTA (see phases.py)."""
        key = (id(self.partner), self.phase, self.bone_lock_hu)
        if self._phase_key != key:
            own, other = self.vol.data, self.partner.data
            arterial, venous = (other, own) if self.phase == "venous" else (own, other)
            m = classify_phases(arterial, venous, self.vol.spacing, self.bone_lock_hu, self.native)
            if self.phase == "venous" and any(m.shift):
                # the masks lie where the arterial series has things; this window shows the venous series
                back = tuple(-v for v in m.shift)
                m.bone, m.arteries, m.veins = (np.roll(x, back, (0, 1, 2)) for x in (m.bone, m.arteries, m.veins))
            self._phase_masks = m
            self._phase_key = key
        return self._phase_masks

    # bone color lock ----------------------------------------------------------
    def bone_region(self):
        if not self.bone_lock or self.vol.modality != "CT":
            return None
        if self.paired:
            self.phase_masks()
        source = self.bone_source
        key = (self.bone_lock_hu, self.bone_close_mm, self.bone_rim_mm, self._phase_key if self.paired else None, source)
        if self._bone_key != key:
            if source == "plain":
                area, cells = self._plain_area()
                ok = self._bone_plain[1]
                if not ok.all():
                    # the planes the plain scan does not line up with keep the rule of the series without it
                    rule, rule_cells = self._rule_area(self._rule_source())
                    area[~ok] = rule[~ok]
                    cz = ok[::self.mask_steps[0]][:cells.shape[0]]
                    cells[~cz] = rule_cells[~cz]
                    self._enhanced[~ok] = True      # no plain scan there to say what got brighter
                self._bone_region, self._bone_coarse = area, cells
                self._bone_key = key
                return self._bone_region
            self._bone_region, self._bone_coarse = self._rule_area(source)
            self._bone_key = key
        return self._bone_region

    def _rule_area(self, source):
        """(bone area, its cells on the mask grid) without the plain scan, by source ('cta', 'thin' or 'ct')."""
        if self.paired:
            m = self.phase_masks()
            bone, vessels = m.bone, m.arteries | m.veins
            sp = tuple(s * k for s, k in zip(self.vol.spacing, m.steps))
        else:
            mean, top, sp = self._coarse_data()
            if source == "cta":
                bone, vessels = bone_and_vessels(top, sp, self.bone_lock_hu)
            else:
                bone, vessels = top >= self.bone_lock_hu, np.zeros(top.shape, bool)
        plates = self.thin_plates()
        if plates is not None:
            bone = bone | plates
        rc = mm_to_radii(self.bone_close_mm, sp)
        closed = box_erode(box_dilate(bone, rc), rc) | bone      # fill the diploe / spongy bone
        region = box_dilate(closed, mm_to_radii(self.bone_rim_mm, sp))
        # a bright vessel and its wall stay out of the bone area, even right next to bone
        out = region & box_dilate(vessels, tuple(max(1, r) for r in mm_to_radii(1.0, sp))) & ~bone
        return self._full_area(bone, closed, vessels, out, plates), region & ~out

    @property
    def bone_source(self):
        """What the bone area is made from:
        - 'plain': the study's plain scan lined up (bone_plain; as a whole or slab by slab): bone is what was bone before
          contrast (PLAIN_BONE_HU within PLAIN_TOL_MM) and did not get brighter, voxel by voxel, so holes stay holes; the
          planes it does not line up with keep the rule below (_rule_source). No HU tells a weak contrast vessel (real
          infant '+ CM' CT: 150-220 HU) from young bone (190-760), and also with both phases of a CTA the plain scan comes
          first for the bone: a patient who moved between the phases had bone edges that 'changed' and took the vessel
          colour (real CTA: the face red);
        - 'cta': both phases of a CTA, a series named arterial or venous, or a CTA's contrast in the head (CTA_TOP_HU):
          from bone_lock_hu, less the bright vessels and aneurysms that pass it (bone_and_vessels);
        - 'thin': a newborn's skull (thin_skull) without a plain scan that lines up: all from bone_lock_hu and its skull
          plates (thin_plates, by shape), without the CTA rule: its bone has no cortical core, and the rule took 58 % of
          its bone above 500 HU for vessels (real newborn CTA), the plates the vessel colour;
        - 'ct': any other CT (a routine '+ CM' head CT without its plain scan): all from bone_lock_hu. Its vessels stay far
          below that, and the CTA rule took a child's facial bones and spine (no cortical core) for vessels: on a real
          infant CT, 60 % of what had the vessel colour was bone on the plain scan."""
        if self.vol.modality != "CT":
            return None
        if self._source is None:
            self._source = "plain" if self.bone_plain()[0] is not None else self._rule_source()
        return self._source

    def _rule_source(self):
        """bone_source without the plain scan (also where it does not line up)."""
        if self._rule is None:
            if self.paired:
                self._rule = "cta"
            elif self.thin_skull:
                self._rule = "thin"
            elif self.phase in ("arterial", "venous"):
                self._rule = "cta"
            else:
                from .phases import intracranial_top_of
                t = intracranial_top_of(*self._coarse_data())
                self._rule = "cta" if t is not None and t >= CTA_TOP_HU else "ct"
        return self._rule

    def fit_bone_curves(self, tf):
        """tf with its bone curves from BONE_CURVE_FROM_HU when the bone lock takes its bone from the plain scan, from
        THIN_BONE_HU on a newborn's skull (its plates are 150-450 HU), else tf."""
        from .tf_model import fit_bone_curves
        if not self.bone_lock or self.bone_source not in ("plain", "thin"):
            return tf
        return fit_bone_curves(tf, BONE_CURVE_FROM_HU if self.bone_source == "plain" else THIN_BONE_HU)

    def bone_note(self):
        """Where the bone area comes from, for the 3D panel ('' without the bone lock)."""
        source = self.bone_source if self.bone_lock else None
        if source == "plain":
            moved = " (lined up first: the patient had moved between the scans)" if self.plain_moved is not None else ""
            return ("Bone from the plain scan of this study" + moved + ": what was bone before contrast is bone, voxel by "
                    "voxel, and only what got brighter with contrast takes a vessel colour. Bone curves start at "
                    f"{BONE_CURVE_FROM_HU:.0f} HU, so thin bone shows.")
        if source == "cta":
            return "CTA: bone from the threshold below, without the bright vessels and aneurysms that pass it."
        if source == "thin":
            return ("A newborn's skull: bone from the threshold below and the skull plates by their shape (thin sheets), "
                    f"without the rule for bright vessels (this bone has no dense core). Bone curves start at {THIN_BONE_HU:.0f}"
                    " HU, so the plates show; the fontanelles and sutures stay open.")
        if source == "ct":
            return ("No plain scan of this study: bone from the threshold below. Young or thin bone under it, and a vessel "
                    "against bone, may take the wrong colour; the study's plain scan, opened with it, tells them apart.")
        return ""

    def _plain_area(self):
        """(bone area, its cells on the mask grid) from the plain scan, and _enhanced (see bone_source)."""
        before = box_max(self.bone_plain()[0], tuple(max(1, r) for r in mm_to_radii(PLAIN_TOL_MM, self.vol.spacing)))
        np.minimum(before, np.iinfo(np.int16).max - int(ENHANCE_HU), out=before)
        enhanced = self.vol.data >= before + int(ENHANCE_HU)
        enhanced &= before < LUMEN_PLAIN_MAX_HU
        enhanced &= box_count(enhanced) >= ENHANCE_NEIGHBOURS
        # a lumen against bone: within PLAIN_TOL_MM the plain scan holds the bone beside it (and a plain scan of thicker
        # slices blurs it into the lumen), so the test above calls it no lumen; where the plain scan at the voxel itself
        # is blood and the series far brighter, joined to a lumen, it is one (else the neck of an aneurysm on the clinoid
        # and its ICA came out as bone). Joined: a bone edge the scans moved apart is no lumen (rings on the skull)
        p = self.bone_plain()[0]
        edge = (self.vol.data >= np.minimum(p, 32000) + int(LUMEN_EDGE_HU)) & (p < LUMEN_PLAIN_MAX_HU)
        edge &= self.vol.data < BONE_CORE_HU
        edge &= box_dilate(enhanced, tuple(max(1, r) for r in mm_to_radii(LUMEN_EDGE_MM, self.vol.spacing)))
        self._enhanced = enhanced | edge
        del enhanced, edge
        area = before >= PLAIN_BONE_HU
        area &= ~self._enhanced
        del before
        steps = self.mask_steps
        cells = np.zeros(tuple(n // s for n, s in zip(area.shape, steps)), bool)
        nz, ny, nx = cells.shape
        for k in range(steps[0]):
            for j in range(steps[1]):
                for i in range(steps[2]):
                    cells |= area[k:nz * steps[0]:steps[0], j:ny * steps[1]:steps[1], i:nx * steps[2]:steps[2]]
        return area, cells

    @property
    def thin_skull(self):
        """A newborn's skull: most of its bright bone below the bone threshold (cells >= 500 HU / >= 250 HU on the mask
        grid: real newborn CT 0.29, an infant 0.76, adult CTAs 0.57-0.59). Its plates (150-450 HU) then took the vessel
        colour: the whole skull came out red."""
        if self.vol.modality != "CT":
            return False
        if self._thin is None:
            _, top, _ = self._coarse_data()
            dense, bright = int((top >= 500).sum()), int((top >= 250).sum())
            self._thin = bright > 0 and dense / bright < THIN_SKULL_SHARE
        return self._thin

    def thin_plates(self):
        """The skull plates of a thin skull on the mask grid (venography.sheets outside the skull's cavity, from
        THIN_BONE_HU), or None for any other head: the bone area takes them as bone."""
        if not self.thin_skull:
            return None
        if self._plates is None:
            from .venography import cavity, sheets
            mean, top, sp = self._coarse_data()
            self._plates = sheets(mean, sp) & ~cavity(top, sp, mean) & (top >= THIN_BONE_HU)
        return self._plates

    def _full_area(self, bone, closed, vessels, out, plates=None):
        """The bone area voxel by voxel, from the cells of the mask grid and this series' own HU. Only upsampled, its edge
        ran along the ~1 mm cells: a sinus or an artery against the skull was cut off in terraces and the cells taken out
        for a vessel took bone with them (real CTA: red blocks and stepped edges on the skull base). So, as before the grid:
        the bone voxels of the bone cells, the cells the closing filled (diploe), and a rim of bone_rim_mm around them;
        beside a vessel's cell, out goes what lies within 1 mm of a voxel of that vessel, and the bone stays in.
        plates (thin_plates): a newborn's skull plates, bone from THIN_BONE_HU."""
        sp = self.vol.spacing
        d = self.vol.data
        hard = d >= self.bone_lock_hu
        base = (self._full(bone) & hard) | self._full(closed & ~bone)
        if plates is not None:
            base |= self._full(plates) & (d >= THIN_BONE_HU)
        area = box_dilate(base, mm_to_radii(self.bone_rim_mm, sp))
        if not out.any():
            return area
        # beside the vessels' cells only (the grid says where they are), in the box around them
        band = box_dilate(out, (1, 1, 1))
        idx = [np.flatnonzero(band.any(axis=tuple(a for a in range(3) if a != ax))) for ax in range(3)]
        steps = self.mask_steps
        box = tuple(slice(int(i[0]) * k, min(n, (int(i[-1]) + 1) * k)) for i, k, n in zip(idx, steps, d.shape))
        cbox = tuple(slice(b.start // k, -(-b.stop // k)) for b, k in zip(box, steps))
        shape = tuple(b.stop - b.start for b in box)
        up = lambda m: upsample(m[cbox], steps, shape)      # noqa: E731  (box starts on a cell corner)
        db = d[box]
        b = up(band)
        in_vessel = up(vessels)
        in_vessel &= b
        # the lumen, not the faint partial-volume voxels around it, which would show as pink voxel steps on the bone
        vessel = in_vessel & (db >= VESSEL_MIN_HU)
        hv = hard[box] & in_vessel
        # in a vessel's cell, only what touches dense bone is bone: contrast in a large artery or an aneurysm passes the
        # bone threshold too (bone_and_vessels)
        solid = hv & box_dilate(db >= max(BONE_CORE_HU, self.bone_lock_hu + 200.0), (1, 1, 1))
        if self.paired:
            # with both phases, voxel by voxel: contrast changes between them, bone does not. A vessel's cell also holds
            # the bone beside it (the thin plate over a sinus, the petrous apex, the clivus), which came out red in blocks
            same = self._unchanged(db, box, vessel | hv)
            solid |= hv & same
            vessel &= ~same
        bone_v = (hard[box] & ~in_vessel) | solid
        vessel &= ~box_dilate(bone_v, (1, 1, 1))
        bone_v &= b
        # out: the vessel, and around it within 1 mm only what no curve draws. The bone's partial-volume edge beside it
        # stays bone: taken out too, it was drawn by the vessel curve as a red band in voxel steps along the bone
        near = box_dilate(vessel, mm_to_radii(1.0, sp))
        near &= db < VEIN_MIN_HU
        near |= vessel
        sub = area[box]
        sub |= bone_v
        sub &= ~(near & b)
        return area

    def _unchanged(self, db, box, where):
        """Of the voxels where (in box), those whose HU differs by less than DIFF_MIN_HU on the other phase, moved by the
        shift found between the two (as in update_render_data)."""
        pm = self._phase_masks
        sh = [int(v * k) for v, k in zip(pm.shift, pm.steps)]
        if self.phase == "venous":
            sh = [-v for v in sh]
        p = self.partner.data
        at = np.nonzero(where)
        there = tuple(np.clip(i + b.start - s, 0, n - 1) for i, b, s, n in zip(at, box, sh, p.shape))
        out = np.zeros(where.shape, bool)
        out[at] = np.abs(db[at].astype(np.int32) - p[there]) < DIFF_MIN_HU
        return out

    # artery / vein separation -------------------------------------------------
    def _coarse_region(self):
        """The bone area on the mask grid."""
        return self._bone_coarse if self.bone_region() is not None else None

    def artery_mask(self):
        if self.paired:
            return self._full(self.phase_masks().arteries)
        return self._full(self._single_masks()[0])

    def vein_mask(self):
        """With both phases: phases.py. One series: the dural sinuses traced by anatomy (phases.sinus_tracing); on an
        arterial CTA also the vessels dimmer than the arterial tree. With both phases: phases.py."""
        if self.paired:
            return self._full(self.phase_masks().veins)
        return self._full(self._single_masks()[1])

    def tracing(self):
        """(sinus mask, vessel threshold HU) of phases.sinus_tracing on the mask grid, once per series."""
        if self._tracing is None:
            mean, top, sp = self._coarse_data()
            self._tracing = sinus_tracing(mean, top, sp)
        return self._tracing

    def fit_contrast_preset(self, tf):
        """For one contrast series that is not a named arterial CTA: tf (an arteries / veins preset) with its vessel
        curves moved onto this scan's vessels (tf_model.fit_vessel_curves), and the floor under which nothing is coloured
        lowered to the scan's vessel threshold; tf unchanged for a CTA-bright series (sinuses at 200 HU and more)."""
        from .tf_model import fit_vessel_curves
        sinus, vessel_hu = self.tracing()
        mean, _, _ = self._coarse_data()
        if not sinus.any():
            return tf
        peak = float(np.median(mean[sinus]))
        if peak >= 200.0:
            return tf
        floor = max(vessel_hu + 10.0, 60.0)
        self.vessel_floor = min(floor, VEIN_MIN_HU)
        return fit_vessel_curves(tf, floor, peak)

    def _single_masks(self):
        """(arteries, veins) on the mask grid for one series. Named arterial CTA: arteries are the bright voxels and what
        connects to them, veins the dimmer vessels and the traced sinuses. Anything else (a routine contrast CT, a venous
        series): arteries and veins are equally bright, so veins are the traced sinuses and 'arteries' every other
        vessel: arteries and smaller veins together, which the panel says."""
        region = self._coarse_region()
        key = (self._bone_key if region is not None else None, self.artery_hu, self.phase)
        if self._artery_key != key:
            mean, top, sp = self._coarse_data()
            free = np.ones(top.shape, bool) if region is None else ~region
            sinus, vessel_hu = self.tracing()
            if self.phase == "arterial":
                core = self.artery_hu or artery_threshold(top, free)
                self.artery_hu_used = core
                arteries = classify_arteries(top, sp, free, core) & ~sinus
                # the mean: the brightest voxel of a cell is mostly noise at 110 HU
                m = (mean >= VEIN_MIN_HU) & ~arteries
                if region is not None:
                    m &= ~box_dilate(region, mm_to_radii(2.0, sp))
                lab = label_regions(m, sp)
                size = np.bincount(lab.ravel())
                big = size * float(np.prod(sp)) >= 30.0      # brain noise and enhancing dura are specks
                big[0] = False
                veins = big[lab] | sinus
            else:
                self.artery_hu_used = None
                veins = sinus
                arteries = (mean >= vessel_hu) & free & ~sinus
            self.sinus_ml = float(sinus.sum() * np.prod(sp)) / 1000
            # Inside the head the vessel atlas decides, with this series' own HU (dcv/atlas.py; leave-one-out on real
            # CTAs from one series: 88-89 % of the vessel voxels right, the tracing alone 57-66 %). Elsewhere (neck,
            # face) and without a skull that fits the atlas, the above.
            from .atlas import classify_head
            bone = top >= 500
            vessel = (mean >= vessel_hu) & free & inside_skull(bone) & ~box_dilate(bone, (1, 1, 1))
            res = classify_head(vessel, lambda idx: mean[tuple(idx.T)], bone, sp)
            self.atlas_used = res is not None
            if res is not None:
                art_h, ven_h, cells, prior = res
                arteries = (arteries & ~cells) | art_h
                veins = (veins & ~cells) | ven_h
                if self.phase != "arterial":
                    # arteries and veins equally bright: each vessel the label of the tree it is joined to
                    from .phases import vessel_trees
                    arteries, veins = vessel_trees(arteries, veins, sinus, prior, sp)
            self._artery_mask, self._vein_mask, self._artery_key = arteries, veins, key
        return self._artery_mask, self._vein_mask

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
        self.veno_built = self.venography
        if br is not None and self.bone_source == "plain" and not self.venography:
            # bone from the plain scan: what did not get brighter with contrast is no vessel either, wherever it lies (the
            # partial-volume edge of bone, noise)
            hide = min(NOT_VESSEL_HU, (self.vessel_floor or VEIN_MIN_HU) - 1)
            d = self.render_data
            d[~self._enhanced[:, ::f, ::f] & ~br & (d > hide)] = hide
        if self.venography:
            from . import venography
            plain = self.plain_data()
            cav, plates = venography.masks_on(self, f)
            # veins only: the arteries (both phases of a CTA; one series: the atlas and phases.vessel_trees) are no
            # vein. Without them 'arteries' are every vessel but the traced sinuses, and hiding them took the cortical
            # veins too: then the venography shows all of them, as before
            art = self.artery_mask()[:, ::f, ::f]
            arteries = box_dilate(art, (1, 1, 1)) if self.paired or self.atlas_used else None
            venography.encode(self.render_data, src, br, cav, None if plain is None else plain[:, ::f, ::f],
                              2 if self.plain_moved is not None else 1, self.render_spacing, plates, arteries)
            br = None
        if br is not None:
            # bone-area voxels are stored mirrored as -BONE_OFFSET - HU, so only bone curves color them.
            # Mirrored (not shifted): interpolating from the area edge (low HU) to the tissue outside passes
            # only through bone-domain values BELOW the edge HU and air values (transparent). A plain shift would
            # pass through the dense-bone part of the bone curve and leave a speckled dark shell at the edge.
            self.render_data[br] = -BONE_OFFSET - np.minimum(self.render_data[br], 32767 - BONE_OFFSET - 1)
            if self.paired and not self.vein_encoded:
                # with both phases and no vein layer: bone outside the area (moved between the phases, as ribs and
                # clavicles do with breathing, or not lined up with the plain scan) is still bone unless both phases
                # call it an artery (real CTA: the clavicles and ribs red); the vein layer does this below
                d = self.render_data
                art = box_dilate(upsample(self.phase_masks().arteries, self.mask_steps, self.vol.shape, f), (1, 1, 1))
                hard = (d >= self.bone_lock_hu) & ~br & ~art
                d[hard] = -BONE_OFFSET - np.minimum(d[hard], 32767 - BONE_OFFSET - 1)
        if self.vein_encoded and not self.venography:
            # vessels that are not arteries go to their own domain, so only vein curves color them and the partial
            # volume wall of an artery (which passes through vein HU) is not painted as a vein
            # Stored above the normal domain, not between it and the bone domain: there every bone surface would
            # cross the vein values and get blue specks. A vein surface crosses the arteries' range instead; clamping
            # that range (ARTERY_TOP_HU) and the large offset make the crossing ~2 % of a voxel.
            d = self.render_data
            if self.paired:
                # each vessel with the HU of the phase that shows it, so the two are on screen together: a mask cell
                # wider so the full-resolution HU, not the coarse grid, draws the surface
                pm = self.phase_masks()
                own, other_src = self.vol.data[:, ::f, ::f], self.partner.data[:, ::f, ::f]
                art_src, ven_src = (other_src, own) if self.phase == "venous" else (own, other_src)
                sh = tuple(int(v * k) for v, k in zip(pm.shift, pm.steps))
                if any(sh):             # the shift holds for venous → arterial; bring the partner over
                    moved = (sh[0], sh[1] // f, sh[2] // f)
                    if self.phase == "venous":
                        art_src = np.roll(art_src, tuple(-v for v in moved), (0, 1, 2))
                    else:
                        ven_src = np.roll(ven_src, moved, (0, 1, 2))
                on_copy = lambda m: upsample(m, pm.steps, self.vol.shape, f)    # noqa: E731
                a = on_copy(box_dilate(pm.arteries, (1, 1, 1)) & ~pm.veins)
                v = on_copy(box_dilate(pm.veins, (1, 1, 1)) & ~pm.arteries)
                v &= ~a
                if br is not None:
                    a &= ~br
                    v &= ~br
                    # bone outside the bone area (a cell given to a vessel, bone that moved between the phases) is still
                    # bone: before, it was clamped into the vessel range and hidden, which left holes in the teeth
                    hard = (d >= self.bone_lock_hu) & ~a
                    d[hard] = -BONE_OFFSET - np.minimum(d[hard], 32767 - BONE_OFFSET - 1)
                    # and where it moved (bright on this series only), its partial-volume edge is no vein either (real
                    # CTA: blue blocks beside molars); the inner table beside a sinus is bright on both and keeps it
                    other_hu = art_src if self.phase == "venous" else ven_src
                    moved_edge = box_dilate(hard & (other_hu < self.bone_lock_hu), (2, 2, 2))
                    v &= ~moved_edge
                # a coarse cell cuts through a wide lumen (a sinus against the skull): the vein takes the rest of the
                # lumen within ~1 mm, so its surface is drawn by the HU and not in steps of the grid
                lumen = (ven_src >= VEIN_MIN_HU) & (ven_src < self.bone_lock_hu) & ~a
                if br is not None:
                    lumen &= ~br & ~moved_edge
                for _ in range(2):
                    v |= box_dilate(v, (1, 1, 1)) & lumen
                # only the vein's own voxels: brain in a vein cell stored as VEIN_OFFSET + 40 HU met its neighbours at
                # 40 HU across the whole range, and every face of the cell drew a thin red and blue sheet (a wire grid)
                v &= ven_src >= VEIN_MIN_HU
                other = (d >= VEIN_MIN_HU) & ~a & ~v
                d[d > ARTERY_TOP_HU] = ARTERY_TOP_HU
                d[other] = VEIN_MIN_HU - 1
                d[a] = np.minimum(art_src[a], ARTERY_TOP_HU)
                d[v] = VEIN_OFFSET + np.minimum(ven_src[v], 3000)
            else:
                arteries = self.artery_mask()[:, ::f, ::f]
                # a routine contrast CT's vessels start lower (fit_contrast_preset), and so does what is hidden
                floor = self.vessel_floor or VEIN_MIN_HU
                veins = self.vein_mask()[:, ::f, ::f] & (src >= floor)     # the vein's own voxels (see above)
                other = (d >= floor) & ~arteries
                d[d > ARTERY_TOP_HU] = ARTERY_TOP_HU
                d[other] = floor - 1              # neither (brain noise, bone edge): transparent in the normal domain
                d[veins] = VEIN_OFFSET + np.minimum(src[veins], 3000)
        if self.removed is not None:
            self.render_data[self.removed[:, ::f, ::f].astype(bool)] = lo
        self.version += 1

    @property
    def vein_encoded(self):
        return bool(self.encode_veins) and self.vol.modality == "CT"

    @property
    def venography(self):
        """The venography preset (venography.py): the inside of the skull in the veins' domain instead of the veins."""
        return self.vein_encoded and getattr(self.tf, "venography", False)

    def plain_data(self):
        """The plain (pre-contrast) scan of this series' study on its grid (venography.plain_on_grid), once; None when
        there is none. plain_loader (set by the main window) finds and reads it."""
        if not self._plain_tried and self.plain_loader is not None and self.thin_skull:
            # a newborn: its plates are told by their shape (venography.sheets), and its plain scan, lined up, still
            # matched too little (real CT: 74 % after ~30 s of reading and lining up)
            self._plain_tried = True
        if not self._plain_tried and self.plain_loader is not None:
            from .venography import MATCH_MIN, lines_up, plain_on_grid, register
            self._plain_tried = True
            loaded = self.plain_loader()
            p = plain_on_grid(loaded, self.vol) if loaded is not None else None
            self.plain_match = None if p is None else lines_up(p, self.vol.data)
            self.plain_moved = None
            if p is not None and self.plain_match < MATCH_MIN:
                # the patient moved between the scans: the skull's motion found and undone (venography.register)
                motion = register(loaded, self.vol)
                if motion is not None:
                    q = plain_on_grid(loaded, self.vol, motion)
                    match = lines_up(q, self.vol.data)
                    if match > self.plain_match:
                        p, self.plain_match, self.plain_moved = q, match, motion
            self._plain = p if p is not None and self.plain_match >= MATCH_MIN else None
            self._plain_any = p
        return self._plain

    def bone_plain(self):
        """(the plain scan lined up for the bone lock, bool per plane where it does) or (None, None): as a whole when it
        lines up (plain_data), else slab by slab (venography.slab_plain), so a CTA whose neck bent between the scans
        takes its head's bone from the plain scan. Not a newborn's (thin_skull, see plain_data)."""
        if self._bone_plain is None:
            self._bone_plain = (None, None)
            p = self.plain_data()
            if p is not None:
                self._bone_plain = (p, np.ones(self.vol.shape[0], bool))
            elif self._plain_any is not None:
                from .venography import slab_plain
                tol = tuple(max(1, r) for r in mm_to_radii(PLAIN_TOL_MM, self.vol.spacing))
                q, ok = slab_plain(self._plain_any, self.vol.data, self.vol.spacing, tol)
                if ok.any():
                    self._bone_plain = (q, ok)
        return self._bone_plain

    def release_3d(self):
        """Free the 3D-only buffers (copy of the volume, bone area, edge mask) when no 3D window uses them."""
        self.render_data = None
        self._bone_region = self._bone_key = self._bone_coarse = self._plates = None
        self._artery_mask = self._artery_key = self._vein_mask = self._vein_key = None
        self._phase_masks = self._phase_key = self._coarse = None
        self.partner = self.native = None
        self._plain, self._plain_tried = None, False
        self._source = self._enhanced = self._rule = self._plain_any = self._bone_plain = None
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
