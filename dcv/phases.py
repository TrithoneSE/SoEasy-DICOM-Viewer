"""Arterial and venous phase of one CTA: which series belong together, and which voxels are arteries or veins.

One phase cannot tell the two apart. On the venous series of real multiphase head CTAs (GE, 120 kVp, ~18 s between the
phases) arteries have washed out to ~95 HU in some patients and still hold ~230 HU in others, veins reach 130-280 HU, and
statistics of a single series do not separate the phases (P90 of the bright soft tissue: arterial 403-508, venous 400-470,
neck and shoulder vessels included).

Two series do, with anatomy where timing is not enough:
- arterial seeds: as bright as the large arteries on the arterial series and washing out;
- venous seeds: brighter on the venous series; keeping clearly more of their enhancement than this patient's arteries,
  which finds thin and weakly filled veins; and the thick sinuses against the skull. Those are needed because the
  "arterial" series is often late: in 3 of 7 real CTAs the sinuses were already full on it (~290 HU against ~400) and fell
  by the same share as the arteries, so the time course alone painted them as arteries;
- both grow through the vessels and meet where the trees touch.
A sinus is shown as a vein even when it fills early (real CCF: ~400-550 HU on the arterial series): the colour says what
the vessel is, not that it is arterialized.

Bone needs no shape analysis here: it is the only thing that stays bright on both series.
All masks are made on a coarse grid (about 1 mm), which is what keeps opening 3D quick; the surface in 3D still comes
from the full-resolution HU.
"""

from __future__ import annotations

import re

import numpy as np

VENOUS_NAME = re.compile(r"(?<![A-Z])(VENA|VENOUS|VENOSA|VENOGRAM|VEN|CTV|DELAY|DELAYED|LATE)(?![A-Z])")
ARTERIAL_NAME = re.compile(r"(?<![A-Z])(ARTER|ARTERI|ARTERIAL|ARTERY)")
NATIVE_NAME = re.compile(r"(?<![A-Z])(PRE|NK|NC|NON ?CONTRAST|PLAIN|NATIVE|UNENHANCED|POLOS)(?![A-Z])")
CONTRAST_NAME = re.compile(r"(\+ ?(CM|K|C|KONTRAS|CONTRAST)(?![A-Z])|(?<![A-Z])(C\+|KONTRAS|CONTRAST|POST|ENHANCED|CE)(?![A-Z]))")
CONTRAST_TOP_HU = 70.0  # intracranial_top: real CTs without contrast <= 66, with contrast >= 74

VESSEL_MIN_HU = 150.0   # on the phase where the vessel is at its brightest
DIFF_MIN_HU = 60.0      # bone edges, noise and enhancing mucosa are alike on both series, a vessel with contrast is not
PIECE_MIN_MM3 = 8.0     # smaller pieces are noise
MAX_SHIFT_MM = 6.0
ARTERY_SHARE = 0.85     # of the large arteries' HU on the arterial series: above it (and washing out) is arterial
PEAK_MAX_HU = 1000.0    # contrast in the neck arteries stays below it (real CTA: 331-595)
ARTERY_SEED_DEPTH_MM = 70.0    # under the vertex: no arterial seeds higher than this
SINUS_THICK_MM = 3.0   # what is left of a dural sinus's lumen beside the bone edge; cortical arteries are narrower
SINUS_REACH_MM = 3.0    # from the inner table
SINUS_DEPTH_MM = 130.0  # under the vertex: down to the transverse and upper sigmoid sinuses
SINUS_MIN_MM3 = 150.0   # of core in one piece
SINUS_ABOVE_BRAIN_HU = 35.0     # the tracing's vessels are at least this far above the brain
SINUS_MAX_ML = 45.0
SKULL_BASE_HALF_MM = 30.0
SINUS_THICK_DEPTH_MM = 70.0    # thick vessels away from bone count as sinus only above this (straight sinus, torcula)
WASHOUT_MARGIN = 0.2    # share of enhancement kept, above what this patient's arteries keep
WASHOUT_MIN_MM3 = 60.0  # smaller pieces are noise in the share
UNENHANCED_HU = 40.0
LATE_GAP = 0.12         # sinuses keeping less than this share above the neck arteries: the time course tells nothing
LATE_MIN_ML = 2.0       # of traced sinus, to compare its share with the arteries'
MOVED_BONE_HU = 700.0   # brightest voxel of a cell: cortical bone and teeth; contrast in the arteries stays below it
MOVED_BONE_MM = 3.0
ARTERY_BASE_DEPTH_MM = 90.0
ARTERY_BASE_FRONT = 0.55
AORTA_DEPTH_MM = 270.0  # under the vertex: below the aortic arch's top, where the descending aorta is
JUGULAR_GAP_MM = 14.0   # across the bright neck vessels of one side: a carotid alone spans ~8 mm, with the jugular ~20


def phase_of_name(description):
    """'arterial', 'venous', 'native' or None, from the series description."""
    d = (description or "").upper()
    if NATIVE_NAME.search(d):
        return "native"
    if VENOUS_NAME.search(d):
        return "venous"
    if ARTERIAL_NAME.search(d):
        return "arterial"
    return None


def has_contrast_name(description):
    """'+ CM', '+K', 'C+', 'post' ...: the name says contrast was given (but not which phase)."""
    return bool(CONTRAST_NAME.search((description or "").upper()))


def intracranial_top(data, spacing):
    """HU of the brightest 0.5 % inside the skull, 15-90 mm under the vertex, away from bone: the vessels and sinuses.
    None when the scan holds too little head to tell. Real CTs: without contrast 42-66 HU, with contrast 74-258 (the
    dimmest: a late venous phase, 81, and a routine '+ CM' head CT, 74). The neck is left out: streaks from the arm
    vein and bone that moved between two scans make it look enhanced either way."""
    steps = coarse_steps(spacing, data.shape)
    mean, top = _pool(data, steps)
    return intracranial_top_of(mean, top, tuple(s * k for s, k in zip(spacing, steps)))


def intracranial_top_of(mean, top, sp):
    """intracranial_top on the mask grid already pooled."""
    from .segment import box_dilate, mm_to_radii
    bone = top >= 500
    vtx = _head_top(bone)
    z0, z1 = max(0, vtx - int(90 / sp[0])), max(1, vtx - int(15 / sp[0]))
    b = bone[z0:z1]
    # between bone on both sides, along x and along y, in the same plane
    inside = (np.maximum.accumulate(b, axis=2) & np.maximum.accumulate(b[:, :, ::-1], axis=2)[:, :, ::-1]
              & np.maximum.accumulate(b, axis=1) & np.maximum.accumulate(b[:, ::-1], axis=1)[:, ::-1])
    inside &= ~box_dilate(bone, mm_to_radii(3.0, sp))[z0:z1] & (mean[z0:z1] > 0)
    if np.count_nonzero(inside) < 20000:
        return None
    return float(np.percentile(mean[z0:z1][inside], 99.5))


def is_phase_pair(a, b, spacing):
    """Whether two series of one study are both contrast phases (intracranial_top of both at least CONTRAST_TOP_HU). A
    plain head CT next to its '+ CM' series is not: pairing those painted a whole skull red and blue. Unknown (no head in
    the scan) counts as a pair: the names decided it."""
    ta, tb = intracranial_top(a, spacing), intracranial_top(b, spacing)
    return ta is None or tb is None or min(ta, tb) >= CONTRAST_TOP_HU


def coarse_steps(spacing, shape, target_mm=1.0, min_voxels=30e6):
    """(step z, y, x) of the mask grid: every axis halved when it stays near target_mm; small volumes keep every voxel."""
    if float(np.prod(shape)) < min_voxels:
        return (1, 1, 1)
    return tuple(max(1, int(round(target_mm / s))) if s < target_mm else 1 for s in spacing)


def _pool(data, steps):
    """(mean as float32, max) of each grid cell. Cell by cell over slices of the volume: ~10× quicker than a reshape."""
    sz, sy, sx = steps
    nz, ny, nx = (n // s for n, s in zip(data.shape, steps))
    if steps == (1, 1, 1):
        return np.asarray(data, np.float32), data
    mean = np.zeros((nz, ny, nx), np.float32)
    top = np.full((nz, ny, nx), np.iinfo(np.int16).min, data.dtype)
    for k in range(sz):
        for j in range(sy):
            for i in range(sx):
                cell = data[k:nz * sz:sz, j:ny * sy:sy, i:nx * sx:sx]
                mean += cell
                np.maximum(top, cell, out=top)
    mean /= sz * sy * sx
    return mean, top


def upsample(mask, steps, shape, f=1):
    """A mask of the coarse grid on the full grid, or on its copy with every f-th row and column (cells at the far edge
    repeat the last one)."""
    if tuple(steps) == (1, 1, 1) and f == 1:
        return mask
    reps = [s // k if s % k == 0 else 0 for s, k in zip(steps, (1, f, f))]
    size = [len(range(0, n, k)) for n, k in zip(shape, (1, f, f))]
    if all(reps):
        # every cell written into its s / k strided copies: ~4× quicker than indexing (the bone area makes several of
        # these on every 3D opening)
        out = np.empty(size, mask.dtype)
        fit = [min(m, -(-n // r)) for m, n, r in zip(mask.shape, size, reps)]
        for i in range(reps[0]):
            for j in range(reps[1]):
                for k in range(reps[2]):
                    dst = out[i::reps[0], j::reps[1], k::reps[2]]
                    w = [min(a, b) for a, b in zip(dst.shape, fit)]
                    dst[:w[0], :w[1], :w[2]] = mask[:w[0], :w[1], :w[2]]
        # past the last whole cell: the last cell again
        for ax in range(3):
            done = min(mask.shape[ax] * reps[ax], size[ax])
            if done < size[ax]:
                sl = (slice(None),) * ax
                out[sl + (slice(done, None),)] = out[sl + (slice(done - 1, done),)]
        return out
    idx = [np.minimum(np.arange(0, n, k) // s, m - 1) for n, s, m, k in zip(shape, steps, mask.shape, (1, f, f))]
    return mask[idx[0]][:, idx[1]][:, :, idx[2]]


def alignment(a_bone, b_bone, spacing, max_mm=MAX_SHIFT_MM):
    """Whole-voxel shift of b relative to a (z, y, x) from their bone, and how much of a's bone it then covers.
    A patient who moved between the phases shows up here; the head is rigid enough for a translation over 18 s."""
    a, b = a_bone[::2], b_bone[::2]         # every second plane is enough to find a whole-voxel shift in y and x
    n = np.count_nonzero(a)
    if not n:
        return (0, 0, 0), 0.0
    best, score = (0, 0, 0), np.count_nonzero(a & b)
    # uphill one voxel at a time: the overlap is smooth around its peak, and a head that did not move costs 6 comparisons.
    # Comparing with a threshold instead misses 2-3 mm: a skull is thick, most of it still overlaps with itself.
    reach = [int(np.ceil(max_mm / s)) for s in spacing]
    better = True
    while better:
        better = False
        for ax in (1, 2, 0):
            for d in (-1, 1):
                sh = list(best)
                sh[ax] += d
                if abs(sh[ax]) > reach[ax]:
                    continue
                s = np.count_nonzero(a & np.roll(b_bone, sh, (0, 1, 2))[::2])
                if s > score:
                    best, score, better = tuple(sh), s, True
    return best, score / n


def _box_mean(m, r=1):
    """Mean over a (2r + 1)^3 box, by cumulative sums along each axis."""
    out = m
    for ax in range(3):
        c = np.cumsum(np.concatenate([np.take(out, [0] * (r + 1), ax), out, np.take(out, [-1] * r, ax)], ax), ax,
                      dtype=np.float32)
        hi = [slice(None)] * 3
        lo = [slice(None)] * 3
        hi[ax], lo[ax] = slice(2 * r + 1, None), slice(None, -(2 * r + 1))
        out = (c[tuple(hi)] - c[tuple(lo)]) / (2 * r + 1)
    return out


class PhaseMasks:
    """Result on the coarse grid: bone (bright on both), arteries, veins; shift = venous relative to arterial."""

    def __init__(self, bone, arteries, veins, steps, shift, overlap, late=False):
        self.bone, self.arteries, self.veins = bone, arteries, veins
        self.steps, self.shift, self.overlap = steps, shift, overlap
        self.late = late    # both series after the arterial peak: separated by anatomy (see classify_phases)
        self.atlas = False  # the head's vessels labelled by the vessel atlas (dcv/atlas.py)


def pooled_native(native, spacing):
    """The pre-contrast series on the mask grid: all that is kept of it (a tenth of its memory)."""
    return _pool(native, coarse_steps(spacing, native.shape))[0]


def classify_phases(arterial, venous, spacing, bone_hu=500.0, native=None):
    """PhaseMasks of two series on the same grid. spacing = (sz, sy, sx) mm. native = pooled_native() of the pre-contrast
    series, when the study has one: what is already bright before contrast is no vessel (thin ribs, teeth, calcium),
    even where a streak from the subclavian contrast or dental metal makes it differ between the phases."""
    from .segment import box_dilate, box_erode, label_regions, mm_to_radii

    steps = coarse_steps(spacing, arterial.shape)
    sp = tuple(s * k for s, k in zip(spacing, steps))
    A, top_a = _pool(arterial, steps)
    V, top_v = _pool(venous, steps)
    high_a, high_v = top_a >= bone_hu, top_v >= bone_hu
    shift, overlap = alignment(high_a, high_v, sp)
    if shift != (0, 0, 0):
        V = np.roll(V, shift, (0, 1, 2))
        high_v = np.roll(high_v, shift, (0, 1, 2))
        top_v = np.roll(top_v, shift, (0, 1, 2))
    # the coarse mean of thin diploe falls below bone_hu; the brightest voxel of the cell does not
    bone = high_a & high_v & (np.minimum(A, V) >= 0.6 * bone_hu)
    # bone that moved on its own between the phases (jaw, shoulders, arms: one translation fits the skull only) is bright
    # on one series and soft tissue on the other, which looked like a vessel filling or washing out (real CTA: blue
    # blocks on the teeth and clavicles)
    # Whole cells of bone against soft tissue, not a bone edge that differs a little between the scans (which took
    # the sinuses along the inner table with it)
    dense_a, dense_v = top_a >= MOVED_BONE_HU, top_v >= MOVED_BONE_HU
    reach = mm_to_radii(MOVED_BONE_MM, sp)
    moved_bone = ((dense_a & (A >= bone_hu) & (V < 0.5 * bone_hu) & box_dilate(dense_v, reach))
                  | (dense_v & (V >= bone_hu) & (A < 0.5 * bone_hu) & box_dilate(dense_a, reach)))
    # the sinuses traced along the venous series, as on one contrast series: the whole transverse and sigmoid sinuses,
    # the straight sinus and torcula, where sinus_seeds holds only the thickest cores against the vault. A skull that
    # turned a little between the scans also moves its inner table: the sinus beside it stays (real CTA: the superior
    # sagittal sinus at the vertex was lost to it)
    traced = sinus_tracing(V, top_v, sp)[0]
    moved_bone &= ~box_dilate(traced, mm_to_radii(MOVED_BONE_MM + 1.0, sp))
    del top_a, top_v, high_v, dense_a, dense_v
    diff = _box_mean(V - A)
    bright = np.maximum(A, V) >= VESSEL_MIN_HU
    near_bone = box_dilate(bone | moved_bone,tuple(max(1, r) for r in mm_to_radii(1.0, sp)))
    vessel = bright & ~near_bone
    # what a shift carried round from the far edge of the volume compares two unrelated places
    for ax, s in enumerate(shift):
        if s:
            edge = [slice(None)] * 3
            edge[ax] = slice(0, s) if s > 0 else slice(s, None)
            vessel[tuple(edge)] = False
    voxel = float(np.prod(sp))
    vessel &= label_regions(A > -200, sp) == 1          # the patient: the largest piece; the table and head holder are not
    clear = moved = None                                # clear: cells that were not bright before contrast
    if native is not None and native.shape == A.shape:
        # the patient may have moved in the minute or two since the native scan (real CTA: 4 mm): align on bone first
        moved, cover = alignment(high_a, native >= 0.6 * bone_hu, sp)
        if cover >= 0.5:
            clear = np.roll(native, moved, (0, 1, 2)) < VESSEL_MIN_HU
    # The sinuses first, before the native rule: a head can also turn a little between the scans (real CTA: 63 % of the
    # bone overlapped after the best shift), and then the native inner table may fall on the sinus beside it. A piece
    # that was mostly bright before contrast is no sinus (real CTA: 651 mm3 of skull base, 314 HU before contrast).
    # A weakly filled sinus (real CTA: ~209 HU arterial, ~120 venous) has no core here: washout_seeds finds it.
    sinus = sinus_seeds(vessel, bone, sp, clear=clear)
    lumen = box_dilate(sinus, tuple(max(1, r) for r in mm_to_radii(SINUS_THICK_MM, sp)))    # the core and all around it
    if clear is not None:
        vessel &= clear | lumen                         # no dilation: the sinuses lie right against the inner table
    lab = label_regions(vessel, sp)                     # specks of noise take no part
    size = np.bincount(lab.ravel())
    small = size * voxel < PIECE_MIN_MM3
    small[0] = False
    vessel[small[lab]] = False
    del lab
    peak = arterial_peak(A, vessel, bone, sp)
    # anatomy goes first: on real CTAs a full sinus reached the arteries' HU (379 against 397), and in a fistula passes it
    sinus = lumen & vessel
    native_used = native if clear is not None else None
    moved_used = moved if clear is not None else None
    traced &= vessel
    late = _late_scan(A, V, native_used, moved_used, vessel, bone, traced, sp)
    top = _head_top(bone)
    if late:
        # Both series after the arterial peak (real CTAs: sinuses and neck arteries at the same HU on each series and
        # keeping the same share, 0.36 against 0.40): the time course says nothing, so anatomy decides. Arteries start
        # where they enter the head: the neck, and the thick bright vessels at the centre of the skull base (carotid
        # siphons, basilar artery, circle of Willis), where the tracing never goes. Veins start from every traced sinus.
        # In the neck the jugular veins are as bright as the carotids (real CTA: 324-373 HU against ~380 on the
        # 'arterial' series): there the one that lies lateral is the jugular.
        thick = box_erode(vessel, (1, 1, 1)) & vessel
        neck_art, neck_ven = _neck_sides(vessel & thick & (A >= 0.6 * peak), A > -200, top, sp)
        # all of the skull base centre, not only its brightest vessels: the siphon lies inside the cavernous sinus, which
        # is as bright, and the vein front came in through it to the middle cerebral artery (real CTA)
        base = _artery_entry(inside_skull(bone), top, sp)
        aorta = _descending_aorta(vessel & thick & (A >= 0.6 * peak), bone, A > -200, top, sp)
        art_seed = vessel & (neck_art | base | aorta) & ~box_dilate(traced | sinus, (1, 1, 1))
        ven_seed = vessel & ((diff >= DIFF_MIN_HU) | sinus | traced | neck_ven) & ~art_seed
    else:
        art_seed = vessel & (A >= ARTERY_SHARE * peak) & (diff <= -DIFF_MIN_HU) & ~lumen & ~traced
        # above the circle of Willis what is this bright is a sinus; cortical arteries are reached from below, through M1
        art_seed[max(0, top - int(ARTERY_SEED_DEPTH_MM / sp[0])):] = False
        slow = washout_seeds(A, V, native_used, moved_used, vessel, bone, sp)
        ven_seed = vessel & ((diff >= DIFF_MIN_HU) | sinus | traced | slow) & ~art_seed
    arteries, veins = compete(vessel, art_seed, ven_seed)
    # what the two fronts could not reach (a piece cut off by bone): the time course, or on a late scan whether it lies
    # against the inside of the skull, where the sinuses are
    rest = vessel & ~arteries & ~veins
    if late:
        to_vein = rest & inside_skull(bone) & box_dilate(bone, mm_to_radii(SINUS_REACH_MM + 1.0, sp))
    else:
        to_vein = rest & (diff >= 0)
    arteries |= rest & ~to_vein
    veins |= to_vein
    by_atlas = False
    if late:
        # inside the head the vessel atlas decides (dcv/atlas.py): where arteries and veins run in other heads, with this
        # patient's own HU on the venous series where it still tells them apart. The neck and chest keep the rules above.
        from .atlas import classify_head
        res = classify_head(vessel, lambda idx: V[tuple(idx.T)], bone, sp)
        if res is not None:
            art_h, ven_h, cells, _ = res
            arteries = (arteries & ~cells) | art_h
            veins = (veins & ~cells) | ven_h
            by_atlas = True
    m = PhaseMasks(bone, arteries, veins, steps, shift, overlap, late)
    m.atlas = by_atlas
    return m


def _late_scan(A, V, native, moved, vessel, bone, traced, sp):
    """Whether the traced sinuses keep about the same share of their enhancement as the neck arteries: then the
    'arterial' series was taken after the arterial peak, like the venous one, and the time course separates nothing."""
    voxel = float(np.prod(sp))
    if np.count_nonzero(traced) * voxel < LATE_MIN_ML * 1000:
        return False
    P = np.roll(native, moved, (0, 1, 2)) if native is not None else np.full(A.shape, UNENHANCED_HU, np.float32)
    ratio = (V - P) / np.maximum(A - P, 20.0)
    neck = _neck_arteries(A, vessel, bone, sp)
    if np.count_nonzero(neck) < 50:
        return False
    return float(np.median(ratio[traced])) - float(np.median(ratio[neck])) < LATE_GAP


def _artery_entry(inside, vtx, sp):
    """Where a late scan's arteries start inside the head: under the circle of Willis (ARTERY_BASE_DEPTH_MM to
    SINUS_DEPTH_MM under the vertex), near the midline and in the front ARTERY_BASE_FRONT of the head: carotid siphons, basilar artery. The vein
    of Galen and the straight sinus are higher and further back (real CTA: painted red when the area reached them)."""
    out = np.zeros(inside.shape, bool)
    z0 = max(0, vtx - int(SINUS_DEPTH_MM / sp[0]))         # down to the foramen magnum: the neck is _neck_sides'
    z1 = max(0, vtx - int(ARTERY_BASE_DEPTH_MM / sp[0]))
    for z in range(z0, z1):
        ys, xs = np.nonzero(inside[z])
        if ys.size < 100:
            continue
        xc, y0, y1 = 0.5 * (xs.min() + xs.max()), ys.min(), ys.max()
        half = SKULL_BASE_HALF_MM / sp[2]
        front = y0 + ARTERY_BASE_FRONT * (y1 - y0)
        out[z, :int(front), max(0, int(xc - half)):int(xc + half) + 1] = True
    return out


def _descending_aorta(cand, bone, body, top, sp):
    """The descending aorta where the scan reaches the chest (AORTA_DEPTH_MM under the vertex and lower): the thick bright
    vessel beside the vertebral body, on the patient's left and in front of the spine's centre. Without it the veins of a
    late scan reached the aortic arch first, through the brachiocephalic vein that crosses in front of it (real CTA)."""
    from .segment import box_erode
    out = np.zeros(cand.shape, bool)
    z1 = max(0, top - int(AORTA_DEPTH_MM / sp[0]))
    if z1 == 0:
        return out
    wide = box_erode(cand[:z1], (0, 2, 2)) & cand[:z1]       # ~5 mm across and more
    for z in range(z1):
        cols = np.flatnonzero(body[z].any(axis=0))
        if cols.size < 10 or not wide[z].any():
            continue
        mid = 0.5 * (cols[0] + cols[-1])
        by, bx = np.nonzero(bone[z])
        spine = np.abs(bx - mid) * sp[2] < 10.0
        if np.count_nonzero(spine) < 10:
            continue
        sy = float(np.median(by[spine][by[spine] >= np.median(by[spine])]))     # the back half: body and arch
        ys, xs = np.nonzero(wide[z])
        left = (xs - mid) * sp[2]                       # columns grow to the patient's left (LPS)
        front = (sy - ys) * sp[1]
        keep = (left > 5.0) & (left < 45.0) & (front > 0.0) & (front < 45.0)
        out[z, ys[keep], xs[keep]] = True
    return out


def _neck_sides(cand, body, top, sp):
    """(arteries, veins) among the bright thick vessels of the neck, 14-24 cm under the vertex, by where they lie: on each
    side of the body's midline, the carotid is medial and the internal jugular lateral to it, in every patient. A side
    whose vessels span less than JUGULAR_GAP_MM holds one vessel only, taken as the carotid (a collapsed jugular).
    Of the two largest pieces on a side, the jugular is the one further back and out (it lies posterolateral to the
    carotid); smaller pieces (external carotid branches, the external jugular) are left to the growing trees."""
    from .segment import label_regions
    art, ven = np.zeros(cand.shape, bool), np.zeros(cand.shape, bool)
    z0, z1 = max(0, top - int(240 / sp[0])), max(0, top - int(140 / sp[0]))
    cell = sp[1] * sp[2]
    for z in range(z0, z1):
        cols = np.flatnonzero(body[z].any(axis=0))
        if cols.size < 10 or not cand[z].any():
            continue
        mid = 0.5 * (cols[0] + cols[-1])
        x = np.arange(cand.shape[2])
        for half in (x < mid, x >= mid):
            m = cand[z] & half[None, :]
            if np.count_nonzero(m) * cell < 10.0:
                continue
            lab = label_regions(m[None], (1.0, sp[1], sp[2]))[0]
            ys, xs = np.nonzero(lab)
            # back and out, in mm (rows grow towards the back of the head: LPS)
            out = np.abs(xs - mid) * sp[2] + ys * sp[1]
            size = np.bincount(lab[ys, xs])
            if size.size > 2 and size[2] * cell >= 10.0:
                one, two = lab[ys, xs] == 1, lab[ys, xs] == 2
                jug = one if out[one].mean() > out[two].mean() else two
                car = two if jug is one else one
                art[z, ys[car], xs[car]] = True
                ven[z, ys[jug], xs[jug]] = True
                continue
            one = lab[ys, xs] == 1
            o = out[one]
            if np.ptp(np.abs(xs[one] - mid)) * sp[2] < JUGULAR_GAP_MM:
                art[z, ys[one], xs[one]] = True
                continue
            back = o > 0.5 * (o.min() + o.max())     # carotid and jugular touching: one piece
            art[z, ys[one][~back], xs[one][~back]] = True
            ven[z, ys[one][back], xs[one][back]] = True
    return art, ven


def _neck_arteries(A, vessel, bone, sp):
    """Cells of the large neck arteries: 12-22 cm under the vertex and as bright as arterial_peak."""
    top = _head_top(bone)
    z0, z1 = max(0, top - int(220 / sp[0])), max(1, top - int(120 / sp[0]))
    peak = arterial_peak(A, vessel, bone, sp)
    neck = np.zeros(A.shape, bool)
    neck[z0:z1] = vessel[z0:z1] & (A[z0:z1] >= ARTERY_SHARE * peak)
    return neck


def washout_seeds(A, V, native, moved, vessel, bone, sp):
    """Veins by how much of their enhancement is left on the venous series. Thin or weakly filled vessels are found this
    way, because a share does not change with partial volume: on real CTAs (V - P) / (A - P) was 0.12-0.40 in the neck
    arteries, depending on the patient's timing, and a sinus too faint for sinus_seeds kept 0.47 where the patient's
    arteries kept 0.19-0.24. The arteries of this patient set the bar, so a late or early scan moves it along.
    Without a native series the baseline is unenhanced soft tissue."""
    from .segment import box_dilate, label_regions, mm_to_radii
    P = np.roll(native, moved, (0, 1, 2)) if native is not None else np.full(A.shape, UNENHANCED_HU, np.float32)
    ratio = _box_mean(V - P) / np.maximum(_box_mean(A - P), 20.0)
    top = _head_top(bone)
    z0, z1 = max(0, top - int(220 / sp[0])), max(1, top - int(120 / sp[0]))
    peak = arterial_peak(A, vessel, bone, sp)
    neck = np.zeros(A.shape, bool)
    neck[z0:z1] = vessel[z0:z1] & (A[z0:z1] >= ARTERY_SHARE * peak)
    inner = vessel & ~box_dilate(bone, mm_to_radii(12.0, sp))
    inner[:max(0, top - int(ARTERY_SEED_DEPTH_MM / sp[0]))] = False
    refs = [float(np.median(ratio[m])) for m in (neck, inner) if np.count_nonzero(m) >= 50]
    if not refs:
        return np.zeros(A.shape, bool)
    # the edge of a large artery is partly tissue, so its share drifts up: no seed in a shell around what is that bright
    m = vessel & (ratio >= min(refs) + WASHOUT_MARGIN) & ~box_dilate(A >= ARTERY_SHARE * peak, mm_to_radii(3.0, sp))
    lab = label_regions(m, sp)
    big = np.bincount(lab.ravel()) * float(np.prod(sp)) >= WASHOUT_MIN_MM3
    big[0] = False
    return big[lab]


def inside_skull(bone):
    """Cells with bone on both sides along x and along y in their plane: the inside of the head (and of the neck's
    vertebrae, which hold no sinus)."""
    return (np.maximum.accumulate(bone, axis=2) & np.maximum.accumulate(bone[:, :, ::-1], axis=2)[:, :, ::-1]
            & np.maximum.accumulate(bone, axis=1) & np.maximum.accumulate(bone[:, ::-1], axis=1)[:, ::-1])


def central_skull_base(inside, vtx, sp):
    """Below SINUS_THICK_DEPTH_MM under the vertex: the middle of the head, within SKULL_BASE_HALF_MM of the midline and
    in front of its back fifth. The cavernous sinus is there, but so are the carotid siphons, the circle of Willis and
    most aneurysms (real CTA: the tracing reached the sellar region of an aneurysm patient). The sinuses that belong
    low in the head run lateral (transverse, sigmoid) or at the back (torcula)."""
    out = np.zeros(inside.shape, bool)
    z1 = max(0, vtx - int(SINUS_THICK_DEPTH_MM / sp[0]))
    for z in range(z1):
        ys, xs = np.nonzero(inside[z])
        if ys.size < 100:
            continue
        xc, y0, y1 = 0.5 * (xs.min() + xs.max()), ys.min(), ys.max()
        half = SKULL_BASE_HALF_MM / sp[2]
        back = y1 - 0.2 * (y1 - y0) if y1 > y0 else y1      # rows grow towards the back of the head (LPS)
        out[z, :int(back), max(0, int(xc - half)):int(xc + half) + 1] = True
    return out


def trace_sinuses(mean, top, sp, vessel_hu=None):
    """The sinus mask of sinus_tracing."""
    return sinus_tracing(mean, top, sp, vessel_hu)[0]


def sinus_tracing(mean, top, sp, vessel_hu=None):
    """The dural sinuses of ONE contrast series, by anatomy: on a routine contrast CT they are as bright as the arteries
    (real CT: 154 against 153 HU) and there is no second phase to tell them apart. A sinus is a thick vessel inside the
    skull against the inner table (superior sagittal, transverse, sigmoid) or, high in the head, against the tentorium
    and falx (straight sinus, torcula). Cores: thick vessel cells within SINUS_REACH_MM of bone in the upper head, in
    long pieces. From there the tracing follows vessels that stay against bone, and thick ones only above the circle of
    Willis, so the carotids and the basilar artery are not reached. Returns the sinus mask on the given grid.
    mean, top: the series on the mask grid (its mean and brightest voxel per cell); sp: that grid's spacing."""
    from .segment import box_dilate, box_erode, label_regions, mm_to_radii
    bone = top >= 500
    inside = inside_skull(bone)
    near_bone = box_dilate(bone, tuple(max(1, r) for r in mm_to_radii(1.0, sp)))
    vtx = _head_top(bone)
    if vessel_hu is None:
        # halfway between brain (~40 HU) and the brightest 0.5 % of the head above the skull base, where the sinuses are:
        # the whole volume's top includes the neck and set it far above a venous series' sinuses (151 against ~102 HU)
        upper = np.zeros(bone.shape, bool)
        upper[max(0, vtx - int(90 / sp[0])):max(1, vtx - int(15 / sp[0]))] = True
        vals = mean[inside & upper & ~box_dilate(bone, mm_to_radii(3.0, sp)) & (mean > 0)]
        if vals.size < 1000:
            return np.zeros(bone.shape, bool), 150.0
        # and well clear of this patient's brain: a faint venous phase (top 89 HU) otherwise took in the cortex
        brain = float(np.median(vals))
        vessel_hu = max(brain + SINUS_ABOVE_BRAIN_HU, 0.5 * (brain + float(np.percentile(vals, 99.5))))
    vessel = inside & ~near_bone & (mean >= vessel_hu)
    voxel = float(np.prod(sp))
    lab = label_regions(vessel, sp)
    size = np.bincount(lab.ravel())
    small = size * voxel < PIECE_MIN_MM3
    small[0] = False
    vessel[small[lab]] = False
    dural =box_dilate(bone, mm_to_radii(SINUS_REACH_MM + 1.0, sp))     # against the inner table
    thick = box_erode(vessel, (1, 1, 1)) & vessel                        # 3 cells: ~3 mm and wider
    high = np.zeros(vessel.shape, bool)
    high[max(0, vtx - int(SINUS_THICK_DEPTH_MM / sp[0])):] = True
    # cores: thick, against bone, in the upper head, long
    core = thick & dural
    core[:max(0, vtx - int(SINUS_DEPTH_MM / sp[0]))] = False
    lab = label_regions(core, sp)
    size = np.bincount(lab.ravel())
    long = size * voxel >= SINUS_MIN_MM3
    long[0] = False
    seeds = long[lab]
    if not seeds.any():
        return seeds, vessel_hu
    # the way along the sinuses: vessels against bone anywhere inside the skull, thick vessels high in the head
    way = vessel & ((dural & box_dilate(thick, (1, 1, 1))) | (thick & high))
    lab = label_regions(way, sp)
    keep = np.bincount(lab[seeds], minlength=int(lab.max()) + 1) > 0
    keep[0] = False
    traced = box_dilate(keep[lab], (1, 1, 1)) & vessel      # with the lumen around it
    traced &= ~central_skull_base(inside, vtx, sp)
    # the dural sinuses of an adult hold ~15-25 ml (real CTAs: 14-24 ml found with both phases); far more is cortex or
    # enhancing tissue, and nothing is better than that
    if traced.sum() * voxel > SINUS_MAX_ML * 1000:
        return np.zeros(bone.shape, bool), vessel_hu
    return traced, vessel_hu


def _head_top(bone):
    """Index of the highest plane with bone: the vertex, on a scan that reaches it."""
    planes = np.flatnonzero(bone.sum(axis=(1, 2)) >= 20)
    return int(planes[-1]) if planes.size else bone.shape[0] - 1


def arterial_peak(A, vessel, bone, sp):
    """HU of the large arteries on the arterial series: the thick vessels of the neck, 12-22 cm under the vertex. Contrast
    in the arm and chest veins is far brighter and lower down; a sinus is higher up and dimmer (real CTA: 0.5-0.75 of it)."""
    from .segment import box_erode
    top = _head_top(bone)
    z0, z1 = max(0, top - int(220 / sp[0])), max(1, top - int(120 / sp[0]))
    thick = box_erode(vessel[z0:z1], (1, 1, 1))
    v = A[z0:z1][thick]
    v = v[v < PEAK_MAX_HU]
    return float(np.percentile(v, 90)) if v.size >= 50 else 400.0


def sinus_seeds(vessel, bone, sp, reach_mm=SINUS_REACH_MM, thick_mm=SINUS_THICK_MM, depth_mm=SINUS_DEPTH_MM, clear=None):
    """The dural sinuses by anatomy: thick vessels against the skull in the upper head. No artery up there is this wide,
    and an aneurysm lies in the cisterns, away from the vault. This is what still finds them when the 'arterial' series
    was late, so the sinuses were full on it and wash out at the same pace as the arteries (3 of 7 real CTAs)."""
    from .segment import box_dilate, box_erode, label_regions, mm_to_radii
    top = _head_top(bone)
    z0 = max(0, top - int(depth_mm / sp[0]))
    out = np.zeros(vessel.shape, bool)
    r = tuple(max(1, k) for k in mm_to_radii(thick_mm / 2, sp))
    core = box_erode(vessel[z0:], r) & box_dilate(bone[z0:], mm_to_radii(reach_mm + thick_mm / 2, sp))
    # long: the carotid in its canal and an aneurysm against the clinoid are short pieces, a sinus runs for centimetres
    lab = label_regions(core, sp)
    size = np.bincount(lab.ravel())
    long = size * float(np.prod(sp)) >= SINUS_MIN_MM3
    if clear is not None:
        long &= np.bincount(lab[clear[z0:]], minlength=size.size) >= 0.5 * size
    long[0] = False
    out[z0:] = long[lab]
    return out


TREE_SURE = 0.9         # vessel_trees: atlas share of arteries for an artery seed (1 - this for a vein seed)
TREE_ARTERY_MM = 1.2    # and a lumen this thick around it: the ICA, M1, the basilar
TREE_VEIN_MM = 1.5


def vessel_trees(arteries, veins, sinus, prior, sp):
    """Arteries and veins of ONE contrast series as two trees that compete for its vessels (compete): veins from the
    traced dural sinuses and the thick vessels the atlas is sure are veins, arteries from the thick ones it is sure are
    arteries. Every vessel goes to the tree that reaches it first along the vessels, so a vessel takes the label of what
    it is joined to instead of the atlas's guess for its place: a sinus stays a vein all along (the atlas alone put part
    of a real infant's superior sagittal sinus with the arteries), the internal jugular vein follows the sigmoid sinus.
    Real CTAs, the venous series alone against its own two phases: 84 % of the head's vessel cells right with the atlas,
    88 % as trees (each of six studies better). Vessels neither tree reaches keep their label. prior: the atlas's
    (dcv/atlas.py classify_head)."""
    vessel = arteries | veins
    pa = prior(vessel)
    seed_v = vessel & (sinus | ((pa <= 1 - TREE_SURE) & _thick(vessel, sp, TREE_VEIN_MM)))
    seed_a = vessel & (pa >= TREE_SURE) & _thick(vessel, sp, TREE_ARTERY_MM) & ~seed_v
    if not seed_a.any() or not seed_v.any():
        return arteries, veins
    a, v = compete(vessel, seed_a, seed_v)
    reached = a | v
    return a | (arteries & ~reached), v | (veins & ~reached)


def _thick(mask, sp, mm):
    """Cells of mask with no cell outside it closer than mm (the grid's edge counts as inside): the lumen around them is
    at least that thick."""
    out = mask.copy()
    r = [int(np.ceil(mm / s)) for s in sp]
    for dz in range(-r[0], r[0] + 1):
        for dy in range(-r[1], r[1] + 1):
            for dx in range(-r[2], r[2] + 1):
                d = np.sqrt((dz * sp[0]) ** 2 + (dy * sp[1]) ** 2 + (dx * sp[2]) ** 2)
                if 0 < d < mm:
                    out &= _shifted(mask, (dz, dy, dx))
    return out


def _shifted(m, off):
    """m moved by off cells, the edge filled with True: out[i] = m[i + off]."""
    out = np.ones(m.shape, bool)
    src, dst = [], []
    for o, n in zip(off, m.shape):
        src.append(slice(max(0, o), n + min(0, o)))
        dst.append(slice(max(0, -o), n + min(0, -o)))
    out[tuple(dst)] = m[tuple(src)]
    return out


def compete(vessel, seeds_a, seeds_b):
    """Grow both seed sets through `vessel`, one layer at a time: a voxel goes to whichever front reaches it first, so
    the two trees meet only where they really touch. Frontier lists over ~1 M vessel voxels of a 20 M grid: well under 1 s."""
    nz, ny, nx = vessel.shape
    labels = np.zeros(vessel.size, np.uint8)        # 0 = free or outside, 1, 2 = taken, 3 = not a vessel
    labels[~vessel.ravel()] = 3
    labels[seeds_b.ravel() & vessel.ravel()] = 2
    labels[seeds_a.ravel() & vessel.ravel()] = 1
    a, b = np.flatnonzero(labels == 1), np.flatnonzero(labels == 2)
    offsets = np.array([1, -1, nx, -nx, nx * ny, -nx * ny])

    def step(front, tag):
        n = (front[:, None] + offsets).ravel()
        n = n[(n >= 0) & (n < labels.size)]
        n = n[labels[n] == 0]
        n = np.unique(n)
        labels[n] = tag
        return n

    while a.size or b.size:
        a = step(a, 1) if a.size else a
        b = step(b, 2) if b.size else b
    out = labels.reshape(vessel.shape)
    return out == 1, out == 2
