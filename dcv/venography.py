"""Venography view of a contrast head CT (preset "CT Venography"): the dural sinuses and the cortical and deep veins in
blue over a faint brain, inside a faint skull and skin.

One HU curve cannot do it: brain and scalp have the same HU, and the edge of bone passes through the vessels' HU. So the
inside of the skull gets a domain of its own in the 3D copy (stored as VEIN_OFFSET + HU, where the vein curves read):
there the brain and the vessels have their own curves, and the scalp and face outside keep theirs.

  cavity: the inside of the skull on the mask grid, with the gaps in it closed first (an infant's fontanelles, sutures,
          foramina: real CT, an open anterior fontanelle left the front of the superior sagittal sinus outside it);
  sheets: bright sheets by the Hessian (sheets()): a newborn's skull plates have the HU of veins, not their shape; they
          bound the cavity and are never a lumen;
  lumen:  what the contrast made bright. With the plain scan of the study: brighter by >= 50 HU than there, and nothing
          bright around it there (bone edge, calcification); a scan that moved by a fraction of a voxel changes the
          bone's edge by hundreds of HU (real infant CT: +-500 HU), so the plain scan's 3x3x3 neighbourhood decides.
          Without one: bright and away from bone, so a sinus right against the skull keeps a voxel less;
  bone area: the bone colour lock's, minus the lumens inside the cavity (a sinus against the inner table), plus what
          was already bright before contrast inside it;
  layer:  the voxels around the cavity's domain are stored as its 0 HU, so the vessel against the bone or the scalp
          ends in its own domain (smooth surface) instead of on the jump to the other code (a ladder of voxel faces).
"""

from __future__ import annotations

import numpy as np

from .phases import _pool, inside_skull, upsample
from .segment import BONE_OFFSET, VEIN_OFFSET, box_dilate, box_erode, label_regions

CLOSE_MM = 10.0         # gaps in the skull up to twice this are closed for the cavity
LUMEN_MIN_HU = 70.0     # contrast in a routine head CT's veins: ~95-175 HU (real CT), brain 30-40
LUMEN_MAX_HU = 350.0
ENHANCE_HU = 50.0       # brighter than on the plain scan by at least this
PLAIN_MAX_HU = 120.0    # the voxel on the plain scan: blood, not bone or calcium
PLAIN_NEAR_HU = 200.0   # nothing this bright around it on the plain scan (a newborn's skull there: 200-400 HU)
BEFORE_HU = 90.0        # bright around it on the plain scan: bone edge, calcification
OUTSIDE_MIN_HU = 120.0  # no plain scan: a lumen outside the skull is at least this bright
RIM_MM = 2.0            # encode(): the cavity's edge, where the skull lies
MEAN_MIN_HU = 60.0      # a lumen with its 6 neighbours: at least this (a thin cortical vein ~85, noise less)
BRAIN_TOP_HU = 55       # what is no lumen inside the skull is stored at most this: under the vein curve
MATCH_MIN = 0.85        # lines_up: the plain scan is used from this share on (a newborn lined up by register(): 74 %, its
                        # sutures then still came out as veins; the sheets alone did better)


def cavity(top, sp, mean=None):
    """Inside of the skull on the mask grid (top / mean = brightest / mean voxel of each cell, sp = cell size mm). The
    skull that bounds it: its dense bone, and its bright sheets (sheets(): a newborn's plates are 150-450 HU, real CT, and
    at 300 HU the cavity ran out through them and through the open sutures into the scalp). The larger of the cavities
    bounded at 500 and at 300 HU."""
    plates = sheets(mean, sp) if mean is not None else np.zeros(top.shape, bool)
    best = np.zeros(top.shape, bool)
    for hu in (500.0, 300.0):
        c = _cavity((top >= hu) | plates, top if mean is None else mean, sp)
        if c.sum() > best.sum():
            best = c
    return best


def _cavity(barrier, mean, sp):
    r = tuple(max(1, int(round(CLOSE_MM / k))) for k in sp)
    shut = box_dilate(barrier, r)
    ins = inside_skull(shut)
    core = box_erode(ins & ~shut, (2, 2, 2))
    if not core.any():
        return np.zeros(barrier.shape, bool)
    # the largest piece: the brain's, not an orbit or the neck's vertebral canal
    lab = label_regions(core, sp)
    size = np.bincount(lab.ravel())
    size[0] = 0
    # grown back to the skull a cell at a time, not across it nor into air: the core starts ~CLOSE_MM inside the skull
    room = ins & ~barrier & (mean > -300)
    cav = lab == size.argmax()
    for _ in range(max(r) + 6):         # ~CLOSE_MM + the erosion, and the box corners: to the inner table
        cav = box_dilate(cav, (1, 1, 1)) & room
    return cav


SHEET_SIGMA_MM = 1.0    # sheets(): the scale of a skull plate (a newborn's is 1-2 mm thick)
SHEET_MIN_HU = 100.0    # cells this bright and up are looked at
SHEET_SHARE = 0.35      # across it the curvature dominates: the second one is at most about this share of it
SHEET_CURVE = 30.0      # HU (Hessian x sigma^2): a weaker sheet is noise (a 2 mm plate of 250 HU: ~50)
SHEET_RIDGE = 0.8       # on the ridge: the gradient (x sigma) at most this share of the curvature (a 2 mm plate between
                        # voxels: 0.75); at the edge of a large bright thing (a sinus) it is ~1, so that edge is no sheet


def sheets(mean, sp):
    """Bright sheets on the mask grid (mean HU of each cell, sp = cell size mm): skull plates, also dura. Their Hessian
    has one strong negative curvature (across) and little along them, where a vessel has two (a tube) and the inside of a
    sinus little at all. HU alone cannot tell a newborn's plates and their edges at the open sutures (150-450 HU) from
    veins (real CT: the sutures came out as veins)."""
    at = np.nonzero(mean >= SHEET_MIN_HU)
    if not len(at[0]):
        return np.zeros(mean.shape, bool)
    g = _gauss(mean.astype(np.float32), [SHEET_SIGMA_MM / k for k in sp])
    grads = np.gradient(g, *sp)
    del g
    w = SHEET_SIGMA_MM ** 2
    h = [np.gradient(grads[a], sp[b], axis=b)[at] * w for a, b in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))]
    slope = np.sqrt(sum(gr[at] ** 2 for gr in grads)) * SHEET_SIGMA_MM
    del grads
    lam = _eig_sym(*h)
    l2, l3 = lam[:, 1], lam[:, 2]
    share = np.abs(l2) / (np.abs(l3) + 1e-9)
    out = np.zeros(mean.shape, bool)
    out[at] = (l3 < -SHEET_CURVE) & (share < SHEET_SHARE) & (slope < SHEET_RIDGE * np.abs(l3))
    return out


def _gauss(a, sig):
    """Gaussian blur of a, sig = sigma in cells per axis (separable, +-2.5 sigma)."""
    out = a
    for ax, s in enumerate(sig):
        r = int(np.ceil(2.5 * s))
        if r < 1:
            continue
        k = np.exp(-0.5 * (np.arange(-r, r + 1) / s) ** 2)
        k /= k.sum()
        pad = [(0, 0)] * 3
        pad[ax] = (r, r)
        p = np.pad(out, pad, mode="edge")
        acc = np.zeros_like(out)
        for i, wt in enumerate(k):
            sl = [slice(None)] * 3
            sl[ax] = slice(i, i + out.shape[ax])
            acc += np.float32(wt) * p[tuple(sl)]
        out = acc
    return out


def _eig_sym(a, b, c, d, e, f):
    """Eigenvalues (n, 3) of the symmetric [[a, d, e], [d, b, f], [e, f, c]], sorted by absolute value."""
    p1 = d * d + e * e + f * f
    q = (a + b + c) / 3
    p = np.sqrt(((a - q) ** 2 + (b - q) ** 2 + (c - q) ** 2 + 2 * p1) / 6) + 1e-12
    b0, b1, b2, b3, b4, b5 = (a - q) / p, d / p, e / p, (b - q) / p, f / p, (c - q) / p
    det = b0 * (b3 * b5 - b4 * b4) - b1 * (b1 * b5 - b4 * b2) + b2 * (b1 * b4 - b3 * b2)
    phi = np.arccos(np.clip(det / 2, -1, 1)) / 3
    l1 = q + 2 * p * np.cos(phi)
    l3 = q + 2 * p * np.cos(phi + 2 * np.pi / 3)
    lam = np.stack([l1, 3 * q - l1 - l3, l3], -1)
    return np.take_along_axis(lam, np.argsort(np.abs(lam), axis=-1), -1)


def plain_on_grid(plain, vol, motion=None):
    """The plain scan's HU on vol's grid (int16), or None. Same grid: as it is. Same patient coordinates (frame of
    reference) but another field of view or tilt: the nearest voxel (the lumen rule looks at 3x3x3 of it anyway).
    motion = (R, t) from register(): a point p of the plain scan lies at R p + t on this series."""
    if plain is None:
        return None
    if motion is None and plain.data.shape == vol.data.shape and np.allclose(plain.spacing, vol.spacing, atol=0.01) \
            and np.allclose(plain.origin, vol.origin, atol=0.5) and np.allclose(plain.direction, vol.direction, atol=1e-3):
        return plain.data
    if not plain.frame_uid or plain.frame_uid != vol.frame_uid:
        return None
    rot, shift = motion if motion is not None else (np.eye(3), np.zeros(3))
    to_plain = lambda zyx: plain.world_to_index((vol.index_to_world(zyx) - shift) @ rot)    # noqa: E731  (R^T (w - t))
    # index on the plain scan = a + z * dz + y * dy + x * dx (both grids are linear in patient coordinates)
    nz, ny, nx = vol.data.shape
    a = to_plain(np.zeros(3))
    dz, dy, dx = (to_plain(np.eye(3)[k]) - a for k in range(3))
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float32)
    plane = [a[k] + yy * dy[k] + xx * dx[k] for k in range(3)]
    pz, py, px = plain.data.shape
    out = np.full(vol.data.shape, -1024, np.int16)
    for z in range(nz):
        qz, qy, qx = (np.rint(plane[k] + z * dz[k]).astype(np.int32) for k in range(3))
        ok = (qz >= 0) & (qz < pz) & (qy >= 0) & (qy < py) & (qx >= 0) & (qx < px)
        out[z][ok] = plain.data[qz[ok], qy[ok], qx[ok]]
    return out


def lines_up(plain, src):
    """Share of this series' bone (>= 300 HU, every 2nd voxel) that has bone (>= 200) within a voxel on the plain scan.
    A patient who moved between the scans (real newborn CT: 9 % instead of ~90 %) would turn every bone edge into a
    vessel, so the plain scan is used only above MATCH_MIN."""
    b = src[::2, ::2, ::2] >= 300
    if not b.any():
        return 0.0
    near = _max3(plain[::2, ::2, ::2])
    return float((near[b] >= 200).mean())


REG_CELL_MM = (2.0, 1.0)        # register(): the coarse and the fine pass (a 0.6 mm pass gained nothing on newborn CT)
REG_CAP_MM = (10.0, 4.0)        # distances beyond this count as this (bone of one scan that the other does not have)
REG_POINTS = 20000
REG_BONE_HU = 300.0         # a newborn's skull is 150-450 HU (real CT)


def register(plain, vol):
    """The motion of the patient between the plain scan and this series, (R, t): a point p of the plain scan lies at
    R p + t here (patient mm). Rigid, as the skull is: the plain scan's bone (points) is moved onto this series' bone
    (a map of the distance to it), first on ~2 mm cells from a search of shifts, then rotations and shifts by halving
    steps, then again on ~1 mm cells. Real newborn CT that moved between the scans: 42 % of the bone matched before,
    see lines_up. None when either has no bone."""
    rot, shift = np.eye(3), np.zeros(3)
    for k, (cell, cap) in enumerate(zip(REG_CELL_MM, REG_CAP_MM)):
        target = _distance_map(vol, cell, cap)
        pts = _bone_points(plain, cell)
        if target is None or pts is None:
            return None
        cost = _cost(vol, target, pts, cap)
        centre = pts.mean(0)
        params = _params_of(rot, shift, centre)
        if k == 0:
            # shifts first: a grid of ±16 mm
            best = (cost(params), params)
            for dz in range(-16, 17, 4):
                for dy in range(-16, 17, 4):
                    for dx in range(-16, 17, 4):
                        p = params + np.array([0, 0, 0, dx, dy, dz], float)
                        c = cost(p)
                        if c < best[0]:
                            best = (c, p)
            params = best[1]
        params = _descend(cost, params, 4.0 if k == 0 else 1.0, 0.25)
        rot, shift = _motion(params, centre)
    return rot, shift


def _distance_map(vol, cell, cap):
    """(distance in mm to this series' bone on cells of ~cell mm, steps, cell size) or None."""
    steps = tuple(max(1, int(round(cell / s))) for s in vol.spacing)
    _, top = _pool(vol.data, steps)
    bone = top >= REG_BONE_HU
    if not bone.any():
        return None
    size = float(np.mean([s * k for s, k in zip(vol.spacing, steps)]))
    dist = np.full(top.shape, cap, np.float32)
    dist[bone] = 0
    cur = bone
    for i in range(1, int(np.ceil(cap / size)) + 1):
        nxt = box_dilate(cur, (1, 1, 1))
        dist[nxt & ~cur] = min(cap, i * size)
        cur = nxt
    return dist, steps


def _bone_points(plain, cell):
    """Patient coordinates (n, 3) of the plain scan's bone on cells of ~cell mm, at most REG_POINTS of them."""
    steps = tuple(max(1, int(round(cell / s))) for s in plain.spacing)
    _, top = _pool(plain.data, steps)
    idx = np.argwhere(top >= REG_BONE_HU).astype(np.float64)
    if len(idx) < 50:
        return None
    if len(idx) > REG_POINTS:
        idx = idx[np.random.default_rng(0).choice(len(idx), REG_POINTS, replace=False)]
    return plain.index_to_world(idx * steps + (np.array(steps) - 1) / 2.0)


def _rotation(v):
    """Rotation matrix of the rotation vector v (radians, Rodrigues)."""
    a = float(np.linalg.norm(v))
    if a < 1e-12:
        return np.eye(3)
    k = v / a
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * kx + (1 - np.cos(a)) * kx @ kx


def _motion(params, centre):
    """(R, t) of params = (rotation vector in degrees, shift in mm) about centre."""
    rot = _rotation(np.radians(params[:3]))
    return rot, centre + params[3:] - rot @ centre


def _params_of(rot, shift, centre):
    """params of (R, t) about centre (inverse of _motion; R small, as register() builds it)."""
    c, s = np.clip((np.trace(rot) - 1) / 2, -1, 1), None
    a = np.arccos(c)
    if a < 1e-9:
        v = np.zeros(3)
    else:
        s = np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]]) / (2 * np.sin(a))
        v = s * a
    return np.concatenate([np.degrees(v), shift + rot @ centre - centre])


def _cost(vol, target, pts, cap):
    """Mean distance (mm, at most cap) from the moved points to this series' bone, trilinear in the map."""
    dist, steps = target
    shape = np.array(dist.shape)
    sp = np.array(steps, float)

    def cost(params):
        rot, shift = _motion(params, centre)
        q = vol.world_to_index(pts @ rot.T + shift)
        q = (q - (sp - 1) / 2.0) / sp
        f = np.floor(q)
        w = q - f
        f = f.astype(np.int64)
        inside = np.all((f >= 0) & (f < shape - 1), axis=1)
        out = np.full(len(q), cap, np.float64)
        fi, wi = f[inside], w[inside]
        acc = np.zeros(len(fi))
        for c in range(8):
            o = np.array([(c >> 2) & 1, (c >> 1) & 1, c & 1])
            wt = np.prod(np.where(o, wi, 1 - wi), axis=1)
            acc += wt * dist[fi[:, 0] + o[0], fi[:, 1] + o[1], fi[:, 2] + o[2]]
        out[inside] = acc
        return float(out.mean())
    centre = pts.mean(0)
    return cost


def _descend(cost, params, step, last):
    """Coordinate descent on the 6 params (degrees, mm) from step down to last, halving."""
    best = cost(params)
    while step >= last:
        better = True
        while better:
            better = False
            for i in range(6):
                for sgn in (1, -1):
                    p = params.copy()
                    p[i] += sgn * step
                    c = cost(p)
                    if c < best - 1e-6:
                        best, params, better = c, p, True
        step /= 2
    return params


SLAB_MM = 40.0          # slab_plain: the plain scan is lined up slab by slab along the body
SLAB_SHIFT_MM = 4.0     # each slab moved by up to this
SLAB_POINTS = 6000      # bone points per slab that score a shift


def slab_plain(plain, src, spacing, tol):
    """The plain scan (on src's grid) lined up with src slab by slab: SLAB_MM along z, each slab moved by up to
    SLAB_SHIFT_MM, and which planes then line up: the share of src's bone (>= 300 HU) with plain bone (>= 200 HU) within
    tol voxels (z, y, x) reaches MATCH_MIN. A CTA from the chest to the vertex: the head lies still while the jaw and neck
    bend between the scans (real CTA: the head 0.94-1.0, the whole 0.79 even after register()). Returns (plain moved per
    slab, -1024 where no slab lines up; bool per plane)."""
    from .segment import box_max
    near = box_max(plain, tol) >= 200
    out = np.full(src.shape, -1024, plain.dtype)
    ok = np.zeros(src.shape[0], bool)
    n = max(1, int(round(SLAB_MM / spacing[0])))
    reach = [int(round(SLAB_SHIFT_MM / s)) for s in spacing]
    shape = np.array(src.shape)
    for z0 in range(0, src.shape[0], n):
        z1 = min(src.shape[0], z0 + n)
        pts = np.argwhere(src[z0:z1] >= 300)
        if len(pts) < 200:
            continue
        pts = pts[::max(1, len(pts) // SLAB_POINTS)]           # every k-th, as the web takes them
        pts[:, 0] += z0

        def share(off):
            q = pts + off
            inside = np.all((q >= 0) & (q < shape), axis=1)
            return float(np.mean(near[tuple(q[inside].T)])) if inside.any() else 0.0
        off = np.zeros(3, int)
        best = share(off)
        # a slab that lines up as it is stays so: shifted alone it left a step in the bone between two slabs
        for step, rr in (() if best >= MATCH_MIN else ((2, reach), (1, (2, 2, 2)))):   # every 2nd voxel, then near the best
            centre = off.copy()
            for dz in range(-rr[0], rr[0] + 1, step):
                for dy in range(-rr[1], rr[1] + 1, step):
                    for dx in range(-rr[2], rr[2] + 1, step):
                        o = centre + np.array([dz, dy, dx])
                        if np.any(np.abs(o) > reach):
                            continue                            # the search around the best stays within reach
                        s = share(o)
                        if s > best or (s == best and np.abs(o).sum() < np.abs(off).sum()):     # a tie: the smaller move
                            best, off = s, o
        if best < MATCH_MIN:
            continue
        ok[z0:z1] = True
        # out[z, y, x] = plain[z + oz, y + oy, x + ox], what lies outside the plain scan stays -1024
        dst, srcs = [slice(z0, z1)], [slice(z0 + off[0], z1 + off[0])]
        for o, size in zip(off[1:], src.shape[1:]):
            dst.append(slice(max(0, -o), size - max(0, o)))
            srcs.append(slice(max(0, o), size + min(0, o)))
        lo, hi = max(0, -(z0 + off[0])), max(0, z1 + off[0] - plain.shape[0])
        dst[0] = slice(z0 + lo, z1 - hi)
        srcs[0] = slice(z0 + off[0] + lo, z1 + off[0] - hi)
        out[tuple(dst)] = plain[tuple(srcs)]
    return out, ok


def _max3(a):
    """Maximum over 3 x 3 x 3 voxels."""
    out = a.copy()
    for ax in range(3):
        src = out.copy()
        lo, hi = [slice(None)] * 3, [slice(None)] * 3
        lo[ax], hi[ax] = slice(1, None), slice(None, -1)
        np.maximum(out[tuple(lo)], src[tuple(hi)], out=out[tuple(lo)])
        np.maximum(out[tuple(hi)], src[tuple(lo)], out=out[tuple(hi)])
    return out


def encode(d, src, area, cav, plain, reach=1, spacing=(1.0, 1.0, 1.0), plates=None, arteries=None):
    """Write the venography domains into d (the 3D copy, int16, values of src = this series on the copy's grid).
    area: the bone colour lock's area (None = off); cav: the cavity on the copy's grid; plain: the plain scan there;
    reach: voxels around a voxel on the plain scan that must hold nothing bright (2 after register(): what is left of
    the motion, ~1 mm on a newborn's skull, would turn its edge into vessels); spacing: of the copy's grid (mm);
    plates: the bright sheets on the copy's grid (masks_on), never a lumen; arteries: the arteries on the copy's grid
    (VolumeState.artery_mask, a voxel wider), no lumen either: a routine contrast CT shows arteries as bright as veins,
    and the venography showed the MCA and ACA as veins.
    Within RIM_MM of the cavity's edge a lumen must be clear of BEFORE_HU on the plain scan, not only of PLAIN_NEAR_HU:
    there lies the skull, which on a newborn is only 150-400 HU (real CT: a blue shell inside the skull).
    Outside the skull the normal domain keeps only the lumens' HU above LUMEN_MIN_HU (the neck's veins, a vessel in an
    encephalocele: real newborn CT), so a newborn's skull and the edge of bone there are not drawn as vessels."""
    lumen = (src >= LUMEN_MIN_HU) & (src < LUMEN_MAX_HU)
    if plates is not None:
        lumen &= ~plates                            # skull plates and their edges, dura: no vessel (sheets())
    if arteries is not None:
        lumen &= ~arteries
    if plain is not None:
        near = _max3(plain)
        for _ in range(reach - 1):
            near = _max3(near)
        # the differences only where they decide: int32 copies of a whole CTA would take ~700 MB each
        at = np.nonzero(lumen)
        p = plain[at].astype(np.int32)
        sm, pm = _mean7(src, at), _mean7(plain, at)
        rim = cav & box_dilate(~cav, tuple(max(1, int(round(RIM_MM / k))) for k in spacing))
        keep = (sm >= MEAN_MIN_HU) & (sm - pm >= ENHANCE_HU) & (p < PLAIN_MAX_HU)             & (near[at] < np.where(rim[at], BEFORE_HU, PLAIN_NEAR_HU))
        del rim
        lumen[tuple(i[~keep] for i in at)] = False
        before = near >= BEFORE_HU
        del near
    else:
        # away from bone; a newborn's skull is 150-450 HU (real CT), so anything from 200 HU outside the cavity counts
        lumen &= ~box_dilate((src >= 300) | ((src >= 200) & ~cav), (1, 2, 2))
        lumen &= cav | (src >= OUTSIDE_MIN_HU)      # outside the skull muscle and glands reach 70-110 HU
        at = np.nonzero(lumen)
        lumen[tuple(i[_mean7(src, at) < MEAN_MIN_HU] for i in at)] = False
        before = ~lumen
    bone = cav & (src >= LUMEN_MIN_HU) & before
    if arteries is not None:
        bone &= ~arteries                           # the arteries taken out are brain here, not bone
    if area is not None:
        bone |= area
    bone &= ~lumen
    inner = cav & ~bone
    layer = box_dilate(inner, (1, 1, 1)) & ~inner
    hide = ~cav & ~bone & ~lumen & (src >= LUMEN_MIN_HU)
    d[hide] = LUMEN_MIN_HU - 1
    if area is not None:
        b = np.clip(src[bone].astype(np.int32), -1024, 32767 - BONE_OFFSET - 1)
        d[bone] = (-BONE_OFFSET - b).astype(np.int16)
    # only a lumen reaches the vein curve: noise in a thin newborn scan put ~15 % of the brain at 60-80 HU (a blue haze)
    d[inner] = (VEIN_OFFSET + np.clip(src[inner].astype(np.int32), -1024, BRAIN_TOP_HU)).astype(np.int16)
    vessel = inner & lumen
    d[vessel] = (VEIN_OFFSET + np.clip(src[vessel].astype(np.int32), -1024, 3000)).astype(np.int16)
    d[layer] = VEIN_OFFSET


def _mean7(a, at):
    """Mean of a at the voxels at (np.nonzero) and their 6 neighbours (edges repeated): a lumen is more than a spike
    of noise."""
    total = a[at].astype(np.int32)
    for ax in range(3):
        n = a.shape[ax]
        for step in (-1, 1):
            j = list(at)
            j[ax] = np.clip(at[ax] + step, 0, n - 1)
            total = total + a[tuple(j)]
    return total / 7.0


def masks_on(state, f):
    """(cavity, sheets a cell wider) of a VolumeState on its 3D copy's grid (every f-th row and column)."""
    mean, top, sp = state._coarse_data()
    plates = sheets(mean, sp)
    cav = cavity(top, sp, mean)
    up = lambda m: upsample(m, state.mask_steps, state.vol.shape, f)      # noqa: E731
    return up(cav), up(box_dilate(plates, (1, 1, 1)))
