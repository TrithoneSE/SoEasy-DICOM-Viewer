"""Where arteries and where veins run in the head: an atlas made from multiphase CTAs (tools/build_vessel_atlas.py), and
its use on a new scan.

A head is matched to the atlas by its skull: an affine transform (12 parameters: position, rotation, size, shear) that
lays the scan's skull surface onto the atlas's average skull (Gauss-Newton on the atlas's distance map; the mandible and
other misfits are down-weighted). The atlas then says, at each place, how often a vessel there was an artery. Real CTAs,
leave-one-out (atlas from 4, scored on the 5th against its own two-phase result): the skull fits within ~1 mm, and the
location alone labels 87-95 % of the vessel voxels right; that, with this patient's own HU learnt from the places the
atlas is sure of and the evidence pooled along each vessel, 88-89 % from ONE series (the anatomy tracing alone: 57-66 %).
Small cortical vessels against the skull are where it errs: their course varies from person to person.
numpy only (the build tool needs scipy)."""

from __future__ import annotations

import json
import os
import sys

import numpy as np

HEAD_MM = 120.0          # the head under the vertex whose skull is matched (skull base in, most of the mandible out)
ATLAS_DEPTH_MM = 150.0   # the atlas reaches this far under the vertex
MIN_FIT = 0.4            # share of the skull points within 2 mm of the atlas skull: below it the atlas is not used (real
                         # scans: CTAs 0.70-0.87, routine head CTs 0.48-0.75, the lower one with a tumour eroding bone)
PITCHES = (0.0, -20.0, 20.0, -40.0, 40.0)    # head pitches (degrees) the registration starts from
PITCH_MARGIN = 0.02      # a tilted start must fit clearly better than the upright one
SURE = 0.85              # atlas share of arteries above which (or below 1 - SURE) a place counts as known
SUPPORT_MIN = 0.001      # vessels per study near a place for its share to count: thin arteries leave little (real CTA:
                         # at 0.05, 62 % of the arteries counted as unknown and the veins around them won)
LLR_CLIP = 4.0
_MAGIC = b"SEVA"


def vertex(bone):
    planes = np.flatnonzero(bone.sum(axis=(1, 2)) >= 20)
    return int(planes[-1]) if planes.size else bone.shape[0] - 1


def surface(bone):
    """Bone cells with a non-bone 6-neighbour."""
    inner = bone.copy()
    for ax in range(3):
        for s in (1, -1):
            inner &= np.roll(bone, s, ax)
    return bone & ~inner


def head_points(bone, sp, n=8000, seed=0):
    """Up to n points (mm, z y x) of the skull surface, HEAD_MM under the vertex and up."""
    s = surface(bone)
    s[:max(0, vertex(bone) - int(HEAD_MM / sp[0]))] = False
    idx = np.argwhere(s)
    if len(idx) > n:
        idx = idx[np.random.default_rng(seed).choice(len(idx), n, replace=False)]
    return idx * np.asarray(sp, float)


def sample(vol, sp, origin, pts, grad=False):
    """Trilinear value (and gradient per mm) of a grid whose cell i lies at origin + i * sp mm; outside: the edge."""
    g = (pts - origin) / np.asarray(sp, float)
    top = np.array(vol.shape) - 1
    g = np.clip(g, 0, top - 1e-6)
    i0 = np.floor(g).astype(int)
    f = g - i0
    i1 = np.minimum(i0 + 1, top)
    val = np.zeros(len(pts))
    gr = np.zeros((len(pts), 3)) if grad else None
    for dz in (0, 1):
        wz = f[:, 0] if dz else 1 - f[:, 0]
        for dy in (0, 1):
            wy = f[:, 1] if dy else 1 - f[:, 1]
            for dx in (0, 1):
                wx = f[:, 2] if dx else 1 - f[:, 2]
                c = vol[(i1 if dz else i0)[:, 0], (i1 if dy else i0)[:, 1], (i1 if dx else i0)[:, 2]].astype(float)
                val += wz * wy * wx * c
                if grad:
                    gr[:, 0] += (1 if dz else -1) * wy * wx * c
                    gr[:, 1] += (1 if dy else -1) * wz * wx * c
                    gr[:, 2] += (1 if dx else -1) * wz * wy * c
    return (val, gr / np.asarray(sp, float)) if grad else val


class Affine:
    """Scan mm -> atlas mm: q = M (p - c) + t."""

    def __init__(self, M, c, t):
        self.M, self.c, self.t = np.asarray(M, float), np.asarray(c, float), np.asarray(t, float)

    def __call__(self, p):
        return (p - self.c) @ self.M.T + self.t

    def inverse(self, q):
        return (q - self.t) @ np.linalg.inv(self.M).T + self.c


class VesselAtlas:
    def __init__(self, dist, prob, support, sp, origin, centre):
        self.dist, self.prob, self.support = dist, prob, support
        self.sp, self.origin, self.centre = np.asarray(sp, float), np.asarray(origin, float), np.asarray(centre, float)

    def register(self, bone, sp, iters=60):
        """(Affine, fit) for a scan's bone mask on its grid of spacing sp (mm, z y x). Starts from the centroids, at
        several head pitches: a routine head CT is often scanned along the orbitomeatal line, ~35 degrees from a CTA (real
        CTs: 37 degrees). Huber weights shrinking from 10 to 3 mm keep the mandible and bright arteries from pulling.
        fit = share of the skull points within 2 mm of the atlas skull afterwards."""
        pts = head_points(bone, sp)
        if len(pts) < 500:
            return None, 0.0
        c = pts.mean(0)
        p = pts - c

        def fit_of(M, t):
            return float(np.mean(sample(self.dist, self.sp, self.origin, p @ M.T + t) < 2.0))

        best = None
        for deg in PITCHES:
            a = np.radians(deg)
            M0 = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
            M, t = self._descend(p, M0, self.centre.copy(), 20)
            f = fit_of(M, t)
            if best is None or f > best[0] + PITCH_MARGIN:
                best = (f, M, t)
        M, t = self._descend(p, best[1], best[2], iters)
        return Affine(M, c, t), fit_of(M, t)

    def _descend(self, p, M, t, iters):
        for it in range(iters):
            r, g = sample(self.dist, self.sp, self.origin, p @ M.T + t, grad=True)
            k = max(3.0, 10.0 * (1 - it / (0.6 * iters)))
            w = np.where(r <= k, 1.0, k / np.maximum(r, 1e-6))
            J = np.concatenate([g[:, :, None] * p[:, None, :], g[:, :, None]], axis=2).reshape(len(p), 12)
            H = (J * w[:, None]).T @ J + 1e-3 * np.eye(12)
            A = np.concatenate([M, t[:, None]], 1) + np.linalg.solve(H, -(J * w[:, None]).T @ r).reshape(3, 4)
            M, t = A[:, :3], A[:, 3]
        return M, t

    def prior(self, q):
        """(share of arteries, support) at atlas mm points; outside the atlas: 0.5 and 0."""
        g = (q - self.origin) / self.sp
        inside = np.all((g >= 0) & (g <= np.array(self.prob.shape) - 1), axis=1)
        pa = np.where(inside, sample(self.prob, self.sp, self.origin, q), 0.5)
        su = np.where(inside, sample(self.support, self.sp, self.origin, q), 0.0)
        return pa, su

    # file: magic, header length, JSON header, then dist (1/8 mm), prob (1/255) and support (scaled) as uint8
    def save(self, path, **extra):
        scale = float(max(self.support.max(), 1e-6))
        head = dict(shape=list(self.dist.shape), sp=self.sp.tolist(), origin=self.origin.tolist(),
                    centre=self.centre.tolist(), support_scale=scale, **extra)
        hb = json.dumps(head).encode()
        body = [np.clip(np.round(self.dist * 8), 0, 255).astype(np.uint8),
                np.clip(np.round(self.prob * 255), 0, 255).astype(np.uint8),
                np.clip(np.round(self.support / scale * 255), 0, 255).astype(np.uint8)]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(_MAGIC + np.uint32(len(hb)).tobytes() + hb + b"".join(b.tobytes() for b in body))

    @classmethod
    def load(cls, path):
        raw = open(path, "rb").read()
        if raw[:4] != _MAGIC:
            raise ValueError("not a vessel atlas")
        n = int(np.frombuffer(raw[4:8], np.uint32)[0])
        head = json.loads(raw[8:8 + n])
        shape = tuple(head["shape"])
        size = int(np.prod(shape))
        arr = [np.frombuffer(raw, np.uint8, size, 8 + n + i * size).reshape(shape).astype(np.float32) for i in range(3)]
        return cls(arr[0] / 8, arr[1] / 255, arr[2] / 255 * head["support_scale"], head["sp"], head["origin"],
                   head["centre"])


_default = None


def default_atlas():
    """The atlas shipped with the app (assets/vessel_atlas.bin), or None when it is missing."""
    global _default
    if _default is None:
        base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(base, "assets", "vessel_atlas.bin")
        _default = VesselAtlas.load(path) if os.path.exists(path) else False
    return _default or None


def _own_llr(f, sure_a, sure_v):
    """log N(f | arteries) - log N(f | veins) per column of f, each class a Gaussian (median, MAD) of this patient's cells
    where the atlas is sure. A column whose two classes overlap (d' < 1: a late scan's HU) adds nothing."""
    out = np.zeros(len(f))
    if sure_a.sum() < 30 or sure_v.sum() < 30:
        return out
    for c in range(f.shape[1]):
        fa, fv = f[sure_a, c], f[sure_v, c]
        ma, mv = np.median(fa), np.median(fv)
        sa = max(1.4826 * np.median(np.abs(fa - ma)), 1e-3)
        sv = max(1.4826 * np.median(np.abs(fv - mv)), 1e-3)
        if abs(ma - mv) < np.sqrt(0.5 * (sa * sa + sv * sv)):
            continue
        la = -0.5 * ((f[:, c] - ma) / sa) ** 2 - np.log(sa)
        lv = -0.5 * ((f[:, c] - mv) / sv) ** 2 - np.log(sv)
        out += np.clip(la - lv, -LLR_CLIP, LLR_CLIP)
    return out


def _pool_along(L, vessel, iters=8):
    """Each vessel cell's evidence pulled half-way to the mean of its vessel neighbours, a few times: a vessel takes one
    label along its course, where single cells were unsure. Within the vessels' bounding box (a tenth of the grid)."""
    idx = np.argwhere(vessel)
    if not len(idx):
        return L
    lo, hi = np.maximum(idx.min(0) - 1, 0), idx.max(0) + 2
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    ves = np.pad(vessel[box], 1)
    own = np.pad(np.where(vessel[box], L[box], 0.0).astype(np.float32), 1)
    n = np.zeros(ves.shape, np.float32)
    for ax in range(3):
        for s in (1, -1):
            n += np.roll(ves, s, ax)
    out = own.copy()
    for _ in range(iters):
        acc = np.zeros(ves.shape, np.float32)
        for ax in range(3):
            for s in (1, -1):
                acc += np.roll(out, s, ax)
        out = np.where(ves, 0.5 * own + 0.5 * acc / np.maximum(n, 1), 0.0).astype(np.float32)
    res = np.zeros(L.shape, np.float32)
    res[box] = out[1:-1, 1:-1, 1:-1]
    return res


def plausible(M):
    """Whether a registration's size change is one between heads: overall 0.7-1.45 (an adult to a small child) and no
    axis stretched past 0.6-1.7. A skull-like shape that is no head (a tube of bone) can fit the skull surface too."""
    size = abs(np.linalg.det(M)) ** (1 / 3)
    cols = np.linalg.norm(M, axis=0)
    return 0.7 <= size <= 1.45 and bool(np.all((cols >= 0.6) & (cols <= 1.7)))


def classify_head(vessel, feats, bone, sp, atlas=None):
    """Arteries and veins among the vessel cells of the head, by the atlas and this patient's own features.
    vessel: bool grid; feats: callable(idx) -> (n, k) features of those cells (HU of one series; with two phases the
    share of enhancement kept and the change between them); bone: the scan's bone on the same grid; sp: its spacing (mm,
    z y x). Returns (arteries, veins, decided, prior) or None when there is no atlas or the skull does not fit it;
    prior(mask) = the atlas share of arteries at the cells of any mask of this grid (0.5 where it knows nothing), for
    phases.vessel_trees."""
    atlas = atlas or default_atlas()
    if atlas is None:
        return None
    T, fit = atlas.register(bone, sp)
    if T is None or fit < MIN_FIT or not plausible(T.M):
        return None
    head = np.zeros(vessel.shape, bool)
    head[max(0, vertex(bone) - int(ATLAS_DEPTH_MM / sp[0])):] = True
    cells = vessel & head
    idx = np.argwhere(cells)
    if not len(idx):
        return None
    pa, su = atlas.prior(T(idx * np.asarray(sp, float)))
    pa = np.where(su > SUPPORT_MIN, np.clip(pa, 0.02, 0.98), 0.5)
    L = np.log(pa / (1 - pa))
    if feats is not None:
        L = L + _own_llr(np.asarray(feats(idx), float).reshape(len(idx), -1), pa > SURE, pa < 1 - SURE)
    full = np.zeros(vessel.shape)
    full[tuple(idx.T)] = L
    full = _pool_along(full, cells)

    def prior(mask):
        out = np.full(mask.shape, 0.5, np.float32)
        at = np.argwhere(mask)
        if len(at):
            p, s = atlas.prior(T(at * np.asarray(sp, float)))
            out[tuple(at.T)] = np.where(s > SUPPORT_MIN, p, 0.5)
        return out
    return cells & (full > 0), cells & (full <= 0), cells, prior
