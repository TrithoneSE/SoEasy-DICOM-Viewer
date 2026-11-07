"""Transfer function model of the 3D color curves.

A transfer function is a set of *curves*. Each curve is a polyline of points (value in HU, opacity 0..1), every
point with its own color; outside its first / last point a curve is 0. Curves are merged: opacity = the maximum,
color = the opacity-weighted mean. The editor's opacity axis is a square root (opacity = y²); in 3D the color is
interpolated across the gaps between curves instead of going black (fill_black=False), and the default lighting
is ambient 0.15 / diffuse 0.9 / specular 0.3 / power 15. A curve with `hide_edges` colors only homogeneous tissue,
not the partial-volume rim of brighter structures (vessel / bone walls pass through the HU range of an enhancing
tumor); it is rendered with a 2D (HU × gradient) transfer function, see VolumeView.apply_tf.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class TFPoint:
    x: float
    y: float
    color: tuple = (1.0, 1.0, 1.0)


def _guess_layer(name):
    n = name.lower()
    if n.startswith("vein") or n.startswith("vena"):
        return "vein"
    return "bone" if n.startswith("tulang") or n.startswith("bone") else "normal"     # "tulang": old presets


@dataclass
class TFCurve:
    name: str = "Curve"
    points: list = field(default_factory=list)
    # "bone" = used inside the bone area (bone color lock); "vein" = vessels that are not arteries
    # (segment.classify_arteries); "normal" = everything else
    layer: str = "normal"
    # True = color only homogeneous tissue (skip the partial-volume rim around brighter structures)
    hide_edges: bool = False

    def sort(self):
        self.points.sort(key=lambda p: p.x)

    def sample(self, xs: np.ndarray):
        px = np.array([p.x for p in self.points], float)
        py = np.array([p.y for p in self.points], float)
        pc = np.array([p.color for p in self.points], float)
        a = np.interp(xs, px, py, left=0.0, right=0.0)
        rgb = np.stack([np.interp(xs, px, pc[:, i]) for i in range(3)], axis=1)
        a[(xs < px[0]) | (xs > px[-1])] = 0.0
        return a, rgb


def combine(samples, n, fill_black=True, keep=None):
    """Merge curve samples: opacity = max, color = opacity-weighted mean. keep: optional (n,) factor applied to
    hide_edges curves (gradient rows of the 2D transfer function)."""
    alpha = np.zeros(n)
    wsum = np.zeros(n)
    rgb = np.zeros((n, 3))
    for a, col, hide in samples:
        if hide and keep is not None:
            a = a * keep
        alpha = np.maximum(alpha, a)
        rgb += col * a[:, None]
        wsum += a
    mask = wsum > 1e-9
    rgb[mask] /= wsum[mask, None]
    if fill_black or not mask.any():
        rgb[~mask] = 0.0
    else:
        idx = np.flatnonzero(mask)
        for k in range(3):
            rgb[:, k] = np.interp(np.arange(n), idx, rgb[idx, k])
    return alpha, np.clip(rgb, 0, 1)


@dataclass
class TransferFunction:
    name: str = "New"
    curves: list = field(default_factory=list)
    shading: bool = True
    ambient: float = 0.15           # default lighting
    diffuse: float = 0.9
    specular: float = 0.3
    specular_power: float = 15.0
    background: tuple = (0.0, 0.0, 0.0)
    edge_threshold: float = 60.0    # gradient (HU, as VTK measures it) above which hide_edges curves stop coloring
    # True = the "vein" curves color the inside of the skull (brain and its vessels) instead of separated veins
    # (venography.py)
    venography: bool = False

    def curve_samples(self, xs, layer=None):
        """[(opacity, rgb, hide_edges)] of each curve at xs; layer None = all curves, 'normal' / 'bone' = that layer."""
        return [(*c.sample(xs), c.hide_edges) for c in self.curves
                if len(c.points) >= 2 and (layer is None or c.layer == layer or c.layer in layer)]

    def sample(self, xmin: float, xmax: float, n: int = 1024, gain: float = 1.0, offset: float = 0.0, layer=None,
               fill_black=True):
        """fill_black: no curve = black (2D slices, MIP / MinIP); False = interpolate the curve colors across
        the gaps (3D VR: low-opacity curve edges keep their hue instead of darkening)."""
        xs = np.linspace(xmin, xmax, n)
        alpha, rgb = combine(self.curve_samples(xs - offset, layer), n, fill_black)
        return xs, np.clip(alpha * gain, 0, 1), rgb

    def copy(self):
        return copy.deepcopy(self)

    # ------------------------------------------------------------ files
    def to_dict(self):
        d = asdict(self)
        return d

    @staticmethod
    def from_dict(d):
        tf = TransferFunction(
            name=d.get("name", "Preset"),
            shading=d.get("shading", True),
            ambient=d.get("ambient", 0.15),
            diffuse=d.get("diffuse", 0.9),
            specular=d.get("specular", 0.3),
            specular_power=d.get("specular_power", 15.0),
            background=tuple(d.get("background", (0, 0, 0))),
            edge_threshold=d.get("edge_threshold", 60.0),
            venography=bool(d.get("venography", False)),
        )
        for c in d.get("curves", []):
            name = c.get("name", "Curve")
            curve = TFCurve(name=name, layer=c.get("layer") or _guess_layer(name),
                            hide_edges=bool(c.get("hide_edges", False)))
            for p in c.get("points", []):
                curve.points.append(TFPoint(float(p["x"]), float(p["y"]), tuple(p.get("color", (1, 1, 1)))))
            curve.sort()
            tf.curves.append(curve)
        return tf

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def load(path):
        with open(path, encoding="utf-8") as f:
            return TransferFunction.from_dict(json.load(f))


def fit_vessel_curves(tf, floor, peak):
    """A copy of an arteries / veins preset with its two vessel curves moved onto one scan's vessels: starting at floor
    (the scan's vessel threshold), a fifth of the way up halfway to peak (the sinuses' median HU), full from peak on. A
    routine contrast head CT's sinuses hold ~100-140 HU (real CTs), where the CTA curves are still nearly clear."""
    out = tf.copy()
    mid = 0.5 * (floor + peak)
    for c in out.curves:
        if c.name not in ("Veins", "Arteries"):
            continue
        lo, hi = c.points[0].color, c.points[-1].color
        end = c.points[-2].x
        c.points = [TFPoint(floor, 0.0, lo), TFPoint(mid, 0.2, lo), TFPoint(max(peak, mid + 10), 0.45, hi),
                    TFPoint(max(end, peak + 20), 0.45, hi), TFPoint(max(end, peak + 20) + 100, 0.0, hi)]
    out.name = tf.name.replace("CT Angio", "Contrast CT") + " (fitted)"
    return out


def fit_bone_curves(tf, start):
    """A copy of tf with the rise of each bone curve moved down to start HU, when it starts higher; the venography
    keeps its own. For a bone area taken from the plain scan (segment.VolumeState.bone_source): all of it is bone, a
    child's thin bone too, which the presets' 380-450 HU left out (real infant CT: holes in the spine and face)."""
    out = tf.copy()
    if getattr(tf, "venography", False):
        return out
    for c in out.curves:
        if c.layer != "bone" or len(c.points) < 3 or c.points[0].x <= start:
            continue
        shift = start - c.points[0].x
        c.points[0].x = start
        c.points[1].x = min(c.points[1].x + shift, c.points[2].x - 1)
    return out


def unchanged_builtin(tf):
    """tf is one of the built-in presets as it comes."""
    d = tf.to_dict()
    return any(p.to_dict() == d for p in builtin_presets())


def _c(name, pts, hide_edges=False, layer=None):
    """pts: list (x, y, (r,g,b) 0..255)."""
    return TFCurve(name, [TFPoint(x, y, tuple(v / 255 for v in col)) for x, y, col in pts], layer or _guess_layer(name),
                   hide_edges)


IVORY = (240, 225, 190)
BONE_HI = (255, 250, 235)
RED = (200, 20, 20)
RED_HI = (255, 70, 60)
VEIN = (40, 70, 200)
VEIN_HI = (80, 130, 255)
SKIN = (225, 160, 120)
MUSCLE = (190, 70, 60)
FAT = (240, 210, 120)
BRAIN = (235, 235, 250)
VENO = (30, 60, 220)
VENO_HI = (70, 120, 255)
SCALP = (230, 200, 70)
BONE_FAINT = (220, 220, 215)


def builtin_presets() -> list[TransferFunction]:
    """Built-in presets (starting points: adjust them, then save your own)."""
    p = []
    p.append(TransferFunction("CT - Bone", [
        _c("Bone", [(180, 0, IVORY), (300, 0.12, IVORY), (650, 0.55, BONE_HI), (3000, 0.85, BONE_HI)]),
    ]))
    p.append(TransferFunction("CT Angio - Vessels + Bone", [
        _c("Vessels", [(130, 0, RED), (190, 0.18, RED), (280, 0.45, RED_HI), (800, 0.45, RED_HI), (1000, 0.0, RED_HI)]),
        _c("Bone", [(380, 0, IVORY), (500, 0.35, IVORY), (900, 0.7, BONE_HI), (3000, 0.9, BONE_HI)]),
    ]))
    # Veins are a layer of their own (segment.classify_arteries: arteries = bright voxels + what connects to them), so
    # the two curves may overlap in HU. Works on an arterial-phase CTA; on a late phase both are equally bright.
    p.append(TransferFunction("CT Angio - Arteries / Veins + Bone", [
        _c("Veins", [(100, 0, VEIN), (140, 0.2, VEIN), (200, 0.45, VEIN_HI), (500, 0.45, VEIN_HI), (600, 0.0, VEIN_HI)]),
        _c("Arteries", [(110, 0, RED), (170, 0.2, RED), (280, 0.45, RED_HI), (450, 0.45, RED_HI), (500, 0.0, RED_HI)]),
        _c("Bone", [(450, 0, IVORY), (600, 0.4, IVORY), (900, 0.7, BONE_HI), (3000, 0.9, BONE_HI)]),
    ]))
    p.append(TransferFunction("CT Angio - Vessels only", [
        _c("Vessels", [(140, 0, RED), (200, 0.25, RED), (320, 0.6, RED_HI), (420, 0.0, RED_HI)]),
    ]))
    p.append(TransferFunction("CT - Skin + Bone", [
        _c("Skin", [(-500, 0, SKIN), (-250, 0.06, SKIN), (-80, 0.12, SKIN), (0, 0.0, SKIN)]),
        _c("Bone", [(200, 0, IVORY), (400, 0.3, IVORY), (900, 0.7, BONE_HI), (3000, 0.85, BONE_HI)]),
    ]))
    p.append(TransferFunction("CT - Muscle + Vessels + Bone", [
        _c("Fat", [(-150, 0, FAT), (-90, 0.015, FAT), (-40, 0.0, FAT)]),
        _c("Muscle/organ", [(10, 0, MUSCLE), (40, 0.04, MUSCLE), (80, 0.06, MUSCLE), (120, 0.0, MUSCLE)]),
        _c("Vessels", [(140, 0, RED), (200, 0.3, RED_HI), (330, 0.45, RED_HI), (400, 0.0, RED_HI)]),
        _c("Bone", [(400, 0, IVORY), (600, 0.5, BONE_HI), (3000, 0.9, BONE_HI)]),
    ]))
    p.append(TransferFunction("CT - Enhancing lesion/tumor (start here)", [
        _c("Soft tissue", [(-20, 0, (150, 150, 170)), (30, 0.02, (150, 150, 170)), (60, 0.0, (150, 150, 170))],
           hide_edges=True),
        _c("Enhancing lesion", [(70, 0, (40, 220, 60)), (95, 0.2, (40, 220, 60)), (140, 0.25, (120, 255, 120)),
                              (170, 0.0, (120, 255, 120))], hide_edges=True),
        _c("Vessels", [(180, 0, RED), (230, 0.35, RED_HI), (380, 0.4, RED_HI), (430, 0.0, RED_HI)]),
        _c("Bone", [(430, 0, IVORY), (650, 0.5, BONE_HI), (3000, 0.9, BONE_HI)]),
    ]))
    p.append(TransferFunction("CT - Lung", [
        _c("Lung", [(-950, 0, (120, 180, 255)), (-850, 0.04, (150, 200, 255)), (-600, 0.08, (200, 220, 255)),
                    (-450, 0.0, (230, 230, 255))]),
        _c("Bone", [(250, 0, IVORY), (500, 0.4, IVORY), (3000, 0.85, BONE_HI)]),
    ]))
    # A contrast head CT's veins over a faint brain, skull and skin (venography.py): the "vein" layer is the inside of
    # the skull here, so brain and vessels have curves of their own; the scalp and the face keep the normal layer.
    p.append(TransferFunction("CT Venography (contrast CT)", [
        _c("Brain", [(10, 0, BRAIN), (25, 0.004, BRAIN), (40, 0.006, BRAIN), (55, 0.0, BRAIN)], layer="vein"),
        _c("Veins", [(60, 0, VENO), (80, 0.35, VENO), (110, 0.85, VENO_HI), (600, 0.9, VENO_HI), (700, 0.0, VENO_HI)]),
        # from 70 HU: what is no vessel there is stored at 69 (venography.encode)
        _c("Vessels outside the skull", [(70, 0, VENO), (90, 0.35, VENO), (120, 0.85, VENO_HI), (600, 0.9, VENO_HI),
                                         (700, 0.0, VENO_HI)], layer="normal"),
        _c("Skin", [(-600, 0, SCALP), (-400, 0.03, SCALP), (-200, 0.03, SCALP), (-120, 0.0, SCALP)]),
        _c("Bone", [(300, 0, BONE_FAINT), (600, 0.02, BONE_FAINT), (1200, 0.035, BONE_FAINT), (3000, 0.05, BONE_FAINT)]),
    ], venography=True))
    return p


def generic_preset(vmin, vmax) -> TransferFunction:
    """MR / modalities without HU: one curve over the upper part of the data range."""
    r = vmax - vmin
    return TransferFunction("Generic (MR / other)", [
        _c("Curve 1", [(vmin + 0.15 * r, 0, (200, 120, 90)), (vmin + 0.35 * r, 0.25, (240, 190, 140)),
                       (vmin + 0.7 * r, 0.7, (255, 250, 230)), (vmax, 0.9, (255, 255, 255))]),
    ])


def user_preset_dir():
    d = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), "DicomColorViewer", "presets")
    os.makedirs(d, exist_ok=True)
    return d


def list_user_presets():
    d = user_preset_dir()
    out = []
    for n in sorted(os.listdir(d)):
        if n.lower().endswith(".json"):
            try:
                out.append((os.path.join(d, n), TransferFunction.load(os.path.join(d, n))))
            except Exception:
                pass
    return out
