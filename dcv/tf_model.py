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
    return "bone" if n.startswith("tulang") or n.startswith("bone") else "normal"     # "tulang": old presets


@dataclass
class TFCurve:
    name: str = "Curve"
    points: list = field(default_factory=list)
    # "bone" = used inside the bone area (bone color lock); "normal" = outside it
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

    def curve_samples(self, xs, layer=None):
        """[(opacity, rgb, hide_edges)] of each curve at xs; layer None = all curves, 'normal' / 'bone' = that layer."""
        return [(*c.sample(xs), c.hide_edges) for c in self.curves
                if len(c.points) >= 2 and (layer is None or c.layer == layer)]

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


def _c(name, pts, hide_edges=False):
    """pts: list (x, y, (r,g,b) 0..255)."""
    return TFCurve(name, [TFPoint(x, y, tuple(v / 255 for v in col)) for x, y, col in pts], _guess_layer(name),
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


def builtin_presets() -> list[TransferFunction]:
    """Built-in presets (starting points: adjust them, then save your own)."""
    p = []
    p.append(TransferFunction("CT - Bone", [
        _c("Bone", [(180, 0, IVORY), (300, 0.12, IVORY), (650, 0.55, BONE_HI), (3000, 0.85, BONE_HI)]),
    ]))
    p.append(TransferFunction("CT Angio - Vessels + Bone", [
        _c("Vessels", [(130, 0, RED), (190, 0.18, RED), (280, 0.45, RED_HI), (380, 0.0, RED_HI)]),
        _c("Bone", [(380, 0, IVORY), (500, 0.35, IVORY), (900, 0.7, BONE_HI), (3000, 0.9, BONE_HI)]),
    ]))
    # Arteries vs veins by HU: only separable on an arterial-phase CTA (arteries 250–500 HU, veins still faint).
    # On a delayed / venous-phase scan both sit at ~130–350 HU and the two curves just split one vessel range.
    p.append(TransferFunction("CT Angio - Arteries / Veins + Bone", [
        _c("Veins", [(120, 0, VEIN), (160, 0.12, VEIN), (210, 0.3, VEIN_HI), (250, 0.0, VEIN_HI)]),
        _c("Arteries", [(220, 0, RED), (270, 0.35, RED), (380, 0.45, RED_HI), (450, 0.0, RED_HI)]),
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
