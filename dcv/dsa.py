"""Digital subtraction angiography (DSA) for XA / fluoroscopy runs: subtraction on / off, the displayed frame as
the mask, reset, manual pixel-shift (drag, 1 px / 0.1 px keys) and auto pixel-shift in a square around the click.

The maths follows DICOM: a mask given by the scanner (Mask Subtraction Sequence (0028,6100), AVG_SUB: mean of the
mask frames, Mask Sub-pixel Shift) is used when present, else frame 2. Pixel Intensity Relationship
(0028,1040) LOG / DISP data are already logarithmic, so live − mask is the attenuation of the contrast; LIN data are
log-transformed first (the plain difference of linear intensities would depend on the bone behind the vessel).
"""

from __future__ import annotations

import numpy as np

LOG_SCALE = 1000.0          # LIN data: log difference × this, so a window of a few hundred shows the vessels
# Default DSA window: WW = 0.6 × the run's own window, "no change" at 64 % grey, so vessels are a firm mid grey
# on a light background instead of pitch black (tuned on real Siemens runs; adjust with the W/L tool).
DSA_WW_FACTOR = 0.6
DSA_BACKGROUND = 0.64


def shift_image(img, dy, dx):
    """img moved by (dy, dx) pixels (rows down, columns right), bilinear for sub-pixel shifts, edges repeated."""
    h, w = img.shape
    iy, ix = int(np.floor(dy)), int(np.floor(dx))
    fy, fx = dy - iy, dx - ix
    pad = max(abs(iy), abs(ix)) + 2
    p = np.pad(img.astype(np.float32, copy=False), pad, mode="edge")

    def take(oy, ox):      # out[r, c] = img[r - oy, c - ox]
        return p[pad - oy:pad - oy + h, pad - ox:pad - ox + w]

    out = take(iy, ix) * ((1 - fy) * (1 - fx))
    if fx:
        out += take(iy, ix + 1) * ((1 - fy) * fx)
    if fy:
        out += take(iy + 1, ix) * (fy * (1 - fx))
    if fx and fy:
        out += take(iy + 1, ix + 1) * (fy * fx)
    return out


def hint_from_dataset(ds, n_frames):
    """What the DICOM header says about subtraction: {'mask': [0-based frames] or None, 'shift': (dy, dx),
    'log': bool (data are linear -> subtract logarithms), 'recommended': Recommended Viewing Mode is SUB}."""
    mask, shift = None, (0.0, 0.0)
    for item in ds.get("MaskSubtractionSequence", []) or []:
        if str(item.get("MaskOperation", "")).upper() != "AVG_SUB":
            continue                                    # TID (time interval differencing) is not supported
        frames = item.get("MaskFrameNumbers")             # 1-based; one number or several (averaged)
        if frames is not None:
            frames = [int(f) - 1 for f in np.atleast_1d(np.asarray(frames, dtype=float))]
            frames = [f for f in frames if 0 <= f < n_frames]
            if frames:
                mask = frames
        sp = item.get("MaskSubPixelShift")
        if sp is not None and len(sp) == 2:
            shift = (float(sp[0]), float(sp[1]))          # row offset, column offset of the mask
        break
    rel = str(ds.get("PixelIntensityRelationship", "")).upper()
    return {"mask": mask, "shift": shift, "log": rel == "LIN",
            "recommended": str(ds.get("RecommendedViewingMode", "")).upper() == "SUB"}


class DSA:
    """Subtraction settings of one viewport: mask frames, mask shift, log transform. Frames are vol.data[k]."""

    def __init__(self, vol):
        self.vol = vol
        n = vol.shape[0]
        hint = getattr(vol, "dsa_hint", None) or {}
        self.default_mask = hint.get("mask") or [min(1, n - 1)]       # the second frame
        self.default_shift = tuple(hint.get("shift", (0.0, 0.0)))
        self.log = bool(hint.get("log", False))
        self.mask = list(self.default_mask)
        self.shift = self.default_shift
        self._cache = None

    # -------------------------------------------------------------- mask
    def set_mask(self, frames):
        self.mask = [int(f) for f in frames]
        self._cache = None

    def reset(self):
        """Back to the default mask and its original shift."""
        self.mask = list(self.default_mask)
        self.shift = self.default_shift
        self._cache = None

    def move(self, dy, dx):
        self.shift = (round(self.shift[0] + dy, 3), round(self.shift[1] + dx, 3))

    def _frame(self, k):
        f = getattr(self.vol, "frame", None)        # Volume.frame waits for frames still being decoded
        return f(k) if f is not None else self.vol.data[k]

    def _prep(self, img):
        img = img.astype(np.float32)
        return np.log(np.maximum(img, 0) + 1.0) * LOG_SCALE if self.log else img

    def mask_image(self):
        key = (tuple(self.mask), self.shift)
        if self._cache is None or self._cache[0] != key:
            m = np.mean([self._prep(self._frame(f)) for f in self.mask], axis=0)
            self._cache = (key, shift_image(m, *self.shift) if any(self.shift) else m)
        return self._cache[1]

    def subtract(self, k):
        """Frame k minus the (shifted) mask: about 0 for background, negative where contrast attenuates."""
        return self._prep(self._frame(k)) - self.mask_image()

    def suggest_window(self, native_ww, current=None):
        """(ww, wl) of the subtracted run: DSA_WW_FACTOR × the window the run was shown with, "no change" at
        DSA_BACKGROUND grey. Log-transformed (LIN) data have other units: 5 × the 99th percentile of |live − mask|
        in a few frames (decoded ones plus the one on screen)."""
        if not self.log and native_ww:
            ww = DSA_WW_FACTOR * float(native_ww)
            return ww, -(DSA_BACKGROUND - 0.5) * ww
        n = self.vol.shape[0]
        loader = getattr(self.vol, "loader", None)
        ks = {int(k) for k in np.linspace(0, n - 1, min(n, 8))}
        if loader is not None:
            ks = {k for k in ks if loader.ready[k]}
        ks = sorted(ks | ({int(current)} if current is not None else set()) or {n - 1})
        ww = max(5 * max(float(np.percentile(np.abs(self.subtract(k)), 99)) for k in ks), 10.0)
        return ww, -(DSA_BACKGROUND - 0.5) * ww

    def label(self):
        m = "+".join(str(f + 1) for f in self.mask)
        return f"Mask: {m}", f"Shift: x {self.shift[1]:+.1f}  y {self.shift[0]:+.1f} px"

    # ------------------------------------------------------- auto pixel-shift
    def auto_shift(self, k, center, half, search=12):
        """Shift of the mask that best removes the background in the square (row, col centre, half side) of frame
        k: least total variation of live − mask there (misaligned bone edges leave strong edges; the contrast in the
        vessels is the same for every shift). Integer search ±`search` px around the current shift, then 0.1 px."""
        live = self._prep(self._frame(k))
        base = np.mean([self._prep(self._frame(f)) for f in self.mask], axis=0)
        h, w = live.shape
        r, c = int(round(center[0])), int(round(center[1]))
        half = int(min(half, h // 2 - 1, w // 2 - 1))
        r, c = int(np.clip(r, half, h - half - 1)), int(np.clip(c, half, w - half - 1))
        m = half + search + 3
        r0, r1, c0, c1 = max(0, r - m), min(h, r + m + 1), max(0, c - m), min(w, c + m + 1)
        live_c, base_c = live[r0:r1, c0:c1], base[r0:r1, c0:c1]
        ys, xs = slice(r - half - r0, r + half + 1 - r0), slice(c - half - c0, c + half + 1 - c0)

        def cost(dy, dx):
            d = (live_c - shift_image(base_c, dy, dx))[ys, xs]
            return float(np.abs(np.diff(d, axis=0)).sum() + np.abs(np.diff(d, axis=1)).sum())

        y0, x0 = round(self.shift[0]), round(self.shift[1])
        best = min(((cost(y0 + dy, x0 + dx), y0 + dy, x0 + dx)
                    for dy in range(-search, search + 1) for dx in range(-search, search + 1)))
        _, by, bx = best
        fine = min(((cost(by + dy / 10, bx + dx / 10), by + dy / 10, bx + dx / 10)
                    for dy in range(-9, 10) for dx in range(-9, 10)))
        self.shift = (round(fine[1], 2), round(fine[2], 2))
        return self.shift
