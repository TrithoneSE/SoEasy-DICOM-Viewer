"""Where the bone lock takes its bone from (VolumeState.bone_source): with the study's plain scan, thin young bone (300 HU)
is bone and a weak contrast vessel against it (200 HU, 40 before) is not, noise on dense bone makes no hole, and what did
not get brighter takes no vessel colour; without a plain scan a routine contrast CT keeps a bright piece without a
cortical core as bone, a CTA does not (an aneurysm); bone curves start lower with the plain scan's bone area."""
import sys

app_dir, data_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, app_dir)

import numpy as np  # noqa: E402

from dcv.loader import Volume  # noqa: E402
from dcv.segment import BONE_CURVE_FROM_HU, BONE_OFFSET, NOT_VESSEL_HU, VolumeState  # noqa: E402
from dcv.tf_model import builtin_presets, fit_bone_curves, unchanged_builtin  # noqa: E402

errors = 0


def check(ok, text):
    global errors
    if not ok:
        errors += 1
    print(("ok   " if ok else "FAIL ") + text)


def scan(contrast, seed):
    """Soft tissue 40 with a dense bone block (1200), a thin young plate (300) beside it, a vessel along the plate (200
    with contrast, 40 without), a bright piece without a cortical core (600, the same on both: young bone, or an
    aneurysm on a CTA), and a patch of tissue at 140 HU on both (it does not enhance)."""
    rng = np.random.default_rng(seed)
    vol = (40 + rng.normal(0, 8, (60, 80, 80))).astype(np.int16)
    vol[10:50, 10:30, 10:30] = 1200 + rng.normal(0, 60, (40, 20, 20)).astype(np.int16)    # noisy dense bone
    vol[10:50, 10:30, 34:37] = 300                                     # the thin plate
    vol[10:50, 14:17, 38:41] = 200 if contrast else 40                # the vessel, one voxel from the plate
    vol[20:40, 50:62, 50:62] = 600                                     # bright, no cortical core
    vol[20:40, 50:60, 10:20] = 140                                     # does not enhance
    return vol


sp = (1.0, 1.0, 1.0)
cm, before = scan(True, 1), scan(False, 2)
plain = Volume(data=before, spacing=sp, origin=(0, 0, 0), modality="CT", description="Head", frame_uid="f")
plate = (slice(10, 50), slice(10, 30), slice(34, 37))
vessel = (slice(10, 50), slice(14, 17), slice(38, 41))
blob = (slice(20, 40), slice(50, 62), slice(50, 62))
quiet = (slice(22, 38), slice(52, 58), slice(12, 18))

# with the plain scan
st = VolumeState(Volume(data=cm, spacing=sp, origin=(0, 0, 0), modality="CT", description="Head + CM", frame_uid="f"))
st.plain_loader = lambda: plain
st.render_factor = 1
check(st.bone_source == "plain", f"with the plain scan the bone comes from it ({st.bone_source}, match {st.plain_match:.2f})")
br = st.bone_region()
check(br[plate].mean() > 0.99, f"the thin plate (300 HU) is bone ({br[plate].mean():.0%})")
check(br[vessel].mean() < 0.05, f"the vessel against it (200 HU, 40 before) is not ({br[vessel].mean():.0%})")
dense = (slice(12, 48), slice(12, 28), slice(12, 28))
check(br[dense].all(), "noise on dense bone makes no hole in it")
check(br[blob].mean() > 0.99, "the bright piece that was there before is bone")
st.tf = next(p for p in builtin_presets() if p.name == "CT Angio - Vessels + Bone").copy()
st.encode_bone, st.encode_veins = True, False
st.update_render_data()
d = st.render_data
check((d[plate] <= -BONE_OFFSET).all(), "the plate is stored in the bone domain")
check(d[quiet].max() <= NOT_VESSEL_HU, f"tissue that did not enhance is stored under every vessel curve (max {d[quiet].max()})")
check((d[vessel][1:-1, 1, 1] >= 190).all(), "the vessel keeps its HU")
fitted = st.fit_bone_curves(st.tf)
bone = next(c for c in fitted.curves if c.layer == "bone")
check(bone.points[0].x == BONE_CURVE_FROM_HU and bone.points[1].x < bone.points[2].x,
      f"bone curves start at {BONE_CURVE_FROM_HU:.0f} HU with the plain scan's bone ({[p.x for p in bone.points]})")
check(unchanged_builtin(st.tf) and not unchanged_builtin(fitted), "a fitted preset is no longer the built-in one")
check("plain scan" in st.bone_note(), "the panel says the bone is from the plain scan")

# without it: a routine contrast CT, then a CTA
st = VolumeState(Volume(data=cm, spacing=sp, origin=(0, 0, 0), modality="CT", description="Head + CM", frame_uid="f"))
check(st.bone_source == "ct", f"no plain scan, no CTA: bone from the threshold ({st.bone_source})")
br = st.bone_region()
check(br[blob].mean() > 0.99, f"a bright piece without a cortical core stays bone ({br[blob].mean():.0%})")
check(st.fit_bone_curves(st.tf) is st.tf, "the curves stay as they are")
st = VolumeState(Volume(data=cm, spacing=sp, origin=(0, 0, 0), modality="CT", description="CTA ARTERI", frame_uid="f"))
check(st.bone_source == "cta", f"a CTA ({st.bone_source})")
br = st.bone_region()
check(br[blob].mean() < 0.05, f"on a CTA it is a vessel (an aneurysm), out of the bone area ({br[blob].mean():.0%})")

# a newborn's skull without a plain scan: no CTA rule, bone curves from THIN_BONE_HU
from dcv.segment import THIN_BONE_HU  # noqa: E402
st = VolumeState(Volume(data=cm, spacing=sp, origin=(0, 0, 0), modality="CT", description="AX THIN ARTERY", frame_uid="f"))
st._thin = True                                            # as thin_skull finds on a newborn
check(st.bone_source == "thin", f"a newborn's skull: its own source ({st.bone_source}), even on an arterial series")
st.thin_plates = lambda: None                              # this phantom has no skull plates to find
br = st.bone_region()
check(br[blob].mean() > 0.99, f"its bright bone without a cortical core stays bone ({br[blob].mean():.0%})")
st.tf = next(p for p in builtin_presets() if p.name == "CT Angio - Vessels + Bone").copy()
bone = next(c for c in st.fit_bone_curves(st.tf).curves if c.layer == "bone")
check(bone.points[0].x == THIN_BONE_HU, f"bone curves from {THIN_BONE_HU:.0f} HU, so its plates show")
check("newborn" in st.bone_note(), "the panel says so")

# a vessel lying right on bone (an aneurysm's ICA on the clinoid): the plain scan beside it holds the bone, yet its edge
# stays a vessel, while bone stays bone
t = np.full((30, 30, 30), 40, np.int16)
t[5:25, 5:25, 10:13] = 900                                   # a plate of cortex
t[5:25, 10:14, 13:17] = 220                                  # a vessel on it (x 13-16, the plate ends at 12)
tp = t.copy()
tp[5:25, 10:14, 13:17] = 40
st = VolumeState(Volume(data=t, spacing=sp, origin=(0, 0, 0), modality="CT", description="CTA + CM", frame_uid="f"))
st.plain_loader = lambda: Volume(data=tp, spacing=sp, origin=(0, 0, 0), modality="CT", description="PRE", frame_uid="f")
br = st.bone_region()
check(st.bone_source == "plain" and not br[6:24, 11:13, 13].any(), "a vessel on bone keeps its edge against the bone (plain scan)")
check(br[6:24, 6:24, 10:12].all(), "  and the plate stays bone")

# slab by slab: a plain scan whose lower part moved (the neck bent between the scans)
from dcv import venography  # noqa: E402
col = np.full((100, 40, 40), 40, np.int16)
for zc in range(5, 100, 10):
    col[zc:zc + 5, 10:30, 18:20] = 900                       # thin plates of bone up a column (as cortex is)
for k, (shift, expect) in enumerate(((3, True), (10, False))):
    pl = col.copy()
    pl[:40] = np.roll(col[:40], shift, axis=2)              # the lowest slab (40 planes of 1 mm) moved along x
    q, ok = venography.slab_plain(pl, col, (1.0, 1.0, 1.0), (1, 1, 1))
    check(ok[40:].all(), f"slab plain, moved {shift} mm below: the planes above line up as they are")
    check(bool(ok[:40].all()) == expect, f"  the moved slab {'is found by its shift' if expect else 'is beyond reach: not used'}")
    if expect:
        from dcv.segment import box_max
        here = col[:40] >= 500
        check(np.mean((box_max(q[:40], (0, 1, 1)) >= 500)[here]) == 1.0, "  and its bone lies where the series has it (within a voxel)")

# fit_bone_curves on every built-in preset: rising, the venography's own
for p in builtin_presets():
    f = fit_bone_curves(p, BONE_CURVE_FROM_HU)
    for c in f.curves:
        xs = [q.x for q in c.points]
        if xs != sorted(xs):
            check(False, f"{p.name} / {c.name}: points out of order {xs}")
ven = next(p for p in builtin_presets() if p.venography)
check(fit_bone_curves(ven, BONE_CURVE_FROM_HU).to_dict() == ven.to_dict(), "the venography keeps its bone curve")

print(f"errors: {errors}")
sys.exit(1 if errors else 0)
