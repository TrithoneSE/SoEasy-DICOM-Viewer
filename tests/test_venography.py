"""Venography preset on a contrast head CT: the skull's cavity with its gaps closed, a sinus against the skull under an
open fontanelle in the cavity's domain, bone mirrored, a vessel outside the skull kept and the bone edge there hidden;
a plain scan that moved is not used; the 3D copy of the VolumeState holds the domains."""
import sys

app_dir, data_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, app_dir)

import numpy as np  # noqa: E402

from dcv import venography  # noqa: E402
from dcv.loader import Volume  # noqa: E402
from dcv.segment import BONE_OFFSET, VEIN_OFFSET, VolumeState  # noqa: E402
from dcv.tf_model import builtin_presets  # noqa: E402

errors = 0


def check(ok, text):
    global errors
    if not ok:
        errors += 1
    print(("ok   " if ok else "FAIL ") + text)


def head(contrast):
    """Skull (1500 HU) with an open fontanelle at the top, brain 35, scalp 40 outside, a sinus along the inner table
    under the fontanelle and a vessel outside the skull (160 HU with contrast, 45 without)."""
    vol = np.full((140, 120, 120), -1000, np.int16)
    z, y, x = np.mgrid[0:140, 0:120, 0:120]
    r = np.sqrt((y - 60) ** 2 + (x - 60) ** 2 + ((z - 60) * 1.0) ** 2)
    vol[r < 56] = 40                                         # scalp
    vol[r < 52] = 1500                                       # skull
    vol[r < 48] = 35                                         # brain
    gap = (z > 100) & (np.abs(x - 60) < 9) & (y < 60)        # the fontanelle: no bone there
    vol[(r >= 48) & (r < 52) & gap] = 40
    sinus = (r >= 44) & (r < 48) & (np.abs(x - 60) < 3) & (z > 70)
    outside = (np.hypot(y - 60, x - 60) < 3) & (z < 6)
    vol[sinus] = 160 if contrast else 45
    vol[(np.hypot(y - 60, x - 60) < 12) & (z < 14) & (vol < 0)] = 40      # the neck, around the vessel outside
    vol[outside] = 160 if contrast else 45
    return vol, sinus, outside


sp = (1.0, 1.0, 1.0)
cm, sinus, outside = head(True)
plain, _, _ = head(False)
vol = Volume(data=cm, spacing=sp, origin=(0, 0, 0), modality="CT", description="Head + CM", frame_uid="f")

# the cavity on the mask grid (this phantom is small: every voxel), its gap closed
from dcv.phases import _pool, coarse_steps  # noqa: E402
steps = coarse_steps(sp, cm.shape)
mean, top = _pool(cm, steps)
cav = venography.cavity(top, tuple(s * k for s, k in zip(sp, steps)))
fontanelle_sinus = sinus & (np.mgrid[0:140, 0:120, 0:120][0] > 100)
check(cav[sinus].mean() > 0.9, f"the sinus is inside the cavity ({cav[sinus].mean():.0%}), also under the open fontanelle "
      f"({cav[fontanelle_sinus].mean():.0%})")
check(not cav[outside].any(), "the vessel outside the skull is not")

# the domains with the plain scan
d = cm.copy()
area = np.zeros(cm.shape, bool)
z, y, x = np.mgrid[0:140, 0:120, 0:120]
r = np.sqrt((y - 60) ** 2 + (x - 60) ** 2 + (z - 60) ** 2)
area[(r >= 46.5) & (r < 53.5)] = True                        # the bone colour lock's area: bone and a rim
venography.encode(d, cm, area, cav, plain)
inner_sinus = sinus & (d >= VEIN_OFFSET + 60)
# the voxel touching the skull has bone beside it on the plain scan too, so it stays bone: 1 of the 4 here
check(inner_sinus.sum() >= 0.65 * sinus.sum(), f"the sinus against the skull is in the cavity's domain "
      f"({inner_sinus.sum() / sinus.sum():.0%}), not hidden in the bone area")
skull = cm >= 1000
check((d[skull] < -2048).mean() > 0.95, "the skull is bone-coded")
check((d[outside] == 160).all(), "the vessel outside the skull keeps its HU (the plain scan says contrast)")
brain = (r < 40)
check(((d[brain] >= VEIN_OFFSET) & (d[brain] < VEIN_OFFSET + 60)).mean() > 0.99, "the brain is in the cavity's domain")

# no plain scan: still the sinus away from the bone
d2 = cm.copy()
venography.encode(d2, cm, area, cav, None)
check((sinus & (d2 >= VEIN_OFFSET + 60)).sum() >= 0.3 * sinus.sum(), "without a plain scan the sinus still shows")

# a plain scan that moved does not line up
moved = np.roll(plain, (6, 5, 0), (0, 1, 2))
check(venography.lines_up(plain, cm) >= venography.MATCH_MIN, f"the same position lines up ({venography.lines_up(plain, cm):.0%})")
check(venography.lines_up(moved, cm) < venography.MATCH_MIN, f"a moved plain scan does not ({venography.lines_up(moved, cm):.0%})")

# a patient who moved: the plain scan's geometry turned by 10 degrees and shifted, so its bone lands elsewhere in patient
# coordinates; register() finds the motion and the plain scan lines up again (a bone ridge makes the skull asymmetric)
cm_r, plain_r = cm.copy(), plain.copy()
for a in (cm_r, plain_r):
    a[20:45, 100:110, 35:60] = 1500
    a[90:100, 15:22, 80:95] = 1500
vol_r = Volume(data=cm_r, spacing=sp, origin=(0, 0, 0), modality="CT", description="Head + CM", frame_uid="f")
ang = np.radians(10.0)
turn = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
moved_vol = Volume(data=plain_r, spacing=sp, origin=(6.0, -4.0, 5.0), modality="CT", description="Head", frame_uid="f",
                   direction=turn)
before = venography.lines_up(venography.plain_on_grid(moved_vol, vol_r), cm_r)
motion = venography.register(moved_vol, vol_r)
after = venography.lines_up(venography.plain_on_grid(moved_vol, vol_r, motion), cm_r)
check(before < venography.MATCH_MIN <= after, f"a moved plain scan is lined up by register(): {before:.0%} -> {after:.0%} of the bone")

# a newborn's skull plate (2 mm, 250 HU) is a sheet, a vein of the same HU (4 mm across) is not
g = np.full((40, 60, 60), 30.0, np.float32)
g[20:22, 5:55, 5:55] = 250
zz, yy, xx = np.mgrid[0:40, 0:60, 0:60]
tube = np.hypot(zz - 30, yy - 45) < 2
g[tube] = 250
sh = venography.sheets(g, (1.0, 1.0, 1.0))
check(sh[20:22, 15:45, 15:45].mean() > 0.8 and sh[tube].mean() < 0.1,
      f"the plate is a sheet ({sh[20:22, 15:45, 15:45].mean():.0%}), the vein beside it not ({sh[tube].mean():.0%})")

# through the VolumeState: preset, plain scan from its loader, the copy holds the domains
st = VolumeState(vol)
st.render_factor = 1
st.plain_loader = lambda: Volume(data=plain, spacing=sp, origin=(0, 0, 0), modality="CT", description="Head",
                                 frame_uid="f")
st.tf = next(p for p in builtin_presets() if p.venography).copy()
st.encode_bone, st.encode_veins = True, True
st.update_render_data()
rd = st.render_data
check(st.venography and st.veno_built, "the state knows it is in venography")
check(st.plain_data() is not None and st.plain_match > 0.9, "the plain scan was read and used")
check((rd[sinus] >= VEIN_OFFSET + 60).mean() > 0.6, "its 3D copy shows the sinus in the cavity's domain")
check((rd[skull] <= -BONE_OFFSET).mean() > 0.9, "and the skull bone-coded")
check(np.array_equal(st.vol.data, cm), "the series itself is untouched")

print(f"errors: {errors}")
sys.exit(1 if errors else 0)
