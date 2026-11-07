"""Phases of a CTA: an artery and a vein told apart with both series, also when the sinus is full on the arterial series;
3D window with the pair; a seed tuned in 2D without any 3D, its panel and the 3D window following each other."""
import sys

app_dir, data_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, app_dir)

import numpy as np  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from dcv import phases, segment  # noqa: E402
from dcv.loader import Volume  # noqa: E402
from dcv.seed_range import SeedPanel, grower_of  # noqa: E402
from dcv.segment import Segment, VolumeState, auto_grow  # noqa: E402
from dcv.tf_model import builtin_presets  # noqa: E402

errors = 0


def check(ok, text):
    global errors
    if not ok:
        errors += 1
    print(("ok   " if ok else "FAIL ") + text)


check(phases.phase_of_name("CTA Carotis/COW 0.625mm VENA") == "venous", "VENA = venous")
check(phases.phase_of_name("CTA Carotis/COW 0.625mm ARTER") == "arterial", "ARTER = arterial")
check(phases.phase_of_name("PRE KONTRAS 0.625 mm NK") == "native", "PRE ... NK = native")
check(phases.phase_of_name("Ventricle") is None, "a word that only starts like VEN is no phase")
check(phases.phase_of_name("Brain 0.625 mm  NC") == "native", "NC = native")
check(phases.has_contrast_name("Helical Head 0.625mm + CM") and not phases.has_contrast_name("Helical Head 0.625mm"),
      "'+ CM' says contrast, the plain name does not")


def plain_and_contrast(contrast):
    """A head: skull, brain at 35 HU, and a sinus and arteries inside; with contrast those are bright."""
    vol = np.full((150, 100, 100), -1000, np.int16)
    z, y, x = np.mgrid[0:150, 0:100, 0:100]
    r = np.hypot(np.hypot(y - 50, x - 50), (z - 40) * 0.8)
    vol[r < 48] = 1500
    vol[r < 44] = 35
    for cx in (35, 50, 65):
        vol[(np.hypot(y - 50, x - cx) < 2.5) & (r < 44) & (z > 20)] = 300 if contrast else 45   # > 0.5 % of the inside
    return vol


check(not phases.is_phase_pair(plain_and_contrast(False), plain_and_contrast(True), (1.0, 1.0, 1.0)),
      "a plain series next to its contrast series is no pair of phases")
check(phases.is_phase_pair(plain_and_contrast(True), plain_and_contrast(True), (1.0, 1.0, 1.0)),
      "two contrast series are")


def head(late=False):
    """Skull, a vertical artery, a sinus against the vault. late: the sinus full on the arterial series and falling too."""
    n, ny, nx = 150, 100, 100
    a, v = np.full((n, ny, nx), 40, np.int16), np.full((n, ny, nx), 40, np.int16)
    z, y, x = np.mgrid[0:n, 0:ny, 0:nx]
    r = np.hypot(y - 50, x - 50)
    a[r > 44], v[r > 44] = 1500, 1500
    art = np.hypot(y - 50, x - 30) < 3
    a[art], v[art] = 450, 120
    sinus = (np.hypot(y - 50, x - 88) < 6) & (r <= 44)
    a[sinus], v[sinus] = (350, 170) if late else (70, 260)
    return a, v, art, sinus


for late in (False, True):
    a, v, art, sinus = head(late)
    m = phases.classify_phases(a, v, (1.0, 1.0, 1.0))
    word = "late series" if late else "timely series"
    check(m.arteries[art].mean() > 0.9, f"{word}: the artery is an artery")
    check(m.veins[sinus].mean() > 0.6, f"{word}: the sinus is a vein ({m.veins[sinus].mean():.2f})")
    check(not m.arteries[sinus].any() or m.arteries[sinus].mean() < 0.2, f"{word}: the sinus is not an artery")
    check(not (m.arteries & m.bone).any() and not (m.veins & m.bone).any(), f"{word}: bone is neither")

a, v, art, sinus = head()
moved = np.roll(v, 3, axis=2)
m = phases.classify_phases(a, moved, (1.0, 1.0, 1.0))
check(m.shift == (0, 0, -3), f"a patient who moved 3 mm is found: {m.shift}")
check(m.arteries[art].mean() > 0.9, "and the artery stays an artery")

# bone against the edge of the volume, carried round by the shift, is not taken for a very bright artery
a6, v6, art6, _ = head()
a6[:, :, :4], v6[:, :, :4] = 1500, 1500
moved6 = np.roll(v6, 3, axis=2)
moved6[:, :, :3] = 0
m6 = phases.classify_phases(a6, moved6, (1.0, 1.0, 1.0))
check(not (m6.arteries | m6.veins)[:, :, 97:].any(), "the edge a shift wrapped round holds no vessels")
check(m6.arteries[art6].mean() > 0.9, "and the artery stays an artery")

# a patient who moved: in either window, each vessel lands where that window's own series has it
for own, other, name, here_art, here_sinus in ((moved, a, "VENA", np.roll(art, 3, axis=2), np.roll(sinus, 3, axis=2)),
                                               (a, moved, "ARTER", art, sinus)):
    stm = VolumeState(Volume(data=own, spacing=(1.0, 1.0, 1.0), origin=(0, 0, 0), modality="CT",
                             description=f"CTA {name}", frame_uid="f"))
    stm.partner = Volume(data=other, spacing=(1.0, 1.0, 1.0), origin=(0, 0, 0), modality="CT", description="other", frame_uid="f")
    stm.tf = next(p for p in builtin_presets() if "Veins" in p.name).copy()
    stm.encode_bone, stm.encode_veins = True, True
    stm.update_render_data()
    dm = stm.render_data
    check((dm[here_art] == 450).mean() > 0.9, f"moved, {name} window: the artery is at its own place with its arterial HU")
    check((dm[here_sinus] >= segment.VEIN_OFFSET).mean() > 0.6, f"moved, {name} window: so is the vein")
    check((dm[here_sinus] == segment.VEIN_OFFSET + 260).mean() > 0.5, f"moved, {name} window: with its venous HU")

# a narrow stretch of a sinus, late series: it joins the thick part. A narrow sinus connected to nothing thick is the
# known limit: anatomy cannot find it, and the time course cannot on a late series.
a2, v2, art2, _ = head(True)
z, y, x = np.mgrid[0:150, 0:100, 0:100]
r2 = np.hypot(y - 50, x - 50)
thick = (np.hypot(y - 50, x - 88) < 6) & (r2 <= 44) & (z >= 110)
thin = (np.hypot(y - 50, x - 90) < 3) & (r2 <= 44) & (z >= 70) & (z < 110)
a2[(np.hypot(y - 50, x - 88) < 6) & (r2 <= 44)] = 40
v2[(np.hypot(y - 50, x - 88) < 6) & (r2 <= 44)] = 40
a2[thick | thin], v2[thick | thin] = 350, 170
m2 = phases.classify_phases(a2, v2, (1.0, 1.0, 1.0))
check(m2.veins[thick].mean() > 0.6, "late series: the thick part of the sinus is a vein")
check(m2.veins[thin].mean() > 0.5, f"late series: its narrow stretch joins it ({m2.veins[thin].mean():.2f})")

# a native scan whose inner table falls on the sinus (a head that turned): the sinus must survive, but a thick piece that
# was bright before contrast all along (bone the threshold missed) is no sinus
a3, v3, art3, sinus3 = head(True)
native = np.full(a3.shape, 40, np.float32)
native[np.hypot(*np.mgrid[0:100, 0:100] - 50.0)[None].repeat(150, 0) > 44] = 1500
z3, y3, x3 = np.mgrid[0:150, 0:100, 0:100]
native[(np.hypot(y3 - 50, x3 - 88) < 6) & (x3 >= 92)] = 400         # the table over the outer third of the sinus
m3 = phases.classify_phases(a3, v3, (1.0, 1.0, 1.0), native=native)
check(m3.veins[sinus3].mean() > 0.6, f"misaligned native: the sinus stays a vein ({m3.veins[sinus3].mean():.2f})")
native[sinus3] = 400
m3 = phases.classify_phases(a3, v3, (1.0, 1.0, 1.0), native=native)
check(m3.veins[sinus3].mean() < 0.2, "a piece bright before contrast too is no sinus")

# a thin, weakly filled sinus and an artery near it: too narrow for the anatomy, but it keeps more of its enhancement
a4, v4, art4, _ = head()
z4, y4, x4 = np.mgrid[0:150, 0:100, 0:100]
r4 = np.hypot(y4 - 50, x4 - 50)
a4[(np.hypot(y4 - 50, x4 - 88) < 6) & (r4 <= 44)] = 40
v4[(np.hypot(y4 - 50, x4 - 88) < 6) & (r4 <= 44)] = 40
weak = (np.hypot(y4 - 50, x4 - 90) < 2.2) & (r4 <= 44)
a4[weak], v4[weak] = 210, 125       # keeps (125-40)/(210-40) = 0.50; the artery (120-40)/(450-40) = 0.20
cortical = np.hypot(y4 - 50, x4 - 60) < 1.5
a4[cortical], v4[cortical] = 230, 75
m4 = phases.classify_phases(a4, v4, (1.0, 1.0, 1.0))
check(m4.veins[weak].mean() > 0.6, f"a weak thin sinus is a vein ({m4.veins[weak].mean():.2f})")
check(m4.arteries[art4].mean() > 0.9, "the artery stays an artery")
check(m4.arteries[cortical].mean() > 0.6 or not m4.veins[cortical].any(), "a small artery is not a vein")

# an aneurysm with slow flow, as bright as the arteries, keeps much of its contrast: still an artery
a5, v5, art5, _ = head()
sac = np.hypot(z4 - 40, np.hypot(y4 - 50, x4 - 38)) < 7
a5[sac], v5[sac] = 450, 260
m5 = phases.classify_phases(a5, v5, (1.0, 1.0, 1.0))
check(m5.arteries[sac].mean() > 0.9, f"a slow aneurysm stays an artery ({m5.arteries[sac].mean():.2f})")

# a late scan's neck, where the jugular is as bright as the carotid: on each side the lateral one is the jugular
nk = np.zeros((300, 100, 100), bool)
zk, yk, xk = np.mgrid[0:300, 0:100, 0:100]
neck = (zk >= 70) & (zk < 150)
carotid = neck & ((np.hypot(yk - 50, xk - 33) < 3) | (np.hypot(yk - 50, xk - 67) < 3))
jugular = neck & ((np.hypot(yk - 57, xk - 18) < 5) | (np.hypot(yk - 57, xk - 82) < 5))
nk[carotid | jugular] = True
na, nv = phases._neck_sides(nk, np.hypot(yk - 50, xk - 50) < 45, 299, (1.0, 1.0, 1.0))
check(na[carotid].mean() > 0.9 and not na[jugular].any(), "late neck: the carotids are the arterial seeds")
check(nv[jugular].mean() > 0.9 and not nv[carotid].any(), "late neck: the lateral jugulars are the venous seeds")

# bone that moved between the phases (a jaw): drawn as bone on the window, never as a vein block
at, vt, _, _ = head()
zt, yt, xt = np.mgrid[0:150, 0:100, 0:100]
tooth_a = (zt >= 20) & (zt < 40) & (yt >= 60) & (yt < 70) & (xt >= 45) & (xt < 55)
tooth_v = (zt >= 20) & (zt < 40) & (yt >= 60) & (yt < 70) & (xt >= 48) & (xt < 58)
at[tooth_a], vt[tooth_v] = 1500, 1500
stt = VolumeState(Volume(data=vt, spacing=(1.0, 1.0, 1.0), origin=(0, 0, 0), modality="CT", description="CTA VENA",
                         frame_uid="f"))
stt.partner = Volume(data=at, spacing=(1.0, 1.0, 1.0), origin=(0, 0, 0), modality="CT", description="CTA ARTER", frame_uid="f")
stt.tf = next(p for p in builtin_presets() if "Veins" in p.name).copy()
stt.encode_bone, stt.encode_veins = True, True
stt.update_render_data()
near = (zt >= 18) & (zt < 42) & (yt >= 58) & (yt < 72) & (xt >= 43) & (xt < 60)
check(not (stt.render_data[near] >= segment.VEIN_OFFSET).any(), "a jaw that moved is no vein")
check((stt.render_data[tooth_v] <= -segment.BONE_OFFSET).all(), "and all of it is drawn as bone, without holes")

# the vessel atlas: a skull 5 % larger and 6 mm further back is laid onto the atlas skull, and vessels take the label
# of where they lie (front: arteries, back: veins, in this made-up atlas)
from dcv import atlas as vatlas  # noqa: E402


def skull_dist(z, y, x, c, s):
    """mm from an egg-shaped vault (radii 70 / 85 / 65 mm, scaled s) and the slab of an anterior skull base."""
    r = np.sqrt(((z - c[0]) / (70 * s)) ** 2 + ((y - c[1]) / (85 * s)) ** 2 + ((x - c[2]) / (65 * s)) ** 2)
    out = lambda v, lo, hi: np.maximum(np.maximum(lo - v, 0), v - hi)  # noqa: E731
    slab = np.sqrt(out(z, c[0] - 45 * s, c[0] - 40 * s) ** 2 + out(y, c[1] - 80 * s, c[1] - 10 * s) ** 2
                   + out(x, c[2] - 35 * s, c[2] + 35 * s) ** 2)
    return np.minimum(np.abs(r - 1) * 72 * s, slab)


za, ya, xa = np.mgrid[0:90, 0:110, 0:90] * 2.0
eg = vatlas.VesselAtlas(np.minimum(skull_dist(za, ya, xa, (90, 110, 90), 1.0), 30).astype(np.float32),
                        np.where(ya < 110, 0.95, 0.05).astype(np.float32), np.ones(za.shape, np.float32),
                        (2, 2, 2), (0, 0, 0), (90, 110, 90))
zs, ys, xs = np.mgrid[0:110, 0:130, 0:110] * 1.5
c_scan = (82, 101, 82)
skull = (skull_dist(zs, ys, xs, c_scan, 1.05) < 1.5) & (zs > c_scan[0] - 50)
T_eg, fit_eg = eg.register(skull, (1.5, 1.5, 1.5))
check(fit_eg > 0.8 and abs(np.cbrt(np.linalg.det(T_eg.M)) - 1 / 1.05) < 0.03,
      f"atlas: a larger, shifted skull fits ({fit_eg:.2f}, scale {np.cbrt(np.linalg.det(T_eg.M)):.3f})")
ves = np.zeros(skull.shape, bool)
ves[60:80, 50, 50:60] = True                        # front (y 75 mm, the skull's centre at 101)
ves[60:80, 85, 50:60] = True                        # back (y 128 mm)
res_eg = vatlas.classify_head(ves, None, skull, (1.5, 1.5, 1.5), eg)
check(res_eg is not None and res_eg[0][60:80, 50, 50:60].all() and res_eg[1][60:80, 85, 50:60].all(),
      "atlas: vessels labelled by where they lie")
check(vatlas.default_atlas() is not None, "the app's vessel atlas is there (assets/vessel_atlas.bin)")

# seed: a noisy mass touching a neighbour of the same HU over a narrow neck, on 0.4 mm voxels (real CT: a sinonasal
# tumour against muscle grew to 209 ml from a ~20 ml mass, because the cut was one voxel and fell back to none)
from dcv.segment import region_grow  # noqa: E402

rng = np.random.default_rng(7)
nb = np.full((40, 60, 120), -100, np.int16)
zb, yb, xb = np.mgrid[0:40, 0:60, 0:120]
body = (zb >= 5) & (zb < 35) & (yb >= 10) & (yb < 50) & (xb >= 5) & (xb < 115) \
    & ((xb < 50) | (xb > 70) | ((abs(yb - 30) < 1) & (abs(zb - 20) < 1)))
nb[body] = 45 + rng.integers(-7, 8, int(body.sum()))
spb = (0.625, 0.4, 0.4)
mass_a = 30 * 40 * 45
cutm = region_grow(nb, spb, (20, 30, 25), 25, 70)
check(0.9 * mass_a < cutm[0].sum() < 1.1 * mass_a, f"the mass alone with the cut ({cutm[0].sum() / mass_a:.2f} of it)")
check(region_grow(nb, spb, (20, 30, 25), 25, 70, 0)[0].sum() > 1.9 * mass_a, "without the cut it runs into the neighbour")
ball = (np.hypot(np.hypot(zb - 20, yb - 30), xb - 25) * 0.4 < 6)
nb2 = np.full(nb.shape, 0, np.int16)
nb2[ball] = 100
check(region_grow(nb2, spb, (20, 30, 25), 90, 110)[0].sum() >= 0.98 * ball.sum(), "a round lesion keeps its whole edge")

# the small windows around the click (quick to re-grow) give what the whole block gives, whichever window is needed
import dcv.segment as dseg  # noqa: E402

same = True
for data, seed, lo, hi, leak in ((nb, (20, 30, 25), 25, 70, 1.6), (nb, (20, 30, 25), 25, 70, 0), (nb2, (20, 30, 25), 90, 110, 1.6),
                                 (nb, (20, 30, 100), 30, 60, 1.6)):
    dseg._last_window.clear()
    by_win = region_grow(data, spb, seed, lo, hi, leak)
    keep, dseg.WINDOWS_MM = dseg.WINDOWS_MM, ()
    whole = region_grow(data, spb, seed, lo, hi, leak)
    dseg.WINDOWS_MM = keep
    same &= by_win[1] == whole[1] and np.array_equal(by_win[0], whole[0])
check(same, "seed: growing in windows around the click is exact")

# CT: the automatic range stays out of the brain next to a heterogeneous tumour (two parts, 65 and 95 HU, bright
# vessels): mean − 2.5·SD reaches the brain's ~35 HU (web: segment.test.ts, the same phantom)
n = 120
zz, yy, xx = np.mgrid[0:n, 0:n, 0:n]
in_t = ((zz - 60) ** 2 + (yy - 60) ** 2 + (xx - 60) ** 2) * 0.25 < 144
rng = np.random.default_rng(5)
tum = np.where(in_t, np.where(xx < 60, 65, 95) + np.where((xx * 7 + yy * 3 + zz) % 23 == 0, 120, 0), 35)
tum = (tum + np.where(in_t, rng.integers(-14, 15, tum.shape), rng.integers(-9, 10, tum.shape))).astype(np.int16)
dseg._smooth_cache.clear()
dseg._last_window.clear()
plain, _ = auto_grow(tum, (0.5, 0.5, 0.5), (60, 60, 59))
guarded, (glo, _) = auto_grow(tum, (0.5, 0.5, 0.5), (60, 60, 59), guard=True)
ball = int(in_t.sum())
check(plain[0].sum() > 3 * ball and glo > 45 and 0.6 * ball < guarded[0].sum() < 1.3 * ball,
      f"seed on CT: the range keeps out of the brain ({plain[0].sum() / ball:.1f}× → {guarded[0].sum() / ball:.2f}× the "
      f"tumour, lower bound {glo:.0f} HU)")

app = QApplication([])
from dcv.vr_window import VRWindow  # noqa: E402

vol = Volume(data=v, spacing=(1.0, 1.0, 1.0), origin=(0, 0, 0), modality="CT", description="CTA VENA", frame_uid="f")
st = VolumeState(vol)
check(st.phase == "venous", "the state reads the phase from the name")

# one venous series alone: the sinus is traced by anatomy, the washed-out artery is no vein
st.encode_bone, st.encode_veins = True, True
check(st.vein_mask()[sinus].mean() > 0.5, f"one series: the sinus is traced as a vein ({st.vein_mask()[sinus].mean():.2f})")
check(not st.vein_mask()[art].any(), "and the artery is not")

# a routine contrast head CT: sinus and artery equally bright; only the sinus by the skull is a vein
rt = np.full((150, 100, 100), 40, np.int16)
zr, yr, xr = np.mgrid[0:150, 0:100, 0:100]
rr = np.hypot(yr - 50, xr - 50)
rt[rr > 44] = 1500
art_r = np.hypot(yr - 50, xr - 30) < 3
sin_r = (np.hypot(yr - 50, xr - 88) < 6) & (rr <= 44)
rt[art_r], rt[sin_r] = 160, 160
srt = VolumeState(Volume(data=rt, spacing=(1.0, 1.0, 1.0), origin=(0, 0, 0), modality="CT", description="Head + CM",
                         frame_uid="f"))
srt.encode_bone, srt.encode_veins = True, True
check(srt.vein_mask()[sin_r].mean() > 0.5 and srt.artery_mask()[art_r].mean() > 0.9 and not srt.vein_mask()[art_r].any(),
      "routine contrast CT: the sinus is blue, the artery of the same HU stays with the other vessels")

st.partner = Volume(data=a, spacing=(1.0, 1.0, 1.0), origin=(0, 0, 0), modality="CT", description="CTA ARTER", frame_uid="f")

# 3D of a CTA: from the venous series arteries / veins, from the arterial series (and a routine contrast CT) vessels + bone
from dcv.main_window import MainWindow  # noqa: E402

art_st = VolumeState(st.partner)
art_st.partner = vol
for s_ in (st, art_st, srt):
    MainWindow._default_vein_preset(s_)
check("Veins" in st.tf.name and art_st.tf.name == "CT Angio - Vessels + Bone" and srt.tf.name == "CT Angio - Vessels + Bone",
      "3D starts on arteries / veins only from the venous series of a CTA")
st.tf = next(p for p in builtin_presets() if "Veins" in p.name).copy()
st.update_render_data()
d = st.render_data
check(d[art].max() == 450 or d[art].max() >= 300, "arteries show the HU of the arterial series on the venous window")
check((d[sinus] >= segment.VEIN_OFFSET).mean() > 0.6, "veins in their own domain")
check(np.array_equal(st.vol.data, v), "the series itself stays untouched")
w = VRWindow(st)
w.show()
app.processEvents()
check("Other phase" in w.phase_note.text(), "the panel says which series it combines")
check(w.combine_chk.isVisibleTo(w), "the combine checkbox shows with a pair")
check(not w.artery_spin.isVisibleTo(w), "no artery threshold with a pair")
w.combine_chk.setChecked(False)
app.processEvents()
check(not st.paired and (st.render_data[sinus] >= segment.VEIN_OFFSET).any(), "unchecked: one series again")
w.combine_chk.setChecked(True)

# choosing arteries / veins: seconds of work behind a wait panel, never a frozen window
import dcv.vr_window as vrw
shown = []


class SpyBusy(vrw.BusyDialog):
    def stage(self, text):
        shown.append(text)
        super().stage(text)


keep, vrw.BusyDialog = vrw.BusyDialog, SpyBusy
pick = {w.preset_combo.itemData(i)[1].name: i for i in range(w.preset_combo.count())
        if w.preset_combo.itemData(i)[0] == "builtin"}
w._on_preset_selected(pick["CT Angio - Vessels + Bone"])
check(not st.vein_encoded, "Vessels + Bone: no vein domain")
w._on_preset_selected(pick["CT Angio - Arteries / Veins + Bone"])
vrw.BusyDialog = keep
app.processEvents()
check(any("arteries and veins" in s for s in shown) and st.vein_encoded and w._busy is None and not w._working,
      "switching to arteries / veins shows the wait panel, then the veins")

# a seed without 3D: no surface until asked for
st2 = VolumeState(Volume(data=v.copy(), spacing=(1.0, 1.0, 1.0), origin=(0, 0, 0), modality="CT", description="x", frame_uid="f"))
seed = (75, 50, 88)
res, (lo, hi) = auto_grow(st2.vol.data, st2.vol.spacing, seed)
seg = Segment("S", res[0], res[1], st2.vol.shape, (0.1, 0.9, 0.2), st2.vol.spacing)
check(seg._polydata is None, "no 3D surface is made for a seed in 2D")
seg.range, seg.seed = (lo, hi), seed
st2.segments.append(seg)
st2.active_segment = seg
panel = SeedPanel()
panel.follow(st2, grower_of(st2))
panel.show()
app.processEvents()
check(panel.editor.isVisible(), "the curve opens in 2D")
g = grower_of(st2)
g.request(lo - 10, hi + 10)
g._job["thread"].join()
g._poll()
app.processEvents()
new = st2.active_segment
check(new is not seg and new.range == (lo - 10, hi + 10), "moving the handles re-grows the segment")
check(panel.editor.lo == lo - 10, "the panel follows")

w2 = VRWindow(st2)
w2.show()
w2.tabs.setCurrentIndex(1)
app.processEvents()
check(w2.seed_range.isVisibleTo(w2), "the 3D window has the same curve")
check(new.polydata.GetNumberOfPoints() > 0, "the surface is made once 3D asks for it")
g.request(lo - 5, hi + 5)
g._job["thread"].join()
g._poll()
app.processEvents()
check(w2.seed_range.lo == lo - 5 and panel.editor.lo == lo - 5, "a range change reaches both curves")
w2.close()
panel.close()
w.close()
print(f"errors: {errors}")
sys.exit(1 if errors else 0)
