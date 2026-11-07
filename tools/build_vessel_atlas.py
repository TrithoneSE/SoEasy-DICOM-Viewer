"""Build assets/vessel_atlas.bin (and web/public/vessel_atlas.bin): where in the head arteries and where veins run, from
multiphase head CTAs (pre-contrast, arterial and venous series of one study, as GE "CTA Carotis/COW" exams).

    python tools/build_vessel_atlas.py cache <folder of study zips> <cache dir>
    python tools/build_vessel_atlas.py build <cache dir> [<repo root>]

cache: each zip's PRE / ARTER / VENA series as int16 HU volumes (numbers only: no names, dates or IDs).
build: arteries and veins of every study from its two phases (dcv.phases.classify_phases); studies whose arterial series
was late (sinuses as full as the arteries: the time course separates nothing) are left out. Every head is registered by
its skull (dcv.atlas.register) onto a template: the AVERAGE skull of the studies, so no single patient's skull is stored.
The atlas holds, on a 2 mm grid of the template head: the distance to the template's skull surface (for registering a
new scan), and per cell the share of arteries among the studies' vessels near it (Gaussian SIGMA_MM) and how many there
were. scipy is needed here only (not by the app)."""
import io
import json
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SIGMA_MM = 4.0          # leave-one-out on 5 CTAs: 3-6 mm alike, 4 a little better
GRID_MM = 2.0


def cache(src, out):
    import pydicom
    out.mkdir(parents=True, exist_ok=True)
    names = {"pre": ("PRE",), "art": ("ARTER",), "ven": ("VENA",)}

    def read_one(zpath, name):
        with zipfile.ZipFile(zpath) as zf:
            ds = pydicom.dcmread(io.BytesIO(zf.read(name)))
        arr = ds.pixel_array.astype(np.float32) * float(ds.get("RescaleSlope", 1)) + float(ds.get("RescaleIntercept", 0))
        return float(ds.ImagePositionPatient[2]), np.clip(arr, -2048, 32000).astype(np.int16)

    for k, z in enumerate(sorted(src.glob("*.zip")), 1):
        groups, first = {}, {}
        with zipfile.ZipFile(z) as zf:
            for i in zf.infolist():
                if i.is_dir() or i.file_size < 100000:
                    continue
                with zf.open(i) as f:
                    ds = pydicom.dcmread(io.BytesIO(f.read(16384)), stop_before_pixels=True, force=True)
                kind = list(ds.get("ImageType", []))
                if "ORIGINAL" not in kind or "AXIAL" not in kind:
                    continue
                desc = str(ds.get("SeriesDescription", "")).upper()
                for p, words in names.items():
                    if any(w in desc for w in words):
                        groups.setdefault(p, []).append(i.filename)
                        first.setdefault(p, ds)
                        break
        if "art" not in groups or "ven" not in groups:
            continue
        meta = {}
        for p, files in groups.items():
            with ThreadPoolExecutor(6) as ex:
                res = sorted(ex.map(lambda n: read_one(z, n), files), key=lambda r: r[0])
            vol = np.stack([r[1] for r in res])
            np.save(out / f"s{k}_{p}.npy", vol)
            zs = [r[0] for r in res]
            meta[p] = dict(shape=vol.shape, dz=float(np.median(np.diff(zs))), px=[float(v) for v in first[p].PixelSpacing])
            del vol, res
        (out / f"s{k}.json").write_text(json.dumps(meta))
        print("cached study", k, flush=True)


def build(src, root, exclude=(), outs=None):
    from scipy import ndimage

    from dcv import atlas as at
    from dcv import phases as ph

    studies = []
    for meta_path in sorted(src.glob("s*.json")):
        k = meta_path.stem
        if k in exclude:
            continue
        meta = json.loads(meta_path.read_text())
        if "ven" not in meta or meta["ven"]["shape"] != meta.get("art", {}).get("shape"):
            continue
        m = meta["ven"]
        sp0 = (m["dz"], *m["px"])
        A, V = np.load(src / f"{k}_art.npy"), np.load(src / f"{k}_ven.npy")
        nat = ph.pooled_native(np.load(src / f"{k}_pre.npy"), sp0) if meta.get("pre", {}).get("shape") == m["shape"] else None
        r = ph.classify_phases(A, V, sp0, 500, nat)
        if r.late:
            print(k, "left out: a late arterial series", flush=True)
            continue
        _, ta = ph._pool(A, r.steps)
        _, tv = ph._pool(V, r.steps)
        sp = np.array([s * f for s, f in zip(sp0, r.steps)])
        studies.append(dict(name=k, sp=sp, bone=(ta >= 500) & (tv >= 500), art=r.arteries, ven=r.veins))
        print(k, "in", flush=True)
        del A, V
    if len(studies) < 2:
        sys.exit("need at least two studies with a timely arterial series")

    # template grid: the first study's head, 2 mm, from 150 mm under its vertex up
    s0 = studies[0]
    vtx = at.vertex(s0["bone"]) * s0["sp"][0]
    lo = np.array([vtx - at.ATLAS_DEPTH_MM, 0.0, 0.0])
    hi = np.array([vtx + 6.0, s0["bone"].shape[1] * s0["sp"][1], s0["bone"].shape[2] * s0["sp"][2]])
    shape = tuple(int(np.ceil(v)) for v in (hi - lo) / GRID_MM)
    gsp = np.full(3, GRID_MM)
    q = np.argwhere(np.ones(shape, bool)) * gsp + lo          # template mm of every grid cell

    def onto_grid(st, T, mask):
        """Share of the cell covered by the mask: 8 points per 2 mm cell, so a 1-2 mm vessel is not skipped."""
        out = np.zeros(len(q), np.float32)
        for off in np.array(np.meshgrid(*[[-0.5, 0.5]] * 3, indexing="ij")).reshape(3, -1).T:
            p = np.round(T.inverse(q + off * GRID_MM / 2) / st["sp"]).astype(int)
            ok = np.all((p >= 0) & (p < np.array(mask.shape)), axis=1)
            out[ok] += mask[p[ok, 0], p[ok, 1], p[ok, 2]]
        return (out / 8).reshape(shape)

    # the average skull: register to the current template, average the bone, twice
    tmpl_bone = onto_grid(s0, at.Affine(np.eye(3), np.zeros(3), np.zeros(3)), s0["bone"]) > 0.5
    for rnd in range(3):
        surf = at.surface(tmpl_bone)
        dist = np.minimum(ndimage.distance_transform_edt(~surf, sampling=gsp), 30.0).astype(np.float32)
        centre = at.head_points(tmpl_bone, gsp).mean(0)
        atlas = at.VesselAtlas(dist, np.full(shape, 0.5, np.float32), np.zeros(shape, np.float32), gsp, lo, centre + lo)
        acc = np.zeros(shape, np.float32)
        for st in studies:
            st["T"], st["fit"] = atlas.register(st["bone"], st["sp"])
            acc += onto_grid(st, st["T"], st["bone"])
        print(f"round {rnd}: fit (share of skull points within 2 mm) " + ", ".join(f"{s['fit']:.2f}" for s in studies), flush=True)
        tmpl_bone = acc / len(studies) >= 0.5

    a = np.zeros(shape, np.float32)
    v = np.zeros(shape, np.float32)
    for st in studies:
        a += ndimage.gaussian_filter(onto_grid(st, st["T"], st["art"]), SIGMA_MM / GRID_MM)
        v += ndimage.gaussian_filter(onto_grid(st, st["T"], st["ven"]), SIGMA_MM / GRID_MM)
    support = (a + v) / len(studies)
    prob = np.where(a + v > 1e-6, a / np.maximum(a + v, 1e-6), 0.5).astype(np.float32)
    # only the head: the skull and the vessels with a margin (a quarter of the first study's field of view)
    keep = np.argwhere(tmpl_bone | (support > 1e-3))
    a0, a1 = np.maximum(keep.min(0) - 8, 0), np.minimum(keep.max(0) + 9, np.array(shape))
    box = tuple(slice(i, j) for i, j in zip(a0, a1))
    final = at.VesselAtlas(dist[box], prob[box], support[box], gsp, lo + a0 * gsp, centre + lo)
    for out in outs or (root / "assets" / "vessel_atlas.bin", root / "web" / "public" / "vessel_atlas.bin"):
        final.save(str(out), studies=len(studies))
        print("wrote", out, Path(out).stat().st_size // 1024, "KB")


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) >= 3 and args[0] == "cache":
        cache(Path(args[1]), Path(args[2]))
    elif len(args) >= 2 and args[0] == "build":
        # --exclude s3,s5 --out path: an atlas without those studies (leave-one-out testing), written only there
        opt = {args[i]: args[i + 1] for i in range(len(args) - 1) if args[i].startswith("--")}
        pos = [a for i, a in enumerate(args[1:], 1) if not a.startswith("--") and not args[i - 1].startswith("--")]
        build(Path(pos[0]), Path(pos[1]) if len(pos) > 1 else ROOT, tuple(filter(None, opt.get("--exclude", "").split(","))),
              [Path(opt["--out"])] if "--out" in opt else None)
    else:
        sys.exit(__doc__)
