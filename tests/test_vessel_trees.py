"""Arteries and veins of one series as two competing trees (phases.vessel_trees): a stretch of a sinus that the atlas
labelled wrongly (without being sure) takes the label of the vessel it belongs to; across a thin contact the other tree
takes only what lies nearer its own seeds than the vessel's (the fronts are not slowed there: on six real CTAs a cost
for thin places changed nothing); a thin vessel the atlas is sure of seeds nothing; with no seed of one kind the labels
stay as they were."""
import sys

app_dir, data_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, app_dir)

import numpy as np  # noqa: E402

from dcv.phases import vessel_trees  # noqa: E402

errors = 0


def check(ok, text):
    global errors
    if not ok:
        errors += 1
    print(("ok   " if ok else "FAIL ") + text)


shape, sp = (40, 40, 80), (1.0, 1.0, 1.0)
z, y, x = np.mgrid[0:40, 0:40, 0:80]
sinus = (np.hypot(z - 20, y - 12) <= 3) & (x >= 5) & (x < 75)          # a thick tube along x
artery = (np.hypot(z - 20, y - 26) <= 2.5) & (x >= 5) & (x < 75)       # another, 14 cells away
bridge = (z == 20) & (y > 12) & (y < 26) & (x == 40)                    # one thin contact between them
vessel = sinus | artery | bridge
traced = sinus & ((x < 30) | (x >= 50))                                 # the tracing missed the middle of the sinus


def prior_with(sure_artery):
    def prior(mask):
        out = np.full(shape, 0.5, np.float32)
        out[sure_artery & mask] = 0.97
        return out
    return prior


# what the atlas gave: the sinus's middle and the bridge as arteries, a stretch of the artery as a vein; sure only of
# the artery's far end
start_a = (artery & ~((x >= 15) & (x < 30))) | (sinus & (x >= 30) & (x < 50)) | bridge
start_v = vessel & ~start_a
a, v = vessel_trees(start_a, start_v, traced, prior_with(artery & (x > 60)), sp)
mid = sinus & (x >= 30) & (x < 50)
check(v[mid].all(), f"the sinus's middle, joined to its traced parts, is a vein ({v[mid].mean():.0%})")
check(v[sinus].all(), "the whole sinus is a vein")
near = artery & (x >= 45)
check(a[near].all(), "the artery from a few cells past the contact to its seeds is an artery")
check(a[artery].mean() > 0.9, f"the artery is an artery nearly all along, also at the contact ({a[artery].mean():.0%})")
check(not (a & v).any() and ((a | v) == vessel).all(), "each vessel cell has one label")
# a thin vessel the atlas is sure of: no seed (TREE_ARTERY_MM), so nothing changes without other seeds
thin = (z == 20) & (y == 5) & (x >= 5) & (x < 75)
a2, v2 = vessel_trees(np.zeros(shape, bool), thin, np.zeros(shape, bool), prior_with(thin), sp)
check(not a2.any() and v2[thin].all(), "a thin vessel seeds no artery tree: the labels stay as they were")

print(f"errors: {errors}")
sys.exit(1 if errors else 0)
