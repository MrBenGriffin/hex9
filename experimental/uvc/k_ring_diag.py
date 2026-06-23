"""
Diagnose the k-ring undercount: is step_region's carry giving geometrically
correct neighbours, does hex_key coalesce the 3 half-hexes, are all siblings reached?
"""
import numpy as np
from collections import defaultdict
from build_stepper import make_chain, step_region
from hhg9.h9.region import regions_xy
from hhg9.h9.addressing import reg_hex_digits, TailStyle
from hhg9 import Registrar

reg = Registrar(); b = reg.domain('b_oct')


def centroid(chain):
    xy = regions_xy(np.asarray(chain, dtype=np.uint8)[None, :])
    return np.asarray(xy[0][:2], dtype=float)


def hkey(chain, oc):
    hv = reg_hex_digits(np.asarray(chain, dtype=np.uint8)[None, :], oc, b, TailStyle.key)
    return tuple(int(x) for x in hv[0])


chain, pb, oc = make_chain(5.0, 25.0)
c0 = centroid(chain)
# unit spacing = distance to the nearest stepped neighbour
nbrs = [step_region(np.asarray(chain, dtype=np.int64), e) for e in range(3)]
dists = [np.linalg.norm(centroid(n) - c0) for n in nbrs if n is not None]
s = min(dists)
print("step_region carry CORRECTNESS (neighbour centroid distance / unit spacing):")
for e, n in enumerate(nbrs):
    if n is None:
        print(f"  edge {e}: None"); continue
    print(f"  edge {e}: {np.linalg.norm(centroid(n)-c0)/s:.2f}   (1.0 = adjacent; >1.7 = wrong/far)")

# flood 3 region-steps; group regions by hexagon key
dist = {tuple(int(x) for x in chain): 0}
frontier = [chain]
for d in range(1, 4):
    nxt = []
    for ch in frontier:
        for e in range(3):
            nb = step_region(np.asarray(ch, dtype=np.int64), e)
            if nb is None:
                continue
            t = tuple(int(x) for x in nb)
            if t not in dist:
                dist[t] = d; nxt.append(nb)
    frontier = nxt

hexgroups = defaultdict(list)
for t in dist:
    hexgroups[hkey(np.asarray(t), oc)].append(t)
ck = hkey(chain, oc)
print(f"\ncentre hexagon key {ck}: {len(hexgroups[ck])} regions belong to it (expect 3 half-hexes)")
sizes = sorted(len(v) for v in hexgroups.values())
print(f"regions-per-hexagon distribution: {dict((s, sizes.count(s)) for s in set(sizes))}")
print(f"distinct hexagons within 3 region-steps: {len(hexgroups)}")
