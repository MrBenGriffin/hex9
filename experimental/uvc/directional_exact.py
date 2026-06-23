"""
Exact directional transition table (artifact-free).

Direction labelling uses the b_oct lattice, not sphere bearings: within a flat
octant the lattice is translation-invariant, so ONE clean cell's 6 neighbour
offsets are the octant's direction templates. Assign each neighbour to the nearest
template -> exact cyclic order (handles the lattice skew, no 60-deg binning, no
projection distortion).

Then build the straight-step table by came-from/opposite (+3) and check Ben's rule:
  next = f(came_from_digit, current_digit)  EXCEPT a doubled came_from needs the
  2-back digit. Expect: artifacts gone; only doubled-neighbour cases need order-3.
"""
import sys
import math
import numpy as np
from collections import defaultdict
from l1_traverse_testbed import build_cells

LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 2
DENS = int(sys.argv[2]) if len(sys.argv) > 2 else 900

cells, _ = build_cells(LAYER, DENS)


def deltas(i):
    r = cells[i]
    out = []
    for j in r['ring']:
        c = cells[j]
        if c['oct'] == r['oct'] and c['interior']:
            d = np.asarray(c['planar']) - np.asarray(r['planar'])
            out.append((j, d / (np.linalg.norm(d) + 1e-12)))
    return out


# per-octant direction templates from one clean reference cell
templates = {}
for i, r in enumerate(cells):
    if r['oct'] in templates or not r['interior']:
        continue
    dl = deltas(i)
    if len(dl) == 6:
        vecs = [v for _, v in dl]
        ang = sorted(range(6), key=lambda k: math.atan2(vecs[k][1], vecs[k][0]))
        templates[r['oct']] = [vecs[k] for k in ang]   # dir 0..5 by angle


def ring_by_dir(i):
    """return [neighbour_idx per dir 0..5] or None if not a clean 6-cell."""
    oc = cells[i]['oct']
    if oc not in templates:
        return None
    dl = deltas(i)
    if len(dl) != 6:
        return None
    slots = [None] * 6
    for j, v in dl:
        d = int(np.argmax([v @ t for t in templates[oc]]))
        if slots[d] is not None:
            return None                     # two neighbours same dir -> ambiguous
        slots[d] = j
    return slots


# build came-from/opposite tables
T1 = defaultdict(set)     # (came_from_digit, current_digit) -> next_digit
T2 = defaultdict(set)     # (2back_digit, came_from_digit, current_digit) -> next_digit
n_clean = 0
for i, r in enumerate(cells):
    if not r['interior']:
        continue
    slots = ring_by_dir(i)
    if slots is None:
        continue
    n_clean += 1
    cur = r['leaf']
    for d in range(6):
        came = cells[slots[d]]['leaf']
        nxt = cells[slots[(d + 3) % 6]]['leaf']
        T1[(came, cur)].add(nxt)
        # 2-back: where the came-from cell came from on a straight line = its
        # neighbour opposite to us, i.e. came-from's dir-d neighbour's digit.
        cf_slots = ring_by_dir(slots[d])
        if cf_slots is not None:
            two = cells[cf_slots[d]]['leaf']        # same dir from the came-from cell
            T2[(two, came, cur)].add(nxt)

print(f"\nclean cells: {n_clean}")
d1 = sum(len(v) == 1 for v in T1.values())
print(f"T1 (came_from,current)->next : {d1}/{len(T1)} deterministic")
print("\nper current-digit ambiguity (came_from list):")
for cur in range(9):
    amb = sorted(k[0] for k in T1 if k[1] == cur and len(T1[k]) > 1)
    print(f"  current {cur}: ambiguous from {amb if amb else '(none)'}")

d2 = sum(len(v) == 1 for v in T2.values())
print(f"\nT2 (2back,came_from,current)->next : {d2}/{len(T2)} deterministic")
