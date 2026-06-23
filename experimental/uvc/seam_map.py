"""
Cross-octant seam map (L1).

For every geometric adjacency that crosses an octant boundary, record
(octA, digitA, c2A) -> (octB, digitB, c2B). Then:
  1. confirm the octant part is governed by H9O.oid_nb[octA, c2]
  2. characterise the digit transform across the seam (is it a fixed relabelling?)

This is the rule step_region needs when a carry runs off the octant
(currently returns None): new_oct = oid_nb[oct, c2], geometry y-flips.
"""
import sys
import numpy as np
from collections import defaultdict
from l1_traverse_testbed import build_cells
from hhg9.h9 import H9O

DENS = int(sys.argv[1]) if len(sys.argv) > 1 else 360
cells, _ = build_cells(1, DENS)
print(f"oid_nb (octant,c2)->neighbour octant:\n{H9O.oid_nb}")
print(f"oid_mo (octant->mode): {H9O.oid_mo.tolist()}")

# all cross-octant adjacencies (use raw ring, include non-interior)
cross = []
for i, r in enumerate(cells):
    for j in r['ring']:
        c = cells[j]
        if c['oct'] != r['oct']:
            cross.append((r, c))
print(f"\ncross-octant adjacencies: {len(cross)}")

# 1. octant pairs vs oid_nb
oct_pairs = defaultdict(int)
nb_set = defaultdict(set)
for a, c in cross:
    oct_pairs[(a['oct'], c['oct'])] += 1
    nb_set[a['oct']].add(c['oct'])
print("\noctant adjacency (from geometry) vs oid_nb rows:")
for o in range(8):
    geo = sorted(nb_set.get(o, set()))
    nb = sorted(set(int(x) for x in H9O.oid_nb[o]))
    print(f"  oct {o} (mode {int(H9O.oid_mo[o])}): geo-neighbours {geo}  | oid_nb {nb}  {'OK' if geo==nb else 'DIFF'}")

# 2. EDGE-crossings only (octB in oid_nb[octA]); crossing-c2 = the oid_nb column.
def crossing_c2(octA, octB):
    row = list(int(x) for x in H9O.oid_nb[octA])
    return row.index(octB) if octB in row else None


print("\nEDGE-seam digit transform  (octA, digitA, crossing_c2) -> (digitB, c2B):")
trans = defaultdict(lambda: defaultdict(int))
for a, c in cross:
    xc2 = crossing_c2(a['oct'], c['oct'])
    if xc2 is None:                       # vertex crossing, not an edge seam
        continue
    trans[(a['oct'], a['leaf'], xc2)][(c['leaf'], c['c2'])] += 1

shown = 0
for k in sorted(trans):
    outs = trans[k]
    best = max(outs, key=outs.get)
    if shown < 18:
        print(f"  oct{k[0]} d{k[1]} xc2={k[2]} -> d{best[0]} c2={best[1]}"
              f"{'' if len(outs) == 1 else '  '+str(dict(outs))}")
        shown += 1
ncons = sum(len(v) == 1 for v in trans.values())
print(f"\nEDGE-seam digit transform consistent: {ncons}/{len(trans)} keys")
# digit relabelling summary (ignore c2): is digitB a function of (octA-mode, digitA)?
dmap = defaultdict(lambda: defaultdict(int))
for a, c in cross:
    if crossing_c2(a['oct'], c['oct']) is not None:
        dmap[(int(H9O.oid_mo[a['oct']]), a['leaf'])][c['leaf']] += 1
print("\ndigit relabel by (octA_mode, digitA) -> digitB:")
for k in sorted(dmap):
    best = max(dmap[k], key=dmap[k].get)
    print(f"  mode{k[0]} d{k[1]} -> d{best}   {dict(dmap[k])}")
