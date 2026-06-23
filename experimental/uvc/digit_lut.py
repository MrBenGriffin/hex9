"""
Digit-neighbour LUT (legible form Ben asked for):
   key = (my_digit, my_c2, parent_region)  ->  multiset of neighbour (digit, c2)

"parent_region" is the combinatorial state (= the phase carrier identified in
bridge_phase: parent rid predicts phase). Built from the geometric ring at L2.
Reports how consistent the LUT is, and prints a sample so you can *see* what
digits surround a given cell.
"""
import sys
import numpy as np
from collections import defaultdict
from l1_traverse_testbed import build_cells
from hhg9.h9 import HEX_LUTS, H9_RA
from hhg9.h9.lattice import H9C

LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 2
DENS = int(sys.argv[2]) if len(sys.argv) > 2 else 900
hr = HEX_LUTS.hex_reg
oob = HEX_LUTS.hex_oob

cells, _ = build_cells(LAYER, DENS)


def parent_region(body, c2_leaf):
    """walk leaf->parent via hex_reg; return the parent level's region id (the state)."""
    c_mo, c2 = 0, c2_leaf
    reg = None
    for i in range(len(body) - 1, 0, -1):
        d = body[i]
        r, pmo, pc2 = (int(x) for x in hr[d, c_mo, c2])
        if reg is None:               # leaf region (skip, it's constant per leaf,c2)
            reg = r
            c_mo, c2 = pmo, pc2
            continue
        return r                       # parent region
    return -1


for r in cells:
    if r['interior']:
        r['preg'] = parent_region(r['body'], r['c2'])

lut = defaultdict(list)
for r in cells:
    if not r['interior']:
        continue
    nb = tuple(sorted((cells[j]['leaf'], cells[j]['c2'])
                      for j in r['ring'] if cells[j]['interior']))
    if len(nb) == 6:
        lut[(r['leaf'], r['c2'], r['preg'])].append(nb)

consistent = sum(len(set(v)) == 1 for v in lut.values())
print(f"\nLUT keys: {len(lut)};  consistent (single neighbour-set): {consistent}/{len(lut)}")

# is the neighbour multiset determined by my_digit ALONE?
by_digit = defaultdict(list)
for r in cells:
    if r['interior']:
        nb = tuple(sorted((cells[j]['leaf'], cells[j]['c2'])
                          for j in r['ring'] if cells[j]['interior']))
        if len(nb) == 6:
            by_digit[r['leaf']].append(tuple(d for d, _ in nb))
print("\nneighbour DIGITS by my_digit alone (most common multiset):")
for d in range(9):
    from collections import Counter
    if by_digit[d]:
        common, n = Counter(by_digit[d]).most_common(1)[0]
        frac = n / len(by_digit[d])
        print(f"  digit {d}: neighbours {list(common)}   ({frac:.0%} of {len(by_digit[d])} cells)")
print("\nsample rows  (my_digit, my_c2 | parent_region) -> neighbour digits:")
for k in sorted(lut)[:14]:
    sets = set(lut[k])
    nb = next(iter(sets))
    digs = [d for d, _ in nb]
    flag = '' if len(sets) == 1 else f'  <{len(sets)} variants>'
    print(f"  d={k[0]} c2={k[1]} | preg={k[2]:2d} -> neighbours {digs}{flag}")
