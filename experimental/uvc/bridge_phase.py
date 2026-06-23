"""
Bridge: what combinatorial quantity equals the empirical phase?

Walk each interior cell's body leaf->root via hex_reg (digit,c_mo,c2)->(region,
parent_mode,parent_c2), collecting the per-level (region, mode) chain. Then test
which feature determines the geometric phase:
  - leaf's own (mode, sib)            [expected: CONSTANT per (leaf,c2) -> can't be phase]
  - the c_mo chain / region at each ancestor level
A feature 'determines phase' if (leaf, c2, feature) -> phase is single-valued.
"""
import sys
import numpy as np
from collections import defaultdict
from l1_traverse_testbed import build_cells
from g_extract import assign_phases
from hhg9.h9 import HEX_LUTS, H9_RA
from hhg9.h9.lattice import H9C

LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 2
DENS = int(sys.argv[2]) if len(sys.argv) > 2 else 900

cells, _ = build_cells(LAYER, DENS)
assign_phases(cells)
hr = HEX_LUTS.hex_reg            # [digit, c_mo, c2] -> [region, parent_mode, parent_c2]
oob = HEX_LUTS.hex_oob


def walk(body, c2_leaf):
    """leaf->root: yield (region, region_mode, region_sib) per body level (cols 2..)."""
    c_mo, c2 = 0, c2_leaf         # leaf parent-mode is 0 after coalesce
    chain = []
    for i in range(len(body) - 1, 0, -1):    # leaf (last col) -> col1 (col0=root is special)
        d = body[i]
        reg, pmo, pc2 = (int(x) for x in hr[d, c_mo, c2])
        mode = int(H9C.mode[H9_RA.rid2cell[reg]]) if reg != oob else -1
        chain.append((reg, mode, 1 if reg >= 6 else 0))
        c_mo, c2 = pmo, pc2
    return chain                   # [leaf_level, ..., toward root]


# attach chain to each interior cell
for r in cells:
    if r['interior']:
        r['chain'] = walk(r['body'], r['c2'])

interior = [r for r in cells if r['interior'] and r['chain'] and r['chain'][0][0] != oob]


def predicts(name, featfn):
    groups = defaultdict(set)
    for r in interior:
        groups[(r['leaf'], r['c2'], featfn(r))].add(r['phase'])
    det = sum(len(v) == 1 for v in groups.values())
    print(f"  (leaf,c2,{name:<22}) -> phase : {det}/{len(groups)} single-valued")


print(f"\ninterior cells with valid chain: {len(interior)}")
# is the leaf's own (mode,sib) constant per (leaf,c2)?  (expected yes -> not the phase)
lvar = defaultdict(set)
for r in interior:
    lvar[(r['leaf'], r['c2'])].add((r['chain'][0][1], r['chain'][0][2]))
nconst = sum(len(v) == 1 for v in lvar.values())
print(f"leaf (mode,sib) constant within (leaf,c2): {nconst}/{len(lvar)} buckets  "
      f"(constant => leaf's own mode/sib is NOT the phase)")

print("\nwhich feature determines phase?")
predicts("leaf_mode,leaf_sib", lambda r: (r['chain'][0][1], r['chain'][0][2]))
predicts("parent_region", lambda r: r['chain'][1][0] if len(r['chain']) > 1 else -1)
predicts("parent_mode,parent_sib", lambda r: r['chain'][1][1:] if len(r['chain']) > 1 else (-1, -1))
predicts("c_mo at parent", lambda r: int(hr[r['body'][-1], 0, r['c2']][1]))
predicts("full mode chain", lambda r: tuple(m for _, m, _ in r['chain']))
predicts("region chain", lambda r: tuple(rg for rg, _, _ in r['chain']))
