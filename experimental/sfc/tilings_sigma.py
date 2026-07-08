"""Enumerate ALL 9-half-hex tilings of the half-hex (expect 49, per
experimental/halfhex.py) and find the sigma-symmetric ones (invariant under
the tile mirror). A sigma-symmetric dissection is the necessary substrate
for a self-mirror curve shape (Ben's 'A'); the canonical pair T1/T2 are
mirror images of each other, so neither qualifies."""

import sys
from fractions import Fraction
from collections import Counter

sys.path.insert(0, "/Users/ben/Documents/Projects/PyCharm/hex9/experimental/sfc")
from sfc_grammar import GROUND, LIB, tri_verts

F = Fraction
# tile mirror in (a,b): sigma(p) = S p + (6,0)
def sigma(p):
    a, b = p
    return (-a - b + 6, b)


def cell_key(cell):
    return frozenset(tri_verts(cell))


# map each triangle cell to its sigma-image cell
CELLS = sorted(GROUND, key=lambda e: (e[1], e[0]))
BY_VERTS = {cell_key(c): c for c in CELLS}
SIG_CELL = {}
for c in CELLS:
    vs = frozenset(sigma(v) for v in tri_verts(c))
    SIG_CELL[c] = BY_VERTS[vs]          # KeyError would mean sigma wrong
assert all(SIG_CELL[SIG_CELL[c]] == c for c in CELLS)

# all placements of all 6 orientations
placements = set()
for k, shape in LIB.items():
    for oy in range(-8, 9, 2):
        for ox in range(-8, 9):
            cells = frozenset((px + ox, py + oy) for (px, py) in shape)
            if cells <= GROUND:
                placements.add(cells)
placements = sorted(placements, key=lambda s: sorted(s))

solutions = []
def cover(uncovered, chosen):
    if not uncovered:
        solutions.append(frozenset(chosen))
        return
    cell = min(uncovered, key=lambda e: (e[1], e[0]))
    for cells in placements:
        if cell in cells and cells <= uncovered:
            chosen.append(cells)
            cover(uncovered - cells, chosen)
            chosen.pop()

cover(frozenset(GROUND), [])
print(f"total 9-half-hex tilings of the half-hex: {len(solutions)}")

symmetric = []
for sol in solutions:
    img = frozenset(frozenset(SIG_CELL[c] for c in piece) for piece in sol)
    if img == sol:
        symmetric.append(sol)
print(f"sigma-symmetric tilings: {len(symmetric)}")
for s in symmetric:
    # self-symmetric pieces within it
    selfsym = [p for p in s if frozenset(SIG_CELL[c] for c in p) == p]
    print(f"  tiling with {len(selfsym)} self-symmetric piece(s)")
    # print compact form: sorted pieces as sorted cell lists
    for p in sorted(s, key=lambda p: sorted(p)):
        mark = " <- self-symmetric" if p in selfsym else ""
        print("   ", sorted(p), mark)
