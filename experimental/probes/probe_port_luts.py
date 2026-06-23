# Port-prep probe: (1) dump the exact LUTs the C cascade needs, ground-truth from
# region.py; (2) over the 521 L29 wing-terminus diverging cells, classify whether
# region_neighbours' fold is octant-"hopped" (root region changes) — i.e. whether a
# projection-free combinatorial cascade in C can close them, or whether the geometric
# reproject in _coalesce_bin (uuid_address.py:129-134) is load-bearing for them.
import sys
sys.path.insert(0, '.')
import numpy as np
from hhg9.h9.region import H9R, region_neighbours, xy_regions
from hhg9.h9 import H9C
from hhg9.h9.uuid_address import h9_encode, h9_bin, _batch_int_to_nibbles

print("=== H9R.proto ===", H9R.proto)
print("=== H9C cell ids in scope (0x.. ) ===")
ids = [i for i in range(H9C.count) if H9R.is_in[i]]
print("in-scope cells:", [hex(i) for i in ids], " count=", H9C.count, " bad=", H9R.invalid_region)
print("=== mode per in-scope cell ===")
print({hex(i): int(H9C.mode[i]) for i in ids})

print("=== H9R.child (mcc2 child terminal): child[mode][c2] ===")
# child = h9c.c2 ; shape (mode, c2, k)
print("child shape:", H9C.c2.shape)
for m in range(H9C.c2.shape[0]):
    print(f" mode {m}: ", [[hex(int(v)) for v in row] for row in H9C.c2[m]])

print("=== H9R.cmc2n live entries [cell][mode][c2] -> (nbr, pmn) ===")
for cell in ids:
    for m in range(2):
        row = H9R.cmc2n[cell, m]
        if np.all(row[:, 0] == H9R.invalid_region):
            continue
        print(f" ({hex(cell)},{m}): " + ", ".join(f"[{hex(int(a))},{int(b)}]" for a, b in row))

print("=== H9R.mcc2 [mode][cell] -> c2 (live cells only) ===")
for m in range(H9R.mcc2.shape[0]):
    live = {hex(c): int(H9R.mcc2[m, c]) for c in ids if H9R.mcc2[m, c] != H9R.invalid_region}
    print(f" mode {m}: {live}")

# ---- (2) hop classification over the 521 diverging L29 cells ----
print("\n=== HOP classification over L29 diverging cells ===")
rng = np.random.default_rng(1)
n = 3000
lats = rng.uniform(-89.5, 89.5, n); lons = rng.uniform(-179.5, 179.5, n)
uuids = h9_encode(lats, lons)
L = 29
pyb = h9_bin(list(uuids), L)
# get region addresses for the L29 bin via xy_regions is geometry-based; instead
# reconstruct from the FULL uuid's body nibbles -> region ids is non-trivial.
# Simpler: replicate _coalesce_bin's region path for these points at layer L.
from hhg9 import Registrar, Points
from hhg9.h9.classifier import location
from hhg9.h9.protocols import BaryLoc
from hhg9.h9 import H9K
reg = Registrar(); g = reg.domain('g_gcd'); b = reg.domain('b_oct')
bp = reg.project(Points(np.column_stack([lats, lons]), g), [g, b])
coords = bp.coords; oc, mo = bp.cm()
x, y = coords[:, 0], coords[:, 1]
regions = xy_regions(coords, mo, L)
locs = location(H9K.R3 * x, y, mo)
active = ((locs != BaryLoc.EXT) & (locs != BaryLoc.UDF) & (H9C.mode[regions[:, -2]] == 1))
idx = np.flatnonzero(active)
nbr, c2 = region_neighbours(regions[idx])
hopped = regions[idx, 0] != nbr[:, 0]

# which of these are the diverging ones? cross-check needs C; approximate by: the
# diverging set ~ active-folded cells whose terminal body wing in {6,7,8}.
print(f" total points: {n}; active(folded): {len(idx)}; hopped(octant-span): {int(hopped.sum())}")
print(f" hopped fraction of folded: {hopped.mean():.4f}")
# terminal body digit of the FULL uuid at nibble L
full_nibs = _batch_int_to_nibbles([u.int for u in uuids], n=32)
wing = np.isin(full_nibs[:, L], [6, 7, 8])
print(f" full-uuid nibble[{L}] in wing(6/7/8): {int(wing.sum())}")
# of the active folded cells, how many are wing at terminus, and of those how many hop
act_wing = wing[idx]
print(f" active & wing: {int(act_wing.sum())}; (active&wing)&hopped: {int((act_wing & hopped).sum())}")
