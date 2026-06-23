"""
Sharp vertex probe: what actually happens at the octahedron's 6 vertices?

Questions:
  1. The "twin" (chord~0) cells — are they CROSS-octant (same hexagon, two octant
     addresses = seam double-addressing) or genuine same-octant coincidences?
  2. True degree distribution of ALL cells (uncapped) — are there any degree!=6
     topological defects (pentagons/squares) at all?
  3. At the 6 octahedron vertices (axis-aligned: poles + 4 equatorial points),
     what is the local cell structure — degree, and how many octants meet there?
"""
import sys
import numpy as np
from collections import defaultdict
from scipy.spatial import cKDTree
from l1_traverse_testbed import build_cells

LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 2
DENS = int(sys.argv[2]) if len(sys.argv) > 2 else 900

cells, _ = build_cells(LAYER, DENS)
C = np.array([r['xyz'] for r in cells])
octs = np.array([r['oct'] for r in cells])
tree = cKDTree(C)

# --- 1. twin cells: cross-octant? ---
twins = [i for i, r in enumerate(cells) if r['nn_ratio'] < 0.15]
cross = same = 0
for i in twins:
    j = cells[i]['twin']
    if j is None:
        d, jj = tree.query(C[i], k=2); j = int(jj[1])
    if octs[j] != octs[i]:
        cross += 1
    else:
        same += 1
print(f"twin cells: {len(twins)}  -> cross-octant: {cross}, same-octant: {same}")
print("  (cross-octant twins = one hexagon with two octant addresses = seam double-address)")

# --- 2. true degree via GAP detection (first ring ends at a big distance jump) ---
s = np.array([r['spacing'] for r in cells])
dist, idx = tree.query(C, k=12)
degree = np.zeros(len(cells), dtype=int)
for i in range(len(cells)):
    nd = [dist[i, t] for t in range(1, idx.shape[1]) if dist[i, t] > 0.4 * s[i]]  # drop twins
    k = 1
    while k < len(nd) and nd[k] < 1.4 * nd[k - 1]:   # ring ends where spacing jumps >40%
        k += 1
    degree[i] = k
dd = defaultdict(int)
for v in degree:
    dd[int(v)] += 1
print(f"true degree distribution (gap-detected): {dict(sorted(dd.items()))}")
defects = [i for i in range(len(cells)) if degree[i] != 6]
print(f"non-hexagonal cells: {len(defects)}  "
      f"(degrees: {sorted(set(int(degree[i]) for i in defects))})")

# --- 3. the 6 octahedron vertices (axis-aligned guess) ---
verts = np.array([[0, 0, 1], [0, 0, -1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], float)
print("\nat each candidate octahedron vertex (nearest cells within ~1.5 spacings):")
med_s = float(np.median(s))
for v in verts:
    near = [i for i in range(len(cells)) if np.dot(C[i], v) > 1 - (1.5 * med_s) ** 2 / 2]
    if not near:
        d, i = tree.query(v, k=1); near = [int(i)]
    deg_here = sorted(degree[i] for i in near)
    octs_here = sorted({int(octs[i]) for i in near})
    lat = np.degrees(np.arcsin(v[2])); lon = np.degrees(np.arctan2(v[1], v[0]))
    print(f"  vertex @({lat:+.0f},{lon:+.0f}): {len(near)} cells, octants={octs_here}, degrees={deg_here}")
