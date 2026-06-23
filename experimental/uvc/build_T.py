"""
Build the step table T[phase, leaf, c2, dir, r_mo] -> (leaf', c2', phase', carry).

Direction is labelled by planar (b_oct) bearing: the 6 neighbours of a hex sit at
~60-degree spacing; per octant we estimate the lattice offset and bin into dir 0..5.
carry = the parent body digit changes (the leaf step crossed a parent boundary).

Validation:
  1. T determinism  — same (phase,leaf,c2,dir,r_mo) always gives same output.
  2. period-9       — walking dir 0 through same-octant neighbours cycles the leaf
                      digit with period 9 (the known 860671782-style cycles).
"""
import sys
import math
import numpy as np
from collections import defaultdict
from l1_traverse_testbed import build_cells
from g_extract import assign_phases

LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 2
DENS = int(sys.argv[2]) if len(sys.argv) > 2 else 900

cells, _ = build_cells(LAYER, DENS)
assign_phases(cells)

# --- per-octant tangent frame (each octant is a flat octahedron face) ---
# b_oct planar coords are skewed, so bearings there don't sit at 60deg. Instead
# build a 2D basis on each face plane from the 3D centroids and measure true angles.
oc_xyz = defaultdict(list)
for r in cells:
    oc_xyz[r['oct']].append(r['xyz'])
frame = {}
for oc, vs in oc_xyz.items():
    n = np.mean(vs, axis=0); n /= np.linalg.norm(n)
    ref = np.array([0, 0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0, 0])
    e1 = ref - n * (ref @ n); e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    frame[oc] = (e1, e2)


def bearing(r, c):
    e1, e2 = frame[r['oct']]
    d = c['xyz'] - r['xyz']
    return math.atan2(d @ e2, d @ e1)


# 6-fold lattice offset per octant
by_oct = defaultdict(list)
for r in cells:
    if not r['interior']:
        continue
    for j in r['ring']:
        if cells[j]['oct'] == r['oct'] and cells[j]['interior']:
            by_oct[r['oct']].append(bearing(r, cells[j]))
offset = {oc: np.angle(np.mean(np.exp(1j * 6 * np.array(bs)))) / 6 for oc, bs in by_oct.items()}


def direction(r, c):
    return int(round((bearing(r, c) - offset[r['oct']]) / (math.pi / 3))) % 6


# --- build T (only from cells whose 6 neighbours snap to 6 DISTINCT directions) ---
T = defaultdict(set)
n_clean = n_ambig = 0
for r in cells:
    if not r['interior']:
        continue
    nbrs = [cells[j] for j in r['ring']
            if cells[j]['oct'] == r['oct'] and cells[j]['interior']]
    if len(nbrs) != 6:
        continue
    drs = [direction(r, c) for c in nbrs]
    if sorted(drs) != [0, 1, 2, 3, 4, 5]:     # distortion made two neighbours share a bin
        n_ambig += 1
        continue
    n_clean += 1
    for c, dr in zip(nbrs, drs):
        carry = c['body'][:-1] != r['body'][:-1]
        T[(r['phase'], r['leaf'], r['c2'], dr, r['r_mo'])].add(
            (c['leaf'], c['c2'], c['phase'], carry))

det = sum(len(v) == 1 for v in T.values())
print(f"\nclean cells: {n_clean}  (dropped {n_ambig} dir-ambiguous)")
print(f"T determinism: {det}/{len(T)}")
if det < len(T):
    shown = 0
    for k, v in T.items():
        if len(v) > 1 and shown < 6:
            print(f"   ({k}) -> {sorted(v)}")
            shown += 1

# --- period-9 walk: follow dir 0 through same-octant neighbours ---
print("\nperiod-9 walk (dir 0, same-octant), leaf-digit sequence per start cell:")
idx_by_body = {(r['oct'], *r['body']): i for i, r in enumerate(cells)}


def step_geo(i, want_dir):
    r = cells[i]
    if not r['interior']:
        return None
    for j in r['ring']:
        if cells[j]['oct'] == r['oct'] and direction(r, cells[j]) == want_dir:
            return j
    return None


starts = [i for i, r in enumerate(cells) if r['interior']][:6]
for i0 in starts:
    seq = []
    i = i0
    for _ in range(20):
        if i is None:
            break
        seq.append(cells[i]['leaf'])
        i = step_geo(i, 0)
    s = ''.join(str(d) for d in seq)
    per9 = len(seq) >= 18 and s[:9] == s[9:18]
    print(f"  oct{cells[i0]['oct']} {tuple(cells[i0]['body'])}: {s}  "
          f"{'period-9 OK' if per9 else '(walk left octant / short)'}")
