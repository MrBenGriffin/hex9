"""
Directional digit transition (Ben's came-from / opposite-edge, in DIGIT space).

state to go straight = (came_from_digit, current_digit [, phase]).  next = the
OPPOSITE-edge neighbour's digit (came_from ring position + 3).  c2 is NOT used
(testing Ben's "entirely c2-independent").

Tests:
  T0[(came_from, current)]        -> next   : ambiguous? (Ben: yes, e.g. (0<-0)->{3,4})
  T1[(came_from, current, phase)] -> next   : deterministic? (phase = the 2-bit prior)
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
assign_phases(cells)                      # r['phase'] in 0..3 (the hidden state)

# per-octant tangent frame for cyclic ordering (robust; came-from is relative)
oc_xyz = defaultdict(list)
for r in cells:
    oc_xyz[r['oct']].append(r['xyz'])
frame = {}
for oc, vs in oc_xyz.items():
    n = np.mean(vs, axis=0); n /= np.linalg.norm(n)
    ref = np.array([0, 0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0, 0])
    e1 = ref - n * (ref @ n); e1 /= np.linalg.norm(e1)
    frame[oc] = (e1, np.cross(n, e1))


def ordered_ring(i):
    r = cells[i]; e1, e2 = frame[r['oct']]
    out = []
    for j in r['ring']:
        c = cells[j]
        if c['oct'] == r['oct'] and c['interior']:
            d = c['xyz'] - r['xyz']
            out.append((math.atan2(d @ e2, d @ e1), j))
    out.sort()
    return [j for _, j in out]


T0 = defaultdict(set)     # (came_from_digit, current_digit) -> next_digit
T1 = defaultdict(set)     # (came_from_digit, current_digit, phase) -> next_digit
for i, r in enumerate(cells):
    if not r['interior']:
        continue
    ring = ordered_ring(i)
    if len(ring) != 6:
        continue
    cur = r['leaf']
    for p in range(6):
        came = cells[ring[p]]['leaf']
        nxt = cells[ring[(p + 3) % 6]]['leaf']     # opposite edge = straight
        T0[(came, cur)].add(nxt)
        T1[(came, cur, r['phase'])].add(nxt)

d0 = sum(len(v) == 1 for v in T0.values())
d1 = sum(len(v) == 1 for v in T1.values())
print(f"\nT0 (came_from, current)         deterministic: {d0}/{len(T0)}")
print(f"T1 (came_from, current, phase)  deterministic: {d1}/{len(T1)}")

print("\nBen's case  (came_from=0, current=0):")
print(f"  without phase -> next in {sorted(T0[(0, 0)])}   (ambiguous if >1)")
for ph in range(4):
    if (0, 0, ph) in T1:
        print(f"  phase {ph}      -> next in {sorted(T1[(0, 0, ph)])}")
