"""
How much walk history does a straight (came-from / opposite-edge) step need?

Walk straight from every interior cell, collect the leaf-digit stream, then test:
  order-k : do the last k digits determine the next digit?
k where it hits 100% = the prior Ben needs (bounded => it's the carried state).
Purely digit-space, c2 not used.
"""
import sys
import math
import numpy as np
from collections import defaultdict
from l1_traverse_testbed import build_cells

LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 3
DENS = int(sys.argv[2]) if len(sys.argv) > 2 else 3000

cells, _ = build_cells(LAYER, DENS)

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


def straight_walk(i0, maxlen=30):
    ring0 = ordered_ring(i0)
    if len(ring0) != 6:
        return []
    prev, cur = i0, ring0[0]
    seq = [cells[i0]['leaf'], cells[cur]['leaf']]
    for _ in range(maxlen):
        ring = ordered_ring(cur)
        if len(ring) != 6 or prev not in ring:
            break
        nxt = ring[(ring.index(prev) + 3) % 6]
        seq.append(cells[nxt]['leaf'])
        prev, cur = cur, nxt
    return seq


seqs = [straight_walk(i) for i, r in enumerate(cells) if r['interior']]
seqs = [s for s in seqs if len(s) >= 6]
print(f"\n{len(seqs)} straight walks collected (avg len {np.mean([len(s) for s in seqs]):.1f})")

for k in range(1, 5):
    table = defaultdict(set)
    for s in seqs:
        for t in range(k, len(s)):
            table[tuple(s[t - k:t])].add(s[t])
    det = sum(len(v) == 1 for v in table.values())
    print(f"  order-{k}  (last {k} digits -> next): {det}/{len(table)} deterministic")

# Ben's claim: order-1 prior (came_from,current)->next suffices EXCEPT current in {0,1}.
t2 = defaultdict(set)     # (came_from, current) -> next
t3 = defaultdict(set)     # (2back, came_from, current) -> next
for s in seqs:
    for t in range(2, len(s)):
        t2[(s[t - 2], s[t - 1])].add(s[t])
    for t in range(3, len(s)):
        t3[(s[t - 3], s[t - 2], s[t - 1])].add(s[t])

print("\nper current-digit: is (came_from, current) -> next deterministic?")
for cur in range(9):
    keys = [k for k in t2 if k[1] == cur]
    amb = [k for k in keys if len(t2[k]) > 1]
    note = '' if not amb else '   AMBIG came_from=' + str(sorted(k[0] for k in amb))
    print(f"  current {cur}: {len(keys)-len(amb)}/{len(keys)} det{note}")

print("\nfor current in {0,1}: does adding the 2-back digit (order-3) resolve it?")
for cur in (0, 1):
    keys = [k for k in t3 if k[2] == cur]
    amb = [k for k in keys if len(t3[k]) > 1]
    print(f"  current {cur}: order-3 {len(keys)-len(amb)}/{len(keys)} det"
          + ('' if not amb else f'  still-ambig {[k for k in amb][:5]}'))
