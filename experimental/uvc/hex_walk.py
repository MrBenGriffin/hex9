"""
Hexagon came-from walk: state = (cell, came_from ring-position). To go straight,
exit the OPPOSITE edge (came_from + 3) mod 6. No global frame — only each cell's
cyclic neighbour order (robust to projection distortion) + the +3 opposite rule.

Validates: straight walk does NOT bounce and the leaf hex digit cycles period-9.
Also prints the digit-neighbour view: "my digit -d-> neighbour digit".
"""
import sys
import math
import numpy as np
from l1_traverse_testbed import build_cells

LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 2
DENS = int(sys.argv[2]) if len(sys.argv) > 2 else 900

cells, _ = build_cells(LAYER, DENS)

# per-octant tangent frame (each octant is a flat face) for cyclic ordering
oc_xyz = {}
for r in cells:
    oc_xyz.setdefault(r['oct'], []).append(r['xyz'])
frame = {}
for oc, vs in oc_xyz.items():
    n = np.mean(vs, axis=0); n /= np.linalg.norm(n)
    ref = np.array([0, 0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0, 0])
    e1 = ref - n * (ref @ n); e1 /= np.linalg.norm(e1)
    frame[oc] = (e1, np.cross(n, e1))


def ordered_ring(i):
    """same-octant interior neighbours of cell i, sorted cyclically by bearing."""
    r = cells[i]; e1, e2 = frame[r['oct']]
    out = []
    for j in r['ring']:
        c = cells[j]
        if c['oct'] != r['oct'] or not c['interior']:
            continue
        d = c['xyz'] - r['xyz']
        out.append((math.atan2(d @ e2, d @ e1), j))
    out.sort()
    return [j for _, j in out]


def came_from_pos(cur, prev_ring_len, prev):
    """ring position of `prev` in `cur`'s ordered ring (the edge we entered by)."""
    ring = ordered_ring(cur)
    return ring.index(prev) if prev in ring else None


print(f"\nhexagon came-from walk (exit = came_from + 3), leaf-digit sequence:")
starts = [i for i, r in enumerate(cells) if r['interior'] and len(ordered_ring(i)) == 6][:4]
for i0 in starts:
    ring0 = ordered_ring(i0)
    prev, cur = i0, ring0[0]          # first move: exit edge 0
    seq = [cells[i0]['leaf'], cells[cur]['leaf']]
    digit_steps = [(cells[i0]['leaf'], cells[cur]['leaf'])]
    ok = True
    for _ in range(18):
        cf = came_from_pos(cur, 6, prev)
        ring = ordered_ring(cur)
        if cf is None or len(ring) != 6:
            ok = False; break
        nxt = ring[(cf + 3) % 6]      # opposite edge = go straight
        digit_steps.append((cells[cur]['leaf'], cells[nxt]['leaf']))
        prev, cur = cur, nxt
        seq.append(cells[cur]['leaf'])
    s = ''.join(str(d) for d in seq)
    # period-9 (ignore the seed digit): check a 9-window repeats
    body = s[1:]
    per9 = len(body) >= 18 and body[:9] == body[9:18]
    tag = 'PERIOD-9' if per9 else ('left-octant' if not ok else 'check')
    print(f"  oct{cells[i0]['oct']} start-digit {cells[i0]['leaf']}: {s:<24} [{tag}]")
