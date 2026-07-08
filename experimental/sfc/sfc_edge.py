"""Face-continuous (edge-to-edge) SFC search on the canonical T1 d-cell
dissection, named-point endpoints, FWD regime (canonical-tree overlay).

Constraint added to the grammar: at every handoff between consecutive
children, the two cells must share a boundary edge and the handoff point
must be an endpoint of one of those shared unit edges — no corner-only
hops. Pipeline: GFP -> greedy shrink -> exhaustive minimality -> exact
expansion -> direct face-continuity check at depths 1..4 (consecutive
cells must share a positive-length boundary segment).
"""

from fractions import Fraction
from itertools import combinations, product
import sys

sys.path.insert(0, "/Users/ben/Documents/Projects/PyCharm/hex9/experimental/sfc")
from sfc_grammar import (TILINGS, PARENT, TEMPLATE, reconstruct, label_piece,
                         boundary_edges, rot)

F = Fraction
LABELS = [0, 1, 2, 3, 4]
NAME = {0: "P0", 1: "P1", 2: "P2", 3: "P3", 4: "M"}


def build_pieces():
    pieces = []
    for k, cells in reconstruct(TILINGS["T1"]):
        j, labels = label_piece(cells)
        pieces.append({"orient": k, "rot": j, "labels": labels,
                       "bedges": boundary_edges(cells)})
    return pieces


def edge_ok(pieces, i, j, p):
    """Handoff point p between pieces i -> j sits on a shared unit edge."""
    for e in pieces[i]["bedges"] & pieces[j]["bedges"]:
        if p in e:
            return True
    return False


_memo = {}

def realizable(pieces, tau, allowed):
    key = (tau, allowed)
    if key in _memo:
        return _memo[key]
    p, q = tau
    start, goal = PARENT[p], PARENT[q]
    opts = []
    for pc in pieces:
        lab = pc["labels"]
        opts.append(list({(lab[u[0]], lab[u[1]]) for u in allowed}))
    found = False
    def dfs(cur, used, k, prev):
        nonlocal found
        if found:
            return
        if k == 9:
            found = cur == goal
            return
        for i in range(9):
            if used >> i & 1:
                continue
            for e, x in opts[i]:
                if e == cur and (prev is None or edge_ok(pieces, prev, i, cur)):
                    dfs(x, used | 1 << i, k + 1, i)
                    if found:
                        return
    dfs(start, 0, 0, None)
    _memo[key] = found
    return found


def gfp(pieces, types):
    viable = set(types)
    while True:
        dead = {t for t in viable
                if not realizable(pieces, t, frozenset(viable))}
        if not dead:
            return viable
        viable -= dead


def find_production(pieces, tau, allowed):
    p, q = tau
    start, goal = PARENT[p], PARENT[q]
    out = []
    def dfs(cur, used, k, prev, path):
        if out:
            return
        if k == 9:
            if cur == goal:
                out.append(path[:])
            return
        for i in range(9):
            if used >> i & 1:
                continue
            for u in allowed:
                lab = pieces[i]["labels"]
                if lab[u[0]] == cur and (prev is None or
                                         edge_ok(pieces, prev, i, cur)):
                    dfs(lab[u[1]], used | 1 << i, k + 1, i, path + [(i, u)])
                    if out:
                        return
    dfs(start, 0, 0, None, [])
    return out[0] if out else None


def shrink(pieces, start_set, seed):
    cur = set(start_set)
    changed = True
    while changed:
        changed = False
        for t in sorted(cur, key=lambda t: ((t[0]*7 + t[1]*3 + seed) % 11, t)):
            rest = gfp(pieces, cur - {t})
            if rest and len(rest) < len(cur):
                cur = rest
                changed = True
                break
    return cur


# ---------- exact expansion + face-continuity ----------
def piece_map(piece):
    j = piece["rot"]
    m0 = rot((F(PARENT[4][0], 3), F(PARENT[4][1], 3)), j)
    t = (piece["labels"][4][0] - m0[0], piece["labels"][4][1] - m0[1])
    def f(x):
        rx = rot((F(x[0], 3), F(x[1], 3)), j)
        return (rx[0] + t[0], rx[1] + t[1])
    return f


CORNERS = [(0, 0), (6, 0), (3, 3), (0, 3)]


def expand(tau, prods, pieces, depth):
    maps = [piece_map(p) for p in pieces]
    def rec(t, d, chain):
        if d == 0:
            def g(pt):
                for f in reversed(chain):
                    pt = f(pt)
                return (F(pt[0]), F(pt[1]))
            e = g(PARENT[t[0]]); x = g(PARENT[t[1]])
            poly = [g(c) for c in CORNERS]
            return [(e, x, poly)]
        out = []
        for i, u in prods[t]:
            out.extend(rec(u, d - 1, chain + [maps[i]]))
        return out
    return rec(tau, depth, [])


def seg_overlap(a1, a2, b1, b2):
    """Positive-length overlap of collinear segments (exact)."""
    da = (a2[0]-a1[0], a2[1]-a1[1])
    db = (b2[0]-b1[0], b2[1]-b1[1])
    if da[0]*db[1] - da[1]*db[0] != 0:
        return False
    dc = (b1[0]-a1[0], b1[1]-a1[1])
    if da[0]*dc[1] - da[1]*dc[0] != 0:
        return False
    # project onto dominant axis of da
    def key(p):
        return p[0]*da[0] + p[1]*da[1]
    lo_a, hi_a = sorted((key(a1), key(a2)))
    lo_b, hi_b = sorted((key(b1), key(b2)))
    return min(hi_a, hi_b) > max(lo_a, lo_b)


def cells_share_edge(poly1, poly2):
    n = len(poly1)
    for i in range(n):
        for j in range(n):
            if seg_overlap(poly1[i], poly1[(i+1) % n],
                           poly2[j], poly2[(j+1) % n]):
                return True
    return False


def face_continuity(tau, prods, pieces, max_depth=4):
    for d in range(1, max_depth + 1):
        seq = expand(tau, prods, pieces, d)
        assert len(seq) == 9 ** d
        for (e1, x1, p1), (e2, x2, p2) in zip(seq, seq[1:]):
            assert x1 == e2, "discontinuity"
            if not cells_share_edge(p1, p2):
                return d
    return None  # face-continuous through max_depth


if __name__ == "__main__":
    pieces = build_pieces()
    all_types = [t for t in product(LABELS, repeat=2)]
    surv = gfp(pieces, all_types)
    print(f"edge-constrained GFP survivors: {len(surv)}: "
          f"{sorted((NAME[a], NAME[b]) for a, b in surv)}")
    if not surv:
        sys.exit()
    best = None
    for seed in range(12):
        s = shrink(pieces, surv, seed)
        if best is None or len(s) < len(best):
            best = s
    print(f"greedy irreducible: size {len(best)}")
    # exhaustive below
    minimum = best
    for size in range(1, len(best)):
        found = None
        for combo in combinations(sorted(surv), size):
            fz = frozenset(combo)
            if all(realizable(pieces, t, fz) for t in combo):
                found = fz
                break
        print(f"  exhaustive size {size}: {'FOUND' if found else 'none'}")
        if found:
            minimum = found
            break
    combo = tuple(sorted(minimum))
    prods = {t: find_production(pieces, t, set(combo)) for t in combo}
    print(f"\nminimal edge-constrained system, size {len(combo)}:")
    for t in combo:
        print(f"  {NAME[t[0]]}->{NAME[t[1]]}  ::=  " +
              " ".join(f"c{i}[{NAME[u[0]]}{NAME[u[1]]}]" for i, u in prods[t]))
    tau = combo[0]
    for t in combo:
        bad = face_continuity(t, prods, pieces, max_depth=4)
        tag = "face-continuous through depth 4" if bad is None else \
              f"CORNER HOP first appears at depth {bad}"
        print(f"  type {NAME[t[0]]}->{NAME[t[1]]}: {tag}")
