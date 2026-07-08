"""Find the minimal closed multi-type curve system on the Hex9 d-cell
dissection (FWD regime: every child traversed forward => digit order can be
made equal to curve order via a finite-state relabelling), then expand the
recursion to depth 3 with exact rational arithmetic, machine-verify
continuity + coverage, and emit an SVG figure.
"""

import os
from fractions import Fraction
from itertools import combinations, product
from math import sqrt

from sfc_grammar import (TILINGS, PARENT, TEMPLATE, reconstruct, label_piece,
                         boundary_edges, rot, add, sub)

LABELS = [0, 1, 2, 3, 4]
NAME = {0: "P0", 1: "P1", 2: "P2", 3: "P3", 4: "M"}


def build_pieces(tname):
    pieces = []
    for k, cells in reconstruct(TILINGS[tname]):
        j, labels = label_piece(cells)
        pieces.append({"orient": k, "rot": j, "labels": labels,
                       "cells": cells, "bedges": boundary_edges(cells)})
    return pieces


def find_production(tau, pieces, allowed):
    """One FWD production for type tau using only 'allowed' types.
    Returns ordered list of (child_index, child_type) or None.
    Prefers productions whose consecutive children share a full edge."""
    p, q = tau
    start, goal = PARENT[p], PARENT[q]
    best = []

    def dfs(cur, used, path, ea):
        if best and best[0][1] >= ea + (9 - len(path)):
            pass  # can't beat best on edge-adjacency; still allow first-found fill
        if len(path) == 9:
            if cur == goal:
                best.append((list(path), ea))
                best.sort(key=lambda t: -t[1])
                del best[1:]
            return
        for i in range(9):
            if used >> i & 1:
                continue
            for u in allowed:
                if pieces[i]["labels"][u[0]] == cur:
                    step_ea = 1
                    if path:
                        prev = path[-1][0]
                        step_ea = 1 if (pieces[i]["bedges"] & pieces[prev]["bedges"]) else 0
                    dfs(pieces[i]["labels"][u[1]], used | 1 << i,
                        path + [(i, u)], ea + step_ea)
    dfs(start, 0, [], 0)
    if not best:
        return None
    return best[0]


_memo = {}

def fast_realizable(tau, pieces, allowed):
    key = (tau, allowed)
    if key in _memo:
        return _memo[key]
    p, q = tau
    start, goal = PARENT[p], PARENT[q]
    # per child: dedup (entry, exit) coord pairs available under 'allowed'
    opts = []
    for pc in pieces:
        lab = pc["labels"]
        opts.append(list({(lab[u[0]], lab[u[1]]) for u in allowed}))
    found = False
    def dfs(cur, used, k):
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
                if e == cur:
                    dfs(x, used | 1 << i, k + 1)
                    if found:
                        return
    dfs(start, 0, 0)
    _memo[key] = found
    return found


def gfp(pieces, types):
    viable = set(types)
    while True:
        dead = {t for t in viable
                if not fast_realizable(t, pieces, frozenset(viable))}
        if not dead:
            return viable
        viable -= dead


def shrink(pieces, start_set, order_seed):
    """Greedy irreducible closed subsystem: delete a type, re-GFP, keep going."""
    cur = set(start_set)
    changed = True
    while changed:
        changed = False
        order = sorted(cur, key=lambda t: ((t[0] * 7 + t[1] * 3 + order_seed) % 11, t))
        for t in order:
            rest = gfp(pieces, cur - {t})
            if rest and len(rest) < len(cur):
                cur = rest
                changed = True
                break
    return cur


def minimal_systems(pieces, exhaustive_below=None):
    all_types = [t for t in product(LABELS, repeat=2)]
    surviving = gfp(pieces, all_types)
    print(f"GFP-surviving FWD types: {len(surviving)}")
    best = None
    for seed in range(12):
        sysm = shrink(pieces, surviving, seed)
        if best is None or len(sysm) < len(best):
            best = sysm
    print(f"greedy irreducible closed system: size {len(best)}")
    # exhaustively confirm nothing smaller exists (sizes 6..len-1); <=5 already ruled out
    lo = 6
    hi = len(best) - 1
    if exhaustive_below and hi >= lo:
        for s in range(lo, hi + 1):
            found = None
            for combo in combinations(sorted(surviving), s):
                fz = frozenset(combo)
                if all(fast_realizable(t, pieces, fz) for t in combo):
                    found = combo
                    break
            print(f"  exhaustive size {s}: {'FOUND' if found else 'none'}")
            if found:
                best = set(found)
                break
    combo = tuple(sorted(best))
    prods = {t: find_production(t, pieces, set(combo)) for t in combo}
    loops = sum(1 for (a, b) in combo if a == b)
    ea = sum(v[1] for v in prods.values())
    return [(len(combo), loops, -ea, combo, prods)]


# ---------- exact expansion + verification ----------
def piece_map(piece):
    """Affine map f(x) = R x/3 + t taking parent-frame coords into the child."""
    j = piece["rot"]
    t = sub(piece["labels"][4], rot((Fraction(PARENT[4][0], 3), Fraction(PARENT[4][1], 3)), j))
    def f(x):
        rx = rot((Fraction(x[0], 3), Fraction(x[1], 3)), j)
        return (rx[0] + t[0], rx[1] + t[1])
    return f


def expand(tau, prods, pieces, depth):
    """Return list of (cell_path, entry, exit, centroid) in curve order."""
    maps = [piece_map(p) for p in pieces]
    # trapezoid centroid, exact: average of triangle centroids weighted equally
    # (27 unit triangles -> just use vertex-mean of the 4 corners weighted by area split:
    #  compute via the three big triangles P0P1P3, P1P2P3 areas 2:1... do simple:
    #  centroid of trapezoid with parallel sides a=6 (line0), b=3 (line3):
    #  distance from long side = h*(2b+a)/(3(a+b)) = 3*(12)/(27)=4/3 -> b-coord 4/3;
    #  a-coord by symmetry axis: x centre... in (a,b) Eisenstein coords use numeric later.
    C = (Fraction(7, 3), Fraction(4, 3))  # a,b: symmetric axis a+b/2=3 -> a=3-2/3=7/3
    def rec(t, d, f_chain, path):
        if d == 0:
            e = f_chain(PARENT[t[0]])
            x = f_chain(PARENT[t[1]])
            c = f_chain(C)
            return [(tuple(path), e, x, c)]
        out = []
        for i, u in prods[t][0]:
            g = maps[i]
            out.extend(rec(u, d - 1, lambda p, f=f_chain, g=g: f(g(p)), path + [i]))
        return out
    return rec(tau, depth, lambda p: p, [])


def verify(seq, tau, depth):
    n = len(seq)
    assert n == 9 ** depth, n
    assert len({s[0] for s in seq}) == n, "cells not distinct"
    assert seq[0][1] == PARENT[tau[0]], "start endpoint wrong"
    assert seq[-1][2] == PARENT[tau[1]], "end endpoint wrong"
    for a, b in zip(seq, seq[1:]):
        assert a[2] == b[1], f"discontinuity between {a[0]} and {b[0]}"
    return True


# ---------- rendering ----------
def xy(p):
    a, b = float(p[0]), float(p[1])
    return (a + b / 2.0, b * sqrt(3) / 2.0)


def svg_figure(pieces, tau, prods, path_out):
    S = 92.0
    pad = 30.0
    W = 6 * S + 2 * pad
    panels = []
    for d in (1, 2, 3):
        seq = expand(tau, prods, pieces, d)
        pts = [xy(c) for _, _, _, c in seq]
        start = xy(seq[0][1]); end = xy(seq[-1][2])
        panels.append((d, pts, start, end))
    H = (3 * sqrt(3) / 2 * S + 90) * 3 + pad
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.0f}" height="{H:.0f}" '
           f'viewBox="0 0 {W:.0f} {H:.0f}" font-family="Helvetica,Arial">',
           f'<rect width="100%" height="100%" fill="white"/>']
    outline = [(0, 0), (6, 0), (Fraction(3), Fraction(3)), (0, 3)]
    oxy = [xy(p) for p in outline]
    y0 = pad
    for d, pts, start, end in panels:
        out.append(f'<g transform="translate({pad},{y0})">')
        # tile outline
        o = " ".join(f"{x*S:.2f},{y*S:.2f}" for x, y in oxy)
        out.append(f'<polygon points="{o}" fill="#FFF7E0" stroke="#555" stroke-width="1.2"/>')
        # child boundaries (level-1 pieces) for context
        for pc in pieces:
            vs = {}
            for e in pc["bedges"]:
                for v in e:
                    vs.setdefault(v, [])
            segs = [tuple(e) for e in pc["bedges"]]
            for (v1, v2) in segs:
                x1, y1 = xy(v1); x2, y2 = xy(v2)
                out.append(f'<line x1="{x1*S:.2f}" y1="{y1*S:.2f}" x2="{x2*S:.2f}" '
                           f'y2="{y2*S:.2f}" stroke="#BBBBBB" stroke-width="0.8"/>')
        # the curve
        p = " ".join(f"{x*S:.2f},{y*S:.2f}" for x, y in pts)
        full = [f"{start[0]*S:.2f},{start[1]*S:.2f}", p,
                f"{end[0]*S:.2f},{end[1]*S:.2f}"]
        out.append(f'<polyline points="{" ".join(full)}" fill="none" '
                   f'stroke="#004488" stroke-width="{2.4 if d < 3 else 1.2}" '
                   f'stroke-linejoin="round"/>')
        out.append(f'<circle cx="{start[0]*S:.2f}" cy="{start[1]*S:.2f}" r="4" fill="#DDAA33"/>')
        out.append(f'<circle cx="{end[0]*S:.2f}" cy="{end[1]*S:.2f}" r="4" fill="#BB5566"/>')
        out.append(f'<text x="{6*S/2:.0f}" y="{3*sqrt(3)/2*S+34:.0f}" text-anchor="middle" '
                   f'font-size="15" fill="#222">level {d} — {9**d} d-cells</text>')
        out.append('</g>')
        y0 += 3 * sqrt(3) / 2 * S + 90
    out.append('</svg>')
    with open(path_out, "w") as f:
        f.write("\n".join(out))


if __name__ == "__main__":
    pieces = build_pieces("T1")
    systems = minimal_systems(pieces, exhaustive_below=True)
    s, loops, negea, combo, prods = systems[0]
    tau = (0, 1) if (0, 1) in combo else combo[0]
    print(f"\nchosen system: {[f'{NAME[a]}->{NAME[b]}' for a,b in combo]}, top type {NAME[tau[0]]}->{NAME[tau[1]]}")
    for t, (pr, ea) in prods.items():
        print(f"  {NAME[t[0]]}->{NAME[t[1]]}  ::=  " +
              " ".join(f"c{i}[{NAME[u[0]]}{NAME[u[1]]}]" for i, u in pr) +
              f"   (edge-adjacent steps {ea}/8)")
    for d in (1, 2, 3):
        seq = expand(tau, prods, pieces, d)
        verify(seq, tau, d)
        print(f"depth {d}: {len(seq)} cells, continuity + coverage + endpoints VERIFIED")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "h9_sfc.svg")
    svg_figure(pieces, tau, prods, out)
    print(f"figure: {out}")
