"""Traversal-grammar (SFC) search on the canonical Hex9 d-cell 9-dissection.

Question: does there exist a base endpoint pair (A,B) on the half-hex trapezoid,
an ordering of the 9 children, and a per-child variant, such that
  entry(child_0) = parent A,  exit(child_i) = entry(child_{i+1}),  exit(child_8) = parent B,
where each child's curve is the (recursively identical) base curve carried into the
child by its placement isometry?

Variant regimes:
  FWD      : forward only                          (digit order = curve order)
  FWD+REV  : traversal direction may flip          (overlay on hex9 as-is)
  +MIRROR  : Hilbert-style mirrored variants       (requires mirror dissection in
             sigma-children at the next level => a *twin* hierarchy, not an overlay,
             because the two 9-dissections are a chiral mirror pair)

Geometry from experimental/halfhex.py: 27 triangle cells (x,y), y in 0..5,
even y = down-pointing, odd y = up-pointing, ref order sorted by (y,x).
Solution strings are the two machine-verified mirror-pair tilings.

Coordinates: triangle-grid points (X half-units on line r) -> Eisenstein (a,b):
  a=(X-r)//2, b=r; |a e1 + b e2|^2 = a^2+a b+b^2 (y-down basis).
rot60 rho(a,b)=(-b,a+b); mirror sigma(a,b)=(b,a).
"""

from collections import Counter
from itertools import product

# ---------- ground + tilings (verbatim from experimental/halfhex.py) ----------
REMOVED = {(3, 5), (4, 3), (4, 4), (4, 5), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5)}
GROUND = set((x, y) for y in range(6) for x in range(6)) - REMOVED
REF = sorted(GROUND, key=lambda e: (e[1], e[0]))
LIB = {
    "0": {(0, 0), (0, 1), (0, 2)},
    "1": {(0, 1), (1, 0), (1, 1)},
    "2": {(0, 1), (1, 0), (0, 2)},
    "3": {(0, 1), (0, 2), (0, 3)},
    "4": {(0, 0), (0, 1), (1, 0)},
    "5": {(1, 1), (1, 2), (0, 3)},
}
TILINGS = {
    "T1": "044044043040230225322512511",
    "T2": "442442425420250205300130113",
}

# ---------- lattice helpers ----------
def rho(p):
    a, b = p
    return (-b, a + b)

def rot(p, j):
    for _ in range(j % 6):
        p = rho(p)
    return p

def add(p, q): return (p[0] + q[0], p[1] + q[1])
def sub(p, q): return (p[0] - q[0], p[1] - q[1])

def tri_verts(cell):
    """Triangle cell (x,y) -> 3 vertices in (a,b)."""
    x, y = cell
    r = y // 2
    if y % 2 == 0:   # down-pointing
        pts = [(r + 2 * x, r), (r + 2 * x + 2, r), (r + 2 * x + 1, r + 1)]
    else:            # up-pointing
        pts = [(r + 2 * x + 1, r + 1), (r + 2 * x + 3, r + 1), (r + 2 * x + 2, r)]
    return [((X - rr) // 2, rr) for (X, rr) in pts]

# ---------- reconstruct the 9 pieces from a solution string ----------
def reconstruct(sol):
    labels = {c: sol[i] for i, c in enumerate(REF)}
    placements = set()
    for k, shape in LIB.items():
        for oy in range(-8, 9, 2):
            for ox in range(-8, 9):
                cells = frozenset((px + ox, py + oy) for (px, py) in shape)
                if cells <= GROUND and all(labels[c] == k for c in cells):
                    placements.add((k, cells))
    placements = sorted(placements)

    solutions = []
    def cover(uncovered, chosen):
        if not uncovered:
            solutions.append(list(chosen))
            return
        cell = min(uncovered, key=lambda e: (e[1], e[0]))
        for k, cells in placements:
            if cell in cells and cells <= uncovered:
                chosen.append((k, cells))
                cover(uncovered - cells, chosen)
                chosen.pop()
    cover(frozenset(GROUND), [])
    assert len(solutions) == 1, f"digit string does not uniquely define tiling: {len(solutions)}"
    return solutions[0]

# ---------- label each piece's vertices by rotation-only template match ----------
# child-scale template, labels 0..3 = P0(long-L) P1(long-R) P2(short-R) P3(short-L), 4 = M
TEMPLATE = {0: (0, 0), 1: (2, 0), 2: (1, 1), 3: (0, 1), 4: (1, 0)}
SIGMA_LAB = {0: 1, 1: 0, 2: 3, 3: 2, 4: 4}
# parent labels (long edge 6 on line 0, short edge on line 3)
PARENT = {0: (0, 0), 1: (6, 0), 2: (3, 3), 3: (0, 3), 4: (3, 0)}

def label_piece(cells):
    cnt = Counter()
    tris = [tri_verts(c) for c in cells]
    for t in tris:
        cnt.update(t)
    m = [v for v, n in cnt.items() if n == 3]
    assert len(m) == 1, cnt
    m = m[0]
    vset = set(cnt)
    for j in range(6):
        t = sub(m, rot(TEMPLATE[4], j))
        mapped = {lab: add(rot(v, j), t) for lab, v in TEMPLATE.items()}
        if set(mapped.values()) == vset:
            return j, mapped
    raise AssertionError("no rotation match (should be impossible for achiral tile)")

def boundary_edges(cells):
    e = Counter()
    for c in cells:
        v = tri_verts(c)
        for i in range(3):
            e[frozenset((tuple(v[i]), tuple(v[(i + 1) % 3])))] += 1
    return {k for k, n in e.items() if n == 1}

# ---------- grammar search ----------
def variants(labels, A, B, allow_rev, allow_mirror):
    out = []
    out.append(("f", labels[A], labels[B]))
    if allow_rev:
        out.append(("r", labels[B], labels[A]))
    if allow_mirror:
        out.append(("mf", labels[SIGMA_LAB[A]], labels[SIGMA_LAB[B]]))
        if allow_rev:
            out.append(("mr", labels[SIGMA_LAB[B]], labels[SIGMA_LAB[A]]))
    # dedupe identical (entry,exit) keeping first tag
    seen, ded = set(), []
    for tag, e, x in out:
        if (e, x) not in seen:
            seen.add((e, x))
            ded.append((tag, e, x))
    return ded

def search(pieces, A, B, allow_rev, allow_mirror, cap=100000):
    labs = [p["labels"] for p in pieces]
    var = [variants(l, A, B, allow_rev, allow_mirror) for l in labs]
    start, goal = PARENT[A], PARENT[B]
    sols = []
    n = len(pieces)
    def dfs(cur, used, path):
        if len(sols) >= cap:
            return
        if len(path) == n:
            if cur == goal:
                sols.append(list(path))
            return
        for i in range(n):
            if used >> i & 1:
                continue
            for tag, e, x in var[i]:
                if e == cur:
                    dfs(x, used | 1 << i, path + [(i, tag)])
    dfs(start, 0, [])
    return sols

def edge_adjacent_chain(sol, pieces):
    for (i, _), (j, _) in zip(sol, sol[1:]):
        if not (pieces[i]["bedges"] & pieces[j]["bedges"]):
            return False
    return True

def main():
    lab_names = {0: "P0", 1: "P1", 2: "P2", 3: "P3", 4: "M"}
    for tname, sol in TILINGS.items():
        raw = reconstruct(sol)
        pieces = []
        for k, cells in raw:
            j, labels = label_piece(cells)
            pieces.append({"orient": k, "cells": cells, "rot": j,
                           "labels": labels, "bedges": boundary_edges(cells)})
        # sibling edge-adjacency graph + plain Hamiltonicity (context)
        adj = {i: set() for i in range(9)}
        for i in range(9):
            for j in range(i + 1, 9):
                if pieces[i]["bedges"] & pieces[j]["bedges"]:
                    adj[i].add(j); adj[j].add(i)
        def ham(path, used):
            if len(path) == 9:
                return True
            return any(ham(path + [n], used | {n})
                       for n in adj[path[-1]] if n not in used)
        any_ham = any(ham([s], {s}) for s in range(9))
        print(f"\n=== {tname} ===  sibling adjacency degrees: "
              f"{sorted(len(v) for v in adj.values())}  plain-Hamiltonian-path: {any_ham}")

        for regime, (rev, mir) in {"FWD": (False, False),
                                   "FWD+REV": (True, False),
                                   "FWD+REV+MIRROR": (True, True)}.items():
            total = {}
            for A, B in product(range(5), repeat=2):
                if A == B:
                    continue
                sols = search(pieces, A, B, rev, mir)
                if sols:
                    ea = sum(edge_adjacent_chain(s, pieces) for s in sols)
                    total[(A, B)] = (len(sols), ea, sols[0])
            print(f"  [{regime}]")
            if not total:
                print("    -- no grammar exists (exhausted all endpoint pairs) --")
            for (A, B), (n, ea, ex) in sorted(total.items()):
                seq = " ".join(f"{i}{'' if t=='f' else t}" for i, t in ex)
                print(f"    {lab_names[A]}->{lab_names[B]}: {n} grammars "
                      f"({ea} fully edge-adjacent)   e.g. {seq}")

if __name__ == "__main__":
    main()
