# Part of the Hex9 (H9) Project
# Copyright ©2026, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""The fourth-letter conjecture, stage 1 (existence CSAT).

Matched pairs with THREE class digits is topologically obstructed on
the closed octahedron: any such labelling is locally a direction
3-colouring, and the cone-point holonomy (rot 240°) 3-cycles the
directions (transport note §4c correction, census 2026-08-03). Ben's
conjecture (2026-08-03): a FOURTH letter dissolves the obstruction —
digit 4 as the holonomy's escape valve, with the plane-interior scheme
untouched (cf. Tait / four-colour: on the sphere, 4 is where these
obstructions die).

This harness machine-checks existence on the REAL layer-L complex:

  * hosts enumerated globally (dense sphere sample -> h9_bin), their
    state-ordered rings via hhg9's own _anchor_hex_latlon;
  * one variable per fine edge child == one per coarse edge (the a4
    ledger: fine centres = coarse centres + edge midpoints), shared
    edges identified by 3D midpoint key;
  * one constraint per (host, half): the half's three edge children
    pairwise distinct. Half 0 covers ring edges (1,2),(2,3),(3,4)
    (the T0 corners are canonical ring indices 1..4), half 1 the rest.

FINDINGS (2026-08-03, run before reading further):

  * k = 3 is SAT — globally, on the closed octahedron, cone points
    included, at layer 1 AND layer 2 (CP-SAT proves it; solutions are
    perfectly balanced, 108x3 / 972x3; independently re-validated).
    The fourth letter is NOT needed for existence. The holonomy
    obstruction is real but narrower than first claimed: it forbids
    DIRECTION-EQUIVARIANT rules (any rule reading the digit off the
    child's geometric direction class, e.g. the state-frame rule in
    hhg9/h9/e4h.py — hence the 0%% cross-host census), not general
    labellings: the per-cell cut freedom (only two rainbow triples
    per hexagon, abc|acb allowed) breaks the direction-field rigidity
    the 3-cycle argument needs.
  * What remains OPEN is Ben's sharper question: a HIERARCHICALLY
    STABLE / TRANSITIVE rule — one local state-computable rule, the
    same at every depth (substitution-stable), producing such a
    labelling. At 3 letters no equivariant rule exists (holonomy);
    whether 4 or 5 letters admit one is stage 2.

Run:  python experimental/e4h_fourth_letter.py [--layer 1] [--viz PNG]
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

VERTS = [(90.0, 0.0), (-90.0, 0.0),
         (0.0, 0.0), (0.0, 90.0), (0.0, 180.0), (0.0, -90.0)]


def to3(la, lo):
    la, lo = math.radians(la), math.radians(lo)
    return np.array([math.cos(la) * math.cos(lo),
                     math.cos(la) * math.sin(lo), math.sin(la)])


def build_complex(layer, reg):
    """(hosts, edges, triples): edges keyed by 3D midpoint; triples =
    per (host, half) the three member edge ids, from the state rings."""
    from hhg9.h9.uuid_address import _anchor_hex_latlon, h9_bin, h9_encode
    n = 40000
    i = np.arange(n)
    ga = np.pi * (3 - math.sqrt(5)) * i                 # Fibonacci sphere
    z = 1 - 2 * (i + 0.5) / n
    r = np.sqrt(1 - z * z)
    lat = np.degrees(np.arcsin(z))
    lon = np.degrees(np.arctan2(np.sin(ga) * r, np.cos(ga) * r))
    hosts = list(dict.fromkeys(h9_bin(h9_encode(lat, lon, reg), layer, reg)))

    eid, emid = {}, []
    triples, host_ll = [], []
    for u in hosts:
        ring = _anchor_hex_latlon(u, layer, reg, inflate=1.0)
        v3 = np.array([to3(la, lo) for la, lo in ring])
        host_ll.append(ring.mean(axis=0))
        eids = []
        for i in range(6):
            m = v3[i] + v3[(i + 1) % 6]
            m /= np.linalg.norm(m)
            key = tuple(np.round(m, 4))
            if key not in eid:
                eid[key] = len(eid)
                emid.append(m)
            eids.append(eid[key])
        triples.append((u, 0, [eids[1], eids[2], eids[3]]))
        triples.append((u, 1, [eids[4], eids[5], eids[0]]))
    deg = np.zeros(len(eid), int)
    for _, _, es in triples:
        for e in es:
            deg[e] += 1
    assert (deg == 2).all(), 'every edge child must belong to exactly 2 halves'
    return hosts, np.array(emid), triples, np.array(host_ll)


def solve(triples, n_edges, k, order=None, limit=20_000_000):
    """Backtracking with MRV; returns labels array or None (UNSAT).
    Exhaustive within `limit` node visits (raises if exceeded)."""
    sys.setrecursionlimit(max(20000, 8 * n_edges))
    cons = [[] for _ in range(n_edges)]
    for t, (_, _, es) in enumerate(triples):
        for e in es:
            cons[e].append(t)
    tri = [es for _, _, es in triples]
    lab = np.zeros(n_edges, int)
    order = order or list(range(1, k + 1))
    visits = 0

    # static BFS variable order over the edge-adjacency graph: local
    # propagation keeps the search near-greedy on these lattices
    nbr = [set() for _ in range(n_edges)]
    for es in tri:
        for e in es:
            nbr[e] |= set(es) - {e}
    seq, seen = [], set()
    from collections import deque
    for s in range(n_edges):
        if s in seen:
            continue
        q = deque([s])
        seen.add(s)
        while q:
            e = q.popleft()
            seq.append(e)
            for o in sorted(nbr[e]):
                if o not in seen:
                    seen.add(o)
                    q.append(o)

    def dom(e):
        used = {int(lab[o]) for o in nbr[e] if lab[o]}
        return [v for v in order if v not in used]

    def bt(i):
        nonlocal visits
        visits += 1
        if visits > limit:
            raise RuntimeError('search limit exceeded')
        if i == len(seq):
            return True
        e = seq[i]
        for v in dom(e):
            lab[e] = v
            if bt(i + 1):
                return True
        lab[e] = 0
        return False

    return lab.copy() if bt(0) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--layer', type=int, default=1)
    ap.add_argument('--viz', metavar='PNG')
    args = ap.parse_args()
    from hhg9 import Registrar
    reg = Registrar()

    hosts, emid, triples, host_ll = build_complex(args.layer, reg)
    print(f'complex   : layer {args.layer}: {len(hosts)} hosts, '
          f'{len(emid)} edge children, {len(triples)} half-triples')

    # k=3, globally, on the closed octahedron (CP-SAT when available)
    r3 = None
    try:
        from ortools.sat.python import cp_model
        m = cp_model.CpModel()
        xs = [m.NewIntVar(1, 3, f'e{i}') for i in range(len(emid))]
        for _, _, es in triples:
            m.AddAllDifferent([xs[e] for e in es])
        sv = cp_model.CpSolver()
        sv.parameters.max_time_in_seconds = 300
        st = sv.Solve(m)
        if sv.StatusName(st) in ('OPTIMAL', 'FEASIBLE'):
            r3 = np.array([sv.Value(x) for x in xs])
        engine = f'CP-SAT ({sv.StatusName(st)})'
    except ImportError:
        r3 = solve(triples, len(emid), 3)
        engine = 'backtracking'
    if r3 is None:
        print(f'k=3 global: UNSAT [{engine}]')
        return False
    bad = sum(len({int(r3[e]) for e in es}) != 3 for _, _, es in triples)
    import collections
    print(f'k=3 global: SAT [{engine}] — independent check violations '
          f'{bad}, balance {dict(collections.Counter(r3.tolist()))}')
    print('          : the 4th letter is NOT needed for existence; the '
          'open question is the hierarchic/transitive rule (docstring)')
    r4 = r3                                        # viz shows the 3-letter map

    if args.viz:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        DIG = ['#2a5fa5', '#DDAA33', '#b7d0ee', '#a97f14']
        fig, ax = plt.subplots(figsize=(13, 6.8))
        lat = np.degrees(np.arcsin(emid[:, 2]))
        lon = np.degrees(np.arctan2(emid[:, 1], emid[:, 0]))
        ax.scatter(lon, lat, c=[DIG[v - 1] for v in r4], s=26, lw=0)
        ax.scatter([w[1] for w in VERTS], [w[0] for w in VERTS],
                   marker='*', color='k', s=180, label='cone vertices')
        ax.set_xlabel('longitude')
        ax.set_ylabel('latitude')
        ax.legend(loc='lower left', fontsize=9)
        ax.set_title(f'three letters suffice, layer {args.layer}: one '
                     'label per fine edge child, each half rainbow, '
                     'closed octahedron (CP-SAT, balanced)',
                     fontsize=11.5)
        fig.savefig(args.viz, dpi=140, bbox_inches='tight')
        print('viz       :', args.viz)
    return True


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
