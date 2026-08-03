# Part of the Hex9 (H9) Project
# Copyright ©2026, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""Hierarchic stability of the matched-digit rule (stage 3).

Depth-1 has a transitive rule (e4h_rule_table.py: five letters,
state (octant, c2), one table for layers 1-2). This harness asks the
substitution question: does ONE rule serve depths 1 AND 2 of the tail
simultaneously — the property Ben named "hierarchically stable /
transitive" and the last step between a per-layer gauge and a real
addressing rule?

Built entirely on the h9e address machinery (no fresh geometry):

  * sample patches covering all octants, an equator seam and the cone
    point (0,0); encode every grid point at depth 1 and depth 2;
  * a NODE is a fine hexagon: a leaf paired with its partner
    (h9e_partner_point -> encode), cross-host pairs included;
  * a SIDE CONTEXT is what a local rule may see: (octant, c2 of the
    host, half, d1, class), with class = the leaf's final class digit
    (the raw direction class in the host frame — the current
    implementation's identity rule makes the digit the class);
  * constraints: T[ctx_A] == T[ctx_B] for the two sides of every
    node (matched), plus AllDifferent over each trapezoid's three
    edge children (addressability).

Context projections sweep what the rule is allowed to depend on; the
JOINT solve shares one table across depths (projections that include
d1 are depth-2-only and solved per-depth instead). k sweeps 3..5.

FINDINGS (2026-08-03, all 8 octants + equator seam + cone vertex +
both poles in the patch set):

  * depth 2 alone is easy (SAT even at k=3, any context) — interior
    pairs dominate; depth 1 is the hard case, exactly as Ben said.
  * the HALF digit is load-bearing context: without it the joint
    solve is UNSAT at every k; with it —
  * ONE five-letter table T[octant, c2, half, class] (144 entries)
    satisfies depths 1, 2 AND 3 JOINTLY (k=4 UNSAT, k=5 SAT):
    the hierarchically stable / transitive rule Ben conjectured
    exists, at least to depth 3, with fully address-local context.
    Witness saved: experimental/e4h_rule_T5.json (digit usage
    48/24/24/24/24 — suspiciously regular; closed form next).

FINDINGS 2 (same day): depth 4 joins the joint solve — SAT at k = 5,
UNSAT at k = 4 (witness e4h_rule_T5_d4.json). The witness is nearly a
formula: digit 1 anchors at class (1 + half) everywhere; class 3
carries the same digit in both halves; per octant the remaining
triple is a 3-cycle over {2,3,4,5} minus one digit, where the missing
digit is constant on ANTIPODAL octant pairs (axes <-> digits), the
cycle rotates with c2 and its orientation flips with octant mode.
Conjectured closed form recorded in transport note §4d; still to pin
by canonicalised re-solve (the witness is one CP-SAT gauge).

Run:  python experimental/e4h_stability.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

LAYER = 6
PATCHES = [  # (name, lat0, lat1, lon0, lon1) — octant + seam + vertex mix
    ('NEA-int', 19.5, 20.5, 34.5, 35.5),
    ('seam-eq', -0.35, 0.35, 34.5, 35.2),
    ('vertex00', -0.28, 0.28, -0.28, 0.28),
    ('NWP-int', 19.5, 20.3, -125.5, -124.8),
    ('S-int', -20.5, -19.8, 100.2, 100.9),
    ('SWP-int', -20.5, -19.8, -100.9, -100.2),
    ('pole-N', 89.55, 89.95, -180.0, 180.0),
    ('pole-S', -89.95, -89.55, -180.0, 180.0),
]


def gather(reg, depth, n=46):
    """(sides, triples): sides[node] = list of (ctx, key); ctx =
    (oid, c2, half, d1, cls). Nodes = fine hexagons via partner."""
    from hhg9.h9.e4h import (h9e_encode, h9e_partner_point, h9e_split)
    from hhg9.h9.tail import tail_unpack_reversible
    from hhg9.h9.uuid_address import h9_dec
    b_oct = reg.domain('b_oct')
    oid_of, c2_of = {}, {}

    def host_state(host):
        if host.int not in oid_of:
            oid_of[host.int] = int(h9_dec([host], b_oct).oid[0])
            c2, _r, _p = tail_unpack_reversible(
                np.array([host.int & 0xF], dtype=np.uint8))
            c2_of[host.int] = int(c2[0])
        return oid_of[host.int], c2_of[host.int]

    def ctx_of(u):
        host, half, digs = h9e_split(u)
        oid, c2 = host_state(host)
        d1 = digs[0] if len(digs) > 1 else None
        return (oid, c2, half, d1, digs[-1]), (host.int, half, tuple(digs))

    leaves = {}
    for name, la0, la1, lo0, lo1 in PATCHES:
        g1 = np.linspace(la0, la1, n)
        g2 = np.linspace(lo0, lo1, n)
        gl, gn = np.meshgrid(g1, g2)
        for u in h9e_encode(gl.ravel(), gn.ravel(), LAYER, depth, reg):
            leaves[u.int] = u
    us = list(leaves.values())
    pl, po = h9e_partner_point(us, reg)
    pus = h9e_encode(pl, po, LAYER, depth, reg)

    sides, triples = {}, {}
    for u, pu in zip(us, pus):
        ca, ka = ctx_of(u)
        cb, kb = ctx_of(pu)
        if ca[4] == 0 or cb[4] == 0:
            continue                        # centre children: fixed digit
        node = min(ka, kb), max(ka, kb)
        sides[node] = [ca, cb]
        triples.setdefault(ka[:2] + (ka[2][:-1],), {})[ka[2][-1]] = node
        triples.setdefault(kb[:2] + (kb[2][:-1],), {})[kb[2][-1]] = node
    tris = [list(d.values()) for d in triples.values() if len(d) == 3]
    return sides, tris


PROJ = [
    ('oid,c2 | cls', lambda c: (c[0], c[1], c[4]), True),
    ('oid,c2,half | cls', lambda c: (c[0], c[1], c[2], c[4]), True),
    ('oid,c2,d1 | cls', lambda c: (c[0], c[1], c[3], c[4]), False),
    ('oid,c2,half,d1 | cls', lambda c: (c[0], c[1], c[2], c[3], c[4]),
     False),
]


def solve(sides_list, tris_list, proj, k):
    """One shared table over every (sides, tris) set. None = UNSAT."""
    from ortools.sat.python import cp_model
    m = cp_model.CpModel()
    V = {}

    def var(ctx):
        key = proj(ctx)
        if key not in V:
            V[key] = m.NewIntVar(1, k, str(key))
        return V[key]

    node_var = {}
    for sides, tris in zip(sides_list, tris_list):
        for node, (ca, cb) in sides.items():
            m.Add(var(ca) == var(cb))
            node_var[node] = var(ca)
        for tri in tris:
            m.AddAllDifferent([node_var[nd] for nd in tri if nd in node_var])
    sv = cp_model.CpSolver()
    sv.parameters.max_time_in_seconds = 120
    st = sv.Solve(m)
    ok = sv.StatusName(st) in ('OPTIMAL', 'FEASIBLE')
    return ({k2: sv.Value(v) for k2, v in V.items()} if ok else None,
            len(V))


def main():
    from hhg9 import Registrar
    reg = Registrar()
    data = {d: gather(reg, d) for d in (1, 2)}
    for d, (sides, tris) in data.items():
        octs = {c[0] for ss in sides.values() for c in ss}
        print(f'depth {d}: {len(sides)} paired nodes, {len(tris)} '
              f'triples, octants seen {sorted(octs)}')
    print()
    print(f'{"context":24s}' + ''.join(f'  k={k}: d1    d2    joint'
                                       for k in (3, 4, 5)))
    for name, proj, joint_ok in PROJ:
        row = f'{name:24s}'
        for k in (3, 4, 5):
            r1, _ = solve([data[1][0]], [data[1][1]], proj, k)
            r2, _ = solve([data[2][0]], [data[2][1]], proj, k)
            if joint_ok:
                rj, nv = solve([data[1][0], data[2][0]],
                               [data[1][1], data[2][1]], proj, k)
                j = 'SAT ' if rj is not None else 'UNSAT'
            else:
                j = ' -- '
            row += (f'  {"SAT " if r1 else "UNSAT":5s} '
                    f'{"SAT " if r2 else "UNSAT":5s} {j:5s}')
        print(row)
    print()
    print('joint = ONE table for depths 1 and 2 together — the')
    print('substitution-stability signal.')
    return True


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
