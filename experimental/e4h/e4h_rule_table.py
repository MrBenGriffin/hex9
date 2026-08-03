# Part of the Hex9 (H9) Project
# Copyright ©2026, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""Stage 2 of the matched-digits question, depth-1 (the simplest case).

Existence of globally matched labellings is settled (three letters
suffice: experimental/e4h_fourth_letter.py, CP-SAT, layers 1-2). This
harness asks Ben's harder question (2026-08-03): is there a
HIERARCHICALLY STABLE / TRANSITIVE rule — digits computed from LOCAL
STATE, not chosen by a global gauge?

Depth 1 is the pure case: every fine child sits on a host edge, so
every pair is cross-host. Rule form under test:

    digit(child) = T[state_class(host), ring_slot]     slot in 0..5

where ring_slot is the child's edge position in the host's state ring
(H9P.hx order — already state-covariant), and T is a finite table.
Constraints, over the REAL layer-L complex:

  * matched   — for every physical edge, its two incident
                (state_class, slot) entries agree;
  * distinct  — per class, T[c, {1,2,3}] all-different and
                T[c, {4,5,0}] all-different (the two half-triples).

The search sweeps alphabet size k = 3, 4, 5 against nested state
granularities (tail state p_mo/c2/r_mo, octant mode, octant id), and
for every SAT table re-verifies it verbatim on the OTHER layer — one
table satisfying both layers is the transitivity signal; a table that
fits L1 but not L2 is a gauge in disguise.

FINDINGS (2026-08-03, first sweep):

  * state (p_mo, c2) / (+r_mo) / (+mo): UNSAT at k = 3, 4 AND 5 —
    tail state alone cannot carry a matched rule at any swept
    alphabet size;
  * state (p_mo, c2, r_mo, oid) — 24 classes, and exactly the same
    24 appear at layer 2 (state alphabet layer-closed): k = 3 and 4
    UNSAT, k = 5 SAT, and the layer-1 table satisfies layer 2
    VERBATIM — one octant-aware five-letter table serves both layers.
    Ben's fifth symbol is exactly where the transitive rule appears:
    existence needs 3 letters (e4h_fourth_letter.py), a
    state-computable rule needs 5 (at this granularity). The found
    table shows structure (every 5-entry has p_mo = 0) but is one
    gauge — no minimisation of 4/5 usage attempted yet.

FINDINGS 2 (2026-08-03, minimisation + Ben's rotating-centre idea):

  * ROTATING CENTRE: freeing the centre digit from the reserved 0
    (centre joins each half's all-different; centre matching is
    within-host, hence free) compresses the TOTAL alphabet: K = 4 is
    UNSAT, K = 5 is SAT with verbatim L2 transfer — five symbols
    TOTAL {0..4}, versus six for the fixed-centre form. Five is
    optimal at this granularity (each half needs 4 distinct digits,
    and 4 total is refuted).
  * MINIMISED fixed-centre table (k = 5, CP-SAT OPTIMAL, edge-weighted
    5-then-4 objective): exactly 27 physical 5-edges of 324 on L1
    (true minimum), 12 table entries, all in half-1 slots (s0/s5),
    exactly two per octant with octants 3 and 4 entirely 5-free; the
    27 sit in a regular superlattice, not clustered at the vertices.
  * STRUCTURE: p_mo == 0 for every host tail at L1/L2 and r_mo equals
    the octant mode, so the EFFECTIVE state is just (octant, c2) — 24
    classes. Slots s3 and s4 carry digit 2 UNIFORMLY across all
    classes: the two edges flanking ring corner 4 (one end of the
    host cut) get a universal digit — an echo of the old positional
    "2 = axis" surviving inside the transitive rule. Closed-form
    factorisation of the remaining slots (c2 rotation x octant
    permutation?) is the next hunt, then the depth-2 substitution
    test.

Run:  python experimental/e4h_rule_table.py
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def to3(la, lo):
    la, lo = math.radians(la), math.radians(lo)
    return np.array([math.cos(la) * math.cos(lo),
                     math.cos(la) * math.sin(lo), math.sin(la)])


def complex_slots(layer, reg):
    """Per host: (state tuple, edge id per ring slot 0..5).
    State tuple = (p_mo, c2, r_mo, oid_mo, oid) — all address-local."""
    from hhg9.h9.constants import H9O
    from hhg9.h9.tail import tail_unpack_reversible
    from hhg9.h9.uuid_address import (_anchor_hex_latlon, h9_bin, h9_dec,
                                      h9_encode)
    n = 40000
    i = np.arange(n)
    ga = np.pi * (3 - math.sqrt(5)) * i
    z = 1 - 2 * (i + 0.5) / n
    r = np.sqrt(1 - z * z)
    lat = np.degrees(np.arcsin(z))
    lon = np.degrees(np.arctan2(np.sin(ga) * r, np.cos(ga) * r))
    hosts = list(dict.fromkeys(h9_bin(h9_encode(lat, lon, reg), layer, reg)))
    oids = h9_dec(hosts, reg.domain('b_oct')).oid

    eid, recs = {}, []
    for u, oid in zip(hosts, oids):
        ring = _anchor_hex_latlon(u, layer, reg, inflate=1.0)
        v3 = np.array([to3(la, lo) for la, lo in ring])
        eids = []
        for s in range(6):
            m = v3[s] + v3[(s + 1) % 6]
            m /= np.linalg.norm(m)
            key = tuple(np.round(m, 4))
            eids.append(eid.setdefault(key, len(eid)))
        c2, r_mo, p_mo = tail_unpack_reversible(
            np.array([u.int & 0xF], dtype=np.uint8))
        recs.append(((int(p_mo[0]), int(c2[0]), int(r_mo[0]),
                      int(H9O.oid_mo[int(oid)]), int(oid)), eids))
    deg = np.zeros(len(eid), int)
    for _, es in recs:
        for e in es:
            deg[e] += 1
    assert (deg == 2).all()
    return recs, len(eid)


PROJ = [  # nested state granularities: name, projection of the tuple
    ('p_mo,c2', lambda st: st[:2]),
    ('p_mo,c2,r_mo', lambda st: st[:3]),
    ('p_mo,c2,mo', lambda st: (st[0], st[1], st[3])),
    ('p_mo,c2,r_mo,mo', lambda st: st[:4]),
    ('p_mo,c2,r_mo,oid', lambda st: (st[0], st[1], st[2], st[4])),
]
HALVES = ([1, 2, 3], [4, 5, 0])                    # ring-slot half triples


def edge_incidence(recs, proj):
    inc = {}
    classes = set()
    for st, eids in recs:
        c = proj(st)
        classes.add(c)
        for s, e in enumerate(eids):
            inc.setdefault(e, []).append((c, s))
    return inc, sorted(classes)


def solve_table(inc, classes, k, minimise=False):
    """CP-SAT over the rule table T[class, slot]; returns T or None.
    minimise: least physical usage of digit 5, then 4 (edge-weighted)."""
    from ortools.sat.python import cp_model
    m = cp_model.CpModel()
    T = {(c, s): m.NewIntVar(1, k, f't{i}_{s}')
         for i, c in enumerate(classes) for s in range(6)}
    for c in classes:
        for tri in HALVES:
            m.AddAllDifferent([T[(c, s)] for s in tri])
    cost = []
    for e, pair in inc.items():
        (c1, s1), (c2, s2) = pair
        m.Add(T[(c1, s1)] == T[(c2, s2)])
        if minimise:
            for v, w in ((5, 1000), (4, 1)):
                if v <= k:
                    b = m.NewBoolVar(f'e{e}_{v}')
                    m.Add(T[(c1, s1)] == v).OnlyEnforceIf(b)
                    m.Add(T[(c1, s1)] != v).OnlyEnforceIf(b.Not())
                    cost.append(w * b)
    if minimise and cost:
        m.Minimize(sum(cost))
    sv = cp_model.CpSolver()
    sv.parameters.max_time_in_seconds = 180
    st = sv.Solve(m)
    if sv.StatusName(st) not in ('OPTIMAL', 'FEASIBLE'):
        return None
    return {(c, s): sv.Value(v) for (c, s), v in T.items()}


def solve_rotating(inc, classes, K, minimise=True):
    """Ben's rotating-centre variant: no privileged 0. Total alphabet
    {0..K-1}; centre digit C[class] joins each half's all-different
    (a half = centre + its 3 edge children, 4 distinct digits needed,
    so K >= 4). Centre matching across the host cut is within-class,
    hence free. Returns (T, C) or None."""
    from ortools.sat.python import cp_model
    m = cp_model.CpModel()
    T = {(c, s): m.NewIntVar(0, K - 1, f't{i}_{s}')
         for i, c in enumerate(classes) for s in range(6)}
    C = {c: m.NewIntVar(0, K - 1, f'c{i}') for i, c in enumerate(classes)}
    for c in classes:
        for tri in HALVES:
            m.AddAllDifferent([C[c]] + [T[(c, s)] for s in tri])
    cost = []
    for e, pair in inc.items():
        (c1, s1), (c2, s2) = pair
        m.Add(T[(c1, s1)] == T[(c2, s2)])
        if minimise and K >= 5:
            b = m.NewBoolVar(f'e{e}h')
            m.Add(T[(c1, s1)] >= 4).OnlyEnforceIf(b)
            m.Add(T[(c1, s1)] < 4).OnlyEnforceIf(b.Not())
            cost.append(b)
    if cost:
        m.Minimize(sum(cost))
    sv = cp_model.CpSolver()
    sv.parameters.max_time_in_seconds = 180
    st = sv.Solve(m)
    if sv.StatusName(st) not in ('OPTIMAL', 'FEASIBLE'):
        return None
    return ({(c, s): sv.Value(v) for (c, s), v in T.items()},
            {c: sv.Value(v) for c, v in C.items()})


def check_table(inc, T):
    """Does a fixed table satisfy a complex? (classes must be covered)"""
    for e, pair in inc.items():
        (c1, s1), (c2, s2) = pair
        if (c1, s1) not in T or (c2, s2) not in T:
            return 'classes missing'
        if T[(c1, s1)] != T[(c2, s2)]:
            return 'mismatch'
    return 'OK'


def main():
    from hhg9 import Registrar
    reg = Registrar()
    data = {L: complex_slots(L, reg) for L in (1, 2)}
    for L, (recs, ne) in data.items():
        print(f'layer {L}: {len(recs)} hosts, {ne} edges')
    print()
    print(f'{"state class":22s} {"#cls":>4s}  ' +
          '  '.join(f'k={k}: L1    xL2' for k in (3, 4, 5)))
    for name, proj in PROJ:
        row = f'{name:22s}'
        inc1, cls1 = edge_incidence(data[1][0], proj)
        inc2, cls2 = edge_incidence(data[2][0], proj)
        row += f' {len(set(cls1) | set(cls2)):>4d}  '
        for k in (3, 4, 5):
            T = solve_table(inc1, cls1, k)
            if T is None:
                row += f'  UNSAT  --   '
                continue
            x = check_table(inc2, T)
            row += f'  SAT    {x:5s}'
        print(row)
    print()
    print('xL2 = the L1 table applied verbatim to layer 2: OK means one')
    print('table serves both layers (transitivity signal); a table can')
    print('also fail xL2 yet a joint L1+L2 solve succeed — try that for')
    print('any promising row.')
    return True


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
