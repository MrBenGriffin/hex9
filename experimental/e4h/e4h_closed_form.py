# Part of the Hex9 (H9) Project
# Copyright ©2026, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""Pinning the closed form of the five-symbol enumeration (stage 4).

The depth-1..4 witness (e4h_stability.py FINDINGS 2) suggested:

  S1  digit 1 anchors at class (1 + half);
  S2  class 3 carries the same digit in both halves (q);
  S3  the per-(octant, c2) triple (p, q, r) rotates right as c2 steps;
  S4  each octant avoids exactly one digit of {2,3,4,5}, constant on
      ANTIPODAL octant pairs (axes <-> digits, a bijection).

This harness adds S1-S4 as CONSTRAINTS on top of the real joint
depth-1..4 complexes (all octants + seam + vertex + poles) and asks:

  1. SAT?      — the structured family is realisable (sanity);
  2. FORCED?   — with the gauge fixed (axis->digit bijection pinned to
                 the witness's, one phase pinned), how many solutions
                 remain? A count of 1 makes the formula machine-fact
                 modulo gauge;
  3. FORMULA   — extract the per-octant base cycles and print the
                 closed form; verify a pure-python implementation of
                 it against every constraint set independently.

FINDINGS (2026-08-03): the structured family is SAT, and with the
gauge fixed (axis->digit bijection + octant-0 phase) the solution is
UNIQUE — the closed form is machine-fact modulo gauge. The
pure-python formula() reproduces the table exactly and passes every
gathered depth-1..4 constraint independently (CLEAN). Bonus clause
read off the forced bases: ANTIPODAL octants satisfy
base(o') = (q, p, r) — swap p and q, keep r — so the whole rule needs
only FOUR base cycles, one per octahedral axis, plus S1-S4. All
inter-octant phases were forced by pinning octant 0 alone: the
coupling is fully rigid. Caveat: at the harness grid density depth 4
contributes pair constraints only (0 complete triples; depth-3
triples partial) — a denser depth-4 gather is the remaining dot on
the i.

Run:  python experimental/e4h_closed_form.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from e4h_stability import gather  # noqa: E402

AXIS = {0: 0, 7: 0, 1: 1, 6: 1, 2: 2, 5: 2, 3: 3, 4: 3}
MISS_W = {0: 4, 1: 5, 2: 2, 3: 3}      # witness gauge: axis -> missing digit


def build(data, fix_gauge):
    """CP-SAT model: T[oid,c2,half,cls] with S1-S4 + all pair/triple
    constraints from every depth's complex."""
    from ortools.sat.python import cp_model
    m = cp_model.CpModel()
    T = {}
    for o in range(8):
        for c2 in range(3):
            for h in range(2):
                for cl in (1, 2, 3):
                    T[(o, c2, h, cl)] = m.NewIntVar(1, 5, f't{o}{c2}{h}{cl}')
    # S1 anchor
    for o in range(8):
        for c2 in range(3):
            for h in range(2):
                m.Add(T[(o, c2, h, 1 + h)] == 1)
                m.AddAllDifferent([T[(o, c2, h, cl)] for cl in (1, 2, 3)])
    # S2 shared q; S3 rotation: (p,q,r) -> (r,p,q) as c2 -> c2+1
    P = lambda o, c2: T[(o, c2, 0, 2)]
    Q = lambda o, c2: T[(o, c2, 0, 3)]
    R = lambda o, c2: T[(o, c2, 1, 1)]
    for o in range(8):
        for c2 in range(3):
            m.Add(T[(o, c2, 1, 3)] == Q(o, c2))
            n = (c2 + 1) % 3
            m.Add(P(o, n) == R(o, c2))
            m.Add(Q(o, n) == P(o, c2))
            m.Add(R(o, n) == Q(o, c2))
    # S4 axis-missing digit, bijective over axes
    miss = [m.NewIntVar(2, 5, f'miss{a}') for a in range(4)]
    m.AddAllDifferent(miss)
    for o in range(8):
        for c2 in range(3):
            for h in range(2):
                for cl in (1, 2, 3):
                    m.Add(T[(o, c2, h, cl)] != miss[AXIS[o]])
    if fix_gauge:
        for a, v in MISS_W.items():
            m.Add(miss[a] == v)
        m.Add(P(0, 0) == 5)                 # pin octant-0 phase (witness)
    # the real complexes
    V = {}

    def var(ctx):
        o, c2, h, d1, cl = ctx
        return T[(o, c2, h, cl)]

    for sides, tris in data:
        node_var = {}
        for node, (ca, cb) in sides.items():
            m.Add(var(ca) == var(cb))
            node_var[node] = var(ca)
        for tri in tris:
            m.AddAllDifferent([node_var[nd] for nd in tri
                               if nd in node_var])
    return m, T, miss


def solve_count(m, T, limit=200):
    from ortools.sat.python import cp_model

    class Cb(cp_model.CpSolverSolutionCallback):
        def __init__(self, T):
            super().__init__()
            self.T = T
            self.sols = []

        def on_solution_callback(self):
            self.sols.append({k: self.Value(v) for k, v in self.T.items()})
            if len(self.sols) >= limit:
                self.StopSearch()

    sv = cp_model.CpSolver()
    sv.parameters.enumerate_all_solutions = True
    sv.parameters.max_time_in_seconds = 300
    cb = Cb(T)
    sv.Solve(m, cb)
    return cb.sols, sv.StatusName(sv.Solve(m))


def formula(o, c2, h, cl, bases):
    """The closed form: S1/S2/S3 mechanics over per-octant base cycles."""
    if cl == 1 + h:
        return 1
    b = bases[o]
    p, q, r = b[(0 - c2) % 3], b[(1 - c2) % 3], b[(2 - c2) % 3]
    return q if cl == 3 else (p if h == 0 else r)


def main():
    from hhg9 import Registrar
    reg = Registrar()
    data = [gather(reg, d) for d in (1, 2, 3, 4)]
    for i, (s, t) in enumerate(data):
        print(f'depth {i + 1}: {len(s)} nodes, {len(t)} triples')

    m, T, miss = build(data, fix_gauge=False)
    from ortools.sat.python import cp_model
    sv = cp_model.CpSolver()
    sv.parameters.max_time_in_seconds = 300
    ok = sv.StatusName(sv.Solve(m)) in ('OPTIMAL', 'FEASIBLE')
    print(f'structured family (S1-S4 + depths 1-4): '
          f'{"SAT" if ok else "UNSAT"}')

    m, T, miss = build(data, fix_gauge=True)
    sols, _ = solve_count(m, T)
    print(f'gauge-fixed solutions (axis map + one phase pinned): '
          f'{len(sols)}{"+" if len(sols) >= 200 else ""}')
    if not sols:
        return False

    t0 = sols[0]
    bases = {}
    for o in range(8):
        bases[o] = (t0[(o, 0, 0, 2)], t0[(o, 0, 0, 3)], t0[(o, 0, 1, 1)])
    print('per-octant base cycles (p,q,r at c2=0):')
    for o in range(8):
        print(f'  oid{o}: {bases[o]}')
    bad = sum(formula(o, c2, h, cl, bases) != t0[(o, c2, h, cl)]
              for (o, c2, h, cl) in t0)
    print(f'pure-python formula reproduces the table: '
          f'{"YES" if bad == 0 else f"NO ({bad} mismatches)"}')

    # independent check of the formula against every complex
    viol = 0
    for sides, tris in data:
        nd = {}
        for node, (ca, cb) in sides.items():
            va = formula(ca[0], ca[1], ca[2], ca[4], bases)
            vb = formula(cb[0], cb[1], cb[2], cb[4], bases)
            viol += va != vb
            nd[node] = va
        for tri in tris:
            vs = [nd[n] for n in tri if n in nd]
            viol += len(set(vs)) != len(vs)
    print(f'formula vs all depth-1..4 constraints: '
          f'{"CLEAN" if viol == 0 else f"{viol} violations"}')
    return viol == 0


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
