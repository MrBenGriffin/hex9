# Part of the Hex9 (H9) Project
# Copyright ©2026, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""The rotating-centre {0..4} total-alphabet variant: establish or
refute (stage 5).

The promoted five-symbol enumeration spends SIX nibble values on the
tail alphabet: a reserved 0 for centre children plus edge digits 1..5.
Ben's rotating-centre idea (2026-08-03): stop privileging 0 — let the
centre digit vary with state and join each trapezoid's all-different,
compressing the total alphabet to FIVE symbols {0..4}. At depth 1
this was SAT with host-layer transfer (e4h_rule_table.py). Here it
faces the full test: JOINT depths 1..3, all octants + seam + vertex +
poles, centre children now first-class nodes (their pairs can cross
hosts and octant seams at depth >= 2).

Model: raw class 0 = centre, 1..3 = direction classes (recovered from
the promoted digits by inverting _digit5). One table
D[oid, c2, half, rawclass] in {0..K-1}; constraints:

  * matched — every fine hexagon's two sides agree (centre pairs
    INCLUDED — the constraint the naive own-axis-to-centre guess
    should fail at octant seams);
  * addressable — each trapezoid's FOUR children (centre + 3 edges)
    all-different.

Sweep K = 4, 5; if SAT, hunt the derivation (is the edge sub-table
the promoted rule relabelled? what generates the centre digits?).

FINDINGS (2026-08-03): REFUTED as a like-for-like replacement,
ESTABLISHED only in a weaker, costlier form.

  * K = 4: UNSAT (each trapezoid needs 4 distinct digits and the
    matching leaves no room — as at depth 1).
  * K = 5 with the SAME depth-uniform context as the promoted rule
    (oid, c2, half, class): UNSAT jointly at depths 1-3. The depth-1
    SAT of e4h_rule_table.py was another shallow-case mirage: centre
    pairs become cross-host constraints at depth >= 2 and kill the
    five-total-symbol alphabet. The reserved-symbol six-value form
    (promoted rule) is minimal AND necessary at this granularity.
  * K = 5 WITH d1 added to the context: SAT — five total symbols is
    possible if the rule takes one symbol of memory (the first tail
    digit). Alphabet width trades against context memory; the
    solution's centre digits vary richly (mostly 4, no clean
    structure apparent).
  * K = 6 sanity: SAT with a constant centre digit (the solver picked
    3; our promoted rule is the centre = 0 gauge) — the reserved
    form embeds, as it must.

Verdict: keep the promoted reserved-0 rule. The nibble affords the
six values, the rule stays memoryless per level, and the rotating
variant buys one symbol at the price of context depth. Caveat: depth-3
quad coverage is partial at this grid density (210 complete quads;
pair coverage strong).

Run:  python experimental/e4h_rotating.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from e4h_stability import PATCHES  # noqa: E402

LAYER = 6


def gather_full(reg, depth, n=46):
    """Like e4h_stability.gather, but: centre children included as
    nodes, contexts carry the RAW class (0..3), and trapezoid groups
    are returned for the 4-way all-different."""
    from hhg9.h9.e4h import (_digit5, h9e_encode, h9e_partner_point,
                             h9e_split)
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
        d = digs[-1]
        if d == 0:
            cls = 0
        else:
            cls = next(c for c in (1, 2, 3)
                       if _digit5(oid, c2, half, c) == d)
        d1 = digs[0] if len(digs) > 1 else -1
        return (oid, c2, half, d1, cls), (host.int, half, tuple(digs))

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

    sides, groups = {}, {}
    for u, pu in zip(us, pus):
        ca, ka = ctx_of(u)
        cb, kb = ctx_of(pu)
        node = min(ka, kb), max(ka, kb)
        sides[node] = [ca, cb]
        groups.setdefault(ka[:2] + (ka[2][:-1],), {})[ca[4]] = node
        groups.setdefault(kb[:2] + (kb[2][:-1],), {})[cb[4]] = node
    quads = [list(g.values()) for g in groups.values() if len(g) == 4]
    return sides, quads


def solve(sets, K, with_d1=False):
    from ortools.sat.python import cp_model
    m = cp_model.CpModel()
    V = {}

    def var(ctx):
        key = ctx if with_d1 else ctx[:3] + ctx[4:]
        if key not in V:
            V[key] = m.NewIntVar(0, K - 1, str(key))
        return V[key]

    for sides, quads in sets:
        nv = {}
        for node, (ca, cb) in sides.items():
            m.Add(var(ca) == var(cb))
            nv[node] = var(ca)
        for quad in quads:
            m.AddAllDifferent([nv[nd] for nd in quad if nd in nv])
    sv = cp_model.CpSolver()
    sv.parameters.max_time_in_seconds = 300
    st = sv.Solve(m)
    if sv.StatusName(st) not in ('OPTIMAL', 'FEASIBLE'):
        return None
    return {k: sv.Value(v) for k, v in V.items()}


def main():
    from hhg9 import Registrar
    reg = Registrar()
    sets = [gather_full(reg, d) for d in (1, 2, 3)]
    for i, (s, q) in enumerate(sets):
        print(f'depth {i + 1}: {len(s)} nodes '
              f'({sum(1 for v in s.values() if v[0][4] == 0)} '
              f'centre pairs), {len(q)} complete quads')
    for K, wd1, tag in ((4, False, ''), (5, False, ''),
                        (5, True, ' + d1 context'),
                        (6, False, ' (sanity: reserved-0 embeds)')):
        D = solve(sets, K, with_d1=wd1)
        print(f'rotating centre, JOINT depths 1-3, K={K}{tag}: '
              f'{"SAT" if D else "UNSAT"}')
        if not D:
            continue
        cen = {k: v for k, v in D.items() if k[-1] == 0}
        import collections
        cu = collections.Counter(cen.values())
        print(f'  centre digit usage: {dict(sorted(cu.items()))}')
        by_oid = {}
        for k, v in cen.items():
            by_oid.setdefault(k[0], set()).add(v)
        print('  centre digits per octant:',
              {o: sorted(vs) for o, vs in sorted(by_oid.items())})
    return True


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
