# Part of the Hex9 (H9) Project
# Copyright ©2026, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""The k=7 / GBT-flavoured septree on the a4 tail (stage 6).

Every hexagon has SEVEN next-level constituents: its centre child plus
the six edge children sitting on its edge midpoints. Ben's k=7 idea
(2026-08-03): label them 0..6 all-different per parent hexagon — a
GBT-style enumeration ON the a4 tower (cf. the Weston/Renoud GBT
correction-set discussion, transport note §3). The ambiguity trades
shape: the half-split (two addresses per hexagon) becomes the classic
two-parent ambiguity (every edge hexagon sits in TWO parents'
seven-sets, H3-style; ownership pin = the usual mode-0 rule). The
prize, if the rule is structured: GBT digit arithmetic — a first
tail-side crack at the odometer.

Engineering (Ben): AllDifferent(7) with a CONJOINED centre — one
colouring per hexagon applied to both halves, so matched pairs hold by
construction (one variable per hexagon; in rule-table form, the two
side contexts are constrained equal). The seven-set of a parent
hexagon P = {centre hexagon} ∪ {6 edge hexagons}; each of P's two
trapezoids contributes its class-0 child (one half of the centre
hexagon) and its three edge-class children (one half each of three
edge hexagons) — so per-trapezoid all-different is implied and the
k=7 constraint is strictly stronger than the promoted rule's.

Built on the harness bones (address machinery only, no fresh
geometry): nodes = partner-paired fine hexagons over PATCHES (all 8
octants + equator seam + cone vertex + both poles); NEW BIT =
seven-cliques per parent hexagon. Parent identification uses
truncation-is-binning: the generating point of each side encodes at
depth-1 to that side's parent trapezoid; the parent trapezoid pairs
with ITS partner into the parent hexagon node. At tail depth 1 the
parent hexagon is the h9 host itself.

Questions:
  * EXISTENCE (per depth, free labelling): is a global 0..6 labelling
    SAT on the real closed octahedron? (On the plane GBT says yes;
    the cone-point 3-cycle holonomy is the thing being tested.)
  * RULE (joint depths, one table): does a state-computable
    T[oid, c2, half, class] exist at K=7? K=7 is the pigeonhole
    floor; sweep K=8 and the +d1 context if 7 is tight.

FINDINGS (2026-08-03, DENSITY grids, 600/668/515 complete seven-sets
at depths 1/2/3):

  * EXISTENCE: K=7 SAT at depths 1, 2 AND 3 on the real closed
    octahedron, near-perfectly balanced (each digit ~1/7 of
    in-clique nodes). The cone-point 3-cycle holonomy does NOT
    obstruct a global GBT-style 0..6 labelling — same shape as the
    three-letter existence result (e4h_fourth_letter.py).
  * RULE: UNSAT at every K for the state-local table — and the
    union-find diagnostic shows WHY, structurally: hundreds of
    seven-sets contain two DIFFERENT edge hexagons whose conjoined
    equality chains land on the SAME table entry (e.g. two edge
    children of one parent both partnering into oid 7, c2 1, half 0,
    class 2). Identical context => identical digit => AllDifferent
    dead at ANY alphabet size. Adding d1 thins the collisions
    (1783 -> 1178) but cannot empty them; depth-2-alone SAT at
    K=7+d1 is the usual shallow-case mirage (interior pairs
    dominate), depths 1 and 3 alone are already UNSAT.
  * Reading: matches the plane. GBT septree labels there are a
    positional linear functional mod 7, not a function of local
    state — position-dependence is intrinsic, and position is
    exactly what a bounded state table cannot see. The five-symbol
    enumeration is the state-computable regime; k=7 buys per-parent
    all-different (and the GBT arithmetic prize) only if the digit
    is computed from the ADDRESS as a number — a mod-7 congruence of
    the folded position, i.e. the digit-arithmetic/odometer road,
    not a bigger table.

Run:  python experimental/e4h/e4h_k7.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from e4h_stability import PATCHES  # noqa: E402

LAYER = 6
# per-depth (patch shrink about centre, grid n): fine pitch quarters
# each tail depth, so patches shrink and densify to keep several grid
# points per fine hexagon — otherwise complete seven-sets never appear
DENSITY = {1: (1, 46), 2: (2, 60), 3: (4, 80)}


def shrunk(patch, s):
    name, la0, la1, lo0, lo1 = patch
    cla, clo = (la0 + la1) / 2, (lo0 + lo1) / 2
    hla, hlo = (la1 - la0) / (2 * s), (lo1 - lo0) / (2 * s)
    if name.startswith('pole'):        # keep the full ring of longitudes
        return cla - hla, cla + hla, lo0, lo1
    return cla - hla, cla + hla, clo - hlo, clo + hlo


def gather7(reg, depth, n=46):
    """(sides, cliques): sides[node] = [ctx_a, ctx_b] with ctx =
    (oid, c2, half, d1, rawclass); cliques = complete seven-sets
    (node lists) per parent hexagon. Nodes = fine hexagons via
    partner, centre children included."""
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

    def ctx_key(u):
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

    leaves, pt_of = {}, {}
    s = DENSITY.get(depth, (1,))[0]
    for la0, la1, lo0, lo1 in (shrunk(p, s) for p in PATCHES):
        g1 = np.linspace(la0, la1, n)
        g2 = np.linspace(lo0, lo1, n)
        gl, gn = np.meshgrid(g1, g2)
        gl, gn = gl.ravel(), gn.ravel()
        for u, la, lo in zip(h9e_encode(gl, gn, LAYER, depth, reg),
                             gl, gn):
            if u.int not in leaves:
                leaves[u.int] = u
                pt_of[u.int] = (la, lo)
    us = list(leaves.values())
    pts = np.array([pt_of[u.int] for u in us])
    pl, po = h9e_partner_point(us, reg)
    pus = h9e_encode(pl, po, LAYER, depth, reg)

    if depth > 1:
        # parent trapezoid of each side = same point one level up
        pa_us = h9e_encode(pts[:, 0], pts[:, 1], LAYER, depth - 1, reg)
        pb_us = h9e_encode(pl, po, LAYER, depth - 1, reg)
        uniq = {}
        for p in list(pa_us) + list(pb_us):
            uniq.setdefault(p.int, p)
        plist = list(uniq.values())
        ql, qo = h9e_partner_point(plist, reg)
        qus = h9e_encode(ql, qo, LAYER, depth - 1, reg)
        hex_of = {}
        for p, q in zip(plist, qus):
            _, kp = ctx_key(p)
            _, kq = ctx_key(q)
            hex_of[p.int] = (min(kp, kq), max(kp, kq))

    sides, cliques = {}, {}
    for i, (u, pu) in enumerate(zip(us, pus)):
        ca, ka = ctx_key(u)
        cb, kb = ctx_key(pu)
        node = min(ka, kb), max(ka, kb)
        sides[node] = [ca, cb]
        if depth == 1:
            pids = (('H', ka[0]), ('H', kb[0]))
        else:
            pids = (hex_of[pa_us[i].int], hex_of[pb_us[i].int])
        for pid in pids:
            cliques.setdefault(pid, set()).add(node)
    full = [sorted(s) for s in cliques.values() if len(s) == 7]

    # composition sanity: each complete seven-set has exactly one
    # centre hexagon (class 0 both sides)
    bad = sum(1 for cl in full
              if sum(1 for nd in cl if sides[nd][0][4] == 0) != 1)
    return sides, full, bad


def solve_exist(sides, cliques, K=7):
    """Free labelling, one var per hexagon node. None = UNSAT."""
    from ortools.sat.python import cp_model
    m = cp_model.CpModel()
    nv = {node: m.NewIntVar(0, K - 1, str(i))
          for i, node in enumerate(sides)}
    for cl in cliques:
        m.AddAllDifferent([nv[nd] for nd in cl])
    sv = cp_model.CpSolver()
    sv.parameters.max_time_in_seconds = 300
    st = sv.Solve(m)
    if sv.StatusName(st) not in ('OPTIMAL', 'FEASIBLE'):
        return None
    return {node: sv.Value(v) for node, v in nv.items()}


def solve_rule(sets, K, with_d1=False):
    """One table over every (sides, cliques) set. None = UNSAT."""
    from ortools.sat.python import cp_model
    m = cp_model.CpModel()
    V = {}

    def var(ctx):
        key = ctx if with_d1 else ctx[:3] + ctx[4:]
        if key not in V:
            V[key] = m.NewIntVar(0, K - 1, str(key))
        return V[key]

    for sides, cliques in sets:
        nv = {}
        for node, (ca, cb) in sides.items():
            m.Add(var(ca) == var(cb))            # conjoined halves
            nv[node] = var(ca)
        for cl in cliques:
            m.AddAllDifferent([nv[nd] for nd in cl if nd in nv])
    sv = cp_model.CpSolver()
    sv.parameters.max_time_in_seconds = 300
    st = sv.Solve(m)
    if sv.StatusName(st) not in ('OPTIMAL', 'FEASIBLE'):
        return None
    return {k: sv.Value(v) for k, v in V.items()}


def rule_collisions(sets, with_d1=False):
    """Union-find over table entries via the conjoined equalities;
    a seven-set holding two nodes with the SAME representative entry
    is UNSAT at ANY alphabet size. Returns (count, example)."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def proj(c):
        return c if with_d1 else c[:3] + c[4:]

    for sides, _cl in sets:
        for ca, cb in sides.values():
            parent[find(proj(ca))] = find(proj(cb))
    hits, example = 0, None
    for sides, cliques in sets:
        for cl in cliques:
            reps = [find(proj(sides[nd][0])) for nd in cl]
            if len(set(reps)) < 7:
                hits += 1
                if example is None:
                    dup = next(r for r in reps if reps.count(r) > 1)
                    example = (dup,
                               [sides[nd][0] for nd, r in zip(cl, reps)
                                if r == dup])
    return hits, example


def main():
    import collections
    from hhg9 import Registrar
    reg = Registrar()
    depths = (1, 2, 3)
    sets = {}
    for d in depths:
        sides, cliques, bad = gather7(reg, d, DENSITY[d][1])
        sets[d] = (sides, cliques)
        cen = sum(1 for v in sides.values() if v[0][4] == 0)
        print(f'depth {d}: {len(sides)} nodes ({cen} centre pairs), '
              f'{len(cliques)} complete seven-sets'
              + (f', {bad} BAD compositions' if bad else ''))

    print()
    for d in depths:
        sol = solve_exist(*sets[d], K=7)
        line = f'EXISTENCE depth {d}, K=7: {"SAT" if sol else "UNSAT"}'
        if sol:
            used = {nd for cl in sets[d][1] for nd in cl}
            cu = collections.Counter(v for nd, v in sol.items()
                                     if nd in used)
            line += f'  usage(in-clique) {dict(sorted(cu.items()))}'
        print(line)

    print()
    for wd1, tag in ((False, ''), (True, ' + d1 context')):
        hits, ex = rule_collisions(list(sets.values()), with_d1=wd1)
        print(f'STRUCTURAL{tag}: {hits} seven-sets with forced '
              f'same-entry collisions (UNSAT at any K)')
        if ex:
            print(f'  e.g. entry {ex[0]} shared by side contexts '
                  f'{ex[1]}')

    print()
    for K, wd1, tag in ((7, False, ''), (8, False, ''),
                        (7, True, ' + d1 context'),
                        (8, True, ' + d1 context')):
        for d in depths:
            Td = solve_rule([sets[d]], K, with_d1=wd1)
            print(f'RULE T[oid,c2,half,cls] depth {d} alone, '
                  f'K={K}{tag}: {"SAT" if Td else "UNSAT"}')
        T = solve_rule(list(sets.values()), K, with_d1=wd1)
        print(f'RULE T[oid,c2,half,cls], JOINT depths {depths}, '
              f'K={K}{tag}: {"SAT" if T else "UNSAT"}')
        if not T:
            continue
        cu = collections.Counter(T.values())
        print(f'  digit usage: {dict(sorted(cu.items()))}')
        by_oid = {}
        for k, v in T.items():
            by_oid.setdefault(k[0], set()).add(v)
        print('  digits per octant:',
              {o: sorted(vs) for o, vs in sorted(by_oid.items())})
        out = Path(__file__).with_name(
            f'e4h_rule_K{K}{"_d1" if wd1 else ""}.json')
        out.write_text(json.dumps(
            {str(k): v for k, v in sorted(T.items())}, indent=1))
        print(f'  witness -> {out.name}')
        break  # first SAT rule is the result; sweep stops here
    return True


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
