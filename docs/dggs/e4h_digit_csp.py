# Part of the Hex9 (H9) Project
# Copyright ©2026, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""E4H digit naming, Fork A: global CSP over the closed octahedron.

The matched-pairs digit scheme (Ben, 2026-07-31) names each fine edge
child by the DIRECTION CLASS of its coarse edge, per face: digits 1..3
= the face's three lattice direction classes, digit 0 = centre child.
Locally this is automatic (a half-hex's three edges are one of each
class; the three centre-touching children are rainbow). The one global
question: can the per-face class->digit namings phi_f be chosen so
that every hinge's own class gets the same digit from both sides —
12 constraints over 8 faces on the closed octahedron?

Ground truth = hex9's own constants (H9O.edges_by_id / oid_nb /
oid_mo): edge slots ARE the c2 classes ("c2 is always equator=0"), so
the CSP is a pure table computation, no geometry rederived.

Checks:
  1. INCIDENCE — the 12 edge labels each join exactly two faces, and
     the slot pairing agrees with oid_nb (table self-consistency).
  2. CSP      — brute-force all 6^7 namings (phi_0 fixed = identity):
     report SAT and the solution count.
  3. MODE LAW — the canonical solution is hex9's own apparatus:
     phi_f = identity on mode-0 octants and the (1 2) class swap on
     mode-1 octants (oid_mo) — i.e. E4H digits = c2 composed with the
     octant mode, "the same place the h9 enumeration came from".

Run:  python docs/dggs/e4h_digit_csp.py
"""
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hhg9.h9.constants import H9O


def main():
    E = H9O.edges_by_id                      # (8, 3) edge labels, c2 order
    NB = H9O.oid_nb                          # (8, 3) neighbour per slot
    MO = H9O.oid_mo                          # (8,)  octant mode

    # 1. INCIDENCE
    inc = {}
    for f in range(8):
        for s in range(3):
            inc.setdefault(str(E[f, s]), []).append((f, s))
    ok = len(inc) == 12 and all(len(v) == 2 for v in inc.values())
    pairs = []
    for lab, ((fa, sa), (fb, sb)) in sorted(inc.items()):
        ok &= NB[fa, sa] == fb and NB[fb, sb] == fa
        pairs.append(((fa, sa), (fb, sb)))
    print(f'incidence : 12 edges, 2 faces each, slot pairing == oid_nb   '
          f'{"PASS" if ok else "FAIL"}')

    # 2. CSP — phi_f in S3, phi_0 = identity
    perms = list(itertools.permutations(range(3)))
    sols = []
    for assign in itertools.product(range(6), repeat=7):
        phi = (perms[0],) + tuple(perms[i] for i in assign)
        if all(phi[fa][sa] == phi[fb][sb] for (fa, sa), (fb, sb) in pairs):
            sols.append(phi)
    sat = len(sols) > 0
    ok &= sat
    print(f'csp       : {len(sols)} solution(s) with phi_0 fixed '
          f'(x6 global relabellings)   {"SAT — PASS" if sat else "UNSAT"}')

    # 3. MODE LAW — phi_f = id (mode 0) / (1 2) swap (mode 1)
    swap = (0, 2, 1)
    law = tuple(perms[0] if MO[f] == 0 else swap for f in range(8))
    law_ok = all(law[fa][sa] == law[fb][sb] for (fa, sa), (fb, sb) in pairs)
    ok &= law_ok and law in [tuple(s) for s in sols]
    print(f'mode law  : phi = id on mode-0 / (1 2) on mode-1 octants '
          f'(oid_mo) satisfies all 12 hinges   '
          f'{"PASS" if law_ok else "FAIL"}')
    if sat:
        print(f'gauge     : {6 * len(sols)} namings total = 6 global '
              f'relabellings x {len(sols)} gauge (equator is class 0 '
              f'everywhere, so e.g. a hemisphere-wide (1 2) swap is '
              f'invisible to the equatorial hinges); the mode law is '
              f'the canonical pick — hex9\'s own c2 apparatus '
              f'("nb.c2=1 == self.c2=2")')
    return ok


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
