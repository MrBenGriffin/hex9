# Part of the Hex9 (H9) Project
# Copyright ©2026, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""The canonical form: deriving the five-symbol enumeration from H9O.

e4h_closed_form.py pinned the rule to a unique table (mod gauge) with
four base cycles as residual data. Hypothesis (2026-08-03): the bases
are not data at all — they derive from hex9's own adjacency:

    base(o) at c2 = 0  ==  reverse( axis-digit of oid_nb[o][slot]
                                    for slot in 0..2 )

i.e. each octant's cycle is the AXES OF ITS THREE NEIGHBOURS in
c2-slot order. Semantically: an edge-class points at a neighbour
octant, and the digit names that neighbour's octahedral axis. The
axis->digit bijection is the one true gauge (a global relabelling);
everything else — S1 anchor, S2 shared q, S3 c2-rotation, S4 own-axis
omission, S5 antipodal swap — then follows from oid_nb + oid_cmp.

Checks:
  1. DERIVE  — build the table purely from H9O (no CP-SAT) under the
               witness gauge; compare with the pinned unique table.
  2. CLEAN   — the derived table against every gathered depth-1..4
               constraint set (all octants + seam + vertex + poles).
  3. DENSE-4 — a dense micro-patch at depth 4 so complete trapezoid
               triples exist there too (closes the earlier caveat).

FINDINGS (2026-08-03): all three checks pass. The H9O-derived bases
equal the CP-SAT-pinned unique table's, the derived table is CLEAN
against 41,367 gathered constraints (depths 1-4), and a dense depth-4
micro-patch (406 complete triples) is CLEAN — the depth-4 caveat is
closed. The five-symbol enumeration is generated entirely by hex9's
own constants: S1-S3 mechanics + oid_nb (neighbour axes, reversed, in
c2-slot order) + oid_cmp (antipodes); the sole gauge is the global
axis->digit bijection. Ben: "oid_nb — literally no raising of
eyebrows at all." The digit of an edge child names the octahedral
axis of the neighbour octant its class points at.

Run:  python experimental/e4h_canonical.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from e4h_closed_form import AXIS, MISS_W, formula  # noqa: E402
import e4h_stability  # noqa: E402
from e4h_stability import gather  # noqa: E402


def h9o_bases(gauge=MISS_W):
    """base(o) = reversed neighbour-axis digits in c2-slot order."""
    from hhg9.h9.constants import H9O
    dax = {a: gauge[a] for a in range(4)}
    bases = {}
    for o in range(8):
        nb = [int(H9O.oid_nb[o][s]) for s in range(3)]
        trip = [dax[AXIS[n]] for n in nb]
        bases[o] = tuple(reversed(trip))
    return bases


def main():
    from hhg9 import Registrar
    reg = Registrar()

    # 1. DERIVE — compare with the CP-SAT-pinned unique table
    bases = h9o_bases()
    print('H9O-derived bases (p,q,r at c2=0):')
    for o in range(8):
        print(f'  oid{o}: {bases[o]}')
    pinned = {0: (5, 2, 3), 1: (3, 4, 2), 2: (4, 3, 5), 3: (2, 5, 4),
              4: (5, 2, 4), 5: (3, 4, 5), 6: (4, 3, 2), 7: (2, 5, 3)}
    same = all(bases[o] == pinned[o] for o in range(8))
    print(f'equals the pinned unique table\'s bases: '
          f'{"YES — bases derive from oid_nb" if same else "NO"}')

    # 2. CLEAN — derived table vs all gathered constraints
    data = [gather(reg, d) for d in (1, 2, 3, 4)]
    viol = tot = 0
    for sides, tris in data:
        nd = {}
        for node, (ca, cb) in sides.items():
            va = formula(ca[0], ca[1], ca[2], ca[4], bases)
            vb = formula(cb[0], cb[1], cb[2], cb[4], bases)
            viol += va != vb
            nd[node] = va
            tot += 1
        for tri in tris:
            vs = [nd[n] for n in tri if n in nd]
            viol += len(set(vs)) != len(vs)
            tot += 1
    print(f'derived table vs depth-1..4 constraints: '
          f'{"CLEAN" if viol == 0 else f"{viol} violations"} '
          f'({tot} constraints)')

    # 3. DENSE-4 — complete depth-4 triples on a micro-patch
    saved = e4h_stability.PATCHES
    e4h_stability.PATCHES = [('dense4', 19.96, 20.11, 34.96, 35.11)]
    try:
        sides4, tris4 = gather(reg, 4, n=110)
    finally:
        e4h_stability.PATCHES = saved
    v4 = 0
    nd = {}
    for node, (ca, cb) in sides4.items():
        va = formula(ca[0], ca[1], ca[2], ca[4], bases)
        vb = formula(cb[0], cb[1], cb[2], cb[4], bases)
        v4 += va != vb
        nd[node] = va
    for tri in tris4:
        vs = [nd[n] for n in tri if n in nd]
        v4 += len(set(vs)) != len(vs)
    print(f'dense depth-4 micro-patch: {len(sides4)} nodes, '
          f'{len(tris4)} complete triples — '
          f'{"CLEAN" if v4 == 0 else f"{v4} violations"}')
    return same and viol == 0 and v4 == 0


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
