# Copyright 2026 Ben Griffin
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Machine verification of the Hex9 half-hexagon tiling chain.

Verifies, end to end, the claims in docs/paper_notes.md
(Mathematical Foundations, Layers 2-3):

  V0. The 6 piece shapes are valid half-hexagons (3 cells, connected,
      two wings sharing a long-side direction) and pairwise distinct.
  V1. The 9-cell equilateral triangle admits exactly 2 tilings by three
      half-hexes — a chiral pair (L and R).
  V2. The 27-cell half-hexagon admits exactly 1 tiling by three 9-cell
      equilateral triangles, so its 3-triangle decomposition is unique.
  V3. The 27-cell half-hexagon admits exactly 49 distinct tilings by
      nine half-hexes: 24 chiral pairs + 1 self-mirror solution.
      Includes a hash-collision check: no two distinct placement-sets
      share an orientation string (validates the dedup in halfhex.py).
  V4. Constraint A (hexagon affordance: every long edge meets a long
      edge): exactly 18 of the 49 survive — 9 chiral pairs.
  V5. Constraint B (3-triangle structure): exactly 8 of the 49 — 4
      chiral pairs — decompose into the 3 equilateral sub-triangles of
      V2, each tiled by an L or R from V1.
  V6. A ∩ B: exactly 2 solutions (1 chiral pair) satisfy both — the
      Hex9 solutions, matching the strings recorded in halfhex.py.

Run from this directory:  python3 halfhex_verify.py
"""

from ortools.sat.python import cp_model

from halfhex_further import (
    HHSet, HHSolver, ShapeCollection, mirror_solution, find_mirror_pairs,
)

# The recorded Hex9 mirror pair (results 09 and 48 of the 49-solution run).
H9_SOLUTIONS = {
    '044044043040230225322512511',
    '442442425420250205300130113',
}

# 27-cell half-hexagon space (long side north), as in halfhex.py __main__.
HH_SPACE = set((x, y) for y in range(6) for x in range(6)) - {
    (3, 5), (4, 3), (4, 4), (4, 5),
    (5, 1), (5, 2), (5, 3), (5, 4), (5, 5)}

# The two 9-cell equilateral triangle shapes (L/R chiralities of the
# 3-half-hex equilateral), as in halfhex_further.py __main__.
EQ_SHAPES = ShapeCollection({
    "0": {(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (1, 2), (2, 0)},
    "1": {(0, 5), (1, 3), (1, 4), (1, 5), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5)},
})


class Collector(cp_model.CpSolverSolutionCallback):
    """Collects every solution as (placement-set, orientation string)."""

    def __init__(self, presence, placement_map, ref):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.presence = presence
        self.placement_map = placement_map
        self.ref = ref
        self.solutions = []

    def on_solution_callback(self):
        chosen = [idx for idx, var in self.presence.items() if self.Value(var)]
        cell_orient = {}
        for idx in chosen:
            for pt in self.placement_map[idx]:
                cell_orient[pt] = idx[0]
        fz = ''.join(cell_orient[c] for c in self.ref)
        self.solutions.append((frozenset(chosen), fz))


def enumerate_tilings(shapes, ground, long_side_matched=False):
    """Exact-cover enumeration, mirroring HHSolver.process, but returning
    full placement-sets rather than printing deduplicated strings."""
    helper = HHSolver()  # for get_long_side_edges / get_neighbors
    model = cp_model.CpModel()

    presence, placement_map = {}, {}
    for name in shapes.lib:
        for point in ground:
            if shapes.legal(name, ground, point):
                idx = (name, *point)
                presence[idx] = model.NewBoolVar(f"{name}:{point}")
                placement_map[idx] = shapes.translate(name, point)

    pos_items = {pt: set() for pt in ground}
    for idx, points in placement_map.items():
        for pt in points:
            pos_items[pt].add(idx)
    for pt, items in pos_items.items():
        model.Add(sum(presence[i] for i in items) == 1)

    if long_side_matched:
        internal_edges = set()
        for u in ground:
            for v in helper.get_neighbors(u):
                if v in ground and (v, u) not in internal_edges:
                    internal_edges.add((u, v))
        for (u, v) in internal_edges:
            long_at_u = [presence[i] for i in pos_items[u]
                         if (u, v) in helper.get_long_side_edges(placement_map[i])]
            long_at_v = [presence[i] for i in pos_items[v]
                         if (v, u) in helper.get_long_side_edges(placement_map[i])]
            model.Add(sum(long_at_u) == sum(long_at_v))

    ref = sorted(ground, key=lambda e: (e[1], e[0]))
    collector = Collector(presence, placement_map, ref)
    solver = cp_model.CpSolver()
    solver.parameters.enumerate_all_solutions = True
    status = solver.Solve(model, solution_callback=collector)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE), 'enumeration failed'
    return collector.solutions, placement_map, ref


def check(label, condition, detail=''):
    print(f"  {'PASS' if condition else 'FAIL'}  {label}{'  — ' + detail if detail else ''}")
    return condition


def main():
    ok = True
    hh_set = HHSet()
    helper = HHSolver()

    # ── V0: piece shapes are valid, distinct half-hexagons ────────────────
    print('V0: piece shape validity')
    normalised = set()
    for name, pts in hh_set.lib.items():
        long = helper.get_long_side_edges(pts)
        ok &= check(f"orientation {name}: 3 cells, long side found",
                    len(pts) == 3 and len(long) == 2)
        mx = min(p[0] for p in pts)
        my = min(p[1] for p in pts) // 2 * 2  # preserve row parity
        normalised.add(frozenset((x - mx, y - my) for x, y in pts))
    ok &= check('6 orientations pairwise distinct under translation',
                len(normalised) == 6)

    # ── V1: equilateral = 3 half-hexes, exactly 2 ways (chiral pair) ─────
    print('V1: 9-cell equilateral by three half-hexes')
    eq_space = EQ_SHAPES.lib['0']
    eq_sols, _, eq_ref = enumerate_tilings(hh_set, eq_space)
    eq_strings = {fz for _, fz in eq_sols}
    ok &= check('exactly 2 tilings', len(eq_sols) == 2,
                f'found {len(eq_sols)}')
    m = mirror_solution(sorted(eq_strings)[0], eq_ref)
    ok &= check('the 2 are a chiral pair', m in eq_strings and m != sorted(eq_strings)[0])

    # ── V2: half-hex = 3 equilaterals, exactly 1 way ──────────────────────
    print('V2: 27-cell half-hexagon by three 9-cell equilaterals')
    dec_sols, dec_map, _ = enumerate_tilings(EQ_SHAPES, HH_SPACE)
    ok &= check('exactly 1 decomposition', len(dec_sols) == 1,
                f'found {len(dec_sols)}')
    regions = [dec_map[idx] for idx in dec_sols[0][0]]
    ok &= check('3 regions of 9 cells covering the space',
                len(regions) == 3 and all(len(r) == 9 for r in regions)
                and set().union(*regions) == HH_SPACE)

    # ── V3: 49 tilings of the half-hex by 9 half-hexes ────────────────────
    print('V3: 27-cell half-hexagon by nine half-hexes')
    all_sols, _, ref = enumerate_tilings(hh_set, HH_SPACE)
    strings = [fz for _, fz in all_sols]
    placements = [pl for pl, _ in all_sols]
    ok &= check('exactly 49 tilings', len(all_sols) == 49,
                f'found {len(all_sols)}')
    ok &= check('no orientation-string collisions (dedup in halfhex.py is safe)',
                len(set(strings)) == len(all_sols) == len(set(placements)))
    pairs = find_mirror_pairs(set(strings), ref)
    self_mirror = [a for a, b in pairs if a == b]
    ok &= check('24 chiral pairs + 1 self-mirror', len(pairs) == 25
                and len(self_mirror) == 1,
                f'{len(pairs) - len(self_mirror)} pairs, {len(self_mirror)} self-mirror')

    # ── V4: constraint A (long-edge-to-long-edge) → 18 = 9 pairs ─────────
    print('V4: constraint A — hexagon affordance')
    a_sols, _, _ = enumerate_tilings(hh_set, HH_SPACE, long_side_matched=True)
    a_strings = {fz for _, fz in a_sols}
    ok &= check('exactly 18 survivors', len(a_sols) == 18, f'found {len(a_sols)}')
    ok &= check('survivors are a subset of the 49', a_strings <= set(strings))
    a_pairs = find_mirror_pairs(a_strings, ref)
    ok &= check('9 chiral pairs, no self-mirror',
                len(a_pairs) == 9 and all(a != b for a, b in a_pairs))

    # ── V5: constraint B (3-triangle structure) → 8 = 4 pairs ────────────
    print('V5: constraint B — decomposes into the 3 equilateral regions')
    b_strings = set()
    all_sols_full, pm49, _ = enumerate_tilings(hh_set, HH_SPACE)
    for pl, fz in all_sols_full:
        if all(any(set(pm49[idx]) <= set(region) for region in regions)
               for idx in pl):
            b_strings.add(fz)
    ok &= check('exactly 8 satisfy constraint B', len(b_strings) == 8,
                f'found {len(b_strings)}')
    b_pairs = find_mirror_pairs(b_strings, ref)
    ok &= check('4 chiral pairs, no self-mirror',
                len(b_pairs) == 4 and all(a != b for a, b in b_pairs))

    # ── V6: A ∩ B = the Hex9 chiral pair ─────────────────────────────────
    print('V6: A ∩ B — the Hex9 solutions')
    both = a_strings & b_strings
    ok &= check('exactly 2 solutions satisfy A and B', len(both) == 2,
                f'found {len(both)}')
    ok &= check('they are the recorded Hex9 mirror pair', both == H9_SOLUTIONS,
                f'{sorted(both)}')

    print()
    print('ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
