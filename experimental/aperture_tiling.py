# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Which apertures k² actually hexagonalise? An OR-Tools reading.

Settles the §6 / Felix-review item 3.2 question. A k-refinement of an octant is a
side-k equilateral triangle of k² triangular cells; a hexagonalisation needs that
triangle to tile by **half-hexagons** (3-cell pieces, the d_cells of
halfhex_verify.py V1). This sweep runs the same CP-SAT exact-cover enumeration over
the side-k triangle for a range of k and reports, for each:

  * whether it tiles by half-hexes at all (feasibility),
  * how many distinct tilings, and their chiral structure (mirror pairs +
    self-mirrors) — the V1 reading generalised.

Reference point: k=3 must reproduce V1 — exactly 2 tilings, one chiral pair.

The two competing hypotheses for the admissible-aperture condition:
  (A) `3 | k`  alone        → k = 3,6,9 tile;            aperture {9, 36, 81, …}
  (B) `3 | k` AND k odd     → only k = 3,9,15 tile cleanly; aperture {9, 81, 225, …}

If k=6 tiles with a clean chiral pair like k=3, (A) holds and A36 is real. If k=6
is infeasible, or tiles only in a degenerate/non-chiral way, (B) holds.

Run from this directory:  python3 aperture_tiling.py
"""
from ortools.sat.python import cp_model

from halfhex_further import HHSet, HHSolver
from halfhex_verify import mirror_solution, find_mirror_pairs


def tri_ground(k):
    """Side-k equilateral triangle of k² triangular cells (matches EQ_SHAPES['0'] at k=3)."""
    return set((x, y) for x in range(k) for y in range(2 * (k - x) - 1))


def enumerate_tilings(shapes, ground, max_solutions=0, time_limit=60.0):
    """Exact-cover enumeration by `shapes` over `ground`. Returns (solutions, ref, status).

    No assert: an infeasible cover returns ([], ref, INFEASIBLE). `max_solutions`>0
    caps enumeration (0 = all). Mirrors halfhex_verify.enumerate_tilings otherwise.
    """
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
        if not items:                       # a cell no piece can cover ⇒ infeasible
            return [], sorted(ground, key=lambda e: (e[1], e[0])), cp_model.INFEASIBLE
        model.Add(sum(presence[i] for i in items) == 1)

    ref = sorted(ground, key=lambda e: (e[1], e[0]))

    class C(cp_model.CpSolverSolutionCallback):
        def __init__(self):
            super().__init__()
            self.sols = []
        def on_solution_callback(self):
            chosen = [idx for idx, var in presence.items() if self.Value(var)]
            cell = {}
            for idx in chosen:
                for pt in placement_map[idx]:
                    cell[pt] = idx[0]
            self.sols.append((frozenset(chosen), ''.join(cell[c] for c in ref)))
            if max_solutions and len(self.sols) >= max_solutions:
                self.StopSearch()

    col = C()
    solver = cp_model.CpSolver()
    solver.parameters.enumerate_all_solutions = True
    solver.parameters.max_time_in_seconds = time_limit
    status = solver.Solve(model, solution_callback=col)
    return col.sols, ref, status


def main():
    hh = HHSet()
    print(f'{"k":>2} {"aperture":>8} {"cells":>6} {"tilings":>9} {"structure":>26} {"verdict":>9}')
    print('-' * 70)
    for k in range(2, 8):
        ground = tri_ground(k)
        cells = len(ground)                                  # = k²
        sols, ref, status = enumerate_tilings(hh, ground, time_limit=120.0)
        if status == cp_model.INFEASIBLE:
            print(f'{k:>2} {k*k:>8} {cells:>6} {0:>9} {"— cannot tile —":>26} {"NO":>9}')
            continue
        n = len(sols)
        strings = {fz for _, fz in sols}
        pairs = find_mirror_pairs(strings, ref)
        self_m = sum(1 for a, b in pairs if a == b)
        struct = f'{len(pairs) - self_m} pairs + {self_m} self-mirror'
        capped = ' (time-capped)' if status == cp_model.FEASIBLE and n else ''
        verdict = 'YES' if n else 'NO'
        print(f'{k:>2} {k*k:>8} {cells:>6} {n:>9} {struct:>26} {verdict:>9}{capped}')


if __name__ == '__main__':
    main()
