# Copyright 2024 Ben Griffin
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

"""Half-hexagon Puzzle Solver
This solves filling a space with a set of defined half-hexagons.

A half-hexagon is defined by a hexagon bisected along it's long diameter,
or three equilateral triangles adjacent to each other. ⬣

The puzzle here is to enumerate the ways in which nine half-hexagons can tile a half-hexagon.

"""
import numpy as np
from ortools.sat.python import cp_model


class SolutionPrinter(cp_model.CpSolverSolutionCallback):

    def __init__(self, ground, pos_items, presence):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.hashes = set()
        self.ground = ground
        self.pos_items = pos_items
        self.presence = presence
        self.solutions = 0
        self.ref = [c for c in self.ground]
        self.ref.sort(key=lambda e: (e[1], e[0]))  # to get string values.

    def on_solution_callback(self):
        p = {}
        for c in self.ground:
            for x in self.pos_items[c]:
                if self.Value(self.presence[x]):
                    p[c] = f'{x[0]}'

        fz = ''.join([p[x] for x in self.ref])
        if fz not in self.hashes:
            self.solutions += 1
            self.hashes.add(fz)
            print(f'{len(self.hashes):02d}:    {fz}')
        return True


class HHGSolver:
    """Solves all half-hexagon tilings."""
    def __init__(self):
        self.allow_gaps = False
        self.maximising = False
        self.status = cp_model.UNKNOWN
        self.solver = cp_model.CpSolver()
        self.solver.parameters.enumerate_all_solutions = True
        self.solver.parameters.cp_model_probing_level = 2000
        # self.solver.parameters.num_search_workers = 20
        self.solver.parameters.cp_model_presolve = True
        # self.solver.parameters.log_search_progress = True
        self.solver.parameters.linearization_level = 20000
        self.solver.parameters.max_presolve_iterations = 20000
        self.solver.parameters.cp_model_probing_level = 20000

        self.pieces = None
        self.ground = None
        self.pos_items = None
        self.presence = {}

    def process(self, pieces, ground):
        """
        Given a problem and the pieces we construct a SAT model and then solve it.
        """
        model = cp_model.CpModel()
        self.pieces = pieces
        self.ground = ground

        """
        Also add it's coordinates into a placement dict so that we can test for overlaps later.
        `placement` is a dict keyed by each legally oriented piece and offset showing the spaces covered. 
        We will invert this to hold the items by position.
        """
        placement = {}
        for named_fix in pieces.lib:
            for point in self.ground:
                if pieces.legal(named_fix, self.ground, point):
                    self.presence[tuple((named_fix, *point))] = model.NewBoolVar(f"{named_fix}:{point}")
                    placement[tuple((named_fix, *point))] = pieces.translate(named_fix, point)

        """
        Compose pos_items - at each legal position in the space, identify which placement(s) can cover it.
        So this is the invert of placement.
        We will use this for constraint setting and for rendering.
        """
        self.pos_items = {x: set() for x in self.ground}
        for fixed in self.presence.keys():
            pp = placement[fixed]
            for point in pp:
                self.pos_items[point].add(fixed)

        """
        For each legal position in the space constrain the sum of pos_items to 1 (1 piece per x,y).
        """
        for pt_points in self.pos_items.values():
            items = [self.presence[i] for i in pt_points]
            model.Add(sum(items) == 1)

        """
        Can now solve.
        """
        solution_printer = SolutionPrinter(self.ground, self.pos_items, self.presence)
        self.solver.enumerate_all_solutions = True
        self.status = self.solver.Solve(model, solution_callback=solution_printer)

    def show(self) -> bool:
        if self.status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            if self.maximising:
                print(f"Solver places {int(self.solver.ObjectiveValue())} pieces in {self.solver.WallTime()}s")
            else:
                print(f"Solved challenge in {self.solver.WallTime()}s")
            return True
        else:
            if self.status == cp_model.INFEASIBLE:
                print(f"Solver says the challenge is infeasible after {self.solver.WallTime()}s.")
            else:
                print(f"Solver ran out of time.")
            return False


class ShapeCollection:

    def __init__(self, points):
        self.lib = points
        self.names = set(points.keys())

    def legal(self, key: tuple, ground: set, offs: tuple) -> bool:
        """
        Given an (x,y) offset, and a 'space' of legal points check to see if this shape can
        be legally placed in it.
        """
        proj = self.translate(key, offs)
        return False if not proj else proj.issubset(ground)

    def translate(self, key: tuple, offs: tuple) -> None | set:
        """
            return the normalised shape translated by offset (x,y).
        """
        pts = self.lib[key]
        for pt in pts:
            if ((pt[1] + offs[1]) % 2) != (pt[1] % 2):
                return None
        return {tuple((pt[0] + offs[0], pt[1] + offs[1])) for pt in pts}


class HHSet(ShapeCollection):
    def __init__(self):
        _half_hex = {
            "0": {(0, 0), (0, 1), (0, 2)},
            "1": {(0, 1), (1, 0), (1, 1)},
            "2": {(0, 1), (1, 0), (0, 2)},
            "3": {(0, 1), (0, 2), (0, 3)},
            "4": {(0, 0), (0, 1), (1, 0)},
            "5": {(1, 1), (1, 2), (0, 3)}
        }
        super().__init__(_half_hex)


class HHSolver:
    """Solves half-hexagon tilings that allow hexagon tilings."""
    def __init__(self):
        self.allow_gaps = False
        self.maximising = False
        self.solver = cp_model.CpSolver()
        self.solver.parameters.enumerate_all_solutions = True
        self.solver.parameters.cp_model_probing_level = 2000
        # self.solver.parameters.num_search_workers = 20
        self.solver.parameters.cp_model_presolve = True
        # self.solver.parameters.log_search_progress = True
        self.solver.parameters.linearization_level = 20000
        self.solver.parameters.max_presolve_iterations = 20000
        self.solver.parameters.cp_model_probing_level = 20000
        self.pieces = None
        self.ground = None
        self.pos_items = None
        self.presence = {}
        self.status = cp_model.UNKNOWN

    def get_neighbors(self, pt):
        """
        Strict neighbour logic for the triangular grid:
        mode 0 triangles point 'down'.
        mode 1 triangles point 'up'.
        """
        x, y = pt
        if y % 2 == 0:  # Even row
            return [(x, y + 1), (x - 1, y + 1), (x, y - 1)]
        else:  # Odd row
            return [(x, y - 1), (x + 1, y - 1), (x, y + 1)]

    def get_long_side_edges(self, piece_points):
        """
        A half-hexagon (3 cells) has a 'middle' cell and two 'wings'.
        The long side consists of the external edges of the two wings
        that share the same direction vector.
        """
        pts = list(piece_points)
        # Identify 'wing' triangles (degree 1 in the piece's internal graph)
        adj = {p: [n for n in self.get_neighbors(p) if n in piece_points] for p in pts}
        wings = [p for p in pts if len(adj[p]) == 1]

        if len(wings) != 2:
            return []

        # Map external edge directions for both wings
        w1_ext = {(n[0] - wings[0][0], n[1] - wings[0][1]): (wings[0], n)
                  for n in self.get_neighbors(wings[0]) if n not in piece_points}
        w2_ext = {(n[0] - wings[1][0], n[1] - wings[1][1]): (wings[1], n)
                  for n in self.get_neighbors(wings[1]) if n not in piece_points}

        # The Long Side is the direction shared by both wings
        for d, edge1 in w1_ext.items():
            if d in w2_ext:
                return [edge1, w2_ext[d]]
        return []

    def process(self, pieces, ground, long_side_matched=False):
        model = cp_model.CpModel()
        self.ground = ground
        self.presence = {}

        # 1. Map legal placements
        placement_map = {}
        for named_fix in pieces.lib:
            for point in self.ground:
                if pieces.legal(named_fix, self.ground, point):
                    var = model.NewBoolVar(f"{named_fix}:{point}")
                    idx = (named_fix, *point)
                    self.presence[idx] = var
                    placement_map[idx] = pieces.translate(named_fix, point)

        # 2. Coverage constraint
        self.pos_items = {pt: set() for pt in self.ground}
        for idx, points in placement_map.items():
            for pt in points:
                self.pos_items[pt].add(idx)

        for pt, items in self.pos_items.items():
            model.Add(sum(self.presence[i] for i in items) == 1)

        # 3. THE CONSTRAINT: Long-to-Long Handshake (hextile constraint).
        if long_side_matched:
            # Iterate over every internal boundary between two triangles
            internal_edges = set()
            for u in self.ground:
                for v in self.get_neighbors(u):
                    if v in self.ground and (v, u) not in internal_edges:
                        internal_edges.add((u, v))

            for (u, v) in internal_edges:
                # Placements at 'u' that use edge (u->v) as part of a LONG side
                long_at_u = []
                for idx in self.pos_items[u]:
                    if (u, v) in self.get_long_side_edges(placement_map[idx]):
                        long_at_u.append(self.presence[idx])

                # Placements at 'v' that use edge (v->u) as part of a LONG side
                long_at_v = []
                for idx in self.pos_items[v]:
                    if (v, u) in self.get_long_side_edges(placement_map[idx]):
                        long_at_v.append(self.presence[idx])

                # Logic: If U is long, V must be long. If U is short, V must be short.
                model.Add(sum(long_at_u) == sum(long_at_v))

        # Solve
        solution_printer = SolutionPrinter(self.ground, self.pos_items, self.presence)
        self.solution_printer = solution_printer
        self.status = self.solver.Solve(model, solution_callback=solution_printer)

    def show(self):
        if self.status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            print(f"Solved challenge in {self.solver.WallTime():.2f}s")
            return True
        return False


# ── Chirality / mirror analysis ───────────────────────────────────────────────

# Vertical (left-right) mirror of the half-hexagon space maps orientations as follows:
#   0 (SW base) ↔ 2 (SE base)   — reflected across vertical axis
#   3 (NE base) ↔ 5 (NW base)   — reflected across vertical axis
#   1 (S base)  → 1  (self)      — lies on the mirror axis
#   4 (N base)  → 4  (self)      — lies on the mirror axis
ORIENT_MIRROR = {'0': '2', '2': '0', '3': '5', '5': '3', '1': '1', '4': '4'}


def mirror_solution(fz, ref):
    """Return the vertical-mirror image of solution string fz.

    Vertical mirror of the half-hexagon (long side up):
      - x-flip within each row: x → max_x[y] - x  (row order unchanged)
      - orientation relabelling via ORIENT_MIRROR
    """
    pos_to_char = {ref[i]: fz[i] for i in range(len(fz))}
    max_x = {}
    for x, y in ref:
        max_x[y] = max(max_x.get(y, 0), x)
    mirrored = {(max_x[y] - x, y): ORIENT_MIRROR[c]
                for (x, y), c in pos_to_char.items()}
    return ''.join(mirrored[pos] for pos in ref)


def find_mirror_pairs(solutions, ref):
    """Pair each solution with its vertical mirror.

    Returns list of (a, b) tuples where b = mirror(a).
    Self-mirror solutions and orphans (mirror not in solutions) are returned as (a, a).
    """
    sol_set = set(solutions)
    paired = set()
    pairs = []
    for s in sorted(sol_set):
        if s in paired:
            continue
        m = mirror_solution(s, ref)
        if m == s or m not in sol_set:
            pairs.append((s, s))
            paired.add(s)
        else:
            pairs.append((s, m))
            paired.add(m)
            paired.add(s)
    return pairs

def find_pieces(sol, positions):
    """Group triangle positions into connected same-orientation components (= placed pieces).

    Returns a list of frozensets, one per piece, sorted by the position of each piece's
    first triangle in ref order so the piece index is stable across solutions.
    """
    pos_set = set(positions)
    pos_to_char = dict(zip(positions, sol))
    visited = set()
    pieces = []
    for pos in positions:
        if pos in visited:
            continue
        c = pos_to_char[pos]
        component = set()
        queue = [pos]
        while queue:
            p = queue.pop()
            if p in visited:
                continue
            visited.add(p)
            component.add(p)
            x, y = p
            nbs = [(x, y+1), (x-1, y+1), (x, y-1)] if y % 2 == 0 else [(x, y-1), (x+1, y-1), (x, y+1)]
            for nb in nbs:
                if nb in pos_set and nb not in visited and pos_to_char[nb] == c:
                    queue.append(nb)
        pieces.append(component)
    return pieces


PIECE_COLOURS = ['#2ecc71', '#3498db', '#fb9837', '#1abc9c', '#5bacd8', '#f1c40f', '#067e22', '#091e63', '#007d8b']

def display_solution(name, positions, sols, pairs):
    """Write all solutions into a single SVG grid, coloured by piece identity."""
    # swaps = {'0': '3', '1': '1', '2': '5', '3': '0', '4': '4', '5': '2'}
    swaps = {'0': '2', '1': '1', '2': '0', '3': '5', '4': '4', '5': '3'}
    lefts = {l for l, r in pairs if l != r}
    rights = {r for l, r in pairs if l != r}
    r_only = rights - lefts
    non_rgt = sols - r_only
    mir = {b: ''.join([swaps[t] for t in b]) for b in r_only}
    solutions = {s: s for s in sols}
    solutions |= {s: mir[s] for s in mir}

    t_wid = 30  # triangle side length in pixels
    h = t_wid * np.sqrt(3) / 2
    gap, label_h = 10, 14

    def centroid(x, y):
        cx = (x + 0.5 * (1 + (y + 1) // 2)) * t_wid
        cy = (1 + y + y // 2) * h / 3
        return cx, cy

    def tri_vertices(x, y):
        cx, cy = centroid(x, y)
        if y % 2 == 0:  # ∇ — apex at bottom
            return [(cx - t_wid / 2, cy - h / 3),
                    (cx + t_wid / 2, cy - h / 3),
                    (cx,             cy + 2 * h / 3)]
        else:           # △ — apex at top
            return [(cx - t_wid / 2, cy + h / 3),
                    (cx + t_wid / 2, cy + h / 3),
                    (cx,             cy - 2 * h / 3)]

    margin = 5
    all_verts = [v for pos in positions for v in tri_vertices(*pos)]
    lx = min(v[0] for v in all_verts) - margin
    ly = min(v[1] for v in all_verts) - margin
    t_w = max(v[0] for v in all_verts) + margin - lx
    t_h = max(v[1] for v in all_verts) + margin - ly
    # Bounding-box centre in raw pixel coords — rotation pivot keeps shape in its cell
    ct = (lx + t_w / 2, ly + t_h / 2)
    PAD = 4  # padding around pair background rects (also expands viewBox)

    # Normal pairs first, self-mirrors last → pairs never start at an odd column
    columns = 6  # even, 3 pairs per row
    sorted_pairs = [(a, b) for a, b in pairs if a != b] + [(a, b) for a, b in pairs if a == b]
    ordered_keys = []
    pair_starts = []
    for a, b in sorted_pairs:
        pair_starts.append(len(ordered_keys))
        if a in sols:
            ordered_keys.append(a)
        if b != a and b in sols:
            ordered_keys.append(b)

    if not ordered_keys:
        return

    n_rows = (len(ordered_keys) + columns - 1) // columns
    svg_w = columns * (t_w + gap) - gap
    svg_h = n_rows * (t_h + label_h + gap) - gap

    els = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w + 2*PAD:.1f}" height="{svg_h + 2*PAD:.1f}" '
           f'viewBox="{-PAD} {-PAD} {svg_w + 2*PAD:.1f} {svg_h + 2*PAD:.1f}">',
           f'  <rect x="{-PAD}" y="{-PAD}" width="{svg_w + 2*PAD:.1f}" height="{svg_h + 2*PAD:.1f}" fill="white"/>']

    # Pair background rects — drawn before solutions so they sit behind
    for (a, b), si in zip(sorted_pairs, pair_starts):
        in_sols = (a in sols) + (b != a and b in sols)
        span = max(1, in_sols)
        ci_s, ri_s = si % columns, si // columns
        rx = ci_s * (t_w + gap) - PAD
        ry = ri_s * (t_h + label_h + gap) - PAD
        rw = span * (t_w + gap) - gap + 2 * PAD
        rh = t_h + label_h + 2 * PAD
        els.append(f'  <rect x="{rx:.1f}" y="{ry:.1f}" width="{rw:.1f}" height="{rh:.1f}" '
                   f'rx="3" fill="#f0f0f0" stroke="#ccc" stroke-width="0.5"/>')

    for i, sol in enumerate(ordered_keys):
        ci, ri = i % columns, i // columns
        tx = ci * (t_w + gap) - lx
        ty = ri * (t_h + label_h + gap) + label_h - ly
        lbl_x = ci * (t_w + gap) + t_w / 2
        lbl_y = ri * (t_h + label_h + gap) + label_h - 3
        els.append(f'  <text x="{lbl_x:.1f}" y="{lbl_y:.1f}" font-size="10" '
                   f'text-anchor="middle" font-family="monospace">{sol[::-1]}</text>')
        t_rot = f'' if sol != solutions[sol] else f' rotate(180 {ct[0]:.2f} {ct[1]:.2f})'

        els.append(f'  <g transform="translate({tx:.2f} {ty:.2f}){t_rot}">')

        # Build pos→piece_index map for this solution
        # pieces = find_pieces(sol, positions)
        # pos_to_piece = {pos: idx for idx, piece in enumerate(pieces) for pos in piece}
        for c, pos, col_to_use in zip(sol, positions, sol):  # solutions[sol]
            # colour = PIECE_COLOURS[pos_to_piece[pos] % len(PIECE_COLOURS)]
            colour = PIECE_COLOURS[int(col_to_use)]
            cx, cy = centroid(*pos)
            verts = tri_vertices(*pos)
            pts_str = ' '.join(f'{px:.2f},{py:.2f}' for px, py in verts)
            els.append(f'    <polygon points="{pts_str}" fill="{colour}" stroke="white" stroke-width="0.5"/>')
            txt_tfm = f' transform="rotate(180 {cx:.2f} {cy:.2f})"' if t_rot else ''
            els.append(f'    <text x="{cx:.1f}" y="{cy + 4:.1f}" font-size="8" '
                       f'text-anchor="middle" fill="white" font-family="monospace"{txt_tfm}>{c}</text>')
        els.append('  </g>')

    els.append('</svg>')
    with open(f'{name}.svg', 'w') as f:
        f.write('\n'.join(els))


def example(name, space_set, long_side_matched=False, shapes=None):
    shapes = HHSet() if shapes is None else shapes
    solver = HHSolver()
    solver.process(shapes, space_set, long_side_matched=long_side_matched)
    solver.show()
    ref = solver.solution_printer.ref
    pairs = find_mirror_pairs(solver.solution_printer.hashes, ref)
    display_solution(name, ref, solver.solution_printer.hashes, pairs)
    print(f"\n{len(pairs)} chiral pair(s):")
    for i, (a, b) in enumerate(pairs):
        tag = "  [self-mirror]" if a == b else ""
        print(f"  {i+1:2d}: {a}  ↔  {b}{tag}")


if __name__ == '__main__':
    # Limited tilings to those where the long-edge of the half-hexagons meet.
    # 9-piece half-hexagon tiling (27 triangular cells = 9 × 3): H9 proof space
    # 18 solutions (confirmed) — the relevant H9 result.
    # H9 solutions (mirror pair): results 09 and 48 from the 49-solution run:
    #   044044043040230225322512511
    #   442442425420250205300130113
    #   The numbers represent possible orientations (or rotations) of a half-hexagon, where
    #   0 has its *base* towards the south-west, 1 to the south, etc., with 5 towards the north-west.
    #   The space to fill is a half-hexagon with the long side north (orientation 4).
    #   The fill is left to right, top to bottom, treating each parity as its own row.
    #   442442|42542|02502|0530|0130|113
    """
       4   4   2   4   4   2  <-- downward pointing triangles
         4   2   5   4   2    <-- upward pointing triangles
         0   2   5   0   2    <-- downward pointing triangles
           0   5   3   0      <-- upward pointing triangles
           0   1   3   0      <-- downward pointing triangles
             1   1   3        <-- upward pointing triangles
    """
    # Both use orientations {0,2,4} twice each and {1,3,5} once each —
    # the even/odd split reflects the diploid crystallographic structure.
    # Fundamental equilateral triangle.
    # Space: 27-cell Half-hexagon
    hh = set((x, y) for y in range(0, 6) for x in range(0, 6)) - {(3, 5), (4, 3), (4, 4), (4, 5), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5)}

    print('–' * 90)
    print('Equilateral: Pack a 9-cell equilateral with 3x(3-cell half-hexagons): 2 solutions: 1 chiral pair')
    # Construct equilateral shapes.
    eqs = ShapeCollection({
        "0": {(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (1, 2), (2, 0)},
        "1": {(0, 5), (1, 3), (1, 4), (1, 5), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5)}
    })
    # Space: 9-cell equilateral.
    eq = eqs.lib['0']
    example('eq.hh', eq)
    print('–' * 90)

    print('Half-hexagon: Pack a 27-cell half-hexagon with 3x(9-cell equilateral): 1 solution.')
    example('hh.eq', hh, False, eqs)
    print('–' * 90)

    print('Half-hexagon: Pack a 27-cell half-hexagon with 9x(3-cell half-hexagons): 49 solutions: 24 chiral pairs, 1 self-mirror.')
    example('hh.hh', hh)
    print('–' * 90)

    print('Half-hexagon: Pack a 27-cell half-hexagon with 9x(3-cell half-hexagons); MUST hex-tile: 18 solutions: 9 chiral pairs.')
    # Uses hh: 27-cell Half-hexagon
    example('hh.hht', hh, long_side_matched=True)
    print('–' * 90)
