# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
H3 hierarchy: exact in the index, divergent in geometry — a demonstrator.

Backs the §12 nesting claim in the Hex9 paper (`docs/paper-draft.md`) and the
figures `docs/h3/001.jpg`–`003.jpeg`.  Two independent facts about H3's
aperture-7 subdivision:

  PART A — the INDEX hierarchy is exact and deterministic.
    `cell_to_parent` is single-valued and transitive, so a shared parent always
    implies a shared grandparent.  (This is why the paper must *not* claim
    otherwise in index terms — an earlier draft did, and a reviewer rightly
    flagged it.)

  PART B — the GEOMETRY does not nest, and the divergence compounds with depth.
    Counting a cell H's index-descendants by their area overlap with H:
      * 1 generation down  — only the centre child is inside; the other six straddle.
      * 2 generations down — boundary grandchildren split ~half-and-half (overlap ≈ 0.5).
      * 3 generations down — some index-descendants lie ENTIRELY outside H
                             (overlap = 0), though their address extends H's.
      * 4 generations down — many more do.
    So H3's hierarchy is an exact index laid over grids that never geometrically
    refine one another.  (The exact counts are cell-dependent; the *existence* of
    entirely-outside descendants by the third generation is the universal claim.)

This is a comparison demonstrator — it uses only the reference `h3` and `shapely`
libraries, no Hex9 code.

Run:
    python docs/h3/h3_nesting.py                 # print Part A + Part B table
    python docs/h3/h3_nesting.py --figure        # also write h3_nesting.png
    python docs/h3/h3_nesting.py --res 6 --lat 51.48 --lng 0 --gens 4

Last tested: 29 Jun 2026 — h3 4.x, shapely 2.x.
"""
import argparse

import h3
from shapely.geometry import Polygon
from shapely.ops import unary_union


def cell_polygon(cell: str) -> Polygon:
    """Lon/lat polygon of an H3 cell boundary (x=lng, y=lat)."""
    return Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(cell)])


def part_a_index_is_exact(lat: float, lng: float, res: int) -> None:
    """Index parent is single-valued + transitive ⇒ shared parent ⇒ shared grandparent."""
    base = h3.latlng_to_cell(lat, lng, res)
    siblings = h3.cell_to_children(base, res + 1)            # all share parent `base`
    grandparents = {h3.cell_to_parent(c, res - 1) for c in siblings}

    # transitivity over a wide sample: parent(parent(c)) == grandparent(c)
    sample = h3.cell_to_children(h3.latlng_to_cell(0.0, 0.0, res - 3), res + 1)
    transitive = all(
        h3.cell_to_parent(h3.cell_to_parent(c, res), res - 1)
        == h3.cell_to_parent(c, res - 1)
        for c in sample
    )

    print('PART A — index hierarchy (exact, deterministic)')
    print(f'  {len(siblings)} children of one cell share exactly '
          f'{len(grandparents)} grandparent  -> shared parent implies shared '
          f'grandparent: {len(grandparents) == 1}')
    print(f'  parent(parent(c)) == grandparent(c) for all {len(sample)} sampled '
          f'cells: {transitive}')
    print()


def part_b_geometry_diverges(lat: float, lng: float, res: int, gens: int) -> None:
    """Tabulate index-descendants of H by area-overlap with H, generation by generation."""
    h0 = h3.latlng_to_cell(lat, lng, res)
    H = cell_polygon(h0)

    print('PART B — geometric containment (divergent, compounding)')
    print(f'  anchor cell H = {h0}  (res {res})')
    print(f'  {"down":>4} {"descendants":>12} {"inside H":>9} {"straddle":>9} '
          f'{"OUTSIDE H":>10} {"min overlap":>12}')
    for k in range(1, gens + 1):
        desc = h3.cell_to_children(h0, res + k)
        fracs = []
        for d in desc:
            D = cell_polygon(d)
            inter = D.intersection(H).area
            fracs.append(inter / D.area if D.area else 0.0)
        outside = sum(f < 1e-9 for f in fracs)
        inside = sum(f > 0.999 for f in fracs)
        straddle = len(fracs) - outside - inside
        print(f'  {k:>4} {len(desc):>12} {inside:>9} {straddle:>9} '
              f'{outside:>10} {min(fracs):>12.3f}')
    print()
    print('  Note: counts are cell-dependent; the universal claim is the *existence*')
    print('  of entirely-outside index-descendants by the third generation.')


def write_figure(lat: float, lng: float, res: int, gens: int, path: str) -> None:
    """Optional: H and its `gens`-down index-descendants, coloured by containment."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MplPolygon
    except ImportError:
        print('  (matplotlib not available — skipping figure)')
        return

    h0 = h3.latlng_to_cell(lat, lng, res)
    H = cell_polygon(h0)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.add_patch(MplPolygon(list(H.exterior.coords), closed=True, fill=False,
                            edgecolor='#222222', linewidth=2.5, zorder=3))
    # CVD-safe: blue=inside, gold=straddle, red=outside (luminance-separated).
    colours = {'in': '#4f86c6', 'straddle': '#efdca0', 'out': '#c0392b'}
    for d in h3.cell_to_children(h0, res + gens):
        D = cell_polygon(d)
        frac = D.intersection(H).area / D.area if D.area else 0.0
        key = 'out' if frac < 1e-9 else 'in' if frac > 0.999 else 'straddle'
        ax.add_patch(MplPolygon(list(D.exterior.coords), closed=True,
                                facecolor=colours[key], edgecolor='#555555',
                                linewidth=0.4, alpha=0.85, zorder=2))
    ax.autoscale()
    ax.set_aspect('equal')
    ax.set_title(f'H3 res-{res} cell and its index-descendants {gens} levels down\n'
                 f'blue=inside  gold=straddle  red=entirely outside')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f'  Saved figure: {path}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--lat', type=float, default=40.0)
    ap.add_argument('--lng', type=float, default=-3.0)
    ap.add_argument('--res', type=int, default=5, help='resolution of the anchor cell H')
    ap.add_argument('--gens', type=int, default=4, help='generations below H to tabulate')
    ap.add_argument('--figure', action='store_true', help='also write h3_nesting.png')
    args = ap.parse_args()

    part_a_index_is_exact(args.lat, args.lng, args.res)
    part_b_geometry_diverges(args.lat, args.lng, args.res, args.gens)
    if args.figure:
        import os
        write_figure(args.lat, args.lng, args.res, 3,
                     os.path.join(os.path.dirname(__file__), 'h3_nesting.png'))
