# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Hierarchy nesting across DGGS: iterated hierarchies decohere, contracted ones don't.

A six-panel comparison backing the §12 nesting argument in the Hex9 paper.  For
each system we take one anchor cell H, enumerate its index-descendants `gens`
resolutions finer (by that system's *own* descendant definition), and colour each
descendant by how much of its area overlaps H:

    blue  = inside H        gold = straddles H's boundary      red = entirely OUTSIDE H

The point is not "Hex9 nests and the others don't".  It is that a hierarchy built
by *iterating a single-level parent/child relation* must propagate its per-level
offset, so descendants drift out of the ancestor (H3, via the aperture-7 √7/19.1°
rotation).  A quadtree (S2) iterates cleanly because squares refine onto squares.
Hex9 avoids iteration entirely: `descendants(H, L+i)` is a direct per-layer
*contract*, not a walk down the tree — which is exactly why the split hexes (the
one place iteration would break) do not decohere.

Fairness: every panel uses that system's native descendant relation — H3/S2/A5
through their reference libraries, ISEA3H through DGGAL's primary-parent lineage
(parents[0] = upstream getZonePrimaryParent; via isea_probe.py in a subprocess),
HEALPix through nested-index arithmetic, Hex9 through its `descendants` contract.
No spatial pre-filtering (that would rig the result; cf. `tile_subgrid`, which
clips to the tile and is therefore NOT used here).

Run:
    python docs/dggs/dggs_nesting.py               # 2x3 figure -> docs/dggs/dggs_nesting.png
    python docs/dggs/dggs_nesting.py --gens 3 --lat 40 --lng -3

Result (anchor 40N,3W, gens 3): H3 12/343 and A5 22/64 outside (iterated, rotated
→ decohere); ISEA3H 20/36 outside (DGGAL primary-parent lineage: a pure
first-parent convention, not geometry-aware, so the descendant patch slides
sideways out of the anchor — the strongest decoherence measured here); S2 and
HEALPix 0 straddle / 0 outside (fully-nested quadtrees: all hierarchy readings
coincide). Hex9: exactly 729 = 9^3 descendants —
702 inside, 27 rim splits straddling by their far (mode-1) half, 0 outside.
This is the leaf-reified d_cell relation (h9_cell_ancestor: the subdivision
tree is on d_cells, which nest exactly; mode-0 reification into x_cells happens
once, at the leaf). Two superseded semantics both decohere and are recorded as
warnings: point-bin tie-breaking gave a wrong COUNT (739 ≠ 9^3, "0 outside");
composing x_cell parents level-by-level gave the right count but re-adjudicated
splits at every layer, growing deep tongues/voids (108/729 entirely outside,
exactly 1/6 of the area displaced). Overlap is classified with EDGE_TOL so that
boundary cells sharing the anchor's (curved, polygon-approximated) edge are
counted as contained, not as spurious straddlers.

Last edited: 2026-07-10 (six panels: +ISEA3H via DGGAL primary-parent lineage,
+HEALPix nested; ISEA3H 20/36 outside, HEALPix 0/64).
"""
import argparse

from shapely.geometry import Polygon

# ── per-system adapters ───────────────────────────────────────────────────────
# Each returns (anchor_polygon, [descendant_polygons], anchor_label) in lon/lat.
# `res` is each system's own resolution/level numbering; `gens` is levels finer.

def h3_adapter(lat, lng, res, gens):
    import h3
    H0 = h3.latlng_to_cell(lat, lng, res)
    anchor = Polygon([(lng_, lat_) for lat_, lng_ in h3.cell_to_boundary(H0)])
    desc = [Polygon([(lng_, lat_) for lat_, lng_ in h3.cell_to_boundary(c)])
            for c in h3.cell_to_children(H0, res + gens)]
    return anchor, desc, f'H3 r{res}->r{res + gens}'


def s2_adapter(lat, lng, res, gens):
    import s2sphere as s2
    H0 = s2.CellId.from_lat_lng(s2.LatLng.from_degrees(lat, lng)).parent(res)

    def boundary(cid):
        cell = s2.Cell(cid)
        pts = [s2.LatLng.from_point(cell.get_vertex(k)) for k in range(4)]
        return Polygon([(p.lng().degrees, p.lat().degrees) for p in pts])

    frontier = [H0]
    for _ in range(gens):                      # quadtree: iterate one level at a time
        frontier = [c for cid in frontier for c in cid.children()]
    return boundary(H0), [boundary(c) for c in frontier], f'S2 L{res}->L{res + gens}'


def a5_adapter(lat, lng, res, gens):
    import a5_fast as a5
    H0 = a5.lonlat_to_cell(float(lng), float(lat), res)
    anchor = Polygon(a5.cell_to_boundary(H0))                      # ring of (lon, lat)
    desc = [Polygon(a5.cell_to_boundary(c))
            for c in a5.cell_to_children(H0, res + gens)]
    return anchor, desc, f'A5 r{res}->r{res + gens}'


def hex9_adapter(lat, lng, res, gens):
    """Hex9 via the DIRECT per-layer contract `h9_descendants` (not iterated children).

    `h9_descendants(b_pts, res, gens)` returns every layer-(res+gens) hexagon whose
    canonical ancestor (`h9_cell_ancestor`, mode-0 convention) is the anchor —
    exactly 9^gens cells, every split hex bound to its one owner.
    `_anchor_hex_latlon` renders any UUID's hexagon as a lat/lon ring (folding
    seam-overhanging vertices into their true octant).
    """
    import numpy as np
    from shapely.validation import make_valid
    from hhg9 import Registrar, Points
    from hhg9.h9.uuid_address import h9_descendants, h9_bin_pts, _anchor_hex_latlon
    reg = Registrar()
    g_gcd, b_oct = reg.domain('g_gcd'), reg.domain('b_oct')
    b_pts = reg.project(Points(np.array([[lat, lng]], float), g_gcd), [g_gcd, b_oct])
    anchor = h9_bin_pts(b_pts, res)[0]
    desc = h9_descendants(b_pts, res, gens, reg)[0]
    target = res + gens

    def poly(u, lay):
        ring = _anchor_hex_latlon(u, lay, reg, inflate=1.0)     # (6,2) [lat, lon]
        P = Polygon([(lo, la) for la, lo in ring])
        return P if P.is_valid else make_valid(P)

    return poly(anchor, res), [poly(d, target) for d in desc], f'Hex9 L{res}->L{target}'


def healpix_adapter(lat, lng, res, gens):
    """HEALPix nested scheme: a strict aperture-4 quadtree on 12 equal-area base
    pixels, so — like S2 — all three hierarchy readings coincide.  `res` is the
    order (nside = 2**res); descendants of pixel p at +gens are the contiguous
    nested range [p·4^gens, (p+1)·4^gens)."""
    from hhg9.accel.libhex9 import backend
    backend()   # healpy's native libs break libhex9 if they load first (segfault)
    import healpy as hp
    import numpy as np
    nside = 1 << res
    p = int(hp.ang2pix(nside, lng, lat, nest=True, lonlat=True))

    def poly(nside_, pix):
        vecs = hp.boundaries(nside_, pix, step=1, nest=True)        # (3, 4) vectors
        lon, la = hp.vec2ang(vecs.T, lonlat=True)
        lon = (lon + 180.0) % 360.0 - 180.0                         # match other panels
        return Polygon(np.column_stack([lon, la]))

    fine = nside << gens
    desc = [poly(fine, q) for q in range((p << (2 * gens)), (p + 1) << (2 * gens))]
    return poly(nside, p), desc, f'HEALPix n{nside}->n{fine}'


def dggal_probe_cmd(*argv):
    """Command to run isea_probe.py in an interpreter where dggal loads: the
    current one if the wheel is healthy, else the repo's `.venv-x86` under
    Rosetta (the dggal 0.0.6 macOS arm64 wheel ships x86_64 dylibs).
    Raises NotImplementedError if neither is available."""
    import os
    import sys
    probe = os.path.join(os.path.dirname(__file__), 'isea_probe.py')
    try:
        import dggal                                                # noqa: F401
        return [sys.executable, probe, *argv]
    except ImportError:
        venv_py = os.path.join(os.path.dirname(__file__), '..', '..',
                               '.venv-x86', 'bin', 'python')
        if not os.path.exists(venv_py):
            raise NotImplementedError('dggal unavailable: pip install dggal, or on '
                                      'arm64 macOS create .venv-x86 (see isea_probe.py)')
        return ['arch', '-x86_64', venv_py, probe, *argv]


def isea3h_adapter(lat, lng, res, gens):
    """ISEA3H via DGGAL, using its primary-parent lineage (see isea_probe.py)."""
    import json
    import subprocess
    cmd = dggal_probe_cmd('--lat', str(lat), '--lng', str(lng),
                          '--res', str(res), '--gens', str(gens))
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if out.returncode:
        raise NotImplementedError(f'isea_probe failed: {out.stderr.strip()[-200:]}')
    data = json.loads(out.stdout)
    return (Polygon(data['anchor']), [Polygon(r) for r in data['descendants']],
            data['label'])


ADAPTERS = {'H3': h3_adapter, 'S2': s2_adapter, 'A5': a5_adapter,
            'ISEA3H': isea3h_adapter, 'HEALPix': healpix_adapter, 'Hex9': hex9_adapter}
# Per-system default resolution for the anchor (own numbering); tuned to comparable
# anchor sizes.  The nesting behaviour is resolution-invariant, so exact values are
# not load-bearing.
DEFAULT_RES = {'H3': 5, 'S2': 8, 'A5': 5, 'ISEA3H': 11, 'HEALPix': 8, 'Hex9': 5}

# CVD-safe (luminance + blue/gold), matching ex0122.
COLOURS = {'in': '#4f86c6', 'straddle': '#efdca0', 'out': '#c0392b'}


# A cell sharing the anchor's boundary reads as slightly <1 — the polygons are
# straight-line approximations of curved cell edges, so a contained boundary cell
# loses a sliver along the shared edge. That is not straddling. Classify with a
# tolerance: a cell straddles only if a real fraction lies on BOTH sides.
EDGE_TOL = 0.02


def classify(anchor, desc):
    """Return (fracs, counts) — per-descendant overlap fraction with the anchor."""
    fracs = []
    for d in desc:
        inter = d.intersection(anchor).area
        fracs.append(inter / d.area if d.area else 0.0)
    counts = {
        'in': sum(f > 1 - EDGE_TOL for f in fracs),
        'out': sum(f < EDGE_TOL for f in fracs),
        'straddle': sum(EDGE_TOL <= f <= 1 - EDGE_TOL for f in fracs),
    }
    return fracs, counts


def main(lat, lng, gens, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    print(f'anchor = ({lat}, {lng})   gens = {gens}')
    print(f'{"system":>6} {"res":>14} {"descendants":>12} {"inside":>7} '
          f'{"straddle":>9} {"OUTSIDE":>8}')

    for ax, name in zip(axes.flat, ADAPTERS):
        res = DEFAULT_RES[name]
        try:
            anchor, desc, label = ADAPTERS[name](lat, lng, res, gens)
        except NotImplementedError as e:
            ax.text(0.5, 0.5, f'{name}\n(stub)\n{e}', ha='center', va='center',
                    fontsize=9, wrap=True, transform=ax.transAxes, color='#888888')
            ax.set_title(name)
            ax.axis('off')
            print(f'{name:>6} {"-":>14} {"(stub — awaiting descendants/cell_to_boundary)":>12}')
            continue

        fracs, counts = classify(anchor, desc)
        ax.add_patch(MplPolygon(list(anchor.exterior.coords), closed=True, fill=False,
                                edgecolor='#222222', linewidth=2.5, zorder=3))
        for d, f in zip(desc, fracs):
            key = 'out' if f < EDGE_TOL else 'in' if f > 1 - EDGE_TOL else 'straddle'
            ax.add_patch(MplPolygon(list(d.exterior.coords), closed=True,
                                    facecolor=COLOURS[key], edgecolor='#555555',
                                    linewidth=0.3, alpha=0.85, zorder=2))
        ax.autoscale()
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'{label}   '
                     f'(out {counts["out"]}/{len(desc)})')
        print(f'{name:>6} {label.split()[1]:>14} {len(desc):>12} {counts["in"]:>7} '
              f'{counts["straddle"]:>9} {counts["out"]:>8}')

    fig.suptitle('DGGS hierarchy nesting — index-descendants coloured by overlap '
                 'with their ancestor\nblue = inside   gold = straddle   red = '
                 'entirely outside', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=140)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    import os
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--lat', type=float, default=40.0)
    ap.add_argument('--lng', type=float, default=-3.0)
    ap.add_argument('--gens', type=int, default=3, help='generations below the anchor')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__),
                                                  'dggs_nesting.png'))
    args = ap.parse_args()
    main(args.lat, args.lng, args.gens, args.out)
