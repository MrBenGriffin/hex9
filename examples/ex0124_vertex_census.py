# Part of the Hex9 (H9) Project
# Copyright ©2026, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Vertex-residual census: classify every L5 hexagon by |area deviation| and
show the neighbourhood of one vertex on each of the three octahedral axes
(poles, 0°/180°, 90°E/W).

Under the via-sphere chain (default since 2026-07-18) the warped
(quasi-authalic) L5 grid deviates from ideal cell area almost nowhere:
all but 252 of the 708,588 cells lie within 0.1% of ideal, no cell exceeds
1%, the extreme is −0.289%, and every cell beyond 0.1% is in deficit. The
exceptions cluster at the six octahedral vertices, which now form ONE
spectral class: 42 cells >0.1% at each vertex, with all six sorted
deviation spectra identical to ~1e-5 percentage points — pairwise AND
across the former classes. Morphology is shared too: every vertex shows
the same deficit (blue) cross of rays along its four meeting seams around
a near-white core.

That equivalence is the sphere engine's symmetry showing through: the
spherical AK core and Sphere-trained warp are equivariant under the full
octahedral rotation group, which maps any vertex — together with its L0
join structure — onto any other, and the authalic reduction is exactly
area-preserving, so the WGS84 census inherits the sphere's vertex
equivalence unchanged.

Historical (v2, classic per-ellipsoid chain via set_ellipsoid(...,
via_sphere=False)): the census split into THREE pairwise-exact classes —

    North/South Pole   48 cells >0.1% each, 12 >1%, extreme +4.80%
    0°N 0°E / 0°N 180°  38 cells >0.1% each,  2 >1%, extreme  2.10%
    0°N 90°E / 90°W     36 cells >0.1% each,  2 >1%, extreme  2.60%

— with class-dependent morphology (excess red polar core with rays on all
four seams; near-white equatorial cores with meridional-only rays). The
split is therefore a property of the ellipsoidal AK realisation and its
mirror-only trained field, not of the construction (paper §11b, inverted
in v3).

Shares the geometry/score cache with ex0081w_warped_authalics.py
(output/ex0081_cache_5.npz); builds it if absent (~10 min at L5). The
cache does not record which chain built it — delete it after changing
engine/chain settings.

Run:
    python examples/ex0124_vertex_census.py         # census table + figure
    python examples/ex0124_vertex_census.py --raw   # control: warp disabled

The --raw control disables the warp only, so under the via-sphere default
it measures the raw *spherical* AK realisation. To reproduce the classic
three-class control (raw ellipsoid field ±20%; equatorial-pair separation
~0.005% absolute, growing to ~0.5% once the shared bulk is warped away),
also set via_sphere=False before the first domain call.

Last tested: 2026-07-19 (via-sphere census + figure as above).
"""
from pathlib import Path

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt, colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

LAYERS = 5
CACHE = Path(f'output/ex0081_cache_{LAYERS}.npz')          # warped (shared with ex0081w)
CACHE_RAW = Path(f'output/ex0081_cache_raw_{LAYERS}.npz')  # raw AK (no warp)

# Stepped |deviation| classes (percent): sign preserved, no clipping — the
# top class is open-ended and the caption carries the true extremes.
BOUNDS = [-5.0, -1.0, -0.1, 0.1, 1.0, 5.0]
CLASS_COLOURS = ['#2166ac', '#a6cee8', '#f6f6f4', '#f4a582', '#b2182b']  # CVD-safe RdBu family
HW = 0.45e6           # panel half-width, m (~900 km across). Classed cells
                      # reach 444 km from their vertices (identical at all
                      # six under via-sphere; v2 classes reached 315/395/423),
                      # so the window keeps the census in-frame at the ray tips.
STROKE = (0.15, 0.15, 0.15, 1.0)
LINEWIDTH = 0.125

VERTICES = [   # (label, unit axis, view elev/azim, panel window builder)
    ('(a) North Pole',   [0, 0, 1], (90, 0)),
    ('(b) 0°N 0°E',      [1, 0, 0], (0, 0)),
    ('(c) 0°N 90°E',     [0, 1, 0], (0, 90)),
]


def build_cache(raw: bool = False):
    """Geometry + geodesic-area pipeline (slow path, ~10 min at L5).

    raw=True disables the authalic warp (b_oct.no_warp()) — the control run
    showing the three vertex classes are already present, pairwise-exact but
    minute, in the unwarped AK realisation (§11b)."""
    from hhg9 import Registrar
    from hhg9.algorithms.distance import wgs84_area, ellipsoid_area_wgs84
    from hhg9.h9.polygon import hex_poly_layer
    import ex0081w_warped_authalics as base
    reg = Registrar()
    if raw:
        reg.domain('b_oct').no_warp()
    b_pts = base.get_data(reg, LAYERS)
    pts, _ = hex_poly_layer(b_pts, LAYERS)
    ideal = ellipsoid_area_wgs84() / (12 * 9 ** LAYERS)
    c_pts = reg.project(pts.copy(), ['b_oct', 'c_oct', 'c_ell'])
    g_pts = reg.project(pts.copy(), ['b_oct', 'g_gcd'])
    score = np.log(np.abs(wgs84_area(reg, g_pts) / ideal) + 1e-12)
    np.savez_compressed(CACHE_RAW if raw else CACHE, coords=c_pts.coords, score=score)


def census(pct, cent):
    u = cent / np.linalg.norm(cent, axis=1, keepdims=True)
    V = np.array([[0, 0, 1], [0, 0, -1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], float)
    names = ['North Pole', 'South Pole', '0°N 0°E', '0°N 180°', '0°N 90°E', '0°N 90°W']
    near = np.argmax(u @ V.T, axis=1)
    print(f'{"vertex":>12} {">0.1%":>7} {">1%":>5} {"max |dev|%":>11}')
    for i, n in enumerate(names):
        m = near == i
        print(f'{n:>12} {int(((np.abs(pct) > 0.1) & m).sum()):>7} '
              f'{int(((np.abs(pct) > 1.0) & m).sum()):>5} {np.abs(pct[m]).max():>11.2f}')
    out = int((np.abs(pct) > 0.1).sum())
    print(f'{"TOTAL":>12} {out:>7} {int((np.abs(pct) > 1.0).sum()):>5}   '
          f'({len(pct) - out} of {len(pct)} cells within 0.1%)')


def panels(polys, rgba):
    mpl.rcParams['path.simplify'] = False
    fig = plt.figure(figsize=(18, 7.2), dpi=300, frameon=False)
    fig.subplots_adjust(top=0.93, bottom=0.10, right=1.0, left=0.0, wspace=0.02)
    cmap = colors.ListedColormap(CLASS_COLOURS)
    norm = colors.BoundaryNorm(BOUNDS, cmap.N)
    for k, (label, vaxis, (elev, azim)) in enumerate(VERTICES):
        ax = fig.add_subplot(1, 3, k + 1, projection='3d')
        ax.view_init(elev=elev, azim=azim)
        az, el = np.deg2rad(azim), np.deg2rad(elev)
        axis = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        mask = polys.mean(axis=1) @ axis >= 0
        ax.set_proj_type('ortho')
        # window: HW square, centred on the vertex; all three spans equal
        # (set_aspect('equal') scales to the largest range).
        lims = []
        for d in range(3):
            c = 5.85e6 * vaxis[d] if vaxis[d] else 0.0   # near-surface centring on the view axis
            lims.append((c - HW, c + HW) if vaxis[d] == 0 else (5.60e6, 5.60e6 + 2 * HW))
        ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1]); ax.set_zlim(*lims[2])
        ax.add_collection(Poly3DCollection([p for p in polys[mask]], ec=STROKE,
                                           facecolors=rgba[mask], linewidth=LINEWIDTH))
        ax.set_aspect('equal', adjustable='box')
        ax.set_axis_off()
        ax.set_title(label, fontsize=14, y=1.0)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=fig.axes, orientation='horizontal', pad=0.02,
                        fraction=0.030, aspect=50, shrink=0.45, spacing='uniform')
    cbar.set_ticks([-1.0, -0.1, 0.1, 1.0])
    cbar.set_label('% area deviation class — at L5, all but 252 of 708,588 cells lie in the central (white) class',
                   fontsize=11)
    out = Path(f'output/ex0124_vertex_census_{LAYERS}.png')
    fig.savefig(out, dpi=300, facecolor='white')
    print(f'fig saved at {out}')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--raw', action='store_true',
                    help='control run: raw AK, warp disabled (census + spectra only)')
    args = ap.parse_args()
    cache = CACHE_RAW if args.raw else CACHE
    if not cache.exists():
        build_cache(raw=args.raw)
    z = np.load(cache)
    coords, score = z['coords'], z['score']
    pct = np.expm1(score) * 100.0
    polys = coords.reshape(-1, 6, 3)
    census(pct, polys.mean(axis=1))
    if args.raw:
        # pairwise-exact class spectra: the split predates the warp
        C = polys.mean(axis=1)
        u = C / np.linalg.norm(C, axis=1, keepdims=True)
        V = np.array([[0, 0, 1], [0, 0, -1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], float)
        near = np.argmax(u @ V.T, axis=1)
        print('\nraw-AK top-10 |dev|% per vertex (sorted desc):')
        for i, n in enumerate(['NP', 'SP', '0°E', '180°', '90°E', '90°W']):
            v = np.sort(np.abs(pct[near == i]))[::-1][:10]
            print(f'{n:>5}: ' + ' '.join(f'{x:7.3f}' for x in v))
    else:
        cmap = colors.ListedColormap(CLASS_COLOURS)
        rgba = cmap(colors.BoundaryNorm(BOUNDS, cmap.N)(pct))
        panels(polys, rgba)
