# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Single-octant area-deviation map in native b_oct face coordinates.

Renders one octant of the H9 triangular sub-grid in the octahedral face
coordinate space (b_oct: 2D Cartesian, origin at face barycentre). Cells
are coloured by geodesic area deviation from ideal.

All 8 octants are mathematically identical under the transform, so one
octant fully characterises the warp quality. This is also the cleanest
possible representation — b_oct is the native coordinate space; no
reprojection artefacts.

Outputs:
  output/ex0063b_octant_L{depth}.png

Last Tested
2026-04-20
"""

import os
import numpy as np
from matplotlib import pyplot as plt, colors
from matplotlib.collections import PolyCollection
from hhg9 import Registrar, Points
from hhg9.algorithms.distance import wgs84_area, ellipsoid_area_wgs84
from hhg9.h9 import H9O, H9K
from hhg9.h9.polygon import region_grid, H9P

os.makedirs('output', exist_ok=True)


def tri_grid_with_modes(layer: int, octant_mode: int):
    """Return (vertices, modes) for all triangles in one octant.

    Returns
    -------
    verts : (N, 3, 2) float64  — b_oct triangle vertices
    modes : (N,) int            — 0 = ∇, 1 = △ for each triangle
    """
    queue = region_grid(layer, octant_mode, H9P)
    n = len(queue)
    verts = np.empty((n, 3, 2), dtype=np.float64)
    modes = np.empty(n, dtype=np.int8)
    for i, (path, mode, origin, scale) in enumerate(queue):
        verts[i] = origin + H9P.sv[mode] * scale
        modes[i] = mode
    return verts, modes


def get_octant_data(reg: Registrar, layer: int, octant_id: int = 0):
    """Return (Points, modes) for one octant in b_oct space."""
    octant_mode = H9O.oid_mo[octant_id]
    verts, modes = tri_grid_with_modes(layer, octant_mode)
    b_oct = reg.domain('b_oct')
    pts = Points(verts.reshape(-1, 2), b_oct, oid=octant_id)
    return pts, modes


def octant_plot(reg: Registrar, depth: int, octant_id: int = 0) -> None:
    data, tri_modes = get_octant_data(reg, depth, octant_id)
    n_tris = data.coords.shape[0] // 3

    # ── Geodesic area per triangle ─────────────────────────────────────────
    g_pts = reg.project(data, ['b_oct', 'g_gcd'])
    g_areas = wgs84_area(reg, g_pts, 3)             # (N_tri,) m²

    # Ideal: one octant = 1/8 sphere; at layer depth, 9^(depth+1) triangles/octant
    n_tris_total = 8 * 9 ** (depth + 1)
    gm2 = ellipsoid_area_wgs84()
    ideal_m2 = gm2 / n_tris_total
    pct_dev = (g_areas / ideal_m2 - 1.0) * 100.0   # (N_tri,)

    # ── Mode-split analysis ────────────────────────────────────────────────
    # Identify systematic bias between ∇ (mode-0) and △ (mode-1) triangles.
    # The Sinkhorn warp was optimised at the hexagon level (mode-0 + mode-1
    # d_cell pairs). Any residual mode asymmetry here persists at ALL finer
    # refinement levels — it is baked into the continuous displacement field.
    # print(f'\nLayer {depth} — mode-split area analysis:')
    # for m, sym in [(0, '∇ mode-0'), (1, '△ mode-1')]:
    #     mask = tri_modes == m
    #     n_m = mask.sum()
    #     mean_pct = pct_dev[mask].mean()
    #     std_pct  = pct_dev[mask].std()
    #     print(f'  {sym}: {n_m:6d} triangles  mean {mean_pct:+.4f}%  std {std_pct:.4f}%')
    # asymmetry = pct_dev[tri_modes == 0].mean() - pct_dev[tri_modes == 1].mean()
    # print(f'  Mode asymmetry (Δ mean): {asymmetry:+.4f}%')
    # if abs(asymmetry) > 0.2:
    #     print(f'  *** Asymmetry > 0.2% — consider re-running Sinkhorn at d_cell level ***')

    # ── b_oct 2D vertices for each triangle ────────────────────────────────
    verts = data.coords.reshape(-1, 3, 2)           # (N_tri, 3, [x, y])

    # ── Colourmap ──────────────────────────────────────────────────────────
    max_abs = float(np.percentile(np.abs(pct_dev), 99))
    norm = colors.TwoSlopeNorm(vcenter=0.0, vmin=-max_abs, vmax=+max_abs)
    cmap = plt.get_cmap('RdBu_r')

    # ── Plot ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 8), dpi=600)
    ax.set_aspect('equal')
    ax.set_facecolor('#f4f4f4')

    col = PolyCollection(
        verts,
        array=pct_dev,
        cmap=cmap,
        norm=norm,
        edgecolors='none',
        linewidths=0.1,
        antialiased=True,
    )
    ax.add_collection(col)
    u, v = H9K.lattice.U, H9K.lattice.V
    ax.set_xlim(H9K.limits.TL - 0.02, H9K.limits.TL + u)
    ax.set_ylim(H9K.limits.VC - v, H9K.limits.VC + 0.02)

    print(f'Layer {depth}: {n_tris} triangles  '
          f'pct_dev min/mean/max: {pct_dev.min():+.3f}% / '
          f'{pct_dev.mean():+.3f}% / {pct_dev.max():+.3f}%')

    out = f'output/ex0063bc_octant_L{depth}.png'
    fig.savefig(out, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')


if __name__ == '__main__':
    rg = Registrar()
    b_oct = rg.domain('b_oct')
    # b_oct.set_warp('../hhg9/data/WGS84_l5_warp_data_tidied.npz')
    for depth in [5]:
        octant_plot(rg, depth, octant_id=0)
