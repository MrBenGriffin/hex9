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
16 Jun 2026 0.1.3a0 (passed) 18.2s
"""

import os
import numpy as np
from matplotlib import pyplot as plt, colors
from matplotlib.collections import PolyCollection
from hhg9 import Registrar, Points
from hhg9.algorithms.distance import wgs84_area, ellipsoid_area_wgs84
from hhg9.h9 import H9O
from hhg9.h9.polygon import octant_grid

os.makedirs('output', exist_ok=True)

def get_octant_data(reg: Registrar, depth: int):
    """Return (vertices, triangles) for all triangles in one octant."""
    verts, grid, _, _, _ = octant_grid(depth, octant_id=0)
    b_oct = reg.domain('b_oct')
    pts = Points(verts, b_oct, oid=0)
    return pts, grid

def octant_plot(reg: Registrar, depth: int, octant_id: int = 0) -> None:
    b_pts, grid = get_octant_data(reg, depth)

    # ── Geodesic area per triangle ─────────────────────────────────────────
    c_pts = reg.project(b_pts, ['b_oct', 'c_oct'])
    e_pts = reg.project(c_pts, ['c_oct', 'c_ell'])
    # Got to convert e_pts to unit ellipsoid, right?
    e_max = np.max(e_pts.coords, axis=0)
    n_pts = e_pts.coords / e_max
    n_tri = n_pts[grid]
    c_tri = c_pts.coords[grid]
    nt = np.mean(n_tri, axis=1)
    ct = np.mean(c_tri, axis=1)
    dx = nt[:, 0] -  ct[:, 0]
    dy = nt[:, 1] -  ct[:, 1]
    dz = nt[:, 2] -  ct[:, 2]
    dist3 = [dx, dy, dz]

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
    # verts = data.coords.reshape(-1, 3, 2)           # (N_tri, 3, [x, y])
    verts = b_pts.coords[grid]
    axes = ['X', 'Y', 'Z']
    for axis in range(3):
        term = axes[axis]
        dist = dist3[axis]
        # ── Colourmap ──────────────────────────────────────────────────────────
        # max_abs = float(np.percentile(np.abs(dist), 99.999))
        d_max = np.max(dist)
        norm = colors.Normalize(vmin=0.0, vmax=d_max)
        cmap = plt.get_cmap('gist_ncar')

        # ── Plot ───────────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(9, 8), dpi=600)
        ax.set_aspect('equal')
        ax.set_facecolor('#f4f4f4')

        col = PolyCollection(
            verts,
            array=dist,
            cmap=cmap,
            norm=norm,
            edgecolors='none',
            linewidths=0.0,
            antialiased=True,
        )
        ax.add_collection(col)
        ax.autoscale_view()

        # Draw octant boundary (convex hull of all triangle vertices)
        # from scipy.spatial import ConvexHull
        # corner_xy = verts.reshape(-1, 2)
        # hull = ConvexHull(corner_xy)
        # hull_verts = corner_xy[hull.vertices]
        # ax.fill(hull_verts[:, 0], hull_verts[:, 1],
        #         fill=False, edgecolor='black', linewidth=0.8)

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical', pad=0.02, fraction=0.04)
        cbar.set_label(f'distance on {term}', fontsize=10)
        cbar.ax.tick_params(labelsize=8)

        ax.set_xlabel('b_oct x  (face-plane Cartesian)', fontsize=9)
        ax.set_ylabel('b_oct y  (face-plane Cartesian)', fontsize=9)
        ax.set_title(
            f'H9 Layer {depth}; Axis {term}\n'
            f'min {dist.min():+.2f} mean {np.mean(np.abs(dist)):.3f}' f'max {dist.max():+.2f}',
            fontsize=10, pad=8,
        )
        n_tris = len(grid)
        out = f'output/ex0065_octant_L{depth}{term}.png'
        fig.savefig(out, dpi=600, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out}{term}')


if __name__ == '__main__':
    rg = Registrar()
    b_oct = rg.domain('b_oct')
    b_oct.no_warp()
    for depth in [4, 5]:
        octant_plot(rg, depth, octant_id=0)
