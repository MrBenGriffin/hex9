# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Global Mollweide area-deviation map.

Renders all H9 hexagonal cells at a given layer, coloured by percentage
deviation from ideal equal area, in a Mollweide projection.

Outputs:
  output/ex0118_mollweide_L{layer}.png

Note: antimeridian-crossing cells (~4 at any layer) are skipped; a count
is printed. This is the known QGIS/display limitation — data is correct.

Usage:
  python ex0118_mollweide.py           # default L4 (~78k hexes, fast)
  python ex0118_mollweide.py --layer 5 # L5 (~944k hexes, paper quality)

Last Tested
16 Jun 2026 0.1.3a0 (time)
"""

import argparse
import os
import numpy as np
from matplotlib import pyplot as plt, colors
from matplotlib.collections import PolyCollection
from hhg9 import Registrar
from hhg9.algorithms.distance import wgs84_area
from hhg9.h9.grid import HexMesh

os.makedirs('output', exist_ok=True)


def mollweide_map(reg: Registrar, depth: int) -> None:
    print(f'Layer {depth}: generating hex polygons...')
    mesh = HexMesh.create([depth], reg)
    pts = mesh.pts
    faces = mesh.faces
    gm2 = reg.ellipsoid_area  # 510_065_621_724km
    bins = 12 * 9 ** depth
    ideal_m2 = gm2 / bins

    # ── Project to lat/lon and compute areas ───────────────────────────────
    g_pts = reg.project(pts, ['b_oct', 'g_gcd'])  # (18140, 2) [lat°, lon°]
    hexes = g_pts.coords[faces]

    latlon_deg = hexes           # (N_hex, 6, [lat, lon])

    # ── Compute geodesic areas ──────────────────────────────────────────────
    # hexes is (N_hex, 6, 2) [lat°, lon°] — pass as raw array so wgs84_area
    # uses the face-indexed grouping rather than reshaping the vertex set.
    w_area_m2 = wgs84_area(reg, hexes)
    pct_dev = (w_area_m2 / ideal_m2 - 1.0) * 100.0       # (N_hex,)

    # ── Convert to radians for Mollweide ───────────────────────────────────
    latlon_rad = np.deg2rad(latlon_deg)                    # (N_hex, 6, [lat, lon])
    # Mollweide: x = longitude, y = latitude
    verts_moll = latlon_rad[:, :, [1, 0]]                 # (N_hex, 6, [lon, lat])

    # ── Skip antimeridian-crossing hexes ───────────────────────────────────
    lon_deg = latlon_deg[:, :, 1]                          # (N_hex, 6)
    lon_span = lon_deg.max(axis=1) - lon_deg.min(axis=1)
    valid = lon_span < 180.0
    n_skipped = int((~valid).sum())
    if n_skipped:
        print(f'  Skipped {n_skipped} antimeridian-crossing hex(es) (data correct, display limitation).')

    verts_plot = verts_moll[valid]
    scores_plot = pct_dev[valid]

    # ── Colourmap ──────────────────────────────────────────────────────────
    # Asymmetric TwoSlopeNorm: each side of zero gets its own scale so both
    # the negative and positive extremes are fully saturated and legible.
    # The colorbar explicitly shows the asymmetric range; note in caption.
    v_min = float(np.percentile(scores_plot, 1))
    v_max = float(np.percentile(scores_plot, 99))
    norm = colors.TwoSlopeNorm(vcenter=0.0, vmin=v_min, vmax=v_max)
    cmap = plt.get_cmap('RdBu_r')

    # ── Plot ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 9), dpi=600)
    ax = fig.add_subplot(111, projection='mollweide')
    ax.set_facecolor('#d0e8f0')

    col = PolyCollection(
        verts_plot,
        array=scores_plot,
        cmap=cmap,
        norm=norm,
        linewidths=0.0,
        antialiased=False,
    )
    ax.add_collection(col)

    # Graticule
    ax.grid(color='white', linewidth=0.4, alpha=0.6)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.04,
                        fraction=0.03, aspect=40)
    cbar.set_label('% deviation from ideal equal area', fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    # Stats annotation
    print(f'  pct_dev: min={pct_dev.min():+.3f}%  mean={pct_dev.mean():+.3f}%  '
          f'max={pct_dev.max():+.3f}%  MAE={np.mean(np.abs(pct_dev)):.3f}%')
    ax.set_title(
        f'H9 cell area deviation — Layer {depth}  ({bins:,} hexagons)\n'
        f'min {pct_dev.min():+.2f}%   MAE {np.mean(np.abs(pct_dev)):.3f}%   '
        f'max {pct_dev.max():+.2f}%',
        fontsize=11, pad=10,
    )

    out = f'output/ex0118_mollweide_L{depth}.png'
    fig.savefig(out, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')


if __name__ == '__main__':
    from hhg9.h9.region import recover_stats_reset, recover_stats_report

    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=int, default=5,
                        help='H9 layer depth (5 for paper quality)')
    args = parser.parse_args()
    recover_stats_reset()
    rg = Registrar()
    mollweide_map(rg, args.layer)
    recover_stats_report()
