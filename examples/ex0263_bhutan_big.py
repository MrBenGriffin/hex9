# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Seam-zoom: prime-meridian crossing at higher resolution.

A tight rectangle inside Greenwich Park straddles the prime meridian
(lon = 0).  This exercises the same cross-seam stitching as ex0262 but
at L15, where individual hexes are ~30 m across, making the seam
continuity easy to inspect visually.

Hexes that span the prime meridian (vertices on both sides of lon = 0)
are drawn in red so the stitching is immediately visible.

Companion to ex0262_greenwich_seam.py.

Last Tested
-----------
23 May 2026
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection

from hhg9 import Registrar
from hhg9.h9.grid import HexMesh

# Tiny rectangle straddling lon = 0 inside Greenwich Park.
# At L20, each hex ≈ 3.5 mm²; this ~22 mm × 21 mm patch yields ~130 L20 cells.
# lat ∈ [51.4799999, 51.4800001]  (span ≈ 22 mm)
# lon ∈ [-1.5e-7, +1.5e-7]       (span ≈ 21 mm)
# 88.73, 26.70, 92.09, 28.36
SEAM_WINDOW = np.array([
    # lat, lon
    [26.70, 88.73],
    [26.70, 92.09],
    [28.36, 92.09],
    [28.36, 88.73],
])


def plot_seam_zoom(mesh, lonlat, polygon, name):
    """Draw the nested hex layers; highlight hexes that cross lon = 0."""
    layers = mesh.layers
    finest = mesh.fine

    plon, plat = polygon[:, 1], polygon[:, 0]
    xmin, xmax = plon.min(), plon.max()
    ymin, ymax = plat.min(), plat.max()
    ratio = (xmax - xmin) / (ymax - ymin)

    fig = plt.figure(figsize=(ratio * 14, 14), dpi=400)
    ax = fig.add_subplot(111)
    pad_x = 0.08 * (xmax - xmin)
    pad_y = 0.08 * (ymax - ymin)
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()

    n_layers = len(layers)
    for i, L in enumerate(layers):
        faces = mesh[L]
        if len(faces) == 0:
            continue
        polys = lonlat[faces][:, :, ::-1]          # (N, 6, 2) lon-lat order
        lw = 0.06 * (n_layers - i)

        if L == finest:
            # Split finest layer: seam-crossing vs normal.
            lons = polys[:, :, 0]
            crosses = (lons.min(axis=1) < 0) & (lons.max(axis=1) > 0)

            ax.add_collection(PolyCollection(
                polys[~crosses], linewidths=lw,
                facecolors='#33994420', edgecolors='#444444'))
            ax.add_collection(PolyCollection(
                polys[crosses], linewidths=lw * 0.25,
                facecolors='#ff000030', edgecolors='#cc0000'))
            print(f'  layer {L}: {len(faces)} hexes '
                  f'({crosses.sum()} cross the seam)')
        else:
            edge = '#1f77b4a0'
            ax.add_collection(PolyCollection(
                polys, linewidths=lw, facecolors='none', edgecolors=edge))
            print(f'  layer {L}: {len(faces)} hexes')

    # Prime meridian.
    ax.axvline(0.0, color='#cc0000', lw=0.1, ls='--', alpha=0.6, zorder=5)

    # Clip rectangle outline.
    ax.add_collection(PolyCollection(
        [np.column_stack([plon, plat])],
        facecolors='none', edgecolors='#888888',
        linewidths=0.15, linestyles='--'))

    fig.savefig(f'output/{name}.png', dpi=400, bbox_inches='tight')
    print(f'fig saved at output/{name}.png')
    plt.close(fig)


if __name__ == '__main__':
    reg = Registrar()
    b_oct = reg.domain('b_oct')
    g_gcd = reg.domain('g_gcd')

    finest = 5
    nest = list(range(3, finest))      # L18, L19, L20

    mesh = HexMesh.create_clipped(nest, SEAM_WINDOW, reg)
    print(mesh)

    lonlat = reg.project(mesh.pts, [b_oct, g_gcd]).coords

    plot_seam_zoom(mesh, lonlat, SEAM_WINDOW, f'ex0263_bhutan_big_L{finest}')

    addrs = mesh.addrs
    print(f'finest layer L{finest}: {len(addrs)} hexes, '
          f'{len(set(addrs))} unique bin UUIDs')
