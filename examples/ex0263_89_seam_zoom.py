# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Seam-zoom near the north pole (≈89°N), rendered in a local ECEF tangent plane.

A small window straddling the prime meridian at ~89°N crosses an octant seam
right next to the north-pole cone vertex.  In g_gcd (lat/lon) this renders with a
badly distorted shape — at 89°N a degree of longitude is ~60× shorter on the
ground than a degree of latitude, so an equal-aspect lon/lat plot squashes the
hexes.  Instead we project the shared vertex pool to ECEF (c_ell, EPSG:4978) and
flatten it onto the plane tangent to the surface at the window centroid (an ENU
local frame), so distances are metric and the hexes keep their true shape.

Hexes that span the prime meridian (vertices on both sides of lon = 0) are drawn
in red so the cross-seam stitching is visible.

Companion to ex0262_greenwich_seam.py (prime-meridian close-seam test) and
ex0263_bhutan_big.py (a different seam, with population data).

Last Tested
-----------
28 Jun 2026 0.1.3a0 (ECEF tangent-plane rendering)
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection

from hhg9 import Points, Registrar
from hhg9.h9.grid import HexMesh

# Window straddling lon = 0 just below the north pole (≈89°N).
SEAM_WINDOW = np.array([
    [88.98, -0.5],
    [88.98,  0.5],
    [89.02,  0.5],
    [89.02, -0.5],
])


def tangent_xy(ecef: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Flatten ECEF points onto the ENU plane tangent to the sphere at *origin*."""
    up = origin / np.linalg.norm(origin)
    east = np.cross([0.0, 0.0, 1.0], up)
    east /= np.linalg.norm(east)
    north = np.cross(up, east)
    rel = ecef - origin
    return np.column_stack([rel @ east, rel @ north])      # metres


def plot_seam_zoom(mesh, xy, lonlat, poly_xy, name):
    """Draw nested hex layers in the local tangent plane; flag seam crossers."""
    layers = mesh.layers
    finest = mesh.fine

    xmin, xmax = poly_xy[:, 0].min(), poly_xy[:, 0].max()
    ymin, ymax = poly_xy[:, 1].min(), poly_xy[:, 1].max()
    ratio = (xmax - xmin) / (ymax - ymin) if ymax > ymin else 1.0

    fig = plt.figure(figsize=(ratio * 12, 12), dpi=200)
    ax = fig.add_subplot(111)
    pad_x, pad_y = 0.08 * (xmax - xmin), 0.08 * (ymax - ymin)
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()

    n_layers = len(layers)
    for i, L in enumerate(layers):
        faces = mesh[L]
        if len(faces) == 0:
            continue
        polys = xy[faces]                              # (N, 6, 2) metres
        lw = 0.06 * (n_layers - i)

        if L == finest:
            lons = lonlat[faces][:, :, 1]             # g_gcd lon per vertex
            crosses = (lons.min(axis=1) < 0) & (lons.max(axis=1) > 0)
            ax.add_collection(PolyCollection(
                polys[~crosses], linewidths=lw,
                facecolors='#33994420', edgecolors='#444444'))
            ax.add_collection(PolyCollection(
                polys[crosses], linewidths=lw * 0.25,
                facecolors='#ff000030', edgecolors='#cc0000'))
            print(f'  layer {L}: {len(faces)} hexes '
                  f'({int(crosses.sum())} cross the seam)')
        else:
            ax.add_collection(PolyCollection(
                polys, linewidths=lw, facecolors='none', edgecolors='#1f77b4a0'))
            print(f'  layer {L}: {len(faces)} hexes')

    # Clip rectangle outline (in the same tangent plane).
    ax.add_collection(PolyCollection(
        [poly_xy], facecolors='none', edgecolors='#888888',
        linewidths=0.4, linestyles='--'))

    # Scale bar (100 m).
    ax.plot([xmin, xmin + 100.0], [ymin, ymin], color='k', lw=1.5)
    ax.text(xmin + 50.0, ymin + 0.01 * (ymax - ymin), '100 m',
            ha='center', va='bottom', fontsize=8)

    fig.savefig(f'output/{name}.png', dpi=200, bbox_inches='tight')
    print(f'fig saved at output/{name}.png')
    plt.close(fig)


if __name__ == '__main__':
    reg = Registrar()
    b_oct = reg.domain('b_oct')
    g_gcd = reg.domain('g_gcd')
    c_oct = reg.domain('c_oct')
    c_ell = reg.domain('c_ell')

    finest = 11
    nest = [finest]
    mesh = HexMesh.create_clipped(nest, SEAM_WINDOW, reg)
    print(mesh)

    ecef = reg.project(mesh.pts, [b_oct, c_oct, c_ell]).coords      # (M, 3) metres
    lonlat = reg.project(mesh.pts, [b_oct, g_gcd]).coords           # for seam test

    origin = ecef.mean(axis=0)
    xy = tangent_xy(ecef, origin)
    poly_ecef = reg.project(Points(SEAM_WINDOW, g_gcd), [g_gcd, c_ell]).coords
    poly_xy = tangent_xy(poly_ecef, origin)

    plot_seam_zoom(mesh, xy, lonlat, poly_xy, f'ex0263_89_zoom_L{finest}')

    addrs = mesh.addrs
    print(f'finest layer L{finest}: {len(addrs)} hexes, '
          f'{len(set(addrs))} unique bin UUIDs')
