# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Polygon-clipped hex grid via the HexMesh hierarchy.

Companion to ex0260_polygrid.py.  Where ex0260 fills the polygon by casting
into n_oct and binning a net field, this example uses the HexMesh supercell
hierarchy directly:

    HexMesh.create_clipped(layers, ll_polygon, reg)

It descends the H9 supercell tree, pruning any branch whose footprint misses
the polygon's bounding box, then keeps hexes whose centroid lies inside the
polygon.  Cost scales with the polygon's area, not the globe — so there is no
12·9**L global enumeration and no n_oct cast.  The HexMesh template assembly
keeps each hex seam-safe (boundary verts reflected into the adjacent face).

Coarser layers form an ancestry "nest" drawn behind the finest layer.

Last Tested
-----------
22 May 2026 (passed: L4-L6 nest, 572 hexes at L6)
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.path import Path

from hhg9 import Registrar
from hhg9.h9.grid import HexMesh


# [birmingham-bristol-london-belgium] polygon, [lat, lon] degrees.
B_DLL = np.array([
    [51.084464524954795, -3.4462861844929585],
    [52.68223344923537, -2.189458562294672],
    [52.60858425701964, -1.432594162686184],
    [51.648685168766754, -2.0537941887799422],
    [51.52446468552721, -0.8970768988122543],
    [51.82555423461282, -0.747132064927554],
    [51.684114370958646, 0.5309691381848917],
    [51.40435883726121, 0.47384729670500597],
    [51.29190823165293, 5.4859054058927266],
    [50.72448838777691, 5.4044368907702074],
])


def plot_mesh(mesh, lonlat, polygon, name):
    """Draw clipped hex layers (finest on top) plus the clip polygon."""
    layers = mesh.layers
    finest = mesh.fine

    plon, plat = polygon[:, 1], polygon[:, 0]
    xmin, xmax = plon.min(), plon.max()
    ymin, ymax = plat.min(), plat.max()
    ratio = (xmax - xmin) / (ymax - ymin)

    fig = plt.figure(figsize=(ratio * 10, 10), dpi=200)
    ax = fig.add_subplot(111)
    pad_x, pad_y = 0.04 * (xmax - xmin), 0.04 * (ymax - ymin)
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()

    n_layers = len(layers)
    for i, L in enumerate(layers):
        faces = mesh[L]
        if len(faces) == 0:
            continue
        # lonlat columns: 0 = lat, 1 = lon -> plot (lon, lat).
        polys = lonlat[faces][:, :, ::-1]            # (N, 6, 2) as (lon, lat)
        lw = 0.03 * (n_layers - i)
        edge = 'black' if L == finest else '#1f77b4a0'
        face = '#33994420' if L == finest else 'none'
        ax.add_collection(PolyCollection(
            polys, linewidths=lw, facecolors=face, edgecolors=edge))
        print(f'  layer {L}: {len(faces)} hexes')

    # Clip polygon outline.
    ax.add_collection(PolyCollection(
        [np.column_stack([plon, plat])],
        facecolors='none', edgecolors='red', linewidths=1.0, linestyles='--'))

    fig.savefig(f'output/{name}.png', dpi=200, bbox_inches='tight')
    print(f'fig saved at output/{name}.png')
    plt.close(fig)


if __name__ == '__main__':
    from hhg9.h9.region import recover_stats_reset, recover_stats_report

    reg = Registrar()
    b_oct = reg.domain('b_oct')
    g_gcd = reg.domain('g_gcd')
    recover_stats_reset()

    finest = 6
    nest = list(range(finest - 2, finest + 1))      # ancestry nest: L4..L6

    mesh = HexMesh.create_clipped(nest, B_DLL, reg)
    print(mesh)

    # Project the shared vertex pool b_oct -> g_gcd once; faces index into it.
    lonlat = reg.project(mesh.pts, [b_oct, g_gcd]).coords
    plot_mesh(mesh, lonlat, B_DLL, f'ex0261_polygrid_mesh_L{finest}')

    # Unique bin addresses for the finest clipped layer.
    addrs = mesh.addrs
    print(f'finest layer L{finest}: {len(addrs)} hexes, '
          f'{len(set(addrs))} unique bin UUIDs')
    for a in addrs[:5]:
        print(f'  {a}')
    recover_stats_report()
