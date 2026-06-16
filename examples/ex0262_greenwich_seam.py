# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Cross-seam polygon-clipped hex grid: Greenwich Park.

Greenwich Park straddles the prime meridian.  This is a small, deep test of
HexMesh.create_clipped — a fine layer (L13) over a sub-kilometre polygon.
The pruned descent makes this feasible: a global L13 mesh would be
12·9**13 ≈ 3·10**13 hexes, but the descent only ever expands branches that
overlap the park's bounding box.

Companion to ex0261_polygrid_mesh.py.

Last Tested
-----------
16 Jun 2026 0.1.3a0 (passed) 10.0s
22 May 2026 (passed: L11-L13 nest, 53297 hexes at L13, seam continuous)
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection

from hhg9 import Registrar
from hhg9.h9.grid import HexMesh


# Greenwich Park boundary, [lat, lon] degrees (crosses the prime meridian).
GREENWICH = np.array([
    [51.481588, 0.002237], [51.481983, 0.001826], [51.482083, 0.001692],
    [51.482271, 0.001410], [51.482221, 0.001084], [51.482130, 0.000648],
    [51.481848, -0.000455], [51.481653, -0.001111], [51.481571, -0.001934],
    [51.481530, -0.002234], [51.481461, -0.002221], [51.481401, -0.002221],
    [51.481321, -0.002253], [51.481270, -0.002297], [51.481237, -0.002338],
    [51.481164, -0.002486], [51.480008, -0.005877], [51.479916, -0.005879],
    [51.479891, -0.005951], [51.479551, -0.006985], [51.479547, -0.007034],
    [51.479560, -0.007097], [51.479590, -0.007134], [51.479494, -0.007335],
    [51.479397, -0.007442], [51.479285, -0.007495], [51.479139, -0.007527],
    [51.478899, -0.007514], [51.478852, -0.007494], [51.478788, -0.007446],
    [51.478565, -0.007245], [51.478424, -0.007046], [51.478220, -0.006676],
    [51.478039, -0.006386], [51.477978, -0.006260], [51.477870, -0.006136],
    [51.477757, -0.006029], [51.477729, -0.006005], [51.477485, -0.005880],
    [51.477322, -0.005815], [51.477014, -0.005750], [51.476990, -0.005731],
    [51.476731, -0.005437], [51.476427, -0.005022], [51.476194, -0.004507],
    [51.475790, -0.003959], [51.475625, -0.003773], [51.474917, -0.003027],
    [51.474912, -0.002994], [51.474894, -0.002969], [51.474891, -0.002916],
    [51.474874, -0.002888], [51.474469, -0.002460], [51.474461, -0.002455],
    [51.474440, -0.002462], [51.474420, -0.002488], [51.473424, -0.001428],
    [51.473429, -0.001416], [51.473408, -0.001357], [51.473396, -0.001351],
    [51.473379, -0.001361], [51.473372, -0.001353], [51.473366, -0.001367],
    [51.472357, -0.000298], [51.472370, -0.000266], [51.472284, -0.000178],
    [51.472207, 0.000050], [51.472178, 0.000024], [51.472151, 0.000105],
    [51.472109, 0.000137], [51.473143, 0.003658], [51.473855, 0.005997],
    [51.474360, 0.007575], [51.474778, 0.008834], [51.475043, 0.009674],
    [51.475070, 0.009726], [51.475115, 0.009770], [51.475177, 0.009800],
    [51.475238, 0.009805], [51.475282, 0.009792], [51.477937, 0.006880],
    [51.477963, 0.006836], [51.478158, 0.006623], [51.478180, 0.006612],
    [51.479505, 0.005183], [51.479728, 0.004899], [51.480037, 0.004296],
    [51.480459, 0.003571], [51.480750, 0.003289], [51.480750, 0.003245],
    [51.480774, 0.003182], [51.481139, 0.002722], [51.481266, 0.002566],
])


def plot_mesh(mesh, lonlat, polygon, name):
    """Draw clipped hex layers (finest on top) plus the clip polygon."""
    layers = mesh.layers
    finest = mesh.fine

    plon, plat = polygon[:, 1], polygon[:, 0]
    xmin, xmax = plon.min(), plon.max()
    ymin, ymax = plat.min(), plat.max()
    ratio = (xmax - xmin) / (ymax - ymin)

    fig = plt.figure(figsize=(ratio * 12, 12), dpi=400)
    ax = fig.add_subplot(111)
    pad_x, pad_y = 0.06 * (xmax - xmin), 0.06 * (ymax - ymin)
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()

    n_layers = len(layers)
    for i, L in enumerate(layers):
        faces = mesh[L]
        if len(faces) == 0:
            continue
        polys = lonlat[faces][:, :, ::-1]            # (N, 6, 2) as (lon, lat)
        lw = 0.05 * (n_layers - i)
        edge = '#444444' if L == finest else '#1f77b4a0'
        face = '#33994420' if L == finest else 'none'
        ax.add_collection(PolyCollection(
            polys, linewidths=lw, facecolors=face, edgecolors=edge))
        print(f'  layer {L}: {len(faces)} hexes')

    # Prime meridian and clip polygon outline.
    ax.axvline(0.0, color='0.6', lw=0.1, ls=':')
    ax.add_collection(PolyCollection(
        [np.column_stack([plon, plat])],
        facecolors='none', edgecolors='red', linewidths=0.05, linestyles='--'))

    fig.savefig(f'output/{name}.png', dpi=400, bbox_inches='tight')
    print(f'fig saved at output/{name}.png')
    plt.close(fig)


if __name__ == '__main__':
    reg = Registrar()
    b_oct = reg.domain('b_oct')
    g_gcd = reg.domain('g_gcd')
    # Disable the authalic warp: it currently delaminates hexes that straddle
    # an octahedral face seam (warp re-training in progress).  The lattice
    # itself stitches the seam correctly — see the cross-seam hex column.
    # b_oct.no_warp()

    finest = 13
    nest = list(range(finest - 0, finest + 1))      # ancestry nest: L11..L13

    mesh = HexMesh.create_clipped(nest, GREENWICH, reg)
    print(mesh)

    lonlat = reg.project(mesh.pts, [b_oct, g_gcd]).coords

    plot_mesh(mesh, lonlat, GREENWICH, f'ex0262_greenwich_L{finest}')

    addrs = mesh.addrs
    print(f'finest layer L{finest}: {len(addrs)} hexes, '
          f'{len(set(addrs))} unique bin UUIDs')
    for a in addrs:
        print(f'  {a}')
