# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Polygon -> hex grid + H9 addresses, rendered in b_oct.

Given a NON-CONVEX lat/lon polygon and a finest hex layer, build the clipped hex
grid with its ancestry "nest" (coarser border-levels behind the finest) and
generate an H9 UUID address for every hex.

This is the addressing / non-convex-clipping showcase of the polygrid trio:
  * ex0260 (this)           -- HexMesh -> b_oct, UUID addresses + common-prefix
                               labels, non-convex clip, ancestry nest.
  * ex0261_polygrid_mesh.py -- the same HexMesh source rendered in g_gcd (lat/lon).
  * ex0261_polygrid.py      -- a simpler grid via hex_poly_layer.

Why HexMesh: ``HexMesh.create_clipped`` keeps a hex when its centroid lies inside
the polygon (matplotlib ``Path.contains_points``), so CONCAVE boundaries are
honoured and cost scales with the polygon's area, not the globe.  The previous
inline ``poly_net_field`` scan filled only the polygon's convex hull.

Addresses come straight from the mesh (``mesh.addr(L)`` -> one ``uuid.UUID`` per
hex, aligned with ``mesh[L]``).  The leading ``L+1`` nibbles of each UUID are the
H9 path, so a common prefix is shared across a clipped region; labels show the
per-hex suffix beyond that prefix.

Last Tested
-----------
28 Jun 2026 0.1.3a0 (migrated to HexMesh; b_oct render, UUID addresses)
13 Mar 2026 0.1.1a1 (passed, inline n_oct version)
26 Dec 2025 0.1.0a4 (working well)
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection

from hhg9 import Points, Registrar
from hhg9.h9 import H9K
from hhg9.h9.grid import HexMesh

# [birmingham-bristol-london-belgium] NON-CONVEX polygon, [lat, lon] degrees.
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


def bary_to_xy(uv: np.ndarray) -> np.ndarray:
    """Equilateral embedding of barycentric b_oct (u, v) into cartesian (x, y)."""
    u, v = uv[..., 0], uv[..., 1]
    return np.stack([u + v / 2.0, v * (H9K.R3 / 2.0)], axis=-1)


def common_prefix(hex_strs, cap):
    """Longest shared leading substring across hex_strs, capped at *cap* chars."""
    if not hex_strs:
        return ''
    p = hex_strs[0][:cap]
    for s in hex_strs[1:]:
        i, m = 0, min(len(p), len(s))
        while i < m and p[i] == s[i]:
            i += 1
        p = p[:i]
        if not p:
            break
    return p


def plot_polygrid(mesh, clip_xy, prefix, labels, ctrs_xy, name):
    """Draw the ancestry nest (finest on top) in b_oct, with suffix labels."""
    layers = mesh.layers
    finest = mesh.fine
    n_layers = len(layers)
    pts_xy = bary_to_xy(mesh.pts.coords)

    fine_v = pts_xy[mesh[finest]].reshape(-1, 2)
    xmin, xmax = float(fine_v[:, 0].min()), float(fine_v[:, 0].max())
    ymin, ymax = float(fine_v[:, 1].min()), float(fine_v[:, 1].max())
    ratio = (xmax - xmin) / (ymax - ymin) if ymax > ymin else 1.0

    fig = plt.figure(figsize=(ratio * 10, 10), dpi=200, frameon=False)
    ax = fig.add_subplot(111)
    pad_x, pad_y = 0.04 * (xmax - xmin), 0.04 * (ymax - ymin)
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()

    # Ancestry nest: coarser layers in muted blue, finest in black. (CVD-safe.)
    for i, L in enumerate(layers):
        faces = mesh[L]
        if len(faces) == 0:
            continue
        polys = pts_xy[faces]
        lw = 0.2 * ((n_layers - 1 - i) + 1)
        edge = 'black' if L == finest else '#0072b4a0'
        face = '#0072b420' if L == finest else 'none'
        ax.add_collection(PolyCollection(polys, linewidths=lw,
                                         facecolors=face, edgecolors=edge))
        print(f'  layer {L}: {len(faces)} hexes')

    # Clip polygon outline (Okabe-Ito vermillion dashed).
    ax.add_collection(PolyCollection([clip_xy], facecolors='none',
                                     edgecolors='#d55e00', linewidths=1.0,
                                     linestyles='--'))

    for pt, lab in zip(ctrs_xy, labels):
        ax.text(pt[0], pt[1], lab, color='k', fontsize=4, ha='center', va='center')
    ax.text(xmax, ymin, f'prefix: {prefix}', color='k', fontsize=18,
            ha='right', va='bottom')

    fig.savefig(f'output/ex0260_hx_{name}.png', dpi=200, bbox_inches='tight')
    print(f'fig saved at output/ex0260_hx_{name}.png')
    plt.close(fig)


if __name__ == '__main__':
    rg = Registrar()
    g_gcd = rg.domain('g_gcd')
    b_oct = rg.domain('b_oct')

    finest = 6
    nest = list(range(finest - 2, finest + 1))      # ancestry nest: L4..L6

    # Area-proportional, non-convex-aware clip; no global enumeration, no n_oct cast.
    mesh = HexMesh.create_clipped(nest, B_DLL, rg)
    print(mesh)

    # H9 UUID address per finest-layer hex (aligned with mesh[finest]).
    fine_addrs = [a.hex for a in mesh.addr(finest)]
    path_len = finest + 1                            # leading nibbles = H9 path
    prefix = common_prefix([a[:path_len] for a in fine_addrs], path_len)
    plen = len(prefix)
    labels = [a[plen:path_len] for a in fine_addrs]  # per-hex suffix
    print(f'finest L{finest}: {len(fine_addrs)} hexes, common prefix '
          f'{prefix!r} ({plen} nibbles)')
    for a in mesh.addr(finest)[:5]:
        print(f'  {a}')

    # Centroids (b_oct -> cartesian) for label placement.
    pts_xy = bary_to_xy(mesh.pts.coords)
    ctrs_xy = pts_xy[mesh[finest]].mean(axis=1)

    # Clip polygon outline in the same b_oct cartesian view.
    clip_b = rg.project(Points(B_DLL, g_gcd), [g_gcd, b_oct]).coords
    clip_xy = bary_to_xy(clip_b)

    plot_polygrid(mesh, clip_xy, prefix, labels, ctrs_xy, f'L{finest}')
