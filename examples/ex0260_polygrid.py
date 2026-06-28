# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Polygon -> hex grid + H9 addresses, rendered on a *dynamic local net*.

Given a NON-CONVEX lat/lon polygon and a finest hex layer, build the clipped hex
grid with its ancestry "nest" and an H9 UUID address per hex, then lay it flat.

The Britain–Belgium polygon spans two octants (0 & 2), so it has no single b_oct
plot — the octants live in disjoint per-octant frames.  Instead of the full fixed
n_oct net, we build a *dynamic* net for just the touched octants:

    layout = n_oct.local_layout(octants)          # hinge from a fundamental
    world  = n_oct.place(mesh.pts.coords, oids, layout.layout)

Each octant keeps its own proj.matrix (regular hexes) and only its offset is
recomputed relative to the fundamental, so the seam closes exactly and the region
lays flat with no tear.  This is n_oct, dynamic rather than defined.

Companions:
  ex0261_polygrid_mesh.py -- the same HexMesh source rendered in g_gcd (lat/lon).
  ex0261_polygrid.py      -- a simpler grid via hex_poly_layer.

Why HexMesh: ``create_clipped`` keeps a hex when its centroid lies inside the
polygon (matplotlib ``Path.contains_points``), so CONCAVE boundaries are honoured
and cost scales with the polygon's area.  Addresses come from ``mesh.addr(L)``
(one ``uuid.UUID`` per hex, aligned with ``mesh[L]``); the leading ``L+1`` UUID
nibbles are the H9 path, so the clipped region shares a common prefix.

Last Tested
-----------
28 Jun 2026 0.1.3a0 (dynamic local net; HexMesh + OctahedralNet.local_layout)
13 Mar 2026 0.1.1a1 (passed, inline n_oct version)
26 Dec 2025 0.1.0a4 (working well)
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection

from hhg9 import Points, Registrar
from hhg9.h9.grid import HexMesh

# [birmingham-bristol-london-belgium] NON-CONVEX polygon, [lat, lon]; spans octant 0 & 2.
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

# A NON-CONVEX C-shaped northern band spanning all FOUR northern octants
# (0,1,2,3).  Crucially it is OPEN over the Pacific, so it does NOT enclose the
# north pole (a cone vertex) — the four octants therefore unfold into one
# continuous arc with no tear.  (A *closed* ring round the pole would be forced
# to cut: the pole's 240° of face-angle cannot lie flat.)  Rendered at a coarse
# layer since it spans a quarter of the globe.  [lat, lon].
C_BAND = np.array([
    [68, -170], [68, -90], [68, 0], [68, 90], [68, 115],     # outer arc (65°N)
    [42, 115], [42, 90], [42, 0], [42, -90], [42, -170],      # inner arc (42°N)
])


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


def plot_polygrid(mesh, world, clip_xy, prefix, labels, ctrs_xy, name):
    """Draw the ancestry nest (finest on top) on the local net, with labels."""
    layers = mesh.layers
    finest = mesh.fine
    n_layers = len(layers)

    fine_v = world[mesh[finest]].reshape(-1, 2)
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
        lw = 0.2 * ((n_layers - 1 - i) + 1)
        edge = 'black' if L == finest else '#0072b4a0'
        face = '#0072b420' if L == finest else 'none'
        ax.add_collection(PolyCollection(world[faces], linewidths=lw,
                                         facecolors=face, edgecolors=edge))
        print(f'  layer {L}: {len(faces)} hexes')

    # Clip polygon outline (Okabe-Ito vermillion dashed).
    ax.add_collection(PolyCollection([clip_xy], facecolors='none',
                                     edgecolors='#d55e00', linewidths=1.0,
                                     linestyles='--'))

    if labels is not None:
        for pt, lab in zip(ctrs_xy, labels):
            ax.text(pt[0], pt[1], lab, color='k', fontsize=4,
                    ha='center', va='center')
    ax.text(xmax, ymin, f'prefix: {prefix!r}', color='k', fontsize=18,
            ha='right', va='bottom')

    fig.savefig(f'output/ex0260_hx_{name}.png', dpi=200, bbox_inches='tight')
    print(f'fig saved at output/ex0260_hx_{name}.png')
    plt.close(fig)


def run_case(rg, b_oct, g_gcd, n_oct, polygon, finest, name, with_labels=True):
    """Clip, lay flat on a dynamic local net, address, and plot one polygon."""
    nest = list(range(max(0, finest - 2), finest + 1))   # ancestry nest
    mesh = HexMesh.create_clipped(nest, polygon, rg)
    print(f'{name}: {mesh}')

    # Dynamic local net for the touched octants; lay the mesh flat.
    oids = np.asarray(mesh.pts.oid)
    octs = sorted(set(oids.tolist()))
    layout = n_oct.local_layout(octs)
    print(f'  octants {octs}: seam residual {layout.residual:.2e}, '
          f'unreached {layout.unreached}')
    world = n_oct.place(mesh.pts.coords, oids, layout.layout)

    # H9 UUID address per finest-layer hex (aligned with mesh[finest]).
    fine_addrs = [a.hex for a in mesh.addr(finest)]
    path_len = finest + 1                            # leading nibbles = H9 path
    prefix = common_prefix([a[:path_len] for a in fine_addrs], path_len)
    print(f'  finest L{finest}: {len(fine_addrs)} hexes, common prefix '
          f'{prefix!r} ({len(prefix)} nibbles)')

    if with_labels:
        labels = [a[len(prefix):path_len] for a in fine_addrs]   # per-hex suffix
        ctrs_xy = world[mesh[finest]].mean(axis=1)
    else:
        labels, ctrs_xy = None, None                 # too many to draw legibly

    clip_pts = rg.project(Points(polygon, g_gcd), [g_gcd, b_oct])
    clip_xy = n_oct.place(clip_pts.coords, clip_pts.oid, layout.layout)
    plot_polygrid(mesh, world, clip_xy, prefix, labels, ctrs_xy, name)


if __name__ == '__main__':
    rg = Registrar()
    g_gcd = rg.domain('g_gcd')
    b_oct = rg.domain('b_oct')
    n_oct = rg.domain('n_oct:butterfly')

    # Two-octant non-convex region (Britain–Belgium), fine layer + labels.
    run_case(rg, b_oct, g_gcd, n_oct, B_DLL, finest=6, name='L6')

    # Four-octant non-convex C-band (no pole enclosed) — dropped to a coarse
    # layer since it spans a quarter of the globe; labels omitted for legibility.
    run_case(rg, b_oct, g_gcd, n_oct, C_BAND, finest=3, name='cband_L3',
             with_labels=False)
