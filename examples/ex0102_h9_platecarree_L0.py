# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Plate carrée painted by L0 hexagon — every pixel coloured into its base cell.

Companion to ex0101_h9_grid.py, which colours the pixels of an *n_oct* net by
their hexagon id.  Here the canvas is instead an equirectangular (plate carrée)
world raster — x = longitude ∈ [-180, 180], y = latitude ∈ [-90, 90] — so the
result reads as a familiar world map carved into the 12 L0 hexagons.

Two roles, kept separate on purpose:

  * FILL  — each pixel centre (lon, lat) is projected g_gcd -> b_oct and
    classified to its L0 hexagon id (0..11) with the same vectorised root-hex
    lookup that ``h9_bin_pts`` uses at L0.  This is what "colour each pixel into
    its L0 hexagon" means, and it is exact everywhere: the colour-region
    boundaries ARE the true (curved) hexagon edges, including across the ±180°
    seam and over the poles, because every pixel is classified independently.

  * MESH  — ``HexMesh.create([0], reg)`` supplies the 12 canonical L0 cells:
    their bin addresses (printed as a legend) and own-octant centroids (the
    in-map id labels).  We deliberately do NOT stroke the mesh polygons: an L0
    hexagon's true edge is a curve in plate carrée (the 4 polar caps even run
    along the pole line), so a straight-edge polygon would contradict the fill.
    The colour-region boundaries already are the exact hexagon edges.

Because the 12 ids are categorical (and a CVD-safe 12-way palette is
impossible), each region is also labelled with its hex id, so the map never
relies on colour alone.

Last Tested
-----------
28 Jun 2026 0.1.3a0 (new; plate carrée L0 pixel classification + HexMesh labels)
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from hhg9 import Registrar, Points
from hhg9.h9 import H9O, H9R
from hhg9.h9.region import xy_regions
from hhg9.h9.grid import HexMesh

WIDTH = 1440          # 0.25° pixels: 1440 x 720 over the whole sphere
HEIGHT = WIDTH // 2


def l0_digits(b_pts):
    """Vectorised L0 hexagon id (0..11) per b_oct point — the L0 body nibble.

    Mirrors the L0 branch of ``h9_bin_pts``: classify into the root region, map
    to its c2 colour, then look up the canonical base-hexagon id per octant.
    """
    oc, mo = b_pts.cm()
    cx = xy_regions(b_pts.coords, mo, 0)
    c2 = H9R.mcc2[mo, cx[:, 1]]
    c2 = np.where(c2 == 0x5F, np.uint8(0), c2)
    return H9O.l0hex_by_id[oc, c2]


def hex_centroids(mesh, reg, b_oct, g_gcd):
    """Own-octant centroid of each L0 hex, as (lat, lon) — for in-map labels."""
    faces = mesh[0]
    cb = np.asarray(mesh.pts.coords)[faces]
    co = np.asarray(mesh.pts.oid)[faces]
    match = (H9O.oid_mo[co] == H9O.oid_mo[co][:, :1])         # own-octant verts
    cen = (cb * match[:, :, None]).sum(1) / match.sum(1, keepdims=True)
    cpts = Points(cen, b_oct, oid=co[:, 0].astype(np.int32))
    return reg.project(cpts, [b_oct, g_gcd]).coords            # (12, 2) lat,lon


def run():
    reg = Registrar()
    g_gcd = reg.domain('g_gcd')
    b_oct = reg.domain('b_oct')

    # Plate carrée pixel-centre grid (row 0 = top = +90° lat).
    lon = (np.arange(WIDTH) + 0.5) / WIDTH * 360.0 - 180.0
    lat = 90.0 - (np.arange(HEIGHT) + 0.5) / HEIGHT * 180.0
    LON, LAT = np.meshgrid(lon, lat)

    pts = reg.project(Points(np.column_stack([LAT.ravel(), LON.ravel()]), g_gcd),
                      [g_gcd, b_oct])
    img = l0_digits(pts).reshape(HEIGHT, WIDTH)
    print(f'{WIDTH}x{HEIGHT} pixels classified into {len(np.unique(img))} L0 hexagons')

    # HexMesh L0: canonical addresses (legend) + centroids/outlines.
    mesh = HexMesh.create([0], reg)
    cents = hex_centroids(mesh, reg, b_oct, g_gcd)
    addrs = mesh.addrs
    # face id -> L0 digit (first nibble of the bin UUID).
    face_id = np.array([(a.int >> 124) & 0xF for a in addrs])
    print('L0 hexagons (id : bin address):')
    for i in np.argsort(face_id):
        print(f'  {int(face_id[i]):2d} : {addrs[i]}')

    # 12-way categorical palette (no greys — they read as nodata; labels carry
    # the id too, so the map never leans on colour alone).
    cmap = plt.colormaps.get_cmap('tab20')
    cols = cmap(np.linspace(0, 1, 20))[[0, 2, 4, 6, 8, 10, 12, 16, 18, 1, 3, 5]]

    fig = plt.figure(figsize=(14, 7.4), dpi=200)
    ax = fig.add_subplot(111)
    ax.imshow(img, extent=[-180, 180, -90, 90], origin='upper',
              cmap=ListedColormap(cols), vmin=-0.5, vmax=11.5,
              interpolation='nearest')

    # Per-hexagon id label at its own-octant centroid.
    for i, f in enumerate(mesh[0]):
        clat, clon = cents[i]
        ax.text(clon, np.clip(clat, -82, 82), f'{int(face_id[i])}',
                ha='center', va='center', fontsize=11, fontweight='bold',
                color='white', path_effects=None,
                bbox=dict(boxstyle='circle,pad=0.18', fc='black', ec='none', alpha=0.55))

    # Graticule + seam + equator.
    for x in range(-180, 181, 30):
        ax.axvline(x, color='white', lw=0.25, alpha=0.35)
    for y in range(-60, 61, 30):
        ax.axhline(y, color='white', lw=0.25, alpha=0.35)
    ax.axvline(180, color='#cc0000', lw=0.8, alpha=0.7)
    ax.axvline(-180, color='#cc0000', lw=0.8, alpha=0.7)

    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    ax.set_xticks(range(-180, 181, 60)); ax.set_yticks(range(-90, 91, 30))
    ax.set_xlabel('longitude'); ax.set_ylabel('latitude')
    ax.set_title('Hex9 L0 — every plate carrée pixel coloured into its base hexagon')

    fig.savefig('output/ex0102_h9_platecarree_L0.png', dpi=200, bbox_inches='tight')
    print('fig saved at output/ex0102_h9_platecarree_L0.png')
    plt.close(fig)


if __name__ == '__main__':
    run()
