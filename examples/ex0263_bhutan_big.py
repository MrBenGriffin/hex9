# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Bhutan population choropleth — H9 bin addresses as the join key, across a seam.

``src/bhutan.geojson`` is 161k population points (lon, lat, population, h9).
Bhutan straddles the 90°E octant seam, so the hex grid crosses it; in g_gcd
(lat/lon) the seam is continuous and the country sits at ~27°N where distortion
is mild.

Join by canonical H9 bin, NOT by string-truncating the UUID (which is fragile —
sentinel boundaries, tail/key bits, and the L0 12-digit case all break it).
Both sides are binned to layer L with the same method, ``h9_bin_pts``:

  * each population point's coordinate -> its layer-L bin UUID  (sum population)
  * each hex's centroid               -> its layer-L bin UUID  (look up population)

so the bin UUIDs compare for exact equality (≈100% of in-box points land on a
hex).  Hexes with no population are left transparent, so the populated cells
trace Bhutan's shape inside the bounding-box clip.  Colour map is the CVD-safe
``cividis`` on a log scale.

NOTE: the file's ``h9`` property does not round-trip through ``h9_dec`` (digit
out of range), so we bin from the authoritative coordinates with ``h9_bin_pts``
rather than trusting/decoding the stored UUID.

Companion to ex0262_greenwich_seam.py / ex0263_89_seam_zoom.py (seam tests).

Last Tested
-----------
28 Jun 2026 0.1.3a0 (new; population choropleth via canonical H9 bin join)
"""
import json
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt, colors, cm
from matplotlib.collections import PolyCollection

from hhg9 import Points, Registrar
from hhg9.h9 import H9O
from hhg9.h9.grid import HexMesh
from hhg9.h9.uuid_address import h9_bin_pts

LAYER = 7                                   # ~3.5 km hexes over Bhutan


def load_population(path):
    """Return (lon, lat, pop) arrays from the point GeoJSON."""
    feats = json.load(open(path))['features']
    lon = np.array([f['properties']['longitude'] for f in feats])
    lat = np.array([f['properties']['latitude'] for f in feats])
    pop = np.array([f['properties']['population'] for f in feats], dtype=float)
    return lon, lat, pop


def hex_centroid_bins(mesh, layer, b_oct):
    """Canonical layer-L bin UUID for each hex (binned from its own-octant centroid)."""
    faces = mesh[layer]
    cb, co = np.asarray(mesh.pts.coords)[faces], np.asarray(mesh.pts.oid)[faces]
    match = (H9O.oid_mo[co] == H9O.oid_mo[co][:, :1])         # own-octant verts
    cen = (cb * match[:, :, None]).sum(1) / match.sum(1, keepdims=True)
    return h9_bin_pts(Points(cen, b_oct, oid=co[:, 0].astype(np.int32)), layer)


if __name__ == '__main__':
    reg = Registrar()
    b_oct = reg.domain('b_oct')
    g_gcd = reg.domain('g_gcd')

    lon, lat, pop = load_population('src/bhutan.geojson')
    print(f'{len(pop)} points, total population {pop.sum():,.0f}')

    # Aggregate population by canonical layer-L bin of each point.
    pts_b = reg.project(Points(np.column_stack([lat, lon]), g_gcd), [g_gcd, b_oct])
    pop_by_bin = defaultdict(float)
    for u, p in zip(h9_bin_pts(pts_b, LAYER), pop):
        pop_by_bin[u] += p

    # Matching hex grid over the data's bounding box, clipped per centroid.
    poly = np.array([[lat.min(), lon.min()], [lat.min(), lon.max()],
                     [lat.max(), lon.max()], [lat.max(), lon.min()]])
    mesh = HexMesh.create_clipped([LAYER], poly, reg)
    print(mesh)

    # Join on the same canonical bin (NOT a UUID-string truncation).
    hex_pop = np.array([pop_by_bin.get(u, 0.0)
                        for u in hex_centroid_bins(mesh, LAYER, b_oct)])
    has = hex_pop > 0
    print(f'L{LAYER}: {len(hex_pop)} hexes, {int(has.sum())} populated, '
          f'joined population {hex_pop.sum():,.0f}')

    # Geometry in g_gcd (lon, lat); seam is continuous there.
    lonlat = reg.project(mesh.pts, [b_oct, g_gcd]).coords
    polys = lonlat[mesh[LAYER]][:, :, ::-1]            # (N, 6, 2) lon-lat

    norm = colors.LogNorm(vmin=max(1.0, hex_pop[has].min()), vmax=hex_pop.max())
    rgba = cm.cividis(norm(np.where(has, hex_pop, np.nan)))
    rgba[~has] = [0, 0, 0, 0]                          # nodata transparent

    fig = plt.figure(figsize=(11, 7), dpi=200)
    ax = fig.add_subplot(111)
    ax.add_collection(PolyCollection(
        polys, facecolors=rgba, edgecolors=rgba, linewidths=0.1))
    ax.set_xlim(lon.min(), lon.max())
    ax.set_ylim(lat.min(), lat.max())
    ax.set_aspect(1.0 / np.cos(np.radians(lat.mean())))   # metric-ish at 27°N
    ax.axvline(90.0, color='#cc000080', lw=0.6, ls='--')  # 90°E octant seam
    ax.set_title(f'Bhutan population — H9 L{LAYER} hexes (join key = H9 bin)')
    ax.set_xlabel('longitude'); ax.set_ylabel('latitude')

    sm = cm.ScalarMappable(norm=norm, cmap='cividis')
    fig.colorbar(sm, ax=ax, label='population per hex', shrink=0.8)

    fig.savefig('output/ex0263_bhutan_big.png', dpi=200, bbox_inches='tight')
    print('fig saved at output/ex0263_bhutan_big.png')
    plt.close(fig)
