# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
This reads in the GCD for Stonehenge, then converts it into a set of nested half-hexagons,
each of which represents a single Stage of the H9 Journey.

Last Tested
13 Mar 2026 0.1.1a1 (passed)
26 Dec 2025 0.1.0a4 (passed)
16 Dec 2025 0.1.0a3 (passed)
25 Nov 2025 (passed)
"""
import os
from pathlib import Path

import numpy as np
import contextily as cx
from hhg9 import Registrar, Points
from hhg9.algorithms.distance import densify_poly_geodesic_by_step
from hhg9.h9.binning import hex_reduce, hex_parents, hex_from_pars


def gather_image(extent, zoom, title):
    """use OSM web source.  Cache ON!"""
    f_name = f'output/ex0310/{title}_{zoom:02}.tif'
    file_to_keep = Path(f_name)
    if not file_to_keep.exists():
        west, south, east, north = extent  # minx, miny, maxx, maxy
        cx.bounds2raster(west, south, east, north, f_name, zoom=zoom, wait=5, ll=True, use_cache=True, source=cx.providers.OpenStreetMap.Mapnik)
    return f_name


def zoom(lon_width, pixels):
    """
    Calculate the required OSM zoom level for an image width in pixels
    This is for Web Mercator (EPSG:3857) composites.
    """
    tiles = pixels / 256.0  # tiles are 256x
    ideal_zoom = np.log2(tiles * (360.0 / lon_width))  # The world is 360 degrees wide.
    return int(min(19, np.ceil(ideal_zoom)))


if __name__ == '__main__':
    """
        Compose a set of zooms into stonehenge..
    """
    reg = Registrar()  # Manage Domains & Projections
    g_gcd = reg.domain('g_gcd')
    b_oct = reg.domain('b_oct')
    p_pix = reg.domain('p_pix')
    pix_gcd = reg.projection('pix_gcd')
    n_oct = reg.domain('n_oct:butterfly:2000')

    name = 'Stonehenge'
    spot = [51.178798000021075, -1.8261898473293334]
    ll0 = Points(np.array([spot]), g_gcd)
    b_pts = reg.project(ll0, [g_gcd, b_oct])

    repo = {
        'name': name,
        'noct': n_oct.name,
        'spot': spot,
        'layers': None
    }
    layer_store = list()
    for layer in range(1, 9):
        # The pattern is hex_reduce → hex_parents → hex_from_pars
        hex_num, hex_v, hex_inv, hex_idx = hex_reduce(b_pts, layer)
        hex_xy, hex_oid, scale = hex_parents(b_oct, hex_v, hex_num, layer)
        hex_p = hex_from_pars(b_oct, hex_xy, hex_oid, scale, hex_v[:, -1])
        hex_n = reg.project(hex_p, [b_oct, n_oct])
        hex_g = reg.project(hex_p, [b_oct, g_gcd])
        dense = densify_poly_geodesic_by_step(hex_g.coords, max_per_edge=25)
        d_pts = Points(dense, g_gcd)
        s_bounds, s_diff = hex_g.bbox(return_diff=True)
        bounds, diff = d_pts.bbox(return_diff=True)  # xmin, ymin, xmax, ymax
        zoom_size = zoom(diff[0], 4096)
        tif_name = gather_image(bounds, zoom_size, f'ex0310_{layer:02}')
        layer_meta = {
                'layer': layer,
                'tif': tif_name,
                'zoom': zoom_size,
                'hex_n_coords': hex_n.coords,
                'hex_n_cmp': hex_n.components,
            }
        layer_store.append(layer_meta)
    np.savez(
        f"output/ex0310/repo.npz",
        name=repo['name'],
        noct=repo['noct'],
        spot=repo['spot'],
        layers=layer_store
    )

