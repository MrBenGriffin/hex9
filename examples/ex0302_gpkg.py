# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Export UK hex grid to GeoPackage and OSM backdrop to GeoTIFF.

Produces:
  output/ex0302_uk.gpkg        — two polygon layers (h9_L04, h9_L05) in WGS84
  output/ex0302_uk_n.gpkg      — same layers in n_oct pixel space
  output/ex0302_uk_osm.tif     — plate-carrée GeoTIFF aligned with the gpkg
  output/ex0302_uk_osm_n.tif   — n_oct-space GeoTIFF with H9 metadata

Requires the OSM cache from ex0075:
  output/ex0075.png            — regional plate-carrée image
  output/ex0075_extents.npy    — [lon_min, lon_max, lat_min, lat_max]

Run ex0075_osm_mesh.py first to generate those files.

Last Tested
13 Mar 2026 0.1.1a1 (passed)
"""

import os

import numpy as np
from matplotlib import image

from hhg9 import Registrar, Points
from hhg9.algorithms.distance import densify_quad_geodesic_by_step
from hhg9.rendering.composition import LayerSpec, Compositor, make_backdrop
from hhg9.geo.export import (
    from_composed, layers_to_gpkg,
    backdrop_to_geotiff, net_to_geotiff,
    load_gpkg, load_net_geotiff,
)

os.makedirs('output', exist_ok=True)


if __name__ == '__main__':
    print('Initialising')
    rg    = Registrar()
    g_gcd = rg.domain('g_gcd')
    b_oct = rg.domain('b_oct')
    c_oct = rg.domain('c_oct')
    c_ell = rg.domain('c_ell')
    n_oct = rg.domain('n_oct:butterfly')

    # ------------------------------------------------------------------ region
    osm_available = (
        os.path.exists('output/ex0075.png') and
        os.path.exists('output/ex0075_extents.npy')
    )

    if osm_available:
        print('Loading OSM image')
        osm_img   = image.imread('output/ex0075.png')[:, :, :3]   # (H, W, 3) float32
        img_h, img_w = osm_img.shape[:2]
        zone      = np.load('output/ex0075_extents.npy')           # [lon_min, lon_max, lat_min, lat_max]
        lon_min, lon_max, lat_min, lat_max = zone
    else:
        print('OSM cache not found — using fixed UK bounding box')
        lon_min, lon_max = -8.0, 2.0
        lat_min, lat_max =  49.5, 61.0
        osm_img = None

    # Build region polygon in n_oct
    corners = np.array([
        [lat_min, lon_min], [lat_min, lon_max],
        [lat_max, lon_max], [lat_max, lon_min],
    ])
    qx   = densify_quad_geodesic_by_step(corners, max_step_m=50_000)
    ntt  = rg.project(Points(qx, g_gcd), [g_gcd, b_oct, n_oct])
    nlt  = float(ntt.coords[:, 0].min())
    nrt  = float(ntt.coords[:, 0].max())
    nbt  = float(ntt.coords[:, 1].min())
    ntp  = float(ntt.coords[:, 1].max())
    bbox_n = (ntp, nlt, nbt, nrt)

    fill_n = Points(
        np.array([[nlt, nbt], [nrt, nbt], [nrt, ntp], [nlt, ntp]]),
        n_oct,
    )

    # ---------------------------------------------------------------- backdrop
    if osm_img is not None:
        def osm_sampler(ctrs: Points) -> np.ndarray:
            gcd  = rg.project(ctrs, [b_oct, c_oct, c_ell, g_gcd])
            lats = gcd.coords[:, 0]
            lons = gcd.coords[:, 1]
            cols = np.floor((lons - lon_min) / (lon_max - lon_min) * img_w).astype(np.int32)
            rows = np.floor((lat_max - lats) / (lat_max - lat_min) * img_h).astype(np.int32)
            valid = (cols >= 0) & (cols < img_w) & (rows >= 0) & (rows < img_h)
            cols  = np.clip(cols, 0, img_w - 1)
            rows  = np.clip(rows, 0, img_h - 1)
            out   = osm_img[rows, cols].copy()
            out[~valid] = 0.0
            return out

        print('Building backdrop (2048 × 2048)')
        backdrop = make_backdrop(rg, b_oct, n_oct, bbox_n, 2048, 2048, osm_sampler)
    else:
        backdrop = None

    # ----------------------------------------------------------- hex compositor
    specs = [
        LayerSpec(level=4, kind='outline', labels=True, reference=True,
                  style={'lw': 0.8, 'ec': '#ffffffa0'}),
        LayerSpec(level=5, kind='outline', labels=True, threshold=2,
                  style={'lw': 0.3, 'ec': '#ffffff60'}),
    ]
    print('Running compositor')
    composed = Compositor(rg, b_oct, n_oct, specs).run(fill_n)
    print(f'  → {len(composed)} layers')

    # ------------------------------------------------------- WGS84 GeoPackage
    print('Building WGS84 HexLayers')
    layers_gcd = [from_composed(cl, rg, crs='g_gcd') for cl in composed]
    print('Writing output/ex0302_uk.gpkg')
    layers_to_gpkg(layers_gcd, 'output/ex0302_uk.gpkg')

    # --------------------------------------------------- n_oct GeoPackage
    print('Building n_oct HexLayers')
    layers_n = [from_composed(cl, rg, crs='n_oct') for cl in composed]
    print('Writing output/ex0302_uk_n.gpkg')
    layers_to_gpkg(layers_n, 'output/ex0302_uk_n.gpkg')

    # -------------------------------------------------------- GeoTIFF rasters
    if osm_img is not None:
        print('Writing output/ex0302_uk_osm.tif  (plate-carrée WGS84)')
        # osm_img row 0 = top (lat_max) — correct for backdrop_to_geotiff
        backdrop_to_geotiff(osm_img, lat_min, lon_min, lat_max, lon_max,
                            'output/ex0302_uk_osm.tif')

        print('Writing output/ex0302_uk_osm_n.tif  (n_oct space)')
        # backdrop row 0 = bottom — net_to_geotiff flips internally
        net_to_geotiff(backdrop, bbox_n, 'output/ex0302_uk_osm_n.tif')

    # --------------------------------------------------------- round-trip test
    print('\nRound-trip verification')
    rt_layers = load_gpkg('output/ex0302_uk.gpkg')
    print(f'  load_gpkg: {len(rt_layers)} layers')
    for rl in rt_layers:
        has_addrs = any(a for a in rl.addresses)
        print(f'    {rl.name}: {len(rl.ctrs)} hexes, '
              f'crs={rl.crs!r}, addresses={"yes" if has_addrs else "empty"}')

    if osm_img is not None:
        arr, bbox_rt, layout_rt = load_net_geotiff('output/ex0302_uk_osm_n.tif')
        print(f'  load_net_geotiff: shape={arr.shape}, layout={layout_rt!r}')
        tp_err = abs(bbox_rt[0] - bbox_n[0])
        print(f'  bbox_n tp error: {tp_err:.2e}  {"OK" if tp_err < 1e-9 else "MISMATCH"}')

    print('\nDone.')
