# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Global authalic-deviation GeoPackage for QGIS.

Builds a GeoPackage of all H9 hexagons at the requested layer, with
pct_deviation and log_ratio attached as attributes. Load into QGIS,
apply a diverging graduated fill on `pct_deviation`, then export in
Mollweide (EPSG:54009) for the paper figure.

Outputs:
  output/ex0119_authalic_L{layer}.gpkg

Usage:
  python ex0119_authalic_gpkg.py           # default L4
  python ex0119_authalic_gpkg.py --layer 5 # L5 (paper quality, ~10 min)

QGIS quick-start:
  1. Layer → Add Layer → Add Vector Layer → choose the .gpkg
  2. Layer CRS: EPSG:4326 (WGS84 geographic)
  3. Properties → Symbology → Graduated → Column: pct_deviation
     Color ramp: RdBu (reversed), Classes: 10, Mode: Equal interval, centred at 0
  4. Project → Properties → CRS → EPSG:54009 (Mollweide) for on-screen view
  5. Project → Import/Export → Export Map to Image / PDF

Last Tested
16 Jun 2026 0.1.3a0 (fail)
"""

import argparse
import os
import numpy as np
from hhg9 import Registrar
from hhg9.algorithms.distance import wgs84_area, ellipsoid_area_wgs84
from hhg9.h9.grid import HexMesh
from hhg9.geo.export import HexLayer, layers_to_gpkg
os.makedirs('output', exist_ok=True)


def build_gpkg(reg: Registrar, depth: int) -> None:
    print(f'Layer {depth}: generating hex polygons...')
    mesh = HexMesh.create([depth], reg)
    pts = mesh.pts
    faces = mesh.faces

    gm2 = ellipsoid_area_wgs84()  # 510_065_621_724_088.509
    bins = 12 * 9 ** depth
    ideal_m2 = gm2 / bins

    # ── Project to lat/lon and compute areas ───────────────────────────────
    g_pts = reg.project(pts, ['b_oct', 'g_gcd'])    # (18140, 2) [lat°, lon°]
    hexes = g_pts.coords[faces]

    w_area_m2 = wgs84_area(reg, hexes)
    ratio = w_area_m2 / ideal_m2
    pct_dev = (ratio - 1.0) * 100.0
    log_ratio = np.log(ratio)
    centroids = hexes.mean(axis=1)                      # (N, 2) [lat, lon]
    n_hex = hexes.shape[0]
    print(f'  {n_hex:,} hexagons')
    print(f'  pct_dev: min={pct_dev.min():+.3f}%  mean={pct_dev.mean():+.3f}%  '
          f'max={pct_dev.max():+.3f}%  MAE={np.mean(np.abs(pct_dev)):.4f}%')

    # ── Build HexLayer directly ────────────────────────────────────────────
    layer = HexLayer(
        level=depth,
        name=f'h9_authalic_L{depth:02d}',
        crs='g_gcd',
        polys=hexes,                              # (N, 6, 2) [lat, lon]
        ctrs=centroids,                           # (N, 2)    [lat, lon]
        addresses=np.full(n_hex, '', dtype=object),
        parent_addresses=np.full(n_hex, '', dtype=object),
        key_tails=np.zeros(n_hex, dtype=np.uint8),
        fields={
            'pct_deviation': pct_dev,
            'log_ratio':     log_ratio,
            'area_m2':       w_area_m2,
        },
    )

    out = f'output/ex0119_authalic_L{depth}.gpkg'
    layers_to_gpkg([layer], out)
    print(f'  Saved: {out}')
    print('  QGIS: style by pct_deviation with RdBu_r diverging colour, centred at 0.')
    print('  Set project CRS to EPSG:54009 (Mollweide) for paper-layout view.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=int, default=5, help='H9 layer depth (default 5)')
    args = parser.parse_args()

    rg = Registrar()
    build_gpkg(rg, args.layer)
