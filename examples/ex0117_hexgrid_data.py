# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
For a given layer, compose a reference address for each
hexagon in that layer - generate the set of hexagons at that layer
and display on the globe.

Last Tested
13 Mar 2026 0.1.1a1 (passed)
26 Dec 2025 0.1.0a4 (passed)
16 Dec 2025 0.1.0a3 (passed)
25 Nov 2025 (passed)
"""
import numpy as np
from hhg9 import Registrar
from hhg9.h9.grid import HexMesh
from hhg9.h9.uuid_address import h9_label
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon


def export_grid(coords_pool, layers_data, output_path):
    """
    Export a multi-layer H9 hex grid to GeoParquet.

    Parameters
    ----------
    coords_pool  : (M, 2) ndarray, lon/lat for every vertex in the mesh
    layers_data  : list of (hex_indices, uuids) — one entry per layer,
                   hex_indices shape (N, V) where V = 6 * 3^delta for
                   densified layers or 6 for the finest layer
    output_path  : str or Path
    """
    all_layer_data = []

    for depth, (hex_indices, uuids) in enumerate(layers_data):
        print(f"Assembling Layer {depth} ({len(hex_indices)} hexes)...")
        layer_coords = coords_pool[hex_indices]          # (N, V, 2)
        polys = [Polygon(h) for h in layer_coords]

        layer_df = pd.DataFrame({
            'h9_id':    [str(u) for u in uuids],         # canonical UUID string
            'h9_label': [h9_label(u) for u in uuids],    # human-readable, roundtrippable
            'layer':    np.full(len(polys), depth, dtype='uint8'),
            'geometry': polys,
        })
        all_layer_data.append(layer_df)

    print("Finalizing GeoParquet...")
    full_df = pd.concat(all_layer_data, ignore_index=True)
    gdf = gpd.GeoDataFrame(full_df, crs="EPSG:4326")
    gdf.to_parquet(output_path, compression='zstd')
    print(f"{len(gdf)} hexagons exported to {output_path}")
    # QGIS display note: antimeridian-crossing hexes render incorrectly in
    # geographic/Mercator views.  Use a feature filter expression to hide them:
    #   (x_max(@geometry) - x_min(@geometry)) < 180


if __name__ == '__main__':
    MAX_LAYER = 6                                        # layers 0..3; finest = 3
    reg = Registrar()
    mesh = HexMesh.create(range(MAX_LAYER+1), reg)

    b_oct = reg.domain('b_oct')
    g_gcd = reg.domain('g_gcd')
    lon_lat = reg.project(mesh.pts, [b_oct, g_gcd]).coords[:, ::-1]  # (M, 2) lon/lat

    layers_data = [
        (mesh.densify(i), mesh.addr(i))
        for i in range(MAX_LAYER+1)
    ]
    export_grid(lon_lat, layers_data, output_path=f'output/grid_0117_{MAX_LAYER}.parquet')


