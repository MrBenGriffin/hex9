# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon


if __name__ == '__main__':

    # https://apps.nationalmap.gov/downloader/ Has geotiff heights.
    # 1. Load the Geology Map
    # The specific hex_layer usually ends in "MapUnit" or "GeoPolygon" inside the GDB/Folder
    # Replace with the actual path to the .shp file extracted from the zip
    # roi_coords = [
    #     (-156.752, 57.15278955),  # Top-Right
    #     (-156.752, 57.01809322),  # Bottom-Right
    #     (-157.1, 57.01809322),  # Bottom-Left
    #     (-157.1, 57.15278955)  # Top-Left
    # ]
    # roi_poly = Polygon(roi_coords)
    # roi_gdf = gpd.GeoDataFrame(index=[0], geometry=[roi_poly], crs="EPSG:4326")
    geo_data = gpd.read_file("src/chiginagak_gems_shapefile_pkg/chiginagak_gems-simple/MapUnitPolys.shp")
    # roi_projected = roi_gdf.to_crs(geo_data.crs)
    # clipped_geology = gpd.clip(geo_data, roi_projected)
    # final_data = clipped_geology[['MapUnit', 'GeoMat', 'RGB', 'geometry']]
    geo_data.to_file("output/volcano.geojson", driver="GeoJSON")

    # Ensure it's in a projected coordinate system (like UTM Zone 4N for Alaska)
    # EPSG:3338 is a common Albers projection for Alaska, but check the file's CRS.
    if geo_data.crs.to_string() != "EPSG:3338":
        geo_data = geo_data.to_crs("EPSG:3338")

    print(geo_data.columns)
    labs = np.array(geo_data['Label'])
    lb = np.unique(labs)
    print(geo_data[['MapUnit', 'IdeConf', 'GeoMat', 'RGB']].head())

