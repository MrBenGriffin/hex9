# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
import json

import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.transform import from_origin
from shapely.geometry import box
import numpy as np


if __name__ == '__main__':

    # 1. Setup: Custom Projection (WKT)
    # We use this to ensure the output matches your other layers perfectly.
    target_crs_wkt = 'PROJCS["WGS_1984_Albers",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Albers_Conic_Equal_Area"],PARAMETER["latitude_of_center",50],PARAMETER["longitude_of_center",-154],PARAMETER["standard_parallel_1",55],PARAMETER["standard_parallel_2",65],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["meters",1],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'

    # 2. Setup: Region of Interest (ROI) boundaries (Lat/Lon)
    # Taken from your coordinates
    # min_lon, min_lat = -157.00023116, 57.01809322
    # max_lon, max_lat = -156.75239106, 57.15278955

    # 3. Load and Reproject Data
    print("Loading Shapefile...")
    # Replace with your actual path
    gdf = gpd.read_file("src/chiginagak_gems_shapefile_pkg/chiginagak_gems-simple/MapUnitPolys.shp")

    print("Reprojecting to custom Albers...")
    gdf = gdf.to_crs(target_crs_wkt)

    # gdf['lookup_id'] = gdf.groupby(['GeoMat', 'RGB']).ngroup() + 1
    gdf['lookup_id'] = gdf.groupby(['GeoMat']).ngroup() + 1  # 0 = no data.
    gdf['lookup_id'] = gdf['lookup_id'].astype('uint8')  # Ensure it fits in 8 bits
    print(f"   - Found {gdf['lookup_id'].max()} unique rock types.")

    # 4. Define the Raster Grid
    # We need to know the bounds of your ROI in the *Target Projection*
    # So we create a bounding box in Lat/Lon and project it
    # roi_box = gpd.GeoDataFrame({'geometry': [box(min_lon, min_lat, max_lon, max_lat)]}, crs="EPSG:4326")
    # roi_projected = roi_box.to_crs(target_crs_wkt)
    # bounds = roi_projected.total_bounds  # [minx, miny, maxx, maxy]
    bounds = gdf.total_bounds

    # Define Resolution (Pixel Size in Meters)
    # 5 meters is a good balance for L9 (approx 100m hexes)
    pixel_size = 5

    width = int((bounds[2] - bounds[0]) / pixel_size)
    height = int((bounds[3] - bounds[1]) / pixel_size)

    # Create the Transform (mapping pixels to coordinates)
    # Note: Rasters usually start from Top-Left (North-West)
    transform = from_origin(bounds[0], bounds[3], pixel_size, pixel_size)

    print(f"Output Raster Size: {width}x{height} pixels")

    # # 5. Prepare Colors
    # # We parse "153,153,255" into three columns: R, G, B
    # # If RGB is missing/null, default to Grey (128,128,128)
    # def parse_rgb(row):
    #     try:
    #         if isinstance(row['RGB'], str):
    #             parts = [int(x) for x in row['RGB'].split(',')]
    #             if len(parts) == 3:
    #                 return parts
    #     except:
    #         pass
    #     return [128, 128, 128]  # Default Grey
    #
    # # Apply parser
    # gdf[['R', 'G', 'B']] = gdf.apply(parse_rgb, axis=1, result_type='expand')
    #
    # unique_units = gdf[['RGB', 'GeoMat']].drop_duplicates()

    # --- JSON GENERATION (NEW STEP) ---
    print("3. Generating JSON Dictionary...")
    lookup_table = {}
    # Iterate over unique rock types to build the master list
    unique_rocks = gdf[['lookup_id', 'GeoMat', 'RGB', 'Label']].drop_duplicates(subset='lookup_id').sort_values('lookup_id')
    for index, row in unique_rocks.iterrows():
        # Helper to clean the RGB string into a list [R, G, B]
        rgb_list = [128, 128, 128]  # Default
        if isinstance(row['RGB'], str):
            try:
                rgb_list = [int(x) for x in row['RGB'].split(',')]
            except:
                pass
        # Structure: ID -> { Description, Color }
        # This gives you the best of both worlds: Light TIF, Rich Metadata
        lookup_table[int(row['lookup_id'])] = {
            "desc": row['GeoMat'].strip() if row['GeoMat'] else "",
            "color": rgb_list,
            "label": row['Label'].strip() if row['Label'] else "",
        }
    output_json = 'src/chiginagak_geology.json'
    with open(output_json, 'w') as f:
        json.dump(lookup_table, f, indent=4)
    print(f" - Saved lookup for {len(lookup_table)} rock types to {output_json}")

    shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf['lookup_id']))
    image_index = features.rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,  # 0 = No Data (Background)
        dtype='uint8'
    )

    # Save as single-band uint8
    output_tif = "src/chiginagak_geology.tif"
    with rasterio.open(
            output_tif, 'w', driver='GTiff',
            height=height, width=width, count=1, dtype='uint8',  # 1 Band Only
            crs=target_crs_wkt, transform=transform, nodata=0
    ) as dst:
        dst.write(image_index, 1)

    print("A baked geology map.")