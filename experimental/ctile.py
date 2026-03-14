# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0


import contextily as cx


def calculate_4k_zoom(minx, maxx):
    """
    Calculates the required OSM zoom level to get approx 4000 pixels across a bounding box.
    Assumes minx and maxx are in Web Mercator (EPSG:3857) meters.
    """
    # Total width of the Earth in Web Mercator meters
    world_width_m = 40075016.686
    bbox_width_m = maxx - minx

    # We want 16 tiles (16 * 256px = 4096px) across our bounding box.
    # Formula: (bbox_width / world_width) * (2^zoom) = 16
    ideal_zoom = math.log2(16 * (world_width_m / bbox_width_m))

    # Round to the nearest available integer zoom level (OSM max is usually 19)
    return min(19, round(ideal_zoom))


# --- Example usage in your loop ---
# zoom_level = calculate_4k_zoom(minx, maxx)
# cx.bounds2raster(minx, miny, maxx, maxy, "hex_zoom.tif", zoom=zoom_level)

# UK roughly: West, South, East, North (in Plate Carrée / EPSG:4326)
west, south, east, north = -8.0, 49.0, 2.0, 61.0

# Download OSM tiles and save directly to a georeferenced GeoTIFF
# ll=True tells contextily that our bounds are in Longitude/Latitude
bb = cx.bounds2raster(west, south, east, north,
                 "uk_osm_native.tif",
                 ll=True, use_cache=True,
                 source=cx.providers.OpenStreetMap.Mapnik)
print(bb)
