"""
Part of the H9 project - Preparation 0003
Load a boundary (in GCD) and retrieve an image from OpenStreetMap.
Last Tested 07 August 2025 √
"""
import io
import math

import numpy as np
import cartopy.crs as ccrs  # Import the Cartopy Coordinate Reference Systems
import pyproj
from cartopy.io.img_tiles import OSM
from matplotlib import pyplot as plt, patches
from hhg9.algorithms.distance import wgs84_angular_ratio
from PIL import Image  # Use Pillow for clean image saving
# from shapely.geometry import Polygon
# import pyproj
# import math

# def flatten_providers(providers_object, parent_name='', repo=None):
#     """
#     Recursively flattens the nested contextily providers object.
#     Args:
#         providers_object: The contextily providers object or a sub-group.
#         parent_name (str): The name of the parent group for constructing full names.
#         repo of results.
#
#     Returns:
#         dict: A flat dictionary of {full_name: provider_object}.
#     """
#     repo = repo if repo is not None else dict()
#     for name, provider in providers_object.items():
#         # Construct the full, dot-separated name
#         full_name = f"{parent_name}.{name}" if parent_name else name
#         if 'url' in provider:
#             repo[full_name] = provider
#         else:
#             flatten_providers(provider, full_name, repo)
#     return repo
#
#
# def find_suitable_providers(required_zoom, required_bounds):
#     """
#     Finds contextily providers that support a given zoom level and extent.
#     Returns:
#         dict: A dictionary of suitable provider names and their objects.
#     """
#     providers = flatten_providers(cx.providers)
#     suitable_providers = {}
#     user_lon_min, user_lon_max, user_lat_min, user_lat_max = required_bounds
#     # Iterate through all available providers in the contextily library
#     for name, provider in providers.items():
#         # --- 1. Check Zoom Level ---
#         # Get provider's max zoom, default to 0 if not specified
#         if 'max_zoom' not in provider:
#             # print(f'{name} shows no max_zoom value')
#             continue
#         provider_max_zoom = provider['max_zoom']
#         if provider_max_zoom < required_zoom:
#             continue  # Skip if it doesn't support the required zoom
#
#         # --- 2. Check Geographic Bounds ---
#         provider_bounds = provider.get('bounds')
#         # If a provider has specific bounds, check for overlap
#         if provider_bounds:
#             [prov_lon_min, prov_lat_min], [prov_lon_max, prov_lat_max] = provider_bounds
#
#             # Check if the user's box is completely outside the provider's box
#             if (user_lon_max < prov_lon_min or
#                     user_lon_min > prov_lon_max or
#                     user_lat_max < prov_lat_min or
#                     user_lat_min > prov_lat_max):
#                 continue  # Skip if there is no overlap
#         if 'accessToken' in provider:
#             # print(f'{name} requires an accessToken')
#             continue
#         # If we passed both checks, this provider is suitable
#         suitable_providers[name] = provider
#     return suitable_providers


def calculate_dpi_aware_zoom(bounds, figsize_inches, dpi):
    """
    Calculates the zoom level needed to match a target DPI.
    Args:
        bounds (list): [lon_min, lon_max, lat_min, lat_max] in WGS84 degrees.
        figsize_inches (tuple): (width, height) of the matplotlib figure in inches.
        dpi (int): The target dots per inch (DPI) of the output image.

    Returns:
        int: The calculated integer zoom level.
    """
    # 1. Calculate required output dimensions in pixels
    fig_w_px = figsize_inches[0] * dpi
    fig_h_px = figsize_inches[1] * dpi

    # 2. Project geographic bounds to Web Mercator meters
    lon_min, lon_max, lat_min, lat_max = bounds
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    merc_min_x, merc_min_y = transformer.transform(lon_min, lat_min)
    merc_max_x, merc_max_y = transformer.transform(lon_max, lat_max)
    merc_w_meters = merc_max_x - merc_min_x
    merc_h_meters = merc_max_y - merc_min_y

    # 3. Determine the required resolution in meters/pixel
    resolution_w = merc_w_meters / fig_w_px
    resolution_h = merc_h_meters / fig_h_px
    required_resolution = min(resolution_w, resolution_h)
    TILE_SIZE = 256  # Standard tile size in pixels
    EARTH_RADIUS_M = 6378137.0
    try:
        zoom_float = math.log2((2 * math.pi * EARTH_RADIUS_M) / (required_resolution * TILE_SIZE))
    except (ValueError, ZeroDivisionError):
        zoom_float = 0  # Handle cases with zero size
    return int(np.ceil(zoom_float))


if __name__ == '__main__':
    # Cartopy appears to be better.
    imagery = OSM()
    file = 'jpn'
    bounds_file = f'src/{file}_lat_lon_bounds.npy'
    lat_lon = np.load(bounds_file).reshape(2, 2)
    lon_lat = lat_lon[::-1].reshape(-1)
    ratio = wgs84_angular_ratio(lon_lat)  # (lon_min, lon_max, lat_min, lat_max)
    lon_min, lon_max, lat_min, lat_max = lon_lat
    dpi = 100
    img_w_pix = 6000
    img_h_pix = int(img_w_pix * ratio)
    fig_w_in, fig_h_in = img_w_pix / dpi, img_h_pix / dpi
    print(f'With lon/lat bounds ({lon_lat}), found ratio {ratio}. Generating image {img_w_pix}x{img_h_pix}px')
    zoom = calculate_dpi_aware_zoom(lon_lat, (fig_w_in, fig_h_in), dpi)
    fig = plt.figure(figsize=(img_w_pix / dpi, img_h_pix / dpi), dpi=dpi, frameon=False)
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(lon_lat, ccrs.PlateCarree())
    ax.axis('off')
    ax.add_image(imagery, zoom - 2, regrid_shape=(img_w_pix, img_h_pix), interpolation='spline36')
    with io.BytesIO() as buff:
        fig.savefig(buff, format='raw')
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    im = data.reshape((img_h_pix, img_w_pix, -1))
    plt.close(fig)
    pil_img = Image.fromarray(im)
    pil_img.save(f'src/{file}_gcd.png')
