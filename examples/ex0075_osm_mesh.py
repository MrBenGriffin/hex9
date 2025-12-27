# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Compose a set of zooms into stonehenge vi cartopy and OSM.
This grabs rectangle boundaries from which we sample the octahedral.
The actual octahedral render series is found in 0076.

Last Tested
26 December 2025 0.1.0a4 (passed)
16 December 2025 0.1.0a3 (passed)
25 November 2025 (passed)
"""
import json
import numpy as np
import cartopy.crs as ccrs
from cartopy.io.img_tiles import OSM
from matplotlib import pyplot as plt
from hhg9 import Registrar, Points
from hhg9.formats import OctahedralH9
from hhg9.h9.polygon import enmesh
from hhg9.h9.region import xy_regions


def json_load(path):
    """Load json file"""
    with (open(path) as infile):
        obj = json.load(infile)
        infile.close()
    return obj


if __name__ == '__main__':
    reg = Registrar()  # Manage Domains & Projections
    b_oct = reg.domain('b_oct')
    g_gcd = reg.domain('g_gcd')
    h9 = OctahedralH9(reg)            # formatter.
    b_oct.register_format(h9)

    # Support Classes
    imagery = OSM()
    layers = 13

    locs = json_load('../assets/locations.json')
    region = locs['NWA']
    spot = region['Stonehenge']
    ll0 = g_gcd.adopt(np.array([spot]))
    bc0 = reg.project(ll0, [g_gcd, b_oct])  # spherical cart
    co, mo = bc0.cm()
    comp = bc0.invert_octant_ids(co)
    uri = xy_regions(bc0.coords, mo, layers)
    poly, msh = enmesh(bc0, layers)
    bnd = Points(poly.reshape([-1, 2]), b_oct, components=comp[0])
    gel = reg.project(bnd, [b_oct, g_gcd])
    gpy = gel.coords.reshape([-1, 4, 2])
    detail = [4, 5, 8, 10, 11, 13, 14, 15, 16, 17, 18, 18, 18, 18, 18, 18]
    BASE_SIZE_METERS = 10000000.0  # 10000 km (example value)
    IMG_PIXELS = 3000
    extents = []

    for i in range(layers+1):  # We need the layer_depth
        hh_vertices = gpy[i]
        layer_depth = i
        current_size_meters = BASE_SIZE_METERS / (3 ** layer_depth)
        center_lat, center_lon = np.mean(hh_vertices, axis=0)
        meters_per_deg_lat = 111132.954
        meters_per_deg_lon = 111320.0 * np.cos(np.radians(center_lat))
        extent_height_deg = current_size_meters / meters_per_deg_lat
        extent_width_deg = current_size_meters / meters_per_deg_lon
        final_extent = [
            center_lon - (extent_width_deg / 2),  # lon_min
            center_lon + (extent_width_deg / 2),  # lon_max
            center_lat - (extent_height_deg / 2),  # lat_min
            center_lat + (extent_height_deg / 2)  # lat_max
        ]
        extents.append(final_extent)
        w, h, dpi = IMG_PIXELS / 100, IMG_PIXELS / 100, 100
        fig = plt.figure(figsize=(w, h), dpi=dpi, frameon=False)
        fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.set_extent(final_extent, ccrs.PlateCarree())
        print(f'Layer {layer_depth}, detail:{detail[layer_depth]}')
        ax.add_image(imagery, detail[layer_depth], regrid_shape=(2*IMG_PIXELS, 2*IMG_PIXELS))
        fig.savefig(f'output/ex0075_{i:02}.png',
                    dpi=dpi,
                    format='png', bbox_inches='tight',
                    pad_inches=0, transparent=True)
        plt.close(fig)
        print(f'file saved at output/ex0075_{i:02}.png')
    np.save('output/ex0075_extents.npy', np.array([extents]))
    print(f'extents file saved at output/ex0075_extents.npy')
