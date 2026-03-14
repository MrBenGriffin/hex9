# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Compose a plate carree of UK vi cartopy and OSM.
Last Tested

13 Mar 2026 0.1.1a1 (passed - complete rewrite/repurpose)
02 Mar 2026 0.1.1a1 (passed)
26 Dec 2025 0.1.0a4 (passed)
16 Dec 2025 0.1.0a3 (passed)
25 Nov 2025 (passed)
"""
import json
import os
import numpy as np
import cartopy.crs as ccrs
from cartopy.io.img_tiles import OSM
from matplotlib import pyplot as plt
from hhg9 import Registrar


def json_load(path):
    """Load json file"""
    with (open(path) as infile):
        obj = json.load(infile)
        infile.close()
    return obj


if __name__ == '__main__':
    imagery = OSM()

    reg = Registrar()  # Manage Domains & Projections
    g_gcd = reg.domain('g_gcd')
    zone = [-11, 2.5, 49.5, 61]  # Rough UK boundary box lon_min/lon_max/lat_min/lat_max
    lon_min, lon_max, lat_min, lat_max = zone
    pw, ph = lon_max-lon_min, lat_max-lat_min
    size, dpi, zoom = 8, 100, 8   # zoom 8 → ~2400×2100 px of tile data for UK
    img_w = int(pw * size * dpi)   # output pixel width  — matches regrid_shape
    img_h = int(ph * size * dpi)   # output pixel height — matches regrid_shape

    out_img = 'output/ex0075.png'
    out_ext = 'output/ex0075_extents.npy'
    if os.path.exists(out_img) and os.path.exists(out_ext):
        print(f'Cached: {out_img} — delete to regenerate')
        raise SystemExit(0)

    fig = plt.figure(figsize=(img_w/dpi, img_h/dpi), dpi=dpi, frameon=False)
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
    ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.PlateCarree())
    ax.set_extent(zone, ccrs.PlateCarree())
    ax.add_image(imagery, zoom, regrid_shape=(img_w, img_h))
    fig.savefig(f'output/ex0075.png',
                dpi=dpi,
                format='png', bbox_inches='tight',
                pad_inches=0, transparent=True)
    plt.close(fig)
    print(f'file saved at output/ex0075.png')
    np.save('output/ex0075_extents.npy', np.array(zone))
    print(f'extents file saved at output/ex0075_extents.npy')
