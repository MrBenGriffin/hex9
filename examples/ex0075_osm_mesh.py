"""
Part of the H9 project
Compose a set of zooms into stonehenge vi cartopy and OSM.
This grabs rectangle boundaries from which we sample the octahedral.
The actual octahedral render series is found in 0076.
Last Tested 10 August 2025 √ (rewrite)
"""

import numpy as np
import cartopy.crs as ccrs
from cartopy.io.img_tiles import OSM

from matplotlib import pyplot as plt
from hhg9 import Registrar, H9Engine, Grid
from hhg9.domains import GeneralGCD, EllipsoidCartesian, OctahedralCartesian, OctahedralBarycentric, PlatePixel
from hhg9.projections import EllipsoidGCD, PlatePixelGCD, AKOctahedralEllipsoid
from hhg9.formats import OctahedralH9, DMS, DecimalDegrees, DecimalCartesian
from support import Util, Display
# from matplotlib.collections import PolyCollection


if __name__ == '__main__':
    reg = Registrar()  # Manage Domains & Projections
    g_gen = GeneralGCD(reg)             # GCD Spherical Domain (latitude/longitude)
    c_ell = EllipsoidCartesian(reg)     # Cartesian Geodesic (xyz)
    c_oct = OctahedralCartesian(reg)    # Cartesian Octahedron (xyz)
    b_oct = OctahedralBarycentric(reg, c_oct)  # 2d Flat for addressing.
    p_plt = PlatePixel(reg)             # 2D Pixel Cartesian Domain

    h9 = OctahedralH9()            # formatter.
    g_gen.register_format(DMS())
    g_gen.register_format(DecimalDegrees())
    c_ell.register_format(DecimalCartesian())
    b_oct.register_format(h9)

    # Projections/Transforms. Bary and Net are loaded by the domains.
    EllipsoidGCD(reg)           # [g_gen, c_ell]
    ak = AKOctahedralEllipsoid(reg)  # [c_ell, c_oct]
    ppg = PlatePixelGCD(reg)

    # Support Classes
    u = Util()
    h9 = H9Engine()
    g = Grid()
    d = Display()
    imagery = OSM()

    acc = ak.set_accuracy(0.01)
    locs = u.json_load('../assets/locations.json')
    region = locs['NWA']
    spot = region['Stonehenge']
    ll0 = g_gen.adopt(np.array([spot]))
    sp0 = reg.project(ll0, [g_gen, c_ell])  # spherical cart
    oc0 = reg.project(sp0, [c_ell, c_oct])
    bc0 = reg.project(oc0, [c_oct, b_oct])

    cmp = tuple(bc0.components[0])
    sdo = bc0.domain.components[cmp]
    co, mo = bc0.cm()
    uri = h9.ugc_regions(bc0.coords, mo, acc)
    plx = h9.enmesh(uri)
    plv = np.array(list(plx.values()))
    plp = plv.reshape([-1, 2])
    bnd = sdo.adopt(plp)
    gel = reg.project(bnd, [b_oct, c_oct, c_ell, g_gen])
    gpy = gel.coords.reshape([-1, 5, 2])
    detail = [4, 6, 8, 10, 12, 14, 16, 18, 18, 18, 18, 18, 18, 18, 18, 18]
    BASE_SIZE_METERS = 10000000.0  # 10000 km (example value)
    IMG_PIXELS = 3000
    extents = []
    for i, hh_vertices in enumerate(gpy):  # You need the layer_depth
        layer_depth = i
        # --- 3. Calculate the correct geographic size for the CURRENT layer ---
        # Each layer is 3x smaller (more zoomed in) than the last.
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

        # --- 7. Create the plot ---
        w, h, dpi = IMG_PIXELS / 100, IMG_PIXELS / 100, 100
        fig = plt.figure(figsize=(w, h), dpi=dpi, frameon=False)
        fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.set_extent(final_extent, ccrs.PlateCarree())
        ax.add_image(imagery, detail[layer_depth], regrid_shape=(2*IMG_PIXELS, 2*IMG_PIXELS))

        # Put on a border for approximated projection testing.
        # # Close the polygon by appending the first vertex to the end
        # closed_poly_lon_lat = np.vstack([hh_vertices[:, ::-1], hh_vertices[0, ::-1]])
        #
        # # Plot the outline
        # ax.plot(closed_poly_lon_lat[:, 0], closed_poly_lon_lat[:, 1],
        #         color='k', linewidth=6.0, transform=ccrs.PlateCarree())

        fig.savefig(f'sh/SH_{i}.png',
                    dpi=dpi,
                    format='png', bbox_inches='tight',
                    pad_inches=0, transparent=True)
        plt.close(fig)
        if layer_depth == 13:  # not much to see beyond level 13!
            break
    np.save('sh/extents.npy', np.array(extents))

