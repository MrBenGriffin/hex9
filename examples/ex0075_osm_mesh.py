"""
Part of the H9 project
Compose a set of zooms into stonehenge vi cartopy and OSM.
This grabs rectangle boundaries from which we sample the octahedral.
The actual octahedral render series is found in 0076.
Last Tested 21 June 2025 √
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

    shb = bc0.coords[0]  # [0.29386167 0.27855944]
    code = f'{bc0:h9}'
    cmp = tuple(bc0.components[0])
    sdo = bc0.domain.components[cmp]
    loc = sdo.tr

    plx = h9.enmesh(shb, loc, acc)  # This is the set of half-hexagons.
    plp = plx.reshape([-1, 2])
    bnd = sdo.adopt(plp)
    gel = reg.project(bnd, [b_oct, c_oct, c_ell, g_gen])
    mmx = []
    gpy = gel.coords.reshape([-1, 4, 2])
    for hh in gpy:
        # longitude is x, latitude is y.
        y0, y1 = np.min(hh[..., 0]), np.max(hh[..., 0])
        x0, x1 = np.min(hh[..., 1]), np.max(hh[..., 1])
        mmx.append([x0, x1, y0, y1])
    plates = np.array(mmx)

    #         √  7  2   3  4    5  6   7   8   9   10  11  12
    scales = [5, 6, 8, 10, 12, 14, 16, 18, 18, 18, 18, 18, 18, 18, 18, 18]
    for i, extent in enumerate(plates):
        lon_min, lon_max, lat_min, lat_max = extent
        w = lon_max - lon_min
        h = lat_max - lat_min
        if w > h:
            hgt = 2000
            wid = np.uint32((w / h) * hgt)
        else:
            wid = 2000  # np.uint32(scale / w)
            hgt = np.uint32((h / w) * wid)
        w, h, dpi = wid / 72, hgt / 72, 72
        fig = plt.figure(figsize=(w, h), dpi=dpi, frameon=False)
        fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
        ax = fig.add_subplot(1, 1, 1, projection=imagery.crs)
        ax.set_extent(extent, ccrs.PlateCarree())
        ax.add_image(imagery, scales[i])  # 5 = zoom level 18 = ring of stonehenge.
        fig.savefig(f'sh/SH_{i}.png', format='png', bbox_inches='tight', pad_inches=0)
