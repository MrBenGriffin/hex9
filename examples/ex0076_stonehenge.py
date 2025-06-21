"""
Part of the H9 project
This reads in the GCD for Stonehenge, then converts it into a set of nested half-hexagons,
each of which represents a single Stage of the H9 Journey.
This uses the images captured via ex0075
"""
import numpy as np
from matplotlib import image, pyplot as plt
from scipy.spatial import KDTree

from hhg9 import Registrar, H9Engine, Grid
from hhg9.domains import GeneralGCD, EllipsoidCartesian, OctahedralCartesian, OctahedralBarycentric, PlatePixel
from hhg9.projections import EllipsoidGCD, PlatePixelGCD, AKOctahedralEllipsoid
from hhg9.formats import OctahedralH9, DMS, DecimalDegrees, DecimalCartesian
from support import Util, Display

if __name__ == '__main__':
    """
        Compose a set of zooms into stonehenge..
    """
    reg = Registrar()  # Manage Domains & Projections
    g_sph = GeneralGCD(reg)           # GCD Spherical Domain (latitude/longitude)
    c_sph = EllipsoidCartesian(reg)     # Cartesian Spherical (xyz)
    c_oct = OctahedralCartesian(reg)    # Cartesian Octahedron (xyz)
    b_oct = OctahedralBarycentric(reg, c_oct)  # 2d Flat for addressing.
    p_plt = PlatePixel(reg)             # 2D Pixel Cartesian Domain
    # Projections/Transforms

    h9 = OctahedralH9()            # formatter.
    g_sph.register_format(DMS())
    g_sph.register_format(DecimalDegrees())
    c_sph.register_format(DecimalCartesian())
    b_oct.register_format(h9)

    # Projections/Transforms. Bary and Net are loaded by the domains.
    EllipsoidGCD(reg)           # g_sph <=> c_sph
    AKOctahedralEllipsoid(reg)  # c_sph <=> (c_oct <=> b_oct)
    ppg = PlatePixelGCD(reg)

    # Support Classes
    u = Util()
    h9 = H9Engine()
    g = Grid()
    d = Display()

    locs = u.json_load('../assets/locations.json')
    region = locs['NWA']
    spot = region['Stonehenge']
    ll0 = g_sph.adopt(np.array([spot]))
    sp0 = reg.project(ll0, [g_sph, c_sph])  # spherical cart
    oc0 = reg.project(sp0, [c_sph, c_oct])
    bc0 = reg.project(oc0, [c_oct, b_oct])

    # Stonehenge: 'NWΛ135724754627513335560466226533Λ1' [0.29387005 0.27854425]
    # [51.17886302 -1.82617712] -
    shb = bc0.coords[0]  # [0.29386167 0.27855944]
    cmp = tuple(bc0.components[0])
    sdo = bc0.domain.components[cmp]
    loc = sdo.tr

    plx = h9.enmesh(shb, loc, 12)
    flt = plx.reshape(-1, 2)
    pls = sdo.adopt(flt)
    ll1 = reg.project(pls, [b_oct, c_oct, c_sph, g_sph])
    brg = ll1.coords.reshape(12, 4, 2)

    h9a = f'{bc0:h9.f}'
    label, h9a = h9a[:3], h9a[3:]

    for i in range(12):
        img = image.imread(f'sh/SH_{i}.png', 'png')
        ll_poly = brg[i]
        lon_min = np.min(ll_poly[..., 1])
        lon_max = np.max(ll_poly[..., 1])
        lat_min = np.min(ll_poly[..., 0])
        lat_max = np.max(ll_poly[..., 0])
        extent = (lon_min, lon_max, lat_min, lat_max)
        pc_px = p_plt.adopt(img)
        ppg.set_dim(pc_px, extent)
        pc_sp = reg.project(pc_px, [p_plt, g_sph, c_sph])
        src = KDTree(pc_sp.coords)  # KDTree of plate_carrée projected onto unit sphere.

        poly = plx[i]
        w, h, pxl = g.qa_grid(poly, 1000)
        pts = sdo.adopt(pxl)
        oc1 = reg.project(pts, [b_oct, c_oct])
        sp1 = reg.project(oc1, [c_oct, c_sph])
        _, idx = src.query(sp1.coords, workers=-1)  # query KDTree and return indices of pc_sp
        pts.samples = pc_sp.samples[idx]
        xx, yy = pts.coords[:, 0], pts.coords[:, 1]
        cols = pts.samples
        fig = plt.figure(figsize=(w/72, h/72), dpi=72, frameon=False)
        fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
        ax = fig.add_subplot(111)
        y_lim = (yy.min(), yy.max())
        x_lim = (xx.min(), xx.max())
        ax.set(xlim=x_lim, ylim=y_lim)
        ax.scatter(xx, yy, marker='.', s=0.5, c=cols)
        ax.set_aspect('equal', adjustable='box')  # 'box'
        ax.text(x_lim[0], y_lim[0], label, fontsize=40)
        label, h9a = h9a[:2], h9a[2:]
        a = plt.gca()
        plt.box(False)
        a.axes.get_xaxis().set_visible(False)
        a.axes.get_yaxis().set_visible(False)
        plt.savefig(f'../output/sh{i}_{label}.png')
