"""
Part of the H9 project
This reads in the GCD for Stonehenge, then converts it into a set of nested half-hexagons,
each of which represents a single Stage of the H9 Journey.
This uses the images captured via ex0075
There seems to be a bug in projection image 0 but otherwise looks ok.
"""
import os

import numpy as np
from matplotlib import image, pyplot as plt
from scipy.spatial import KDTree

from hhg9 import Registrar, H9Engine, Grid
from hhg9.domains import GeneralGCD, EllipsoidCartesian, OctahedralCartesian, OctahedralBarycentric, PlatePixel
from hhg9.projections import EllipsoidGCD, PlatePixelGCD, AKOctahedralEllipsoid
from hhg9.formats import OctahedralH9, DecimalDegrees, DecimalCartesian
from support import Util, Display

if __name__ == '__main__':
    """
        Compose a set of zooms into stonehenge..
    """
    reg = Registrar()  # Manage Domains & Projections
    g_gen = GeneralGCD(reg)             # GCD Spherical Domain (latitude/longitude)
    c_ell = EllipsoidCartesian(reg)     # Cartesian Geodesic (xyz)
    c_oct = OctahedralCartesian(reg)    # Cartesian Octahedron (xyz)
    b_oct = OctahedralBarycentric(reg, c_oct)  # 2d Flat for addressing.
    p_plt = PlatePixel(reg)             # 2D Pixel Cartesian Domain

    h9 = OctahedralH9()            # formatter.
    g_gen.register_format(DecimalDegrees())
    c_ell.register_format(DecimalCartesian())
    b_oct.register_format(h9)

    # Projections/Transforms. Bary and Net are loaded by the domains.
    eg = EllipsoidGCD(reg)           # [g_gen, c_ell]
    ak = AKOctahedralEllipsoid(reg)  # [c_ell, c_oct]
    pg = PlatePixelGCD(reg)

    # Support Classes
    u = Util()
    h9 = H9Engine()
    g = Grid()
    d = Display()

    # acc = ak.set_accuracy(0.000000000001)
    # acc = ak.set_accuracy(0.000000001)  # nanometre.
    acc = ak.set_accuracy(0.01)  # nanometre.

    locs = u.json_load('../assets/locations.json')
    region = locs['NWA']
    spot = region['Stonehenge']
    ll0 = g_gen.adopt(np.array([spot]))
    bc0 = reg.project(ll0, [g_gen, c_ell, c_oct, b_oct])  # spherical cart

    shb = bc0.coords[0]  # [0.29386167 0.27855944]
    code = f'{bc0:h9.19}'   # 'NWΛ013572475462751333Λ2'
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

    h9a = f'{bc0:h9.f}'
    label, h9a = h9a[:3], h9a[3:]

    for i, extent in enumerate(plates):
        lon_min, lon_max, lat_min, lat_max = extent
        w = lon_max - lon_min
        h = lat_max - lat_min
        img_file = f'sh/SH_{i}.png'
        if os.path.isfile(img_file):
            img = image.imread(img_file, 'png')
        else:
            break
        pc_px = p_plt.adopt(img)
        pg.set_dim(pc_px, extent)
        pc_sp = reg.project(pc_px, [p_plt, g_gen])
        src = KDTree(pc_sp.coords)  # KDTree of plate_carrée projected onto unit sphere.

        poly = plx[i]
        w, h, pxl = g.qa_grid(poly, 1000)
        pts = sdo.adopt(pxl)
        sp1 = reg.project(pts.copy(), [b_oct, c_oct, c_ell, g_gen])
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
