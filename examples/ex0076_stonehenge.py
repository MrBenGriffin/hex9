"""
Part of the H9 project
This reads in the GCD for Stonehenge, then converts it into a set of nested half-hexagons,
each of which represents a single Stage of the H9 Journey.
Last Tested 10 August 2025 √ (rewrite)
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

    # metres = [100.]  # 0.01 = cm
    locs = u.json_load('../assets/locations.json')
    # np.set_printoptions(precision=22)
    region = locs['NWA']
    spot = region['Stonehenge']
    ll0 = g_gen.adopt(np.array([spot]))
    sp0 = reg.project(ll0, [g_gen, c_ell])  # spherical cart
    oc0 = reg.project(sp0, [c_ell, c_oct])
    bc0 = reg.project(oc0, [c_oct, b_oct])
    cmp = tuple(bc0.components[0])
    sdo = bc0.domain.components[cmp]
    co, mo = bc0.cm()
    ur0 = h9.ugc_regions(bc0.coords, mo, 18)

    pld = h9.enmesh(ur0)
    plx = np.array(list(pld.values()))

    extents = np.load('sh/extents.npy')
    for i, extent in enumerate(extents):
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
        poly = plx[i][:4]  # don't need the fifth point here.
        # ww, hh, pxl, msk, (grid_org, grid_scale) = grid.qa_grid(grid_shape, img_w)
        w, h, pxl, msk, (grid_org, grid_scale) = g.qa_grid(poly, 2000)
        pts = sdo.adopt(pxl[msk])
        sp1 = reg.project(pts.copy(), [b_oct, c_oct, c_ell, g_gen])
        _, idx = src.query(sp1.coords, workers=-1)  # query KDTree and return indices of pc_sp
        pts.samples = pc_sp.samples[idx]
        xx, yy = pts.coords[:, 0], pts.coords[:, 1]
        cols = pts.samples
        # TODO.md: set alpha to 0 for msk...
        fig = plt.figure(figsize=(w/72, h/72), dpi=72, frameon=False)
        fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
        ax = fig.add_subplot(111)
        y_lim = (yy.min(), yy.max())
        x_lim = (xx.min(), xx.max())
        ax.set(xlim=x_lim, ylim=y_lim)
        ax.scatter(xx, yy, marker='.', s=0.5, c=cols)
        ax.set_aspect('equal', adjustable='box')  # 'box'
        a = plt.gca()
        plt.box(False)
        a.axes.get_xaxis().set_visible(False)
        a.axes.get_yaxis().set_visible(False)
        plt.savefig(f'../output/sh{i}.png')
