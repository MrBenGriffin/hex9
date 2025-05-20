"""
Part of the H9 project
This follows ex0030_plate_glb.py (which loaded the png and displayed it as a unit sphere).
This loads a Plate Carrée png image,
converts it to Octagon NET, via latitude/longitude and sphere and displays it.
It's SLOW because it has to manage an inverse of the AK projection.
"""
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from hhg9 import Registrar
from hhg9.domains import PlatePixel, SphericalGCD, SphericalCartesian, OctahedralCartesian, OctahedralBarycentric, \
    OctahedralNet
from hhg9.projections import PlatePixelGCD, CartesianGCD, AKOctahedralSpherical
from support import Photo, Display, Util


def load_cache(pc_map: str = 'world1350x675'):
    """
    Load cached octahedral and colours from original.
    cf. 0040 / 0041
    """
    cache = Path(f'{pc_map}.npy')
    if not cache.exists():
        raise ValueError('Far better to load a cache. See ex_0041')
    ps.load(f'../preparatory/{pc_map}.png')
    pc_pix = p_plt.adopt(ps.img)
    oc = np.load(cache)
    oc_px = c_oct.adopt(oc)
    oc_px.samples = pc_pix.samples
    return oc_px


if __name__ == '__main__':
    """
    Load a photo, adopt into PlatePixel points, transform to NetValue via GCD.
    Then display it as the Net
    """
    reg = Registrar()  # Manage Domains & Projections
    # Domains - 2D image and GCD Spherical.
    p_plt = PlatePixel(reg)             # 2D Pixel Cartesian Domain
    g_sph = SphericalGCD(reg)           # GCD Spherical Domain (latitude/longitude)
    c_sph = SphericalCartesian(reg)     # Cartesian Spherical (xyz)
    c_oct = OctahedralCartesian(reg)    # Cartesian Octahedron (xyz)
    b_oct = OctahedralBarycentric(reg, c_oct)  # 2d Flat for addressing.
    n_oct = OctahedralNet(reg, c_oct, b_oct)   # 2d Flat for display.

    # Projections/Transforms. Bary and Net are loaded by the domains.
    PlatePixelGCD(reg)   # Transform (Pixel Cartesian <=> GCD)
    CartesianGCD(reg)
    AKOctahedralSpherical(reg)

    # Support Classes
    d = Display()
    ps = Photo()
    u = Util()

    oc_px = load_cache('world1350x675')  #'world1350x675'
    ob_px = reg.project(oc_px, [c_oct, b_oct])
    c1_px = reg.project(ob_px, [b_oct, c_oct])

    on_px = reg.project(oc_px, [c_oct, b_oct, n_oct])
    c2_px = reg.project(on_px, [n_oct, b_oct, c_oct])

    d.show_pts_3d(c2_px)  # after reverse
    d.show_pts_2d(on_px)  # net.

    # w, h = int(675*n_oct.ratio()), 675
    # img = p_plt.image(on_px, w, h)   # convert points back to an [675,1350,4] image.
    # plt.imshow(img, origin='upper')
    # plt.show()


