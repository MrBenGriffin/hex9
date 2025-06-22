"""
Part of the H9 project
This loads a Plate Carrée png image,
converts it to Octagon NET, via latitude/longitude and sphere and displays it.
It's SLOW because it has to manage an inverse of the AK projection.
Last Tested 21 June 2025 √
"""
from pathlib import Path

import numpy as np
from matplotlib import image, pyplot as plt
from hhg9 import Registrar
from hhg9.domains import PlatePixel, GeneralGCD, EllipsoidCartesian, OctahedralCartesian, OctahedralBarycentric, \
    OctahedralNet
from hhg9.projections import PlatePixelGCD, AKOctahedralEllipsoid
from support import Photo, Display, Util


def load_cache(pc_map: str = 'world1350x675'):
    """
    Load cached octahedral and colours from original.
    cf. 0041 / 0042
    """
    cache = Path(f'src/{pc_map}.npy')
    if not cache.exists():
        raise ValueError('Far better to load a cache. See ex_0041')
    img = image.imread(f'src/{pc_map}.png', 'png')
    pc_pix = p_plt.adopt(img)
    oc = np.load(cache)
    _oc_px = c_oct.adopt(oc)
    _oc_px.samples = pc_pix.samples
    return _oc_px


if __name__ == '__main__':
    """
    Load a photo, adopt into PlatePixel points, transform to NetValue via GCD.
    Then display it as the Net
    """
    reg = Registrar()  # Manage Domains & Projections
    # Domains - 2D image and GCD Spherical.
    p_plt = PlatePixel(reg)             # 2D Pixel Cartesian Domain
    g_sph = GeneralGCD(reg)           # GCD Spherical Domain (latitude/longitude)
    c_sph = EllipsoidCartesian(reg)     # Cartesian Spherical (xyz)
    c_oct = OctahedralCartesian(reg)    # Cartesian Octahedron (xyz)
    b_oct = OctahedralBarycentric(reg, c_oct)  # 2d Flat for addressing.
    n_oct = OctahedralNet(reg, c_oct, b_oct)   # 2d Flat for display.

    # Projections/Transforms. Bary and Net are loaded by the domains.
    PlatePixelGCD(reg)   # Transform (Pixel Cartesian <=> GCD)
    GeneralGCD(reg)
    AKOctahedralEllipsoid(reg)

    # Support Classes
    d = Display()
    ps = Photo()
    u = Util()

    oc_px = load_cache('world1350x675')  # 'world1350x675' on octahedral, with samples.
    on_px = reg.project(oc_px.copy(), [c_oct, b_oct, n_oct])
    d.show_pts_2d(on_px, ratio=1.34724743)  # net.
    img = p_plt.image(on_px, (1350, 675))   # convert points back to an image.
    plt.imshow(img, origin='upper')
    plt.show()


