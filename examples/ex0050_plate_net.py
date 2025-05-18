"""
Part of the H9 project
This follows ex0030_plate_glb.py (which loaded the png and displayed it as a unit sphere).
This loads a Plate Carrée png image,
converts it to Octagon NET, via latitude/longitude and sphere and displays it.
It's SLOW because it has to manage an inverse of the AK projection.
"""
import numpy as np
from matplotlib import pyplot as plt

from hhg9 import Registrar
from hhg9.domains import PlatePixel, SphericalGCD, SphericalCartesian, OctahedralCartesian, OctahedralBarycentric, \
    OctahedralNet
from hhg9.projections import PlatePixelGCD, CartesianGCD, AKOctahedralSpherical
from support import Photo, Display, Util


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

    ps.load('../preparatory/world360x180.png')
    # use plate_carrée pc_px domain to adopt the ps.image (shape [675,1350,4], with RGBA)
    pc_px = p_plt.adopt(ps.img)
    oc_px = reg.project(pc_px, [p_plt, g_sph, c_sph, c_oct, b_oct, n_oct])
    d.show_pts_2d(oc_px)
    w, h = int(300*n_oct.ratio()), 300
    img = p_plt.image(np.asarray(oc_px), w, h)   # convert points back to an [675,1350,4] image.
    plt.imshow(img, origin='upper')
    plt.show()


