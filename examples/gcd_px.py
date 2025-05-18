"""
Part of the H9 project
"""
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root
from hhg9 import (
    Registrar, Points, Projection, SphericalGCD,
    SphericalCartesian, OctahedralCartesian, OctahedralBarycentric,
    OctahedralNet, DMS, DecimalDegrees, PlateGCD, PlatePixel,
    AKOctahedralSpherical, CartesianGCD, PlatePixelGCD
)
from support import Util, Photo, Display

if __name__ == '__main__':
    """
    Sample values in hhg are possibly going to be a standard storage of np.uint64
    So I need to use a conversion for Pixels such that RGBA => unit64
    
    """
    u = Util()
    d = Display()
    ps = Photo()   # source image
    ps.load('../preparatory/world1350x675.png')
    reg = Registrar()  # Manage coordinate sets & projections
    g_sph = SphericalGCD(reg)  # 3D Spherical (2-value Polar)
    c_sph = SphericalCartesian(reg)  # 3D Spherical (3-value Cartesian)
    c_oct = OctahedralCartesian(reg)  # 3D Octahedral (surface) coordinate set
    b_oct = OctahedralBarycentric(reg, c_oct)  # 2d Flat coordinate set.
    n_oct = OctahedralNet(reg, c_oct, b_oct)  # 2d Flat coordinate set.
    p_plt = PlatePixel(reg)
    g_plt = PlateGCD(reg)
    PlatePixelGCD(reg)          # p_plt<=>g_plt
    CartesianGCD(reg)           # g_sph<=>c_sph 3D/Polar
    AKOctahedralSpherical(reg, True)  # c_oct<=>c_sph (initialising jac fn)
    reg.register_projection('chain', [g_sph, c_sph, c_oct])  # latlon to octahedral via spherical_euclidean
    reg.register_projection('chain', [c_oct, b_oct, n_oct])  # octahedral to net via barycentric.
    reg.register_projection('chain', [g_sph, c_oct, n_oct])  # latlon to octahedral_net via octahedral
    plate, w, h, t = p_plt.adopt(ps.img)
    d.show_pts_2d(plate)
    img = p_plt.image(plate, w, h, t)

    g_pts = reg.project(plate, [p_plt, g_plt])
    d.show_global(g_pts)
    s_pts = g_sph.adopt(g_pts)  # not 100% necessary.
    c_pts = reg.project(s_pts, [g_sph, c_sph])   # These are now Cartesian
    d.show_pts_3d(c_pts)
    o_pts = reg.project(c_pts, [c_sph, c_oct])   # Cartesian Octahedral (this is slow).
    d.show_pts_3d(o_pts)
    n_pts = reg.project(c_pts, [c_oct, n_oct])   # Octahedral Network.
    p_pts = reg.project(g_pts, [g_plt, p_plt])
    ps.img = p_pts
    ps.save('the_map')
    done = True



