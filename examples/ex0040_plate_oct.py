"""
Part of the H9 project
This follows ex0030_plate_glb.py (which loaded the png and displayed it as a unit sphere).
This loads a Plate Carrée png image,
converts it to XYZ Octagon, via latitude/longitude and sphere and displays it.
It's SLOW because it has to manage an inverse of the AK projection.
"""
import numpy as np

from hhg9 import Registrar
from hhg9.domains import PlatePixel, SphericalGCD, SphericalCartesian, OctahedralCartesian
from hhg9.projections import PlatePixelGCD, CartesianGCD, AKOctahedralSpherical
from support import Photo, Display


if __name__ == '__main__':
    """
    Load a photo, adopt into PlatePixel points, transform to XYZ via GCD.
    Then display it as the unit sphere..
    """
    reg = Registrar()  # Manage Domains & Projections
    # Domains - 2D image and GCD Spherical.
    p_plt = PlatePixel(reg)             # 2D Pixel Cartesian Domain
    g_sph = SphericalGCD(reg)           # GCD Spherical Domain (latitude/longitude)
    c_sph = SphericalCartesian(reg)     # Cartesian Spherical (xyz)
    c_oct = OctahedralCartesian(reg)    # Cartesian Octahedron (xyz)

    # Projections/Transforms
    PlatePixelGCD(reg)   # Transform (Pixel Cartesian <=> GCD)
    CartesianGCD(reg)
    # Initialise (spherical to octahedral) warm-up takes a couple of seconds.
    AKOctahedralSpherical(reg)

    # Support Classes
    d = Display()
    ps = Photo()

    # ps.load('../preparatory/world1350x675.png')
    ps.load('../preparatory/world360x180.png')
    # use plate_carrée pc_px domain to adopt the ps.image (shape [675,1350,4], with RGBA)
    pc_pix = p_plt.adopt(ps.img)
    # project it into Cartesian spherical via GCD.
    sp_gcd = reg.project(pc_pix, [p_plt, g_sph])
    sp_pix = reg.project(sp_gcd, [g_sph, p_plt])
    pc_dif = np.unique(abs(pc_pix - sp_pix), axis=0)


    sp_xyz = reg.project(sp_gcd, [g_sph, c_sph])
    oc_xyz = reg.project(sp_xyz, [c_sph, c_oct])
    # oc_xyz = reg.project(pc_px, [p_plt, g_sph, c_sph, c_oct])
    # Display it as a unit Octagon.
    d.show_pts_3d(oc_xyz)
