"""
Part of the H9 project
This follows ex0020_plate_glb.py (which loaded the png and displayed it on a globe).
This loads a Plate Carrée png image,
converts it to cartesian XYZ sphere, via latitude/longitude, and displays it
"""
from hhg9 import Registrar
from hhg9.domains import PlatePixel, SphericalGCD, SphericalCartesian
from hhg9.projections import PlatePixelGCD, CartesianGCD
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
    # Projections/Transforms
    PlatePixelGCD(reg)   # Transform (Pixel Cartesian <=> GCD)
    CartesianGCD(reg)

    # Support Classes
    d = Display()
    ps = Photo()

    ps.load('../preparatory/world1350x675.png')
    # use plate_carrée pc_px domain to adopt the ps.image (shape [675,1350,4], with RGBA)
    pc_px = p_plt.adopt(ps.img)
    # project it into Cartesian spherical via GCD.
    sp_xyz = reg.project(pc_px, [p_plt, g_sph, c_sph])
    # Display it as a unit sphere.
    d.show_pts_3d(sp_xyz)
