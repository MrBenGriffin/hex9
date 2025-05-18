"""
Part of the H9 project
This follows ex0010_plate_px.py (which loaded a png and displayed it).
This loads a Plate Carrée png, converts it to latitude/longitude, and displays it.
"""

from hhg9 import Registrar
from hhg9.domains import PlatePixel, SphericalGCD
from hhg9.projections import PlatePixelGCD
from support import Photo, Display

if __name__ == '__main__':
    """
    Load a photo, adopt into PlatePixel points, transform to Sphere via GCD (latitude/longitude).
    Then display as a Unit Sphere
    """
    reg = Registrar()  # Manage Domains & Projections
    # Domains - 2D image and GCD Spherical.
    p_plt = PlatePixel(reg)      # 2D Pixel Cartesian Domain
    g_sph = SphericalGCD(reg)    # GCD Spherical Domain (latitude/longitude)
    # Projections/Transforms
    pix_gcd = PlatePixelGCD(reg)   # Transform (Pixel Cartesian <=> GCD)

    # Support Classes
    d = Display()
    ps = Photo()

    ps.load('../preparatory/world1350x675.png')
    # use plate_carrée pc_px domain to adopt the ps.image (shape [675,1350,4], with RGBA)
    pc_px = p_plt.adopt(ps.img)
    # project it into GCD spherical
    sp_ll = reg.project(pc_px, [p_plt, g_sph])
    # Display it as an overlay on the globe.
    d.show_global(sp_ll, alpha=0.1)
    # Round trip to Pixels
    sp_rt = reg.project(sp_ll, [g_sph, p_plt])
    d.show_pts_2d(sp_rt)



