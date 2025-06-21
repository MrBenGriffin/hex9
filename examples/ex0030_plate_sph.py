"""
Part of the H9 project
This follows ex0020_plate_glb.py (which loaded the png and displayed it on a globe).
This loads a Plate Carrée png image,
converts it to cartesian XYZ sphere, via latitude/longitude, and displays it
Last Tested 21 June 2025 √
"""
from matplotlib import image, pyplot as plt
from hhg9 import Registrar
from hhg9.domains import PlatePixel, GeneralGCD, EllipsoidCartesian
from hhg9.projections import EllipsoidGCD, PlatePixelGCD

from support import Display


if __name__ == '__main__':
    """
    Load a photo, adopt into PlatePixel points, transform to XYZ via GCD.
    Then display it as the unit sphere..
    """
    reg = Registrar()  # Manage Domains & Projections

    p_plt = PlatePixel(reg)             # 2D Pixel Cartesian Domain
    g_gcd = GeneralGCD(reg)             # GCD Domain (latitude/longitude)
    c_ell = EllipsoidCartesian(reg)     # Cartesian Geodesic (xyz)
    # Projections/Transforms

    PlatePixelGCD(reg)   # (g_gcd p_plt) Project (Pixel Cartesian <=> GCD)
    EllipsoidGCD(reg)    # (g_gcd c_ell) Project (GCD <=> Geodesic Cartesian)

    # Support Classes
    d = Display()

    img = image.imread(f'src/world1350x675.png', 'png')
    pc_px = p_plt.adopt(img)     # ps.img shape is [1350x675x4]
    # project it into GCD Geodesic
    el_xyz = reg.project(pc_px, [p_plt, g_gcd, c_ell])
    d.show_pts_3d(el_xyz)
    # Roundtrip back to Pixels
    sp_pl = reg.project(el_xyz, [c_ell, g_gcd, p_plt])
    rmg = p_plt.image(sp_pl)     # convert points back to an [1280,2560,3] image.
    fig = plt.figure(figsize=(18, 9), dpi=100, frameon=False)
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
    plt.imshow(rmg)
    plt.show()


