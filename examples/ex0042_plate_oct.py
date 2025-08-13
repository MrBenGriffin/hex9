"""
Part of the H9 project - Uses AK projection, so H9 indirectly
This follows ex0030_plate_glb.py (which loaded the png and displayed it as a unit ellipsoid).
This loads a Plate Carrée png image, converts it to XYZ Octahedron, via latitude/longitude and sphere and displays it.
As the inverse of the AK projection is slow we use the cache from ex0041_cache.py
Last Tested 13 August 2025 √
"""
from pathlib import Path
import numpy as np
from matplotlib import image, pyplot as plt
from hhg9 import Registrar
from hhg9.domains import PlatePixel, GeneralGCD, EllipsoidCartesian, OctahedralCartesian, OctahedralBarycentric
from hhg9.projections import PlatePixelGCD, EllipsoidGCD, AKOctahedralEllipsoid
from support import Photo, Display


if __name__ == '__main__':
    """
    Load a photo, adopt into PlatePixel points, transform to XYZ via GCD.
    Then display it as the unit sphere..
    """
    reg = Registrar()  # Manage Domains & Projections
    # Domains - 2D image and GCD Ellipsoid.
    p_plt = PlatePixel(reg)             # 2D Pixel Cartesian Domain
    g_gen = GeneralGCD(reg)             # GCD Spherical Domain (latitude/longitude)
    c_ell = EllipsoidCartesian(reg)     # Cartesian Geodesic (xyz)
    c_oct = OctahedralCartesian(reg)    # Cartesian Octahedron (xyz)
    b_oct = OctahedralBarycentric(reg, c_oct)  # Barycentric Octahedron (xy)

    # Projections/Transforms
    ppp = PlatePixelGCD(reg)    # [p_plt, g_gen] Project (Pixel Cartesian <=> GCD)
    EllipsoidGCD(reg)           # [p_plt, c_ell]
    AKOctahedralEllipsoid(reg)  # [c_ell, c_oct]

    # Support Classes
    d = Display()
    ps = Photo()

    # Load cached octahedral and colours from original.
    pc_map = 'world1350x675'
    cache = Path(f'src/{pc_map}.npy')
    if not cache.exists():
        raise ValueError('Far better to load a cache. See ex_0041')
    img = image.imread(f'src/{pc_map}.png', 'png')
    pc_pix = p_plt.adopt(img)
    oc = np.load(cache)
    oc_px = c_oct.adopt(oc)
    oc_px.samples = pc_pix.samples
    # Now loaded cached octahedral and colours from original.
    d.show_pts_3d(oc_px)
    sp_pl = reg.project(oc_px, [c_oct, c_ell, g_gen, p_plt])
    rmg = p_plt.image(sp_pl, (1350, 675))     # convert points back to an [1350,675,3] image.
    fig = plt.figure(figsize=(18, 9), dpi=100, frameon=False)
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
    plt.imshow(rmg)
    plt.show()


