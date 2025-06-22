"""
Part of the H9 project
This loads a Plate Carrée png image,
converts the entire image as coordinates into XYZ Octagon,
Normally we sample the image via the forward projection, instead of converting from geodesic
Last Tested 21 June 2025 √
"""
import numpy as np
from hhg9 import Registrar
from hhg9.domains import PlatePixel, GeneralGCD, EllipsoidCartesian, OctahedralCartesian, OctahedralBarycentric
from hhg9.projections import PlatePixelGCD, EllipsoidGCD, AKOctahedralEllipsoid
from support import Photo, Display
from pathlib import Path


if __name__ == '__main__':
    """
    Load a photo, adopt into PlatePixel points, transform to XYZ via GCD.
    Then display it as the unit sphere..
    """
    reg = Registrar()  # Manage Domains & Projections
    # Domains - 2D image and GCD Spherical.
    p_plt = PlatePixel(reg)             # 2D Pixel Cartesian Domain
    g_gen = GeneralGCD(reg)             # GCD Spherical Domain (latitude/longitude)
    c_ell = EllipsoidCartesian(reg)     # Cartesian Geodesic (xyz)
    c_oct = OctahedralCartesian(reg)    # Cartesian Octahedron (xyz)
    b_oct = OctahedralBarycentric(reg, c_oct)  # Barycentric Octahedron (xy)

    # Projections/Transforms
    ppp = PlatePixelGCD(reg)         # [p_plt, g_gen] Project (Pixel Cartesian <=> GCD)
    EllipsoidGCD(reg)                # [p_plt, c_ell]
    ak = AKOctahedralEllipsoid(reg)  # [c_ell, c_oct]

    # Support Classes
    d = Display()
    ps = Photo()

    # circumference of earth: 40075017m
    # 1 pix = 40075017/1350 m = about 30km
    cache = Path('src/world1350x675.npy')
    ps.load('src/world1350x675')
    pc_pix = p_plt.adopt(ps.img)
    ak.set_accuracy(30000.0)  # 30km

    if not cache.exists():
        # Project it into Cartesian Octahedral via GCD.
        el_xyz = reg.project(pc_pix, [p_plt, g_gen, c_ell])
        oc_xyz = reg.project(el_xyz, [c_ell, c_oct])
        np.save(cache, oc_xyz.coords)
    else:
        oc = np.load(cache)
        c_pix = c_oct.adopt(oc)
        c_pix.samples = pc_pix.samples
        d.show_pts_3d(c_pix)
