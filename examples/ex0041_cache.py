"""
Part of the H9 project
This loads a Plate Carrée png image,
converts it to XYZ Octagon, and stores it as an np file if the file is not present.
This is because the AK inverse projection (sphere to octahedron) is slow.
"""
import numpy as np

from hhg9 import Registrar
from hhg9.domains import PlatePixel, SphericalGCD, SphericalCartesian, OctahedralCartesian
from hhg9.projections import PlatePixelGCD, CartesianGCD, AKOctahedralSpherical
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
    g_sph = SphericalGCD(reg)           # GCD Spherical Domain (latitude/longitude)
    c_sph = SphericalCartesian(reg)     # Cartesian Spherical (xyz)
    c_oct = OctahedralCartesian(reg)    # Cartesian Octahedron (xyz)

    # Projections/Transforms
    PlatePixelGCD(reg)   # Transform (Pixel Cartesian <=> GCD)
    CartesianGCD(reg)
    AKOctahedralSpherical(reg)

    # Support Classes
    d = Display()
    ps = Photo()
    # ps.load('../preparatory/world1350x675.png')

    cache = Path('world1350x675.npy')
    ps.load('../preparatory/world1350x675.png')
    pc_pix = p_plt.adopt(ps.img)
    if not cache.exists():
        # project it into Cartesian spherical via GCD.
        oc_xyz = reg.project(pc_pix, [p_plt, g_sph, c_sph, c_oct])
        np.save(cache, oc_xyz.coords)
    else:
        oc = np.load(cache)
        c_pix = c_oct.adopt(oc)
        c_pix.samples = pc_pix.samples
        d.show_pts_3d(c_pix)
    # oc_xyz = reg.project(pc_px, [p_plt, g_sph, c_sph, c_oct])
    # Display it as a unit Octagon.
    # d.show_pts_3d(oc_xyz)
