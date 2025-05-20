"""
Part of the H9 project
This follows ex0030_plate_glb.py (which loaded the png and displayed it as a unit sphere).
This loads a Plate Carrée png image,
converts it to XYZ Octagon, via latitude/longitude and sphere and displays it.
The inverse of the AK projection is slow, so we use a cache. cf. ex0041_cache.py
"""
from pathlib import Path

import numpy as np

from hhg9 import Registrar, Points
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
    ppp = PlatePixelGCD(reg)   # Transform (Pixel Cartesian <=> GCD)
    CartesianGCD(reg)
    # Initialise (spherical to octahedral) warm-up takes a couple of seconds.
    AKOctahedralSpherical(reg)

    # Support Classes
    d = Display()
    ps = Photo()

    # Load cached octahedral and colours from original.
    pc_map = 'world1350x675'
    cache = Path(f'{pc_map}.npy')
    if not cache.exists():
        raise ValueError('Far better to load a cache. See ex_0041')
    ps.load(f'../preparatory/{pc_map}.png')
    pc_pix = p_plt.adopt(ps.img)
    oc = np.load(cache)
    oc_px = c_oct.adopt(oc)
    oc_px.samples = pc_pix.samples
    # Now loaded cached octahedral and colours from original.
    # project it back to the plate carree.
    # PlatePixelGCD needs to have its dimensions set - that is currently unknown.
    # we can change the size here if we like. It's possibly the worst way of scaling anything, though...
    # ppp.set_dim(Points(np.array([[720, 360]])))
    ppp.set_dim(pc_pix)
    p_px = reg.project(oc_px, [c_oct, c_sph, g_sph, p_plt])
    d.show_pts_3d(oc_px)
    d.show_pts_2d(p_px)
