"""
Part of the H9 project
Compose pixel grids of each barycentric triangle for the octahedral net,
fill them with a colour (for each triangle), and display.
"""
import numpy as np
from hhg9 import Registrar, H9Engine, Grid
from hhg9.domains import SphericalGCD, SphericalCartesian, OctahedralCartesian, OctahedralBarycentric, OctahedralNet
from hhg9.projections import CartesianGCD, AKOctahedralSpherical
from hhg9.formats import OctahedralH9
from support import Util, Display

def add_sample(pts, sample):
    """Prefix a group of pixels with sample (e.g. colour) values"""
    sample_array = np.tile(sample, (pts.shape[0], 1))  # shape (N, 4)
    return np.hstack([sample_array, pts])  # shape (N, 6)


if __name__ == '__main__':
    """
        Convert a set of locations into a set of h9 formats.
    """
    reg = Registrar()                   # Manage Domains & Projections
    g_sph = SphericalGCD(reg)           # GCD Spherical Domain (latitude/longitude)
    c_sph = SphericalCartesian(reg)     # Cartesian Spherical (xyz)
    c_oct = OctahedralCartesian(reg)    # Cartesian Octahedron (xyz)
    b_oct = OctahedralBarycentric(reg, c_oct)  # 2d Flat for addressing.
    n_oct = OctahedralNet(reg, c_oct, b_oct)   # 2d Flat for display.

    h9 = OctahedralH9()            # formatter.
    h9e = H9Engine()
    b_oct.register_format(h9)

    # Projections/Transforms. Bary and Net are loaded by the domains.
    CartesianGCD(reg)           # g_sph <=> c_sph
    AKOctahedralSpherical(reg)  # c_sph <=> (c_oct <=> b_oct)

    # Support Classes
    d = Display()
    u = Util()
    g = Grid()

    # Triangle edge length will be np.sqrt(2) in unit octahedron.
    col = d.colours(8)
    c0, c1 = col[1], col[2]
    scale = 315.0/np.sqrt(2)  # scale: divide by sqrt2 to get pixels per edge. 1350 / 4 * 3.5 = 315
    wid, hgt = n_oct.width, n_oct.height

    f_org = [g.px_grid(p.offset, scale, p.rev_cs.ud) for _, p in n_oct.projs.items()]
    stuff = np.vstack([add_sample(v/scale, col[i]) for i, v in enumerate(f_org)])
    npt = n_oct.adopt(stuff)
    d.show_pts_2d(npt)
