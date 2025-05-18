"""
Part of the H9 project
Compose pixel grids of each barycentric triangle for the octahedral net and normalise.
Load a Plate Carrée map to use for sampling and compose a sample lookup based upon latitude and longitude.
Project each point of the grid onto GCD and grab sample for each.
Use the sample on the octahedral net and display the result.
"""
import numpy as np
from hhg9 import Registrar, H9Engine, Grid
from hhg9.domains import SphericalGCD, SphericalCartesian, OctahedralCartesian, OctahedralBarycentric, OctahedralNet, \
    PlatePixel
from hhg9.projections import CartesianGCD, AKOctahedralSpherical
from hhg9.formats import OctahedralH9
from support import Util, Display, Photo
from scipy.spatial import KDTree


def add_sample(pts, sample):
    """Prefix a group of pixels with sample (e.g. colour) values"""
    sample_array = np.tile(sample, (pts.shape[0], 1))  # shape (N, 4)
    return np.hstack([sample_array, pts])  # shape (N, 6)


if __name__ == '__main__':
    """
        Convert a set of locations into a set of h9 formats.
    """
    reg = Registrar()                   # Manage Domains & Projections
    p_plt = PlatePixel(reg)             # 2D Pixel Cartesian Domain
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
    p = Photo()

    p.load('../preparatory/world360x180.png')
    # p.load('../preparatory/world1350x675.png')
    # use plate_carrée pc_px domain to adopt the ps.image (shape [675,1350,4], with RGBA)
    pc_px = p_plt.adopt(p.img)
    pc_ll = reg.project(pc_px, [p_plt, g_sph])
    sample = KDTree(pc_ll[..., -2:])  # KDTree of the plate_carrée in GCD (latitude/longitude).
    # sample.query(lab, workers=-1)

    # Triangle edge length will be np.sqrt(2) in unit octahedron.
    col = d.colours(8)
    c0, c1 = col[1], col[2]
    # scale = 315.0/np.sqrt(2)  # scale: divide by sqrt2 to get pixels per edge. 1350 / 4 * 3.5 = 315
    scale = 32.0/np.sqrt(2)  # scale: divide by sqrt2 to get pixels per edge. 1350 / 4 * 3.5 = 315
    wid, hgt = n_oct.width, n_oct.height
    f_org = [g.px_grid(p.offset, scale, p.rev_cs.ud) for _, p in n_oct.projs.items()]
    stuff = np.vstack([v/scale for i, v in enumerate(f_org)])  # remove the scale to normalise.
    npt = n_oct.adopt(stuff)  # This is our projection points.
    # We can now get our sample references by back-projecting this onto the GCD sample space.
    # This is *much* faster than attempting to move the samples to the network space.
    # It depends on each point maintaining its position in the array.
    spt = reg.project(npt, [n_oct, b_oct, c_oct, c_sph, g_sph])
    rx = sample.query(spt, workers=-1)


    # d.show_pts_2d(npt)
