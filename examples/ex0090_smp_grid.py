"""
Part of the H9 project
Compose pixel grids of each barycentric triangle for the octahedral net and normalise.
Load a Plate Carrée colour map for sampling, project onto Cartesian Unit Sphere and assign KDTree for sample queries.
Project each point of the grid onto Cartesian Unit Sphere and sample them.
Apply the samples to the octahedral net and save the result into a png.
The octahedral->spherical projection is relatively fast, so we can handle larger images this way.
Using a pixel grid provides us the ability to map colours to the pixels we need.
Notable feature is that, once adopted, points maintain their position.
Last Tested 21 June 2025
"""
import numpy as np
from matplotlib import image, pyplot as plt
from scipy.spatial import KDTree
from hhg9 import Registrar, H9Engine, Grid
from hhg9.domains import GeneralGCD, EllipsoidCartesian, OctahedralCartesian, OctahedralBarycentric, PlatePixel, \
    OctahedralNet
from hhg9.projections import EllipsoidGCD, PlatePixelGCD, AKOctahedralEllipsoid
from hhg9.formats import OctahedralH9


if __name__ == '__main__':
    reg = Registrar()                   # Manage Domains & Projections
    g_gen = GeneralGCD(reg)           # GCD Spherical Domain (latitude/longitude)
    c_ell = EllipsoidCartesian(reg)     # Cartesian Spherical (xyz)
    c_oct = OctahedralCartesian(reg)    # Cartesian Octahedron (xyz)
    b_oct = OctahedralBarycentric(reg, c_oct)  # 2d Flat for addressing.
    n_oct = OctahedralNet(reg, c_oct, b_oct)   # 2d Flat for display.
    p_plt = PlatePixel(reg)             # 2D Pixel Cartesian Domain

    h9 = OctahedralH9()            # formatter.
    h9e = H9Engine()
    b_oct.register_format(h9)

    # Projections/Transforms. Bary and Net are loaded by the domains.
    PlatePixelGCD(reg)               # Transform (Pixel Cartesian <=> GCD)
    EllipsoidGCD(reg)                # c_ell <=> g_gen
    ak = AKOctahedralEllipsoid(reg)  # c_ell <=> (c_oct <=> b_oct)

    # Support Classes
    g = Grid()

    # Load in blue marble - will use 5400x2700 here.
    img = image.imread(f'src/world5400x2700.png', 'png')
    # img = image.imread(f'src/tissot_2560x1280.png', 'png')
    # img = image.imread(f'src/grid75.png', 'png')
    # Register it as a PlatePixel domain.
    pc_px = p_plt.adopt(img)
    # We could project this merely onto GCD. But here we will do to Cartesian Unit.
    pc_sp = reg.project(pc_px, [p_plt, g_gen, c_ell])
    src = KDTree(pc_sp.coords)  # KDTree of plate_carrée projected onto unit sphere.
    # We now have our sample source.
    # Set up a pixel grid of the octahedral net.
    # Scale is the number of pixels used to express 90º longitude at the equator in the Plate Carrée.
    # The net map will be about 445º across and 330º high, and includes white. 5400x2700 => 6681×4959
    scale = 2700
    f_org = [g.sq_grid(scale, p.fwd_cs.mode)+p.offset for p in n_oct.projs.values()]
    pts = n_oct.adopt(np.vstack(f_org))  # These are our net points.
    # project the grid onto cartesian sphere.
    ref = reg.project(pts, [n_oct, b_oct, c_oct, c_ell])  # ref. addresses on a cartesian unit sphere
    # Use KDTree to map the grid points onto the sample points.
    _, idx = src.query(ref.coords, workers=-1)  # query KDTree and return indices of pc_sp
    pts.samples = pc_sp.samples[idx]
    img = n_oct.image(pts, (6681, 4959))
    plt.imsave(f'ex0090_net_{scale}.png', img)
