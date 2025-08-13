"""
Part of the H9 project
Compose pixel grids of each barycentric triangle.
Load a Plate Carrée colour map for sampling, project onto Cartesian Unit Sphere and assign KDTree for sample queries.
Project each point of the grid onto Cartesian Unit Sphere and sample them.
The octahedral->spherical projection is relatively fast, so we can handle larger images this way.
Using a pixel grid provides us the ability to map colours to the pixels we need.
Notable feature is that, once adopted, points maintain their position.
Last Tested 21 June 2025
Last Tested 03 August √ 2025
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
    g_sph = GeneralGCD(reg)           # GCD Spherical Domain (latitude/longitude)
    c_sph = EllipsoidCartesian(reg)     # Cartesian Spherical (xyz)
    c_oct = OctahedralCartesian(reg)    # Cartesian Octahedron (xyz)
    b_oct = OctahedralBarycentric(reg, c_oct)  # 2d Flat for addressing.
    n_oct = OctahedralNet(reg, c_oct, b_oct)   # 2d Flat for display.
    p_plt = PlatePixel(reg)             # 2D Pixel Cartesian Domain

    h9 = OctahedralH9()            # formatter.
    h9e = H9Engine()
    b_oct.register_format(h9)

    # Projections/Transforms. Bary and Net are loaded by the domains.
    PlatePixelGCD(reg)          # Transform (Pixel Cartesian <=> GCD)
    EllipsoidGCD(reg)           # g_sph <=> c_sph
    AKOctahedralEllipsoid(reg)  # c_sph <=> (c_oct <=> b_oct)

    # Support Classes
    g = Grid()

    # Load in plate carree - will use 2700x1350 here.
    img = image.imread(f'src/world5400x2700.png', 'png')
    # Register it as a PlatePixel domain.
    pc_px = p_plt.adopt(img)
    # We could project this merely onto GCD. But here we will do to Cartesian Unit.
    pc_sp = reg.project(pc_px, [p_plt, g_sph, c_sph])
    src = KDTree(pc_sp.coords)  # KDTree of plate_carrée projected onto unit sphere.
    # We now have our sample source.
    # Set up a pixel grid of the octahedral net.
    # scale should be 1/4 * the width of a plate carrée (or less).
    scale = 1350
    u_grid = g.sq_grid(scale, 'Λ')
    d_grid = g.sq_grid(scale, 'V')

    for side, octant in b_oct.sides.items():
        grd = u_grid if octant.mode == 'Λ' else d_grid
        pts = octant.adopt(grd)  # These are our net points.
        # project the grid onto cartesian sphere.
        ref = reg.project(pts, [b_oct, c_oct, c_sph])  # ref. addresses on a cartesian unit sphere
        # Use the KDTree to map the grid points onto the sample points.
        _, idx = src.query(ref.coords, workers=-1)  # query KDTree and return indices of pc_sp
        pts.samples = pc_sp.samples[idx]
        img = n_oct.image(pts)
        plt.imsave(f'{side}_map_{scale}.png', img)
