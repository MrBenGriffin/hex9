"""
Part of the H9 project
Compose pixel grids of each barycentric triangle for the octahedral net and normalise.
Project each point of the grid onto GCD and grab sample for each.
Use the sample on the octahedral net and save the result into a png.
The octahedral->spherical projection is relatively fast, so we can handle larger images this way.
Using a pixel grid provides us the ability to map colours to the pixels we need.
"""
import numpy as np
from matplotlib import image, pyplot as plt
from hhg9 import Registrar, H9Engine, Grid
from hhg9.domains import OctahedralCartesian, OctahedralBarycentric, OctahedralNet
from hhg9.formats import OctahedralH9
from support import Display

if __name__ == '__main__':
    reg = Registrar()                   # Manage Domains & Projections
    c_oct = OctahedralCartesian(reg)    # Cartesian Octahedron (xyz)
    b_oct = OctahedralBarycentric(reg, c_oct)  # 2d Flat for addressing.
    n_oct = OctahedralNet(reg, c_oct, b_oct)   # 2d Flat for display.

    h9 = OctahedralH9()            # formatter.
    h9e = H9Engine()
    b_oct.register_format(h9)

    # Support Classes
    g = Grid()
    d = Display()

    scale = 100
    f_org = [g.px_grid(p.offset, scale, p.rev_cs.ud) for p in n_oct.projs.values()]
    pts = n_oct.adopt(np.vstack(f_org))  # These are our net points.
    # project the grid onto barycentric octahedron.
    ref = reg.project(pts, [n_oct, b_oct])  # ref. addresses on a cartesian unit sphere
    # adr = f'{ref:h9}'
    h9a = h9.format_arr(ref, 'i2', False)
    pts.samples = h9.format_arr(ref, 'i2', False)
    d.show_pts_2d(pts)

    # img = n_oct.image(ref)
    # plt.imshow(img, interpolation='nearest')
    # plt.show()
    # plt.imsave(f'net_cmap2_{scale}.png', img)
