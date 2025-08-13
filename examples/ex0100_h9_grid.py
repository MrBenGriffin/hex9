"""
Part of the H9 project
Compose pixel grids of each barycentric triangle for the octahedral net and normalise.
Project each point of the grid onto GCD and grab sample for each.
Use the sample on the octahedral net and save the result into a png.
The octahedral->spherical projection is relatively fast, so we can handle larger images this way.
Using a pixel grid provides us the ability to map colours to the pixels we need.
Last Tested  13 August 2025 √
"""
import numpy as np
from matplotlib import image, pyplot as plt
from hhg9 import Registrar, H9Engine, Grid, Points
from hhg9.domains import OctahedralCartesian, OctahedralBarycentric, OctahedralNet
from hhg9.formats import OctahedralH9
from support import Display

if __name__ == '__main__':
    reg = Registrar()                   # Manage Domains & Projections
    c_oct = OctahedralCartesian(reg)    # Cartesian Octahedron (xyz)
    b_oct = OctahedralBarycentric(reg, c_oct)  # 2d Flat for addressing.
    n_oct = OctahedralNet(reg, c_oct, b_oct)   # 2d Flat for display.
    h9e = H9Engine()

    # Support Classes
    g = Grid()
    d = Display()
    rng = np.random.default_rng()

    scale = 2700  #
    cmap = plt.colormaps.get_cmap('terrain')
    cols = cmap(np.linspace(0, 1, 20))
    lut = n_oct.o.sides
    f_org = {lut[k].sig: g.sq_grid(scale, p.fwd_cs.mode)+p.offset for k, p in n_oct.projs.items()}
    pt_list = [Points(v, n_oct, k) for (k, v) in f_org.items()]
    pts = Points.concat(pt_list)  # These are our net points.
    ref = reg.project(pts, [n_oct, b_oct])  # ref. barycentric octahedron
    prf, addr, treg, thx = h9e.addr(ref, 11, False)
    bad = treg == 'X'
    pts.coords[bad] = (0, 0)
    for layer in range(1, 5):
        ai = addr[:, layer]
        pts.samples = cols[ai*2]
        img = n_oct.image(pts, (6681, 4959))
        plt.imsave(f'ex0100_net_{layer}.png', img)

