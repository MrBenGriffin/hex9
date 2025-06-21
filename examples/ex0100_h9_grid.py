"""
Part of the H9 project
# June 2025 fixed..
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
    rng = np.random.default_rng()

    scale = 2700  #
    cmap = plt.colormaps.get_cmap('terrain')
    cols = cmap(np.linspace(0, 1, 12))
    f_org = []
    f_org = [g.sq_grid(scale, p.fwd_cs.mode)+p.offset for p in n_oct.projs.values()]
    values = np.vstack(f_org)
    rng.shuffle(values)
    pts = n_oct.adopt(values)  # These are our net points.
    # This logic could be migrated against the TODO in n_oct.
    # for p in n_oct.projs.values():
    #     mode = p.fwd_cs.mode
    #     # mpi = p.theta % (2 * np.pi)
    #     # if mpi < 1e-8:
    #     #     ud = p.rev_cs.mode
    #     # else:
    #     #     ud = {'V': 'Λ', 'Λ': 'V'}[p.rev_cs.mode]
    #     f_org.append([g.sq_grid(scale, mode)+p.offset])

    ref = reg.project(pts, [n_oct, b_oct])  # ref. barycentric octahedron
    layer = 3  # The layer of hexagons to colour. 0=12, 1=108 hexes.
    addr = h9.format_arr(ref, f'x11', False)
    # cph = reg.project(ref, [b_oct, c_oct])
    for layer in range(11):
        ai = np.array([int(a[layer]) for a in addr])
        pts.samples = cols[ai]
        img = n_oct.image(pts, (6681, 4959))
        plt.imsave(f'ex0100_net_{layer}.png', img)
