"""
Part of the H9 project
This shows the hexagon ids as colours, for each layer. Note that
layer zero is calculated differently, as it has a named hexagon NA, SW, etc.
Last Tested 08 October 2025 √
"""
import numpy as np
from matplotlib import pyplot as plt
from hhg9 import Registrar, Points
from hhg9.algorithms.grid import sq_grid, fit
from PIL import Image  # image saving
from hhg9.formats import OctahedralH9
from hhg9.h9.addressing import hex_layer


def run(layout, scale):
    """Runner"""
    reg = Registrar()                        # Manage Domains & Projections
    b_oct = reg.domain(f'b_oct')
    n_oct = reg.domain(f'n_oct:{layout}')   # 'mortar', 'butterfly', etc.

    h9f = OctahedralH9(reg)
    b_oct.register_format(h9f)

    img_w, img_h = n_oct.image_dims(pixels=scale)
    gbd = sq_grid(scale, 0)
    gbu = sq_grid(scale, 1)

    # wid, hgt, rec, trx, px_idx, py_idx
    full_grid_u = gbu[2]
    full_grid_d = gbd[2]
    mask_u = gbu[3]
    mask_d = gbd[3]
    grid_u = full_grid_u[mask_u]
    grid_d = full_grid_d[mask_d]

    cmap = plt.colormaps.get_cmap('plasma')
    cols = cmap(np.linspace(0, 1, 16))
    cols[15, 3] = 0  # stick alpha=transparent onto illegal.
    pt_list = []
    for sig, composite in n_oct.components.items():
        name = n_oct.c_oct.signs[sig]
        prj = n_oct.projs[name]
        grid_base = [grid_d, grid_u][prj.fwd_cs.mode]
        grid = grid_base.copy() + prj.offset
        pt_list.append(Points(grid, n_oct, sig))
    pts = Points.concat(pt_list)  # These are our net points.
    px, py = fit(pts, img_w, img_h)
    # now generate the sample pts.
    ref = reg.project(pts, [n_oct, b_oct])
    for layer in range(5):
        addr = hex_layer(ref, layer)
        idt = addr[:, layer]  # should be same as -1
        samples = cols[idt.astype(np.uint8)]
        rgba = samples if samples.shape[1] == 4 else np.hstack(
            (samples, np.ones((samples.shape[0], 1), dtype=samples.dtype))
        )
        out = np.zeros((img_h, img_w, 4), dtype=float)
        out[py, px] = rgba
        f_name = f'../output/h9_{layout}_L{layer}.png'
        Image.fromarray((out * 255).astype(np.uint8)).save(f_name)
        print(f'Saved {f_name}')


if __name__ == '__main__':
    run('butterfly', 1000)
