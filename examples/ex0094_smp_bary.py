# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Part of the H9 project
Compose pixel grids of each barycentric triangle.
Load a Plate Carrée colour map for sampling, project onto Cartesian Unit Sphere and assign KDTree for sample queries.
Project each point of the grid onto Cartesian Unit Sphere and sample them.
The octahedral->spherical projection is relatively fast, so we can handle larger images this way.
Using a pixel grid provides us the ability to map colours to the pixels we need.
Notable feature is that, once adopted, points maintain their position.

Last Tested
26 December 2025 0.1.0a4 (passed)
16 December 2025 0.1.0a3 (?passed)
08 October 2025 (passed)
"""
import numpy as np
from matplotlib import image
from scipy.spatial import KDTree
from hhg9 import Registrar, Points
from hhg9.algorithms.grid import sq_grid_vx, fit
from PIL import Image  # image saving


def run(scale=2003):
    """Runner"""
    reg = Registrar()  # Manage Domains & Projections
    p_pix = reg.domain('p_pix')
    b_oct = reg.domain('b_oct')

    # Load in plate carrée - will use 2700x1350 Blue Marble here.
    img = image.imread(f'src/world5400x2700.png', 'png')
    pc_px = p_pix.adopt(img)
    pc_el = reg.project(pc_px, [p_pix, 'g_gcd', 'c_ell'])
    src = KDTree(pc_el.coords)  # KDTree of plate_carrée projected onto unit sphere.
    # We now have our sample source.

    grid_bits_u = sq_grid_vx(scale, 1)
    grid_bits_d = sq_grid_vx(scale, 0)
    # These are the pixel grids we will use

    for side in ["NWA", "NEA"]:
        octant = b_oct.sides[side]
        if octant.mode == 1:
            wid, hgt, rec, msk, _, _ = grid_bits_u
        else:
            wid, hgt, rec, msk, _, _ = grid_bits_d

        rec_in = rec[msk]  # (M,2)

        pts = Points(rec_in, b_oct, components=octant.sig())
        px, py = fit(pts, wid, hgt)
        ref = reg.project(pts, [b_oct, 'c_oct', 'c_ell'])

        _, nn = src.query(ref.coords, workers=-1)
        samples = pc_el.samples[nn]
        rgba = samples if samples.shape[1] == 4 else np.hstack(
            (samples, np.ones((samples.shape[0], 1), dtype=samples.dtype))
        )

        out = np.zeros((hgt, wid, 4), dtype=float)
        out[py, px] = rgba
        Image.fromarray((out * 255).astype(np.uint8)).save(f"output/ex0094_{side}.png")


if __name__ == '__main__':
    run()
