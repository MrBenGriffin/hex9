# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Compose pixel grids of each barycentric triangle.
Load a Plate Carrée colour map for sampling, project onto Cartesian Unit Sphere and assign KDTree for sample queries.
Project each point of the grid onto Cartesian Unit Sphere and sample them.
The octahedral->spherical projection is relatively fast, so we can handle larger images this way.
Using a pixel grid provides us the ability to map colours to the pixels we need.
Notable feature is that, once adopted, points maintain their position.
This is functionally no different from 0094, as we are currently not implementing warps.
Though it -does- do a roundtrip to simplex.

Last Tested
02 March 2026 0.1.1a1 (passed)
26 December 2025 0.1.0a4 (passed)
16 December 2025 0.1.0a3 (passed)
08 October 2025 (passed)
"""
import numpy as np
from matplotlib import image
from scipy.spatial import KDTree
from hhg9 import Registrar, Points
from hhg9.h9.grid import fit, qa_grid
from PIL import Image  # image saving
from hhg9.h9 import H9P


def run(scale=800):
    """Runner"""
    reg = Registrar()  # Manage Domains & Projections
    p_pix = reg.domain('p_pix')
    b_oct = reg.domain('b_oct')

    # Load in plate carrée - will use 2700x1350 Blue Marble here.
    img = image.imread(f'src/tissot_3600x1800.png', 'png')
    pc_px = p_pix.adopt(img)
    pc_el = reg.project(pc_px, [p_pix, 'g_gcd', 'c_ell'])
    src = KDTree(pc_el.coords)  # KDTree of plate_carrée projected onto unit sphere.

    grids = {}
    for mode in [0, 1]:
        for c2 in [0, 1, 2]:
            k = (mode, c2)
            poly = H9P.hh[mode, c2, ...]
            sc = scale - 2/3 if c2 > 0 else scale * 4/3
            gw, gh, pxl, msk, (grid_org, (sx, sy)) = qa_grid(poly, sc)
            in_scope = pxl[msk]
            grids[k] = (gw, gh, in_scope)

    for side in b_oct.sides:
        octant = b_oct.sides[side]
        for c2 in [0, 1, 2]:
            wid, hgt, rec_in = grids[(octant.mode, c2)]
            pts = Points(rec_in, b_oct, components=octant.sig())
            px, py = fit(pts, wid, hgt)
            ref = reg.project(pts, [b_oct, 'c_oct', 'c_ell'])
            # h9h = addr.hex_layer(pts, 0, addr.TailStyle.key)
            # groups, inv = np.unique(h9h, axis=0, return_inverse=True)

            _, nn = src.query(ref.coords, workers=-1)
            samples = pc_el.samples[nn]
            rgba = samples if samples.shape[1] == 4 else np.hstack(
                (samples, np.ones((samples.shape[0], 1), dtype=samples.dtype))
            )
            out = np.zeros((hgt, wid, 4), dtype=float)
            out[py, px] = rgba
            Image.fromarray((out * 255).astype(np.uint8)).save(f"output/hh_l0/ex0094A_{side}_{c2}.png")


if __name__ == '__main__':
    run()
