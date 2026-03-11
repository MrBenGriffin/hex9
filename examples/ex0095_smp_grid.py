# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Compose pixel grids of each barycentric triangle for the octahedral net and normalise.
Load a Plate Carrée colour map for sampling, project onto Cartesian Unit Sphere and assign KDTree for sample queries.
Project each point of the grid onto Cartesian Unit Sphere and sample them.
Apply the samples to various octahedral net flavours and save the result into a png.
The octahedral->spherical projection is relatively fast (about 40x spherical->octahedral),
so we can handle larger images this way.
Using a pixel grid provides us the ability to map colours to the pixels we need.
Notable feature is that, once adopted, points maintain their position.

Last Tested
26 December 2025 0.1.0a4 (passed)
16 December 2025 0.1.0a3 (?passed)
12 October 2025 (passed)
"""
import numpy as np
from matplotlib import image
from scipy.spatial import KDTree
from hhg9 import Registrar, Points
from hhg9.h9 import H9O, H9P
from hhg9.h9.grid import fit, qa_grid
from PIL import Image  # image saving


def run(*, flavours=None, file='src/world360x180.png', scale=1350):
    """Runner"""
    reg = Registrar()                   # Manage Domains & Projections
    p_pix = reg.domain('p_pix')
    for flavour in flavours:
        reg.domain(f'n_oct:{flavour}')  # or 'butterfly', etc.
    b_oct = reg.domain('b_oct')

    # Load in plate carrée
    img = image.imread(file, 'png')
    pc_px = p_pix.adopt(img)
    pc_el = reg.project(pc_px, [p_pix, 'g_gcd'])
    src = KDTree(pc_el.coords)  # KDTree of plate_carrée projected onto unit sphere.
    for flavour in flavours:
        n_oct = reg.domain(f'n_oct:{flavour}')
        img_w, img_h = n_oct.image_dims(pixels=scale)
        pt_list = []
        for name, sdom in n_oct.sides.items():
            oid = sdom.oid
            mo = H9O.oid_mo[oid]
            cmp = H9O.oid_cmp[oid]
            prj = n_oct.projs[name]
            for c2 in [0, 1, 2]:
                hhp = H9P.hh[mo, c2]
                grid = qa_grid(hhp, scale, affine=prj.c2_affine(c2))
                pix, msk = grid[2], grid[3]
                if np.any(msk):
                    pix = Points(pix[msk] + prj.offset, n_oct, cmp)
                    pt_list.append(pix)
                else:
                    print(f'No points for {name} {c2}')
        pts = Points.concat(pt_list)
        px, py = fit(pts, img_w, img_h)
        ref = reg.project(pts, [n_oct, b_oct, 'g_gcd'])  # ref. addresses on a cartesian unit sphere

        ok = np.all(np.isfinite(ref.coords), axis=1)
        px_ok = px[ok]
        py_ok = py[ok]

        _, idx = src.query(ref.coords[ok], workers=-1)  # query KDTree and return indices of pc_sp
        samples = pc_px.samples[idx]
        rgba = samples if samples.shape[1] == 4 else np.hstack(
            (samples, np.ones((samples.shape[0], 1), dtype=samples.dtype))
        )
        out = np.zeros((img_h, img_w, 4), dtype=float)
        out[py_ok, px_ok] = rgba
        Image.fromarray((out * 255).astype(np.uint8)).save(f'output/ex0095_{flavour}.png')


if __name__ == '__main__':
    flavours = ['rhombus', 'statue'] #, 'mortar',  'pacific_windmill', 'diamonds', 'mortar', 'butterfly']  # 'mortar', 'butterfly', 'c_butterfly', 'windmill',
    run(flavours=flavours, scale=600, file='src/bm_3600x1800.png')
