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
from pathlib import Path

import numpy as np
from matplotlib import image
from scipy.spatial import KDTree
from hhg9 import Registrar, Points
from hhg9.h9 import H9O
from hhg9.algorithms.grid import sq_grid, fit
from PIL import Image  # image saving


def load_grid(layer: int):
    """
    Cached loader for the grid NPZ for a given hex_layer.
    This avoids re-reading the same file from disk when `run` is called
    repeatedly for the same (hex_layer,e) pair.
    """
    f_name = Path(f"grid_l{layer}.npz")
    repo = np.load(f_name, allow_pickle=True)
    cmp = repo['cmp']
    xy_vert = repo['xy_vert']
    v_ell = repo['v_ell']
    oc_vtx = repo['oc_vtx']
    oc_edg = repo['oc_edg']
    grid = repo['grid']
    locs = repo['locs'] if 'locs' in repo.files else None
    return cmp, xy_vert, v_ell, oc_vtx, oc_edg, grid, locs


def run(*, flavours=None, file='src/world360x180.png', scale=1350):
    """Runner"""
    reg = Registrar()                   # Manage Domains & Projections
    p_pix = reg.domain('p_pix')
    for flavour in flavours:
        reg.domain(f'n_oct:{flavour}')  # or 'butterfly', etc.

    b_oct = reg.domain('b_oct')
    b_oct.set_warp('output/l4_polished.npz')

    # Load in plate carrée - will use 2700x1350 Blue Marble here.
    img = image.imread(file, 'png')
    pc_px = p_pix.adopt(img)
    pc_el = reg.project(pc_px, [p_pix, 'g_gcd'])
    src = KDTree(pc_el.coords)  # KDTree of plate_carrée projected onto unit sphere.
    # We now have our sample source.
    gbu = sq_grid(scale, 1)
    gbd = sq_grid(scale, 0)
    full_grid_u, mask_u = gbu[2], gbu[3]
    full_grid_d, mask_d = gbd[2], gbd[3]
    grid_u = full_grid_u[mask_u]
    grid_d = full_grid_d[mask_d]


    for flavour in flavours:
        n_oct = reg.domain(f'n_oct:{flavour}')
        img_w, img_h = n_oct.image_dims(pixels=scale)
        pt_list = []

        for name, sdom in n_oct.sides.items():
            oid = sdom.oid
            sig = H9O.oid_cmp[oid]
            prj = n_oct.projs[name]
            offset = prj.offset
            grid_base = [grid_d, grid_u][sdom.mode]
            grid = grid_base.copy()+offset
            pt_list.append(Points(grid, n_oct, sig))
        pcs = Points.concat(pt_list)  # These are our net points.
        pts = pcs
        px, py = fit(pts, img_w, img_h)
        ref = reg.project(pts, [n_oct, 'b_oct', 'g_gcd']) # ref. addresses on a cartesian unit sphere
        _, idx = src.query(ref.coords, workers=-1)  # query KDTree and return indices of pc_sp
        samples = pc_px.samples[idx]
        rgba = samples if samples.shape[1] == 4 else np.hstack(
            (samples, np.ones((samples.shape[0], 1), dtype=samples.dtype))
        )
        out = np.zeros((img_h, img_w, 4), dtype=float)
        out[py, px] = rgba
        Image.fromarray((out * 255).astype(np.uint8)).save(f'output/ex0095_{flavour}.png')


if __name__ == '__main__':
    flavours = ['mortar']  # 'mortar', 'butterfly', 'c_butterfly', 'windmill',
    run(flavours=flavours, scale=3600, file='output/tissot.png')
