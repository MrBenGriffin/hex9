"""
Part of the H9 project
Compose pixel grids of each barycentric triangle for the octahedral net and normalise.
Load a Plate Carrée colour map for sampling, project onto Cartesian Unit Sphere and assign KDTree for sample queries.
Project each point of the grid onto Cartesian Unit Sphere and sample them.
Apply the samples to the octahedral net and save the result into a png.
The octahedral->spherical projection is relatively fast (about 40x spherical->octahedral),
so we can handle larger images this way.
Using a pixel grid provides us the ability to map colours to the pixels we need.
Notable feature is that, once adopted, points maintain their position.
Last Tested 19 November 2025 √
"""
import numpy as np
from matplotlib import image
from scipy.spatial import KDTree
from hhg9 import Registrar, Points
from hhg9.h9.grid import sq_grid, fit
from PIL import Image  # image saving

from experimental.algorithms.warp import Warper


def run(*, flavours=None, file='src/world360x180.png', scale=1350):
    """Runner"""
    reg = Registrar()                   # Manage Domains & Projections
    warper = Warper()
    b_oct = reg.domain('b_oct')
    s_oct = reg.domain('s_oct')
    p_pix = reg.domain('p_pix')
    g_gcd = reg.domain('g_gcd')
    for flavour in flavours:
        reg.domain(f'n_oct:{flavour}')  # or 'butterfly', etc.

    greenwich = np.array([
        [51.47783010901254, +0.00000000001],
        [51.47783010901254, +0.25],
        [51.47783010901254, +0.5],
        [51.47783010901254, +0.75],
        [51.47783010901254, +1.0],
        [51.47783010901254, +1.25],
        [51.47783010901254, +1.5],
        [51.47783010901254, +1.75],
        [51.47783010901254, +2.0],
        [51.47783010901254, +2.25],
        [51.47783010901254, +2.5],
        [51.47783010901254, +2.75],
        [51.47783010901254, +3.0],
        [51.47783010901254, +1.25],
        [51.47783010901254, +3.5],
        [51.47783010901254, +3.75],
        [51.47783010901254, +4.0],
    ], dtype=np.float64)  # NEA
    gwp = Points(greenwich, g_gcd)
    bwp = reg.project(gwp, [g_gcd, b_oct])
    gbk = reg.project(bwp, [b_oct, g_gcd])
    assert np.allclose(gbk.coords, gwp.coords)
    swp = reg.project(bwp, [b_oct, s_oct])
    bbk = reg.project(swp, [s_oct, b_oct])
    assert np.allclose(bbk.coords, bwp.coords)
    sww = warper.warp(swp)  # No unwarp yet..
    bbw = reg.project(sww, [s_oct, b_oct])
    gww = reg.project(bbw, [b_oct, g_gcd])
    print(f'Greenwich: {gwp.coords}')
    print(f'Greenwich Post-Warp: {gww.coords}')

    # Load in plate carrée - eg 2700x1350 Blue Marble here.
    img = image.imread(file, 'png')
    pc_px = p_pix.adopt(img)
    pc_el = reg.project(pc_px, [p_pix, g_gcd])
    src = KDTree(pc_el.coords)  # KDTree of plate_carrée projected onto unit sphere.
    # We now have our sample source.
    gbu = sq_grid(scale, 1)   # barymetric xy rectilinear pixel grid.
    gbd = sq_grid(scale, 0)
    full_grid_u, mask_u = gbu[2], gbu[3]
    full_grid_d, mask_d = gbd[2], gbd[3]
    grid_u = full_grid_u[mask_u]
    grid_d = full_grid_d[mask_d]

    for flavour in flavours:
        n_oct = reg.domain(f'n_oct:{flavour}')
        img_w, img_h = n_oct.image_dims(pixels=scale)
        pt_list = []

        for sig, composite in n_oct.components.items():
            name = n_oct.c_oct.signs[sig]
            prj = n_oct.projs[name]
            offset = prj.offset
            grid_base = [grid_d, grid_u][prj.fwd_cs.mode]
            grid = grid_base.copy()+offset
            pt_list.append(Points(grid, n_oct, sig))
        pcs = Points.concat(pt_list)  # These are our net points.
        pts = pcs
        px, py = fit(pts, img_w, img_h)

        # Project net points into barycentric octahedral space, then into simplex space
        p_boct = reg.project(pts, [n_oct, b_oct])
        p_soct = reg.project(p_boct, [b_oct, s_oct])

        # Apply the MA warp in simplex coordinates (warper has its own internal scale/config)
        p_soct_warp = warper.warp(p_soct)

        # Map warped points back through b_oct to the unit sphere for sampling
        ref = reg.project(p_soct_warp, [s_oct, b_oct, g_gcd])  # ref. addresses on a cartesian unit sphere

        _, idx = src.query(ref.coords, workers=-1)  # query KDTree and return indices of pc_sp
        samples = pc_px.samples[idx]
        rgba = samples if samples.shape[1] == 4 else np.hstack(
            (samples, np.ones((samples.shape[0], 1), dtype=samples.dtype))
        )
        out = np.zeros((img_h, img_w, 4), dtype=float)
        out[py, px] = rgba
        Image.fromarray((out * 255).astype(np.uint8)).save(f'output/{flavour}_x0096.png')


if __name__ == '__main__':
    flavours = ['mortar']
    run(flavours=flavours, scale=600, file='../tissot_2560x1280.png')
