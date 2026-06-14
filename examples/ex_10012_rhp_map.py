# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Render a world map onto the rhombus octahedral net.

Uses:
  ex_10011_rhp   — RhombusNet / RHOMBUS_LAYOUT for the rhombus net layout

Pipeline (forward, per face per c2):
  H9P.hh[bmo, c2] vertices → affine → net-space quad → qa_grid → KDTree sample → composite

For each octant face and each c2 sub-region, the net-space quad is computed
directly from the per-c2 affine (prj.matrix @ c2_matr, c2_ofs + prj.offset),
bypassing BaryNet.forward() which classifies c2 using net net_mode rather than
barycentric net_mode (these differ for NWP and SEA in the rhombus layout).
qa_grid samples within each net-space quad at uniform density, and the backward
chain (n_oct → b_oct → g_gcd) provides the geodetic sampling address.

Last Tested
08 March 2026
"""
import numpy as np
from matplotlib import image
from scipy.spatial import KDTree
from PIL import Image
from hhg9 import Registrar, Points
from hhg9.h9 import H9O, H9P
from hhg9.h9.grid import fit, qa_grid
from hhg9.domains import nets
# from ex_10011_rhp import RHOMBUS_LAYOUT


def run(*, file='src/world360x180.png', scale=300):
    """Render a plate carrée world map onto the rhombus octahedral net."""
    reg = Registrar()
    b_oct = reg.domain('b_oct')
    n_oct = reg.domain('n_oct:rhombus')
    img_w, img_h = n_oct.image_dims(pixels=scale)

    # Load plate carrée and build a KDTree on the unit sphere.
    p_pix = reg.domain('p_pix')
    src_img = image.imread(file, 'png')
    pc_px = p_pix.adopt(src_img)
    pc_el = reg.project(pc_px, [p_pix, 'g_gcd'])
    src = KDTree(pc_el.coords)

    # For each face and c2 sub-region, apply the per-c2 affine directly to
    # H9P.hh[bmo, c2] to get the net-space quad, then sample it with qa_grid.
    # Pixel density: pixels per metric coordinate unit (same convention as qa_grid / ex0095).
    density = float(scale)

    pt_list = []
    for name, sdom in n_oct.sides.items():
        oid = sdom.oid
        bmo = int(H9O.oid_mo[oid])         # barycentric net_mode (from H9O constants)
        prj = n_oct.projs[name]

        for c2 in range(3):
            # Compute per-c2 affine directly: net_xy = bary_xy @ (prj.matrix @ c2_matr) + c2_ofs + prj.offset
            # This avoids BaryNet.forward()'s c2 classification, which uses net net_mode
            # rather than barycentric net_mode (these differ for NWP and SEA in rhombus).
            if prj.c2trans is not None:
                _mo, _rt, ofs, matr = prj.c2trans[c2]
                matrix = prj.matrix @ matr
                offset = np.array(ofs) + np.array(prj.offset)
            else:
                matrix = prj.matrix
                offset = np.array(prj.offset)
            grid = qa_grid(H9P.hh[bmo, c2], density, affine=(matrix, offset))
            pix, msk = grid[2], grid[3]
            if np.any(msk):
                pt_list.append(Points(pix[msk], n_oct, oid))

    pts = Points.concat(pt_list)
    px, py = fit(pts, img_w, img_h)

    # Project net → sphere, query KDTree, composite.
    ref = reg.project(pts, [n_oct, b_oct, 'g_gcd'])
    ok = np.all(np.isfinite(ref.coords), axis=1)

    _, idx = src.query(ref.coords[ok], workers=-1)
    samples = pc_px.samples[idx]
    rgba = samples if samples.shape[1] == 4 else np.hstack(
        (samples, np.ones((samples.shape[0], 1), dtype=samples.dtype))
    )

    out = np.zeros((img_h, img_w, 4), dtype=float)
    out[py[ok], px[ok]] = rgba
    Image.fromarray((out * 255).astype(np.uint8)).save('output/ex_10012_rhombus.png')
    print(f'Saved output/ex_10012_rhombus.png  ({img_w}x{img_h})')


if __name__ == '__main__':
    # scale ≈ 3 * desired_px_per_face (img_w = scale * layout_width/tri_w = scale * 2/3).
    # scale=3000 → ~2000×866 px  (1000 px per face side ≈ 90 degrees).
    run(scale=600, file='src/tissot_3600x1800.png')
