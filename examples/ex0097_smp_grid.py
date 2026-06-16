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
16 Jun 2026 0.1.3a0 (passed) 41.6s
13 Mar 2026 0.1.1a1 (passed)
26 Dec 2025 0.1.0a4 (passed)
16 Dec 2025 0.1.0a3 (?passed)
12 Oct 2025 (passed)
"""
import numpy as np
from matplotlib import image, pyplot as plt
from matplotlib.collections import PolyCollection
from scipy.spatial import KDTree
from hhg9 import Registrar, Points
from hhg9.base import registrar
from hhg9.domains.nets import net_layouts
from hhg9.h9 import H9O, H9P
from hhg9.h9.addressing import hex_str_encode
from hhg9.h9.binning import hex_reduce, hex_parents
from hhg9.h9.grid import fit, qa_grid, hex_verts_in_noct
from PIL import Image  # image saving

from hhg9.h9.tail import tail_unpack_reversible, TailStyle


def run(*, flavours=None, file='src/world360x180.png', scale=1350):
    """Runner"""
    reg = Registrar()                   # Manage Domains & Projections
    p_pix = reg.domain('p_pix')
    for flavour in flavours:
        reg.domain(f'n_oct:{flavour}')  # or 'butterfly', etc.
    b_oct = reg.domain('b_oct')

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
            prj = n_oct.projs[name]
            for c2 in [0, 1, 2]:
                hhp = H9P.hh[mo, c2]
                placed_mode = prj.c2trans[c2][0] if prj.c2trans is not None else -1
                grid = qa_grid(hhp, scale, affine=prj.c2_affine(c2), net_mode=placed_mode)
                pix, msk = grid[2], grid[3]
                if np.any(msk):
                    pix = Points(pix[msk] + prj.offset, n_oct, oid)
                    pt_list.append(pix)
                else:
                    print(f'No points for {name} {c2}')
        pts = Points.concat(pt_list)  # The n_oct grid of points
        px, py = fit(pts, img_w, img_h)
        b_pxs = reg.project(pts, [n_oct, b_oct])
        ok = b_oct.valid(b_pxs)
        px = px[ok]
        py = py[ok]
        b_pts = b_pxs.select(ok)
        ref = reg.project(b_pts, [b_oct, 'g_gcd'])  # ref. addresses on a cartesian unit sphere
        _, idx = src.query(ref.coords, workers=-1)  # query KDTree and return indices of pc_sp
        samples = pc_px.samples[idx]
        fig = plt.figure(figsize=(img_w/100, img_h/100), dpi=100, frameon=False)
        fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
        ax = fig.add_axes([0, 0, 1, 1])  # span the whole figure

        rgba = samples if samples.shape[1] == 4 else np.hstack(
            (samples, np.ones((samples.shape[0], 1), dtype=samples.dtype))
        )
        out = np.zeros((img_h, img_w, 4), dtype=float)
        out[py, px] = rgba
        plt.imshow(out, extent=[0, n_oct.wi, 0, n_oct.he], alpha=1.0, origin='upper', aspect='auto')
        lw = [2, 1]
        for li, layer in enumerate([1, 2]):
            hex_num, hex_v_k, hex_inv, counts = hex_reduce(b_pts, layer)
            # tails = tail_key_from_reversible(hex_v_k[:, -1])
            hex_par, hex_oid, scale = hex_parents(b_oct, hex_v_k, hex_num)
            xpm, xc2, _, _ = tail_unpack_reversible(hex_v_k[:, -1])
            verts_n = hex_verts_in_noct(hex_par, hex_oid, xpm, xc2, scale, n_oct)
            ctrs_n = np.mean(verts_n.coords.reshape(-1, 6, 2), axis=1)
            hexes = verts_n.coords.reshape(-1, 6, 2)
            collection = PolyCollection(hexes, facecolors='none', ec=(1, 1, 1, 0.5), linewidth=lw[li], antialiaseds=True)
            ax.add_collection(collection)

            # for ctr, lab in zip(ctrs_n, hex_v_k):
            #     hex_addr = lab
            #     label = f'{hex_addr[layer]}'
            #     ax.text(ctr[0], ctr[1], label,
            #             fontsize=12, ha='center', va='center',
            #             color='black', zorder=100, clip_on=True)
        f_name = f"output/ex0097_{flavour}_012.png"
        fig.savefig(f_name, dpi=100)


if __name__ == '__main__':
    flavours = ['butterfly:0500']  #'c_butterfly', 'mortar',  'pacific_windmill', 'diamonds', 'mortar', 'butterfly']  # 'mortar', 'butterfly', 'c_butterfly', 'windmill',
    # flavours = net_layouts  2600/78732
    run(flavours=flavours, scale=1201, file='src/bm_3600x1800.png')  # 'tissot_3600x1800' bm_3600x1800
