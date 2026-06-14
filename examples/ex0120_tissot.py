# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Tissot indicatrices on the H9 octahedral net.

Samples Kunimune's Tissot indicatrix plate carrée (tissot_3600x1800.png)
through the H9 octahedral projection and renders it in the chosen net
layout, overlaid with L1 and L2 hex grids.

The hex overlay demonstrates that H9 cells are perfectly regular hexagons
spanning the entire globe — the 12 pentagonal cells sit at face vertices
on the net boundary and are not visible in the interior.

Tissot SVG source: https://github.com/jkunimune15/Map-Projections

Outputs:
  output/ex0120_tissot_{flavour}.png

Usage:
  python ex0120_tissot.py                        # default: butterfly
  python ex0120_tissot.py --flavour mortar
  python ex0120_tissot.py --flavour rhombus
  python ex0120_tissot.py --flavour butterfly:0500
  python ex0120_tissot.py --all                  # all registered layouts

Last Tested
20 Apr 2026
"""

import argparse
import os
import numpy as np
from matplotlib import image, pyplot as plt
from matplotlib.collections import PolyCollection
from scipy.spatial import KDTree
from hhg9 import Registrar, Points
from hhg9.domains.nets import net_layouts
from hhg9.h9 import H9O, H9P, H9K
from hhg9.h9.grid import fit, qa_grid, HexMesh

os.makedirs('output', exist_ok=True)

SOURCE     = 'src/tissot_7200x3600.png'   # re-render SVG at 2× for clean graticules
SOURCE_LO  = 'src/tissot_3600x1800.png'   # fallback for quick drafts
SCALE      = 2400       # triangle side in pixels — matches ex0096/ex0097
HEX_COLOUR = (0.12156862745098039, 0.8588235294117647, 0.2823529411764706)  # dark forest green; legible on cream land and blue sea


def run(flavours: list[str], scale: int = SCALE, file: str = SOURCE) -> None:
    reg   = Registrar()
    p_pix = reg.domain('p_pix')
    for flavour in flavours:
        reg.domain(f'n_oct:{flavour}')
    b_oct = reg.domain('b_oct')

    img   = image.imread(file, 'png')
    pc_px = p_pix.adopt(img)
    pc_el = reg.project(pc_px, [p_pix, 'g_gcd'])
    src   = KDTree(pc_el.coords)

    for flavour in flavours:
        print(f'Rendering flavour: {flavour}')
        n_oct        = reg.domain(f'n_oct:{flavour}')
        img_w, img_h = n_oct.image_dims(pixels=scale)
        pt_list      = []

        for name, sdom in n_oct.sides.items():
            oid = sdom.oid
            mo  = H9O.oid_mo[oid]
            prj = n_oct.projs[name]
            for c2 in [0, 1, 2]:
                hhp         = H9P.hh[mo, c2]
                placed_mode = prj.c2trans[c2][0] if prj.c2trans is not None else -1
                grid        = qa_grid(hhp, scale, affine=prj.c2_affine(c2), net_mode=placed_mode)
                pix, msk    = grid[2], grid[3]
                if np.any(msk):
                    pt_list.append(Points(pix[msk] + prj.offset, n_oct, oid))

        pts        = Points.concat(pt_list)
        px, py     = fit(pts, img_w, img_h)
        b_pxs      = reg.project(pts, [n_oct, b_oct])
        ok         = b_oct.valid(b_pxs)
        px, py     = px[ok], py[ok]
        b_pts      = b_pxs.select(ok)
        ref        = reg.project(b_pts, [b_oct, 'g_gcd'])
        _, idx     = src.query(ref.coords, workers=-1)
        samples    = pc_px.samples[idx]

        fig = plt.figure(figsize=(img_w / 100, img_h / 100), dpi=100, frameon=False)
        fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
        ax  = fig.add_axes([0, 0, 1, 1])

        rgba = samples if samples.shape[1] == 4 else np.hstack(
            (samples, np.ones((samples.shape[0], 1), dtype=samples.dtype))
        )
        out = np.zeros((img_h, img_w, 4), dtype=float)
        out[py, px] = rgba
        ax.imshow(out, extent=[0, n_oct.wi, 0, n_oct.he],
                  alpha=1.0, origin='upper', aspect='auto')

        # Hex overlay: L1 bold, L2 fine — proves full-globe coverage, no visible pentagons.
        # Seam filter: drop hexes with any edge longer than the face edge W = √2.
        # Hexes spanning net-connected adjacent faces are valid — their shared-boundary
        # vertices have consistent n_oct coords so all edges stay short.
        # Only seam-crossing hexes (vertices jumping to disconnected face regions)
        # produce edges ≥ W and are correctly rejected.
        mesh = HexMesh.create([1, 2], reg)
        n_pts = reg.project(mesh.pts, [b_oct, n_oct])
        face_edge = H9K.derived.W   # √2 — the face triangle edge length in n_oct space
        r, g, b = HEX_COLOUR
        for li, layer in enumerate([1, 2]):
            lw    = 1.5 if li == 0 else 0.5
            alpha = 0.50 if li == 0 else 0.25
            hexes = n_pts.coords[mesh[layer]].reshape(-1, 6, 2)
            edges = np.roll(hexes, -1, axis=1) - hexes           # (N, 6, 2)
            max_edge = np.hypot(edges[:, :, 0], edges[:, :, 1]).max(axis=1)
            valid = max_edge <= (face_edge / (3**layer))
            col = PolyCollection(hexes[valid], facecolors='none',
                                 ec=(r, g, b, alpha), linewidth=lw, antialiaseds=True)
            ax.add_collection(col)

        safe   = flavour.replace(':', '_')
        f_name = f'output/ex0120_tissot_{safe}.png'
        fig.savefig(f_name, dpi=200)
        plt.close(fig)
        print(f'  Saved: {f_name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--flavour', default='butterfly:0500',
                        help='Net layout name (default: butterfly:0500)')
    parser.add_argument('--all', action='store_true',
                        help='Render all registered net layouts')
    parser.add_argument('--scale', type=int, default=SCALE,
                        help=f'Triangle side in pixels (default: {SCALE})')
    args = parser.parse_args()

    chosen = list(net_layouts.keys()) if args.all else [args.flavour]
    run(chosen, scale=args.scale)
