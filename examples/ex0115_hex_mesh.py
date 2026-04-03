# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""
Direct hex mesh generation from H9 lattice LUTs.

Uses HexMesh.create() from hhg9.h9.grid: sc_mode=0 supercells only, boundary
hex exterior verts reflected y→-y into the adjacent face via H9O.oid_nb.
Each hex is generated exactly once; no seam artefacts, no duplication.

Last Tested
-----------
"""
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection

from hhg9 import Registrar, Points
from hhg9.domains import PlatePixelCarree
from hhg9.h9.grid import HexMesh, valid_ll

if __name__ == '__main__':
    LAYER = 5
    DPI = 500

    reg   = Registrar()
    b_oct = reg.domain('b_oct')
    g_gcd = reg.domain('g_gcd')
    c_oct = reg.domain('c_oct')
    c_ell = reg.domain('c_ell')
    p_pix = PlatePixelCarree(reg)

    pil_img = Image.open('src/bm_3600x1800.png').convert('RGBA')
    img     = np.array(pil_img)
    ph, pw  = img.shape[:2]
    pc_px   = p_pix.adopt(img, extent=(-180.0, -90.0, 180.0, 90.0), y_up=True, center=True)
    image   = p_pix.image(pc_px)

    mesh = HexMesh.create(LAYER, reg)

    # L0 densified with L3 verts: delta=3, factor=27, 28 verts per edge
    # for lev in range(LAYER):
    #     d0 = mesh.densify(lev)
    #     print(f'densify({lev}) shape: {d0.shape}')
    #     print(f'  expected: ({12*(9**lev)})')

    # mesh     = HexMesh.create(LAYER, reg)                    # shared-vertex mesh in b_oct
    ll_pts   = reg.project(mesh.pts, [b_oct, g_gcd])       # project unique verts once
    ll_hexes = ll_pts.coords[mesh.faces]                   # (12*9^L, 6, 2) g_gcd
    ok       = valid_ll(ll_hexes)                          # drop antimeridian-crossing hexes
    p_pts    = reg.project(ll_pts, [g_gcd, p_pix])         # project unique verts to pixel
    full_hexes = p_pts.coords[mesh.faces[ok]]              # (N_ok, 6, 2)

    print(f'b_oct native L{LAYER}: {ll_hexes.shape[0]} total, {ok.sum()} displayed')

    fig = plt.figure(figsize=(pw / 100, ph / 100), dpi=DPI, frameon=False)
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(-180.0, +180.0)
    ax.imshow(image, alpha=1.0, extent=[-180.0, +180.0, -90.0, +90.0],
              origin='upper', aspect='auto')
    ax.set_ylim(-90.0, +90.0)
    ax.add_collection(PolyCollection(
        full_hexes, facecolors='none',
        edgecolors=[(0.95, 0.95, 0.2, 1.0)], linewidth=0.5,
    ))
    f_name = f'output/ex0115_flat_l{LAYER}.png'
    fig.savefig(f_name, dpi=DPI)
    plt.close(fig)
    print(f'  saved {f_name}')