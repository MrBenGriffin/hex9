# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
This shows the hexagon ids as colours, for each layer. Note that
layer zero has 12 hexagons, whereas every other layer has 9.

Last Tested
13 Mar 2026 0.1.1a1 (passed)
06 Mar 2026 0.1.1a1 (passed - fixed)
26 Dec 2025 0.1.0a4 (passed)
16 Dec 2025 0.1.0a3 (passed - and written better)
08 Oct 2025 (passed)
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection

from hhg9 import Registrar, Points
from PIL import Image  # image saving
from hhg9.formats import OctahedralH9
from hhg9.h9.addressing import TailStyle, hex_str_encode
from hhg9.h9.binning import hex_reduce, hex_parents, ctr_from_pars
from hhg9.h9.grid import hex_verts_in_noct
from hhg9.h9.tail import tail_key_from_reversible, tail_unpack_reversible


def run(layout, scale, depths):
    """Runner"""
    cmap = plt.colormaps.get_cmap("plasma")
    cols = cmap(np.linspace(0, 1, 14))
    cols[13] = np.array([1, 1, 1, 1])         # transparent/illegal fallback

    reg = Registrar()                        # Manage Domains & Projections
    b_oct = reg.domain("b_oct")
    n_oct = reg.domain(f"n_oct:{layout}")   # 'mortar', 'butterfly', etc.
    c_oct = reg.domain("c_oct")

    h9f = OctahedralH9(reg)
    b_oct.register_format(h9f)

    pix_w, pix_h = n_oct.image_dims(pixels=scale)

    # Full pixel grid
    xx, yy = np.meshgrid(np.arange(pix_w), np.arange(pix_h))
    px = xx.ravel()
    py = yy.ravel()

    # Sample pixel centres using the actual returned dimensions
    ux = n_oct.wi * ((px + 0.5) / pix_w)
    uy = n_oct.he * ((py + 0.5) / pix_h)
    xy = np.column_stack([ux, uy])

    # Classify to faces
    oids = n_oct.pt_face(xy)  # (N,) uint8; OID_INVALID (255) for outside net
    in_net = oids != 255

    # Keep only pixels in the net
    xy = xy[in_net]
    px = px[in_net]
    py = py[in_net]
    oids = oids[in_net]

    pts = Points(xy, domain=n_oct, oid=oids)

    bas = reg.project(pts, [n_oct, b_oct])  # Move from net to barycentric
    good = b_oct.valid(bas)
    ref = bas.select(good)   # Eliminate outliers
    px = px[good]
    py = pix_h - 1 - py[good]

    for layer in depths:
        hex_num, hex_v_k, hex_inv, _ = hex_reduce(ref, layer)
        hex_par, hex_oid, scale = hex_parents(b_oct, hex_v_k, hex_num)
        xpm, xc2, _, _ = tail_unpack_reversible(hex_v_k[:, -1])

        verts_n2 = hex_verts_in_noct(hex_par, hex_oid, xpm, xc2, scale, n_oct)

        # Roundtrip test.
        verts_b = reg.project(verts_n2, [n_oct, b_oct])
        hexb_oids = verts_b.oid.reshape(-1, 6)
        hexb_vrts  = verts_b.coords.reshape(-1, 6, 2)
        for v, o in zip(hexb_vrts, hexb_oids):
            vc = [f'({x:.04f}, {y:.04f})' for x,y in v]
            print(', '.join(vc), o)
        verts_n = reg.project(verts_b, [b_oct, n_oct])

        hex_oids = verts_n.oid.reshape(-1, 6)
        mask = np.any(hex_oids == 255, axis=1)   # Find hexes extending outside the net.
        hex_vrts = verts_n.coords.reshape(-1, 6, 2)
        full_hexes = hex_vrts[~mask, ...]
        semi_hexes = hex_vrts[mask, :4, ...]

        hx_v = hex_v_k[~mask, ...]
        ff_v = hex_v_k[mask, ...]
        py_v = np.vstack([hx_v, ff_v])

        hx_ctrs_n = np.mean(full_hexes, axis=1)
        hh_ctrs_n = np.mean(semi_hexes, axis=1)

        adr = hex_str_encode(ref, layer=layer, tail_style=TailStyle.key)
        idt = np.array([int(a[layer:layer+1], 16) for a in adr], dtype=np.uint8)
        rgba = cols[idt] * 255

        # Colour each unique hex with a repeatable palette index
        out = np.ones((pix_h, pix_w, 4), dtype=float)
        out[py, px] = rgba

        fig = plt.figure(figsize=(pix_w/100, pix_h/100), dpi=100, frameon=False)
        fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)

        ax = fig.add_axes([0, 0, 1, 1])  # span the whole figure
        plt.imshow(out.astype(np.uint8),
                   extent=[0, n_oct.wi, 0, n_oct.he], alpha=0.3,
                   origin='upper', aspect='auto')

        collection = PolyCollection(full_hexes, facecolors='none', ec=(0, 0, 0, 0.4), linewidth=0.25, antialiaseds=True)
        ax.add_collection(collection)
        collection = PolyCollection(semi_hexes, facecolors='none', ec=(0, 0, 0, 0.5), linewidth=0.25, antialiaseds=True)
        ax.add_collection(collection)

        poly_ctrs = np.vstack([hx_ctrs_n, hh_ctrs_n])

        for ctr, lab in zip(poly_ctrs, py_v):
            hex_addr = lab
            label = f'{hex_addr[layer]}'
            ax.text(ctr[0], ctr[1], label,
                    fontsize=4*[6-layer], ha='center', va='center',
                    color='black', zorder=100, clip_on=True)

        f_name = f"output/ex0102_{layout}_L{layer}.png"
        fig.savefig(f_name, dpi=200)
        print(f'fig saved at {f_name}')


if __name__ == "__main__":
    from hhg9.domains.nets import net_layouts
    for layout in ['rhombus']:
    # for layout in net_layouts: range(5)
        run(layout, 1200, depths=[0])
