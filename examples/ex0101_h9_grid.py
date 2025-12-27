# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
This shows the hexagon ids as colours, for each layer. Note that
layer zero has 11 colours, whereas every other layer has 9.

Last Tested
26 December 2025 0.1.0a4 (passed)
16 December 2025 0.1.0a3 (passed - and written better)
08 October 2025 (passed)
"""
import numpy as np
from matplotlib import pyplot as plt
from hhg9 import Registrar, Points
from PIL import Image  # image saving
from hhg9.formats import OctahedralH9
from hhg9.h9.addressing import hex_layer, hex_key, TailStyle, hex_str_encode


def run(layout, scale, depths):
    """Runner"""
    cmap = plt.colormaps.get_cmap('plasma')
    cols = cmap(np.linspace(0, 1, 16))
    cols[15] = np.array([1, 1, 1, 1])  # stick alpha=transparent onto illegal.

    reg = Registrar()                        # Manage Domains & Projections
    b_oct = reg.domain(f'b_oct')
    n_oct = reg.domain(f'n_oct:{layout}')   # 'mortar', 'butterfly', etc.

    h9f = OctahedralH9(reg)
    b_oct.register_format(h9f)

    pix_w, pix_h = n_oct.image_dims(pixels=scale)

    # full pixel grid
    xx, yy = np.meshgrid(np.arange(pix_w), np.arange(pix_h))
    px = xx.ravel()
    py = yy.ravel()

    # Use the actual pixel dimensions returned by `image_dims` (layout-specific), rather than
    # reconstructing floats from triangle counts; otherwise some layouts will stretch/crop.
    # Sample pixel centres.
    ux = n_oct.wi * ((px + 0.5) / pix_w)
    uy = n_oct.he * ((py + 0.5) / pix_h)
    xy = np.column_stack([ux, uy])

    # classify to faces
    signs = n_oct.pt_face(xy)  # (N,3) int8
    in_net = np.any(signs != 0, axis=1)

    # only keep pixels in the net
    xy = xy[in_net]
    px = px[in_net]
    py = py[in_net]
    signs = signs[in_net]

    pts = Points(xy, domain=n_oct, components=signs)

    bas = reg.project(pts, [n_oct, b_oct])  # Now move from net to bary.
    good = b_oct.valid(bas)
    ref = bas.select(good)   # Eliminate any outliers.
    px = px[good]
    py = py[good]
    for layer in depths:
        # h_val = hex_layer(ref, layer, tail_style=TailStyle.reversible)
        # h_key = hex_key(addr)
        adr = hex_str_encode(ref, layer=layer, tail_style=TailStyle.key)
        h_key = hex_layer(ref, layer=layer, tail_style=TailStyle.key)
        hex_k, idx, inv_hex = np.unique(h_key, axis=0, return_index=True, return_inverse=True)
        hex_num = hex_k.shape[0]
        # idx = range(12) if layer == 0 else range(9)
        # sum_wt = np.bincount(inv_hex, weights=ref.samples, minlength=hex_num)
        # pp_hx = np.bincount(inv_hex, minlength=hex_num)  # aka cnt

        # The following method uses the hex-ids themselves.
        # For layers 0..5 this should be fine.
        idt = h_key[:, layer]  # should be same as -1
        samples = cols[idt.astype(np.uint8)]
        rgba = samples if samples.shape[1] == 4 else np.hstack(
            (samples, np.ones((samples.shape[0], 1), dtype=samples.dtype))
        )
        out = np.ones((pix_h, pix_w, 4), dtype=float)
        out[pix_h - 1 - py, px] = rgba
        f_name = f'output/ex0101_{layout}_L{layer}.png'
        Image.fromarray((out * 255).astype(np.uint8)).save(f_name)
        print(f'Saved {f_name}')


if __name__ == '__main__':
    from hhg9.domains.nets import net_layouts
    for layout in net_layouts:
        run(layout, 500, depths=[0, 1, 2, 3])
