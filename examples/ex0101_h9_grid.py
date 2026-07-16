# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
This shows the hexagon ids as colours, for each layer. Note that
layer zero has 12 hexagons, whereas every other layer has 9.

Last Tested
16 Jun 2026 0.1.3a0 (passed) 17.5s
13 Mar 2026 0.1.1a1 (passed)
06 Mar 2026 0.1.1a1 (passed - fixed)
26 Dec 2025 0.1.0a4 (passed)
16 Dec 2025 0.1.0a3 (passed - and written better)
08 Oct 2025 (passed)
"""
import numpy as np
from matplotlib import pyplot as plt
from hhg9 import Registrar, Points
from PIL import Image  # image saving
from hhg9.formats import OctahedralH9
from hhg9.h9.addressing import TailStyle, hex_str_encode


def run(layout, scale, depths):
    """Runner"""
    cmap = plt.colormaps.get_cmap("tab20")
    cols = cmap(np.linspace(0, 1, 14))
    cols[13] = np.array([1, 1, 1, 1])         # transparent/illegal fallback

    reg = Registrar()                        # Manage Domains & Projections
    b_oct = reg.domain("b_oct")
    n_oct = reg.domain(f"n_oct:{layout}")   # 'mortar', 'butterfly', etc.

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
        adr = hex_str_encode(ref, layer=layer, tail_style=TailStyle.key)
        idt = np.array([int(a[layer:layer+1], 16) for a in adr], dtype=np.uint8)
        rgba = cols[idt] * 255

        # Colour each unique hex with a repeatable palette index
        out = np.ones((pix_h, pix_w, 4), dtype=float)
        out[py, px] = rgba
        f_name = f"output/ex0101_{layout}_L{layer}.png"
        Image.fromarray(out.astype(np.uint8)).save(f_name)
        print(f"Saved {f_name}")


if __name__ == "__main__":
    from hhg9.domains.nets import net_layouts
    for layout in ['turbine']:
    # for layout in net_layouts:
        run(layout, 1200, depths=[0])
