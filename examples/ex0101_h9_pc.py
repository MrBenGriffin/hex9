# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
This shows the hexagon ids as colours, for each layer. Note that
layer zero has 12 hexagons, whereas every other layer has 9.

Last Tested
16 Jun 2026 0.1.3a0 (passed) 17.5s
"""
from pickletools import uint8

import numpy as np
from matplotlib import pyplot as plt
from hhg9 import Registrar, Points
from PIL import Image  # image saving
from hhg9.formats import OctahedralH9
from hhg9.h9.addressing import TailStyle, hex_str_encode
from matplotlib import image, pyplot as plt
from hhg9.h9.uuid_address import h9_bin_pts


def run(base):
    """Runner"""
    reg = Registrar()  # Manage Domains & Projections
    p_pix = reg.domain('p_pix')  # PlatePixel
    g_gcd = reg.domain('g_gcd')  # lat/lon
    b_oct = reg.domain('b_oct')  # lat/lon

    cmap = plt.colormaps.get_cmap("tab20")
    cols = cmap(np.linspace(0, 1, 14))
    cols[13] = np.array([1, 1, 1, 1])         # transparent/illegal fallback
    p_cols = np.array(np.array(cols) * 255, dtype=np.uint8)  # *255 (256 wraps 1.0->0)

    img = image.imread(base)
    pc_extent = (-180.0, -90.0, 180.0, 90.0)
    pc_px = p_pix.adopt(img, extent=pc_extent, y_up=True, center=False)
    pc_h9 = reg.project(pc_px, [p_pix, g_gcd, b_oct])
    h9_u = h9_bin_pts(pc_h9, 4)
    unique_values, inverse_indices = np.unique(np.array(h9_u), return_inverse=True)
    cols = p_cols[1 + (inverse_indices % 9)]            # one RGBA per pixel
    # Paint the per-pixel hex-id colours back into an [y,x,4] raster.
    pc_col = Points(pc_px.coords, p_pix, samples=cols)
    img = p_pix.image(pc_col)    # convert coloured points back to an [y,x,4] image.
    y, x, channels = img.shape
    fig = plt.figure(figsize=(x/100, y/100), dpi=100, frameon=False)
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    ax.imshow(img)
    fig.savefig(f"output/ex0101pc.png", dpi=100)
    plt.close(fig)
    print("fig saved at output/ex0101pc.png")


if __name__ == "__main__":
    base = 'src/tissot_3600x1800.png'
    run(base)
