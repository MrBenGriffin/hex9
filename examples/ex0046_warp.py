# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0


# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Display warp matrix
28 February 2026 0.1.1a1 (passed)

"""
import numpy as np
from matplotlib import image, pyplot as plt
from hhg9 import Registrar, Points
from hhg9.h9 import H9K


def run():
    """
    Load a photo, adopt into PlatePixel points, transform to XYZ via GCD.
    Then display it as the unit sphere.
    """

    reg = Registrar()  # Manage Domains & Projections
    b_oct = reg.domain('b_oct')  # Barycentric Octahedral
    src = b_oct.warp.src
    dst = b_oct.warp.dst

    sx, sy = src[:, 0], src[:, 1]
    dx, dy = dst[:, 0], dst[:, 1]
    fig = plt.figure(figsize=(10, 10), dpi=200, frameon=False)
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
    ax = fig.add_axes([0, 0, 1, 1])  # span the whole figure
    ax.set_aspect('equal', adjustable='box')
    ax.scatter(sx, sy, marker='.', ec='none', s=12, color='green', alpha=0.5)
    ax.scatter(dx, dy, marker='.', ec='none', s=12, color='blue', alpha=0.25)
    ax.set_axis_off()
    fig.savefig(f"output/ex0046_warp.png", dpi=200)
    print(f'fig saved at output/ex0046_warp.png')


if __name__ == '__main__':
    run()
