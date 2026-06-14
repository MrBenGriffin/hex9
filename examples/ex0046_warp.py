# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Display warp matrix
13 Mar 2026 0.1.1a1 (passed)
28 Feb 2026 0.1.1a1 (passed)

"""
import numpy as np
from matplotlib import image, pyplot as plt
from hhg9 import Registrar, Points
from hhg9.h9 import H9K


def run():
    """
        Display the b_oct warp matrix src/destination
    """
    reg = Registrar()  # Manage Domains & Projections
    b_oct = reg.domain('b_oct')  # Barycentric Octahedral
    src = b_oct.warp.src
    dst = b_oct.warp.dst

    sx, sy = src[:, 0], src[:, 1]
    dx, dy = dst[:, 0], dst[:, 1]
    fig = plt.figure(figsize=(12, 10), dpi=400, frameon=False)
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
    ax = fig.add_axes([0, 0, 1, 1])  # span the whole figure
    ax.set_aspect('equal', adjustable='box')
    # ax.scatter(sx, sy, marker='.', ec='none', s=2, color='green', alpha=0.5)
    ax.scatter(dx, dy, marker='.', ec='none', s=2, color='blue', alpha=0.8)
    ax.set_axis_off()
    fig.savefig(f"output/ex0046_warp.png", dpi=400)
    print(f'fig saved at output/ex0046_warp.png')
    d = ['x', 'y']
    z = ['src', 'dst']
    file = f"output/ex0046_warp.txt"
    with open(file, 'w') as f:
        for i, a in enumerate([src, dst]):
            f.write(f'\n\n# warp {z[i]}')
            for p in [0, 1]:
                f.write(f'# dim: {d[p]}')
                f.write('static const float h9_warp_{d[p]} = {\n')
                for v in a[:, p]:
                    f.write(f'{v:.12f}, ')
                    if v % 10 == 9:
                        f.write('\n')
                f.write('\n};\n')
        f.close()


if __name__ == '__main__':
    run()
