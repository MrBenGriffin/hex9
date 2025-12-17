# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
octahedral to octahedron net.
This loads a Plate Carrée png image,
loads the octahedral cache and
displays it as an octahedron net, under different layouts.
Normally I do not project maps this way, but use grid sampling instead.
It's still relevant for managing datasets captured over GCD.
16 December 2025 0.1.0a3 (passed)
25 November 2025 (passed)
"""
from pathlib import Path
import numpy as np
from matplotlib import image, pyplot as plt
from hhg9 import Registrar, Points


def load_cache(reg, pc_map: str = 'tissot_2560x1280'):
    """Load cache from ex_0041"""
    cache = Path(f'src/{pc_map}.npy')
    p_pix = reg.domain('p_pix')
    c_oct = reg.domain('c_oct')
    if not cache.exists():
        raise ValueError('You need to generate the map cache. See ex_0041_cache')
    oc_px = Points(np.load(cache), c_oct)
    pc_img = image.imread(f'src/{pc_map}.png', 'png')
    pc_pix = p_pix.adopt(pc_img)
    oc_px.samples = pc_pix.samples
    return oc_px


def show_pts_2d(arr: Points, ratio=None, name='base'):
    """Display points in 2d image"""
    xx, yy = arr.coords[:, 0], arr.coords[:, 1]
    cols = arr.samples
    if ratio is not None:
        fig_x = 16
        fig_y = fig_x/ratio
        fig = plt.figure(figsize=(fig_x, fig_y), dpi=150, frameon=False)
    else:
        fig = plt.figure(figsize=(10, 10), dpi=150, frameon=False)
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
    ax = fig.add_subplot(111)
    ax.scatter(xx, yy, marker='.', ec='none', s=2.5, alpha=0.95, c=cols)
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()
    plt.box(False)
    fig.savefig(f'output/ex0045_{name}.png',
                dpi=300, format='png', bbox_inches='tight',
                pad_inches=0, transparent=True)
    plt.close(fig)


def run():
    """
    Load a photo, adopt into PlatePixel points, transform to NetValue via GCD.
    Then display it as various nets.
    """
    reg = Registrar()  # Manage Domains & Projections
    oc_px = load_cache(reg, 'tissot_2560x1280')  # 'world1350x675' on octahedral, with samples.
    bb_px = reg.project(oc_px, ['c_oct', 'b_oct'])
    for layout in ['butterfly']:  # 'diamonds', 'mortar', 'butterfly', 'turbine'
        n_oct = reg.domain(f'n_oct:{layout}')
        o_net = reg.project(bb_px, ['b_oct', n_oct])
        show_pts_2d(o_net, ratio=n_oct.ratio(), name=layout)  # net.


if __name__ == '__main__':
    run()
