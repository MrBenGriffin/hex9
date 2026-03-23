# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
octahedral to octahedron net.
This loads a Plate Carrée png image,
loads the octahedral cache and
displays it as an octahedron net, under different layouts.
Normally I do not project maps this way, but use grid sampling instead.
Because without sampling, one gets a bit patchy.
And also the ECEF=>COCT=>BOCT is slow. (upstream)
13 Mar 2026 0.1.1a1 (passed)
28 Feb 2026 0.1.1a1 (passed)
26 Dec 2025 0.1.0a4 (passed)
16 Dec 2025 0.1.0a3 (passed)
25 Nov 2025 (passed)
"""
from pathlib import Path
import numpy as np
from matplotlib import image, pyplot as plt
from hhg9 import Registrar, Points
from hhg9.domains import NetPixel


def load_cache(reg, pc_map: str = 'tissot_3600x1800'):
    """Load cache from ex_0041"""
    cache = Path(f'src/{pc_map}.npz')
    if not cache.exists():
        raise ValueError('You need to generate the map cache. See ex_0041_cache')
    b_oct = reg.domain('b_oct')
    repo = np.load(cache, allow_pickle=True)
    b_pix = Points(repo['coords'], b_oct, oid=repo['cmp'], samples=repo['smp'])
    return b_pix


def show_pts_2d(arr: Points, w=360, h=180, bounds=None, name='base'):
    """Display points in 2d image"""
    xx, yy = arr.coords[:, 0], arr.coords[:, 1]
    cols = arr.samples
    fw = w/100
    fh = h/100
    fig = plt.figure(figsize=(fw, fh), dpi=100, frameon=False)
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
    ax = fig.add_axes([0, 0, 1, 1])  # span the whole figure
    ax.set_aspect('equal', adjustable='box')
    if bounds is not None:
        xmin, xmax, ymin, ymax = bounds
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    ax.set_axis_off()
    ax.scatter(xx, yy, marker='.', ec='none', s=1.0, alpha=1.0, c=cols)
    fig.savefig(f'output/ex0045_{name}.png', dpi=100, format='png', transparent=True)
    plt.close(fig)


def run():
    """
    Load a photo, adopt into PlatePixel points, transform to NetValue via GCD.
    Then display it as various nets.
    """
    reg = Registrar()  # Manage Domains & Projections
    bb_px = load_cache(reg, 'tissot_3600x1800')  # 'world3600x1800' on octahedral, with samples.
    tpx = 900  # pixels in 90 degrees:  90*3600/360

    for layout in ['butterfly', 'diamonds', 'mortar']:  # 'diamonds', 'mortar', 'butterfly', 'turbine'
        n_oct = reg.domain(f'n_oct:{layout}')
        n_pix = NetPixel(reg, n_oct)
        wid, hgt = n_pix.image_dims(tpx)
        bounds = (0, n_oct.wi, 0, n_oct.he)
        o_net = reg.project(bb_px, ['b_oct', n_oct])
        show_pts_2d(o_net, w=wid, h=hgt, bounds=bounds, name=layout)  # net.


if __name__ == '__main__':
    run()
