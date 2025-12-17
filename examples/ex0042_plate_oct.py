# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Uses AK projection, so H9 indirectly
This follows ex0030_plate_glb.py (which loaded the png and displayed it as a unit ellipsoid).
This loads a Plate Carrée png image, converts it to Unit Octahedron, via latitude/longitude and sphere and displays it.
As the inverse of the AK projection is slow we use the cache from ex0041_cache.py
Added chain registration example.
16 December 2025 0.1.0a3 (passed)
25 November 2025 (passed)
"""
from pathlib import Path
import numpy as np
from matplotlib import image, pyplot as plt
from hhg9 import Registrar, Points


def show_octahedron(arr: Points, x_lim=None, y_lim=None, z_lim=None, label=None, clip=False):
    """Display a 3D point cloud using matplotlib"""
    xx, yy, zz = arr.coords[:, 0], arr.coords[:, 1], arr.coords[:, 2]
    cols = arr.samples
    fig = plt.figure(figsize=(10, 10), dpi=200, frameon=False)
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_proj_type('ortho')  # FOV = 0 deg
    ax.set_xlim(-0.52, 0.52)  # fill the area with the map.
    ax.set_ylim(-0.52, 0.52)
    ax.set_zlim(-0.52, 0.52)
    ax.scatter(xx, yy, zz, marker=',', ec='none', s=2.5, c=cols)
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()
    fig.savefig(f"output/ex0042_octahedron.png", dpi=200)
    print(f'fig saved at output/ex0042_octahedron.png')


def run():
    """
        Load a Plate Carrée png image, project onto Unit Octahedron, and display.
    """
    reg = Registrar()  # Manage Domains & Projections
    p_pix = reg.domain('p_pix')           # Pixel Plate Carrée
    c_oct = reg.domain('c_oct')

    # Load cached octahedral and colours from original.
    pc_map = 'world1350x675'
    cache = Path(f'src/{pc_map}.npy')
    if not cache.exists():
        raise ValueError('Far better to load a cache. See ex_0041')

    img = image.imread(f'src/{pc_map}.png', 'png')
    oc = np.load(cache)
    c_pix = Points(oc, c_oct, samples=img.reshape(-1, 4))
    show_octahedron(c_pix)

    # Example Registration of a specific chain.
    # We don't need to do this, we can be explicit in the projection.
    # But it allows us to short-cut end-points.
    reg.register_projection('chain', ['c_oct', 'c_ell', 'g_gcd', 'p_pix'])

    # Now project back to Plate Carrée and display.
    sp_pl = reg.project(c_pix, [c_oct, p_pix])  # can use endpoints from chain.
    rmg = p_pix.image(sp_pl, (1350, 675))     # convert points back to an [1350,675,3] image.
    fig = plt.figure(figsize=(18, 9), dpi=100, frameon=False)
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
    plt.imshow(rmg)
    fig.savefig(f"output/ex0042_recovered.png", dpi=100)
    plt.close(fig)
    print(f'fig saved at output/ex0042_recovered.png')


if __name__ == '__main__':
    run()
