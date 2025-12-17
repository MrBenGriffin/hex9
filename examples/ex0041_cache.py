# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Uses H9 LUTS only (for inverse AK projection beam root-finding)
This loads a Plate Carrée png image,
converts the entire image as coordinates into XYZ Octagon,
Normally we sample the image via the forward projection, instead of converting from geodesic.
The ellipsoid -> octahedral projection is slow, and depends upon using H9 for root-finding,
so here we use a cache. Recent work on vectorising the forward projection has improved timings,
but root-finding is inherently slow.
This takes about 25s to project 1e+6 points at 3e+4m accuracy.
16 December 2025 0.1.0a3 (passed)
25 November 2025 (passed)
"""
import numpy as np
from pathlib import Path
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
    # plt.show()
    fig.savefig(f"output/ex0041.png", dpi=200)
    print(f'fig saved at output/ex0041.png')


def run(*, force=False):
    """
    Load a photo, adopt into Plate Carrée points,
    project to Octahedral via GCD->Barycentric.
    Then display it as the unit octahedron.
    """
    reg = Registrar()  # Manage Domains & Projections
    # Domains - 2D image and GCD Spherical.
    p_pix = reg.domain('p_pix')           # Pixel Plate Carrée
    c_oct = reg.domain('c_oct')

    # circumference of earth: 40075017m
    # 1 pix = 40075017/1350 m = about 30km
    cache = Path('src/world1350x675.npy')
    img = image.imread(f'src/world1350x675.png', 'png')
    pc_pix = p_pix.adopt(img)

    if not cache.exists() or force:
        import time
        # circumference of earth: 40075017m
        # 1 pix = 40075017/1350 m = about 30km
        acc = 3e+4  # 23s for 30km, 60s for 1nm. 1e-9 is default
        reg.projection('gcd_bry').set_accuracy(acc)

        gcd = reg.project(pc_pix, ['p_pix', 'g_gcd'])
        start_time = time.perf_counter()
        bry = reg.project(gcd, ['g_gcd', 'b_oct'])
        seconds = time.perf_counter() - start_time
        print(f'{seconds:.6f} seconds to process {len(gcd)} points at {acc}m accuracy.')
        oc_xyz = reg.project(bry, ['b_oct', 'c_oct'])
        np.save(cache, oc_xyz.coords)

    oc = np.load(cache)
    c_pix = c_oct.adopt(oc)
    c_pix.samples = pc_pix.samples
    show_octahedron(c_pix)


if __name__ == '__main__':
    run(force=True)
