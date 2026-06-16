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
This processes about 7k points / second at full accuracy - 3600x1800 cache_building will be about 10 minutes
16 Jun 2026 0.1.3a0 (passed) 35.2s cache used.
16 Jun 2026 0.1.3a0 (passed) 40.9s (accelerated).
13 Mar 2026 0.1.1a1 (passed)
28 Feb 2026 0.1.1a1 (passed)
26 Dec 2025 0.1.0a4 (fixed/passed)
16 Dec 2025 0.1.0a3 (passed)
25 Nov 2025 (passed)
"""
import numpy as np
from pathlib import Path
from matplotlib import image, pyplot as plt
from hhg9 import Registrar, Points
from hhg9.domains.octahedral_barycentric import WarpTolerance


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
    b_oct = reg.domain('b_oct')
    # b_oct.warp.tolerance = WarpTolerance.MACH

    # circumference of earth: 40075017m
    # 1 pix = 40075017/1350 m = about 30km
    cache = Path('src/tissot_3600x1800.npz')
    img = image.imread(f'src/tissot_3600x1800.png', 'png')
    pc_extent = (-180.0, -90.0, 180.0, 90.0)
    pc_px = p_pix.adopt(img, extent=pc_extent, y_up=True, center=False)

    if not cache.exists() or force:
        import time
        # circumference of earth: 40075017m
        # 1 pix = 40075017/1350 m = about 30km
        # acc = 3e+4  # 23s for 30km, 60s for 1nm. 1e-9 is default
        # reg.projection('gcd_bry').set_accuracy(acc)
        start_time = time.perf_counter()
        bry = reg.project(pc_px, ['p_pix', 'g_gcd', 'b_oct'])
        seconds = time.perf_counter() - start_time
        print(f'{seconds:.6f} seconds to process {len(bry)} points.')
        np.savez(
            cache,
            cmp=bry.oid,  # octant-identity (3,)
            coords=bry.coords,
            smp=bry.samples
        )

    repo = np.load(cache, allow_pickle=True)
    c_pix = Points(repo['coords'], b_oct, oid=repo['cmp'], samples=repo['smp'])
    o_pix = reg.project(c_pix, ['b_oct', 'c_oct'])
    show_octahedron(o_pix)


if __name__ == '__main__':
    run(force=True)
