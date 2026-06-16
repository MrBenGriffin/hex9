# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Uses AK projection, so H9 indirectly
This follows ex0030_plate_glb.py (which loaded the png and displayed it as a unit ellipsoid).
This loads a Plate Carrée png image, converts it to Unit Octahedron, via latitude/longitude and sphere and displays it.
As the inverse of the AK projection is slow we use the cache from ex0041_cache.py
Added chain registration example.
16 Jun 2026 0.1.3a0 (passed) 40.0s
13 Mar 2026 0.1.1a1 (passed)
28 Feb 2026 0.1.1a1 (passed)
26 Dec 2025 0.1.0a4 (passed)
16 Dec 2025 0.1.0a3 (passed)
25 Nov 2025 (passed)
"""
from pathlib import Path
import numpy as np
from PIL import Image
from matplotlib import image, pyplot as plt
from hhg9 import Registrar, Points
from hhg9.domains import PlatePixelCarree


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
    b_oct = reg.domain('b_oct')
    g_gcd = reg.domain('g_gcd')
    # Create the output raster FIRST so PlatePixelGCD captures it as rev_cs (extent-aware).
    fs = PlatePixelCarree.full_sphere(reg, 3600, 1800, center=False)

    # Load cached octahedral and colours from original.
    pc_map = 'tissot_3600x1800'
    cache = Path(f'src/{pc_map}.npz')
    if not cache.exists():
        raise ValueError('Far better to load a cache for this thing. See ex_0041')
    repo = np.load(cache, allow_pickle=True)
    b_pix = Points(repo['coords'], b_oct, oid=repo['cmp'], samples=repo['smp'])
    c_pix = reg.project(b_pix, [b_oct, 'c_oct'])
    show_octahedron(c_pix)

    # Now project back to Plate Carrée and display.
    sp_pl = reg.project(b_pix, [b_oct, g_gcd, fs])  # can use endpoints from chain.
    flat = fs.image(sp_pl)     # convert points back to an image.
    if flat.dtype != np.uint8:
        flat = (flat * 255).astype(np.uint8)
    img = Image.fromarray(flat)  # net_mode auto-detected from shape: (H,W,3)→RGB, (H,W,4)→RGBA
    img.save("output/ex0042_recovered.png")  # PNG keeps alpha
    print(f'fig saved at output/ex0042_recovered.png')


if __name__ == '__main__':
    run()
