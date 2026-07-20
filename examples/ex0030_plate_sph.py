# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Supporting System Test - No octahedral projection or H9
This follows ex0020_plate_glb.py (which loaded the png and displayed it on a globe).
Converts PlateCarree to ECEF via GCD and displays it
16 Jun 2026 0.1.3a0 (passed) 27.6s
13 Mar 2026 0.1.1a1 (passed)
28 Feb 2026 0.1.1a1 (passed)
26 Dec 2025 0.1.0a4 (passed)
16 Dec 2025 0.1.0a3 (passed)
25 Nov 2025 (passed)
"""
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from hhg9 import Registrar, Points


def show_globe(arr: Points, x_lim=None, y_lim=None, z_lim=None, label=None, clip=False):
    """Display a 3D point cloud using matplotlib"""
    xx, yy, zz = arr.coords[:, 0], arr.coords[:, 1], arr.coords[:, 2]
    cols = arr.samples.astype(np.float64)/255
    fig = plt.figure(figsize=(10, 10), dpi=200, frameon=False)
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_proj_type('ortho')  # FOV = 0 deg
    ax.set_xlim(-4e+6, 4e+6)  # fill the area with the map.
    ax.set_ylim(-4e+6, 4e+6)
    ax.set_zlim(-4e+6, 4e+6)
    ax.scatter(xx, yy, zz, marker=',', ec='none', s=2.5, c=cols)
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()
    fig.savefig(f"output/ex0030_global.png", dpi=100)
    print(f'fig saved at output/ex0030_global.png')


def run():
    """
    Load a photo, adopt into PlatePixel points, transform to XYZ via GCD.
    Then display it as the unit sphere.
    """
    reg = Registrar()  # Manage Domains & Projections
    p_pix = reg.domain('p_pix')  # PlatePixel


    # Load as uint8 RGB. matplotlib.imread yields uint8 0..255 for JPEG but
    # float32 0..1 for PNG, and show_globe scales by /255 — so read via PIL to
    # get one predictable dtype regardless of the source format.
    img = np.array(Image.open('src/bm_3600x1800.png').convert('RGB'))
    pc_extent = (-180.0, -90.0, 180.0, 90.0)
    pc_px = p_pix.adopt(img, extent=pc_extent, y_up=True, center=False)

    el_xyz = reg.project(pc_px, ['p_pix', 'g_gcd', 'c_ell'])  # project it onto ECEF
    sp_pl = reg.project(el_xyz, ['c_ell', 'g_gcd', 'p_pix'])
    show_globe(el_xyz)

    rmg = p_pix.image(sp_pl, (3600, 1800))     # convert points back to an [1800,3600,3] image.
    fig = plt.figure(figsize=(36, 18), dpi=100, frameon=False)
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
    plt.imshow(rmg)
    fig.savefig(f"output/ex0030_flat.png", dpi=100)
    print(f'fig saved at output/ex0030_flat.png')


if __name__ == '__main__':
    run()
