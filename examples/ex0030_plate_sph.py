# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Supporting System Test - No octahedral projection or H9
This follows ex0020_plate_glb.py (which loaded the png and displayed it on a globe).
Converts PlateCarree to ECEF via GCD and displays it
16 December 2025 0.1.0a3 (passed)
25 November 2025 (passed)
"""
from matplotlib import image, pyplot as plt
from hhg9 import Registrar, Points


def show_globe(arr: Points, x_lim=None, y_lim=None, z_lim=None, label=None, clip=False):
    """Display a 3D point cloud using matplotlib"""
    xx, yy, zz = arr.coords[:, 0], arr.coords[:, 1], arr.coords[:, 2]
    cols = arr.samples
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

    img = image.imread(f'src/world1350x675.png', 'png')
    pc_px = p_pix.adopt(img)     # ps.img shape is [1350x675x4]

    el_xyz = reg.project(pc_px, ['p_pix', 'g_gcd', 'c_ell'])  # project it onto ECEF
    sp_pl = reg.project(el_xyz, ['c_ell', 'g_gcd', 'p_pix'])
    show_globe(el_xyz)

    rmg = p_pix.image(sp_pl, (1350, 675))     # convert points back to an [1280,2560,3] image.
    fig = plt.figure(figsize=(18, 9), dpi=100, frameon=False)
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
    plt.imshow(rmg)
    fig.savefig(f"output/ex0030_flat.png", dpi=100)
    print(f'fig saved at output/ex0030_flat.png')


if __name__ == '__main__':
    run()
