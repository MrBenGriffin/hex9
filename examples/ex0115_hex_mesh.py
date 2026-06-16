# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""
Direct hex mesh generation from H9 lattice LUTs.

Uses HexMesh.create() from hhg9.h9.grid: sc_mode=0 supercells only, boundary
hex exterior verts reflected y→-y into the adjacent face via H9O.oid_nb.
Each hex is generated exactly once; no seam artefacts, no duplication.
Do not try running the entire sphere on L8! It will flounder in memory.
It would work - but the result isn't useful.

Last Tested
-----------
16 Jun 2026 0.1.3a0 (passed) 393.3s
"""
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from hhg9 import Registrar, Points
from hhg9.h9.grid import HexMesh


def mplot_ax_vector(ax):
    """mplot3d uses azim around z and elev from xy-plane"""
    az = np.deg2rad(ax.azim)
    el = np.deg2rad(ax.elev)
    return np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])


def cull_backface(arr, axis):
    """back-face culling"""
    centroids = arr.mean(axis=1)
    sides = centroids @ axis
    return sides >= 0

def show_global(ctr, crop, poly):
    """Display GCD points on the globe using WGS84-scaled meters"""
    xyz = poly.reshape(-1, 3)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    fig = plt.figure(figsize=(12, 12), dpi=300)
    fig.patch.set_alpha(0)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_proj_type('ortho')  # FOV = 0 deg
    ax.set_aspect('equal', adjustable='box')
    lat, lon = ctr
    elev, azim = lat, lon - 90
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_box_aspect([1, 1, 1], zoom=1.5)
    axis = mplot_ax_vector(ax)
    mask = cull_backface(poly, axis)
    front_poly = poly[mask]

    if crop is None:
        x_mid, y_mid, z_mid = np.median(x), np.median(y), np.median(z)
        max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0
        ax.set_xlim(x_mid - max_range, x_mid + max_range)
        ax.set_ylim(y_mid - max_range, y_mid + max_range)
        ax.set_zlim(z_mid - max_range, z_mid + max_range)
    else:
        (x_min, y_min, z_min), (x_max, y_max, z_max) = crop
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)

    # 2. Add the polygons. facecolor='none' will cause trouble.
    poly_collection = Poly3DCollection(front_poly, ec='black', facecolor=(0, 0, 0, 0), linewidth=1.0)
    ax.add_collection(poly_collection)

    fig.savefig(f"output/ex0115_global.png",
                dpi=300,
                bbox_inches='tight',
                pad_inches=0,
                transparent=True)

    print(f'fig saved at output/ex0115_global.png')
    plt.close(fig)

def calc_crop(ctr, reg, deg):
    """Calculate the crop region in degrees"""
    g_gcd = reg.domain('g_gcd')
    c_ell = reg.domain('c_ell')
    lat0, lon0 = ctr
    d = deg
    n = 20
    # edges
    west = np.linspace(lat0 - d, lat0 + d, n)
    east = np.linspace(lat0 - d, lat0 + d, n)
    south = np.linspace(lon0 - d, lon0 + d, n)
    north = np.linspace(lon0 - d, lon0 + d, n)

    # build perimeter clockwise
    lats = np.concatenate([
        west,
        np.full(n, lat0 + d),
        east[::-1],
        np.full(n, lat0 - d),
    ])
    lons = np.concatenate([
        np.full(n, lon0 - d),
        north,
        np.full(n, lon0 + d),
        south[::-1],
    ])
    ll = Points(np.stack([lats, lons], axis=-1), g_gcd)
    crop_rect = reg.project(ll, [g_gcd, c_ell])
    return crop_rect.bbox()  # (x_min, y_min, z_min) (x_max, y_max, z_max)


if __name__ == '__main__':

    reg = Registrar()
    b_oct = reg.domain('b_oct')
    g_gcd = reg.domain('g_gcd')
    c_ell = reg.domain('c_ell')

    mesh = HexMesh.create([6], reg)
    # layer_0 = mesh.densify(0)
    layer_4 = mesh.densify(6)

    el_pts = reg.project(mesh.pts, [b_oct, g_gcd, c_ell])
    el_hexes = el_pts.coords[layer_4]
    ctr = (51.5074, -0.1278)
    crop = calc_crop(ctr, reg, 2.0)
    show_global(ctr, crop, el_hexes)
