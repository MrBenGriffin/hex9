# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
For a given layer, compose a reference address for each
hexagon in that layer - generate the set of hexagons at that layer
and display on the globe.

Last Tested
16 Jun 2026 0.1.3a0 (passed) 94.7
13 Mar 2026 0.1.1a1 (passed)
26 Dec 2025 0.1.0a4 (passed)
16 Dec 2025 0.1.0a3 (passed)
25 Nov 2025 (passed)
"""
import numpy as np
from matplotlib import pyplot as plt, colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from hhg9 import Registrar, Points
import matplotlib as mpl
from hhg9.h9 import H9K
from hhg9.h9.grid import HexMesh


def rgba_from(arr: np.ndarray, cmap_name: str = "plasma", norm=None, alpha: float = 1.0):
    """Return RGBA array from a 1D array of values.

    Parameters
    ----------
    arr : array-like
        Scalar values to map to colours.
    cmap_name : str
        Name of the Matplotlib colormap.
    norm : matplotlib.colors.Normalize or None
        Normalization object. If None, a simple Normalize based on arr
        is constructed.
    alpha : float
        Global alpha to apply to the colours.
    """
    arr = np.asarray(arr, dtype=float)
    if norm is None:
        norm = colors.Normalize(vmin=arr.min(), vmax=arr.max())

    base_cmap = plt.get_cmap(cmap_name)

    # If the colormap exposes a `.colors` table (ListedColormap), build a
    # new ListedColormap with an explicit alpha channel so we don't mutate
    # the global colormap in-place.
    if hasattr(base_cmap, "colors"):
        base_colors = np.asarray(base_cmap.colors)
        if base_colors.shape[1] == 3:
            # Append alpha channel
            alpha_col = np.full((base_colors.shape[0], 1), alpha, dtype=float)
            rgba_colors = np.concatenate([base_colors, alpha_col], axis=1)
        else:
            rgba_colors = base_colors.copy()
            rgba_colors[:, 3] = alpha
        cmap = colors.ListedColormap(rgba_colors, name=base_cmap.name + "_with_alpha")
    else:
        # For continuous maps, just use the base cmap and apply alpha after
        cmap = base_cmap

    rgba = cmap(norm(arr))

    # If the colormap didn't already encode alpha, enforce it here.
    if rgba.shape[1] == 4:
        rgba[:, 3] = alpha

    return rgba, norm


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


def draw_globe(poly_groups: Points, layers='x'):
    """Display a set of polygons (hexagons) across ECEF ellipsoid using matplotlib"""
    mpl.rcParams['path.simplify'] = False
    fig = plt.figure(figsize=(15, 15), dpi=200, frameon=False)
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    ax.view_init(elev=30, azim=40)
    ax.set_box_aspect([1, 1, 1], zoom=7.8)
    ax.set_proj_type('persp')  # FOV = 0 deg
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()
    axis = mplot_ax_vector(ax)
    for poly_list in poly_groups:
        pts = poly_list.reshape(-1, 3)
        x, y,  z = pts[:, 0], pts[:, 1], pts[:, 2]
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
        ax.set_zlim(z.min(), z.max())
        mask = cull_backface(poly_list, axis)
        front = poly_list[mask]
        polys = [p for p in front]
        faces = [(0, 0, 0, 0)] * len(front)

        collection = Poly3DCollection(polys, ec='black', facecolors=faces, linewidth=0.1)
        ax.add_collection3d(collection)
    fig.savefig(f"output/ex0116_{layers}.png", dpi=300)
    print(f'fig saved at output/ex0116_{layers}.png')


if __name__ == '__main__':
    LAYER = 0  # 0,...5 √
    MAX_LAYER = 1
    reg = Registrar()  # Manage Domains & Projections
    mesh = HexMesh.create(range(MAX_LAYER), reg)
    layer_f = mesh.faces

    b_oct = reg.domain('b_oct')
    # U1 = H9K.lattice.U
    # V3 = 3.0 * H9K.lattice.V

    pts = mesh.pts
    c_pts = reg.project(pts, ['b_oct', 'c_oct', 'c_ell'])
    hf = c_pts.coords[layer_f]

    layers = []
    for i in [0,1,2,3,4,5]:
    # for i in range(MAX_LAYER):
        a = mesh.densify(i)
        print(f'Layer {i} has {len(a)} hexagons')
        layers.append(c_pts.coords[a])
    # layers.append(hf)

    draw_globe(layers, layers=f'{LAYER}_{MAX_LAYER}')


