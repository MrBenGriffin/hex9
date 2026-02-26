# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
For a given layer, compose a reference address for each
hexagon in that layer - generate the set of hexagons at that layer
and display on the globe.

Last Tested
26 December 2025 0.1.0a4 (passed)
16 December 2025 0.1.0a3 (passed)
25 November 2025 (passed)
"""
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt, colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import CloughTocher2DInterpolator

from hhg9 import Registrar, Points
from hhg9.algorithms.distance import wgs84_area
from hhg9.h9 import H9_RA, H9O
from hhg9.h9.region import regions_xy
from hhg9.h9.polygon import hex_poly_layer
import matplotlib as mpl


def chain_generator(initial_seed, depth, props=H9_RA.props, modes=H9_RA.modes):
    """Generator for comprehensive region chain generation"""
    def _recurse(current_chain):  # Recursive Closure
        if len(current_chain) - 2 == depth:  # Stop condition
            yield current_chain
            return
        seed = current_chain[-1]  # Get the current seed (last element)
        children = props[modes[seed]].flatten()
        for child in children:  # Iterate and dive deeper
            yield from _recurse(current_chain + [child])  # Create new list
    yield from _recurse([initial_seed])  # yield from closure.


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


def snow_globe(arr: Points, poly_len: int = 6, scores=None, layers='x', base=1.0):
    """Display a 3D point cloud using matplotlib"""
    mpl.rcParams['path.simplify'] = False
    fig = plt.figure(figsize=(15, 15), dpi=200, frameon=False)
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=30, azim=40)
    axis = mplot_ax_vector(ax)
    all_polys = arr.coords.reshape(-1, poly_len, 3)
    mask = cull_backface(all_polys, axis)
    front = all_polys[mask]
    abs_val = np.abs(scores)
    max_abs = float(np.max(abs_val))
    norm = colors.TwoSlopeNorm(vcenter=0.0, vmin=-max_abs, vmax=+max_abs)
    cmap_name = 'RdBu_r'
    sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=norm)
    sm.set_array([])
    rgba, _norm = rgba_from(scores, cmap_name, norm=norm)
    ax.set_proj_type('ortho')  # FOV = 0 deg
    if True:
        ax.set_xlim(-4e+6, 4e+6)  # fill the area with the map.
        ax.set_ylim(-4e+6, 4e+6)
        ax.set_zlim(-4e+6, 4e+6)
    polys = [p for p in front]

    collection = Poly3DCollection(polys, ec='black', facecolors=rgba[mask], alpha=0.9, linewidth=0.02)
    ax.add_collection(collection)
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()
    mae = np.mean(np.abs(base - 1.0))
    authalic_p99 = np.quantile(np.abs(base - 1.0), 0.99)
    ax.title.set_text(f'mae:{mae}, p99:{authalic_p99:.4f}')
    plt.tight_layout()
    fig.savefig(f"output/ex0011w_{layers}.png", dpi=400)
    print(f'fig saved at output/ex0011w_{layers}.png')


def get_data(reg: Registrar, depth, mode=None):
    """Load up global sample data"""
    # grab generation for given depth
    b_oct = reg.domain('b_oct')
    all_rgn = [   # these are 0..11
        list(chain_generator(H9_RA.proto[0], depth)),
        list(chain_generator(H9_RA.proto[1], depth))
    ]
    rgn = H9_RA.rid2cell[np.array(all_rgn)]  # cell addresses.
    sides = []
    for oc in range(8):  # all octants
        mo = H9O.oid_mo[oc]
        cmp = H9O.oid_cmp[oc]
        if mode is not None and mo != mode:
            continue
        rgc = rgn[mo]
        xym = regions_xy(rgc)
        xy = xym[:, :-1]
        sides.append(Points(xy, b_oct, cmp))
    result = Points.concat(sides)
    return result


def load_grid(layer: int):
    """
    Cached loader for the grid NPZ for a given hex_layer.
    This avoids re-reading the same file from disk when `run` is called
    repeatedly for the same (hex_layer,e) pair.
    """
    f_name = Path(f"grid_l{layer}.npz")
    repo = np.load(f_name, allow_pickle=True)
    cmp = repo['cmp']
    xy_vert = repo['xy_vert']
    v_ell = repo['v_ell']
    oc_vtx = repo['oc_vtx']
    oc_edg = repo['oc_edg']
    grid = repo['grid']
    locs = repo['locs'] if 'locs' in repo.files else None
    return cmp, xy_vert, v_ell, oc_vtx, oc_edg, grid, locs


def hexify(reg: Registrar, b_pts: Points, warp=None, layers: int = 4):
    """
    Find hexagons for data, and display on a 'globe'.
    """
    pts, pops = hex_poly_layer(b_pts, layers)

    # Now calculate their area as a metric. (ignore pops).
    h_pts = pts.copy()
    oc, mo = h_pts.cm()

    b_wrp = warp(h_pts, mo)
    h_pts.coords = b_wrp

    c_pts = reg.project(h_pts, ['b_oct', 'c_oct', 'c_ell'])  # use bary.
    g_pts = reg.project(h_pts, ['b_oct', 'g_gcd'])

    w_area_m2 = wgs84_area(reg, g_pts)  # default value is 6
    gm2 = 510_065_621_724_154.6   # total surface area of WGS-84 (m²)
    bins = 12*9**layers           # number of hexes at this layer
    w_area_m2_ideal = gm2/bins    # ideal equal-area per hex
    w_adj = np.abs(w_area_m2 / w_area_m2_ideal)
    score = np.log(w_adj + 1e-16)  # authalic log-density ℓ
    mae = np.mean(w_adj - 1)
    np.mean(np.abs(w_adj - 1))

    snow_globe(c_pts, 6, score, f'{layers}', base=w_adj)


class AuthalicWarp:
    """
    Generate a warp prior to AK
    """
    def __init__(self, file_name='output/data/curr_fb_iter19_0010131.npz'):
        """
        Builds a C1-continuous warp field from a file.

        Parameters:
        source_pts (Nx2): The perfect, regular grid (a_p)
        target_pts (Nx2): The Sinkhorn-optimized grid (x_prime)
        """
        # f_name = 'output/data/curr_fb_iter19_0010131.npz'
        repo = np.load(file_name, allow_pickle=True)
        # a_p = repo['a_p']  # original value
        # t_p = repo['t_p']  # target value.
        cmp, xy_vert, v_ell, oc_vtx, oc_edg, t_grid, locs = load_grid(layer=4)
        a_p = xy_vert
        t_p = repo['grid']
        # mae = repo['mae']
        # a_p = repo['source_pts']
        # t_p = repo['target_pts']

        # print(f"Building Clough-Tocher Interpolator ({len(t_p)} points)...")
        # We build two separate interpolators: one for X-shift, one for Y-shift.
        # This is often more robust than interpolating the absolute position directly.
        diff = t_p - a_p
        self.dx_interp = CloughTocher2DInterpolator(a_p, diff[:, 0])
        self.dy_interp = CloughTocher2DInterpolator(a_p, diff[:, 1])
        # print("Warp Ready.")

    def __call__(self, b_pts: Points, mode=0):
        """
        Warps an array of points (Mx2).
        Returns mapped points (Mx2).
        """
        xy = np.asarray(b_pts.coords)
        mode = np.asarray(mode)
        m1 = mode == 1
        xy[m1, 1] *= -1
        # if xy.ndim == 1: xy = xy[None, :]  # Handle single point

        # 1. Predict Displacement
        dx = self.dx_interp(xy)
        dy = self.dy_interp(xy)

        # 2. Handle 'Out of Bounds' (NaNs)
        mask_nan = np.isnan(dx) | np.isnan(dy)
        if np.any(mask_nan):
            # print(f"Warning: {np.sum(mask_nan)} points outside warp domain.")
            dx[mask_nan] = 0.0
            dy[mask_nan] = 0.0

        # 3. Apply
        wxy = xy + np.stack([dx, dy], axis=1)
        wxy[m1, 1] *= -1
        return wxy


if __name__ == '__main__':
    depth = 4  # 0,...5 √
    # better_fb_iter24_0480000.npz
    # model_f = 'output/data/best_l3_g5afl_iter99.npz'
    model_f = 'output/data/L4_Final_Ironed.npz'
    # model_f = 'output/q_l4_iter0.npz'
    warp = AuthalicWarp(model_f)
    rg = Registrar()  # Manage Domains & Projections
    data = get_data(rg, depth)  # should be 8*9**depth  (eg, depth=0: 72 points, 9 points on each face, and six points in each hexagon)
    hexify(rg, data, warp, layers=depth)


