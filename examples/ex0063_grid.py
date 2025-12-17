# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
This is a warp testing render - which is currently a closed line of research.
It saves a colouring of the authalic score (deviation from mean) of an octant of the hex9 grid after
it has been projected onto WGS84.
All renders follow the triangular sub-grid.
Another way of calculating this has been used with hexagons in ex0080_authalics
16 December 2025 0.1.0a3 (passed)
25 November 2025 (passed)
"""
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt, colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from hhg9 import Registrar, Points
from hhg9.algorithms.distance import wgs84_area
# from experimental.algorithms.warp import Warper
from hhg9.h9.polygon import tri_grid
import matplotlib as mpl


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


def snow_globe(arr: Points, poly_len: int = 6, pop=None):
    """Display a 3D point cloud using matplotlib"""
    mpl.rcParams['path.simplify'] = False
    fig = plt.figure(figsize=(15, 15), dpi=400, frameon=False)
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_proj_type('ortho')  # FOV = 0 deg
    ax.view_init(elev=30, azim=40)
    axis = mplot_ax_vector(ax)
    all_polys = arr.coords.reshape(-1, poly_len, 3)
    mask = cull_backface(all_polys, axis)
    front = all_polys[mask]
    rx = front.reshape(-1, 3)
    x_min, x_max = rx[:, 0].min(), rx[:, 0].max()
    y_min, y_max = rx[:, 1].min(), rx[:, 1].max()
    z_min, z_max = rx[:, 2].min(), rx[:, 2].max()
    if True:
        ax.set_xlim(x_min, x_max)  # fill the area with the map.
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
    polys = [p for p in front]
    if pop is not None:
        authalic_error = np.mean(np.abs(pop))
        pops = pop[mask]
        # v_min = np.min(pops)
        # v_max = np.max(pops)
        # norm = colors.Normalize(vmin=v_min, vmax=v_max)
        # cmap = plt.get_cmap('RdBu_r')
        col_map_name = 'RdBu_r'
        max_abs = float(np.max(np.abs(pops)))
        norm = colors.TwoSlopeNorm(vcenter=0.0, vmin=-max_abs, vmax=+max_abs)
        sm = plt.cm.ScalarMappable(cmap=col_map_name, norm=norm)
        sm.set_array([])

        # Map authalicity values (pops) to colours using the symmetric TwoSlopeNorm
        cmap = mpl.colormaps[col_map_name]
        facecols = cmap(norm(pops))

        # Optional colourbar (uncomment if/when needed)
        plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)

        collection = Poly3DCollection(
            polys,
            ec=(0, 0, 0, 0.3),
            facecolors=facecols,
            alpha=1.0,
            linewidth=0.05,
        )
        ax.add_collection(collection)
        ax.title.set_text(f'Authalic Error: {authalic_error:.3f}')
    else:
        collection = Poly3DCollection(polys, ec='black', alpha=1.0, linewidth=0.05)
        ax.add_collection(collection)

    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(f"output/ex0063_o{octant}_l{depth}.png", dpi=400)
    plt.close(fig)
    print(f'file saved at output/ex0063_o{octant}_l{depth}.png')


def get_data(reg: Registrar, layer=3, octant_id=None):
    """Load up global sample data"""
    tg0 = tri_grid(layer, 0).reshape([-1, 2])  # triangle polygons.
    tg1 = tri_grid(layer, 1).reshape([-1, 2])  # triangle polygons.
    tgx = [tg0, tg1]
    b_oct = reg.domain('b_oct')
    if octant_id is None:
        repo = []
        for octant in b_oct.signs.keys():
            o_id = b_oct.sign_to_id[octant]
            mode = b_oct.oid_mo[o_id]
            repo.append(Points(tgx[mode].copy(), b_oct, components=octant))
        return Points.concat(repo)
    else:
        o_id = octant_id
        mode = b_oct.oid_mo[o_id]
        cmp = b_oct.signs_by_id[o_id]
        return Points(tgx[mode], b_oct, components=cmp)


def octa_layer_stats(l: int, r=6371000.0):
    """
    Return triangle/vertex/edge counts and safe spacing estimates for a global
    octahedral grid with layer indexing based on emergent hexagons.

    Parameters
    ----------
    l : int
        Layer index (0-based; L=0 → 72 triangles → 12 hexes)
    r : float
        Mean Earth radius in metres (default 6,371 km)

    Returns
    -------
    dict with keys:
        s        : subdivisions per octa edge (=3**(L+1))
        triangles: total number of triangular cells
        vertices : global vertex count
        edges    : global edge count
        d_geo    : great-circle edge spacing (m)
        d_chord  : equivalent chord spacing (m)
        tol_m    : recommended dedupe tolerance (m)
    """
    s = 3 ** (l + 1)  # splits per octa edge
    tri = 8 * s * s  # 8 faces * s²
    vert = 4 * s * s + 2  # from combinatorial derivation
    edge = 12 * s * s  # 3/2 * T, or 3V - 6
    d_geo = (np.pi * r) / (2 * s)  # ~90°/s great-circle
    d_chord = 2 * r * np.sin((np.pi / (4 * s)))  # exact chord
    tol_m = 0.4 * d_geo  # conservative dedupe tolerance
    return dict(L=l, s=s, triangles=tri, vertices=vert, edges=edge,
                d_geo=d_geo, d_chord=d_chord, tol_m=tol_m)


def quantised_ecef_key(xyz: np.ndarray, tol_m: float = 50.0):
    """Return integer 3-tuple key for seam deduplication.
    `tol_m` is the positional merge tolerance in metres."""
    q = tol_m
    return tuple(np.round(xyz / q).astype(int))


if __name__ == '__main__':
    """
    Triangular grid will be 9 triangles per octant at layer 0
    At each subsequent layer, the number of triangles will increase by 9 per triangle.
    So the number of triangles will be 8*9**(layer+1)
    """
    rg = Registrar()  # Manage Domains & Projections
    octant = 0  # entire sphere = None.
    # warper = Warper()
    s_oct = rg.domain('s_oct')
    for depth in [4]:  # range(6):
        l_stats = octa_layer_stats(depth)
        net_f = Path(f"experiments/graph_{octant}_{depth}.json")
        pos_f = Path(f"experiments/graph_pos__{octant}_{depth}.json")
        b_pts = get_data(rg, layer=depth, octant_id=octant)
        u, v = b_pts.coords[:, 0], b_pts.coords[:, 1]
        print(f"\npre_simplex:",
              "x[min,max]=", u.min(), u.max(),
              "y[min,max]=", v.min(), v.max())

        s_pts = rg.project(b_pts, ['b_oct', 's_oct'])
        u, v = s_pts.coords[:, 0], s_pts.coords[:, 1]
        s = u+v
        print(f"pre-warp:",
              "u[min,max]=", u.min(), u.max(),
              "v[min,max]=", v.min(), v.max(),
              "u+v[min,max]=", s.min(), s.max())

        # s_wrp = warper.warp(s_pts)
        s_wrp = s_pts
        u, v = s_wrp.coords[:, 0], s_wrp.coords[:, 1]
        s = u+v
        print(f"pre-clamp:",
              "u[min,max]=", u.min(), u.max(),
              "v[min,max]=", v.min(), v.max(),
              "u+v[min,max]=", s.min(), s.max())

        s_oct.clamp(s_wrp, eps=1e-6)

        u, v = s_wrp.coords[:, 0], s_wrp.coords[:, 1]
        s = u+v

        print(f"post-clamp:",
              "u[min,max]=", u.min(), u.max(),
              "v[min,max]=", v.min(), v.max(),
              "u+v[min,max]=", s.min(), s.max())

        b_pts = rg.project(s_wrp, ['s_oct', 'b_oct'])
        g_pts = rg.project(b_pts, ['b_oct', 'g_gcd'])
        c_pts = rg.project(b_pts, ['b_oct', 'c_oct', 'c_ell'])
        areas = wgs84_area(rg, g_pts, 3)
        tr_pts = b_pts.coords.reshape([-1, 3, 2])
        tr_cts = tr_pts.mean(axis=1)

        t_num = len(areas)
        t_sum = np.sum(areas)
        t_avg = t_sum / t_num
        score = (areas / t_avg) - 1  # fractional deviation
        snow_globe(c_pts, 3, score)
