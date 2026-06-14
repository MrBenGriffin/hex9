# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
from pathlib import Path

import ot
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.collections import PolyCollection
from matplotlib.ticker import FuncFormatter
from hhg9 import Registrar, Points
from hhg9.algorithms.distance import wgs84_area
from hhg9.h9 import H9K, H9O
from hhg9.h9.polygon import tri_mesh


"""
sinkhorn
Parameters:
    a (array-like, shape (dim_a,)) – samples weights in the source domain
    b (array-like, shape (dim_b,) or ndarray, shape (dim_b, n_hists)) – samples in the target domain, compute sinkhorn with multiple targets and fixed if is a matrix (return OT loss + dual variables in log)
    M (array-like, shape (dim_a, dim_b)) – loss matrix
    reg (float) – Regularization term >0
    method (str) – method used for the solver either ‘sinkhorn’,’sinkhorn_log’, ‘greenkhorn’, ‘sinkhorn_stabilized’ or ‘sinkhorn_epsilon_scaling’, see those function for specific parameters
    numItermax (int, optional) – Max number of iterations
    stopThr (float, optional) – Stop threshold on error (>0)
    verbose (bool, optional) – Print information along iterations
    log (bool, optional) – record log if True
    warn (bool, optional) – if True, raises a warning if the algorithm doesn’t convergence.
    warmstart (tuple of arrays, shape (dim_a, dim_b), optional) – Initialization of dual potentials. 
    If provided, the dual potentials should be given (that is the logarithm of the u,v sinkhorn scaling vectors)

Returns:
    gamma (array-like, shape (dim_a, dim_b)) – Optimal transportation matrix for the given parameters
    log (dict) – log dictionary return only if log==True in parameters
"""


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


def set_axis(mfig, simplex: bool = False, cols=1, box=1):
    """Axis template"""
    ax = mfig.add_subplot(1, cols, box)  # (*nrows*, *ncols*, *index*)

    ax.set_aspect('equal', adjustable='box')
    if not simplex:
        ax.set_xlim(H9K.limits.TL, H9K.limits.TR)  # Use TL/TR with a 5% margin
        ax.set_ylim(H9K.limits.VF, H9K.limits.VC)  # Use VF/ΛC with a 5% margin
    else:
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.0)
    ax.set_axis_off()
    return ax


def plot_grid(grid_pts, simplex: bool = False,
              shrink: float = 1.0,
              fc_n: tuple = None, title: str | None = None,
              cmap_name: str = "plasma"):
    """
    :param grid_pts: Points(n*3, 2) of triangle grid. Resizable to (n,3,2) triangles.
    :param simplex: if True, grid_pts is in simplex coordinates (u,v).
    :param shrink: factor in (0,1] to shrink each triangle towards its centroid.
    :param fc_n:   optional (n,4),norm RGBA array giving per-triangle colours. If None,
                       a colormap is applied based on triangle index.
    :param title: optional title for the figure.
    :param cmap_name: Matplotlib colormap name to use when facecolors is None.
    """
    tris = grid_pts.coords.reshape([-1, 3, 2])
    n_tri = tris.shape[0]
    facecolors, norm = fc_n

    if shrink != 1.0:
        # Shrink each triangle towards its centroid so edges are visually separated
        centroids = tris.mean(axis=1, keepdims=True)  # (hex_layer,1,2)
        tris = centroids + (tris - centroids) * shrink

    if facecolors is None:
        # Colour-code by triangle index so that corresponding triangles can be tracked
        cmap = mpl.colormaps[cmap_name]
        idx = np.linspace(0.0, 1.0, max(n_tri, 2), endpoint=True)
        facecolors = cmap(idx[:n_tri])

    ratio = H9K.derived.H / H9K.derived.W if not simplex else 1.0
    size = 10
    fig = plt.figure(figsize=(size, ratio * size), dpi=300, frameon=False)
    fig.subplots_adjust(top=0.98, bottom=0.02, right=0.98, left=0.02)
    ax = set_axis(fig, simplex)
    collection = PolyCollection(
        tris,
        ec=(0, 0, 0, 0.25),
        facecolors=facecolors,
        alpha=1.0,
        linewidth=0.0001,
    )
    x_min, x_max = grid_pts.coords[:, 0].min(), grid_pts.coords[:, 0].max()
    y_min, y_max = grid_pts.coords[:, 1].min(), grid_pts.coords[:, 1].max()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=norm)
    sm.set_array([])  # no data needed
    cb = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cb.ax.tick_params(labelsize=20)  # or 14, etc.
    cb.formatter = FuncFormatter(lambda v, pos: f"{v:+.3f}")
    cb.update_normal(sm)
    ax.add_collection(collection)
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(f"output/{title}.png", dpi=300)


def plot_density(b_oct, c_den, vts, trx, octant_id, layer, marker):
    cmp = H9O.oid_cmp[octant_id]
    tris = np.array([vts[v] for t in trx for v in t], dtype=np.float64)
    t_grid = tris.reshape([-1, 2])  # triangle CW
    b_grid = Points(t_grid, b_oct, components=cmp)
    c_total = np.sum(c_den)
    c_norm = c_den / c_total
    c_mean = np.mean(c_norm)
    c_val = c_norm / c_mean - 1.0
    acol_norm = rgba_from(c_val)
    plot_grid(b_grid, False, fc_n=acol_norm, title=f"t_{marker}_{layer}")


def grid(rg: Registrar, layer: int = 3, octant_id: int = 0, plotting: bool = False, save=True):
    """
    Triangular grid will be 9 triangles per octant at hex_layer 0.
    At each subsequent hex_layer, the number of triangles will increase by 9 per triangle.
    So the number of triangles will be 9**(1+hex_layer) per octant.
    """
    b_oct = rg.domain('b_oct')
    g_gcd = rg.domain('g_gcd')
    cmp = H9O.oid_cmp[octant_id]
    mode = H9O.oid_mo[octant_id]

    # Base grid in barycentric XY (b_oct)
    verts, _, trx = tri_mesh(layer, mode)
    b_vert = Points(verts, b_oct, components=cmp)
    v_ell = get_density(rg, b_vert, octant_id)

    if plotting:
        gpts = rg.project(b_vert, [b_oct, g_gcd])
        t_grd = np.array([gpts.coords[v] for t in trx for v in t], dtype=np.float64)
        c_den = wgs84_area(rg, Points(t_grd, g_gcd), 3)
        #
        # t_den = np.array([v_ell[v] for t in trx for v in t], dtype=np.float64)
        # c_den = t_den.reshape(-1, 3).mean(axis=1)
        plot_density(b_oct, c_den, verts, trx, octant_id, layer, 'grid')

    if save:
        np.savez(
            f"grid_l{layer}.npz",
            cmp=cmp,                 # octant-identity (3,)
            depth=layer,             # mesh size = 9**(hex_layer+1)
            xy_vert=b_vert.coords,   # co-ordinates of vertices.
            grid=trx,                # vert to triangles.
            v_ell=v_ell,             # authalic log-density ℓ  of vertices.
        )


def load_grid(layer: int):
    """
    Cached loader for the grid NPZ for a given hex_layer.
    This avoids re-reading the same file from disk when `run` is called
    repeatedly for the same (hex_layer, net_mode) pair.
    """
    f_name = Path(f"grid_l{layer}.npz")
    repo = np.load(f_name, allow_pickle=True)
    cmp = repo['cmp']
    xy_vert = repo['xy_vert']
    v_ell = repo['v_ell']
    grid = repo['grid']
    return cmp, xy_vert, v_ell, grid


def get_density(reg: Registrar, pts: Points, octant_id: int = 0):
    """
    Compute authalic log-density ℓ = log(area_scale) for points in one octant.
    :param reg: H9 Registrar
    :param pts: Points object to compute density for.
    :param octant_id: octant index (0) for matrix application
    :return: authalic log-density ℓ
    """
    if not (0 <= octant_id < 8):
        raise ValueError(f"octant_id must be in [0, 7]")
    if not isinstance(pts, Points):
        raise ValueError("Points must be a Points object in c_oct, b_oct, or s_oct domain")
    ake = reg.projection('oct_ell')
    b_oct = reg.domain('b_oct')
    face = H9O.oid_str[octant_id]
    prj = b_oct.projs[face]
    q = prj.matrix.T @ prj.orient
    e1_xyz = q[:, 0]  # 3-vector
    e2_xyz = q[:, 1]  # 3-vector
    pts_to_use = None
    dom = pts.domain
    match dom.name:
        case 'c_oct':
            pts_to_use = pts
        case 'b_oct':
            pts_to_use = reg.project(pts, ['b_oct', 'c_oct'])
        case 's_oct':
            pts_to_use = reg.project(pts, ['s_oct', 'b_oct', 'c_oct'])
        case _:
            raise ValueError("Points must be in c_oct, b_oct, or s_oct domain")
    j_pts = ake.jacobian(pts_to_use.coords)
    # det = np.linalg.det(j_pts)
    # kmin = np.argmin(det)
    # kmax = np.argmax(det)
    # mn, mx = pts.coords[kmin], pts.coords[kmax]
    v1 = j_pts @ e1_xyz  # (hex_layer, 3)
    v2 = j_pts @ e2_xyz  # (hex_layer, 3)
    cross = np.cross(v1, v2)  # (hex_layer, 3)
    area_scale = np.linalg.norm(cross, axis=1)  # (hex_layer,)
    area_clip = np.clip(area_scale, 1e-20, None)
    return np.log(area_clip)  # authalic log-density ℓ


if __name__ == '__main__':
    rg = Registrar()
    octant_id = 0
    layer = 5
    # for i in [4]:
    grid(rg, layer, octant_id, plotting=True, save=True)
    g_gcd = rg.domain('g_gcd')
    b_oct = rg.domain('b_oct')
    cmp, xy_vert, v_ell, t_grid = load_grid(layer)
    pts = Points(xy_vert, b_oct, cmp)
    num = len(pts)

    # a_w_raw = 1.0 - np.exp(v_ell - v_ell.max()) + 1e-20
    a_w_raw = np.exp(v_ell - v_ell.max()) + 1e-20
    a_w = a_w_raw / a_w_raw.sum()
    a_p = pts.coords
    t_p = pts.coords.copy()
    t_w = np.full([num], 1 / num)
    loss_matrix = ot.dist(a_p, t_p, metric='sqeuclidean')

    # a_w = np.full([num], 1 / num)  # uniform source
    # b_w_raw = np.exp(v_ell - v_ell.max())  # J ∝ exp(ell)
    # b_w = b_w_raw / b_w_raw.sum()

    gamma = ot.sinkhorn(a_w, t_w, loss_matrix, reg=5e-3, numItermax=800, verbose=True)
    # gamma = ot.bregman.sinkhorn_epsilon_scaling(a_w, t_w, loss_matrix, numItermax=400, numInnerItermax=100, reg=1e-2, verbose=True)
    # Persist plan + inputs to make downstream barycentric projection/debugging reproducible.
    np.savez_compressed(
        f"gamma_l{layer}.npz",
        gamma=gamma,
        loss_matrix=loss_matrix,
        a_w=a_w,
        t_w=t_w,
        a_p=a_p,
        t_p=t_p,
        layer=layer,
    )

    # x_prime = (gamma @ t_p) / b_w[:, None]
    # row_sum = gamma.sum(axis=1, keepdims=True)  # (n,1) should equal a_w[:,None]
    # row_sum = np.maximum(row_sum, 1e-30)  # guard
    # x_prime = (gamma @ t_p) / row_sum  # (n,2)
    # x_prime = (gamma @ t_p) / gamma.sum(axis=1, keepdims=True)
    # delta = (gamma - np.diag(a_w)) @ a_p  # (n,2)
    eps = 0.15  # small ish.
    row_sum = gamma.sum(axis=1, keepdims=True)  # ≈ a_w[:,None]
    row_sum = np.maximum(row_sum, 1e-30)
    x_bar = (gamma @ t_p) / row_sum  # barycentric target for each source
    delta = x_bar - a_p
    print("delta min/max:", delta.min(), delta.max())
    x_prime = a_p + eps * delta
    dpts = Points(x_prime, b_oct, cmp)
    gpts = rg.project(dpts, [b_oct, g_gcd])
    t_grd = np.array([gpts.coords[v] for t in t_grid for v in t], dtype=np.float64)
    rx = wgs84_area(rg, Points(t_grd, g_gcd), 3)
    dens = rx

    plot_density(b_oct, dens, x_prime, t_grid, octant_id, layer, 'xp_grid')


