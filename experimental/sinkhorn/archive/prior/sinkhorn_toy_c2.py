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
from hhg9.h9.classifier import location
from hhg9.h9.polygon import tri_mesh
from scipy.spatial import ConvexHull

from hhg9.h9.protocols import BaryLoc

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


def rgba_from(arr: np.ndarray, cmap_name: str = 'RdBu_r', norm=None, alpha: float = 1.0):
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
              cmap_name: str = 'RdBu_r'):
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
    plt.savefig(f"output/{title}.jpg", dpi=150)
    plt.close()


def plot_density(b_oct, c_den, vts, trx, octant_id, layer, marker, oc_vtx=None, oc_edg=None):
    cmp = H9O.oid_cmp[octant_id]
    tris = np.array([vts[v] for t in trx for v in t], dtype=np.float64)
    t_grid = tris.reshape([-1, 2])  # triangle CW
    b_grid = Points(t_grid, b_oct, components=cmp)
    c_total = np.sum(c_den)                 # sum of all areas = entire octant.
    c_ideal = c_total / (9 ** (layer + 1))  # number of triangles
    c_val = c_den / c_ideal - 1.0           # normalise and zero
    abs_val = np.abs(c_val)

    # Robust colour scaling: clamp to a high percentile so a single outlier
    # (often near a pinned corner) doesn't wash out the rest of the grid.
    clip = float(np.percentile(abs_val, 99))
    clip = float(np.maximum(clip, 1e-12))
    norm = colors.TwoSlopeNorm(vmin=-clip, vcenter=0.0, vmax=clip)
    acol_norm = rgba_from(c_val, norm=norm)

    worst = int(np.argmax(abs_val))
    worst_val = float(c_val[worst])
    worst_abs = float(abs_val[worst])

    touches_corner = None
    if oc_vtx is not None:
        oc_vtx_set = set(int(i) for i in np.asarray(oc_vtx).ravel())
        touches_corner = any(int(v) in oc_vtx_set for v in trx[worst])
    touches_edge = None
    if oc_edg is not None:
        oc_edg_set = set(int(i) for i in np.asarray(oc_edg).ravel())
        touches_edge = any(int(v) in oc_edg_set for v in trx[worst])

    c_mae = float(np.mean(abs_val))
    c_std = float(np.std(c_val))
    c_min = float(np.min(c_val))
    c_max = float(np.max(c_val))
    c_p90 = np.percentile(abs_val, 90)
    c_p99 = np.percentile(abs_val, 99)

    print(
        f"mae: {c_mae}; std: {c_std}; min: {c_min}; max: {c_max}; p90: {c_p90}; p99: {c_p99}; "
        f"clip99(abs): {clip}; worst(abs): {worst_abs} (val={worst_val}) tri={worst} "
        f"touches_corner={touches_corner}, touches_edge={touches_edge}"
    )
    plot_grid(b_grid, False, fc_n=acol_norm, title=f"c2_{marker}_L{layer}")


def get_grid(layer: int = 3, octant_id: int = 0):
    """
    Triangular grid will be 9 triangles per octant at hex_layer 0.
    At each subsequent hex_layer, the number of triangles will increase by 9 per triangle.
    So the number of triangles will be 9**(1+hex_layer) per octant.
    """
    mode = H9O.oid_mo[octant_id]
    m = 3 ** (layer + 1)
    pts_expected = int((m + 1) * (m + 2) / 2)
    tri_expected = 9 ** (layer + 1)
    vtx_expected = 3
    edg_expected = 3 * m - 3
    verts, _, trx = tri_mesh(layer, mode)
    ẋ, y = verts[:, 0] * H9K.R3, verts[:, 1]
    locs = location(ẋ, y, mode)
    oc_vtx = np.flatnonzero(locs == BaryLoc.VTX)  # On an octant vertex (1 of 3)
    oc_edg = np.flatnonzero(locs == BaryLoc.EDG)  # On the octant seam
    if verts.shape[0] != pts_expected:
        print(f"{layer}: octant points {verts.shape[0]} is not {pts_expected}")
    if trx.shape[0] != tri_expected:
        print(f"{layer}: octant triangles {trx.shape[0]} is not {tri_expected}")
    if oc_vtx.shape[0] != vtx_expected:
        print(f"{layer}: octant vert_pts {oc_vtx.shape[0]} is not {vtx_expected}")
    if oc_edg.shape[0] != edg_expected:
        print(f"{layer}: octant seam_pts {oc_edg.shape[0]} is not {edg_expected}")
    return verts, trx, oc_vtx, oc_edg


def grid(rg: Registrar, layer: int = 3, octant_id: int = 0, plotting: bool = False, save=True):
    """
    Triangular grid will be 9 triangles per octant at hex_layer 0.
    At each subsequent hex_layer, the number of triangles will increase by 9 per triangle.
    So the number of triangles will be 9**(1+hex_layer) per octant.
    """
    b_oct = rg.domain('b_oct')
    g_gcd = rg.domain('g_gcd')
    cmp = H9O.oid_cmp[octant_id]
    verts, trx, oc_vtx, oc_edg = get_grid(layer=layer, octant_id=octant_id)

    b_vert = Points(verts, b_oct, components=cmp)
    v_ell = get_density(rg, b_vert, octant_id)

    if plotting:
        gpts = rg.project(b_vert, [b_oct, g_gcd])
        t_grd = np.array([gpts.coords[v] for t in trx for v in t], dtype=np.float64)
        c_den = wgs84_area(rg, Points(t_grd, g_gcd), 3)
        plot_density(b_oct, c_den, verts, trx, octant_id, layer, 'grid', oc_vtx=oc_vtx)

    if save:
        np.savez(
            f"grid_l{layer}.npz",
            cmp=cmp,  # octant-identity (3,)
            depth=layer,  # mesh size = 9**(hex_layer+1)
            xy_vert=b_vert.coords,  # co-ordinates of vertices.
            grid=trx,  # vert to triangles.
            v_ell=v_ell,  # authalic log-density ℓ  of vertices.
            oc_vtx=oc_vtx,  # indices of octant vertices
            oc_edg=oc_edg,  # indices of octant seam vertices
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
    oc_vtx = repo['oc_vtx']
    oc_edg = repo['oc_edg']
    grid = repo['grid']
    return cmp, xy_vert, v_ell, oc_vtx, oc_edg, grid


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
    v1 = j_pts @ e1_xyz  # (hex_layer, 3)
    v2 = j_pts @ e2_xyz  # (hex_layer, 3)
    cross = np.cross(v1, v2)  # (hex_layer, 3)
    area_scale = np.linalg.norm(cross, axis=1)  # (hex_layer,)
    area_clip = np.clip(area_scale, 1e-20, None)
    return np.log(area_clip)  # authalic log-density ℓ


if __name__ == '__main__':
    rg = Registrar()
    b_oct = rg.domain('b_oct')
    g_gcd = rg.domain('g_gcd')
    octant_id = 0
    layer = 3

    REG = 1.0e-4  # blur radius or "temperature" of the system. High = stiff, Low=watery
    CENTER_BOOST = 1.0  # Interior boost multiplier (keep near 1.0; scan tightly around 1)
    CORNER_BOOST = 1.211613  # Low: tips tear away or stretch violently. High: Corners get crushed
    TAPER_POW = 3.0  # Shape of interior taper: higher -> only deep interior gets boosted

    mid_floor = 0.35  # try 0.15..0.35
    boundary_boost = 1.075  # Best fit.
    boundary_pc = 0.18  # thickness as fraction of d_max
    boundary_pow = 2.0  # shape
    mid_pow = 2.0  # concentrates boundary boost toward mid-edge (0 at corners, 1 at mid-edge)
    corner_ring_pc = 0.10  # thickness as fraction of d_corner_max
    corner_ring_pow = 2.0
    corner_ring_boost = 0.97  # <1 shrinks corner-adjacent areas; >1 grows them

    cmp, xy_vert, v_ell, oc_vtx, oc_edg, t_grid = load_grid(layer)
    pts = Points(xy_vert, b_oct, cmp)
    num = len(pts)

    a_p = pts.coords
    t_p = pts.coords.copy()

    # Tuned Weights (Refined Profile) ---
    a_w = np.ones(num, dtype=np.float64) / num
    b_w_raw = np.exp(v_ell - np.median(v_ell))
    b_w_base = np.clip(b_w_raw, np.percentile(b_w_raw, 2), np.percentile(b_w_raw, 98))

    corners = a_p[oc_vtx]

    d_corner = np.min(np.linalg.norm(a_p[:, None, :] - corners[None, :, :], axis=2), axis=1)
    d_corner_max = float(d_corner.max())

    t_corner = np.clip(1.0 - d_corner / (corner_ring_pc * d_corner_max), 0.0, 1.0) ** corner_ring_pow
    mask_no_corner = np.ones(num, dtype=bool)
    mask_no_corner[oc_vtx] = False

    b_w_raw[mask_no_corner] *= (1.0 + (corner_ring_boost - 1.0) * t_corner[mask_no_corner])

    # Define the 3 boundary lines (in b_oct 2D coordinates)
    lines = [(corners[0], corners[1]), (corners[1], corners[2]), (corners[2], corners[0])]

    # Helper: project points to a line segment and return the parametric position `t` in [0, 1]
    def _project_and_get_t(p, v1, v2):
        line_vec = v2 - v1
        line_len2 = np.sum(line_vec ** 2)
        p_vec = p - v1
        t = np.sum(p_vec * line_vec, axis=1) / line_len2
        t = np.clip(t, 0.0, 1.0)
        proj = v1 + np.outer(t, line_vec)
        return proj, t

    # Precompute an edge-midpoint factor for all points:
    #  - 0 at the edge endpoints (corners)
    #  - 1 at the edge midpoint
    best_t_all = np.zeros(num, dtype=np.float64)
    min_dists_all = np.full(num, np.inf, dtype=np.float64)
    for v_start, v_end in lines:
        proj, t = _project_and_get_t(a_p, v_start, v_end)
        dists = np.linalg.norm(proj - a_p, axis=1)
        mask = dists < min_dists_all
        best_t_all[mask] = t[mask]
        min_dists_all[mask] = dists[mask]

    mid_base = 4.0 * best_t_all * (1.0 - best_t_all)  # 0 at corners, 1 at mid-edge

    h_start, h_stop, h_step = 2.0, 4.0, 10
    hyper = np.linspace(h_start, h_stop, h_step)
    print(f"c. hyper: ({h_start} to {h_stop}; {h_step} steps) across mid_pow")
    for mid_pow in hyper:
        cts = f'_mf{int(1000 * mid_floor):04d}'
        mpw = f'_mp{int(1000 * mid_pow):04d}'
        cns = f'_cn{int(1000 * CORNER_BOOST):04d}'
        tps = f'_tp{int(1000 * TAPER_POW):04d}'
        bpc = f'_bpc{int(1000 * boundary_pc):04d}'
        b_w_raw = b_w_base.copy()

        # Mid-edge concentration factor for this sweep value.
        mid_factor = np.power(mid_base, mid_pow)

        # Protect corners (helps avoid violent stretching / tip tearing)
        b_w_raw[oc_vtx] *= CORNER_BOOST

        # Interior boost (fights the inward "vacuum" without fattening the whole boundary)
        # Measure "center-ness" as distance to the nearest edge.
        d_edge = np.full(num, np.inf, dtype=np.float64)
        for v_start, v_end in lines:
            proj, _t = _project_and_get_t(a_p, v_start, v_end)
            d_edge = np.minimum(d_edge, np.linalg.norm(proj - a_p, axis=1))

        d_max = float(d_edge.max())
        if d_max > 0:
            # Normalised "center-ness" in [0, 1], then sharpen so only deep interior changes much
            taper = np.clip(d_edge / d_max, 0.0, 1.0) ** TAPER_POW

            # Apply boost only to true interior points (edges/corners remain governed by their own rules)
            mask_int = np.ones(num, dtype=bool)
            mask_int[oc_vtx] = False
            mask_int[oc_edg] = False
            b_w_raw[mask_int] *= (1.0 + (CENTER_BOOST - 1.0) * taper[mask_int])

        # Boundary hex_layer boost: strong near the boundary, concentrated toward mid-edge
        t_edge = np.clip(1.0 - d_edge / (boundary_pc * d_max), 0.0, 1.0) ** boundary_pow
        # boundary_shape = t_edge * mid_factor

        boundary_shape = t_edge * (mid_floor + (1.0 - mid_floor) * mid_factor)

        mask_no_corner = np.ones(num, dtype=bool)
        mask_no_corner[oc_vtx] = False
        b_w_raw[mask_no_corner] *= (1.0 + (boundary_boost - 1.0) * boundary_shape[mask_no_corner])

        # Normalize
        b_w = b_w_raw / np.sum(b_w_raw)
        b_w = b_w * (a_w.sum() / b_w.sum())

        # Increase reg slightly to 1e-3 to dampen any "ringing" (ripples)
        # caused by the sharp boundary constraints.
        loss_matrix = ot.dist(a_p, t_p, metric="sqeuclidean")
        loss_matrix /= loss_matrix.max()

        # print(f"Solving with (reg={reg})...")

        gamma_log = ot.sinkhorn(
            a_w, b_w, loss_matrix,
            reg=REG,  # Lower reg = less inward "suction".
            method='sinkhorn_log',
            numItermax=20000,
            stopThr=1e-6,
            verbose=False
        )

        row_sum = gamma_log.sum(axis=1, keepdims=True)
        x_prime = (gamma_log @ t_p) / np.maximum(row_sum, 1e-30)

        # --- 3. 1D RELAXATION & SORTING ---
        # The "Spikes" happen because points slide out of order.
        # We project them to the edge, then SORT them to guarantee clean edges.

        # Iterate over boundary points (excluding corners for now)
        edge_pts = x_prime[oc_edg]

        # 1. Assign each point to the closest edge
        best_projections = np.zeros_like(edge_pts)
        best_t = np.zeros(len(edge_pts))
        min_dists = np.full(len(edge_pts), np.inf)
        edge_labels = np.zeros(len(edge_pts), dtype=int)  # 0, 1, 2

        for i, (v_start, v_end) in enumerate(lines):
            proj, t = _project_and_get_t(edge_pts, v_start, v_end)
            dists = np.linalg.norm(proj - edge_pts, axis=1)

            mask = dists < min_dists
            best_projections[mask] = proj[mask]
            best_t[mask] = t[mask]
            min_dists[mask] = dists[mask]
            edge_labels[mask] = i

        # 2. Zipper pass (after labels are final): sort along each edge to prevent crossings
        for j in range(3):
            mask = edge_labels == j
            if not np.any(mask):
                continue

            indices = np.where(mask)[0]
            raw_t = best_t[indices]

            order = np.argsort(raw_t)
            sorted_indices = indices[order]
            t_final = np.sort(raw_t)

            v_start, v_end = lines[j]
            line_vec = v_end - v_start
            perfect_pos = v_start + np.outer(t_final, line_vec)

            x_prime[oc_edg[sorted_indices]] = perfect_pos

        # Pin corners
        x_prime[oc_vtx] = a_p[oc_vtx]

        # --- Metrics & Plot ---
        delta = x_prime - a_p
        dn = np.linalg.norm(delta, axis=1)

        dpts = Points(x_prime, b_oct, cmp)
        gpts = rg.project(dpts, [b_oct, g_gcd])
        t_grd_new = np.array([gpts.coords[v] for t in t_grid for v in t], dtype=np.float64)
        rx = wgs84_area(rg, Points(t_grd_new, g_gcd), 3)
        marker = f'{mpw}{bpc}{tps}{cns}{cts}'
        print(f'\nPlotting: {marker}')
        print(f'BOUND_BOOST: {boundary_boost:.6f}; '
              f'boundary_pc: {boundary_pc:.6f}; '
              f'mid_floor: {mid_floor:.6f}; '
              f'mid_pow: {mid_pow:.6f}; '
              f'CORNER_BOOST: {CORNER_BOOST:.6f}; '
              f'TAPER_POW: {TAPER_POW:.6f}; '
              f'REG: {REG:.6f}')

        plot_density(b_oct, rx, x_prime, t_grid, octant_id, layer, marker, oc_vtx=oc_vtx, oc_edg=oc_edg)
