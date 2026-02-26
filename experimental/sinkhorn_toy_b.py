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
from hhg9.h9.classifier import location
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
    face_colors, norm = fc_n

    if shrink != 1.0:
        # Shrink each triangle towards its centroid so edges are visually separated
        centroids = tris.mean(axis=1, keepdims=True)  # (hex_layer,1,2)
        tris = centroids + (tris - centroids) * shrink

    if face_colors is None:
        # Colour-code by triangle index so that corresponding triangles can be tracked
        cmap = mpl.colormaps[cmap_name]
        idx = np.linspace(0.0, 1.0, max(n_tri, 2), endpoint=True)
        face_colors = cmap(idx[:n_tri])

    ratio = H9K.derived.H / H9K.derived.W if not simplex else 1.0
    size = 10
    fig = plt.figure(figsize=(size, ratio * size), dpi=300, frameon=False)
    fig.subplots_adjust(top=0.98, bottom=0.02, right=0.98, left=0.02)
    ax = set_axis(fig, simplex)
    collection = PolyCollection(
        tris,
        ec=(0, 0, 0, 0.25),
        facecolors=face_colors,
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


def plot_density(b_oct, c_den, xy_vertices: np.ndarray, t_grid: np.ndarray, layer: int, marker: str,
                 octant_id: int | None = None):
    """Plot per-triangle density/area values on a vertex grid.

    Parameters
    ----------
    b_oct : Domain
        The b_oct domain (used only to construct Points for plotting).
    c_den : (n_tri,) array
        Per-triangle scalar (e.g. area) to colour by.
    xy_vertices : (n_v, 2) array
        Vertex coordinates in b_oct.
    t_grid : (n_tri, 3) int array
        Triangle vertex indices.
    layer : int
        Layer for title.
    marker : str
        Marker for title.
    octant_id : int | None
        Optional octant id for title only.
    """
    xy_vertices = np.asarray(xy_vertices)
    t_grid = np.asarray(t_grid)

    # Build flattened triangle point list for plot_grid
    tris = np.asarray(xy_vertices[t_grid], dtype=np.float64)  # (n_tri, 3, 2)
    t_pts = tris.reshape([-1, 2])
    b_grid = Points(t_pts, b_oct)

    c_den = np.asarray(c_den, dtype=np.float64)
    c_total = float(np.sum(c_den))
    if not np.isfinite(c_total) or c_total <= 0.0:
        c_norm = np.full_like(c_den, 1.0 / max(len(c_den), 1), dtype=np.float64)
    else:
        c_norm = c_den / c_total

    c_mean = float(np.mean(c_norm))
    c_val = c_norm / max(c_mean, 1e-30) - 1.0

    acol_norm = rgba_from(c_val)
    prefix = f"o{octant_id}_" if octant_id is not None else ""
    plot_grid(b_grid, False, fc_n=acol_norm, title=f"{prefix}t_{marker}_{layer}")


def get_grid(layer: int = 3, octant_id: int = 0):
    """
    Triangular grid will be 9 triangles per octant at hex_layer 0.
    At each subsequent hex_layer, the number of triangles will increase by 9 per triangle.
    So the number of triangles will be 9**(1+hex_layer) per octant.
    """
    mode = H9O.oid_mo[octant_id]
    m = 3**(layer+1)
    pts_expected = int((m+1)*(m+2)/2)
    tri_expected = 9**(layer+1)
    vtx_expected = 3
    edg_expected = 3*m-3
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
        plot_density(b_oct, c_den, verts, trx, layer, 'grid', octant_id=octant_id)

    if save:
        np.savez(
            f"grid_l{layer}.npz",
            cmp=cmp,                 # octant-identity (3,)
            depth=layer,             # mesh size = 9**(hex_layer+1)
            xy_vert=b_vert.coords,   # co-ordinates of vertices.
            grid=trx,                # vert to triangles.
            v_ell=v_ell,             # authalic log-density ℓ  of vertices.
            oc_vtx=oc_vtx,           # indices of octant vertices
            oc_edg=oc_edg,           # indices of octant seam vertices
        )


def load_grid(layer: int):
    """
    Cached loader for the grid NPZ for a given hex_layer.
    This avoids re-reading the same file from disk when `run` is called
    repeatedly for the same (hex_layer, mode) pair.
    """
    f_name = Path(f"grid_l{layer}.npz")
    repo = np.load(f_name, allow_pickle=True)
    cmp = repo['cmp']
    xy_vert = repo['xy_vert']
    v_ell = repo['v_ell']
    grid = repo['grid']
    oc_vtx = repo['oc_vtx']
    oc_edg = repo['oc_edg']
    return cmp, xy_vert, v_ell, grid, oc_vtx, oc_edg


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


def tri_area_error_for_vertices(xy_vertices: np.ndarray, label: str, tri_mask: np.ndarray | None = None):
        """Project the vertex grid to WGS84 and compute per-triangle relative area.

        Returns (areas, rel) where rel = areas/mean(areas) - 1.

        If tri_mask is provided, it must be a boolean mask of length len(t_grid) selecting
        which triangles to include.
        """
        vpts = Points(xy_vertices, b_oct, cmp)
        gpts = rg.project(vpts, [b_oct, g_gcd])
        t_grd_ll = np.array([gpts.coords[v] for t in t_grid for v in t], dtype=np.float64)
        areas = wgs84_area(rg, Points(t_grd_ll, g_gcd), 3)
        areas = np.asarray(areas, dtype=np.float64)
        if tri_mask is not None:
            areas = areas[np.asarray(tri_mask, dtype=bool)]
        rel = areas / areas.mean() - 1.0
        print(f"{label}: tri area rel min/med/max:", float(rel.min()), float(np.median(rel)), float(rel.max()))
        return areas, rel


if __name__ == '__main__':
    rg = Registrar()
    b_oct = rg.domain('b_oct')
    g_gcd = rg.domain('g_gcd')
    octant_id = 0
    layer = 3

    # ---- knobs ----
    invert_density = False
    auto_flip_density = True

    reg = 5e-4
    num_itermax = 20000
    stop_thr = 1e-9

    n_outer = 25         # outer fixed-point iterations
    tau = 0.15           # damping for vertex moves (0<tau<=1); smaller is stabler

    boundary_boost = 1.0  # keep 1.0 while debugging (boosting fights pinning)

    # ---- load grid ----
    cmp, xy_vert, v_ell, t_grid, oc_vtx, oc_edg = load_grid(layer)
    pts0 = Points(xy_vert, b_oct, cmp)
    num = len(pts0)

    # source positions that move
    a_p_full = pts0.coords.copy()
    # target support is fixed (critical)
    t_p = pts0.coords.copy()

    # masks
    boundary_mask = np.zeros(num, dtype=bool)
    boundary_mask[oc_vtx] = True
    boundary_mask[oc_edg] = True
    int_mask = ~boundary_mask
    n_int = int(int_mask.sum())

    print(f"oc_vtx count={oc_vtx.size} oc_edg count={oc_edg.size} oc_int count={n_int} (num={num})")

    # triangle interior mask (triangles with all-3 vertices interior)
    t_grid = np.asarray(t_grid)
    if t_grid.ndim != 2 or t_grid.shape[1] != 3:
        raise ValueError(f"t_grid must have shape (n_tri, 3); got {t_grid.shape}")

    tri_int_mask = np.ones(t_grid.shape[0], dtype=bool)
    for ti in range(t_grid.shape[0]):
        i0, i1, i2 = (int(t_grid[ti, 0]), int(t_grid[ti, 1]), int(t_grid[ti, 2]))
        if boundary_mask[i0] or boundary_mask[i1] or boundary_mask[i2]:
            tri_int_mask[ti] = False
    print(f"tri_int_mask count={int(tri_int_mask.sum())} / {t_grid.shape[0]}")

    # uniform source weights
    a_w = np.full(num, 1.0 / num, dtype=np.float64)

    def _tri_area_rel_stats(xy_boct: np.ndarray, label: str, tri_mask: np.ndarray | None = None) -> None:
        dpts = Points(xy_boct, b_oct, cmp)
        gpts = rg.project(dpts, [b_oct, g_gcd])
        t_grd = np.array([gpts.coords[v] for t in t_grid for v in t], dtype=np.float64)
        tri_area = np.asarray(wgs84_area(rg, Points(t_grd, g_gcd), 3), dtype=np.float64)
        if tri_mask is not None:
            tri_area = tri_area[tri_mask]
        m = float(np.mean(tri_area))
        rel = tri_area / m - 1.0
        print(f"{label}: tri area rel min/med/max: {float(rel.min())} {float(np.median(rel))} {float(rel.max())}")

    def _vertex_mean_area_rel(xy_boct: np.ndarray) -> np.ndarray:
        """Per-vertex mean relative triangle area (WGS84)."""
        dpts = Points(xy_boct, b_oct, cmp)
        gpts = rg.project(dpts, [b_oct, g_gcd])
        t_grd = np.array([gpts.coords[v] for t in t_grid for v in t], dtype=np.float64)
        tri_area = np.asarray(wgs84_area(rg, Points(t_grd, g_gcd), 3), dtype=np.float64).reshape(t_grid.shape[0])

        v_sum = np.zeros(num, dtype=np.float64)
        v_cnt = np.zeros(num, dtype=np.float64)
        for ti, (i0, i1, i2) in enumerate(t_grid):
            a = tri_area[ti]
            v_sum[i0] += a; v_sum[i1] += a; v_sum[i2] += a
            v_cnt[i0] += 1.0; v_cnt[i1] += 1.0; v_cnt[i2] += 1.0

        v_mean = v_sum / np.maximum(v_cnt, 1.0)
        m = float(np.mean(v_mean))
        return v_mean / m - 1.0

    def _build_b_w(v_ell_local: np.ndarray, inv: bool) -> np.ndarray:
        sgn = -1.0 if inv else 1.0
        b = np.exp(sgn * (v_ell_local - np.median(v_ell_local)))
        b = np.clip(b, np.percentile(b, 2), np.percentile(b, 98))
        if boundary_boost != 1.0:
            b = b.copy()
            b[boundary_mask] *= float(boundary_boost)
        b = b / float(np.sum(b))
        # ensure identical total mass to a_w
        b = b * (float(a_w.sum()) / float(b.sum()))
        return b

    def _dn_stats(delta_xy: np.ndarray, name: str) -> None:
        dn = np.linalg.norm(delta_xy, axis=1)
        dn_int = dn[int_mask]
        print(f"{name}: move dn[int] med={float(np.median(dn_int))} max={float(np.max(dn_int))}")

    # baseline
    _tri_area_rel_stats(a_p_full, "baseline")
    _tri_area_rel_stats(a_p_full, "baseline_int_tris", tri_mask=tri_int_mask)

    # weights + optional flip based on correlation with actual WGS84 area error
    v_area0 = _vertex_mean_area_rel(a_p_full)
    b_w = _build_b_w(v_ell, invert_density)
    corr0 = float(np.corrcoef(v_area0, b_w)[0, 1])
    if auto_flip_density and np.isfinite(corr0) and corr0 < 0.0:
        invert_density = not invert_density
        b_w = _build_b_w(v_ell, invert_density)
        corr1 = float(np.corrcoef(v_area0, b_w)[0, 1])
        print(f"auto_flip_density: corr(v_area,b_w) {corr0:+.3f} -> flipped invert_density={invert_density} corr={corr1:+.3f}")
    else:
        print(f"corr(v_area,b_w)={corr0:+.3f} invert_density={invert_density}")

    print(f"b_w range: {float(b_w.min())} {float(b_w.max())} invert_density={invert_density} auto_flip_density={auto_flip_density}")

    # fixed cost matrix (source moves, target support fixed)
    loss_matrix = ot.dist(a_p_full, t_p, metric="sqeuclidean")
    loss_matrix /= float(loss_matrix.max())
    print(f"loss median: {float(np.median(loss_matrix))} reg: {reg}")

    # outer loop
    a_p = a_p_full.copy()
    for it in range(n_outer):
        print(f"it{it:02d}: loss median {float(np.median(loss_matrix)):.6g} reg {reg:.6g}")
        _tri_area_rel_stats(a_p, f"it{it:02d}_pre")

        # diagnostic corr on current geometry (optional but helpful)
        v_area = _vertex_mean_area_rel(a_p)
        corr = float(np.corrcoef(v_area, b_w)[0, 1])
        if np.isfinite(corr):
            print(f"it{it:02d}: corr(v_area,b_w_fixed)={corr:+.3f}")

        gamma = ot.sinkhorn(
            a_w, b_w, loss_matrix,
            reg=reg,
            method="sinkhorn_log",
            numItermax=num_itermax,
            stopThr=stop_thr,
            verbose=False,
        )

        row_sum = gamma.sum(axis=1, keepdims=True)
        x_bar = (gamma @ t_p) / np.maximum(row_sum, 1e-30)

        # tau = damping: without this, you can “ratchet” into worse geometry fast
        a_p_new = (1.0 - tau) * a_p + tau * x_bar

        # pin boundary
        a_p_new[boundary_mask] = a_p[boundary_mask]

        _dn_stats(a_p_new - a_p, f"it{it:02d}")
        a_p = a_p_new

        # keep cost fixed (recommended early); if you later want, you can refresh it every k iters:
        # loss_matrix = ot.dist(a_p, t_p, metric="sqeuclidean"); loss_matrix /= loss_matrix.max()

    # final report + plot
    _tri_area_rel_stats(a_p, "after_ot")
    _tri_area_rel_stats(a_p, "after_ot_int_tris", tri_mask=tri_int_mask)

    dpts = Points(a_p, b_oct, cmp)
    gpts = rg.project(dpts, [b_oct, g_gcd])
    t_grd_new = np.array([gpts.coords[v] for t in t_grid for v in t], dtype=np.float64)
    rx = wgs84_area(rg, Points(t_grd_new, g_gcd), 3)

    plot_density(b_oct, rx, a_p, t_grid, layer, "xp_sinkhorn_fp", octant_id=octant_id)
