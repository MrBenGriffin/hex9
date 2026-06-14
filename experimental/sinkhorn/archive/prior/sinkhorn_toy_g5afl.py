# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
from pathlib import Path

import ot
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.ticker import FuncFormatter
from hhg9 import Registrar, Points
from hhg9.algorithms.distance import wgs84_area
from hhg9.h9 import H9O, H9K
from hhg9.h9.classifier import location
from hhg9.h9.polygon import tri_mesh
from hhg9.h9.protocols import BaryLoc
import time
import warnings

lim = H9K.limits

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


def set_axis(mfig, cols=1, box=1):
    """Axis template"""
    ax = mfig.add_subplot(1, cols, box)  # (*nrows*, *ncols*, *index*)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(H9K.limits.TL, H9K.limits.TR)  # Use TL/TR with a 5% margin
    ax.set_ylim(H9K.limits.VF, H9K.limits.VC)  # Use VF/ΛC with a 5% margin
    ax.set_axis_off()
    return ax


def plot_grid(grid_pts,
              shrink: float = 1.0,
              fc_n: tuple = None, title: str | None = None, text=None,
              cmap_name: str = 'RdBu_r'):
    """
    :param grid_pts: Points(n*3, 2) of triangle grid. Resizable to (n,3,2) triangles.
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

    ratio = H9K.derived.H / H9K.derived.W
    size = 10
    fig = plt.figure(figsize=(size, ratio * size), dpi=300, frameon=False)
    fig.subplots_adjust(top=0.98, bottom=0.02, right=0.98, left=0.02)
    ax = set_axis(fig)
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
    if text:
        ax.set_title(text, fontsize=20)
    plt.tight_layout()
    plt.savefig(f"output/{title}.jpg", dpi=150)
    plt.close()

# --- Area deviation and compare plot utilities ---
def _area_deviation_for_grid(rg: Registrar,
                             pts_xy: np.ndarray,
                             grid: np.ndarray,
                             cmp: np.ndarray,
                             layer: int,
                             b_oct,
                             g_gcd) -> np.ndarray:
    """Return per-triangle relative area error (area/ideal - 1) for a b_oct grid."""
    pts = Points(pts_xy, b_oct, components=cmp)
    gpts = rg.project(pts, [b_oct, g_gcd])

    # Vectorised triangle gather (n_tri, 3, 2) -> (n_tri*3, 2)
    tri_g = gpts.coords[grid].reshape(-1, 2)
    c_den = wgs84_area(rg, Points(tri_g, g_gcd), 3)

    c_total = float(np.sum(c_den))
    c_ideal = c_total / (9 ** (layer + 1))
    return (c_den / c_ideal) - 1.0


# --- Displacement visualisation utilities ---

def plot_displacement_field(xy_a: np.ndarray,
                            xy_b: np.ndarray,
                            octant_id: int,
                            layer: int,
                            title: str,
                            stride: int = 1,
                            scale: float | None = None,
                            min_mag: float = 0.0,
                            alpha: float = 0.85,
                            headwidth: float = 3.0,
                            headlength: float = 4.0):
    """Plot a sparse vector field of vertex displacements (b_oct space)."""
    xy_a = np.asarray(xy_a, dtype=float)
    xy_b = np.asarray(xy_b, dtype=float)
    delta = xy_b - xy_a
    mag = np.linalg.norm(delta, axis=1)

    if stride < 1:
        stride = 1

    idx = np.arange(xy_a.shape[0])
    if stride > 1:
        idx = idx[::stride]

    if min_mag > 0.0:
        idx = idx[mag[idx] >= float(min_mag)]

    x = xy_a[idx, 0]
    y = xy_a[idx, 1]
    u = delta[idx, 0]
    v = delta[idx, 1]

    ratio = H9K.derived.H / H9K.derived.W
    size = 10
    fig = plt.figure(figsize=(size, ratio * size), dpi=250, frameon=False)
    fig.subplots_adjust(top=0.98, bottom=0.02, right=0.98, left=0.02)
    ax = set_axis(fig)

    # If scale is None, choose a scale so arrows are visible but not overwhelming.
    # Matplotlib quiver uses `scale` as: arrow_length = vector_length / scale.
    if scale is None:
        # Aim for typical arrow length ~ 1% of triangle width
        w = float(H9K.radical.W)
        tgt = 0.01 * w
        med = float(np.median(mag[idx])) if idx.size else 0.0
        scale = (med / tgt) if (med > 0.0 and tgt > 0.0) else 1.0

    q = ax.quiver(
        x, y, u, v,
        angles='xy',
        scale_units='xy',
        scale=scale,
        width=0.002,
        alpha=alpha,
        headwidth=headwidth,
        headlength=headlength,
    )

    # Add a small legend arrow (quiverkey) based on p90 magnitude.
    if idx.size:
        p90 = float(np.percentile(mag[idx], 90))
        if p90 > 0.0:
            ax.quiverkey(q, X=0.88, Y=0.07, U=p90, label=f"p90 |Δ| = {p90:.3e}", labelpos='E', fontproperties={'size': 14})

    ax.set_title(f"disp_{title}", fontsize=18)
    plt.tight_layout()
    file_title = f"disp_{title}_L{layer}_o{octant_id}"
    plt.savefig(f"output/{file_title}.jpg", dpi=150)
    plt.close()


def plot_line_displacement(xy_a: np.ndarray,
                           xy_b: np.ndarray,
                           p0: np.ndarray,
                           p1: np.ndarray,
                           layer: int,
                           octant_id: int,
                           title: str,
                           eps: float | None = None,
                           max_points: int = 4000):
    """Plot along-line displacement components for vertices close to a given line.

    Selects vertices whose *original* positions are within `eps` of the line (p0->p1),
    then plots parallel and perpendicular displacement components vs distance along
    the line.
    """
    xy_a = np.asarray(xy_a, dtype=float)
    xy_b = np.asarray(xy_b, dtype=float)
    p0 = np.asarray(p0, dtype=float).reshape(2)
    p1 = np.asarray(p1, dtype=float).reshape(2)

    d = p1 - p0
    L = float(np.linalg.norm(d))
    if L <= 0:
        return
    t_hat = d / L
    n_hat = np.array([-t_hat[1], t_hat[0]], dtype=float)

    rel = xy_a - p0
    s = rel @ t_hat              # distance along line
    dist = rel @ n_hat           # signed distance from line

    if eps is None:
        # Heuristic: ~0.35 of the vertex spacing.
        m = 3 ** (layer + 1)
        eps = 0.35 * (float(H9K.radical.W) / float(m))

    mask = np.abs(dist) <= float(eps)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return

    # Limit count for plotting speed; keep uniform coverage along s.
    order = np.argsort(s[idx])
    idx = idx[order]
    if idx.size > max_points:
        step = int(np.ceil(idx.size / max_points))
        idx = idx[::step]

    delta = xy_b - xy_a
    dp = delta[idx] @ t_hat      # parallel component
    dn = delta[idx] @ n_hat      # perpendicular component
    ss = s[idx]

    fig = plt.figure(figsize=(12, 6), dpi=200, frameon=False)
    ax = fig.add_subplot(1, 1, 1)
    ax.axhline(0.0, linewidth=1.0, alpha=0.4)
    ax.plot(ss, dp, linewidth=1.0, label='d_par')
    ax.plot(ss, dn, linewidth=1.0, label='d_perp')
    ax.set_xlabel('s (distance along line)', fontsize=12)
    ax.set_ylabel('displacement component (b_oct units)', fontsize=12)
    ax.legend(loc='best', fontsize=11)
    ax.set_title(f"line_{title} (eps={float(eps):.3e}, n={idx.size})", fontsize=14)
    ax.grid(True, linewidth=0.5, alpha=0.25)

    plt.tight_layout()
    file_title = f"line_{title}_L{layer}_o{octant_id}"
    plt.savefig(f"output/{file_title}.jpg", dpi=150)
    plt.close()


def plot_density(b_oct, c_den, vts, trx, octant_id, layer, marker, oc_vtx=None):
    cmp = H9O.oid_cmp[octant_id]
    # net_mode = H9O.oid_mo[octant_id]
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

    c_mae = float(np.mean(abs_val))
    c_std = float(np.std(c_val))
    c_min = float(np.min(c_val))
    c_max = float(np.max(c_val))
    c_p90 = np.percentile(abs_val, 90)
    c_p99 = np.percentile(abs_val, 99)

    print(
        f"mae: {c_mae}; std: {c_std}; min: {c_min}; max: {c_max}; p90: {c_p90}; p99: {c_p99}; "
        f"clip99(abs): {clip}; worst(abs): {worst_abs} (val={worst_val}) tri={worst} "
        f"touches_corner={touches_corner}"
    )
    file_title = f"{marker}_L{layer}"
    plot_grid(b_grid, fc_n=acol_norm, title=file_title, text=f'mae:{c_mae}; std:{c_std}')


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
    return verts, trx, oc_vtx, oc_edg, locs


def grid(rg: Registrar, layer: int = 3, octant_id: int = 0, plotting: bool = False, save=True):
    """
    Triangular grid will be 9 triangles per octant at hex_layer 0.
    At each subsequent hex_layer, the number of triangles will increase by 9 per triangle.
    So the number of triangles will be 9**(1+hex_layer) per octant.
    """
    b_oct = rg.domain('b_oct')
    g_gcd = rg.domain('g_gcd')
    cmp = H9O.oid_cmp[octant_id]
    verts, trx, oc_vtx, oc_edg, locs = get_grid(layer=layer, octant_id=octant_id)

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
            locs=locs,      # barycentric locations
        )


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


def get_ideal_corners(mode: int):
    """
    Returns ULP-precise [Left, Right, Apex] corners based on H9K constants.
    """
    tr = H9K.limits.TR
    vf = H9K.limits.VF
    vc = H9K.limits.VC

    if int(mode) == 0:  # Pointing DOWN
        # Base is at Top (VC), Apex is at Bottom (VF)
        left  = np.array([-tr, vc])
        right = np.array([ tr, vc])
        apex  = np.array([0.0, vf])
    else:               # Pointing UP (Mode 1)
        # Base is at Bottom (VF), Apex is at Top (VC)
        left  = np.array([-tr, vf])
        right = np.array([ tr, vf])
        apex  = np.array([0.0, vc])

    return np.array([left, right, apex])

def barycentric_uvwx(xy_c: np.ndarray):
    """Return (u, v, w, x_n=x/t) for canonical triangle."""
    xy_c = np.asarray(xy_c, dtype=float)
    x = xy_c[:, 0]
    y = xy_c[:, 1]
    u = (y - H9K.limits.VF) / H9K.H
    one_minus_u = 1.0 - u
    x_n = x / float(H9K.limits.TR)
    v = 0.5 * (one_minus_u - x_n)
    w = 0.5 * (one_minus_u + x_n)
    return u, v, w, x_n

def poly_features(u: np.ndarray, x_n: np.ndarray, total_deg: int):
    """Monomials u^i * x_n^j with i+j <= total_deg."""
    cols = []
    powers = []
    for i in range(total_deg + 1):
        for j in range(total_deg + 1 - i):
            cols.append((u ** i) * (x_n ** j))
            powers.append((i, j))
    return np.stack(cols, axis=1), powers  # (n, n_feat)

############################################################

def snap_boundary_analytic(xy, oc_edg, oc_vtx, mode):
    """
    Snaps boundary points to the ideal H9K geometry while allowing
    them to slide freely along the lines.
    """
    out = xy.copy()
    corners = get_ideal_corners(mode)

    # 1. Lock Corners Exactly
    # We map the 3 corners in the order [Left, Apex, Right] usually,
    # but let's stick to the order oc_vtx implies.
    # NOTE: Ensure oc_vtx indices match [Left, Right, Apex] order
    # or loop through to find closest.
    # For safety, let's just snap oc_vtx to the *nearest* ideal corner.
    for idx in oc_vtx:
        dists = np.sum((corners - out[idx]) ** 2, axis=1)
        out[idx] = corners[np.argmin(dists)]

    # 2. Project Edges to Ideal Lines
    if len(oc_edg) == 0: return out

    edg_pts = out[oc_edg]

    # Define line segments
    lines = [
        (corners[0], corners[1]),  # Base
        (corners[1], corners[2]),  # Right
        (corners[2], corners[0])  # Left
    ]

    starts = np.array([l[0] for l in lines])
    vecs = np.array([l[1] - l[0] for l in lines])
    lens2 = np.sum(vecs ** 2, axis=1)

    # Vectorized Projection (Same as before, but using Ideal Lines)
    P = edg_pts[:, None, :]
    A = starts[None, :, :]
    V = vecs[None, :, :]

    AP = P - A
    t = np.sum(AP * V, axis=2) / lens2[None, :]  # (N, 3)
    t_clamped = np.clip(t, 0.0, 1.0)

    projs = A + t_clamped[:, :, None] * V

    # Distance to each ideal line
    dists2 = np.sum((P - projs) ** 2, axis=2)
    best = np.argmin(dists2, axis=1)

    rows = np.arange(len(edg_pts))
    out[oc_edg] = projs[rows, best, :]

    return out

def create_ghost_padding(pts, weights):
    """
    Creates a 'Super-Set' of points including the original octant
    plus 3 ghost reflections (one for each side).

    Parameters:
    pts (Nx2): The 2D coordinates of your grid (t_p or a_p).
    weights (N): The 1D Sinkhorn weights.
     """

    # Define the 3 boundary lines from the corners
    corners = get_ideal_corners(0)

    # 3 Lines: Base(L->R), Right(R->A), Left(A->L)
    lines = [
        (corners[0], corners[1]),
        (corners[1], corners[2]),
        (corners[2], corners[0])
    ]

    super_pts = [pts]
    super_w = [weights]

    # Track which indices are 'real' vs 'ghost' so we can crop later
    real_mask = [np.ones(len(pts), dtype=bool)]

    for i, (p_start, p_end) in enumerate(lines):
        # --- 1. Calculate Reflection Geometry ---

        # Line vector and normal
        line_vec = p_end - p_start
        # Perpendicular (-y, x) gives us the normal vector
        normal = np.array([-line_vec[1], line_vec[0]])
        norm_len = np.linalg.norm(normal)
        if norm_len == 0:
            continue
        normal /= norm_len

        # Vector from line start to all points
        vec_to_pts = pts - p_start

        # Signed distance from line (dot product with normal)
        dist_perp = np.dot(vec_to_pts, normal)

        # Reflection Formula: R = P - 2 * dist * normal
        # This effectively flips the point to the other side of the line
        ghost_xy = pts - 2 * normal * dist_perp[:, None]

        # --- 2. Mirror the Weights ---
        # We copy the EXACT weights from the real side.
        # This creates the "Symmetry Cancellation" that keeps the line straight.
        ghost_w = weights.copy()

        # Append to lists
        super_pts.append(ghost_xy)
        super_w.append(ghost_w)
        real_mask.append(np.zeros(len(pts), dtype=bool))

    # Stack everything into one massive arrays
    full_pts = np.vstack(super_pts)
    full_w = np.concatenate(super_w)
    full_mask = np.concatenate(real_mask)

    # Renormalize weights (Sinkhorn requires sum=1)
    full_w /= full_w.sum()
    return full_pts, full_w, full_mask


def afl(reg, b_oct, octant_id, mode, layer, marker, iterations):
    # --- CONFIGURATION ---
    # Use the stable settings from your heatmap run
    REG = reg
    FEEDBACK_ITERATIONS = iterations
    # FEEDBACK_STRENGTH = 0.75  # Slightly aggressive to kill the bands

    print(f"--- RUNNING FEEDBACK LOOP (L3 / Full Ghost) ---")
    print(f"Goal: Flatten the Red/Blue edge bands.")

    # 1. Load Data (L3)
    cmp, xy_vert, v_ell, oc_vtx, oc_edg, t_grid, locs = load_grid(layer=layer)
    pts = Points(xy_vert, b_oct, cmp)
    a_p = pts.coords
    t_p = pts.coords.copy()

    # 2. Setup Weights (Standard)
    num = len(pts)
    a_w = np.ones(num, dtype=np.float64) / num
    b_w_raw = np.exp(v_ell - np.median(v_ell))
    b_w_base = np.clip(b_w_raw, np.percentile(b_w_raw, 2), np.percentile(b_w_raw, 98))
    # Apply your boosts here (Center/Corner boosts from your main script)...
    # [Insert your boosting logic here, resulting in 'b_w_original']
    b_w_original = b_w_base / b_w_base.sum()  # Placeholder if you don't paste the boost code

    current_b_w = b_w_original.copy()

    # 3. Create Full Ghosts ONCE (Geometry doesn't change)
    # We use Full Ghost (limit=None) because L3 fits in RAM.
    # This guarantees stability.
    print("Generating Full Ghosts...")
    ga, gaw_static, _ = create_ghost_padding(a_p, a_w)
    gt_coords, _, mask = create_ghost_padding(t_p, b_w_original)

    # Pre-calc Matrix (Optimization)
    print("Pre-calculating Matrix...")
    M_cache = ot.dist(ga, gt_coords, metric="sqeuclidean")
    M_cache /= M_cache.max()
    x_prime = None
    v_prev_corr = 1.0

    # --- FEEDBACK LOOP ---
    for i in range(FEEDBACK_ITERATIONS):
        print(f"\n[Iter {i + 1}] Solving...")

        # A. Generate Weights
        # We must regenerate the ghost weights because current_b_w changes every loop
        _, full_b_w, _ = create_ghost_padding(t_p, current_b_w)

        # CRITICAL FIX for Stability: Renormalize sums!
        # Even in Full Ghost, floating point drift can occur.
        gaw_static /= gaw_static.sum()
        full_b_w /= full_b_w.sum()

        # B. Solve
        gamma_log = ot.sinkhorn(
            gaw_static, full_b_w, M_cache,
            reg=REG,
            # method='sinkhorn_log',
            numItermax=50000, stopThr=1e-7, verbose=False
        )

        # C. Reconstruct
        row_sum = gamma_log.sum(axis=1, keepdims=True)
        x_super = (gamma_log @ gt_coords) / np.maximum(row_sum, 1e-30)
        x_prime = x_super[mask]

        # D. Snap
        x_prime = snap_boundary_analytic(x_prime, oc_edg, oc_vtx, mode)

        # E. Measure Error (Triangle Areas)
        dpts = Points(x_prime, b_oct, cmp)
        gpts = rg.project(dpts, [b_oct, g_gcd])
        t_coords_gcd = np.array([gpts.coords[v] for t in t_grid for v in t])
        areas = wgs84_area(rg, Points(t_coords_gcd, g_gcd), 3)

        c_ideal = np.sum(areas) / len(areas)
        ratios = areas / c_ideal
        c_total = np.sum(areas)  # sum of all areas = entire octant.
        c_ideal = c_total / (9 ** (layer + 1))  # number of triangles
        c_val = areas / c_ideal - 1.0  # normalise and zero
        abs_val = np.abs(c_val)

        mae = np.mean(np.abs(ratios - 1.0))
        p99 = np.percentile(abs_val, 99)

        print(f"   MAE: {mae:.6f} | Min/Max Ratio: {ratios.min():.3f} / {ratios.max():.3f}; P99:{p99:.5f}")
        # print(f"   (Ideal is 1.0. Outer Band > 1.0, Inner Band < 1.0)")

        # F. Update Weights (The Correction)
        v_corr = np.zeros_like(current_b_w)
        v_cnt = np.zeros_like(current_b_w)

        for t_idx, tri in enumerate(t_grid):
            v_corr[tri] += ratios[t_idx]
            v_cnt[tri] += 1

        mask_v = v_cnt > 0
        v_corr[mask_v] /= v_cnt[mask_v]
        v_corr[~mask_v] = 1.0

        # F. Correction with Diffusion (Anti-Ringing)
        v_corr = np.zeros(len(current_b_w))
        v_cnt = np.zeros(len(current_b_w))
        for t_idx, tri in enumerate(t_grid):
            v_corr[tri] += ratios[t_idx]
            v_cnt[tri] += 1
        mask_v = v_cnt > 0
        v_corr[mask_v] /= v_cnt[mask_v]
        v_corr[~mask_v] = 1.0

        # --- ERROR DIFFUSION STEP ---
        strength = 1.0
        # Simple averaging with previous value (Momentum) to dampen oscillation
        if i > 0:
            v_corr = 0.7 * v_corr + 0.3 * v_prev_corr
        v_prev_corr = v_corr.copy()

        current_b_w *= np.power(v_corr, strength)
        current_b_w /= current_b_w.sum()

        print(f"   iter:{i:02d}; strength:{strength:.6f}")

        # Save Plot
        marker = f"g5afl_iter{i:02d}"
        plot_density(b_oct, areas, x_prime, t_grid, octant_id, layer=3, marker=marker, oc_vtx=oc_vtx)

        np.savez(
            f"output/data/{marker}.npz",
            octant_id=octant_id,
            cmp=cmp,
            a_p=a_p,
            t_p=x_prime,
            grid=t_grid,
            layer=layer,
        )
    print("Done. Check the last plot. The bands should be gone.")


if __name__ == '__main__':
    rg = Registrar()
    b_oct = rg.domain('b_oct')
    g_gcd = rg.domain('g_gcd')
    octant_id = 0
    mode = H9O.oid_mo[octant_id]
    layer = 3
    iterations = 100

    marker = 'g5_afl_l3'
    afl(0.00020869, b_oct, octant_id, mode, layer, marker, iterations)
