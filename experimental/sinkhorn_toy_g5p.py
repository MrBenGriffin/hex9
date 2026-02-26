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
              fc_n: tuple = None, title: str | None = None,
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
    if title:
        ax.set_title(title, fontsize=20)
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
    # mode = H9O.oid_mo[octant_id]
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
    np.savez(
        f"output/data/{file_title}.npz",
        octant_id=octant_id,
        cmp=cmp,
        pts=vts,
        grid=trx
    )
    plot_grid(b_grid, fc_n=acol_norm, title=file_title)


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


def snap_boundary_preserve_distribution(
        xy: np.ndarray,
        oc_edg: np.ndarray,
        oc_vtx: np.ndarray
) -> np.ndarray:
    """
    Constrains boundary points to the triangle geometry but allows them
    to slide freely along the lines to preserve authalic density.
    """
    out = xy.copy()

    # 1. Hard Pin Corners (Topology requirement)
    # Ideally, Sinkhorn puts them close, but we lock them exactly.
    corners_xy = get_ideal_corners(0)
    out[oc_vtx] = corners_xy

    # 2. Project Edge Points to nearest Boundary Line
    # We do NOT sort them. We let Sinkhorn decide where they sit along the line.

    # Define lines: (A->B), (B->C), (C->A)
    # corners_xy must be [left, right, apex] or similar loopable order
    lines = [
        (corners_xy[0], corners_xy[1]),
        (corners_xy[1], corners_xy[2]),
        (corners_xy[2], corners_xy[0])
    ]

    edg_pts = out[oc_edg]

    # Vectorized projection to closest line segment
    # For every edge point, find distance to all 3 lines, pick min, project.

    # Pre-calc line vectors
    starts = np.array([l[0] for l in lines])  # (3, 2)
    ends = np.array([l[1] for l in lines])  # (3, 2)
    vecs = ends - starts  # (3, 2)
    lens2 = np.sum(vecs ** 2, axis=1)  # (3,)

    # This shape manipulation allows broadcasting:
    # P: (N, 1, 2) vs Lines: (1, 3, 2)
    P = edg_pts[:, None, :]
    A = starts[None, :, :]
    AB = vecs[None, :, :]

    # Projection factor t = (AP . AB) / |AB|^2
    AP = P - A
    dot = np.sum(AP * AB, axis=2)  # (N, 3)
    t = dot / lens2[None, :]  # (N, 3)

    # Clamped t for segment projection
    t_clamped = np.clip(t, 0.0, 1.0)

    # Projected candidates
    projections = A + t_clamped[:, :, None] * AB  # (N, 3, 2)

    # Distance squared to candidates
    dists2 = np.sum((P - projections) ** 2, axis=2)  # (N, 3)

    # Pick best line for each point
    best_line_idx = np.argmin(dists2, axis=1)

    # Select the coordinates corresponding to the best line
    # (Advanced indexing)
    n_pts = len(edg_pts)
    rows = np.arange(n_pts)
    best_projs = projections[rows, best_line_idx, :]

    out[oc_edg] = best_projs
    return out


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

def _sym_bubble_features(u: np.ndarray, x_n: np.ndarray, u_deg: int, x2_deg: int):
    """Features for symmetric bubble model.

    For dy:  u^i * (x_n^2)^j
    For dx:  x_n * u^i * (x_n^2)^j  (odd in x)

    Returns (feat_y, feat_x, powers)
    where powers are (i,j) for the underlying u^i*(x_n^2)^j part.
    """
    u = np.asarray(u, dtype=float)
    x_n = np.asarray(x_n, dtype=float)
    x2 = x_n * x_n

    cols = []
    powers = []
    for i in range(u_deg + 1):
        for j in range(x2_deg + 1):
            cols.append((u ** i) * (x2 ** j))
            powers.append((i, j))

    base = np.stack(cols, axis=1)
    feat_y = base
    feat_x = base * x_n[:, None]
    return feat_y, feat_x, powers


def apply_bubble_symmetric(xy: np.ndarray, corners_xy: np.ndarray, warp: dict) -> np.ndarray:
    xy_c = xy
    corners_c = corners_xy

    base_y = float(warp['base_y'])
    apex_y = float(warp['apex_y'])
    t = float(warp['t'])

    u, v, w, x_n = barycentric_uvwx(xy_c, base_y, apex_y, t)
    uvw = u * v * w

    fy, fx, _ = _sym_bubble_features(u, x_n, u_deg=int(warp['u_deg']), x2_deg=int(warp['x2_deg']))
    dx = (uvw[:, None] * fx) @ np.asarray(warp['cx'])
    dy = (uvw[:, None] * fy) @ np.asarray(warp['cy'])

    out_c = xy_c.copy()
    out_c[:, 0] += dx
    out_c[:, 1] += dy

    out = out_c
    return out


def bubble_delta_symmetric(xy: np.ndarray, corners_xy: np.ndarray, warp: dict) -> np.ndarray:
    """Return the displacement field (dx,dy) produced by a symmetric bubble warp."""
    xy_c = xy
    corners_c = corners_xy

    base_y = float(warp['base_y'])
    apex_y = float(warp['apex_y'])
    t = float(warp['t'])

    u, v, w, x_n = barycentric_uvwx(xy_c, base_y, apex_y, t)
    uvw = u * v * w

    fy, fx, _ = _sym_bubble_features(u, x_n, u_deg=int(warp['u_deg']), x2_deg=int(warp['x2_deg']))
    dx_c = (uvw[:, None] * fx) @ np.asarray(warp['cx'])
    dy_c = (uvw[:, None] * fy) @ np.asarray(warp['cy'])

    dx = dx_c
    dy = dy_c
    return np.stack([dx, dy], axis=1)

############################################################

def fit_warp_lstsq(a_xy, b_xy, corners_xy, total_deg=4, ridge=0.0):
    a_c = a_xy
    b_c = b_xy

    u, v, w, x_n = barycentric_uvwx(a_c)
    uvw = u * v * w

    feats, powers = poly_features(u, x_n, total_deg)
    A = (uvw[:, None]) * feats  # boundary-vanishing

    d = (b_c - a_c)
    dx = d[:, 0]
    dy = d[:, 1]

    if ridge and ridge > 0.0:
        ata = A.T @ A
        ata += float(ridge) * np.eye(ata.shape[0])
        cx = np.linalg.solve(ata, A.T @ dx)
        cy = np.linalg.solve(ata, A.T @ dy)
    else:
        cx, *_ = np.linalg.lstsq(A, dx, rcond=None)
        cy, *_ = np.linalg.lstsq(A, dy, rcond=None)

    return dict(total_deg=int(total_deg), powers=powers, cx=cx, cy=cy)


def apply_warp(xy, corners_xy, warp):
    xy_c = xy
    corners_c = corners_xy

    base_y = float(warp["base_y"])
    apex_y = float(warp["apex_y"])
    t = float(warp["t"])
    total_deg = int(warp["total_deg"])

    u, v, w, x_n = barycentric_uvwx(xy_c)
    uvw = u * v * w
    feats, _ = poly_features(u, x_n, total_deg)
    A = (uvw[:, None]) * feats

    out_c = xy_c.copy()
    out_c[:, 0] += A @ warp["cx"]
    out_c[:, 1] += A @ warp["cy"]

    out = out_c
    return out


def report_fit(a_xy, b_xy, b_fit_xy, boundary_idx, label):
    delta = b_xy - a_xy
    resid = b_xy - b_fit_xy
    mag = np.linalg.norm(delta, axis=1)
    rmag = np.linalg.norm(resid, axis=1)

    def pct(x, p): return float(np.percentile(x, p))
    print(f"[{label}] |Δ|   p50={pct(mag,50):.3e} p90={pct(mag,90):.3e} p99={pct(mag,99):.3e} max={float(mag.max()):.3e}")
    print(f"[{label}] |res| p50={pct(rmag,50):.3e} p90={pct(rmag,90):.3e} p99={pct(rmag,99):.3e} max={float(rmag.max()):.3e}")
    if boundary_idx is not None and len(boundary_idx):
        rb = np.linalg.norm(resid[boundary_idx], axis=1)
        print(f"[{label}] boundary |res| p50={pct(rb,50):.3e} p90={pct(rb,90):.3e} max={float(rb.max()):.3e} n={len(boundary_idx)}")


def tri_signed_area2(xy: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Return signed twice-area for each triangle in b_oct (positive = CCW)."""
    tris = np.asarray(xy, dtype=float)[grid]  # (n_tri,3,2)
    a = tris[:, 0, :]
    b = tris[:, 1, :]
    c = tris[:, 2, :]
    return (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])


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



import time
import warnings

# --- Configuration ---
SEARCH_EPOCHS = 10  # How many times to "zoom in"
SAMPLES_PER_EPOCH = 10  # How many solves per zoom level
START_LOW = 0.00004    # 4e-5 (Very aggressive lower bound)
START_HIGH = 0.00040  # 2e-4 (Safe upper bound)


def evaluate_reg(target_reg, a_p, a_w, t_p, b_w, t_grid, rg, oc_edg, oc_vtx):
    """
    Runs one instance of Ghost Sinkhorn and returns the Area MAE.
    Returns infinity if the solver fails or diverges.
    """

    rg = Registrar()
    b_oct = rg.domain('b_oct')
    g_gcd = rg.domain('g_gcd')
    octant_id = 0
    mode = H9O.oid_mo[octant_id]
    layer = 4

     # 1. Prepare Ghosts (Analytic)
    # Re-generating ghosts is cheap; keeps this function stateless
    ghost_a, full_a_w, _ = create_ghost_padding(a_p, a_w)
    ghost_t_coords, _, _ = create_ghost_padding(t_p, b_w)
    _, full_b_w, mask = create_ghost_padding(t_p, b_w)

    loss_matrix = ot.dist(ghost_a, ghost_t_coords, metric="sqeuclidean")
    loss_matrix /= loss_matrix.max()

    try:
        # Catch convergence warnings as errors so we can reject the result
        gamma_log = ot.sinkhorn(
            full_a_w, full_b_w, loss_matrix,
            reg=target_reg,
            method='sinkhorn_log',
            numItermax=30000,
            stopThr=1e-7,  # High precision
            verbose=False,
            warn=True
        )

        # 2. Reconstruct
        row_sum = gamma_log.sum(axis=1, keepdims=True)
        if np.any(row_sum < 1e-30) or np.any(np.isnan(gamma_log)):
            return float('inf'), None

        x_super = (gamma_log @ ghost_t_coords) / np.maximum(row_sum, 1e-30)
        x_prime = x_super[mask]

        # 3. Analytic Snap
        x_prime = snap_boundary_analytic(x_prime, oc_edg, oc_vtx, mode)

        # 4. Compute Metric (MAE)
        dpts = Points(x_prime, b_oct, cmp)
        gpts = rg.project(dpts, [b_oct, g_gcd])
        t_grd_new = np.array([gpts.coords[v] for t in t_grid for v in t], dtype=np.float64)
        areas = wgs84_area(rg, Points(t_grd_new, g_gcd), 3)

        # Calculate Deviation
        c_total = np.sum(areas)
        c_ideal = c_total / len(areas)
        c_val = (areas / c_ideal) - 1.0
        mae = np.mean(np.abs(c_val))

        return mae, x_prime

    except Exception as e:
        # If solver crashes (LinAlgError etc), return Inf
        return float('inf'), None


# --- Main Subdivision Loop ---
if __name__ == '__main__':
    marker = 'g5p'
    rg = Registrar()

    b_oct = rg.domain('b_oct')
    g_gcd = rg.domain('g_gcd')
    octant_id = 0
    mode = H9O.oid_mo[octant_id]
    layer = 3
    do_compare = False
    do_vectors = True
    do_line = True

    cmp, xy_vert, v_ell, oc_vtx, oc_edg, t_grid, locs = load_grid(layer)
    if locs is None:
        ẋ, y = xy_vert[:, 0] * H9K.R3, xy_vert[:, 1]
        locs = location(ẋ, y, mode)
    pts = Points(xy_vert, b_oct, cmp)
    num = len(pts)

    a_p = pts.coords
    t_p = pts.coords.copy()

    mode = H9O.oid_mo[octant_id]
    boundary_idx = np.unique(np.concatenate([oc_vtx, oc_edg]).astype(int))

    # Tuned Weights (Refined Profile) ---
    a_w = np.ones(num, dtype=np.float64) / num
    b_w_raw = np.exp(v_ell - np.median(v_ell))
    b_w_base = np.clip(b_w_raw, np.percentile(b_w_raw, 2), np.percentile(b_w_raw, 98))

    boundary_indices = np.concatenate([oc_vtx, oc_edg])
    b_w_raw = b_w_base.copy()

    # Normalize
    b_w = b_w_raw / np.sum(b_w_raw)
    b_w = b_w * (a_w.sum() / b_w.sum())

    current_low = START_LOW
    current_high = START_HIGH

    best_mae = float('inf')
    best_reg = None
    best_grid = None

    print(f"--- Starting AFK Subdivision Optimization ---")
    print(f"Range: {current_low:.2e} to {current_high:.2e}")

    for epoch in range(SEARCH_EPOCHS):
        print(f"\n[Epoch {epoch + 1}/{SEARCH_EPOCHS}] Zooming in...")

        # Create test points (Linear space)
        candidates = np.linspace(current_low, current_high, SAMPLES_PER_EPOCH)
        results = []

        for r_val in candidates:
            start_t = time.time()
            mae, grid = evaluate_reg(
                r_val, a_p, a_w, t_p, b_w,
                t_grid, rg, oc_edg, oc_vtx
            )
            dur = time.time() - start_t

            if mae < float('inf'):
                print(f"   Reg: {r_val:.2e} | MAE: {mae:.6f} | Time: {dur:.1f}s")
                results.append((mae, r_val, grid))

                # Keep track of global best
                if mae < best_mae:
                    best_mae = mae
                    best_reg = r_val
                    best_grid = grid

                    # SAVE IMMEDIATELY (Safety against crashes)
                    np.savez(f"output/AFK_BEST_L{layer}.npz",
                             grid=best_grid, reg=best_reg, mae=best_mae, layer=layer)
            else:
                print(f"   Reg: {r_val:.2e} | FAILED (Unstable)")
                results.append((float('inf'), r_val, None))

        # --- Subdivision Logic ---
        # Sort by MAE
        results.sort(key=lambda x: x[0])
        epoch_winner_mae, epoch_winner_reg, _ = results[0]

        if epoch_winner_mae == float('inf'):
            print("!!! All samples in this epoch failed. Backing off upwards.")
            # Shift window up 50%
            current_low = current_high
            current_high = current_high * 1.5
            continue

        # Define new narrow window around the winner
        # We take the interval between the neighbors of the winner
        # e.g., if indices 4, 5, 6 were tested, and 5 won, new range is 4 to 6.

        # Find index of winner in the original candidate list
        # (Using close comparison because floats)
        w_idx = np.argmin(np.abs(candidates - epoch_winner_reg))

        # Determine bounds
        left_bound = candidates[max(0, w_idx - 1)]
        right_bound = candidates[min(len(candidates) - 1, w_idx + 1)]

        # Zoom in
        current_low = left_bound
        current_high = right_bound

        print(f"-> Winner: {epoch_winner_reg:.2e} (MAE: {epoch_winner_mae:.6f})")
        print(f"-> New Window: {current_low:.2e} to {current_high:.2e}")

    print(f"\n--- Optimization Complete ---")
    print(f"Global Best Reg: {best_reg:.2e}")
    print(f"Global Best MAE: {best_mae:.6f}")

    # Final Plot
    marker = f"OPT_rg{int(best_reg * 1e7):04d}"
    rx = wgs84_area(rg, Points(best_grid, b_oct, cmp), 3)  # Re-calc for plot
    plot_density(b_oct, rx, best_grid, t_grid, octant_id, layer, marker, oc_vtx=oc_vtx)
