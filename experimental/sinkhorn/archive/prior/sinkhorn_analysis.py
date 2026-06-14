# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
from pathlib import Path
import numpy as np
from hhg9 import Registrar, Points
from hhg9.h9.classifier import location
from hhg9.algorithms.distance import wgs84_area
from hhg9.h9 import (H9O, H9K)
from hhg9.h9.protocols import BaryLoc

def save_eb_model_npz(path: str,
                      *,
                      octant_id: int,
                      layer: int,
                      mode: int,
                      cmp: np.ndarray,
                      corners_xy: np.ndarray,
                      edge_maps: dict,
                      bubble: dict,
                      alpha: float,
                      alpha_raw: float,
                      alpha_max: float,
                      alpha_strategy: str,
                      alpha_objective: str,
                      # optional: reproducibility knobs
                      sens_k: float = None,
                      sens_max: float = None,
                      irls_enable: bool = None,
                      irls_beta: float = None,
                      irls_pct: float = None,
                      irls_max: float = None,
                      irls_p: float = None,
                      irls_cap: float = None,
                      u_deg: int = None,
                      x2_deg: int = None,
                      env_alpha: float = None,
                      ridge: float = None,
                      eps_uvw: float = None,
                      enforce_lr_symmetry: bool = True):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        octant_id=int(octant_id),
        layer=int(layer),
        mode=int(mode),
        cmp=np.asarray(cmp),
        corners_xy=np.asarray(corners_xy, dtype=float),
        enforce_lr_symmetry=bool(enforce_lr_symmetry),
        edge_maps=edge_maps,     # pickled dict
        bubble=bubble,           # pickled dict
        alpha=float(alpha),
        alpha_raw=float(alpha_raw),
        alpha_max=float(alpha_max),
        alpha_strategy=str(alpha_strategy),
        alpha_objective=str(alpha_objective),
        sens_k=np.nan if sens_k is None else float(sens_k),
        sens_max=np.nan if sens_max is None else float(sens_max),
        irls_enable=False if irls_enable is None else bool(irls_enable),
        irls_beta=np.nan if irls_beta is None else float(irls_beta),
        irls_pct=np.nan if irls_pct is None else float(irls_pct),
        irls_max=np.nan if irls_max is None else float(irls_max),
        irls_p=np.nan if irls_p is None else float(irls_p),
        irls_cap=np.nan if irls_cap is None else float(irls_cap),
        u_deg=-1 if u_deg is None else int(u_deg),
        x2_deg=-1 if x2_deg is None else int(x2_deg),
        env_alpha=np.nan if env_alpha is None else float(env_alpha),
        ridge=np.nan if ridge is None else float(ridge),
        eps_uvw=np.nan if eps_uvw is None else float(eps_uvw),
    )

def load_eb_model_npz(path: str) -> dict:
    repo = np.load(path, allow_pickle=True)
    out = {k: repo[k] for k in repo.files}

    # unwrap pickled dicts/strings
    for k in ("edge_maps", "bubble", "alpha_strategy", "alpha_objective"):
        if k in out and isinstance(out[k], np.ndarray) and out[k].shape == ():
            out[k] = out[k].item()

    # convenience casts
    out["octant_id"] = int(out["octant_id"])
    out["hex_layer"] = int(out["hex_layer"])
    out["net_mode"] = int(out["net_mode"])
    out["alpha"] = float(out["alpha"])
    out["alpha_raw"] = float(out["alpha_raw"])
    out["alpha_max"] = float(out["alpha_max"])
    return out


# --- Loc normalization helpers ---
def loc_code(x):
    """Return an int code for a BaryLoc-like value (BaryLoc, IntEnum, int, or string)."""
    if x is None:
        return -1
    try:
        # IntEnum / Enum with .value
        return int(x.value)
    except Exception:
        pass
    try:
        return int(x)
    except Exception:
        pass
    # fall back for string labels
    if isinstance(x, str):
        s = x.strip().upper()
        if s == '' or s == 'NONE' or s == 'NULL':
            return -1
        if s == 'INT':
            return int(getattr(BaryLoc, 'INT').value)
        if s == 'EDG':
            return int(getattr(BaryLoc, 'EDG').value)
        if s == 'VTX':
            return int(getattr(BaryLoc, 'VTX').value)
    raise TypeError(f'Unsupported loc value: {x!r}')


def normalize_locs(locs: np.ndarray) -> np.ndarray:
    """Normalize a locs array to an int array of codes compatible with BaryLoc.*.value."""
    if locs is None:
        return None
    locs = np.asarray(locs)
    if locs.dtype == object or locs.dtype.kind in ('U', 'S'):
        # Vectorize mapping for object / string arrays
        v = np.vectorize(loc_code, otypes=[int])
        return v(locs)
    # numeric already
    return locs.astype(int, copy=False)

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
    locs = repo['locs'] if 'locs' in repo.files else None
    return cmp, xy_vert, v_ell, oc_vtx, oc_edg, grid, locs

def classify_locs_b_oct(xy: np.ndarray, mode: int) -> np.ndarray:
    """Compute BaryLoc codes for b_oct points via hhg9.classifier.location.

    location() expects the equilateral triangle x-axis scaling, so multiply x by H9K.R3.
    Returns int codes compatible with BaryLoc.*.value.
    """
    xy = np.asarray(xy, dtype=float)
    x_sc = xy[:, 0] * float(H9K.R3)
    y = xy[:, 1]
    locs = location(x_sc, y, int(mode))
    # Normalize to ints (handles IntEnum arrays)
    locs = np.asarray(locs)
    if locs.dtype == object:
        locs = np.vectorize(lambda z: int(z.value) if hasattr(z, 'value') else int(z), otypes=[int])(locs)
    else:
        locs = locs.astype(int, copy=False)
    return locs

def canonicalize_xy(xy: np.ndarray, mode: int) -> tuple[np.ndarray, float]:
    """Map b_oct coords to canonical net_mode-1 via y-flip."""
    s = 1.0 if int(mode) == 1 else -1.0
    xy = np.asarray(xy, dtype=float)
    xy_c = xy.copy()
    xy_c[:, 1] *= s
    return xy_c, s

def bubble_delta_symmetric(xy: np.ndarray, corners_xy: np.ndarray, warp: dict) -> np.ndarray:
    """Return the displacement field (dx,dy) produced by a symmetric bubble warp."""
    mode = int(warp['net_mode'])
    xy_c, s = canonicalize_xy(xy, mode)
    corners_c, _ = canonicalize_xy(corners_xy, mode)

    base_y = float(warp['base_y'])
    apex_y = float(warp['apex_y'])
    t = float(warp['t'])

    u, v, w, x_n = barycentric_uvwx(xy_c, base_y, apex_y, t)
    uvw = u * v * w
    env_alpha = float(warp.get('env_alpha', 1.0))
    env = np.power(np.maximum(uvw, 0.0), env_alpha)

    fy, fx, _ = _sym_bubble_features(u, x_n, u_deg=int(warp['u_deg']), x2_deg=int(warp['x2_deg']))
    dx_c = (env[:, None] * fx) @ np.asarray(warp['cx'])
    dy_c = (env[:, None] * fy) @ np.asarray(warp['cy'])

    # Map delta back to original net_mode coordinates
    dx = dx_c
    dy = dy_c * s
    return np.stack([dx, dy], axis=1)

def triangle_params_from_corners(corners_c: np.ndarray) -> tuple[float, float, float]:
    """Infer (base_y, apex_y, t) from the 3 canonical-net_mode corner points."""
    corners_c = np.asarray(corners_c, dtype=float).reshape(3, 2)
    apex_i = int(np.argmax(corners_c[:, 1]))
    base_idx = [i for i in range(3) if i != apex_i]
    apex_y = float(corners_c[apex_i, 1])
    base_y = float(np.mean(corners_c[base_idx, 1]))
    x0 = float(corners_c[base_idx[0], 0])
    x1 = float(corners_c[base_idx[1], 0])
    t = 0.5 * abs(x1 - x0)
    return base_y, apex_y, t

def barycentric_uvwx(xy_c: np.ndarray, base_y: float, apex_y: float, t: float):
    """Return (u, v, w, x_n=x/t) for canonical triangle."""
    xy_c = np.asarray(xy_c, dtype=float)
    x = xy_c[:, 0]
    y = xy_c[:, 1]
    denom = float(apex_y - base_y)
    u = (y - base_y) / denom
    one_minus_u = 1.0 - u
    x_n = x / float(t)
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

def _project_and_get_t(pts: np.ndarray, v0: np.ndarray, v1: np.ndarray):
    """Project pts onto segment v0->v1 and return (proj, t in [0,1])."""
    pts = np.asarray(pts, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    v1 = np.asarray(v1, dtype=float)
    e = v1 - v0
    ee = float(np.dot(e, e))
    if ee == 0.0:
        t = np.zeros((pts.shape[0],), dtype=float)
        proj = np.repeat(v0[None, :], pts.shape[0], axis=0)
        return proj, t
    t = ((pts - v0) @ e) / ee
    t = np.clip(t, 0.0, 1.0)
    proj = v0 + t[:, None] * e
    return proj, t

# --- Edge+Bubble warp helpers ---
def _edge_id_and_t_for_points(
    xy: np.ndarray,
    corners: np.ndarray,
    mode: int,
    eps: float = 1e-12,
):
    """Robust edge-id + t for (near-)boundary points using barycentric coords.

    corners must be ordered as [left, right, apex].

    Edge ids correspond to directed segments:
      0: left -> right   (base)        u == 0
      1: right -> apex   (right side)  v == 0
      2: apex -> left    (left side)   w == 0

    Returns:
      edge_id: (n,) int in {0,1,2}
      t:       (n,) float in [0,1] along the directed segment above
    """
    xy = np.asarray(xy, dtype=float)
    corners = np.asarray(corners, dtype=float).reshape(3, 2)

    # Canonicalize so u/v/w meanings are stable across net_mode
    xy_c, _ = canonicalize_xy(xy, int(mode))
    corners_c, _ = canonicalize_xy(corners, int(mode))

    base_y, apex_y, t_half = triangle_params_from_corners(corners_c)
    u, v, w, x_n = barycentric_uvwx(xy_c, base_y, apex_y, t_half)

    uvw = np.stack([u, v, w], axis=1)
    edge_id = np.argmin(uvw, axis=1).astype(int)

    t_out = np.empty(xy.shape[0], dtype=float)

    # Edge 0: base left->right: x_n ∈ [-1,+1]
    m0 = edge_id == 0
    if np.any(m0):
        t_out[m0] = 0.5 * (x_n[m0] + 1.0)

    # Edge 1: right->apex: u goes 0->1
    m1 = edge_id == 1
    if np.any(m1):
        t_out[m1] = u[m1]

    # Edge 2: apex->left: u goes 1->0 so t = 1-u
    m2 = edge_id == 2
    if np.any(m2):
        t_out[m2] = 1.0 - u[m2]

    t_out = np.clip(t_out, 0.0, 1.0)
    return edge_id, t_out

def build_edge_maps(a_xy: np.ndarray,
                    x_prime: np.ndarray,
                    corners_xy: np.ndarray,
                    mode: int,
                    oc_edg: np.ndarray,
                    enforce_lr_symmetry: bool = True):
    corners_xy = np.asarray(corners_xy, dtype=float).reshape(3, 2)
    oc_edg = np.asarray(oc_edg, dtype=int).ravel()

    a_edge = np.asarray(a_xy, dtype=float)[oc_edg]
    b_edge = np.asarray(x_prime, dtype=float)[oc_edg]

    # edge ids + parameters for BOTH sets; group by source edge id
    edge_id_src, t_src = _edge_id_and_t_for_points(a_edge, corners_xy, int(mode))

    # IMPORTANT: compute destination t on the SAME edge as the source.
    # Using `_edge_id_and_t_for_points(b_edge, ...)` can mis-assign near-corner points to
    # an adjacent edge, which folds the edge map and can teleport vertices (tri=6560).
    lines = [(corners_xy[0], corners_xy[1]), (corners_xy[1], corners_xy[2]), (corners_xy[2], corners_xy[0])]

    maps = {}
    for j in range(3):
        m = edge_id_src == j
        if not np.any(m):
            maps[j] = {'t_raw': np.array([0.0, 1.0]), 't_map': np.array([0.0, 1.0])}
            continue

        ts = t_src[m]

        # Destination parameters measured on this SAME edge (source edge j)
        idx_edge = np.where(m)[0]
        v0, v1 = lines[j]
        _, td = _project_and_get_t(b_edge[idx_edge], v0, v1)

        # Topology-preserving 1D transport along the edge:
        # sort source parameters, sort destination parameters independently, then pair by rank.
        # This removes crossings while preserving the destination distribution.
        order_ts = np.argsort(ts)
        ts_s = ts[order_ts]
        td_s = np.sort(td)

        # Dedupe for np.interp (ts may contain exact duplicates).
        # For duplicates, average the corresponding destination parameters so the map
        # preserves the rank-paired distribution without arbitrary selection.
        ts_u, inv, counts = np.unique(ts_s, return_inverse=True, return_counts=True)
        td_u = np.zeros_like(ts_u, dtype=float)
        np.add.at(td_u, inv, td_s)
        td_u /= counts

        if ts_u.size < 2:
            ts_u = np.array([0.0, 1.0])
            td_u = np.array([0.0, 1.0])

        maps[j] = {'t_raw': ts_u, 't_map': np.clip(td_u, 0.0, 1.0)}

    if enforce_lr_symmetry:
        # corners_xy=[left,right,apex] => side edges are 1 (right->apex) and 2 (apex->left)
        t1r, t1m = maps[1]['t_raw'], maps[1]['t_map']
        t2r, t2m = maps[2]['t_raw'], maps[2]['t_map']

        # reflect edge2 to align direction: t_ref = 1 - t
        t2r_ref = 1.0 - t2r
        t2m_ref = 1.0 - t2m

        grid = np.linspace(0.0, 1.0, 2049)
        m1 = np.interp(grid, t1r, t1m)

        ord2 = np.argsort(t2r_ref)
        m2 = np.interp(grid, t2r_ref[ord2], t2m_ref[ord2])

        m_avg = np.clip(0.5 * (m1 + m2), 0.0, 1.0)

        # Edge 1 native t runs: right(0) -> apex(1), which matches `grid` meaning.
        maps[1] = {'t_raw': grid, 't_map': m_avg}

        # Edge 2 native t runs: apex(0) -> left(1).
        # Our symmetry coordinate is s = 1 - t2 (left(0) -> apex(1)).
        # We therefore need: t2' = 1 - m_avg(s) = 1 - m_avg(1 - t2).
        t2_grid = grid  # native t2 in [0,1] (apex->left)
        s = 1.0 - t2_grid
        m_at_s = np.interp(s, grid, m_avg)
        t2_map = 1.0 - m_at_s
        maps[2] = {'t_raw': t2_grid, 't_map': np.clip(t2_map, 0.0, 1.0)}

    return maps


def apply_edge_maps_to_points(xy: np.ndarray,
                              corners_xy: np.ndarray,
                              mode: int,
                              edge_maps: dict,
                              locs: np.ndarray):
    """Apply edge reparameterisation to points flagged as EDG/VTX.

    Points on EDG are moved along their nearest edge via t -> t'.
    VTX are left unchanged.

    Returns a new array.
    """
    xy = np.asarray(xy, dtype=float)
    out = xy.copy()

    corners_xy = np.asarray(corners_xy, dtype=float).reshape(3, 2)
    lines = [(corners_xy[0], corners_xy[1]), (corners_xy[1], corners_xy[2]), (corners_xy[2], corners_xy[0])]

    locs_i = normalize_locs(locs)
    is_edg = (locs_i == int(BaryLoc.EDG.value))
    idx = np.flatnonzero(is_edg)
    if idx.size == 0:
        return out

    pts = out[idx]
    edge_id, t = _edge_id_and_t_for_points(pts, corners_xy, int(mode))

    for j in range(3):
        m = edge_id == j
        if not np.any(m):
            continue

        tr = edge_maps[j]['t_raw']
        tm = edge_maps[j]['t_map']
        t_new = np.interp(t[m], tr, tm)

        v0, v1 = lines[j]
        e = v1 - v0
        out[idx[m]] = v0 + np.outer(t_new, e)

    return out


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


def fit_bubble_symmetric(a_xy: np.ndarray,
                         b_xy: np.ndarray,
                         corners_xy: np.ndarray,
                         mode: int,
                         locs: np.ndarray,
                         u_deg: int = 3,
                         x2_deg: int = 1,
                         env_alpha: float = 1.0,
                         ridge: float = 0.0,
                         eps_uvw: float = 0.0,
                         sens_weights: np.ndarray | None = None):
    """Fit a symmetric interior bubble warp on residuals after edge mapping.

    Boundary constraint is analytic: we multiply by uvw.

    eps_uvw != 0 turns on preconditioning: weight rows by 1/sqrt(max(uvw, eps)) (capped). If eps_uvw < 0, eps is auto-chosen from an interior percentile.
    """
    a_c, s = canonicalize_xy(a_xy, mode)
    b_c, _ = canonicalize_xy(b_xy, mode)
    corners_c, _ = canonicalize_xy(corners_xy, mode)

    base_y, apex_y, t = triangle_params_from_corners(corners_c)
    u, v, w, x_n = barycentric_uvwx(a_c, base_y, apex_y, t)
    uvw = u * v * w
    env_alpha = float(env_alpha)
    env = np.power(np.maximum(uvw, 0.0), env_alpha)

    # Fit only on interior points.
    # Prefer `locs` if available, but fall back to a barycentric test when locs is missing
    # or uses incompatible codes (common when loading cached npz across versions).
    idx = np.array([], dtype=int)
    if locs is not None:
        locs_i = normalize_locs(locs)
        try:
            is_int = (locs_i == int(BaryLoc.INT.value))
            idx = np.flatnonzero(is_int)
        except Exception:
            idx = np.array([], dtype=int)

    if idx.size == 0:
        # Barycentric fallback: interior points have all u,v,w > tol
        tol_int = 1e-12
        idx = np.flatnonzero((u > tol_int) & (v > tol_int) & (w > tol_int))

    if idx.size == 0:
        raise ValueError('no interior points to fit (locs missing/mismatched and barycentric test found none)')

    fy, fx, powers = _sym_bubble_features(u[idx], x_n[idx], u_deg=u_deg, x2_deg=x2_deg)

    a_mat_y = (env[idx, None]) * fy
    a_mat_x = (env[idx, None]) * fx

    d = (b_c - a_c)
    dx = d[idx, 0]
    dy = d[idx, 1]

    if eps_uvw is not None and float(eps_uvw) != 0.0:
        epsv = float(eps_uvw)
        if epsv < 0.0:
            # Auto: choose a floor from interior uvw so weights are not extreme
            epsv = float(np.percentile(env[idx], 10))
        epsv = float(np.maximum(epsv, 1e-12))

        # Gentler than 1/uvw and reduces fold-inducing coefficient blow-ups
        wrow = 1.0 / np.sqrt(np.maximum(env[idx], epsv))

        # Cap weights so a few near-edge interior points can't dominate the fit
        wcap = float(np.percentile(wrow, 99))
        wrow = np.minimum(wrow, wcap)

        a_mat_x = a_mat_x * wrow[:, None]
        a_mat_y = a_mat_y * wrow[:, None]
        dx = dx * wrow
        dy = dy * wrow

    # Optional sensitivity weights (e.g. from AK area-error sensitivity).
    # We use sqrt so weights scale like a standard weighted least squares.
    if sens_weights is not None:
        sw = np.asarray(sens_weights, dtype=float)
        if sw.shape[0] != a_xy.shape[0]:
            raise ValueError(f"sens_weights must have shape (n_pts,), got {sw.shape} vs n_pts={a_xy.shape[0]}")
        w_sens = np.sqrt(np.maximum(sw[idx], 0.0))
        a_mat_x = a_mat_x * w_sens[:, None]
        a_mat_y = a_mat_y * w_sens[:, None]
        dx = dx * w_sens
        dy = dy * w_sens

    if ridge and ridge > 0.0:
        ata_x = a_mat_x.T @ a_mat_x
        ata_x += float(ridge) * np.eye(ata_x.shape[0])
        cx = np.linalg.solve(ata_x, a_mat_x.T @ dx)

        ata_y = a_mat_y.T @ a_mat_y
        ata_y += float(ridge) * np.eye(ata_y.shape[0])
        cy = np.linalg.solve(ata_y, a_mat_y.T @ dy)
    else:
        cx, *_ = np.linalg.lstsq(a_mat_x, dx, rcond=None)
        cy, *_ = np.linalg.lstsq(a_mat_y, dy, rcond=None)

    return dict(mode=int(mode), s=float(s), base_y=base_y, apex_y=apex_y, t=float(t),
                u_deg=int(u_deg), x2_deg=int(x2_deg), powers=powers, cx=cx, cy=cy,
                kind='sym_bubble', eps_uvw=float(eps_uvw), env_alpha=float(env_alpha))


def apply_bubble_symmetric(xy: np.ndarray, corners_xy: np.ndarray, warp: dict) -> np.ndarray:
    mode = int(warp['net_mode'])
    xy_c, s = canonicalize_xy(xy, mode)
    corners_c, _ = canonicalize_xy(corners_xy, mode)

    base_y = float(warp['base_y'])
    apex_y = float(warp['apex_y'])
    t = float(warp['t'])

    u, v, w, x_n = barycentric_uvwx(xy_c, base_y, apex_y, t)
    uvw = u * v * w
    env_alpha = float(warp.get('env_alpha', 1.0))
    env = np.power(np.maximum(uvw, 0.0), env_alpha)

    fy, fx, _ = _sym_bubble_features(u, x_n, u_deg=int(warp['u_deg']), x2_deg=int(warp['x2_deg']))
    dx = (env[:, None] * fx) @ np.asarray(warp['cx'])
    dy = (env[:, None] * fy) @ np.asarray(warp['cy'])

    out_c = xy_c.copy()
    out_c[:, 0] += dx
    out_c[:, 1] += dy

    out = out_c
    out[:, 1] *= s
    return out

def load_pts_from_plot_npz(path: str) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    repo = np.load(path, allow_pickle=True)
    octant_id = int(repo["octant_id"])
    cmp = repo["cmp"]
    pts = repo["pts"]     # (n_pts,2) in b_oct
    grid = repo["grid"]   # (n_tri,3) vertex indices
    return octant_id, cmp, pts, grid

def area_rel_err(rg: Registrar, pts_xy: np.ndarray, grid: np.ndarray, cmp: np.ndarray) -> np.ndarray:
    b_oct = rg.domain("b_oct")
    g_gcd = rg.domain("g_gcd")
    gpts = rg.project(Points(pts_xy, b_oct, components=cmp), [b_oct, g_gcd])
    tri_g = gpts.coords[grid].reshape(-1, 2)
    c_den = wgs84_area(rg, Points(tri_g, g_gcd), 3)
    c_total = float(np.sum(c_den))
    c_ideal = c_total / grid.shape[0]
    return (c_den / c_ideal) - 1.0, c_den

def print_worst_triangle(rg: Registrar, pts_xy: np.ndarray, grid: np.ndarray, cmp: np.ndarray,
                         locs: np.ndarray | None, label: str, detail: bool = False):
    rel, c_den = area_rel_err(rg, pts_xy, grid, cmp)
    abs_rel = np.abs(rel)

    mae = float(np.mean(abs_rel))
    std = float(np.std(rel))
    rmin = float(np.min(rel))
    rmax = float(np.max(rel))
    p90 = float(np.percentile(abs_rel, 90))
    p99 = float(np.percentile(abs_rel, 99))

    tri = int(np.argmax(abs_rel))
    vids = grid[tri].astype(int)

    b_oct = rg.domain("b_oct")
    g_gcd = rg.domain("g_gcd")
    gpts = rg.project(Points(pts_xy, b_oct, components=cmp), [b_oct, g_gcd])

    print(f"\n[{label}] worst tri={tri}  rel={float(rel[tri]):+.6f}  abs={float(abs_rel[tri]):.6f}")
    print(f"[{label}] mae={mae:.6f}; std={std:.6f}; min={rmin:.6f}; max={rmax:.6f}; "
          f"p90={p90:.6f}; p99={p99:.6f}; clip99(abs)={p99:.6f}; worst(abs)={float(abs_rel[tri]):.6f}")

    if detail:
        ...
    return tri, vids


if __name__ == "__main__":
    rg = Registrar()

    # Debug/verbosity controls
    verbose = False            # set True to print loc counts and invariants
    tri_detail = False         # set True to print full vertex/cmp/coords for worst triangles
    cross_surface_detail = False  # set True to print a_p/x_prime/x_stage1/x_edge_bubble for the worst tri

    # load grid metadata (for locs / oc_vtx / oc_edg)
    layer = 3
    cmp0, a_p, v_ell, oc_vtx, oc_edg, grid_l3, g_locs = load_grid(layer)

    # load cached Sinkhorn surface and (optionally) cached EB surface
    sinkhorn_npz = f"output/data/c3p__rg1520_tf0250_cn1165_ct1070_L3.npz"
    # Save the EB surface we compute from this cached run (fast iteration).
    eb_npz = f"output/data/c3p_layer{layer}_eb.npz"
    oct_id, cmp_s, x_prime, grid_s = load_pts_from_plot_npz(sinkhorn_npz)
    mode = H9O.oid_mo[oct_id]

    # Fast BaryLoc classification for this octant surface (x scaled by H9K.R3)
    locs_xp = classify_locs_b_oct(x_prime, mode)

    # sanity: these must match the grid you loaded
    assert np.array_equal(grid_s, grid_l3), "grid mismatch (wrong file / hex_layer?)"
    assert np.allclose(cmp_s, cmp0), "cmp mismatch (wrong octant?)"

    # rerun Edge+Bubble ONLY (no Sinkhorn)
    left, apex, right = a_p[oc_vtx]
    corners_xy = np.array([left, right, apex])

    # Classify locations for the base grid and for the Sinkhorn surface
    locs_ap = classify_locs_b_oct(a_p, mode)

    def loc_counts(name: str, locs_arr: np.ndarray):
        n_udf = int(np.sum(locs_arr == int(BaryLoc.UDF.value)))
        n_ext = int(np.sum(locs_arr == int(BaryLoc.EXT.value)))
        n_int = int(np.sum(locs_arr == int(BaryLoc.INT.value)))
        n_edg = int(np.sum(locs_arr == int(BaryLoc.EDG.value)))
        n_vtx = int(np.sum(locs_arr == int(BaryLoc.VTX.value)))
        print(f"{name}: UDF={n_udf} EXT={n_ext} INT={n_int} EDG={n_edg} VTX={n_vtx} total={locs_arr.size}")

    if verbose:
        loc_counts('locs a_p', locs_ap)
        loc_counts('locs x_prime', locs_xp)

    # Invariants we expect (up to classifier tolerance):
    # - the three corners should stay VTX
    # - edge index set should remain EDG
    vtx_ok_ap = int(np.sum(locs_ap[oc_vtx] == int(BaryLoc.VTX.value)))
    vtx_ok_xp = int(np.sum(locs_xp[oc_vtx] == int(BaryLoc.VTX.value)))
    edg_ok_ap = int(np.sum(locs_ap[oc_edg] == int(BaryLoc.EDG.value)))
    edg_ok_xp = int(np.sum(locs_xp[oc_edg] == int(BaryLoc.EDG.value)))
    if verbose:
        print(f"vtx VTX ok: a_p {vtx_ok_ap}/{oc_vtx.size}  x_prime {vtx_ok_xp}/{oc_vtx.size}")
        print(f"edg EDG ok: a_p {edg_ok_ap}/{oc_edg.size}  x_prime {edg_ok_xp}/{oc_edg.size}")

    # Also: report how many points changed class between a_p and x_prime
    n_changed = int(np.sum(locs_ap != locs_xp))
    if verbose:
        print(f"loc class changed a_p->x_prime: {n_changed}/{locs_ap.size}")

    edge_maps = build_edge_maps(
        a_xy=a_p,
        x_prime=x_prime,
        corners_xy=corners_xy,
        mode=mode,
        oc_edg=oc_edg,
        enforce_lr_symmetry=True,
    )

    x_stage1 = apply_edge_maps_to_points(
        a_p,
        corners_xy=corners_xy,
        mode=mode,
        edge_maps=edge_maps,
        locs=locs_ap,
    )
    x_stage1[oc_vtx] = a_p[oc_vtx]

    locs_stage1 = classify_locs_b_oct(x_stage1, mode)
    if verbose:
        loc_counts('locs x_stage1', locs_stage1)

    target = x_prime.copy()
    target[oc_edg] = x_stage1[oc_edg]
    target[oc_vtx] = a_p[oc_vtx]

    # --- Sensitivity weights: accumulate |rel_xp| per vertex from incident triangles ---
    rel_xp, _ = area_rel_err(rg, x_prime, grid_l3, cmp0)
    tri_abs = np.abs(rel_xp)

    n_pts = x_prime.shape[0]
    v_sum = np.zeros((n_pts,), dtype=float)
    v_cnt = np.zeros((n_pts,), dtype=float)
    flat_vids = grid_l3.reshape(-1).astype(int)
    flat_vals = np.repeat(tri_abs, 3)
    np.add.at(v_sum, flat_vids, flat_vals)
    np.add.at(v_cnt, flat_vids, 1.0)
    v_mean = v_sum / np.maximum(v_cnt, 1.0)

    # Normalize over interior points so weights are relative and stable.
    int_idx = np.flatnonzero(locs_stage1 == int(BaryLoc.INT))
    denom = float(np.percentile(v_mean[int_idx], 95)) if int_idx.size else float(np.percentile(v_mean, 95))
    denom = max(denom, 1e-12)

    sens_k = 4.0   # strength (tune 1..8)
    sens_max = 30.0
    sens_w = 1.0 + sens_k * (v_mean / denom)
    sens_w = np.clip(sens_w, 1.0, sens_max)

    print(
        "sens_w (INT) p50/p90/p99/max:",
        float(np.percentile(sens_w[int_idx], 50)),
        float(np.percentile(sens_w[int_idx], 90)),
        float(np.percentile(sens_w[int_idx], 99)),
        float(np.max(sens_w[int_idx])),
    )

    # IRLS-style refinement: use Δrel (eb-xp) to reweight interior points and refit once.
    irls_enable = True
    irls_beta = 3.0
    irls_pct = 99.0
    irls_max = 30.0

    # Nonlinearity and soft-cap controls for pass-2 weighting.
    # tri_points>1 concentrates extra weight on the worst Δrel tail; tri_points=1 is linear.
    irls_p = 1.0
    irls_cap = float(irls_max)

    # Global bubble scale clamp. alpha_raw is the least-squares scale for the bubble delta.
    # We usually expect alpha to be near 1; allowing a small overshoot can reduce persistent under-correction.
    alpha_max = 1.02  # tune in ~[1.0, 1.10] if needed

    # How to choose alpha (bubble scale):
    # - "lsq"  : least-squares projection toward x_prime (good for matching x_prime)
    # - "area" : pick alpha to minimize equal-area error (recommended if equal-area is the goal)
    alpha_strategy = "area"   # "lsq" or "area"
    alpha_objective = "p99"   # "p99" or "mae" (objective for alpha_strategy="area")


    def _alpha_obj_value(rel: np.ndarray) -> float:
        abs_rel = np.abs(rel)
        mae = float(np.mean(abs_rel))
        p90 = float(np.percentile(abs_rel, 90))
        p99 = float(np.percentile(abs_rel, 99))
        # Hybrid: keep tail pressure but stop destroying the bulk
        return mae + 0.25 * p99 + 0.25 * p90

    # def _alpha_obj_value(rel: np.ndarray) -> float:
    #     abs_rel = np.abs(rel)
    #     if alpha_objective == "mae":
    #         return float(np.mean(abs_rel))
    #     # default: p99
    #     return float(np.percentile(abs_rel, 99))

    def choose_alpha_area(
        x_base: np.ndarray,
        delta: np.ndarray,
        alpha_hi: float,
        n_coarse: int = 21,
        n_refine: int = 15,
        refine_span: float = 0.25,
    ) -> tuple[float, float, float]:
        """Pick alpha in [0, alpha_hi] by minimizing an equal-area objective.

        Returns (alpha_best, obj_best, alpha_best_raw_like).
        The third value is just alpha_best (kept for consistent logging).
        """
        # Coarse grid
        alphas = np.linspace(0.0, float(alpha_hi), int(n_coarse))
        objs = np.empty_like(alphas)

        for k, a in enumerate(alphas):
            x_try = x_base + float(a) * delta
            x_try[oc_vtx] = a_p[oc_vtx]
            x_try[oc_edg] = x_stage1[oc_edg]
            rel_try, _ = area_rel_err(rg, x_try, grid_l3, cmp0)
            objs[k] = _alpha_obj_value(rel_try)

        k0 = int(np.argmin(objs))
        a0 = float(alphas[k0])

        # Refine around best coarse alpha
        lo = max(0.0, a0 * (1.0 - float(refine_span)))
        hi = min(float(alpha_hi), a0 * (1.0 + float(refine_span)))
        if hi <= lo + 1e-15:
            return a0, float(objs[k0]), a0

        alphas2 = np.linspace(lo, hi, int(n_refine))
        objs2 = np.empty_like(alphas2)
        for k, a in enumerate(alphas2):
            x_try = x_base + float(a) * delta
            x_try[oc_vtx] = a_p[oc_vtx]
            x_try[oc_edg] = x_stage1[oc_edg]
            rel_try, _ = area_rel_err(rg, x_try, grid_l3, cmp0)
            objs2[k] = _alpha_obj_value(rel_try)

        k1 = int(np.argmin(objs2))
        a1 = float(alphas2[k1])
        return a1, float(objs2[k1]), a1

    def run_edge_bubble(sens_w_in: np.ndarray):
        bubble_loc = fit_bubble_symmetric(
            a_xy=x_stage1,
            b_xy=target,
            corners_xy=corners_xy,
            mode=mode,
            locs=locs_stage1,
            u_deg=5,
            x2_deg=3,
            env_alpha=0.6,
            ridge=1e-5,
            eps_uvw=1e-4,
            sens_weights=sens_w_in,
        )
        if float(bubble_loc.get('env_alpha', 1.0)) > 1.0:
            print(
                f"WARNING: env_alpha={bubble_loc.get('env_alpha')} (>1) will strongly suppress near-edge corrections; typical is <1 (e.g. 0.6)."
            )
        idx_loc = np.flatnonzero(locs_stage1 == int(BaryLoc.INT))

        delta_loc = bubble_delta_symmetric(x_stage1, corners_xy=corners_xy, warp=bubble_loc)
        delta_loc[oc_vtx] = 0.0
        delta_loc[oc_edg] = 0.0
        r = (x_prime - x_stage1)[idx_loc]  # (n,2)
        d = delta_loc[idx_loc]  # (n,2)

        if alpha_strategy == "lsq":
            # Least-squares projection of residual r onto delta d
            num = float(np.sum(r * d))
            den = float(np.sum(d * d))
            alpha_raw = 0.0 if den == 0.0 else float(num / den)
            alpha = float(np.clip(alpha_raw, 0.0, float(alpha_max)))
            if alpha == float(alpha_max) and alpha_raw > float(alpha_max):
                print(
                    f"alpha clipped: raw={alpha_raw:.6f} -> {alpha:.6f} (alpha_max={float(alpha_max):.6f})"
                )
        else:
            # Equal-area optimal alpha (1D search)
            alpha, obj_best, alpha_raw = choose_alpha_area(
                x_base=x_stage1,
                delta=delta_loc,
                alpha_hi=float(alpha_max),
            )
            print(f"alpha(area) obj={alpha_objective} best={obj_best:.6f}")

        x_eb_loc = x_stage1 + alpha * delta_loc

        x_eb_loc[oc_vtx] = a_p[oc_vtx]
        x_eb_loc[oc_edg] = x_stage1[oc_edg]

        return bubble_loc, delta_loc, x_eb_loc, alpha_raw, alpha

    # Pass 1
    bubble, delta_b, x_edge_bubble, alpha_raw, alpha = run_edge_bubble(sens_w)
    print(f"alpha_raw={alpha_raw:.6f}")
    print(f"alpha={alpha:.6f} (alpha_max={float(alpha_max):.6f})")

    # Optional IRLS pass 2: build vertex weights from triangle |Δrel| then refit once.
    if irls_enable:
        rel_eb_1, _ = area_rel_err(rg, x_edge_bubble, grid_l3, cmp0)
        abs_drel_1 = np.abs(rel_eb_1 - rel_xp)

        d_sum = np.zeros((n_pts,), dtype=float)
        d_cnt = np.zeros((n_pts,), dtype=float)
        flat_d = np.repeat(abs_drel_1, 3)
        np.add.at(d_sum, flat_vids, flat_d)
        np.add.at(d_cnt, flat_vids, 1.0)
        d_mean = d_sum / np.maximum(d_cnt, 1.0)

        d_scale = float(np.percentile(d_mean[int_idx], irls_pct)) if int_idx.size else float(np.percentile(d_mean, irls_pct))
        d_scale = max(d_scale, 1e-12)

        w_d = 1.0 + float(irls_beta) * np.power(d_mean / d_scale, float(irls_p))
        w_d = np.clip(w_d, 1.0, float(irls_max))

        # Apply IRLS reweighting. Use a soft-cap so large weights don't all collapse to the hard max.
        sens_w2_raw = sens_w * w_d
        cap = float(irls_cap)
        sens_w2 = 1.0 + (sens_w2_raw - 1.0) / (1.0 + (sens_w2_raw - 1.0) / cap)

        print(
            "irls w_d (INT) p50/p90/p99/max:",
            float(np.percentile(w_d[int_idx], 50)),
            float(np.percentile(w_d[int_idx], 90)),
            float(np.percentile(w_d[int_idx], 99)),
            float(np.max(w_d[int_idx])),
        )
        print(
            "sens_w2 (INT) p50/p90/p99/max:",
            float(np.percentile(sens_w2[int_idx], 50)),
            float(np.percentile(sens_w2[int_idx], 90)),
            float(np.percentile(sens_w2[int_idx], 99)),
            float(np.max(sens_w2[int_idx])),
        )

        bubble, delta_b, x_edge_bubble, alpha_raw, alpha = run_edge_bubble(sens_w2)
        print(f"alpha_raw={alpha_raw:.6f}")
        print(f"alpha={alpha:.6f} (alpha_max={float(alpha_max):.6f})")
        eb_model_npz = f"output/data/eb_model_L{layer}.npz"
        save_eb_model_npz(
            eb_model_npz,
            octant_id=oct_id,
            layer=layer,
            mode=mode,
            cmp=cmp0,
            corners_xy=corners_xy,
            edge_maps=edge_maps,
            bubble=bubble,
            alpha=alpha,
            alpha_raw=alpha_raw,
            alpha_max=alpha_max,
            alpha_strategy=alpha_strategy,
            alpha_objective=alpha_objective,
            sens_k=sens_k,
            sens_max=sens_max,
            irls_enable=irls_enable,
            irls_beta=irls_beta,
            irls_pct=irls_pct,
            irls_max=irls_max,
            irls_p=irls_p,
            irls_cap=irls_cap,
            u_deg=5,
            x2_deg=3,
            env_alpha=0.6,
            ridge=1e-5,
            eps_uvw=1e-4,
            enforce_lr_symmetry=True,
        )
        print(f"saved eb_model_npz: {eb_model_npz}")


    # Per-point residuals against Sinkhorn target (interior only)
    ex = (x_stage1[:, 0] + alpha * delta_b[:, 0]) - x_prime[:, 0]
    ey = (x_stage1[:, 1] + alpha * delta_b[:, 1]) - x_prime[:, 1]

    # Use the same interior index set used for weighting.
    if int_idx.size == 0:
        # Fallback: compute interior indices if for some reason int_idx wasn't populated.
        int_idx = np.flatnonzero(locs_stage1 == int(BaryLoc.INT))

    ex_i = ex[int_idx]
    ey_i = ey[int_idx]
    e2_i = np.stack([ex_i, ey_i], axis=1)
    en_i = np.linalg.norm(e2_i, axis=1)

    # Percentiles for |e| (2D), and also per-axis |e_x| / |e_y|
    exa = np.abs(ex_i)
    eya = np.abs(ey_i)

    print(
        "b_oct |e| p50/p90/p99/max:",
        float(np.percentile(en_i, 50)), float(np.percentile(en_i, 90)), float(np.percentile(en_i, 99)), float(np.max(en_i)),
    )
    print(
        "b_oct |ex| p50/p90/p99/max:",
        float(np.percentile(exa, 50)), float(np.percentile(exa, 90)), float(np.percentile(exa, 99)), float(np.max(exa)),
        " |ey| p50/p90/p99/max:",
        float(np.percentile(eya, 50)), float(np.percentile(eya, 90)), float(np.percentile(eya, 99)), float(np.max(eya)),
    )

    ed = np.linalg.norm(x_stage1[oc_edg] - x_prime[oc_edg], axis=1)
    print(
        "edge |x_stage1-x_prime| p50/p90/p99/max:",
        float(np.percentile(ed, 50)),
        float(np.percentile(ed, 90)),
        float(np.percentile(ed, 99)),
        float(np.max(ed)),
    )

    rel_xp, _ = area_rel_err(rg, x_prime, grid_l3, cmp0)
    rel_eb, _ = area_rel_err(rg, x_edge_bubble, grid_l3, cmp0)
    drel = rel_eb - rel_xp
    abs_drel = np.abs(drel)

    print("Δrel=eb-xp  mae/p90/p99/max:",
          float(np.mean(abs_drel)),
          float(np.percentile(abs_drel, 90)),
          float(np.percentile(abs_drel, 99)),
          float(np.max(abs_drel)))

    tri_d = int(np.argmax(abs_drel))
    vids = grid_l3[tri_d]
    touches_edge = np.any(locs_xp[vids] == int(BaryLoc.EDG.value))
    touches_corner = np.any(locs_xp[vids] == int(BaryLoc.VTX.value))
    print("worst Δ tri", tri_d, "Δrel", float(drel[tri_d]), "touches_edge", touches_edge, "touches_corner",
          touches_corner)

    vids = grid_l3[tri_d].astype(int)
    print("tri_d vids:", vids.tolist())
    print("b_oct Δ (eb-xp):\n", (x_edge_bubble[vids] - x_prime[vids]))
    print("|Δ|:", np.linalg.norm(x_edge_bubble[vids] - x_prime[vids], axis=1))

    # If an EB file already exists, print it BEFORE overwriting so we can compare prior vs new.
    try:
        if Path(eb_npz).exists():
            _, cmp_prev, x_prev, grid_prev = load_pts_from_plot_npz(eb_npz)
            # Be defensive: only compare if this is the same surface.
            if np.array_equal(grid_prev, grid_l3) and np.allclose(cmp_prev, cmp0):
                locs_prev = classify_locs_b_oct(x_prev, mode)
                print_worst_triangle(rg, x_prev, grid_prev, cmp_prev, locs_prev, "prior eb_current", detail=tri_detail)
            else:
                print(f"prior eb_current present but incompatible (grid/cmp mismatch): {eb_npz}")
    except Exception as e:
        print(f"warning: failed to load prior eb_current: {e}")

    Path(eb_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez(eb_npz, octant_id=oct_id, cmp=cmp0, pts=x_edge_bubble, grid=grid_l3)
    if verbose:
        print(f"saved eb_npz: {eb_npz}")

    locs_eb = classify_locs_b_oct(x_edge_bubble, mode)
    if verbose:
        loc_counts('locs x_edge_bubble', locs_eb)

    # print worst triangles for both surfaces
    _, _ = print_worst_triangle(rg, x_prime, grid_l3, cmp0, locs_xp, "sinkhorn x_prime", detail=tri_detail)
    tri_bad, vids_bad = print_worst_triangle(rg, x_edge_bubble, grid_l3, cmp0, locs_eb, "edge+bubble", detail=tri_detail)

    # Cross-surface comparison for the EB worst triangle
    b_oct = rg.domain("b_oct")
    g_gcd = rg.domain("g_gcd")

    def proj3(name: str, pts_xy: np.ndarray):
        gpts = rg.project(Points(pts_xy, b_oct, components=cmp0), [b_oct, g_gcd])
        print(f"\n[tri {tri_bad} @ {name}] b_oct verts:\n", pts_xy[vids_bad])
        print(f"[tri {tri_bad} @ {name}] g_gcd verts:\n", gpts.coords[vids_bad])

    if cross_surface_detail:
        print(f"\nEB worst tri={tri_bad} vids={vids_bad.tolist()}")
        try:
            print("EB worst tri cmp:", np.asarray(cmp0)[vids_bad].tolist())
        except Exception:
            pass

        proj3('a_p', a_p)
        proj3('x_prime', x_prime)
        proj3('x_stage1', x_stage1)
        proj3('x_edge_bubble', x_edge_bubble)

    # Compare against the EB surface we just saved
    try:
        _, cmp_eb, x_eb, grid_eb = load_pts_from_plot_npz(eb_npz)
        locs_eb_saved = classify_locs_b_oct(x_eb, mode)
        print_worst_triangle(rg, x_eb, grid_eb, cmp_eb, locs_eb_saved, "saved eb_current", detail=tri_detail)
    except FileNotFoundError:
        pass
