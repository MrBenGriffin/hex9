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


def save_eb_model_npz(
        path: str,
        *,
        octant_id: int,
        layer: int,
        mode: int,
        cmp: np.ndarray,
        corners_xy: np.ndarray,
        edge_maps: dict,
        bubble: dict,
        alpha: float,
        alpha_max: float,
        alpha_strategy: str,
        alpha_objective: str,
        meta: dict | None = None,
):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        octant_id=int(octant_id),
        layer=int(layer),
        mode=int(mode),
        cmp=np.asarray(cmp),
        corners_xy=np.asarray(corners_xy, dtype=float),
        edge_maps=edge_maps,  # pickled dict
        bubble=bubble,  # pickled dict
        alpha=float(alpha),
        alpha_max=float(alpha_max),
        alpha_strategy=str(alpha_strategy),
        alpha_objective=str(alpha_objective),
        meta={} if meta is None else dict(meta),  # pickled dict
    )
    np.savez(path, **payload)


def load_eb_model_npz(path: str) -> dict:
    repo = np.load(path, allow_pickle=True)
    out = {k: repo[k] for k in repo.files}

    # unwrap 0-d object arrays (pickled dict/string)
    for k in ("edge_maps", "bubble", "alpha_strategy", "alpha_objective", "meta"):
        if k in out and isinstance(out[k], np.ndarray) and out[k].shape == ():
            out[k] = out[k].item()

    # convenience casts
    out["octant_id"] = int(out["octant_id"])
    out["layer"] = int(out["layer"])
    out["net_mode"] = int(out["net_mode"])
    out["alpha"] = float(out["alpha"])
    out["alpha_max"] = float(out["alpha_max"])

    out.setdefault("meta", {})
    return out


def _model_score_from_rel(rel: np.ndarray, tri_mask: np.ndarray) -> dict:
    r = rel[tri_mask] if tri_mask is not None else rel
    ar = np.abs(r)
    return dict(
        mae=float(np.mean(ar)),
        p90=float(np.percentile(ar, 90)),
        p99=float(np.percentile(ar, 99)),
        worst=float(np.max(ar)),
    )


def _objective_from_score(s: dict) -> float:
    # match your alpha objective (or whatever you want to sort by)
    return float(s["mae"] + 0.25 * s["p90"] + 0.25 * s["p99"])


def save_npz_atomic(path: str, **kwargs):
    np.savez(path, **kwargs)
    Path(path).parent.mkdir(parents=True, exist_ok=True)


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


def apply_bubble_teacher_only(a_xy: np.ndarray, *, mode: int, corners_xy: np.ndarray, locs: np.ndarray, model: dict):
    """
    Teacher transfer across layers: bubble-only (no edge maps).
    Keeps VTX and EDG exactly unchanged.
    """
    a_xy = np.asarray(a_xy, float)
    locs_i = normalize_locs(locs)
    is_vtx = (locs_i == int(BaryLoc.VTX.value))
    is_edg = (locs_i == int(BaryLoc.EDG.value))

    # bubble delta evaluated on a_xy directly (or on a stage1 if you have one)
    delta = bubble_delta_symmetric(a_xy, corners_xy=np.asarray(corners_xy, float), warp=model["bubble"])
    delta[is_vtx] = 0.0
    delta[is_edg] = 0.0

    alpha = float(model.get("alpha", 1.0))
    x = a_xy + alpha * delta

    # invariants
    x[is_vtx] = a_xy[is_vtx]
    x[is_edg] = a_xy[is_edg]
    return x


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
    # env_alpha = float(warp.get('env_alpha', 1.0))
    # env = np.power(np.maximum(uvw, 0.0), env_alpha)
    env_alpha = float(warp.get('env_alpha', 1.0))
    env_order = float(warp.get('env_order', 1.0))
    env = np.power(np.maximum(uvw, 0.0), env_alpha * env_order)

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
def _guard_edge_map_endpoints(t_raw: np.ndarray,
                              t_map: np.ndarray,
                              *,
                              m_end: int = 32,
                              power: float = 2.0) -> np.ndarray:
    """
    Blend edge map back toward identity near endpoints to prevent corner-adjacent triangles
    from exploding (teleport/jump at t≈0 or t≈1).
    m_end: how many samples at each end are protected (on the *map array*, not points).
    """
    t_raw = np.asarray(t_raw, dtype=float)
    t_map = np.asarray(t_map, dtype=float).copy()
    n = int(t_raw.size)
    if n < 4 or m_end <= 0:
        return np.clip(t_map, 0.0, 1.0)

    m = min(int(m_end), n // 2)
    if m < 2:
        return np.clip(t_map, 0.0, 1.0)

    idx = np.arange(n, dtype=float)

    # weight = 1 at the endpoints, tapering to 0 over m samples
    w = np.zeros(n, dtype=float)
    w[:m] = 1.0 - (idx[:m] / float(m - 1)) ** float(power)
    w[-m:] = np.maximum(w[-m:], 1.0 - ((n - 1 - idx[-m:]) / float(m - 1)) ** float(power))

    t_map = (1.0 - w) * t_map + w * t_raw

    # hard endpoint constraints + monotone cleanup
    t_map[0] = 0.0
    t_map[-1] = 1.0
    t_map = np.maximum.accumulate(t_map)

    return np.clip(t_map, 0.0, 1.0)


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
                    enforce_lr_symmetry: bool = True,
                    guard_m_end: int = 32,
                    guard_power: float = 2.0,
                    edge_beta: float = 0.0,
                    sym_grid_n: int = 2049):
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

        # Optional bias toward identity to reduce over-aggressive edge transport.
        if float(edge_beta) > 0.0:
            b = float(np.clip(edge_beta, 0.0, 1.0))
            td_u = (1.0 - b) * td_u + b * ts_u

        maps[j] = {'t_raw': ts_u, 't_map': np.clip(td_u, 0.0, 1.0)}
        maps[j]["t_map"] = _guard_edge_map_endpoints(maps[j]["t_raw"], maps[j]["t_map"], m_end=guard_m_end,
                                                     power=guard_power)

    if enforce_lr_symmetry:
        # corners_xy=[left,right,apex] => side edges are 1 (right->apex) and 2 (apex->left)
        t1r, t1m = maps[1]['t_raw'], maps[1]['t_map']
        t2r, t2m = maps[2]['t_raw'], maps[2]['t_map']

        # reflect edge2 to align direction: t_ref = 1 - t
        t2r_ref = 1.0 - t2r
        t2m_ref = 1.0 - t2m

        grid = np.linspace(0.0, 1.0, int(sym_grid_n))
        m1 = np.interp(grid, t1r, t1m)

        ord2 = np.argsort(t2r_ref)
        m2 = np.interp(grid, t2r_ref[ord2], t2m_ref[ord2])

        m_avg = np.clip(0.5 * (m1 + m2), 0.0, 1.0)
        if float(edge_beta) > 0.0:
            b = float(np.clip(edge_beta, 0.0, 1.0))
            m_avg = (1.0 - b) * m_avg + b * grid

        # Edge 1 native t runs: right(0) -> apex(1), which matches `grid` meaning.
        maps[1] = {'t_raw': grid, 't_map': m_avg}
        maps[1]["t_map"] = _guard_edge_map_endpoints(maps[1]["t_raw"], maps[1]["t_map"], m_end=guard_m_end,
                                                     power=guard_power)

        # Edge 2 native t runs: apex(0) -> left(1).
        # Our symmetry coordinate is s = 1 - t2 (left(0) -> apex(1)).
        # We therefore need: t2' = 1 - m_avg(s) = 1 - m_avg(1 - t2).
        t2_grid = grid  # native t2 in [0,1] (apex->left)
        s = 1.0 - t2_grid
        m_at_s = np.interp(s, grid, m_avg)
        t2_map = 1.0 - m_at_s
        maps[2] = {'t_raw': t2_grid, 't_map': np.clip(t2_map, 0.0, 1.0)}
        maps[2]["t_map"] = _guard_edge_map_endpoints(maps[2]["t_raw"], maps[2]["t_map"], m_end=guard_m_end,
                                                     power=guard_power)

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
        tr = np.asarray(tr, float)
        if tr.size >= 2:
            dt = np.diff(tr)
            dt_pos = dt[dt > 0]
            if dt_pos.size:
                eps_t = 0.5 * float(np.min(dt_pos))  # half the smallest step
                t_new = np.clip(t_new, eps_t, 1.0 - eps_t)

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
                         env_order: float = 1.0,
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
    env_order = float(env_order)
    env = np.power(np.maximum(uvw, 0.0), env_alpha * env_order)

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
                kind='sym_bubble',
                eps_uvw=float(eps_uvw), env_alpha=float(env_alpha), env_order=float(env_order),
                )


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
    env_order = float(warp.get('env_order', 1.0))
    env = np.power(np.maximum(uvw, 0.0), env_alpha * env_order)

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
    pts = repo["pts"]  # (n_pts,2) in b_oct
    grid = repo["grid"]  # (n_tri,3) vertex indices
    return octant_id, cmp, pts, grid


# --- Fast planar area approximations on WGS84 from lat/lon (degrees) ---

_wgs84_a = 6378137.0
_wgs84_f = 1.0 / 298.257223563
_wgs84_e2 = _wgs84_f * (2.0 - _wgs84_f)


def _wrap_pi(x):
    """Wrap radians to [-pi, +pi] to avoid dateline issues in dlon."""
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def tri_area_wgs84_local_planar(latlon_deg_tri: np.ndarray) -> np.ndarray:
    """
    Very fast triangle areas in m^2 using a local tangent-plane metric derived from WGS84
    radii of curvature at the triangle centroid.

    latlon_deg_tri: (n_tri, 3, 2) as [lat_deg, lon_deg]
    returns: (n_tri,) positive areas in m^2
    """
    ll = np.asarray(latlon_deg_tri, dtype=float).reshape((-1, 3, 2))
    lat = np.deg2rad(ll[..., 0])
    lon = np.deg2rad(ll[..., 1])

    lat0 = lat.mean(axis=1)
    lon0 = lon.mean(axis=1)

    s = np.sin(lat0)
    w = 1.0 - _wgs84_e2 * (s * s)

    n_rad = _wgs84_a / np.sqrt(w)  # prime vertical radius
    m_rad = _wgs84_a * (1.0 - _wgs84_e2) / (w ** 1.5)  # meridian radius

    dlon = _wrap_pi(lon - lon0[:, None])
    dlat = lat - lat0[:, None]

    # local (east,north) in meters
    x = dlon * (n_rad * np.cos(lat0))[:, None]
    y = dlat * m_rad[:, None]

    # shoelace for triangles
    x0, y0 = x[:, 0], y[:, 0]
    x1, y1 = x[:, 1], y[:, 1]
    x2, y2 = x[:, 2], y[:, 2]
    area = 0.5 * np.abs(x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1))
    return area


def tri_area_sphere_equirect(latlon_deg_tri: np.ndarray, r: float = 6371008.8) -> np.ndarray:
    """
    Even cheaper than WGS84-local: spherical equirectangular metric at centroid.
    Good as 'coarse' (quality=1). Units m^2.
    """
    ll = np.asarray(latlon_deg_tri, dtype=float).reshape((-1, 3, 2))
    lat = np.deg2rad(ll[..., 0])
    lon = np.deg2rad(ll[..., 1])

    lat0 = lat.mean(axis=1)
    lon0 = lon.mean(axis=1)

    dlon = _wrap_pi(lon - lon0[:, None])
    dlat = lat - lat0[:, None]

    x = dlon * (r * np.cos(lat0))[:, None]
    y = dlat * r

    x0, y0 = x[:, 0], y[:, 0]
    x1, y1 = x[:, 1], y[:, 1]
    x2, y2 = x[:, 2], y[:, 2]
    area = 0.5 * np.abs(x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1))
    return area


def area_rel_err(
        rg: Registrar,
        pts_xy: np.ndarray,
        grid: np.ndarray,
        cmp: np.ndarray,
        *,
        quality: int = 3,  # 1=coarse, 2=medium, 3=fine
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      rel_err: (n_tri,) (area/ideal)-1
      areas:   (n_tri,) in m^2

    quality:
      3: GeographicLib via wgs84_area (slow, best)
      2: WGS84 local planar (fast, very good for small triangles)
      1: spherical equirect planar (fastest, ok for ranking)
    """
    b_oct = rg.domain("b_oct")
    g_gcd = rg.domain("g_gcd")

    gpts = rg.project(Points(pts_xy, b_oct, components=cmp), [b_oct, g_gcd])
    tri_g = gpts.coords[grid]  # (n_tri, 3, 2) lat/lon degrees

    q = int(quality)
    if q >= 3:
        tri_flat = tri_g.reshape(-1, 2)
        areas = wgs84_area(rg, Points(tri_flat, g_gcd), 3)
    elif q == 2:
        areas = tri_area_wgs84_local_planar(tri_g)
    else:
        areas = tri_area_sphere_equirect(tri_g)

    total = float(np.sum(areas))
    ideal = total / float(grid.shape[0])
    rel = (areas / ideal) - 1.0
    return rel, areas


def worst_stats(rel: np.ndarray, *, tri_mask: np.ndarray | None):
    abs_rel = np.abs(rel)
    if tri_mask is None:
        m = np.ones(rel.shape[0], dtype=bool)
    else:
        m = np.asarray(tri_mask, bool)

    abs_m = abs_rel[m]
    tri_local = int(np.argmax(abs_m))
    tri = int(np.flatnonzero(m)[tri_local])

    return dict(
        tri=tri,
        rel=float(rel[tri]),
        worst=float(abs_rel[tri]),
        mae=float(np.mean(abs_m)),
        p90=float(np.percentile(abs_m, 90)),
        p99=float(np.percentile(abs_m, 99)),
    )


# --- Extra helper functions for triangle subset reporting ---
def tri_touch_masks(grid: np.ndarray, locs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-triangle masks (touches_edge, touches_corner) based on vertex loc codes."""
    grid = np.asarray(grid, dtype=int)
    locs_i = normalize_locs(locs)

    touches_edge = np.any(locs_i[grid] == int(BaryLoc.EDG.value), axis=1)
    touches_corner = np.any(locs_i[grid] == int(BaryLoc.VTX.value), axis=1)
    return touches_edge, touches_corner


def print_subset_report(label: str,
                        rel: np.ndarray,
                        *,
                        grid: np.ndarray,
                        locs: np.ndarray | None,
                        tri_mask: np.ndarray | None):
    """Print stats for useful triangle subsets (masked, seam band, interior, corner-adjacent)."""
    if locs is None:
        # Fall back: just report the masked and full distributions.
        s_all = _model_score_from_rel(rel, None)
        s_m = _model_score_from_rel(rel, tri_mask)
        n_all = int(rel.shape[0])
        n_m = int(np.sum(tri_mask)) if tri_mask is not None else n_all
        print(f"\n[{label}] subset report (no locs):")
        print(
            f"[{label}] ALL  n={n_all} mae={s_all['mae']:.6f} p90={s_all['p90']:.6f} p99={s_all['p99']:.6f} worst={s_all['worst']:.6f}")
        print(
            f"[{label}] MASK n={n_m} mae={s_m['mae']:.6f} p90={s_m['p90']:.6f} p99={s_m['p99']:.6f} worst={s_m['worst']:.6f}")
        return

    edge, corner = tri_touch_masks(grid, locs)
    seam = edge & (~corner)
    interior = (~edge) & (~corner)
    corner_adj = corner

    def _fmt(name: str, m: np.ndarray | None):
        s = _model_score_from_rel(rel, m)
        n = int(rel.shape[0]) if m is None else int(np.sum(m))
        return f"[{label}] {name:<7} n={n} mae={s['mae']:.6f} p90={s['p90']:.6f} p99={s['p99']:.6f} worst={s['worst']:.6f}"

    print(f"\n[{label}] subset report:")
    print(_fmt('ALL', None))
    print(_fmt('MASK', tri_mask))
    print(_fmt('SEAM', seam))
    print(_fmt('INTER', interior))
    print(_fmt('CORNER', corner_adj))


def print_worst_triangle(rg: Registrar, pts_xy: np.ndarray, grid: np.ndarray, t_mask: np.ndarray, cmp: np.ndarray,
                         locs: np.ndarray | None, label: str, detail: bool = False):
    rel, _ = area_rel_err(rg, pts_xy, grid, cmp)
    s_all = worst_stats(rel, tri_mask=None)
    s_m = worst_stats(rel, tri_mask=t_mask)
    abs_rel = np.abs(rel)

    mae = float(np.mean(abs_rel))
    std = float(np.std(rel))
    rmin = float(np.min(rel))
    rmax = float(np.max(rel))
    p90 = float(np.percentile(abs_rel, 90))
    p99 = float(np.percentile(abs_rel, 99))

    tri = int(np.argmax(abs_rel))
    vids = grid[tri].astype(int)

    touches_edge = False
    touches_corner = False
    if locs is not None:
        locs_i = normalize_locs(locs)
        touches_edge = bool(np.any(locs_i[vids] == int(BaryLoc.EDG.value)))
        touches_corner = bool(np.any(locs_i[vids] == int(BaryLoc.VTX.value)))

    worst = float(abs_rel.max())
    n_worst = int(np.sum(np.isclose(abs_rel, worst, rtol=0, atol=1e-12)))
    print(f"\n[{label}] worst(abs)={worst:.6f} tied_count={n_worst} "
          f"worst tri={tri} rel={float(rel[tri]):+.6f} abs={float(abs_rel[tri]):.6f} "
          f"mae={mae:.6f}; std={std:.6f}; min={rmin:.6f}; max={rmax:.6f}; "
          f"p90={p90:.6f}; p99={p99:.6f}; clip99(abs)={p99:.6f}; worst(abs)={float(abs_rel[tri]):.6f} "
          f"touches_edge={touches_edge} touches_corner= {touches_corner}"
          )

    # masked distribution (this is what you actually optimise against)
    abs_m = abs_rel[np.asarray(t_mask, bool)]
    mae_m = float(np.mean(abs_m))
    p90_m = float(np.percentile(abs_m, 90))
    p99_m = float(np.percentile(abs_m, 99))
    print(f"\n[{label}] ALL  worst tri={s_all['tri']} rel={s_all['rel']:+.6f} abs={s_all['worst']:.6f} "
          f"mae={float(np.mean(abs_rel)):.6f}; p90={float(np.percentile(abs_rel, 90)):.6f}; p99={float(np.percentile(abs_rel, 99)):.6f}")
    print(f"[{label}] MASK worst tri={s_m['tri']} rel={s_m['rel']:+.6f} abs={s_m['worst']:.6f} "
          f"mae={mae_m:.6f}; p90={p90_m:.6f}; p99={p99_m:.6f}")

    return tri, vids


def apply_eb_model_to_points(
        a_xy: np.ndarray,
        *,
        mode: int,
        corners_xy: np.ndarray,
        locs: np.ndarray,
        model: dict,
) -> np.ndarray:
    """Apply a saved EB model to a set of b_oct points a_xy.
    Returns x_eb (same shape as a_xy).
    """
    a_xy = np.asarray(a_xy, dtype=float)
    corners_xy = np.asarray(corners_xy, dtype=float).reshape(3, 2)

    locs_i = normalize_locs(locs)
    is_vtx = (locs_i == int(BaryLoc.VTX.value))
    is_edg = (locs_i == int(BaryLoc.EDG.value))

    # 1) edges only
    x_stage1 = apply_edge_maps_to_points(
        a_xy,
        corners_xy=corners_xy,
        mode=int(mode),
        edge_maps=model["edge_maps"],
        locs=locs_i,
    )
    x_stage1[is_vtx] = a_xy[is_vtx]  # keep vertices exact

    # 2) bubble
    delta = bubble_delta_symmetric(
        x_stage1,
        corners_xy=corners_xy,
        warp=model["bubble"],
    )
    alpha = float(model["alpha"])
    x_eb = x_stage1 + alpha * delta
    # enforce invariants
    x_eb[is_vtx] = a_xy[is_vtx]
    x_eb[is_edg] = x_stage1[is_edg]
    return x_eb


def run(m_end, pwr, beta, env_order):
    # env_alpha = a  # env_alpha
    # eps_uvw = u
    # ridge = r
    lim = H9K.limits
    rg = Registrar()

    layer = 5
    oct_id = 0
    mode = H9O.oid_mo[oct_id]

    verbose = True
    tri_detail = False

    eb_npz = f"output/data/cp2_layer{layer}_eb.npz"
    best_path = f"output/data/eb2_model_L{layer}_best.npz"

    cmp0, a_p, v_ell, oc_vtx, oc_edg, grid_idx, g_locs = load_grid(layer)

    # corners for this net_mode (same for all layers)
    corners_xy = np.array([[lim.TL, lim.VC], [lim.TR, lim.VC], [0.0, lim.VF]], dtype=float)

    # classify base locs on L5 grid
    locs_ap = classify_locs_b_oct(a_p, mode)

    def loc_counts(name: str, locs_arr: np.ndarray):
        n_udf = int(np.sum(locs_arr == int(BaryLoc.UDF.value)))
        n_ext = int(np.sum(locs_arr == int(BaryLoc.EXT.value)))
        n_int = int(np.sum(locs_arr == int(BaryLoc.INT.value)))
        n_edg = int(np.sum(locs_arr == int(BaryLoc.EDG.value)))
        n_vtx = int(np.sum(locs_arr == int(BaryLoc.VTX.value)))
        print(f"{name}: UDF={n_udf} EXT={n_ext} INT={n_int} EDG={n_edg} VTX={n_vtx} total={locs_arr.size}")

    if verbose:
        loc_counts("locs a_p", locs_ap)

    # ---- Teacher: use best result ----
    teacher_model = load_eb_model_npz(best_path)
    # Reuse the previously-swept bubble hyperparameters from the current best model.
    env_alpha = float(teacher_model.get("env_alpha", 0.60))
    eps_uvw = float(teacher_model.get("eps_uvw", 0.0))
    ridge = float(teacher_model.get("ridge", 0.0))

    # Some historical models store these as NaN when unset; fall back to defaults.
    if not np.isfinite(env_alpha):
        env_alpha = 0.60
    if not np.isfinite(eps_uvw):
        eps_uvw = 0.0
    if not np.isfinite(ridge):
        ridge = 0.0
    x_prime = apply_eb_model_to_points(
        a_p,
        mode=mode,
        corners_xy=corners_xy,
        locs=locs_ap,
        model=teacher_model,
    )
    # net_mode 0
    corners_xy = np.array([[lim.TL, lim.VC], [lim.TR, lim.VC], [0.0, lim.VF]], dtype=float)

    locs_xp = classify_locs_b_oct(x_prime, mode)

    if verbose:
        loc_counts("locs teacher x_prime", locs_xp)

    vtx_ok_ap = int(np.sum(locs_ap[oc_vtx] == int(BaryLoc.VTX.value)))
    vtx_ok_xp = int(np.sum(locs_xp[oc_vtx] == int(BaryLoc.VTX.value)))
    edg_ok_ap = int(np.sum(locs_ap[oc_edg] == int(BaryLoc.EDG.value)))
    edg_ok_xp = int(np.sum(locs_xp[oc_edg] == int(BaryLoc.EDG.value)))
    print(f"vtx VTX ok: a_p {vtx_ok_ap}/{oc_vtx.size}  x_prime {vtx_ok_xp}/{oc_vtx.size}")
    print(f"edg EDG ok: a_p {edg_ok_ap}/{oc_edg.size}  x_prime {edg_ok_xp}/{oc_edg.size}")

    n_changed = int(np.sum(locs_ap != locs_xp))
    print(f"loc class changed a_p->x_prime: {n_changed}/{locs_ap.size}")

    # ---- Build fresh edge maps on L5, using teacher x_prime as target ----
    edge_maps = build_edge_maps(
        a_xy=a_p,
        x_prime=x_prime,
        corners_xy=corners_xy,
        mode=mode,
        oc_edg=oc_edg,
        guard_m_end=int(m_end),
        guard_power=float(pwr),
        edge_beta=float(beta),
        sym_grid_n=2049,
        enforce_lr_symmetry=True,
    )
    x_stage1 = apply_edge_maps_to_points(
        a_p, corners_xy=corners_xy, mode=mode, edge_maps=edge_maps, locs=locs_ap
    )
    x_stage1[oc_vtx] = a_p[oc_vtx]
    locs_stage1 = classify_locs_b_oct(x_stage1, mode)
    locs_stage1 = normalize_locs(locs_stage1).copy()
    locs_stage1[oc_vtx] = int(BaryLoc.VTX.value)
    locs_stage1[oc_edg] = int(BaryLoc.EDG.value)
    if verbose:
        loc_counts("locs x_stage1", locs_stage1)
    n_edg_to_vtx = int(np.sum((locs_stage1[oc_edg] != int(BaryLoc.EDG.value))))
    n_vtx_to_edg = int(np.sum((locs_stage1[oc_vtx] != int(BaryLoc.VTX.value))))
    print(f"stage1 loc drift: edg!=EDG {n_edg_to_vtx}/{oc_edg.size}, vtx!=VTX {n_vtx_to_edg}/{oc_vtx.size}")

    # Teacher-consistent “target”: teacher interior, but exact edges from stage1, exact vertices from a_p
    target = x_prime.copy()
    target[oc_edg] = x_stage1[oc_edg]
    target[oc_vtx] = a_p[oc_vtx]
    n_pts = a_p.shape[0]

    is_vtx_pt = np.zeros((n_pts,), dtype=bool);
    is_vtx_pt[oc_vtx] = True
    tri_touches_vtx = np.any(is_vtx_pt[grid_idx], axis=1)
    tri_mask = ~tri_touches_vtx

    # Triangle subsets for scoring (computed from base locs so they are stable across candidates).
    edge_t, corner_t = tri_touch_masks(grid_idx, locs_ap)
    seam_mask = edge_t & (~corner_t)

    # ---- Sensitivity weights from teacher area error on L5 ----
    rel_xp, _ = area_rel_err(rg, x_prime, grid_idx, cmp0)
    tri_abs = np.abs(rel_xp)

    tri_abs_m = tri_abs.copy()
    tri_abs_m[~tri_mask] = 0.0

    v_sum = np.zeros((n_pts,), dtype=float)
    v_cnt = np.zeros((n_pts,), dtype=float)

    flat_vids = grid_idx.reshape(-1).astype(int)
    flat_vals = np.repeat(tri_abs_m, 3)

    np.add.at(v_sum, flat_vids, flat_vals)
    np.add.at(v_cnt, flat_vids, 1.0)  # keep counts unchanged (fine)

    v_mean = v_sum / np.maximum(v_cnt, 1.0)

    int_idx = np.flatnonzero(locs_stage1 == int(BaryLoc.INT.value))
    denom = float(np.percentile(v_mean[int_idx], 95)) if int_idx.size else float(np.percentile(v_mean, 95))
    denom = max(denom, 1e-12)

    sens_k = 4.0
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

    # ---- Alpha controls (carry forward from teacher, but allow search) ----
    irls_enable = True
    irls_beta = 3.0
    irls_pct = 99.0
    irls_max = 30.0
    irls_p = 1.0
    irls_cap = float(irls_max)

    alpha_max = 1.05
    alpha_strategy = "area"
    alpha_objective = "p99"

    def _alpha_obj_value(rel: np.ndarray) -> float:
        # Primary goal: global authalicity (ALL triangles).
        abs_all = np.abs(rel)
        mae_all = float(np.mean(abs_all))
        p90_all = float(np.percentile(abs_all, 90))
        p99_all = float(np.percentile(abs_all, 99))

        # Secondary: lightly discourage seam-band blow-ups.
        if seam_mask is not None and np.any(seam_mask):
            abs_seam = abs_all[seam_mask]
            p99_seam = float(np.percentile(abs_seam, 99))
        else:
            p99_seam = 0.0

        # Keep seam weight small so global dominates.
        seam_lambda = 0.05
        return (mae_all + 0.25 * p90_all + 0.25 * p99_all) + seam_lambda * p99_seam

    def choose_alpha_area(
            x_base: np.ndarray,
            delta: np.ndarray,
            alpha_hi: float,
            n_coarse: int = 21,
            n_refine: int = 15,
            refine_span: float = 0.25,
    ):
        alphas = np.linspace(0.0, float(alpha_hi), int(n_coarse))
        objs = np.empty_like(alphas)
        for k, a in enumerate(alphas):
            x_try = x_base + float(a) * delta
            x_try[oc_vtx] = a_p[oc_vtx]
            x_try[oc_edg] = x_stage1[oc_edg]
            rel_try, _ = area_rel_err(rg, x_try, grid_idx, cmp0, quality=2)
            objs[k] = _alpha_obj_value(rel_try)

        k0 = int(np.argmin(objs))
        a0 = float(alphas[k0])

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
            rel_try, _ = area_rel_err(rg, x_try, grid_idx, cmp0, quality=2)
            objs2[k] = _alpha_obj_value(rel_try)

        k1 = int(np.argmin(objs2))
        on_bound = (k1 == 0) or (k1 == len(alphas2) - 1)
        a1 = float(alphas2[k1])
        return a1, float(objs2[k1]), a1, bool(on_bound)

    def run_edge_bubble(sens_w_in: np.ndarray, calc_alpha: bool = True):
        bubble_loc = fit_bubble_symmetric(
            a_xy=x_stage1,
            b_xy=target,
            corners_xy=corners_xy,
            mode=mode,
            locs=locs_stage1,
            u_deg=5,
            x2_deg=3,
            env_alpha=env_alpha,
            env_order=env_order,
            ridge=ridge,
            eps_uvw=eps_uvw,
            sens_weights=sens_w_in,
        )
        idx_loc = np.flatnonzero(locs_stage1 == int(BaryLoc.INT.value))

        delta_loc = bubble_delta_symmetric(x_stage1, corners_xy=corners_xy, warp=bubble_loc)
        delta_loc[oc_vtx] = 0.0
        delta_loc[oc_edg] = 0.0

        # default: start near teacher alpha, but we typically re-pick by area search
        _alpha = float(teacher_model.get("alpha", 0.925286))

        if calc_alpha:
            _alpha, obj_best, _, on_bound = choose_alpha_area(
                x_base=x_stage1,
                delta=delta_loc,
                alpha_hi=float(alpha_max),
            )
            print(
                f"alpha(area) q=2 obj={alpha_objective} best={obj_best:.6f} "
                f"alpha={_alpha:.6f} (alpha_max={float(alpha_max):.6f}); on_bound={on_bound}"
            )

        x_eb_loc = x_stage1 + _alpha * delta_loc
        x_eb_loc[oc_vtx] = a_p[oc_vtx]
        x_eb_loc[oc_edg] = x_stage1[oc_edg]
        return bubble_loc, delta_loc, x_eb_loc, _alpha

    # Pass 1
    bubble, delta_b, x_edge_bubble, alpha = run_edge_bubble(sens_w)
    print(f"alpha={alpha:.6f} (alpha_max={float(alpha_max):.6f})")

    # IRLS Pass 2 (optional)
    if irls_enable:
        rel_eb_1, _ = area_rel_err(rg, x_edge_bubble, grid_idx, cmp0)
        abs_drel_1 = np.abs(rel_eb_1 - rel_xp)

        # ignore corner-touching triangles in IRLS delta
        abs_drel_1_m = abs_drel_1.copy()
        abs_drel_1_m[~tri_mask] = 0.0

        d_sum = np.zeros((n_pts,), dtype=float)
        d_cnt = np.zeros((n_pts,), dtype=float)

        flat_d = np.repeat(abs_drel_1_m, 3)
        np.add.at(d_sum, flat_vids, flat_d)

        # optionally mask counts here too (recommended)
        flat_cnt = np.repeat(tri_mask.astype(float), 3)
        np.add.at(d_cnt, flat_vids, flat_cnt)

        d_mean = d_sum / np.maximum(d_cnt, 1.0)
        d_scale = float(np.percentile(d_mean[int_idx], irls_pct)) if int_idx.size else float(
            np.percentile(d_mean, irls_pct))
        d_scale = max(d_scale, 1e-12)

        w_d = 1.0 + float(irls_beta) * np.power(d_mean / d_scale, float(irls_p))
        w_d = np.clip(w_d, 1.0, float(irls_max))

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

        bubble, delta_b, x_edge_bubble, alpha = run_edge_bubble(sens_w2)
        print(f"alpha={alpha:.6f} (alpha_max={float(alpha_max):.6f})")

    # ---- Save model + surface for L5 ----
    rel_fine, _ = area_rel_err(rg, x_edge_bubble, grid_idx, cmp0, quality=3)

    # Scores on useful subsets
    score_all = _model_score_from_rel(rel_fine, None)
    score_mask = _model_score_from_rel(rel_fine, tri_mask)
    score_seam = _model_score_from_rel(rel_fine, seam_mask)

    # Primary objective: global (ALL) authalicity
    obj_all = float(score_all["mae"] + 0.25 * score_all["p90"] + 0.25 * score_all["p99"])

    # Soft constraints: discourage seam / worst regression vs current best.
    seam_lambda = 0.50
    worst_lambda = 0.10

    seam_ref = np.nan
    worst_ref = np.nan
    if Path(best_path).exists():
        m_prev = load_eb_model_npz(best_path)
        seam_ref = float(m_prev.get("seam_p99", np.nan))
        worst_ref = float(m_prev.get("all_worst", np.nan))

    seam_pen = 0.0
    worst_pen = 0.0
    if np.isfinite(seam_ref):
        seam_pen = seam_lambda * max(0.0, float(score_seam["p99"]) - seam_ref)
    if np.isfinite(worst_ref):
        worst_pen = worst_lambda * max(0.0, float(score_all["worst"]) - worst_ref)

    obj = obj_all + seam_pen + worst_pen

    print("candidate scores:", {"all": score_all, "seam": score_seam, "mask": score_mask}, "obj_all:", obj_all, "pen:",
          (seam_pen + worst_pen), "obj:", obj)
    latest_path = f"output/data/eb2_model_L{layer}.npz"  # current path
    prev_obj = None
    if Path(best_path).exists():
        m_prev = load_eb_model_npz(best_path)
        prev_obj = float(m_prev.get("obj", np.nan))

    save_kwargs = dict(
        octant_id=oct_id,
        layer=layer,
        mode=mode,
        cmp=cmp0,
        corners_xy=corners_xy,
        edge_maps=edge_maps,
        bubble=bubble,
        alpha=alpha,
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
        env_alpha=env_alpha,
        env_order=env_order,
        ridge=ridge,
        eps_uvw=eps_uvw,
        enforce_lr_symmetry=True,
        edge_guard_m_end=int(m_end),
        edge_guard_power=float(pwr),
        edge_beta=float(beta),
    )
    save_kwargs.update(dict(
        obj=obj,
        obj_all=obj_all,

        # Back-compat (masked metrics) retained for inspection
        mae=score_mask["mae"],
        p90=score_mask["p90"],
        p99=score_mask["p99"],
        worst=score_mask["worst"],

        # Primary: global authalicity (ALL triangles)
        all_mae=score_all["mae"],
        all_p90=score_all["p90"],
        all_p99=score_all["p99"],
        all_worst=score_all["worst"],

        # Secondary: seam band (edge-touching, non-corner)
        seam_mae=score_seam["mae"],
        seam_p90=score_seam["p90"],
        seam_p99=score_seam["p99"],
        seam_worst=score_seam["worst"],
    ))
    improved = False
    save_npz_atomic(latest_path, **save_kwargs)
    if prev_obj is None or (np.isfinite(prev_obj) and obj < prev_obj) or (not np.isfinite(prev_obj)):
        improved = True
        save_npz_atomic(best_path, **save_kwargs)
        print(f"updated BEST model: {best_path} obj={obj:.6f}")
    else:
        print(f"BEST model unchanged: {best_path} prev_obj={prev_obj:.6f} cand_obj={obj:.6f}")

    # Save this model.
    Path(eb_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez(eb_npz, **save_kwargs)
    if verbose:
        print(f"saved eb_npz: {eb_npz}")

    # ---- Print worst triangles (teacher vs new) ----
    locs_eb = classify_locs_b_oct(x_edge_bubble, mode)
    if verbose:
        loc_counts("locs x_edge_bubble", locs_eb)

    print_worst_triangle(rg, x_prime, grid_idx, tri_mask, cmp0, locs_xp, "teacher x_prime", detail=tri_detail)
    tri_bad, vids_bad = print_worst_triangle(rg, x_edge_bubble, grid_idx, tri_mask, cmp0, locs_eb, "edge+bubble",
                                             detail=tri_detail)
    # Extra reporting: separate seam-band behaviour from corner-adjacent triangles.
    print_subset_report("teacher x_prime", rel_xp, grid=grid_idx, locs=locs_xp, tri_mask=tri_mask)
    print_subset_report("edge+bubble", rel_fine, grid=grid_idx, locs=locs_eb, tri_mask=tri_mask)
    print('---------------- iteration done ----------------\n')
    return improved


if __name__ == "__main__":
    m_end_list = [16, 32, 48]
    pwr_list = [1.5, 2.0, 3.0]
    beta_list = [0.0, 0.05, 0.10]
    env_order_list = np.array([1.10, 1.09, 1.11], dtype=float)

    for env_order in env_order_list:
        for m_end in m_end_list:
            for pwr in pwr_list:
                for beta in beta_list:
                    for i in range(10):
                        print(f'm_end:{m_end}; pwr:{pwr}; beta:{beta}; env_order:{env_order}; iteration:{i}')
                        if run(m_end, pwr, beta, env_order):  # True only when best improves
                            continue
                        break
