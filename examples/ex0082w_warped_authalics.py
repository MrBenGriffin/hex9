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
import numpy as np
from matplotlib import pyplot as plt, colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from hhg9 import Registrar, Points
from hhg9.algorithms.distance import wgs84_area
from hhg9.h9 import H9_RA, H9O, H9K
from hhg9.h9.classifier import location
from hhg9.h9.protocols import BaryLoc
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


def snow_globe(arr: Points, poly_len: int = 6, scores=None, layers='x'):
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
    # front = all_polys
    # max_abs = float(np.max(np.abs(scores)))
    max_abs = 0.15
    norm = colors.TwoSlopeNorm(vcenter=0.0, vmin=-max_abs, vmax=+max_abs)
    cmap_name = 'RdBu_r'
    sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=norm)
    sm.set_array([])
    rgba, _norm = rgba_from(scores, cmap_name, norm=norm)
    # pops = scores[mask]
    ax.set_proj_type('ortho')  # FOV = 0 deg
    lim = 3.75e+6
    ax.set_xlim(-lim, lim)  # fill the area with the map.
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    polys = [p for p in front]

    collection = Poly3DCollection(polys, ec='black', facecolors=rgba[mask], alpha=0.9, linewidth=0.02)
    ax.add_collection(collection)
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()
    lil, big = np.min(scores), np.max(scores)
    # authalic_p98 = np.quantile(np.abs(scores), 0.98)
    # p98_frac = 100 * np.expm1(authalic_p98)
    ax.title.set_text(f'min:{lil}, max:{big} deviation from ideal.')
    plt.tight_layout()
    fig.savefig(f"output/ex0080w_{layers}.png", dpi=400)
    print(f'fig saved at output/ex0080w_{layers}.png')


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


def hexify(reg: Registrar, b_pts: Points, warp_m=None, layers: int = 4, num: int=8):
    """
    Find hexagons for data, and display on a 'globe'.
    """
    pts, pops = hex_poly_layer(b_pts, layers)

    # Now calculate their area as a metric. (ignore pops).
    gm2 = 510_065_621_724_154.6  # total surface area of WGS-84 (m²)
    bins = 12*9**layers          # number of hexes at this layer
    w_area_m2_mean = gm2/bins    # ideal equal-area per hex
    h_pts = pts.copy()
    oc, mo = h_pts.cm()

    # warp_m
    b_wrp = warp(
        h_pts,
        in_mode=mo,
        model=warp_m
    )
    h_pts.coords = b_wrp

    c_pts = reg.project(h_pts, ['b_oct', 'c_oct', 'c_ell'])  # use bary.
    g_pts = reg.project(h_pts, ['b_oct', 'g_gcd'])
    w_area_m2 = wgs84_area(reg, g_pts)  # default value is 6
    w_adj = np.abs(w_area_m2 / w_area_m2_mean) + 1e-12
    score = np.log(w_adj)  # authalic log-density ℓ
    snow_globe(c_pts, 6, score, f'{layers}_{num}')

# ---- Warp Functions ------
def pt_loc(coords):
    """Identify where points lie in their octants."""
    x, y = coords[:, 0], coords[:, 1]
    ẋ = H9K.R3 * x
    return location(ẋ, y, 1, detailed=True)


def triangle_params_from_corners(corners_c: np.ndarray) -> tuple[float, float, float]:
    """Infer (base_y, apex_y, t) from the 3 canonical-mode corner points."""
    corners_c = np.asarray(corners_c, dtype=float).reshape(3, 2)
    apex_i = int(np.argmax(corners_c[:, 1]))
    base_idx = [i for i in range(3) if i != apex_i]
    apex_y = float(corners_c[apex_i, 1])
    base_y = float(np.mean(corners_c[base_idx, 1]))
    x0 = float(corners_c[base_idx[0], 0])
    x1 = float(corners_c[base_idx[1], 0])
    t = 0.5 * abs(x1 - x0)
    return base_y, apex_y, t

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


def _edge_id_and_t_for_points(
    xy: np.ndarray,
    corners: np.ndarray,
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
    xy_c = np.asarray(xy, dtype=float)
    corners_c = np.asarray(corners, dtype=float).reshape(3, 2)

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


def apply_edge_maps_to_points(xy: np.ndarray,
                              corners_xy: np.ndarray,
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
    edge_id, t = _edge_id_and_t_for_points(pts, corners_xy)

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


def bubble_delta_symmetric(xy: np.ndarray, corners_xy: np.ndarray, warp: dict) -> np.ndarray:
    """Return the displacement field (dx,dy) produced by a symmetric bubble warp."""
    xy_c = xy
    corners_c = corners_xy

    base_y = float(warp['base_y'])
    apex_y = float(warp['apex_y'])
    t = float(warp['t'])

    u, v, w, x_n = barycentric_uvwx(xy_c, base_y, apex_y, t)
    uvw = u * v * w
    env_alpha = float(warp.get('env_alpha', 1.0))
    env_order = float(warp.get('env_order', 1.0))
    env = np.power(np.maximum(uvw, 0.0), env_alpha * env_order)

    fy, fx, _ = _sym_bubble_features(u, x_n, u_deg=int(warp['u_deg']), x2_deg=int(warp['x2_deg']))
    dx_c = (env[:, None] * fx) @ np.asarray(warp['cx'])
    dy_c = (env[:, None] * fy) @ np.asarray(warp['cy'])

    # Map delta back to original mode coordinates
    dx = dx_c
    dy = dy_c
    return np.stack([dx, dy], axis=1)


def load_warp_npz(path: str) -> dict:
    """Load warp model npz"""
    repo = np.load(path, allow_pickle=True)
    out = {k: repo[k] for k in repo.files}

    # unwrap 0-d object arrays (pickled dict/string)
    for k in ("edge_maps", "bubble", "alpha_strategy", "alpha_objective", "meta"):
        if k in out and isinstance(out[k], np.ndarray) and out[k].shape == ():
            out[k] = out[k].item()

    # convenience casts
    out["octant_id"] = int(out["octant_id"])
    out["layer"] = int(out["layer"])
    out["mode"] = int(out["mode"])
    out["alpha"] = float(out["alpha"])
    out["alpha_max"] = float(out["alpha_max"])

    out.setdefault("meta", {})
    return out


def warp(
    b_pts: Points,
    *,
    in_mode: int,
    locs: np.ndarray = None,
    model: dict,
) -> np.ndarray:
    """Apply a saved EB model to a set of b_oct points a_xy.
    Returns x_eb (same shape as a_xy).
    """
    mo0 = (in_mode == 0)

    # Work on a local copy so we don't mutate the caller's Points.
    a_xy = np.asarray(b_pts.coords, dtype=float).copy()
    a_xy[mo0, 1] *= -1.0  # canonicalize to mode-1 by flipping y for mode-0 points
    lim = H9K.limits
    corners = np.array([[lim.TL, lim.ΛF], [lim.TR, lim.ΛF], [0.0, lim.ΛC]], dtype=np.float64)
    corners_xy = np.asarray(corners, dtype=float).reshape(3, 2)
    if locs is None:
        locs = pt_loc(a_xy)  # locs must be computed in the same canonical space we warp in.
    locs_i = normalize_locs(locs)
    is_vtx = (locs_i == int(BaryLoc.VTX.value))
    is_edg = (locs_i == int(BaryLoc.EDG.value))

    # 1) edges only
    x_stage1 = apply_edge_maps_to_points(
        a_xy,
        corners_xy=corners_xy,
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
    x_eb[mo0, 1] *= -1.0
    return x_eb


if __name__ == '__main__':
    depth = 3  # 0,...5 √
    models = ['data/eb1_model_L5_best.npz', 'data/eb2_model_L5_best.npz']
    for m_no, model_f in enumerate(models):
        model = load_warp_npz(model_f)
        rg = Registrar()  # Manage Domains & Projections
        data = get_data(rg, depth)  # should be 8*9**depth  (eg, depth=0: 72 points, 9 points on each face, and six points in each hexagon)
        hexify(rg, data, model, layers=depth, num=m_no)


