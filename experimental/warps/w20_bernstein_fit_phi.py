from math import comb
from functools import lru_cache
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from hhg9 import Registrar, Points
from hhg9.h9.polygon import H9P
from w99_plot import plot_grid, rgba_from, plot_mesh

# Finite-difference step for Laplacian in (u,v)
LAPLACIAN_H = 1e-4


def make_vertex_taper_weights(
    uv,
    r0=0.01,
    r1=0.05,
    floor=0.1,
    power=2.0,
):
    """
    Build per-sample weights in [floor, 1] that taper down near the
    3 triangle vertices in (u,v)-space.

    r0 : radius where taper starts (full down-weight near vertices)
    r1 : radius where we reach full weight 1.0
    floor : minimum weight near a vertex
    power : smoothness of ramp (2 = quadratic, >2 sharper)
    """
    uv = np.asarray(uv, dtype=float)
    verts = np.array([[0.0, 0.0],
                      [1.0, 0.0],
                      [0.0, 1.0]], dtype=float)

    # distance to each vertex, then min over vertices
    d = np.linalg.norm(uv[:, None, :] - verts[None, :, :], axis=-1)
    d_min = d.min(axis=1)

    # map [r0, r1] -> [0,1] and clip
    t = (d_min - r0) / (r1 - r0)
    t = np.clip(t, 0.0, 1.0)
    t = t**power

    # near vertex: t≈0 → weight≈floor
    # interior:   t≈1 → weight≈1
    w = floor + (1.0 - floor) * t
    return w

# --- Equilateral face geometry / metric for Laplacian ---
# We derive the XY triangle for a given octant mode from the H9P simplex vertices
# and expose the inverse metric entries g11,g12,g22 as module-level globals
# so that laplacian_uv can use them.

# g11 = 2.0 / 3.0  # sensible defaults for an equilateral reference face
# g12 = -1.0 / 3.0
# g22 = 2.0 / 3.0

@lru_cache(maxsize=16)
def metric_from_mode(mode: int):
    """
    Initialise the metric g^{-1} for the given octant mode.

    mode should typically come from b_oct.oid_mo[octant_id].
    We use H9P.sv[mode] to get the three XY vertices of the equilateral face,
    then build the affine map (u,v) -> XY and its induced metric.
    """
    # global g11, g12, g22
    # H9P.sv[mode] is expected to be shape (3,2): three XY vertices for this mode
    verts = np.asarray(H9P.sv[mode], dtype=float)
    if verts.shape != (3, 2):
        raise ValueError(f"Expected H9P.sv[{mode!r}] to have shape (3,2), got {verts.shape}")

    a_xy, b_xy, c_xy = verts[0], verts[1], verts[2]

    # Affine map (u,v) -> XY: P = A + (B-A) u + (C-A) v
    j_uv_to_xy = np.column_stack((b_xy - a_xy, c_xy - a_xy))  # 2x2
    g = j_uv_to_xy.T @ j_uv_to_xy                             # 2x2 metric
    g_inv = np.linalg.inv(g)
    return g_inv[0, 0], g_inv[0, 1], g_inv[1, 1]


# Helper: cached loader for simplex grid NPZs
@lru_cache(maxsize=None)
def load_simplex_grid(layer: int, mode: int):
    """
    Cached loader for the simplex grid NPZ for a given (layer, mode).

    This avoids re-reading the same file from disk when `run` is called
    repeatedly for the same (layer, mode) pair.
    """
    fname = Path(f"grid_l{layer}_m{mode}_simplex.npz")
    repo = np.load(fname, allow_pickle=True)

    # Extract the arrays we actually need; closing the NPZ file afterwards is fine.
    # xy_vert = repo['xy_vert']
    # tri_uv = repo['uv_vert']
    uv_vert = repo['uv_vert']
    # cmp = repo['components']
    v_ell = repo['v_ell']
    return uv_vert, v_ell


def bernstein_terms_deg(n):
    # list of (i,j,k) with i+j+k=n
    terms = []
    for i in range(n+1):
        for j in range(n+1-i):
            k = n - i - j
            terms.append((i, j, k))
    return terms  # length (n+1)(n+2)/2


def bernstein_eval(a, b, c, n, terms=None):
    if terms is None:
        terms = bernstein_terms_deg(n)
    vals = np.empty(len(terms), dtype=np.float64)
    for t, (i, j, k) in enumerate(terms):
        coef = comb(n, i) * comb(n - i, j) * (a**i)*(b**j)*(c**k)
        vals[t] = coef
    return vals  # shape (K,)


def bernstein_eval_vec(a, b, c, n, terms, coeffs):
    """
    Evaluate a Bernstein expansion at many barycentric points.

    Parameters
    ----------
    a, b, c : array_like, shape (N,)
        Barycentric coords (a+b+c=1).
    n : int
        Total Bernstein degree.
    terms : sequence of (i,j,k)
        Multi-indices of the basis terms.
    coeffs : array_like, shape (K,)
        Coefficients for each term.

    Returns
    -------
    y : ndarray, shape (N,)
        Fitted values at each point.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    coeffs = np.asarray(coeffs, dtype=float)

    if not (a.shape == b.shape == c.shape):
        raise ValueError(f"a,b,c must have same shape, got {a.shape}, {b.shape}, {c.shape}")

    n_pts = a.size
    k_terms = len(terms)
    if coeffs.shape[0] != k_terms:
        raise ValueError(f"coeffs length {coeffs.shape[0]} != number of terms {k_terms}")

    # Build basis matrix B (N, K)
    b_matrix = np.empty((n_pts, k_terms), dtype=float)
    for idx, (i, j, k) in enumerate(terms):
        # sanity: i+j+k should equal n
        if i + j + k != n:
            raise ValueError(f"term {(i,j,k)} inconsistent with degree n={n}")
        binom = comb(n, i) * comb(n - i, j)
        b_matrix[:, idx] = binom * (a**i) * (b**j) * (c**k)

    # Apply coefficients → shape (N,)
    return b_matrix @ coeffs


def bernstein_grads_uv(u, v, n, terms=None):
    """Gradients of Bernstein basis wrt (u,v) on the triangle.
    Supports scalar (u,v) → returns (K,) arrays; or vectorized u,v of shape (M,) → returns (M,K).
    Convention: a=1-u-v, b=u, c=v, and \nabla_u = d/db - d/da, \nabla_v = d/dc - d/da.
    """
    if terms is None:
        terms = bernstein_terms_deg(n)
    # Detect scalar vs vector
    scalar_input = np.isscalar(u) and np.isscalar(v)
    u = np.atleast_1d(np.asarray(u, dtype=float))
    v = np.atleast_1d(np.asarray(v, dtype=float))
    assert u.shape == v.shape, "u and v must have the same shape"

    a = 1.0 - u - v
    b = u
    c = v
    m = u.size
    K = len(terms)
    du = np.empty((m, K), dtype=np.float64)
    dv = np.empty((m, K), dtype=np.float64)

    for t, (i, j, k) in enumerate(terms):
        coef = comb(n, i) * comb(n - i, j)
        # partials of monomial a^i b^j c^k
        dB_da = 0.0
        dB_db = 0.0
        dB_dc = 0.0
        if i > 0:
            dB_da = coef * i * (a ** (i - 1)) * (b ** j) * (c ** k)
        if j > 0:
            dB_db = coef * j * (a ** i) * (b ** (j - 1)) * (c ** k)
        if k > 0:
            dB_dc = coef * k * (a ** i) * (b ** j) * (c ** (k - 1))
        du[:, t] = dB_db - dB_da
        dv[:, t] = dB_dc - dB_da

    if scalar_input:
        return du[0], dv[0]
    return du, dv


def _proj_uv_scalar(u, v, eps=1e-12):
    """Project a single (u,v) to the closed simplex {u>=0,v>=0,u+v<=1}."""
    uu = float(np.clip(u, eps, 1.0 - eps))
    vv = float(np.clip(v, eps, 1.0 - eps))
    s = uu + vv
    if s > (1.0 - eps):
        scale = (1.0 - eps) / s
        uu *= scale
        vv *= scale
    return uu, vv

# Helper: distance to nearest edge (0 at boundary, >0 in interior)
def _edge_dist(u, v):
    return max(0.0, min(u, v, 1.0 - u - v))


def laplacian_uv(u, v, n, metric, terms=None):
    # Δ = ∂²/∂u² + ∂²/∂v² + 2 ∂²/∂u∂v  (a=1-u-v)
    # Central differences; avoid projection unless near edges for unbiased interior stencil.
    g11, g12, g22 = metric
    if terms is None:
        terms = bernstein_terms_deg(n)
    # Adaptive step: shrink h as we approach an edge to reduce projection bias
    d_edge = max(0.0, min(u, v, 1.0 - u - v))
    h = min(LAPLACIAN_H, 0.5 * d_edge)
    # Use a small numerical floor to avoid zero step at the boundary
    h_floor = 1e-5
    if h <= 0:
        h = h_floor
    else:
        h = max(h, h_floor)

    def eval_B(uu, vv):
        a = 1.0 - uu - vv; b = uu; c = vv
        return bernstein_eval(a, b, c, n, terms)

    # If safely inside the simplex, use raw offsets (faster, unbiased)
    if (u > h) and (v > h) and (u + v < 1.0 - h):
        f0   = eval_B(u,    v)
        fu_p = eval_B(u+h,  v)
        fu_m = eval_B(u-h,  v)
        fv_p = eval_B(u,    v+h)
        fv_m = eval_B(u,    v-h)
        fpp  = eval_B(u+h,  v+h)
        fpm  = eval_B(u+h,  v-h)
        fmp  = eval_B(u-h,  v+h)
        fmm  = eval_B(u-h,  v-h)
    else:
        # Near edges, project offsets back into the simplex to stay valid
        u_p, v_p   = _proj_uv_scalar(u + h, v)
        u_m, v_m   = _proj_uv_scalar(u - h, v)
        u_0p, v_0p = _proj_uv_scalar(u, v + h)
        u_0m, v_0m = _proj_uv_scalar(u, v - h)
        u_pp, v_pp = _proj_uv_scalar(u + h, v + h)
        u_pm, v_pm = _proj_uv_scalar(u + h, v - h)
        u_mp, v_mp = _proj_uv_scalar(u - h, v + h)
        u_mm, v_mm = _proj_uv_scalar(u - h, v - h)

        f0   = eval_B(u,    v)
        fu_p = eval_B(u_p,  v_p)
        fu_m = eval_B(u_m,  v_m)
        fv_p = eval_B(u_0p, v_0p)
        fv_m = eval_B(u_0m, v_0m)
        fpp  = eval_B(u_pp, v_pp)
        fpm  = eval_B(u_pm, v_pm)
        fmp  = eval_B(u_mp, v_mp)
        fmm  = eval_B(u_mm, v_mm)

    d2uu = (fu_p - 2*f0 + fu_m) / (h*h)
    d2vv = (fv_p - 2*f0 + fv_m) / (h*h)
    d2uv = (fpp - fpm - fmp + fmm) / (4*h*h)
    # Metric-correct Laplacian in XY: Δ = g11 * f_uu + 2*g12 * f_uv + g22 * f_vv
    lap = g11 * d2uu + 2.0 * g12 * d2uv + g22 * d2vv
    return lap


# --- Boundary condition helpers ---
def edge_samples(p0, p1, n_s=48):
    """Sample n_s points along the edge between p0 and p1 in (u,v) space."""
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    t = np.linspace(0.0, 1.0, n_s)
    return (1.0 - t)[:, None] * p0 + t[:, None] * p1


def build_dirichlet_rows(n, terms_all, terms, n_s_factor):
    """Build Dirichlet boundary rows φ=0 on all three triangle edges.

    Returns (B_dir, b_dir) where B_dir has shape (N_dir, K) and b_dir is zeros.
    If n_s_factor <= 0, returns (None, None).
    """
    if n_s_factor <= 0:
        return None, None

    idx_map = {t: i for i, t in enumerate(terms_all)}
    n_s = n_s_factor * n

    edges = [
        edge_samples(np.array([0.0, 0.0]), np.array([1.0, 0.0]), n_s),  # AB (v=0)
        edge_samples(np.array([0.0, 0.0]), np.array([0.0, 1.0]), n_s),  # AC (u=0)
        edge_samples(np.array([1.0, 0.0]), np.array([0.0, 1.0]), n_s),  # BC (u+v=1)
    ]

    rows = []
    for edge_pts in edges:
        for (u, v) in edge_pts:
            a = 1.0 - u - v
            b = u
            c = v
            b_all = bernstein_eval(a, b, c, n, terms_all)
            rows.append([b_all[idx_map[t]] for t in terms])

    if not rows:
        return None, None

    b_dir_mat = np.asarray(rows, dtype=float)
    b_dir_vec = np.zeros(b_dir_mat.shape[0], dtype=float)
    return b_dir_mat, b_dir_vec


def build_neumann_rows(n, terms_all, terms, n_s_factor):
    """Build optional Neumann rows enforcing ∂n φ ≈ 0 on edges.
    Returns (b_neu_mat, b_neu_vec) or (None, None) if n_s_factor<=0.
    """
    if n_s_factor <= 0:
        return None, None

    idx_map = {t: i for i, t in enumerate(terms_all)}
    n_s = n_s_factor * n

    edges = [
        edge_samples(np.array([0.0, 0.0]), np.array([1.0, 0.0]), n_s),  # AB (v=0)
        edge_samples(np.array([0.0, 0.0]), np.array([0.0, 1.0]), n_s),  # AC (u=0)
        edge_samples(np.array([1.0, 0.0]), np.array([0.0, 1.0]), n_s),  # BC (u+v=1)
    ]
    normals = [
        np.array([0.0, 1.0]),
        np.array([1.0, 0.0]),
        np.array([1.0/np.sqrt(2.0), 1.0/np.sqrt(2.0)]),
    ]

    rows = []
    for edge_pts, nvec in zip(edges, normals):
        for (u, v) in edge_pts:
            du_all, dv_all = bernstein_grads_uv(u, v, n, terms_all)
            rows.append([
                du_all[idx_map[t]] * nvec[0] + dv_all[idx_map[t]] * nvec[1]
                for t in terms
            ])

    if not rows:
        return None, None

    b_neu_mat = np.asarray(rows, dtype=float)
    b_neu_vec = np.zeros(b_neu_mat.shape[0], dtype=float)
    return b_neu_mat, b_neu_vec


def fit_phi(uv, r, metric, n=5, bc_weight=1e-2,
            neumann_weight=0.0, n_s_factor=12, lam=1e-5,
            row_weight=None
            ):
    terms_all = bernstein_terms_deg(n)
    idx_map = {t: i for i, t in enumerate(terms_all)}
    terms = terms_all
    k = len(terms)

    # build m_mat: each row is Laplacian of each basis at a sample
    m = uv.shape[0]
    m_mat = np.empty((m, k), dtype=np.float64)
    for i, (u, v) in enumerate(uv):
        lap_all = laplacian_uv(u, v, n, metric, terms_all)
        row = []
        for t in terms:
            idx = idx_map[t]
            row.append(lap_all[idx])
        m_mat[i, :] = row

    rhs = r.astype(np.float64, copy=False)
    if row_weight is not None:
        w = np.asarray(row_weight, dtype=np.float64)
        if w.shape != (m,):
            raise ValueError(f"row_weight shape {w.shape} != (m,)={m}")
        # avoid negative weights, tiny eps for stability
        w = np.clip(w, 0.0, None)
        w = np.sqrt(w)
        m_mat *= w[:, None]
        rhs *= w

    # ---- Column normalization for conditioning ----
    col_norm = np.sqrt(np.mean(m_mat**2, axis=0))
    col_norm[col_norm == 0] = 1.0
    m_mat /= col_norm
    print(f"Column norms (mean={np.mean(col_norm):.3e}, min={np.min(col_norm):.3e}, max={np.max(col_norm):.3e})")

    # ---- Boundary constraints ----
    b_dir_mat, b_dir_vec = build_dirichlet_rows(n, terms_all, terms, n_s_factor)
    b_neu_mat, b_neu_vec = (None, None)
    if neumann_weight > 0.0:
        b_neu_mat, b_neu_vec = build_neumann_rows(n, terms_all, terms, n_s_factor)

    m_aug = m_mat
    r_aug = r
    if bc_weight > 0.0 and b_dir_mat is not None:
        m_aug = np.vstack([m_aug, np.sqrt(bc_weight) * b_dir_mat])
        r_aug = np.concatenate([r_aug, np.sqrt(bc_weight) * b_dir_vec])
    if neumann_weight > 0.0 and b_neu_mat is not None:
        m_aug = np.vstack([m_aug, np.sqrt(neumann_weight) * b_neu_mat])
        r_aug = np.concatenate([r_aug, np.sqrt(neumann_weight) * b_neu_vec])

    # augmented ridge regression via row augmentation (numerically safer)
    col_scale = m_aug.std(axis=0, ddof=0)
    col_scale[col_scale == 0] = 1.0
    m_s = m_aug / col_scale

    # Ridge via row augmentation (avoid forming M^T M)
    kI = np.sqrt(lam) * np.eye(k)
    m_ridge = np.vstack([m_s, kI])
    r_ridge = np.concatenate([r_aug, np.zeros(k)])
    c_scaled, *_ = np.linalg.lstsq(m_ridge, r_ridge, rcond=None)
    c = c_scaled / col_scale

    # Undo earlier column normalization
    c /= col_norm

    # diagnostics
    print("cond(M_aug):", np.linalg.cond(m_aug))
    print("cond(M_s):",   np.linalg.cond(m_s))
    return terms, c, col_norm


def run(rg, layer, octant_id, conf, *, tweak='base', plot=False, save=True, diagnostics=True):
    b_oct = rg.domain('b_oct')
    # s_oct = rg.domain('s_oct')
    mode = int(b_oct.oid_mo[octant_id])
    metric = metric_from_mode(mode)
    uv_vert, v_ell = load_simplex_grid(layer, mode)
    ell = v_ell - v_ell.mean()
    if diagnostics:
        print(f"[w20] layer={layer} mode={mode} ell stats: min={ell.min():.4f}, max={ell.max():.4f}, mean={ell.mean():.4f}, std={ell.std():.4f}")
    n_fit = conf['n_fit']        # number of Bernstein terms to fit
    neumann = conf['neumann']    # edge sampling for BC/Neumann rows
    ns_fac = conf['ns_fac']      # Neumann damping factor
    bc_w = conf['bc_w']          # bc_weight
    lam = conf['lam']            # ridge regularisation
    rhs = -ell                   # Laplacian target: Δφ ≈ -ℓ (we fit φ such that Δφ tracks -ℓ)
    # Per-row weights: gently down-weight samples very close to the three vertices
    row_w = make_vertex_taper_weights(
        uv_vert,
        # vtw010820
        r0=0.01,   # start relaxing a bit further out
        r1=0.08,   # don’t give full weight until ~8% of edge length
        floor=0.2, # vertices still matter, but only at 20% strength
        power=2.0, # quadratic is fine; can go 3.0 if you want a sharper shoulder
        # original 'basis'.
        # r0=0.003,    # start of taper very close to vertices
        # r1=0.03,     # full weight by here
        # floor=0.05,  # down-weight to 5% at the vertices
        # power=2.0,   # smooth quadratic ramp
    )

    # --- Diagnostics: row weight stats and BC rows ---
    if diagnostics:
        if row_w is not None:
            print(f"row_weight stats: min={row_w.min():.4f}, max={row_w.max():.4f}, mean={row_w.mean():.4f}")
        n_s = ns_fac * n_fit
        n_dir = 3 * n_s if bc_w > 0.0 else 0
        n_neu = 3 * n_s if neumann > 0.0 else 0
        print(f"BC sampling: n_s_per_edge={n_s}, Dirichlet rows={n_dir}, Neumann rows={n_neu}")
    terms, c, col_norm = fit_phi(
        uv_vert,
        rhs,
        metric,
        n=n_fit,
        n_s_factor=ns_fac,
        bc_weight=bc_w,
        lam=lam,
        neumann_weight=neumann,
        row_weight=row_w,
    )
    print(f'L:{layer}; n_fit:{n_fit}; ns_fac:{ns_fac}; bc_w:{bc_w} neumann:{neumann}; lam:{lam}; fitted {len(c)} terms')
    terms_all = bernstein_terms_deg(n_fit)

    if save:
        np.savez(
            f"phi_fit_l{layer}_m{mode}_{tweak}_n{n_fit}.npz",
            terms=np.array(terms, dtype=object),
            c=c,
            col_norm=col_norm,
            n_fit=n_fit,
            depth=layer,
            bc_weight=bc_w,
            neumann_weight=neumann,
            lam=lam,
            n_s_factor=ns_fac,
            metric=metric,
            laplacian_h=LAPLACIAN_H,
            row_weight=row_w,
            uv_vert=uv_vert,
        )
    if diagnostics:
        g11, g12, g22 = metric
        print(f"laplacian_h: {LAPLACIAN_H}")
        print(f"metric Ginv: [[{g11:.6f},{g12:.6f}],[{g12:.6f},{g22:.6f}]]")
        idx = {t: i for i, t in enumerate(terms_all)}
        sel = [idx[t] for t in terms]  # use the exact fitted interior term order
        lap = np.vstack([laplacian_uv(u, v, n_fit, metric, terms_all) for u, v in uv_vert])
        m_mat = lap[:, sel]
        print("cond(M):", np.linalg.cond(m_mat))

        # --- Sample all three edges to inspect normal flux ∂n φ ---
        n_edge_samples = 50
        edges = [
            edge_samples(np.array([0.0, 0.0]), np.array([1.0, 0.0]), n_edge_samples),  # AB (v=0)
            edge_samples(np.array([0.0, 0.0]), np.array([0.0, 1.0]), n_edge_samples),  # AC (u=0)
            edge_samples(np.array([1.0, 0.0]), np.array([0.0, 1.0]), n_edge_samples),  # BC (u+v=1)
        ]
        normals = [
            np.array([0.0, 1.0]),                                  # outward normal from v=0 edge
            np.array([1.0, 0.0]),                                  # outward normal from u=0 edge
            np.array([1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)]),    # outward normal from u+v=1 edge
        ]
        tangents = [
            np.array([1.0, 0.0]),                             # tangent along AB (u increases)
            np.array([0.0, 1.0]),                             # tangent along AC (v increases)
            np.array([-1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)]),  # tangent along BC from B→C
        ]
        edge_names = ["AB (v=0)", "AC (u=0)", "BC (u+v=1)"]
        flux_rms_all = []
        tan_rms_all = []

        for edge_pts, nvec, tvec, name in zip(edges, normals, tangents, edge_names):
            u_e = edge_pts[:, 0]
            v_e = edge_pts[:, 1]
            du_all, dv_all = bernstein_grads_uv(u_e, v_e, n_fit, terms_all)
            row = du_all[:, sel] * nvec[0] + dv_all[:, sel] * nvec[1]
            edge_flux = row @ c
            rms = float(np.sqrt(np.mean(edge_flux ** 2)))
            flux_rms_all.append(rms)
            tan_row = du_all[:, sel] * tvec[0] + dv_all[:, sel] * tvec[1]
            edge_tan = tan_row @ c
            tan_rms = float(np.sqrt(np.mean(edge_tan ** 2)))
            tan_rms_all.append(tan_rms)
            print(f"edge {name}: flux_rms={rms:.6e}, tan_rms={tan_rms:.6e}")

        if flux_rms_all:
            print("edge flux rms (max over edges):", max(flux_rms_all))
        # optional, but helpful:
        # if tan_rms_all:
        #     print("edge tan rms (max over edges):", max(tan_rms_all))

        # --- New diagnostic: Dirichlet edge φ RMS ---
        phi_rms_all = []
        for edge_pts, name in zip(edges, edge_names):
            u_e = edge_pts[:, 0]
            v_e = edge_pts[:, 1]
            a = 1.0 - u_e - v_e
            b = u_e
            c_b = v_e  # avoid shadowing the fitted coeffs array 'c'
            phi_edge = bernstein_eval_vec(a, b, c_b, n_fit, terms, c)
            phi_rms = float(np.sqrt(np.mean(phi_edge ** 2)))
            phi_rms_all.append(phi_rms)
            print(f"edge {name}: phi_rms={phi_rms:.6e}")
        if phi_rms_all:
            print("edge phi rms (max over edges):", max(phi_rms_all))

        r_pred = m_mat @ c
        print(f"rhs range: [{rhs.min():.3g},{rhs.max():.3g}]  "
              f"r_pred range: [{r_pred.min():.3g},{r_pred.max():.3g}]")
        # extra diagnostics: how well does the fit reproduce rhs?
        rhs_flat = rhs.ravel()
        r_pred_flat = r_pred.ravel()
        # guard against pathological constant rhs
        if rhs_flat.std() > 0 and r_pred_flat.std() > 0:
            corr = np.corrcoef(rhs_flat, r_pred_flat)[0, 1]
        else:
            corr = np.nan
        rhs_rmse = float(np.sqrt(np.mean(rhs_flat**2)))
        fit_rmse = float(np.sqrt(np.mean(r_pred_flat**2)))
        print(f"corr(rhs, r_pred)={corr:+.6f}  RMSE(rhs vs 0)={rhs_rmse:.6f}  RMSE(r_pred vs 0)={fit_rmse:.6f}")
        res = rhs - r_pred
        print("RMSE:", np.sqrt(np.mean(res ** 2)), " MaxAbs:", np.max(np.abs(res)))
        # Diagnostic: report worst residual and its proximity to edges
        i_max = int(np.argmax(np.abs(res)))
        u_max, v_max = uv_vert[i_max]
        d_edge_max = max(0.0, min(u_max, v_max, 1.0 - u_max - v_max))
        where = "boundary-adjacent" if d_edge_max < 5e-3 else "interior-ish"
        print(f"worst |res| at (u,v)=({u_max:.6f},{v_max:.6f}), "
              f"|res|={abs(res[i_max]):.6g}, d_edge={d_edge_max:.3e} → {where}")


if __name__ == '__main__':
    reg = Registrar()
    layer = 6
    octant = 0
    c_template = [
        'n_fit',    # number of Bernstein basis terms M e R^(NK).
        'ns_fac',   # Node sample factor — how densely integration grid is sampled.
                    # Eg n_fit=16 and ns_fac=20, the algorithm builds a (n_fit * ns_fac)
        'bc_w',     # Dirichlet boundary condition weight High bc_w → values low boundary
        'neumann',  # Neumann boundary condition weight, neumann = 0.0: ignore derivative
        'lam',      # Tikhonov regularisation weight to damp large coefficients
                    # smaller lam → sharper fit, larger lam → smoother, more stable, but biased.
    ]
    configs = {
        'v0100': [16, 20, 1e-2, 1e-5, 3e-6],    #
        'v0305': [16, 30, 0.03, 3.0e-4, 3e-5],  # Neumann best between 0.0001 = 0.1
        'v0408': [16, 50, 0.075, 0, 3e-6],      # <-- v0408 original wins - but delaminates.
    }
    for using in ['v0305']:
    # for using in configs:
    # for v in range(8):
        # using = f'v05{v:02d}'
        config = {n: v for (n, v) in zip(c_template, configs[using])}
        run(reg, layer, octant, config, tweak=using, save=True, plot=False, diagnostics=True)
