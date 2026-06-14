from math import comb
from pathlib import Path
import numpy as np

# Finite-difference step for Laplacian in (u,v)
LAPLACIAN_H = 1e-4

# --- Equilateral face geometry (centroid at origin), metric for Laplacian ---
PARITY = "female"  # apex down for your net_mode-0 octant
S = np.sqrt(2.0)
H = np.sqrt(6.0) / 2.0
if PARITY == "male":
    A_xy = np.array([0.0,  2.0*H/3.0])
    B_xy = np.array([-S/2.0, -H/3.0])
    C_xy = np.array([ S/2.0, -H/3.0])
else:
    A_xy = np.array([0.0, -2.0*H/3.0])
    B_xy = np.array([ S/2.0,  H/3.0])
    C_xy = np.array([-S/2.0,  H/3.0])
# Affine map (u,v) -> XY: P = A + (B-A) u + (C-A) v
J_uv_to_xy = np.column_stack((B_xy - A_xy, C_xy - A_xy))  # 2x2
G = J_uv_to_xy.T @ J_uv_to_xy                             # 2x2 metric
Ginv = np.linalg.inv(G)
G11, G12, G22 = Ginv[0,0], Ginv[0,1], Ginv[1,1]
# print(f"metric G=\n{G}\nG^-1=\n{Ginv}")


def bernstein_terms_deg(n):
    # list of (i,j,k) with i+j+k=n
    terms = []
    for i in range(n+1):
        for j in range(n+1-i):
            k = n - i - j
            terms.append((i,j,k))
    return terms  # length (n+1)(n+2)/2

def bernstein_eval(a, b, c, n, terms=None):
    if terms is None:
        terms = bernstein_terms_deg(n)
    vals = np.empty(len(terms), dtype=np.float64)
    for t, (i,j,k) in enumerate(terms):
        vals[t] = comb(n, i) * comb(n - i, j) * (a**i)*(b**j)*(c**k)
    return vals  # shape (K,)

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

def laplacian_uv(u, v, n, terms=None):
    # Δ = ∂²/∂u² + ∂²/∂v² + 2 ∂²/∂u∂v  (a=1-u-v)
    # Central differences; avoid projection unless near edges for unbiased interior stencil.
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

    # If we're extremely close to a boundary, keep a minimal step to avoid zero
    if h <= 0:
        h = LAPLACIAN_H

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
    # Metric-correct Laplacian in XY: Δ = Ginv11 * f_uu + 2*Ginv12 * f_uv + Ginv22 * f_vv
    lap  = G11 * d2uu + 2.0 * G12 * d2uv + G22 * d2vv
    return lap

def interior_terms(n):
    """Keep only basis terms that vanish on every edge (i>0, j>0, k>0)."""
    terms = bernstein_terms_deg(n)
    keep = []
    for (i, j, k) in terms:
        if i > 0 and j > 0 and k > 0:
            keep.append((i, j, k))
    return keep

# def interior_terms(n):
#     terms = bernstein_terms_deg(n)
#     keep = []
#     for (i,j,k) in terms:
#         # drop pure-vertex terms (one index == n)
#         if i == n or j == n or k == n:
#             continue
#         keep.append((i,j,k))
#     return keep  # ~18 terms for n=5


def fit_phi(uv, r, n=5, bc_weight=1e-2, neumann_weight=0.0, n_s_factor=12, lam= 1e-5):
    terms_all = bernstein_terms_deg(n)
    idx_map = {t:i for i,t in enumerate(terms_all)}
    terms = interior_terms(n)  # drop vertex-only terms
    k = len(terms)

    # build M: each row is Laplacian of each basis at a sample
    m = uv.shape[0]
    M = np.empty((m, k), dtype=np.float64)
    for i, (u, v) in enumerate(uv):
        lap_all = laplacian_uv(u, v, n, terms_all)
        # pick columns corresponding to 'terms'
        # map terms to their indices in terms_all
        # (build index map once outside loop for speed if you like)
        row = []
        for t in terms:
            idx = idx_map[t]
            row.append(lap_all[idx])
        M[i, :] = row

    # ---- Column normalization for conditioning ----
    col_norm = np.sqrt(np.mean(M**2, axis=0))
    col_norm[col_norm == 0] = 1.0
    M /= col_norm
    print(f"Column norms (mean={np.mean(col_norm):.3e}, min={np.min(col_norm):.3e}, max={np.max(col_norm):.3e})")

    # ---- Boundary constraints ----
    # Build Dirichlet rows: φ(u,v) = 0 on edges (keeps boundary fixed)
    B_dir = []
    b_dir = []

    # Build Neumann rows: ∂n φ ≈ 0 on edges (optional extra damping)
    B_neu = []
    b_neu = []

    def edge_samples(p0, p1, n_s=48):
        t = np.linspace(0, 1, n_s)
        return (1 - t)[:, None] * p0 + t[:, None] * p1

    n_s = n_s_factor * n
    edges = [
        edge_samples(np.array([0, 0]), np.array([1, 0]), n_s),  # AB (v=0)
        edge_samples(np.array([0, 0]), np.array([0, 1]), n_s),  # AC (u=0)
        edge_samples(np.array([1, 0]), np.array([0, 1]), n_s),  # BC (u+v=1)
    ]
    normals = [
        np.array([0, 1.0]),
        np.array([1.0, 0]),
        np.array([1/np.sqrt(2), 1/np.sqrt(2)]),
    ]

    for edge_pts, nvec in zip(edges, normals):
        for (u, v) in edge_pts:
            # Dirichlet row: basis values
            a = 1.0 - u - v; b = u; c = v
            B_all = bernstein_eval(a, b, c, n, terms_all)
            B_row = [B_all[idx_map[t]] for t in terms]
            B_dir.append(B_row); b_dir.append(0.0)
            # Optional Neumann row: grad · n
            if neumann_weight > 0.0:
                du_all, dv_all = bernstein_grads_uv(u, v, n, terms_all)
                G_row = [du_all[idx_map[t]] * nvec[0] + dv_all[idx_map[t]] * nvec[1] for t in terms]
                B_neu.append(G_row); b_neu.append(0.0)

    M_aug = M
    r_aug = r
    if bc_weight > 0 and B_dir:
        B_dir = np.asarray(B_dir, float)
        b_dir = np.asarray(b_dir, float)
        M_aug = np.vstack([M_aug, np.sqrt(bc_weight) * B_dir])
        r_aug = np.concatenate([r_aug, np.sqrt(bc_weight) * b_dir])
    if neumann_weight > 0 and B_neu:
        B_neu = np.asarray(B_neu, float)
        b_neu = np.asarray(b_neu, float)
        M_aug = np.vstack([M_aug, np.sqrt(neumann_weight) * B_neu])
        r_aug = np.concatenate([r_aug, np.sqrt(neumann_weight) * b_neu])

    # augmented ridge regression via row augmentation (numerically safer)
    col_scale = M_aug.std(axis=0, ddof=0)
    col_scale[col_scale == 0] = 1.0
    M_s = M_aug / col_scale

    # Ridge via row augmentation (avoid forming M^T M)
    kI = np.sqrt(lam) * np.eye(k)
    M_ridge = np.vstack([M_s, kI])
    r_ridge = np.concatenate([r_aug, np.zeros(k)])
    c_scaled, *_ = np.linalg.lstsq(M_ridge, r_ridge, rcond=None)
    c = c_scaled / col_scale

    # Undo earlier column normalization
    c /= col_norm

    # diagnostics
    print("cond(M_aug):", np.linalg.cond(M_aug))
    print("cond(M_s):",   np.linalg.cond(M_s))

    return terms, c, col_norm


def d3_symmetrise(uv, r, decimals=12):
    """Return (uv, r_sym) where r_sym is r averaged over the D3 orbit of each point.
    Assumes uv contains the full symmetric centroid set; uses rounded-key lookup.
    """
    uv = np.asarray(uv, dtype=float)
    r = np.asarray(r, dtype=float)
    # Build lookup from rounded (u,v) -> index
    keys = np.round(uv, decimals)
    key_to_idx = { (k[0], k[1]) : i for i, k in enumerate(keys) }

    def perms(p):
        u, v = p
        a = 1.0 - u - v
        return (
            (u, v),      # identity
            (v, u),      # swap
            (a, v),      # rotate
            (v, a),      # rotate
            (u, a),      # rotate
            (a, u),      # swap
        )

    r_sym = np.empty_like(r)
    for i, p in enumerate(uv):
        vals = []
        for q in perms(p):
            k = (round(q[0], decimals), round(q[1], decimals))
            j = key_to_idx.get(k)
            if j is not None:
                vals.append(r[j])
        r_sym[i] = np.mean(vals) if vals else r[i]
    return uv, r_sym


def predict_r(uv_pts, n, terms_all, terms, c):
    """Evaluate fitted r = Δφ at arbitrary (u,v) points."""
    idx = {t: i for i, t in enumerate(terms_all)}
    sel = [idx[t] for t in terms]
    rows = []
    for (u, v) in uv_pts:
        lap_all = laplacian_uv(u, v, n, terms_all)
        rows.append(lap_all[sel])
    Mq = np.asarray(rows)
    return Mq @ c

def grad_phi(uv_pts, n, terms_all, terms, c):
    """Evaluate ∇φ at arbitrary (u,v) points (for visualising displacement).
    Returns arrays (gx, gy) of shape (m,).
    """
    idx = {t: i for i, t in enumerate(terms_all)}
    sel = [idx[t] for t in terms]
    gx_list, gy_list = [], []
    for (u, v) in uv_pts:
        du_all, dv_all = bernstein_grads_uv(u, v, n, terms_all)
        du = du_all[sel]
        dv = dv_all[sel]
        gx_list.append(du @ c)
        gy_list.append(dv @ c)
    return np.asarray(gx_list), np.asarray(gy_list)

# --- Barycentric/UV helpers ---
Minv = np.linalg.inv(J_uv_to_xy)  # for XY->UV via (u,v) = Minv @ (P - A)

def xy_to_bary(xy):
    XY = np.asarray(xy, float)
    rel = XY - A_xy
    uv = rel @ Minv.T
    u = uv[:, 0]
    v = uv[:, 1]
    a = 1.0 - u - v
    b = u
    c = v
    return np.column_stack([a, b, c])

def bary_to_uv(bary):
    a, b, c = bary[:, 0], bary[:, 1], bary[:, 2]
    return np.column_stack([b, c])

def is_uv_simplex(arr, tol=1e-6):
    arr = np.asarray(arr, float)
    u = arr[:, 0]; v = arr[:, 1]
    ok = (u >= -tol) & (v >= -tol) & (u + v <= 1 + tol)
    return np.mean(ok) > 0.95


if __name__ == '__main__':
    depth = 5
    octant = 0
    cache = Path(f'O{octant}_L{depth}_CA.npy')
    data = np.load(cache)
    xy_or_uv = data[:, :2]
    areas = data[:, 2]
    areas_mean = np.mean(areas)
    rlog = -np.log(areas / areas_mean)

    # Auto-detect coordinate type and convert to (u,v) if needed
    if is_uv_simplex(xy_or_uv):
        uv = xy_or_uv.copy()
        print("Detected simplex (u,v) input from cache.")
    else:
        print("Detected XY input from cache; converting to (u,v) via barycentrics.")
        bary = xy_to_bary(xy_or_uv)
        uv = bary_to_uv(bary)

    u_sum = uv[:, 0] + uv[:, 1]
    print(
        f"UV ranges: u∈[{uv[:, 0].min():.6f},{uv[:, 0].max():.6f}] v∈[{uv[:, 1].min():.6f},{uv[:, 1].max():.6f}] u+v∈[{u_sum.min():.6f},{u_sum.max():.6f}]")

    # Symmetrise the target over D3 to reduce outliers and enforce equilateral symmetry
    uv_sym, r_sym = d3_symmetrise(uv, rlog)
    uv_b, r_b = uv_sym, r_sym
    n_fit = 16
    ns_fac = 20        # denser edge sampling for BC/Neumann rows
    neumann = 1e-3     # small Neumann damping on edges
    lam = 3e-6         # slightly higher ridge for n=16 stability
    bc_w = 1e-2  # 5e-2
    terms, c, col_norm = fit_phi(uv_b, r_b, n=n_fit, n_s_factor=ns_fac, bc_weight=bc_w, lam=lam, neumann_weight=neumann)

    np.savez(f"phi_fit_L{depth}_n{n_fit}.npz",
             terms=np.array(terms, dtype=object),
             c=c,
             col_norm=col_norm,
             n_fit=n_fit,
             depth=depth,
             bc_weight=bc_w,
             neumann_weight=neumann,
             lam=lam,
             n_s_factor=ns_fac,
             laplacian_h=LAPLACIAN_H)

    # terms, c = fit_phi(uv, r)
    print(f'L:{depth}; n_fit:{n_fit}; ns_fac:{ns_fac}; bc_w:{bc_w} neumann:{neumann}; lam:{lam}; fitted {len(c)} terms')
    print(f"laplacian_h: {LAPLACIAN_H}")
    print(f"metric Ginv: [[{G11:.6f},{G12:.6f}],[{G12:.6f},{G22:.6f}]]")
    terms_all = bernstein_terms_deg(n_fit)
    idx = {t: i for i, t in enumerate(terms_all)}
    sel = [idx[t] for t in terms]  # use the exact fitted interior term order
    lap = np.vstack([laplacian_uv(u, v, n_fit, terms_all) for u, v in uv_b])
    M = lap[:, sel]
    print("cond(M):", np.linalg.cond(M))

    # sample BC edge u+v=1
    us = np.linspace(0, 1, 50)
    vs = 1 - us
    du, dv = bernstein_grads_uv(us, vs, n_fit, terms_all)
    nvec = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
    row = du[:, sel] * nvec[0] + dv[:, sel] * nvec[1]
    edge_flux = row @ c
    print("edge flux rms:", np.sqrt(np.mean(edge_flux ** 2)))

    r_pred = M @ c
    print(f"r_b range: [{r_b.min():.3g},{r_b.max():.3g}]  r_pred range: [{r_pred.min():.3g},{r_pred.max():.3g}]")
    res = r_b - r_pred
    print("RMSE:", np.sqrt(np.mean(res ** 2)), " MaxAbs:", np.max(np.abs(res)))
    # Diagnostic: report worst residual and its proximity to edges
    i_max = int(np.argmax(np.abs(res)))
    u_max, v_max = uv_b[i_max]
    d_edge_max = max(0.0, min(u_max, v_max, 1.0 - u_max - v_max))
    where = "boundary-adjacent" if d_edge_max < 5e-3 else "interior-ish"
    print(f"worst |res| at (u,v)=({u_max:.6f},{v_max:.6f}), |res|={abs(res[i_max]):.6g}, d_edge={d_edge_max:.3e} → {where}")

    # exit(0)
    # =====================
    # Visualisations
    # =====================
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    # cache = Path(f'O{octant}_L{depth}_CA.npy')
    # data = np.load(cache)
    # uv = data[:, :2]
    # tri = mtri.Triangulation(uv_b[:, 0], uv_b[:, 1])

    # Triangle outline in (u,v)
    tri_uv = np.array([[0,0],[1,0],[0,1],[0,0]])

    # 1) Raw data (scatter/tricontourf over the sampled centroids)
    tri = mtri.Triangulation(uv_b[:,0], uv_b[:,1])
    fig1 = plt.figure(figsize=(12,12))
    # --- Color scale setup for comparability ---
    # (raw_min, raw_max, vmin, vmax, etc. are set below)
    # Raw log-density
    # (r_b is unchanged here; color scale will be set after we compute vmin/vmax)

    # 2) Fitted field on a dense barycentric grid
    Ngrid = 600
    u = np.linspace(0,1,Ngrid)
    v = np.linspace(0,1,Ngrid)
    UU, VV = np.meshgrid(u, v)
    # mask = (UU>=0) & (VV>=0) & (UU+VV<=1)
    eps_grid = 5e-3
    mask = (UU >= eps_grid) & (VV >= eps_grid) & (UU + VV <= 1.0 - eps_grid)
    U = UU[mask]; V = VV[mask]
    uv_grid = np.column_stack([U, V])
    r_fit_grid = predict_r(uv_grid, n_fit, terms_all, terms, c)

    # --- Align fitted field statistics to raw and clip outliers for plotting ---
    # Align mean so color scale is comparable (no constant-offset ambiguity)
    r_pred_samples = M @ c  # already used later for residuals
    mu_pred = float(r_pred_samples.mean())
    mu_raw  = float(r_b.mean())
    r_fit_grid_adj = r_fit_grid - mu_pred + mu_raw

    # Use the raw range to set a stable color scale
    raw_min, raw_max = float(r_b.min()), float(r_b.max())
    raw_rng = raw_max - raw_min
    vmin = raw_min
    vmax = raw_max

    # Soft clip fitted field to avoid plot domination by rare outliers
    clip_min = raw_min - 0.10 * raw_rng
    clip_max = raw_max + 0.10 * raw_rng
    r_fit_grid_plot = np.clip(r_fit_grid_adj, clip_min, clip_max)

    # Diagnostics for sanity
    print(f"[viz] raw range: [{raw_min:.3f},{raw_max:.3f}]  fitted(grid) pre-clip: "
          f"[{r_fit_grid_adj.min():.3f},{r_fit_grid_adj.max():.3f}]  post-clip: "
          f"[{r_fit_grid_plot.min():.3f},{r_fit_grid_plot.max():.3f}]")

    # 1) Raw data (scatter/tricontourf over the sampled centroids)
    fig1 = plt.figure(figsize=(12,12))
    tcf1 = plt.tricontourf(tri, r_b, levels=30, vmin=vmin, vmax=vmax)
    plt.plot(tri_uv[:,0], tri_uv[:,1])
    plt.gca().set_aspect('equal','box')
    plt.title(f"Raw log-density ℓ (depth {depth})")
    plt.colorbar(tcf1)
    fig1.savefig(f"viz_raw_L{depth}.png", dpi=160)

    # 2) Fitted field on a dense barycentric grid
    fig2 = plt.figure(figsize=(12,12))
    tri_grid = mtri.Triangulation(U, V)
    tcf2 = plt.tricontourf(tri_grid, r_fit_grid_plot, levels=30, vmin=vmin, vmax=vmax)
    plt.plot(tri_uv[:,0], tri_uv[:,1])
    plt.gca().set_aspect('equal','box')
    plt.title(f"Fitted log-density ℓ̂ (depth {depth}, n={n_fit})")
    plt.colorbar(tcf2)
    fig2.savefig(f"viz_fit_L{depth}_n{n_fit}.png", dpi=160)

    # 2b) Fitted field evaluated at the sample centroids (same support as raw)
    fig2b = plt.figure(figsize=(12, 12))
    tcf2b = plt.tricontourf(tri, r_pred_samples, levels=30, vmin=vmin, vmax=vmax)
    plt.plot(tri_uv[:,0], tri_uv[:,1])
    plt.gca().set_aspect('equal','box')
    plt.title(f"Fitted ℓ̂ at centroids (depth {depth}, n={n_fit})")
    plt.colorbar(tcf2b)
    fig2b.savefig(f"viz_fit_centroids_L{depth}_n{n_fit}.png", dpi=160)

    # 3) Residuals at sample centroids
    r_pred_samples = M @ c
    resid = r_b - r_pred_samples
    fig3 = plt.figure(figsize=(12,12))
    tcf3 = plt.tricontourf(tri, resid, levels=30)
    # plt.plot(tri_uv[:,0], tri_uv[:,1])
    plt.gca().set_aspect('equal','box')
    plt.title(f"Residual ℓ - ℓ̂ (depth {depth}, n={n_fit})")
    plt.colorbar(tcf3)
    fig3.savefig(f"viz_resid_L{depth}_n{n_fit}.png", dpi=160)

    # # Optional: edge-flux profile plot along u+v=1
    # fig4 = plt.figure(figsize=(6,3))
    # t = np.linspace(0,1,edge_flux.size)
    # plt.plot(t, edge_flux)
    # plt.axhline(0, lw=1)
    # plt.title("Edge normal flux along u+v=1")
    # plt.xlabel("t along edge")
    # plt.ylabel("∂n φ")
    # fig4.savefig(f"viz_edgeflux_L{depth}_n{n_fit}.png", dpi=160)

    plt.show()
