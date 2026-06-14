import numpy as np
from math import comb
from pathlib import Path

"""
MA Solver (Bernstein, discrete collocation)
-----------------------------------------
This script assumes you have already run `ma_precondition.py` to produce
`precond_O{octant}_L{depth}_n{deg}.npy`, which stores per-centroid samples in the
(u,v) simplex with an associated log-density field ℓ = -log(A/Ā).

Workflow:
1) Load preconditioned (u,v) and ℓ from the file.
2) Solve for a Bernstein potential ψ so that log det(Hψ_xy) ≈ -ℓ (up to a constant).
   In uv-coordinates, log det(Hψ_uv) ≈ (-ℓ) + 2 log|det J|. We center the RHS to
   remove the constant net_mode ambiguity.
3) Save ψ coefficients and plot Raw ℓ, Fitted ℓ̂, and Residual with shared color scales.

Key conventions:
- (u,v) are barycentric simplex coordinates (u=b, v=c).
- xy geometry is the √2 equilateral; metric handled by J_uv_to_xy and Ginv.
- Output `ma_psi_*.npz` contains (terms, c, depth, parity, J, Ginv, detJ).
"""


# Finite-difference step for Laplacian in (u,v); will be overridden by NPZ if present
global LAPLACIAN_H

# Parity of the octant triangle: "male" = apex up, "female" = apex down.
PARITY = "female"  # set to "male" if your face uses apex-up orientation

# === Geometry of the √2 equilateral (centroid at origin) ===
# Side s = √2; altitude h = s*√3/2 = √6/2
S = np.sqrt(2.0)
H = np.sqrt(6.0) / 2.0
if PARITY == "male":
    # Vertices in XY (counter-clockwise): top A, bottom-left B, bottom-right C (apex up)
    A_xy = np.array([0.0,  2.0*H/3.0])
    B_xy = np.array([-S/2.0, -H/3.0])
    C_xy = np.array([ S/2.0, -H/3.0])
else:
    # "female": apex down. Keep CCW order: bottom A, top-right B, top-left C
    A_xy = np.array([0.0, -2.0*H/3.0])
    B_xy = np.array([ S/2.0,  H/3.0])
    C_xy = np.array([-S/2.0,  H/3.0])
print(f"Using parity: {PARITY}")

# Precompute linear map for barycentric <-> XY
# For a point P = a*A + b*B + c*C with a+b+c=1. Using B as origin:
M = np.column_stack((A_xy - B_xy, C_xy - B_xy))   # 2x2
Minv = np.linalg.inv(M)
J_uv_to_xy = np.column_stack((B_xy - A_xy, C_xy - A_xy))  # 2x2
G = J_uv_to_xy.T @ J_uv_to_xy                             # 2x2 metric
Ginv = np.linalg.inv(G)
G11, G12, G22 = Ginv[0,0], Ginv[0,1], Ginv[1,1]

def xy_to_bary(xy):
    """Vectorized: XY -> barycentric (a,b,c) w.r.t. (A,B,C). xy: (hex_layer,2)."""
    XY = np.asarray(xy, float)
    rel = (XY - B_xy)  # (hex_layer,2)
    lam = rel @ Minv.T  # (hex_layer,2) gives [a, c]
    a = lam[:, 0]
    c = lam[:, 1]
    b = 1.0 - a - c
    return np.column_stack([a, b, c])

def bary_to_uv(bary):
    """(a,b,c) -> (u,v) with u=b, v=c for MA basis."""
    a, b, c = bary[:,0], bary[:,1], bary[:,2]
    return np.column_stack([b, c])

def is_uv_simplex(uv_like, tol=1e-6):
    """Heuristic: return True if points lie in (u>=0,v>=0,u+v<=1) for most samples."""
    U = uv_like[:, 0]; V = uv_like[:, 1]
    ok = (U >= -tol) & (V >= -tol) & (U + V <= 1 + tol)
    return np.mean(ok) > 0.95

def rmse_for_perm(bary, rlog, n_fit, terms_all, terms, c, abc_idx):
    a_idx, b_idx, c_idx = abc_idx
    u = bary[:, b_idx]
    v = bary[:, c_idx]
    idx = {t:i for i,t in enumerate(terms_all)}
    sel = [idx[t] for t in terms]
    M = np.vstack([laplacian_uv(uu, vv, n_fit, terms_all)[sel] for (uu, vv) in zip(u, v)])
    r_pred = M @ c
    resid = rlog - r_pred
    rmse = float(np.sqrt(np.mean(resid**2)))
    return rmse, u, v, r_pred, resid

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


# === Analytic Hessian of Bernstein basis in (u,v) ===
def bernstein_hess_uv(u, v, n, terms=None):
    """Analytic Hessian components of the Bernstein basis in (u,v).
    Returns three arrays (huu, huv, hvv) each of shape (K,), where
    f_uu = sum c_k * huu[k], etc. Uses a=1-u-v, b=u, c=v and
    ∂/∂u = ∂/∂b - ∂/∂a,  ∂/∂v = ∂/∂c - ∂/∂a.
    """
    if terms is None:
        terms = bernstein_terms_deg(n)
    a = 1.0 - u - v
    b = u
    c = v
    K = len(terms)
    huu = np.empty(K, dtype=np.float64)
    huv = np.empty(K, dtype=np.float64)
    hvv = np.empty(K, dtype=np.float64)
    for t, (i, j, k) in enumerate(terms):
        C = comb(n, i) * comb(n - i, j)
        # Base monomial and its logs are not needed; compute second partials directly.
        # d^2/da^2 (a^i b^j c^k) = i*(i-1)*a^{i-2} b^j c^k (if i>=2) else 0, etc.
        d2_da2 = 0.0
        d2_db2 = 0.0
        d2_dc2 = 0.0
        d2_dadb = 0.0
        d2_dadc = 0.0
        d2_dbdc = 0.0
        if i >= 2:
            d2_da2 = C * i * (i-1) * (a**(i-2)) * (b**j) * (c**k)
        if j >= 2:
            d2_db2 = C * j * (j-1) * (a**i) * (b**(j-2)) * (c**k)
        if k >= 2:
            d2_dc2 = C * k * (k-1) * (a**i) * (b**j) * (c**(k-2))
        if (i >= 1) and (j >= 1):
            d2_dadb = - C * i * j * (a**(i-1)) * (b**(j-1)) * (c**k)
        if (i >= 1) and (k >= 1):
            d2_dadc = - C * i * k * (a**(i-1)) * (b**j) * (c**(k-1))
        if (j >= 1) and (k >= 1):
            d2_dbdc =   C * j * k * (a**i)     * (b**(j-1)) * (c**(k-1))
        # Transform to (u,v): ∂u = ∂b - ∂a, ∂v = ∂c - ∂a.
        # Using multi-variable chain rules:
        # f_uu = f_bb + f_aa - 2 f_ab
        # f_vv = f_cc + f_aa - 2 f_ac
        # f_uv = f_bc - f_ab - f_ac + f_aa
        f_uu = d2_db2 + d2_da2 - 2.0 * d2_dadb
        f_vv = d2_dc2 + d2_da2 - 2.0 * d2_dadc
        f_uv = d2_dbdc - d2_dadb - d2_dadc + d2_da2
        huu[t] = f_uu
        hvv[t] = f_vv
        huv[t] = f_uv
    return huu, huv, hvv

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

def laplacian_uv(u, v, n, terms=None):
    # Δ = ∂²/∂u² + ∂²/∂v² + 2 ∂²/∂u∂v  (a=1-u-v)
    # Central differences; avoid projection unless near edges for unbiased interior stencil.
    if terms is None:
        terms = bernstein_terms_deg(n)
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


# === Monge–Ampère solver: discrete collocation, analytic Bernstein Hessians ===
def solve_ma_bernstein(uv, ell, deg, lam=1e-6, iters=20, damping=0.5):
    """Solve log det(Hψ_xy) ≈ -ℓ at centroid samples using a Bernstein basis of degree `deg`.
    Returns (terms, c_psi, history) where c_psi are the coefficients of ψ.
    Uses Gauss–Newton on the residual r = log det(Hψ_xy) + ℓ.
    The XY metric is handled by H_xy = J^{-T} H_uv J^{-1}, so det(H_xy)=det(H_uv)/det(J)^2.
    """
    terms_all = bernstein_terms_deg(deg)
    K = len(terms_all)
    alpha_edge = 0.4  # unchanged, currently unused

    u, v = uv[:, 0], uv[:, 1]
    m = uv.shape[0]

    # Corner/edge emphasis (very light): upweight samples within ~5% of edges.
    # This helps shave the small residual lobes seen near the three corners.
    s_edge = np.minimum.reduce([u, v, 1.0 - u - v])           # distance to closest edge
    edge_band = 0.04                                           # width of the edge band
    edge_gain = 1.5                                            # how much extra weight at the very edge
    edge = np.clip(edge_band - s_edge, 0.0, None) / edge_band  # 0 in interior → 1 at edge
    w_edge = 1.0 + edge_gain * (edge ** 2)                     # smooth, gentle upweighting

    # Precompute per-sample Hessian basis in uv
    Huu = np.empty((m, K), dtype=np.float64)
    Huv = np.empty((m, K), dtype=np.float64)
    Hvv = np.empty((m, K), dtype=np.float64)
    for i, (u, v) in enumerate(uv):
        huu, huv, hvv = bernstein_hess_uv(u, v, deg, terms_all)
        Huu[i, :] = huu
        Huv[i, :] = huv
        Hvv[i, :] = hvv
    # --- Bernstein fix #1: scale Hessian basis by n(n-1) to tame magnitudes ---
    scale = max(deg * (deg - 1), 1)
    Huu /= scale
    Hvv /= scale
    Huv /= scale

    # Precompute normal-equation pieces; whiten Huv so cross-term penalty bites in comparable units
    HT_Huu = Huu.T @ Huu
    HT_Hvv = Hvv.T @ Hvv
    # Column-whiten Huv (solver and plot must do the same)
    _col_huv = np.linalg.norm(Huv, axis=0) + 1e-12
    Huv /= _col_huv
    HT_Huv = Huv.T @ Huv

    # Metric transform: H_xy = J^{-T} H_uv J^{-1}
    # For determinants: det(H_xy) = det(H_uv) / det(J)^2. Constant scale factor.
    detJ = float(np.linalg.det(J_uv_to_xy))
    log_detJ2 = 2.0 * np.log(abs(detJ))
    # Target: log det(Hψ_uv) ≈ (-ell) + 2 log|detJ|
    rhs = (-ell) + log_detJ2
    # Remove constant net_mode: MA only identifies ψ up to additive quadratics/affine terms.
    rhs_mean = float(rhs.mean())
    rhs = rhs - rhs_mean
    print(f"  MA target centering: log|detJ|^2={log_detJ2:.6f}, rhs_mean(before)={rhs_mean:.6f}")
    allow_beta = False  # keep absolute contrast; only initial centering is applied
    beta_cum = 0.0

    # --- SPD tether & scaling guards (constants; per-stage μ computed inside the loop) ---
    mu_tether   = 0.015     # keep light tether
    gamma_cross = 32.0     # a touch stronger shear suppression
    s_floor     = 3e-5     # slightly looser floors to permit more curvature variation
    det_floor   = 3e-5
    max_step_k  = 0.12     # slightly smaller steps to avoid edge overshoot
    print(f"  SPD tether: mu={mu_tether:g}, gamma={gamma_cross:g}; hess-scale=1/(n(n-1))")
    # Notes: lighter mu_tether + looser floors let log-det deviate from 0 more readily.
    # Larger max_step_k and finer homotopy alphas help escape the nearly-constant basin.

    # --- Homotopy continuation: start from constant log-det then morph to rhs ---
    rhs_const = np.full_like(rhs, rhs.mean())

    def ls_init_for(target_rhs):
        """Least-squares initialization for a given target RHS (log-det(H_uv)).
        Uses column-normalized ridge LS to avoid enormous coefficients and gently
        prefers C≈0 so the initial Hessian is close to isotropic SPD.
        """
        s_target = np.exp(0.5 * target_rhs)  # want A≈B≈s, C≈0 ⇒ det≈s^2
        # Stack operators for A, B, C
        H_stack = np.vstack([Huu, Hvv, Huv])          # (3m, K)
        t_stack = np.concatenate([s_target, s_target, np.zeros_like(s_target)])
        # Column-normalize to reduce condition number
        coln = np.linalg.norm(H_stack, axis=0) + 1e-12
        Hn = H_stack / coln
        # Lightly down-weight the C rows so they don't dominate
        wA = 1.0; wB = 1.0; wC = 0.5
        W = np.concatenate([np.full(m, wA), np.full(m, wB), np.full(m, wC)])
        Hnw = Hn * W[:, None]
        tnw = t_stack * W
        # Ridge solve with safe lambda
        lam0 = max(lam, 1e-3)
        JTJ0 = Hnw.T @ Hnw
        rhs0 = Hnw.T @ tnw
        JTJ0.flat[::JTJ0.shape[0] + 1] += lam0
        try:
            c0_scaled = np.linalg.solve(JTJ0, rhs0)
        except np.linalg.LinAlgError:
            c0_scaled = np.linalg.lstsq(JTJ0, rhs0, rcond=None)[0]
        # Undo column scaling
        c0 = c0_scaled / coln
        return c0

    # Initialize at constant target
    c = ls_init_for(rhs_const)
    # Evaluate initial residual
    A = np.maximum(Huu @ c, s_floor)
    B = np.maximum(Hvv @ c, s_floor)
    C = Huv @ c
    det_uv = np.maximum(A * B - C * C, det_floor)
    r = np.log(det_uv) - rhs_const
    rmse0 = float(np.sqrt(np.mean(r * r)))
    print(f"  MA init(const): rmse0={rmse0:.6f}")
    hist = [rmse0]
    # SPD/scale diagnostics at init
    det_min, det_med = float(det_uv.min()), float(np.median(det_uv))
    tiny = np.mean(det_uv < det_floor)
    print(f"  init det(H_uv): min={det_min:.3e} median={det_med:.3e} tiny_frac<{det_floor:.0e}={tiny:.3f}")
    if rmse0 > 5.0:
        print("  init rmse large → re-initialize with stronger ridge")
        c = ls_init_for(np.zeros_like(rhs))  # s_target=1 baseline
        A = np.maximum(Huu @ c, s_floor); B = np.maximum(Hvv @ c, s_floor); C = Huv @ c
        det_uv = np.maximum(A * B - C * C, det_floor)
        r = np.log(det_uv) - rhs_const
        rmse0 = float(np.sqrt(np.mean(r * r)))
        print(f"  MA init(retry): rmse0={rmse0:.6f}")
        hist[0] = rmse0

    # Homotopy blend fractions from constant → full target (tunable)
    alphas = [0.03, 0.10, 0.22, 0.38, 0.55, 0.72, 0.85, 0.92, 0.955, 0.975, 0.987, 0.994, 1.00]
    # alphas = [0.05, 0.15, 0.35, 0.65, 0.85, 0.92, 0.97, 0.985, 1.00]  # extra refinement near 1
    max_iter_per_stage = 32
    rmse = rmse0

    for a in alphas:
        rhs_a = (1.0 - a) * rhs_const + a * rhs
        print(f"  MA stage α={a:.2f}: target=blend(const,{a:.2f})")
        # Do not let the tether vanish; leave a 10% floor so scale stays controlled.
        mu_stage = mu_tether * ((1.0 - a)**2 + 0.05) * (1.0 + 0.5*rmse)
        # Re-center residual to current target before GN
        A = np.maximum(Huu @ c, s_floor)
        B = np.maximum(Hvv @ c, s_floor)
        C = Huv @ c
        det_uv = np.maximum(A * B - C * C, det_floor)
        print(f"    diag α={a:.2f}: C/√(AB) med={np.median(np.abs(C)/np.sqrt(A*B)):.3f}")
        r = np.log(det_uv) - rhs_a
        if allow_beta:
            beta = float(r.mean())
            r -= beta
            beta_cum += beta
        rmse = float(np.sqrt(np.mean(r * r)))
        hist.append(rmse)
        dmp = damping
        lam_gn = max(lam, 3e-4)
        # lam_gn = max(lam, 1e-5)
        for it in range(max_iter_per_stage):
            # Jacobian at current c for rhs_a
            Jk = (Huu * B[:, None]) + (Hvv * A[:, None]) - 2.0 * (Huv * C[:, None])
            Jk /= det_uv[:, None]
            # --- Bernstein fix #3: row-normalize Jacobian to reduce row dominance ---
            rown = np.sqrt(1.0 + np.sum(Jk * Jk, axis=1))
            Jk /= rown[:, None]
            r_scaled = r / rown
            Jk *= w_edge[:, None]
            r_scaled = r_scaled * w_edge
            JTJ = Jk.T @ Jk
            rhs_gn = - Jk.T @ r_scaled
            # --- SPD tether: tether A,B toward the desired per‑sample scale s_target, and C toward 0.
            s_target = np.exp(0.5 * rhs_a)  # shape (m,)
            A_err = (s_target - A)
            B_err = (s_target - B)
            C_err = (- C)
            JTJ += mu_stage * (HT_Huu + HT_Hvv) + (mu_stage * gamma_cross) * HT_Huv
            rhs_gn += mu_stage * (Huu.T @ A_err + Hvv.T @ B_err) + (mu_stage * gamma_cross) * (Huv.T @ C_err)
            # Small coefficient ridge to discourage huge coefficients
            JTJ.flat[::JTJ.shape[0] + 1] += 1e-4
            JTJ.flat[::JTJ.shape[0] + 1] += lam_gn
            try:
                delta = np.linalg.solve(JTJ, rhs_gn)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(JTJ, rhs_gn, rcond=None)[0]

            # simple trust-region on coefficient step
            step_norm = np.linalg.norm(delta)
            max_step = max_step_k * (np.linalg.norm(c) + 1.0)  # was 0.10*
            if step_norm > max_step:
                delta *= (max_step / step_norm)

            c_new = c + dmp * delta

            # Evaluate new residual
            A_new = np.maximum(Huu @ c_new, s_floor)
            B_new = np.maximum(Hvv @ c_new, s_floor)
            C_new = Huv @ c_new
            det_new = np.maximum(A_new * B_new - C_new * C_new, max(det_floor, 1e-3))
            r_new = np.log(det_new) - rhs_a
            if allow_beta:
                r_new -= float(r_new.mean())
            rmse_new = float(np.sqrt(np.mean(r_new * r_new)))
            if rmse_new <= rmse:
                c = c_new
                A, B, C = A_new, B_new, C_new
                det_uv = det_new
                r = r_new
                rmse = rmse_new
                hist[-1] = rmse
                print(f"    it {it + 1:02d}: rmse={rmse:.6f} (accepted, damping={dmp})")
                dmp = min(1.0, dmp * 1.2)
                lam_gn = max(lam, lam_gn / 1.5)
            else:
                dmp *= 0.5
                lam_gn = lam_gn * 3.0
                print(f"    it {it + 1:02d}: no improvement, damping→{dmp}, lam→{lam_gn:.1e}")
                if dmp < 1e-3:
                    break
        print(f"  stage α={a:.2f}: det min/med = {det_uv.min():.2e}/{np.median(det_uv):.2e}  (β_cum={beta_cum:.3e})")

    # After homotopy, do a final polish against full rhs for any remaining iterations
    remaining = max(128, iters - (len(alphas) + 1) * max_iter_per_stage)  # guarantee at least a few polish steps
    lam_gn = max(lam, 3e-4)
    mu_polish = mu_tether * 0.02   # lighter tether for final fine-tuning
    for it in range(remaining):
        A = np.maximum(Huu @ c, s_floor)
        B = np.maximum(Hvv @ c, s_floor)
        C = Huv @ c
        det_uv = np.maximum(A * B - C * C, det_floor)
        print(f"  polish diag: C/√(AB) med={np.median(np.abs(C)/np.sqrt(A*B)):.3f}")
        r = np.log(det_uv) - rhs
        if allow_beta:
            r -= float(r.mean())
        rmse = float(np.sqrt(np.mean(r * r)))
        hist.append(rmse)
        Jk = (Huu * B[:, None]) + (Hvv * A[:, None]) - 2.0 * (Huv * C[:, None])
        Jk /= det_uv[:, None]
        # --- Bernstein fix #3: row-normalize Jacobian to reduce row dominance ---
        rown = np.sqrt(1.0 + np.sum(Jk * Jk, axis=1))
        Jk /= rown[:, None]
        r_scaled = r / rown
        # apply edge weights consistently in polish as well
        Jk *= w_edge[:, None]
        r_scaled = r_scaled * w_edge
        JTJ = Jk.T @ Jk
        rhs_gn = - Jk.T @ r_scaled
        # --- SPD tether: Tether to final target scale as well
        s_target = np.exp(0.5 * rhs)
        A_err = (s_target - A)
        B_err = (s_target - B)
        C_err = (- C)
        JTJ += mu_polish * (HT_Huu + HT_Hvv) + (mu_polish * gamma_cross) * HT_Huv
        rhs_gn += mu_polish * (Huu.T @ A_err + Hvv.T @ B_err) + (mu_polish * gamma_cross) * (Huv.T @ C_err)
        JTJ.flat[::JTJ.shape[0] + 1] += lam_gn + 1e-4
        try:
            delta = np.linalg.solve(JTJ, rhs_gn)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(JTJ, rhs_gn, rcond=None)[0]
        c_new = c + damping * delta
        A_new = np.maximum(Huu @ c_new, s_floor)
        B_new = np.maximum(Hvv @ c_new, s_floor)
        C_new = Huv @ c_new
        det_new = np.maximum(A_new * B_new - C_new * C_new, det_floor)
        r_new = np.log(det_new) - rhs
        if allow_beta:
            r_new -= float(r_new.mean())
        rmse_new = float(np.sqrt(np.mean(r_new * r_new)))
        if rmse_new <= rmse:
            c = c_new
            hist[-1] = rmse_new
            print(f"  polish it {it + 1:02d}: rmse={rmse_new:.6f} (accepted)")
            lam_gn = max(lam, lam_gn / 1.5)
        else:
            damping *= 0.5
            lam_gn = lam_gn * 3.0
            print(f"  polish it {it + 1:02d}: no improvement, damping→{damping}, lam→{lam_gn:.1e}")
            if damping < 1e-3:
                break

    return terms_all, c, np.array(hist)


if __name__ == '__main__':
    # --- Configuration ---
    PARITY = "female"      # ensure matches your octant parity
    depth = 5               # choose octant hex_layer
    octant = 0              # which octant to process
    ma_deg = 16             # MA potential ψ
    pcon_d = 16             # Preconditioned Bernstein degree

    # Geometry constants (already defined above): J_uv_to_xy, Ginv, etc.

    # --- Load MA input bundle (from prewarp phase) ---
    ma_input_path = Path(f"ma_alt_input_L{depth}_n{pcon_d}.npz")
    ma_in = np.load(ma_input_path, allow_pickle=True)
    uv = ma_in["uv_cent"]
    rlog = ma_in["ell"]
    terms = ma_in["terms"]
    pcon_d = int(ma_in["degree"])
    c_init = ma_in.get("c_init", None)
    meta = ma_in.get("meta", {})

    print(f"Loaded MA input: {ma_input_path}")
    print(f"  uv_cent shape={uv.shape}, ell shape={rlog.shape}, terms={len(terms)}, degree={pcon_d}")
    print(f"  ell stats: min={rlog.min():.4f}, max={rlog.max():.4f}, mean={rlog.mean():.4f}, std={rlog.std():.4f}")

    # --- Solve Monge–Ampère in Bernstein basis ---
    print("Starting MA solve (discrete collocation, Bernstein basis)...")
    terms_ma, c_ma, hist = solve_ma_bernstein(uv, rlog, ma_deg, lam=1e-6, iters=30, damping=0.5)
    print(f"MA done: deg={ma_deg}, iters={len(hist)}, final rmse={hist[-1]:.6f}")

    # Save ψ coefficients and metadata
    np.savez(f"ma_alt_psi_L{depth}_n{ma_deg}.npz",
             terms=np.array(terms_ma, dtype=object), c=c_ma, depth=depth,
             parity=PARITY, J=J_uv_to_xy, Ginv=Ginv, detJ=np.linalg.det(J_uv_to_xy))


    # --- Visualise raw ℓ, fitted ℓ̂ from ψ, and residual ---
    try:
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri
        import matplotlib.colors as mcolors

        tri = mtri.Triangulation(uv[:,0], uv[:,1])
        tri_uv = np.array([[0,0],[1,0],[0,1],[0,0]])

        # Reconstruct ℓ̂ from ψ: log det(Hψ_xy) = log det(Hψ_uv) - 2 log|detJ|
        terms_all = bernstein_terms_deg(ma_deg)
        m = uv.shape[0]
        K = len(terms_all)

        # Build Hessian basis at the same scale used in the solver
        Huu = np.empty((m, K), dtype=np.float64)
        Huv = np.empty((m, K), dtype=np.float64)
        Hvv = np.empty((m, K), dtype=np.float64)
        for i, (u_i, v_i) in enumerate(uv):
            huu, huv, hvv = bernstein_hess_uv(u_i, v_i, ma_deg, terms_all)
            Huu[i, :] = huu
            Huv[i, :] = huv
            Hvv[i, :] = hvv
        # Apply the same Bernstein Hessian scaling: divide by n(n-1)
        scale = max(ma_deg * (ma_deg - 1), 1)
        Huu /= scale
        Huv /= scale
        Hvv /= scale

        # Apply the same Huv whitening as in the solver
        _col_huv = np.linalg.norm(Huv, axis=0) + 1e-12
        Huv /= _col_huv

        # Form Hessian components and enforce SPD floors, exactly like the solver
        s_floor_plot = 1e-5
        det_floor_plot = 1e-5
        A = np.maximum(Huu @ c_ma, s_floor_plot)   # f_uu
        B = np.maximum(Hvv @ c_ma, s_floor_plot)   # f_vv
        C = Huv @ c_ma                             # f_uv
        det_uv = np.maximum(A * B - C * C, det_floor_plot)
        print(f"plot diag: A[min/med/max]={A.min():.3e}/{np.median(A):.3e}/{A.max():.3e} "
              f"B[min/med/max]={B.min():.3e}/{np.median(B):.3e}/{B.max():.3e} "
              f"C[min/med/max]={C.min():.3e}/{np.median(C):.3e}/{C.max():.3e} "
              f"det[min/med]={det_uv.min():.3e}/{np.median(det_uv):.3e} "
              f"C/√(AB) med={np.median(np.abs(C)/np.sqrt(A*B)):.3f}")

        # Convert to XY determinant (constant metric factor), then center and flip sign
        log_det_xy = np.log(det_uv) - 2.0 * np.log(abs(np.linalg.det(J_uv_to_xy)))
        lhat = - (log_det_xy - log_det_xy.mean())  # predicted ℓ̂ aligned to raw ℓ centering
        resid = rlog - lhat

        # Diagnostics for fitted field (must come after lhat is defined)
        lhat_std = float(lhat.std())
        corr = float(np.corrcoef(rlog, lhat)[0, 1]) if lhat_std > 0 else np.nan
        print(f"diag: std(lhat)={lhat_std:.4f}  corr(raw, fit)={corr:.3f}")

        # shared color scaling for raw & fit, zero-centred for residual
        vmin = float(min(rlog.min(), lhat.min()))
        vmax = float(max(rlog.max(), lhat.max()))
        raw_fit_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        raw_fit_levels = np.linspace(vmin, vmax, 30)
        rabs = float(max(abs(resid.min()), abs(resid.max())))
        resid_norm = mcolors.TwoSlopeNorm(vmin=-rabs, vcenter=0.0, vmax=rabs)
        resid_levels = np.linspace(-rabs, rabs, 30)
        print(f"Plot norms: raw/fit vmin={vmin:.3f} vmax={vmax:.3f}; resid ±{rabs:.3f}")

        # Raw ℓ
        fig1 = plt.figure(figsize=(8, 8))
        tcf1 = plt.tricontourf(tri, rlog, levels=raw_fit_levels, norm=raw_fit_norm, cmap='viridis')
        plt.plot(tri_uv[:,0], tri_uv[:,1], 'k-')
        plt.gca().set_aspect('equal','box'); plt.title("Raw ℓ (simplex coords)")
        plt.xlabel('u'); plt.ylabel('v'); plt.colorbar(tcf1)
        fig1.savefig(f"ma_raw_L{depth}.png", dpi=160)

        # Fitted ℓ̂
        fig2 = plt.figure(figsize=(8, 8))
        tcf2 = plt.tricontourf(tri, lhat, levels=raw_fit_levels, norm=raw_fit_norm, cmap='viridis')
        plt.plot(tri_uv[:,0], tri_uv[:,1], 'k-')
        plt.gca().set_aspect('equal','box'); plt.title(f"MA fitted ℓ̂ (n={ma_deg})")
        plt.xlabel('u'); plt.ylabel('v'); plt.colorbar(tcf2)
        fig2.savefig(f"ma_fit_L{depth}_n{ma_deg}.png", dpi=160)

        # Residual
        fig3 = plt.figure(figsize=(8, 8))
        tcf3 = plt.tricontourf(tri, resid, levels=resid_levels, norm=resid_norm, cmap='viridis')
        plt.plot(tri_uv[:,0], tri_uv[:,1], 'k-')
        plt.gca().set_aspect('equal','box'); plt.title("MA residual ℓ - ℓ̂")
        plt.xlabel('u'); plt.ylabel('v'); plt.colorbar(tcf3)
        fig3.savefig(f"ma_resid_L{depth}_n{ma_deg}.png", dpi=160)
        plt.close('all')
    except Exception as e:
        print("Plotting skipped:", e)
