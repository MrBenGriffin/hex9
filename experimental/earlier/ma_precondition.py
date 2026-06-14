import numpy as np
from itertools import permutations

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

from math import comb
from pathlib import Path
import numpy as np


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


if __name__ == '__main__':
    # Finite-difference step for Laplacian in (u,v); will be overridden by NPZ if present
    LAPLACIAN_H = 1e-4
    depth = 5
    octant = 0
    # Load base data
    cache = Path(f'O{octant}_L{depth}_CA.npy')
    data = np.load(cache)
    xy = data[:, :2]
    areas = data[:, 2]   # in sqm (covering approx 1/8 global surface area)

    # Force XY→bary→(u,v) to avoid mis-detection; your caches store XY on the √2 equilateral.
    bary = xy_to_bary(xy)          # (a,b,c)

    areas_mean = np.mean(areas)
    rlog = -np.log(areas / areas_mean)

    # Load Bernstein coefficients
    d = np.load(f'phi_fit_L{depth}_n12.npz', allow_pickle=True)
    terms = [tuple(x) for x in d["terms"]]
    c = d["c"]
    n_fit = int(d["n_fit"])
    depth_fit = int(d.get("depth", depth))
    print(f"Loaded φ fit: depth={depth_fit}, n_fit={n_fit}")
    assert depth_fit == depth, f"Depth mismatch: fit depth {depth_fit} vs data depth {depth}"
    # Synchronize Laplacian step with the fit
    if "laplacian_h" in d:
        LAPLACIAN_H = float(d["laplacian_h"])  # noqa: F841 (used inside laplacian_uv via closure/global)
        print(f"Using laplacian_h from fit: {LAPLACIAN_H:g}")
    else:
        print("Warning: laplacian_h missing in NPZ; using default 1e-4")

    # Build full term list for n_fit
    terms_all = bernstein_terms_deg(n_fit)

    # We will choose the (a,b,c) → (u,v) mapping that best matches the fitted coefficients
    # by minimizing RMSE between rlog and r_pred across the 6 permutations.
    perms = [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]
    best = None
    best_tuple = None
    for p in perms:
        rm, u_tmp, v_tmp, r_pred_tmp, resid_tmp = rmse_for_perm(bary, rlog, n_fit, bernstein_terms_deg(n_fit), terms, c, p)
        if (best is None) or (rm < best):
            best = rm
            best_tuple = (p, u_tmp, v_tmp, r_pred_tmp, resid_tmp)
    (a_idx, b_idx, c_idx), u, v, r_pred, resid = best_tuple
    print(f"Chosen vertex mapping a,b,c ← bary indices {a_idx},{b_idx},{c_idx} (RMSE={best:.6f})")
    uv = np.column_stack([u, v])

    # Diagnostics
    s = u + v
    print(f"XY ranges: x∈[{xy[:,0].min():.3f},{xy[:,0].max():.3f}] y∈[{xy[:,1].min():.3f},{xy[:,1].max():.3f}]")
    print(f"UV ranges (after perm): u∈[{u.min():.6f},{u.max():.6f}] v∈[{v.min():.6f},{v.max():.6f}] u+v∈[{s.min():.6f},{s.max():.6f}]")

    rmse = float(np.sqrt(np.mean(resid**2)))
    maxabs = float(np.max(np.abs(resid)))
    print(f"Precond fit on depth {depth}: RMSE={rmse:.6f}  MaxAbs={maxabs:.6f}")

    print(f"rlog range: [{rlog.min():.3g},{rlog.max():.3g}]  r_pred range: [{r_pred.min():.3g},{r_pred.max():.3g}]")
    # Try a scalar rescale to detect Laplacian-step mismatches
    denom = float(np.dot(r_pred, r_pred))
    if denom > 0:
        s_star = float(np.dot(r_pred, rlog) / denom)
        rmse_s = float(np.sqrt(np.mean((rlog - s_star*r_pred)**2)))
        print(f"Best scalar scale s*={s_star:.3g} → RMSE={rmse_s:.6f}")
    else:
        print("Warning: r_pred has zero norm; check basis/terms alignment.")

    # Compute gradient of phi at centroids
    gx, gy = grad_phi(uv, n_fit, terms_all, terms, c)   # arrays (hex_layer,)

    # Seam-safe taper (zero displacement on edges) + **step clipping**
    # u = uv[:,0]; v = uv[:,1]   # Removed redundant redefinition
    a = bary[:, a_idx]
    d_edge = np.minimum.reduce([a, u, v])
    delta = 0.04                                       # ~4% of edge length
    # Smooth ramp that is exactly 0 on edges and ~1 in the interior
    w = (d_edge / (delta + d_edge))**2

    # Raw displacement
    du = -w * gx
    dv = -w * gy

    # Clip step so we **never** cross an edge in one update
    # Constraints: u+du >= eps, v+dv >= eps, u+v+du+dv <= 1-eps
    eps = 1e-12
    s1 = np.where(du < 0.0, (u - eps) / (-du + eps), 1.0)     # keep u >= eps
    s2 = np.where(dv < 0.0, (v - eps) / (-dv + eps), 1.0)     # keep v >= eps
    s3 = np.where(du + dv > 0.0, ((1.0 - eps) - (u + v)) / (du + dv + eps), 1.0)  # keep u+v <= 1-eps
    s = np.minimum(np.minimum(s1, s2), s3)

    # Also cap by a fraction of the distance-to-edge so near-edge points barely move
    alpha = 0.9
    step_norm = np.hypot(du, dv) + eps
    s_cap = alpha * d_edge / step_norm
    s = np.minimum(s, s_cap)

    # Apply clipped step
    u_new = u + s * du
    v_new = v + s * dv

    uv_corr = np.column_stack([u_new, v_new])

    # Seam check: max |displacement| within 1e-4 of edges should be ~0 (due to taper)
    near_edge = d_edge < 1e-4
    print(f"Near-edge points counted: {near_edge.sum()} / {near_edge.size}")
    seam_disp = np.hypot((u_new - u)[near_edge], (v_new - v)[near_edge])
    if seam_disp.size:
        print(f"Seam displacement (near-edge) max: {np.max(seam_disp):.3e}  rms: {np.sqrt(np.mean(seam_disp**2)):.3e}")

    # Save outputs for pipeline
    out = np.column_stack([uv, uv_corr, w, gx, gy, rlog, r_pred, resid])
    out_path = Path(f"precond_O{octant}_L{depth}_n{n_fit}.npy")
    np.save(out_path, out)
    print(f"Saved preconditioned centroids → {out_path}")

    # Optional visualisations
    try:
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri
        import matplotlib.colors as mcolors
        tri = mtri.Triangulation(uv[:,0], uv[:,1])
        tri_uv = np.array([[0,0],[1,0],[0,1],[0,0]])

        # Shared scaling for RAW and FITTED
        vmin = float(min(rlog.min(), r_pred.min()))
        vmax = float(max(rlog.max(), r_pred.max()))
        raw_fit_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        raw_fit_levels = np.linspace(vmin, vmax, 30)
        # Symmetric, zero-centered scaling for RESIDUAL
        rabs = float(max(abs(resid.min()), abs(resid.max())))
        resid_norm = mcolors.TwoSlopeNorm(vmin=-rabs, vcenter=0.0, vmax=rabs)
        resid_levels = np.linspace(-rabs, rabs, 30)
        print(f"Plot norms: raw/fit vmin={vmin:.3f} vmax={vmax:.3f}; resid ±{rabs:.3f}")

        # raw
        fig1 = plt.figure(figsize=(5.6,5.6))
        tcf1 = plt.tricontourf(tri, rlog, levels=raw_fit_levels, norm=raw_fit_norm, cmap='viridis')
        plt.plot(tri_uv[:,0], tri_uv[:,1], 'k-')
        plt.gca().set_aspect('equal','box'); plt.title("Raw ℓ (simplex coords)")
        plt.xlabel('u'); plt.ylabel('v')
        plt.colorbar(tcf1); fig1.savefig(f"ma_raw_L{depth}.png", dpi=160)

        # fit
        fig2 = plt.figure(figsize=(8,8))
        tcf2 = plt.tricontourf(tri, r_pred, levels=raw_fit_levels, norm=raw_fit_norm, cmap='viridis')
        plt.plot(tri_uv[:,0], tri_uv[:,1], 'k-')
        plt.gca().set_aspect('equal','box'); plt.title(f"Fitted ℓ̂ (n={n_fit})")
        plt.xlabel('u'); plt.ylabel('v')
        plt.colorbar(tcf2); fig2.savefig(f"ma_fit_L{depth}_n{n_fit}.png", dpi=160)

        # residual
        fig3 = plt.figure(figsize=(8,8))
        tcf3 = plt.tricontourf(tri, resid, levels=resid_levels, norm=resid_norm, cmap='viridis')
        plt.plot(tri_uv[:,0], tri_uv[:,1], 'k-')
        plt.gca().set_aspect('equal','box')
        plt.title(f"Residual ℓ-ℓ̂ (a,b,c←{a_idx},{b_idx},{c_idx})")
        plt.xlabel('u'); plt.ylabel('v')
        plt.colorbar(tcf3); fig3.savefig(f"ma_resid_L{depth}_n{n_fit}.png", dpi=160)

        # quiver of tapered displacement (subsample for clarity)
        step = max(1, len(uv)//3000)
        plt.figure(figsize=(8,8))
        plt.plot(tri_uv[:,0], tri_uv[:,1], 'k-')
        plt.quiver(u[::step], v[::step], (s*du)[::step], (s*dv)[::step], angles='xy', scale_units='xy', scale=1)
        plt.gca().set_aspect('equal','box'); plt.title("Tapered, clipped displacement −∇φ")
        plt.savefig(f"ma_disp_L{depth}_n{n_fit}.png", dpi=160)

        plt.close('all')
    except Exception as e:
        print("Plotting skipped:", e)
