"""
Part of the H9 project
For a given hex_layer, generate the canonical triangle grid
and display on the globe.
Last Tested 19 October 2025 √
"""
from pathlib import Path

import csv
import time
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt, colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from hhg9 import Registrar, Points
from hhg9.algorithms.distance import wgs84_area
from hhg9.h9.polygon import tri_grid
import matplotlib as mpl

# Bernstein basis utilities (from your examples/src/bernstein.py)
try:
    from bernstein import bernstein_grads_uv, bernstein_terms_deg
except ImportError:
    # Fallback if module path differs; adjust as needed in your env
    from examples.experiments.bernstein import bernstein_grads_uv, bernstein_terms_deg

from math import comb

def load_phi(path):
    """
    General loader for Bernstein potential fits.

    Supports NPZ files that may contain:
      - 'n_fit' or 'degree' (int)
      - 'terms' as (K,3) triplets (i,j,k) or (K,2) pairs (i,j) — we expand to (i,j,k)
      - coefficients under 'c' or 'coeff'
      - optional 'J' (ignored if absent)
    Returns: terms_triplets(list[(i,j,k)]), c(np.ndarray[K]), J(or None), n_fit(int)
    """
    z = np.load(path, allow_pickle=True)
    # degree
    n = None
    if 'n_fit' in z:
        n = int(z['n_fit'])
    elif 'degree' in z:
        n = int(z['degree'])
    # terms
    terms_arr = np.asarray(z['terms'])
    terms_list = [tuple(int(x) for x in row.tolist()) for row in terms_arr]
    # expand (i,j) → (i,j,k)
    if len(terms_list) and len(terms_list[0]) == 2:
        if n is None:
            raise ValueError("Cannot expand 2-term (i,j) entries without a known degree (n_fit/degree).")
        terms_list = [(i, j, int(n - i - j)) for (i, j) in terms_list]
    # coefficients
    if 'c' in z:
        c = np.asarray(z['c'], dtype=float)
    elif 'coeff' in z:
        c = np.asarray(z['coeff'], dtype=float)
    else:
        raise KeyError("NPZ does not contain coefficient array under keys 'c' or 'coeff'.")
    # optional J
    J = z['J'] if 'J' in z else None
    # degree fallback: infer from triplets if not provided
    if n is None:
        n = max(int(i) + int(j) + int(k) for (i, j, k) in terms_list)
    # basic validation
    assert all((int(i) + int(j) + int(k)) == n for (i, j, k) in terms_list), "Inconsistent Bernstein term degrees."
    return terms_list, c, J, n


def degree_from_terms(terms):
    """
    Infer Bernstein degree n from 'terms' which may be (i,j,k) or (i,j).
    If (i,j), n is inferred as max(i+j) over terms; callers must ensure this matches the intended degree.
    """
    if not terms:
        raise ValueError("Empty terms list.")
    arity = len(terms[0])
    if arity == 3:
        n = max(int(i) + int(j) + int(k) for (i, j, k) in terms)
        assert all((int(i) + int(j) + int(k)) == n for (i, j, k) in terms), "Inconsistent Bernstein term degrees."
        return n
    elif arity == 2:
        n = max(int(i) + int(j) for (i, j) in terms)
        return n
    else:
        raise ValueError(f"Unsupported term arity: {arity}")

def bernstein_grads_uv_batch(UV, n, terms, grad_fn):
    # grad_fn is your existing bernstein_grads_uv(u,v,n,terms)
    m = len(UV); K = len(terms)
    du = np.empty((m,K)); dv = np.empty((m,K))
    for i,(u,v) in enumerate(UV):
        gu, gv = grad_fn(u,v,n,terms)  # (K,), (K,)
        du[i] = gu; dv[i] = gv
    return du, dv

def bernstein_vals_uv(u, v, n, terms):
    """Bernstein basis values on the 2-simplex at (u,v) with w=1-u-v."""
    w = 1.0 - u - v
    out = np.empty(len(terms), dtype=float)
    for t, (i, j, k) in enumerate(terms):
        i = int(i); j = int(j); k = int(k)
        out[t] = comb(n, i) * comb(n - i, j) * (u ** i) * (v ** j) * (w ** k)
    return out
def bernstein_vals_uv_batch(UV, n, terms):
    m = len(UV); K = len(terms)
    B = np.empty((m, K), dtype=float)
    for idx, (u, v) in enumerate(UV):
        B[idx] = bernstein_vals_uv(u, v, n, terms)
    return B


# --- Helper: sample points along simplex edges (excluding corners) ---
def edge_samples(m=96, eps=5e-9):
    """Sample m points per edge on the simplex (excluding exact corners)."""
    t = np.linspace(eps, 1.0 - eps, m)
    # edges: v=0, u=0, and u+v=1
    e1 = np.stack([t, np.zeros_like(t)], axis=1)
    e2 = np.stack([np.zeros_like(t), t], axis=1)
    e3 = np.stack([t, 1.0 - t], axis=1)
    return np.vstack([e1, e2, e3])


# --- Helper: solve LS with constraint that constant Bernstein coefficient is zero ---
def solve_constrained_constant_zero(JTJ, rhs, terms, n_fit):
    K = JTJ.shape[0]
    # try to find (0,0,n)
    idx_const = None
    for idx, (i, j, k) in enumerate(terms):
        if int(i) == 0 and int(j) == 0 and int(k) == n_fit:
            idx_const = idx
            break

    if idx_const is None:
        # No constant term present → just solve unconstrained
        try:
            c = np.linalg.solve(JTJ, rhs)
        except np.linalg.LinAlgError:
            c = np.linalg.lstsq(JTJ, rhs, rcond=None)[0]
        return c, -1

    mask = np.ones(K, dtype=bool); mask[idx_const] = False
    JTJ_red = JTJ[mask][:, mask]
    rhs_red = rhs[mask]
    try:
        c_red = np.linalg.solve(JTJ_red, rhs_red)
    except np.linalg.LinAlgError:
        c_red = np.linalg.lstsq(JTJ_red, rhs_red, rcond=None)[0]
    c = np.zeros(K, dtype=float)
    c[mask] = c_red
    c[idx_const] = 0.0
    return c, idx_const


def grad_xy_from_uv(du, dv, J_uv_to_xy):
    # ∇_xy ψ = J^{-T} ∇_uv ψ
    Jinvt = np.linalg.inv(J_uv_to_xy).T
    g = np.stack([du, dv], axis=-1)   # (m,K,2) after @c → (m,2)
    return Jinvt @ g

def apply_prewarp(uv_pts, c, terms, n, alpha, grad_fn, step_mode="rms", eps=5e-9, vel=None, beta=0.0):
    """
    One UV step along the Bernstein potential gradient.
    We move in UV directly (simplex), then project back to the simplex:
      u>=eps, v>=eps, u+v<=1-eps.
    step_mode:
      - "rms": per-point normalize gradient to unit length; step size = alpha
      - "raw": use raw gradient (alpha is small ~1e-4..1e-3)
    Returns: (uv_new, grad_uv, vel_new) with shapes (m,2), (m,2), (m,2) or None

    Momentum:
      Pass a persistent `vel` array (m,2) and set beta∈[0,1). When beta>0, the step becomes:
         vel = beta*vel + step
         delta = vel
      Trial steps should use beta=0 (no momentum) to avoid mutating state.
    """
    m = len(uv_pts)
    K = len(terms)
    du = np.empty((m, K)); dv = np.empty((m, K))
    for i, (u, v) in enumerate(uv_pts):
        gu, gv = grad_fn(u, v, n, terms)  # (K,), (K,)
        du[i] = gu; dv[i] = gv
    # gradient of ψ in UV
    grad_uv = np.stack([du @ c, dv @ c], axis=1)  # (m,2)

    if step_mode == "rms":
        gnorm = np.linalg.norm(grad_uv, axis=1, keepdims=True)
        gnorm = np.maximum(gnorm, 1e-12)
        step = alpha * (grad_uv / gnorm)
    else:
        step = alpha * grad_uv

    # Momentum (optional): vel shape (m,2). If None or beta==0, behave as before.
    if vel is not None and beta > 0.0:
        if vel.shape != step.shape:
            raise ValueError("vel must have shape (m,2) matching uv_pts")
        vel = beta * vel + step
        step = vel

    u = uv_pts[:, 0] + step[:, 0]
    v = uv_pts[:, 1] + step[:, 1]

    # Project back to the simplex
    u = np.clip(u, eps, 1.0 - eps)
    v = np.clip(v, eps, 1.0 - eps)
    s = u + v
    over = s > (1.0 - eps)
    if np.any(over):
        scale = (1.0 - eps) / s[over]
        u[over] *= scale
        v[over] *= scale

    return np.column_stack([u, v]), grad_uv, (vel if vel is not None and beta > 0.0 else None)


# --- Helpers to evaluate ψ and its Hessian (finite-difference in UV) ---

def clamp_to_simplex(u, v, eps=5e-9):
    u = float(np.clip(u, eps, 1.0 - eps))
    v = float(np.clip(v, eps, 1.0 - eps))
    s = u + v
    if s > 1.0 - eps:
        scale = (1.0 - eps) / s
        u *= scale
        v *= scale
    return u, v


def psi_scalar(u, v, n, terms, c):
    """Evaluate the scalar Bernstein potential ψ(u,v) = B(u,v)·c on the simplex."""
    w = 1.0 - u - v
    if (u <= 0.0) or (v <= 0.0) or (w <= 0.0):
        u, v = clamp_to_simplex(u, v)
    # reuse bernstein_vals_uv
    return float(bernstein_vals_uv(u, v, n, terms) @ c)


def hessian_fd(u, v, n, terms, c, eps=1e-4):
    """Finite-difference Hessian of ψ at (u,v) in UV coordinates.
    Returns 2x2 matrix [[ψ_uu, ψ_uv],[ψ_uv, ψ_vv]].
    """
    e = eps
    u_p, v_0 = clamp_to_simplex(u + e, v)
    u_m, _    = clamp_to_simplex(u - e, v)
    u_0, v_p = clamp_to_simplex(u, v + e)
    _,   v_m = clamp_to_simplex(u, v - e)
    up_vp = clamp_to_simplex(u + e, v + e)
    up_vm = clamp_to_simplex(u + e, v - e)
    um_vp = clamp_to_simplex(u - e, v + e)
    um_vm = clamp_to_simplex(u - e, v - e)

    f_00   = psi_scalar(u,      v,      n, terms, c)
    f_up0  = psi_scalar(u_p,    v_0,    n, terms, c)
    f_um0  = psi_scalar(u_m,    v,      n, terms, c)
    f_0vp  = psi_scalar(u_0,    v_p,    n, terms, c)
    f_0vm  = psi_scalar(u,      v_m,    n, terms, c)
    f_upvp = psi_scalar(up_vp[0], up_vp[1], n, terms, c)
    f_upvm = psi_scalar(up_vm[0], up_vm[1], n, terms, c)
    f_umvp = psi_scalar(um_vp[0], um_vp[1], n, terms, c)
    f_umvm = psi_scalar(um_vm[0], um_vm[1], n, terms, c)

    f_uu = (f_up0 - 2.0 * f_00 + f_um0) / (e * e)
    f_vv = (f_0vp - 2.0 * f_00 + f_0vm) / (e * e)
    f_uv = (f_upvp - f_upvm - f_umvp + f_umvm) / (4.0 * e * e)
    return np.array([[f_uu, f_uv], [f_uv, f_vv]], dtype=float)


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


def snow_globe(arr: Points, poly_len: int = 6, pop=None):
    """Display a 3D point cloud using matplotlib"""
    mpl.rcParams['path.simplify'] = False
    fig = plt.figure(figsize=(15, 15), dpi=200, frameon=False)
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_proj_type('ortho')  # FOV = 0 deg
    ax.view_init(elev=30, azim=40)
    axis = mplot_ax_vector(ax)
    all_polys = arr.coords.reshape(-1, poly_len, 3)
    mask = cull_backface(all_polys, axis)
    front = all_polys[mask]
    if True:
        ax.set_xlim(-4e+5, 4e+5)  # -4e+5 fill the area with the map.
        ax.set_ylim(-4e+5, 4e+5)
        ax.set_zlim(-4e+5, 4e+5)
    polys = [p for p in front]
    if pop is not None:
        authalic_error = np.mean(np.abs(pop))
        pops = pop[mask]
        v_min = np.min(pops)
        v_max = np.max(pops)
        norm = colors.Normalize(vmin=v_min, vmax=v_max)
        cmap = plt.get_cmap('plasma')
        al = np.full((cmap.N,), 0.5)
        cmap.colors = np.column_stack([cmap.colors, al])
        col = cmap(norm(pops))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # no data needed
        plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
        collection = Poly3DCollection(polys, ec=(0,0,0,0.2), facecolors=col, alpha=1.0, linewidth=0.05)
        ax.add_collection(collection)
        ax.title.set_text(f'Authalic Error: {authalic_error:.2f}')

    else:
        collection = Poly3DCollection(polys, ec='black', alpha=1.0, linewidth=0.05)
        ax.add_collection(collection)

    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


def get_data(reg: Registrar, layer=3, octant_id=None):
    """Load up global sample data"""
    tg0 = tri_grid(layer, 0).reshape([-1, 2])  # triangle polygons.
    tg1 = tri_grid(layer, 1).reshape([-1, 2])  # triangle polygons.
    tgx = [tg0, tg1]
    b_oct = reg.domain('b_oct')
    if octant_id is None:
        repo = []
        for octant in b_oct.signs.keys():
            o_id = b_oct.sign_to_id[octant]
            mode = b_oct.oid_mo[o_id]
            repo.append(Points(tgx[mode].copy(), b_oct, components=octant))
        return Points.concat(repo)
    else:
        o_id = octant_id
        mode = b_oct.oid_mo[o_id]
        cmp = b_oct.signs_by_id[o_id]
        return Points(tgx[mode], b_oct, components=cmp)


if __name__ == '__main__':
    """
    Triangular grid will be 9 triangles per octant at hex_layer 0
    At each subsequent hex_layer, the number of triangles will increase by 9 per triangle.
    So the number of triangles will be 8*9**(hex_layer+1)
    """
    # --- Config ---
    depth = 5
    octant = 0
    RUN_FOREVER = True          # keep iterating until Ctrl-C
    CYCLE_SLEEP_S = 0.0         # optional pause between cycles
    SAVE_EVERY_CYCLES = 1       # checkpoint cadence
    ALPHA_GEOM_MIN = 2e-6       # base alpha range (UV units)
    ALPHA_GEOM_MAX = 2e-3
    ALPHA_STEPS    = 8
    FIT = 16
    DO_PLOT = False             # plotting disabled for long runs (keeps memory steady)
    # ma_psi_L4_n12.npz

    # --- Setup Registrar / data ---
    rg = Registrar()  # Manage Domains & Projections
    data = get_data(rg, layer=depth, octant_id=octant)
    # --- Load φ (Bernstein) potential fit ---
    phi_npz = Path(f"../experimental/phi_fit_L{depth}_n{FIT}.npz")
    alt_npz = Path(f"../experimental/ma_psi_L{depth}_n{FIT}.npz")  # legacy fallback name
    if phi_npz.exists():
        terms, c_ls, J, n_fit = load_phi(phi_npz)
        c_phi_prefit = c_ls.copy()
        print(f"[φ-fit] loaded {phi_npz.name}: n={n_fit}, K={len(terms)}")
    elif alt_npz.exists():
        # Legacy MA file; try to read with the same loader
        terms, c_ls, J, n_fit = load_phi(alt_npz)
        c_phi_prefit = c_ls.copy()
        print(f"[φ-fit] loaded legacy {alt_npz.name}: n={n_fit}, K={len(terms)}")
    else:
        raise FileNotFoundError(f"Could not find φ fit at {phi_npz} (or legacy {alt_npz}).")
    # retain the file's coefficients as a stable fallback field
    c_phi_prefit = c_phi_prefit.copy()

    # --- Domain bridges ---
    b_oct = rg.domain('b_oct')
    uv_from_xy = b_oct.x2b[0, 0]
    xy_from_uv = b_oct.x2b[0, 1]

    # --- Build original triangle XY and baseline authalic against its own mean ---
    tr_pts_xy = data.coords.copy().reshape(-1, 2)
    g_pts = rg.project(data, ['b_oct', 'g_gcd'])
    areas_baseline = wgs84_area(rg, g_pts, 3).astype(float)
    t_avg = float(np.mean(areas_baseline))
    ell_raw = (areas_baseline / t_avg) - 1.0  # fractional area error relative to baseline mean
    rmse_raw = float(np.sqrt(np.mean(ell_raw**2)))
    print(f"Authalic baseline: RMSE={rmse_raw:.6f}  MaxAbs={float(np.max(np.abs(ell_raw))):.6f}")
    print(f"  ℓ range: [{ell_raw.min():.3f}, {ell_raw.max():.3f}]  mean={ell_raw.mean():.3f}")

    # --- Geometry sanity: xy -> uv -> xy round-trip ---
    tr_uv = uv_from_xy(tr_pts_xy)
    XY_round = xy_from_uv(tr_uv.reshape(-1, 2)).reshape(-1, 2)
    dXY = XY_round - tr_pts_xy
    rt_rms = float(np.sqrt(np.mean(dXY ** 2)))
    rt_max = float(np.max(np.abs(dXY)))
    print(f"[roundtrip] xy→uv→xy  RMS={rt_rms:.3e}  max|Δ|={rt_max:.3e} (should be ~machine-eps to tiny)")

    # --- Prepare long-run outputs ---
    out_csv = Path(f"../experimental/prewarp_ls_L{depth}_n{n_fit}.csv")
    ckpt_npz = Path(f"../experimental/prewarp_ls_L{depth}_n{n_fit}.npz")

    # CSV header (append if exists)
    if not out_csv.exists():
        with out_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "timestamp","cycle","alpha_eff","rmse_rawMean","rmse_ownMean",
                "corr_ls","std_pred","std_ell","min_ell","max_ell","best_rmse_so_far"
            ])

    # --- Helper to evaluate RMSE against baseline mean or own mean ---
    def eval_rmse_from_XY(XY_tri_flat):
        pts = Points(XY_tri_flat, b_oct, components=data.components)
        g = rg.project(pts, ['b_oct', 'g_gcd'])
        a = wgs84_area(rg, g, 3).astype(float)
        # Against baseline mean (fixed)
        score_vs_raw = (a / t_avg) - 1.0
        rmse_vs_raw = float(np.sqrt(np.mean(score_vs_raw**2)))
        # Against own mean (purely diagnostic)
        score_vs_own = (a / float(np.mean(a))) - 1.0
        rmse_vs_own = float(np.sqrt(np.mean(score_vs_own**2)))
        return rmse_vs_raw, rmse_vs_own, score_vs_raw, a

    # --- Start from current uv positions (centroids derived per-triangle) ---
    uv3 = tr_uv.reshape(-1, 2)            # (Nverts, 2)
    XY3 = tr_pts_xy.reshape(-1, 2)

    # Momentum state (UV space)
    vel_uv = np.zeros_like(uv3)  # (m,2)
    beta_mom = 0.0              # momentum OFF for accepted steps; searches already use beta=0.0

    # Adaptive alpha range state
    alpha_cur_min = ALPHA_GEOM_MIN
    alpha_cur_max = ALPHA_GEOM_MAX
    edge_low_hits = 0
    edge_high_hits = 0

    best_rmse_global = np.inf
    cycle = 0
    grad_fn = bernstein_grads_uv
    # lam_ls now set dynamically per cycle (see below)
    # Track last accepted RMSE (start from baseline)
    rmse_prev = rmse_raw

    try:
        while True:
            cycle += 1
            t_cycle0 = time.perf_counter()

            # Recompute centroids in UV at current positions
            uv_tri = uv3.reshape(-1, 3, 2)
            uv_cent = uv_tri.mean(axis=1)  # (Ntri,2)
            # --- Quadrature samples per triangle: centroid + 3 edge midpoints ---
            uv_e01 = 0.5 * (uv_tri[:, 0, :] + uv_tri[:, 1, :])
            uv_e12 = 0.5 * (uv_tri[:, 1, :] + uv_tri[:, 2, :])
            uv_e20 = 0.5 * (uv_tri[:, 2, :] + uv_tri[:, 0, :])
            uv_quad = np.vstack([uv_cent, uv_e01, uv_e12, uv_e20])  # (4*Ntri, 2)

            # Recompute current XY from uv3
            XY_now = xy_from_uv(uv3.reshape(-1, 2)).reshape(-1, 2)

            # Build per-cycle authalic fluctuations around the *cycle* mean (centering)
            pts_now = Points(XY_now, b_oct, components=data.components)
            g_now = rg.project(pts_now, ['b_oct', 'g_gcd'])
            areas_now = wgs84_area(rg, g_now, 3).astype(float)
            ell_cyc = (areas_now / float(np.mean(areas_now))) - 1.0  # zero-mean target for fitting

            # Bernstein design at triangle centroids (for diagnostics) and quadrature (for LS fit)
            B_cent = bernstein_vals_uv_batch(uv_cent, n_fit, terms)   # (Ntri,K) for diagnostics/corr
            B_quad = bernstein_vals_uv_batch(uv_quad, n_fit, terms)   # (4*Ntri,K) for LS fit
            K = B_cent.shape[1]
            # Ridge scales with basis size (reference K≈91 @ n=12)
            lam_ls = 1e-5 * (K / 91.0) ** 2

            ell_quad = np.repeat(ell_cyc, 4, axis=0)                  # (4*Ntri,) target at quadrature samples

            # --- Initial centered solve with constant term fixed to zero ---
            JTJ = B_quad.T @ B_quad
            JTJ.flat[::K+1] += lam_ls
            rhs = B_quad.T @ ell_quad
            c_ls, idx_const = solve_constrained_constant_zero(JTJ, rhs, terms, n_fit)

            # --- One-pass Huber IRLS to suppress outliers (e.g., centre spike) ---
            pred0 = B_quad @ c_ls
            res0 = ell_quad - pred0
            tau = 3.0 * np.median(np.abs(res0)) + 1e-12
            w = np.clip(tau / np.maximum(tau, np.abs(res0)), 0.2, 1.0)  # weights in [0.2,1]
            WB = (w[:, None] * B_quad)

            JTJ = WB.T @ B_quad
            JTJ.flat[::K+1] += lam_ls
            rhs = WB.T @ ell_quad

            # --- Soft boundary pinning ψ≈0 along edges to avoid interior "balloon" ---
            E = edge_samples(96)  # denser edge pin for higher degree
            BE = bernstein_vals_uv_batch(E, n_fit, terms)
            lambda_boundary = 1e-4 * (K / 91.0)  # scale with basis size
            JTJ += lambda_boundary * (BE.T @ BE)
            # rhs unaffected (targets are 0 on the boundary)

            # Conditioning diagnostic before final solve
            try:
                cond_jtj = float(np.linalg.cond(JTJ))
            except Exception:
                cond_jtj = float('inf')

            # Final constrained solve (constant term fixed to zero), with optional stabilization
            if cond_jtj > 1e8:
                lam_boost = 10.0
                lambda_boundary *= 3.0
                # rebuild normal equations with boosted ridge and boundary pin
                JTJ = (WB.T @ B_quad)
                JTJ.flat[::K+1] += lam_boost * lam_ls
                JTJ += lambda_boundary * (BE.T @ BE)
                rhs = (WB.T @ ell_quad)
                c_ls, _ = solve_constrained_constant_zero(JTJ, rhs, terms, n_fit)
                print(f"[fit] K={K} cond(JTJ)≈{cond_jtj:.2e}  ||c||={np.linalg.norm(c_ls):.3e}  [boost ridge x{lam_boost}, edge pin x3]")
            else:
                c_ls, _ = solve_constrained_constant_zero(JTJ, rhs, terms, n_fit)
                print(f"[fit] K={K} cond(JTJ)≈{cond_jtj:.2e}  ||c||={np.linalg.norm(c_ls):.3e}")

            # Diagnostics and step scaling (use centroids for correlation/sign, as before)
            pred_ls = B_cent @ c_ls
            # Correlate against the centered per-cycle target to choose a consistent step sign
            corr_ls = float(np.corrcoef(pred_ls, ell_cyc)[0, 1])
            s_l = float(np.std(ell_raw))
            s_h = float(np.std(pred_ls))
            scale_alpha = s_l / max(s_h, 1e-12)
            # cap overly aggressive scaling
            scale_alpha = min(scale_alpha, 0.8)
            sgn = 1.0 if corr_ls >= 0 else -1.0

            # choose coefficient field for stepping
            use_c = c_ls
            if abs(corr_ls) < 0.6:
                print("[step] weak corr → using pre-fit φ coefficients for stepping")
                use_c = c_phi_prefit

            # if predicted field is much smaller than target fluctuations, shrink the alpha window
            ratio = s_h / max(s_l, 1e-12)
            if ratio < 0.4:
                alpha_cur_min = max(ALPHA_GEOM_MIN, 0.25 * ALPHA_GEOM_MIN)
                alpha_cur_max = min(ALPHA_GEOM_MAX, 0.25 * ALPHA_GEOM_MAX)
                print(f"[alpha] stdψ tiny (ratio={ratio:.2f}) → shrink range to {alpha_cur_min:.2e}..{alpha_cur_max:.2e}")

            # Adaptive alpha ladder (magnitude only); sign applied later via sgn
            alphas_base = np.geomspace(alpha_cur_min, alpha_cur_max, ALPHA_STEPS)
            alphas = alphas_base * scale_alpha

            print(f"\n[cycle {cycle}] [LS ψ] λ={lam_ls:.1e} corr={corr_ls:+.3f} stdψ={s_h:.4f} stdℓ={s_l:.4f}")
            print(f"[alpha] sign={'+' if sgn>0 else '-'} scale={scale_alpha:.3e} range={alphas[0]:.2e}..{alphas[-1]:.2e}")

            best_this = (np.inf, None, None, None, None)  # rmse_raw, a_eff, rmse_own, score, XYp
            no_improve = 0
            flipped = False
            for idx, alpha in enumerate(alphas):
                a_eff = sgn * float(alpha)
                t0 = time.perf_counter()
                uv3p, _, _ = apply_prewarp(uv3, use_c, terms, n_fit, a_eff, grad_fn, step_mode="rms", beta=0.0)
                XYp = xy_from_uv(uv3p.reshape(-1, 2)).reshape(-1, 3, 2).reshape(-1, 2)
                rmse_vs_raw, rmse_vs_own, score_vs_raw, areas_p = eval_rmse_from_XY(XYp)
                dt = (time.perf_counter() - t0)*1e3
                print(f"  α={a_eff:.2e} → RMSE_raw={rmse_vs_raw:.6f} RMSE_own={rmse_vs_own:.6f} ({dt:.0f} ms)")

                if rmse_vs_raw + 1e-7 < best_this[0]:
                    best_this = (rmse_vs_raw, a_eff, rmse_vs_own, score_vs_raw, XYp)
                    no_improve = 0
                else:
                    no_improve += 1
                    # If the first 3 probes fail to improve, flip the sign once
                    if idx == 2 and best_this[0] == np.inf and not flipped:
                        sgn *= -1.0
                        flipped = True
                        print("[alpha] no improvement in first 3 → flipping sign for remaining probes")
                        continue
                    if idx >= 4 and no_improve >= 3:
                        print("  [early-stop] alpha ladder shows no improvement — breaking sweep")
                        break

            # Accept best in this cycle
            rmse_best, alpha_best, rmse_own_best, score_best, XY_best = best_this

            # --- Local bracket + refine around alpha_best (no momentum during search) ---
            if np.isfinite(rmse_best) and alpha_best is not None:
                # Probe scaled neighbors around winner
                scales = np.array([0.5, 0.75, 1.0, 1.33, 1.75, 2.25], dtype=float)
                for s in scales:
                    a_try = s * alpha_best
                    # Clip magnitude to current adaptive bounds (keep sign)
                    sign = 1.0 if a_try >= 0 else -1.0
                    mag = np.clip(abs(a_try), alpha_cur_min, alpha_cur_max)
                    a_try = sign * mag
                    t0r = time.perf_counter()
                    uv_try, _, _ = apply_prewarp(uv3, use_c, terms, n_fit, a_try, grad_fn, step_mode="rms", beta=0.0)
                    XYt = xy_from_uv(uv_try.reshape(-1, 2)).reshape(-1, 3, 2).reshape(-1, 2)
                    r_raw, r_own, s_raw, _ = eval_rmse_from_XY(XYt)
                    dtr = (time.perf_counter() - t0r) * 1e3
                    print(f"  [refine] α={a_try:.2e} → RMSE_raw={r_raw:.6f} RMSE_own={r_own:.6f} ({dtr:.0f} ms)")
                    if r_raw < rmse_best:
                        rmse_best, alpha_best, rmse_own_best, score_best, XY_best = r_raw, a_try, r_own, s_raw, XYt

            # --- Two-sided micro line search around alpha_best ---
            if np.isfinite(rmse_best) and (alpha_best is not None):
                micro = np.array([-1.5, -1.0, -0.5, 0.5, 1.0, 1.5], dtype=float)
                for s in micro:
                    a_try = s * abs(alpha_best) * np.sign(alpha_best) if s > 0 else s * abs(alpha_best)
                    # clip to current bounds
                    sign = 1.0 if a_try >= 0 else -1.0
                    mag = np.clip(abs(a_try), alpha_cur_min, alpha_cur_max)
                    a_try = sign * mag
                    t0m = time.perf_counter()
                    uv_try, _, _ = apply_prewarp(uv3, use_c, terms, n_fit, a_try, grad_fn, step_mode="rms", beta=0.0)
                    XYt = xy_from_uv(uv_try.reshape(-1, 2)).reshape(-1, 3, 2).reshape(-1, 2)
                    r_raw, r_own, s_raw, _ = eval_rmse_from_XY(XYt)
                    dtm = (time.perf_counter() - t0m) * 1e3
                    print(f"  [micro] α={a_try:.2e} → RMSE_raw={r_raw:.6f} RMSE_own={r_own:.6f} ({dtm:.0f} ms)")
                    if r_raw < rmse_best:
                        rmse_best, alpha_best, rmse_own_best, score_best, XY_best = r_raw, a_try, r_own, s_raw, XYt

            # Trust-region acceptance test: only accept if we beat last accepted RMSE
            if np.isfinite(rmse_best) and (rmse_best < rmse_prev - 1e-7):
                # Accept and advance momentum
                uv3, _, vel_uv = apply_prewarp(uv3, use_c, terms, n_fit, alpha_best, grad_fn,
                                               step_mode="rms", vel=vel_uv, beta=beta_mom)
                rmse_prev = rmse_best
                best_rmse_global = min(best_rmse_global, rmse_best)
            else:
                # Reject: keep state, reset momentum, and shrink alpha range
                vel_uv[:] = 0.0
                alpha_cur_max = max(ALPHA_GEOM_MIN*2, alpha_cur_max * 0.5)
                alpha_cur_min = max(ALPHA_GEOM_MIN, min(alpha_cur_min * 0.5, alpha_cur_max / 10.0))
                print(f"[reject] step not improving (best={rmse_best:.6f} ≥ prev={rmse_prev:.6f}); "
                      f"shrink alpha range to {alpha_cur_min:.2e}..{alpha_cur_max:.2e}")

            # --- Adaptive alpha range recenter/resize if we keep hitting bounds ---
            if alpha_best is not None and np.isfinite(alpha_best):
                mag = abs(alpha_best)
                # Detect hits
                if mag <= 1.02 * alpha_cur_min:
                    edge_low_hits += 1
                else:
                    edge_low_hits = 0
                if mag >= 0.98 * alpha_cur_max:
                    edge_high_hits += 1
                else:
                    edge_high_hits = 0

                # Recenter around current best (gentle)
                target_min = max(ALPHA_GEOM_MIN, mag / 3.0)
                target_max = min(ALPHA_GEOM_MAX, mag * 3.0)
                alpha_cur_min = max(ALPHA_GEOM_MIN, min(alpha_cur_min * 0.8 + target_min * 0.2, target_max))
                alpha_cur_max = min(ALPHA_GEOM_MAX, max(alpha_cur_max * 0.8 + target_max * 0.2, alpha_cur_min * 1.5))

                # If stuck at edges twice, shrink/expand aggressively
                if edge_low_hits >= 2:
                    span = max(target_max / max(target_min, 1e-12), 10.0)
                    alpha_cur_min = max(ALPHA_GEOM_MIN, mag / max(3.0, np.sqrt(span)))
                    alpha_cur_max = min(ALPHA_GEOM_MAX, mag * max(3.0, np.sqrt(span)))
                    print(f"[alpha] adjust: near lower edge → new range {alpha_cur_min:.2e}..{alpha_cur_max:.2e}")
                    edge_low_hits = 0
                if edge_high_hits >= 2:
                    alpha_cur_min = max(ALPHA_GEOM_MIN, mag / 3.0)
                    alpha_cur_max = min(ALPHA_GEOM_MAX, mag * 6.0)
                    print(f"[alpha] adjust: near upper edge → new range {alpha_cur_min:.2e}..{alpha_cur_max:.2e}")
                    edge_high_hits = 0

            # --- MA export bundle (for downstream MA solver) ---
            ma_npz = Path(f"../experimental/ma_input_L{depth}_n{n_fit}.npz")

            # Use the accepted geometry for this cycle
            uv_tri_for_ma = uv3.reshape(-1, 3, 2)
            uv_cent = uv_tri_for_ma.mean(axis=1)  # (Ntri,2)

            # Areas and ℓ at current XY (use XY_best which corresponds to alpha_best)
            pts_now = Points(XY_best, b_oct, components=data.components)
            g_now = rg.project(pts_now, ['b_oct', 'g_gcd'])
            areas_now = wgs84_area(rg, g_now, 3).astype(float)
            ell_now = (areas_now / t_avg) - 1.0  # relative to fixed baseline mean

            np.savez(
                ma_npz,
                terms=np.array(terms, dtype=object),
                degree=n_fit,
                uv_cent=uv_cent,
                ell=ell_now,
                J=(J if J is not None else np.array([])),
                c_init=c_ls,
                meta=dict(depth=depth, cycle=cycle, alpha=float(alpha_best), timestamp=float(time.time()))
            )
            print(f"[ma-export] wrote {ma_npz.name}: uv_cent={uv_cent.shape}, ell std={np.std(ell_now):.3g}")

            # Save checkpoint every SAVE_EVERY_CYCLES
            if (cycle % SAVE_EVERY_CYCLES) == 0:
                np.savez(ckpt_npz, terms=np.array(terms, dtype=object),
                         c=c_ls, alpha=alpha_best, depth=depth, uv=uv3)
                # Append CSV row
                with out_csv.open("a", newline="") as f:
                    w = csv.writer(f)
                    w.writerow([
                        datetime.now().isoformat(timespec="seconds") + "Z",
                        cycle, f"{alpha_best:.6e}",
                        f"{rmse_best:.6f}", f"{rmse_own_best:.6f}",
                        f"{corr_ls:.6f}", f"{s_h:.6f}", f"{s_l:.6f}",
                        f"{ell_raw.min():.6f}", f"{ell_raw.max():.6f}",
                        f"{best_rmse_global:.6f}"
                    ])
                print(f"[saved] cycle={cycle} α*={alpha_best:.3e} rmse_raw={rmse_best:.6f} → {ckpt_npz.name}")

            t_cycle = time.perf_counter() - t_cycle0
            print(f"[cycle {cycle}] done in {t_cycle:.1f}s  (best_try={rmse_best:.6f}, accepted={rmse_prev:.6f}, best_global={best_rmse_global:.6f})")

            if not RUN_FOREVER:
                break
            if CYCLE_SLEEP_S > 0:
                time.sleep(CYCLE_SLEEP_S)

    except KeyboardInterrupt:
        print("\n[halted] KeyboardInterrupt — saving final checkpoint…")
        try:
            np.savez(ckpt_npz, terms=np.array(terms, dtype=object),
                     c=c_ls, alpha=alpha_best, depth=depth, uv=uv3)
            print(f"[saved] final → {ckpt_npz}")
        except Exception as e:
            print(f"[warn] could not save final checkpoint: {e}")

    # Optional quick-look plot at the end (disabled in long runs)
    if DO_PLOT:
        data_best = Points(XY_best, b_oct, components=data.components)
        c_pts_post = rg.project(data_best, ['b_oct', 'c_ell'])
        try:
            snow_globe(c_pts_post, 3, score_best)
        except Exception:
            pass
