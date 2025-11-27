"""
Part of the H9 project
For a given layer, generate the canonical triangle grid
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
from hhg9.algorithms.distance import enu_planar_polygon_area
from hhg9.h9.polygon import tri_grid
import matplotlib as mpl

# Bernstein basis utilities (from your examples/src/bernstein.py)
try:
    from bernstein import bernstein_grads_uv, bernstein_terms_deg
except ImportError:
    # Fallback if module path differs; adjust as needed in your env
    from examples.experiments.bernstein import bernstein_grads_uv, bernstein_terms_deg

from math import comb

def load_psi(path):
    z = np.load(path, allow_pickle=True)
    terms = [tuple(t) for t in z["terms"]]
    return terms, z["c"], z["J"]

def degree_from_terms(terms):
    """
    Infer Bernstein degree n from the triplet list 'terms' of (i,j,k) with i+j+k=n.
    """
    n = max(int(i) + int(j) + int(k) for (i, j, k) in terms)
    # Optional consistency check: all terms should have the same total degree.
    assert all((int(i) + int(j) + int(k)) == n for (i, j, k) in terms), "Inconsistent Bernstein term degrees."
    return n

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

def grad_xy_from_uv(du, dv, J_uv_to_xy):
    # ∇_xy ψ = J^{-T} ∇_uv ψ
    Jinvt = np.linalg.inv(J_uv_to_xy).T
    g = np.stack([du, dv], axis=-1)   # (m,K,2) after @c → (m,2)
    return Jinvt @ g

def apply_prewarp(uv_pts, c, terms, n, alpha, grad_fn, step_mode="rms", eps=1e-9):
    """
    One UV step along the Bernstein potential gradient.
    We move in UV directly (simplex), then project back to the simplex:
      u>=eps, v>=eps, u+v<=1-eps.
    step_mode:
      - "rms": per-point normalize gradient to unit length; step size = alpha
      - "raw": use raw gradient (alpha is small ~1e-4..1e-3)
    Returns: (uv_new, grad_uv) with shapes (m,2), (m,2)
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

    return np.column_stack([u, v]), grad_uv


# --- Helpers to evaluate ψ and its Hessian (finite-difference in UV) ---

def clamp_to_simplex(u, v, eps=1e-9):
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
        collection = Poly3DCollection(polys, ec=(0,0,0,1.0), facecolors=col, alpha=1.0, linewidth=0.25)
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
    Triangular grid will be 9 triangles per octant at layer 0
    At each subsequent layer, the number of triangles will increase by 9 per triangle.
    So the number of triangles will be 8*9**(layer+1)
    """
    # --- Config ---
    depth = 5
    octant = 0
    RUN_FOREVER = True          # keep iterating until Ctrl-C
    CYCLE_SLEEP_S = 0.0         # optional pause between cycles
    SAVE_EVERY_CYCLES = 1       # checkpoint cadence
    ALPHA_GEOM_MIN = 2e-6       # base alpha range (UV units)
    ALPHA_GEOM_MAX = 2e-3
    ALPHA_STEPS    = 10
    FIT = 12
    DO_PLOT = False             # plotting disabled for long runs (keeps memory steady)
    # --- Speed knobs ---
    EVAL_SAMPLE_FRAC = 0.12     # evaluate α on a ~12% stratified subset; full eval for the winner
    EVAL_SAMPLE_SEED = 12345    # fixed seed so the subset is stable across cycles
    # ma_psi_L4_n12.npz

    # --- Setup Registrar / data ---
    rg = Registrar()  # Manage Domains & Projections
    data = get_data(rg, layer=depth, octant_id=octant)
    psi = Path(f'../experimental/ma_psi_L{depth}_n{FIT}.npz')  # MA fit (for J only if present)

    # --- Load MA potential terms/J if present (optional) ---
    try:
        terms, c_ma, J = load_psi(psi)
    except Exception:
        print(f"MA fit not found at {psi}.  Loading Bernstein terms from PFT.")
        # Fallback: load Bernstein terms from PFT if needed
        pft = Path(f'../experimental/ma_pft_L{depth}_n{FIT}.npz')
        terms, _, J = load_psi(pft)
    n_fit = degree_from_terms(terms)

    # --- Domain bridges ---
    b_oct = rg.domain('b_oct')
    uv_from_xy = b_oct.x2b[0, 0]
    xy_from_uv = b_oct.x2b[0, 1]

    # --- Build original triangle XY and baseline authalic against its own mean ---
    tr_pts_xy = data.coords.copy().reshape(-1, 2)
    c_pts = rg.project(data, ['b_oct', 'c_oct', 'c_ell'])
    areas_baseline = enu_planar_polygon_area(rg, c_pts, 3).astype(float)
    t_avg = float(np.mean(areas_baseline))
    ell_raw = (areas_baseline / t_avg) - 1.0  # fractional area error relative to baseline mean
    rmse_raw = float(np.sqrt(np.mean(ell_raw**2)))
    print(f"Authalic baseline: RMSE={rmse_raw:.6f}  MaxAbs={float(np.max(np.abs(ell_raw))):.6f}")
    print(f"  ℓ range: [{ell_raw.min():.3f}, {ell_raw.max():.3f}]  mean={ell_raw.mean():.3f}")

    # Build a deterministic evaluation subset of triangles (used to rank α quickly)
    # We use a simple stride selector so it is deterministic and cache-friendly.
    tri_count = int(len(areas_baseline))           # number of triangles
    tri_idx_sample = None
    if 0.0 < EVAL_SAMPLE_FRAC < 1.0:
        stride = max(1, int(round(1.0 / EVAL_SAMPLE_FRAC)))
        tri_idx_sample = np.arange(tri_count)[::stride]
        # Ensure poles/edges are not systematically skipped by offsetting one step each run start
        # (keeps determinism but avoids accidental aliasing with triangle ordering)
        if len(tri_idx_sample) > 0 and (tri_idx_sample[0] == 0):
            tri_idx_sample = tri_idx_sample
        print(f"[subset] evaluating α on {len(tri_idx_sample)}/{tri_count} triangles "
              f"({len(tri_idx_sample)/tri_count:.1%}); full eval for the cycle winner")

    # --- Geometry sanity: xy -> uv -> xy round-trip ---
    tr_uv = uv_from_xy(tr_pts_xy)
    XY_round = xy_from_uv(tr_uv.reshape(-1, 2)).reshape(-1, 2)
    dXY = XY_round - tr_pts_xy
    rt_rms = float(np.sqrt(np.mean(dXY ** 2)))
    rt_max = float(np.max(np.abs(dXY)))
    print(f"[roundtrip] xy→uv→xy  RMS={rt_rms:.3e}  max|Δ|={rt_max:.3e} (should be ~machine-eps to tiny)")

    # --- Prepare long-run outputs ---
    out_csv = Path(f"../experimental/prewarp_ls_cL{depth}_n{n_fit}.csv")
    ckpt_npz = Path(f"../experimental/prewarp_ls_cL{depth}_n{n_fit}.npz")

    # CSV header (append if exists)
    if not out_csv.exists():
        with out_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "timestamp","cycle","alpha_eff","rmse_rawMean","rmse_ownMean",
                "corr_ls","std_pred","std_ell","min_ell","max_ell","best_rmse_so_far"
            ])

    # --- Helper to evaluate RMSE against baseline mean or own mean, supports subsetting ---
    def eval_rmse_from_XY(XY_tri_flat, tri_idx=None):
        """
        Evaluate RMSE on all triangles (default) or a subset specified by tri_idx
        (indices over triangles). XY_tri_flat is shaped (-1,2) for 3-vertex triangles.
        """
        # Prepare XY for either the full set or a subset of triangles
        if tri_idx is not None:
            tri_xy = XY_tri_flat.reshape(-1, 3, 2)[tri_idx]   # (Ns,3,2)
            XY_use = tri_xy.reshape(-1, 2)                    # (Ns*3,2)
            # Components must be per-triangle (Ns,3) to match the subset
            comps_tri = data.components.reshape(-1, 3)[tri_idx]
        else:
            XY_use = XY_tri_flat                              # (Ntri*3,2)
            # Components must be per-triangle (Ntri,3) for the full set
            comps_tri = data.components.reshape(-1, 3)

        # Build Points with components sized per triangle, not per-vertex
        pts = Points(XY_use, b_oct, components=comps_tri)

        # Project and compute areas
        g = rg.project(pts, ['b_oct', 'c_oct', 'c_ell'])
        a = enu_planar_polygon_area(rg, g, 3).astype(float)

        # Against baseline mean (fixed); safe for subset or full
        score_vs_raw = (a / t_avg) - 1.0
        rmse_vs_raw = float(np.sqrt(np.mean(score_vs_raw**2)))
        # Against own mean (diagnostic)
        score_vs_own = (a / float(np.mean(a))) - 1.0
        rmse_vs_own = float(np.sqrt(np.mean(score_vs_own**2)))
        return rmse_vs_raw, rmse_vs_own, score_vs_raw, a

    # --- Start from current uv positions (centroids derived per-triangle) ---
    uv3 = tr_uv.reshape(-1, 2)            # (Nverts, 2)
    XY3 = tr_pts_xy.reshape(-1, 2)

    best_rmse_global = np.inf
    cycle = 0
    grad_fn = bernstein_grads_uv
    lam_ls = 1e-5

    try:
        while True:
            cycle += 1
            t_cycle0 = time.perf_counter()

            # Recompute centroids in UV at current positions
            uv_tri = uv3.reshape(-1, 3, 2)
            uv_cent = uv_tri.mean(axis=1)  # (Ntri,2)

            # LS ψ against current ℓ (baseline areas still used for t_avg; we compare RMSE vs baseline mean)
            B_cent = bernstein_vals_uv_batch(uv_cent, n_fit, terms)
            K = B_cent.shape[1]
            JTJ = B_cent.T @ B_cent
            JTJ.flat[::K+1] += lam_ls
            rhs = B_cent.T @ ell_raw
            try:
                c_ls = np.linalg.solve(JTJ, rhs)
            except np.linalg.LinAlgError:
                c_ls = np.linalg.lstsq(JTJ, rhs, rcond=None)[0]

            pred_ls = B_cent @ c_ls
            corr_ls = float(np.corrcoef(pred_ls, ell_raw)[0, 1])
            s_l = float(np.std(ell_raw))
            s_h = float(np.std(pred_ls))
            scale_alpha = s_l / max(s_h, 1e-12)
            sgn = 1.0 if corr_ls >= 0 else -1.0

            alphas_base = np.geomspace(ALPHA_GEOM_MIN, ALPHA_GEOM_MAX, ALPHA_STEPS)
            alphas = alphas_base * scale_alpha

            print(f"\n[cycle {cycle}] [LS ψ] λ={lam_ls:.1e} corr={corr_ls:+.3f} stdψ={s_h:.4f} stdℓ={s_l:.4f}")
            print(f"[alpha] sign={'+' if sgn>0 else '-'} scale={scale_alpha:.3e} range={alphas[0]:.2e}..{alphas[-1]:.2e}")

            best_this = (np.inf, None, None, None, None, None)  # rmse_raw, a_eff, rmse_own, score, XYp_full_or_subset, uv3p_best
            for alpha in alphas:
                a_eff = sgn * float(alpha)
                t0 = time.perf_counter()
                uv3p_trial, _ = apply_prewarp(uv3, c_ls, terms, n_fit, a_eff, grad_fn, step_mode="rms")
                XYp_trial = xy_from_uv(uv3p_trial.reshape(-1, 2)).reshape(-1, 3, 2).reshape(-1, 2)

                # Fast ranking on the subset
                rmse_vs_raw, rmse_vs_own, score_vs_raw, _ = eval_rmse_from_XY(XYp_trial, tri_idx=tri_idx_sample)

                dt = (time.perf_counter() - t0)*1e3
                print(f"  α={a_eff:.2e} [subset] → RMSE_raw={rmse_vs_raw:.6f} RMSE_own={rmse_vs_own:.6f} ({dt:.0f} ms)")
                if rmse_vs_raw < best_this[0]:
                    best_this = (rmse_vs_raw, a_eff, rmse_vs_own, score_vs_raw, XYp_trial, uv3p_trial)

            # Accept best in this cycle (do one full evaluation for the winner)
            rmse_sub, alpha_best, rmse_own_sub, _, XYp_trial_best, uv3p_best = best_this

            # Full projection + area only once for the selected α
            XY_best = xy_from_uv(uv3p_best.reshape(-1, 2)).reshape(-1, 3, 2).reshape(-1, 2)
            rmse_best, rmse_own_best, score_best, areas_best = eval_rmse_from_XY(XY_best, tri_idx=None)

            # Update UV to the accepted new state
            uv3 = uv3p_best

            best_rmse_global = min(best_rmse_global, rmse_best)

            # --- MA export bundle (for downstream MA solver) ---
            ma_npz = Path(f"../experimental/ma_input_L{depth}_n{n_fit}.npz")

            # Use the accepted geometry for this cycle
            uv_tri_for_ma = uv3.reshape(-1, 3, 2)
            uv_cent = uv_tri_for_ma.mean(axis=1)  # (Ntri,2)

            # Areas and ℓ at current XY (use XY_best which corresponds to alpha_best)
            pts_now = Points(XY_best, b_oct, components=data.components)
            # pts = Points(XY_tri_flat, b_oct, components=data.components)
            g_now = rg.project(pts_now, ['b_oct', 'c_oct', 'c_ell'])
            areas_now = enu_planar_polygon_area(rg, g_now, 3).astype(float)
            #
            # g_now = rg.project(pts_now, ['b_oct', 'g_gcd'])
            # areas_now = wgs84_area(rg, g_now, 3).astype(float)
            ell_now = (areas_now / t_avg) - 1.0  # relative to fixed baseline mean

            np.savez(
                ma_npz,
                terms=np.array(terms, dtype=object),
                degree=n_fit,
                uv_cent=uv_cent,
                ell=ell_now,
                J=J,
                c_init=c_ls,
                meta=dict(depth=depth, cycle=cycle, alpha=float(alpha_best), timestamp=float(time.time()))
            )
            print(f"[ma-export] wrote {ma_npz.name}: uv_cent={uv_cent.shape}, ell std={np.std(ell_now):.3g}")

            # Save checkpoint every SAVE_EVERY_CYCLES
            if (cycle % SAVE_EVERY_CYCLES) == 0:
                np.savez(ckpt_npz, terms=np.array(terms, dtype=object),
                         c=c_ls, alpha=alpha_best, depth=depth, uv=uv3)
                # Append CSV row
                # ... existing code ...
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
                # ... existing code ...
                # with out_csv.open("a", newline="") as f:
                #     w = csv.writer(f)
                #     w.writerow([
                #         datetime.isoformat(timespec="seconds")+"Z",
                #         cycle, f"{alpha_best:.6e}",
                #         f"{rmse_best:.6f}", f"{rmse_own_best:.6f}",
                #         f"{corr_ls:.6f}", f"{s_h:.6f}", f"{s_l:.6f}",
                #         f"{ell_raw.min():.6f}", f"{ell_raw.max():.6f}",
                #         f"{best_rmse_global:.6f}"
                #     ])
                # print(f"[saved] cycle={cycle} α*={alpha_best:.3e} rmse_raw={rmse_best:.6f} → {ckpt_npz.name}")

            t_cycle = time.perf_counter() - t_cycle0
            print(f"[cycle {cycle}] done in {t_cycle:.1f}s  (best_raw={rmse_best:.6f}, best_global={best_rmse_global:.6f})")

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
