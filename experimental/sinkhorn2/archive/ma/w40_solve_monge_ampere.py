"""
MA Solver (Bernstein, discrete collocation)
------------------------------------------
This script assumes you have already run `ma_precondition.py` to produce
`precond_O{octant}_L{depth}_n{deg}.npy`, which stores per-centroid samples in the
(u,v) simplex with an associated log-density field ℓ = log(A/Ā).

Workflow:
1) Load preconditioned (u,v) and ℓ from the file, where ℓ = log(A/Ā) as in w10_grid.
2) Solve for a Bernstein potential ψ so that log det(Hψ_xy) ≈ -ℓ (up to a constant), i.e. the MA-implied log‑density l̂ ≈ -ℓ. In uv-coordinates, log det(Hψ_uv) ≈ (-ℓ) + 2 log|det J|. We center the RHS to remove the constant net_mode ambiguity.
3) Save ψ coefficients and plot Raw ℓ, Fitted ℓ̂, and Residual with shared color scales.

Key conventions:
- (u,v) are barycentric simplex coordinates (u=b, v=c).
- xy geometry is the √2 equilateral; metric handled by J_uv_to_xy and Ginv.
- Output `ma_psi_*.npz` contains (terms, c, depth, parity, J, Ginv, detJ).
"""

import numpy as np
from math import comb
from pathlib import Path
from collections import defaultdict
import yaml
from hhg9 import Registrar
from w20_bernstein_fit_phi import bernstein_terms_deg

# Finite-difference step for Laplacian in (u,v); will be overridden by NPZ if present
LAPLACIAN_H = 1e-4  # default; may be overridden from NPZ


# === Monge–Ampère helpers for right-hand side, logdet, and least-squares init ===
def build_rhs(ell_rhs, det_j, use_minus_ell=True, center=True):
    """Construct MA right-hand side and log|detJ|^2.

    Given authalic log-density ell_rhs = log(A/Ā) and a Jacobian determinant det_j
    for the uv→xy mapping, return:
    - rhs: target field for log det(Hψ_uv), shape (m,)
    - log_detJ2: scalar 2*log|det_j|
    - rhs_mean: mean of the uncentered rhs (for diagnostics).

    Parameters
    ----------
    ell_rhs : array_like
        Authalic log-density field ℓ = log(A/Ā) (or a variant, e.g. residual).
    det_j : float
        Scalar Jacobian determinant |det J| for uv→xy.
    use_minus_ell : bool, optional
        If True (default), use rhs = -ℓ + 2 log|detJ|.
        If False, use rhs = +ℓ + 2 log|detJ|. This is useful for experiments
        with alternative sign conventions.
    center : bool, optional
        If True (default), subtract the mean of rhs so that the RHS is centered
        and the constant net_mode is removed. If False, leave rhs uncentered.
    """
    ell_rhs = np.asarray(ell_rhs, dtype=float)
    log_detJ2 = 2.0 * np.log(abs(det_j))
    sign = -1.0 if use_minus_ell else 1.0
    rhs = sign * ell_rhs + log_detJ2
    rhs_mean = float(rhs.mean())
    if center:
        rhs = rhs - rhs_mean
    return rhs, log_detJ2, rhs_mean


def eval_logdet_uv(c, Huu, Hvv, Huv, s_floor, det_floor):
    """Evaluate log det(Hψ_uv) and SPD components for coefficients c.

    Returns (log_det_uv, det_uv, A, B, C) with shapes (m,).
    """
    c = np.asarray(c, dtype=float)
    A = np.maximum(Huu @ c, s_floor)
    B = np.maximum(Hvv @ c, s_floor)
    C = Huv @ c
    det_uv = np.maximum(A * B - C * C, det_floor)
    log_det_uv = np.log(det_uv)
    return log_det_uv, det_uv, A, B, C


def ls_init_for(rhs_target, Huu, Hvv, Huv, lam):
    """Least-squares initialization for a given target RHS (log-det(H_uv)).

    Uses column-normalized ridge LS to avoid enormous coefficients and gently
    prefers C≈0 so the initial Hessian is close to isotropic SPD.
    """
    rhs_target = np.asarray(rhs_target, dtype=float)
    s_target = np.exp(0.5 * rhs_target)  # want A≈B≈s, C≈0 ⇒ det≈s^2
    m, k = Huu.shape
    # Stack operators for A, B, C
    H_stack = np.vstack([Huu, Hvv, Huv])          # (3m, K)
    t_stack = np.concatenate([s_target, s_target, np.zeros_like(s_target)])
    # Column-normalize to reduce condition number
    coln = np.linalg.norm(H_stack, axis=0) + 1e-12
    Hn = H_stack / coln
    # Lightly down-weight the C rows so they don't dominate
    wA = 1.0
    wB = 1.0
    wC = 0.5
    W = np.concatenate([np.full(m, wA), np.full(m, wB), np.full(m, wC)])
    Hnw = Hn * W[:, None]
    tnw = t_stack * W
    # Ridge solve with safe lambda
    lam0 = max(float(lam), 1e-3)
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


def check_d3_symmetry(terms, c, tol=1e-6, verbose=True):
    """
    Group Bernstein terms (i,j,k) by their permutation orbits and
    check how far the coefficients deviate from full D3 symmetry.

    :param terms: array-like, shape (K,3), integer indices (i,j,k)
    :param c: array-like, shape (K,), coefficients
    :param tol: tolerance for "almost symmetric"
    :return: dict with summary stats
    """
    terms = np.asarray(terms, dtype=int)
    c = np.asarray(c, dtype=float)

    groups = defaultdict(list)
    for idx, (i, j, k) in enumerate(terms):
        key = tuple(sorted((i, j, k)))
        groups[key].append(idx)

    max_dev = 0.0
    worst_key = None
    worst_span = None

    for key, idxs in groups.items():
        vals = c[idxs]
        span = vals.max() - vals.min()
        if span > max_dev:
            max_dev = span
            worst_key = key
            worst_span = (vals.min(), vals.max())

    if verbose:
        print(f"[symmetry] groups={len(groups)}")
        print(f"[symmetry] max coeff span across an orbit = {max_dev:.3e}")
        if worst_key is not None:
            vmin, vmax = worst_span
            print(f"           worst orbit key={worst_key}, "
                  f"coeffs in [{vmin:.6g}, {vmax:.6g}]")

        if max_dev <= tol:
            print(f"[symmetry] OK: D3 symmetry holds within tol={tol:g}")
        else:
            print(f"[symmetry] WARNING: noticeable asymmetry (> {tol:g})")

    return {
        "n_groups": len(groups),
        "max_span": max_dev,
        "worst_key": worst_key,
        "worst_span": worst_span,
    }


def symmetrise_d3(terms, c):
    """
    Return a new coefficient array where each permutation orbit
    has been averaged to enforce D3 symmetry.

    :param terms: (K,3) integer indices
    :param c: (K,) float coefficients
    :return: c_sym (K,)
    """
    terms = np.asarray(terms, dtype=int)
    c = np.asarray(c, dtype=float)
    c_sym = c.copy()

    groups = defaultdict(list)
    for idx, (i, j, k) in enumerate(terms):
        key = tuple(sorted((i, j, k)))
        groups[key].append(idx)

    for idxs in groups.values():
        vals = c[idxs]
        mean_val = vals.mean()
        c_sym[idxs] = mean_val
    return c_sym


def load_ma_config(name="default", path="ma_config.yaml"):
    """Load a named Monge–Ampère solver configuration from a YAML file.

    Supports simple inheritance via an ``overloads`` field in the YAML, e.g.::

        base_soft:
          lam: 1.0e-6
          iters: 40
          damping: 0.5

        strong_edge:
          overloads: [base_soft]
          edge_band: 0.05
          edge_gain: 2.0

        l5_experiment:
          overloads: [strong_edge, weak_tether]
          iters: 80

    Semantics:
    - The first entry in ``overloads`` is treated as the primary base config.
    - Subsequent entries override that base where they define keys.
    - The named config itself is then applied last, overriding all parents.
    - ``overloads`` may be a single string or a list of strings.
    - Cycles in overload chains raise a ValueError.
    """
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"MA config file '{cfg_path}' not found. "
            f"Create it with a '{name}' section as documented in load_ma_config()."
        )

    with cfg_path.open("r") as f:
        data = yaml.safe_load(f) or {}

    if name not in data:
        raise KeyError(
            f"MA config '{name}' not found in '{cfg_path}'. "
            f"Available configs: {', '.join(sorted(data.keys())) or '(none)'}"
        )

    if not isinstance(data[name], dict):
        raise TypeError(f"Config '{name}' in '{cfg_path}' must be a mapping/dict.")

    def _resolve(one_name: str, stack=None):
        if stack is None:
            stack = []
        if one_name in stack:
            chain = " -> ".join(stack + [one_name])
            raise ValueError(f"Cyclic 'overloads' chain detected in ma_config.yaml: {chain}")
        if one_name not in data:
            raise KeyError(f"Config '{one_name}' referenced in 'overloads' but not defined in '{cfg_path}'.")
        raw = data[one_name]
        if not isinstance(raw, dict):
            raise TypeError(f"Config '{one_name}' in '{cfg_path}' must be a mapping/dict.")

        # Start from an empty dict and merge parents (if any) first
        merged = {}
        parents = raw.get("overloads", [])
        if isinstance(parents, str):
            parents = [parents]
        for parent_name in parents:
            parent_cfg = _resolve(parent_name, stack + [one_name])
            merged.update(parent_cfg)

        # Then overlay this config's own keys (excluding 'overloads')
        for k, v in raw.items():
            if k == "overloads":
                continue
            merged[k] = v
        return merged

    cfg = _resolve(name)
    return cfg


def bernstein_hess_uv(u, v, n, terms):
    """Analytic Hessian components of the Bernstein basis in (u,v).
    Returns three arrays (huu, huv, hvv) each of shape (K,), where
    f_uu = sum c_k * huu[k], etc. Uses a=1-u-v, b=u, c=v and
    ∂/∂u = ∂/∂b - ∂/∂a,  ∂/∂v = ∂/∂c - ∂/∂a.
    """
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
            d2_dadb = C * i * j * (a**(i-1)) * (b**(j-1)) * (c**k)
        if (i >= 1) and (k >= 1):
            d2_dadc = C * i * k * (a**(i-1)) * (b**j) * (c**(k-1))
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


# === Diagnostic: one-time numerical check of bernstein_hess_uv ===
_BERNSTEIN_HESS_CHECKED = False

def _debug_check_bernstein_hess_uv():
    """
    One-time numerical sanity check for bernstein_hess_uv.

    For a random degree, term, and (u,v) in the simplex, compare the analytic
    Hessian components (f_uu, f_uv, f_vv) from bernstein_hess_uv against
    finite-difference estimates of the same Bernstein monomial
    B(a,b,c) = C * a**i * b**j * c**k, where a=1-u-v, b=u, c=v.

    This guards against sign mistakes in the mixed partials.
    """
    global _BERNSTEIN_HESS_CHECKED
    if _BERNSTEIN_HESS_CHECKED:
        return
    _BERNSTEIN_HESS_CHECKED = True

    try:
        rng = np.random.default_rng(0)
    except AttributeError:
        # Older NumPy: fall back to RandomState
        rng = np.random.RandomState(0)

    n = 4
    terms = bernstein_terms_deg(n)
    h = 1e-5
    max_abs_err = 0.0
    max_rel_err = 0.0

    # Sample a few random points away from the simplex edges
    for _ in range(5):
        u = 0.15 + 0.6 * rng.random()
        v = 0.15 + 0.6 * rng.random()
        if u + v >= 0.95:
            v = 0.95 - u
        huu, huv, hvv = bernstein_hess_uv(u, v, n, terms)
        # pick a random term
        t = int(rng.integers(len(terms)))
        i, j, k = terms[t]
        C = comb(n, i) * comb(n - i, j)

        def B(u_, v_):
            a_ = 1.0 - u_ - v_
            b_ = u_
            c_ = v_
            if a_ <= 0.0 or b_ <= 0.0 or c_ <= 0.0:
                # Outside simplex interior; skip
                return 0.0
            return C * (a_ ** i) * (b_ ** j) * (c_ ** k)

        f00 = B(u, v)
        fpp_u = B(u + h, v)
        fmm_u = B(u - h, v)
        fpp_v = B(u, v + h)
        fmm_v = B(u, v - h)
        fpp_uv = B(u + h, v + h)
        fpm_uv = B(u + h, v - h)
        fmp_uv = B(u - h, v + h)
        fmm_uv = B(u - h, v - h)

        f_uu_num = (fpp_u - 2.0 * f00 + fmm_u) / (h * h)
        f_vv_num = (fpp_v - 2.0 * f00 + fmm_v) / (h * h)
        f_uv_num = (fpp_uv - fpm_uv - fmp_uv + fmm_uv) / (4.0 * h * h)

        f_uu_ana = huu[t]
        f_vv_ana = hvv[t]
        f_uv_ana = huv[t]

        for ana, num in ((f_uu_ana, f_uu_num),
                         (f_vv_ana, f_vv_num),
                         (f_uv_ana, f_uv_num)):
            abs_err = abs(ana - num)
            max_abs_err = max(max_abs_err, abs_err)
            denom = max(1.0, abs(num))
            rel_err = abs_err / denom
            max_rel_err = max(max_rel_err, rel_err)

    print(f"[bernstein_hess_uv] numeric check: max_abs_err={max_abs_err:.3e}, "
          f"max_rel_err={max_rel_err:.3e}")


_debug_check_bernstein_hess_uv()


# === Helper: Build MA Hessian operators and edge weights ===
def build_ma_operators(uv, deg, terms, edge_band, edge_gain):
    """Build MA Hessian operators and edge weights for a given uv cloud.

    Returns:
      Huu, Hvv, Huv  : (m, K) arrays for the Bernstein Hessian components in (u,v)
      HT_Huu, HT_Hvv : (K, K) normal-equation blocks for A and B
      HT_Huv         : (K, K) normal-equation block for the cross-term C (with column whitening)
      w_edge         : (m,) per-sample edge weights for upweighting near the simplex boundary
    """
    uv = np.asarray(uv, dtype=float)
    u = uv[:, 0]
    v = uv[:, 1]
    m = uv.shape[0]
    K = len(terms)

    # Corner/edge emphasis: upweight samples within ~edge_band of edges.
    s_edge = np.minimum.reduce([u, v, 1.0 - u - v])
    edge = np.clip(edge_band - s_edge, 0.0, None) / edge_band
    w_edge = 1.0 + edge_gain * (edge ** 2)

    # Precompute per-sample Hessian basis in uv
    Huu = np.empty((m, K), dtype=np.float64)
    Huv = np.empty((m, K), dtype=np.float64)
    Hvv = np.empty((m, K), dtype=np.float64)
    for i, (ui, vi) in enumerate(uv):
        huu, huv, hvv = bernstein_hess_uv(ui, vi, deg, terms)
        Huu[i, :] = huu
        Huv[i, :] = huv
        Hvv[i, :] = hvv

    # Scale Hessian basis by n(n-1) to tame magnitudes
    scale = max(deg * (deg - 1), 1)
    Huu /= scale
    Hvv /= scale
    Huv /= scale

    # Precompute normal-equation pieces; whiten Huv so cross-term penalty bites in comparable units
    HT_Huu = Huu.T @ Huu
    HT_Hvv = Hvv.T @ Hvv
    col_huv = np.linalg.norm(Huv, axis=0) + 1e-12
    Huv /= col_huv
    HT_Huv = Huv.T @ Huv

    return Huu, Hvv, Huv, HT_Huu, HT_Hvv, HT_Huv, w_edge


# === Monge–Ampère solver: discrete collocation, analytic Bernstein Hessians ===
def solve_ma_bernstein(uv, ell_rhs, deg, config, det_j):
    """Solve log det(Hψ_xy) ≈ -ℓ_rhs at centroid samples using a Bernstein basis of degree `deg`.
    Here ℓ_rhs is the authalic log-density field as produced by w10/w30, i.e. ℓ_rhs = log(A/Ā)
    (possibly with residuals and scaling applied). Jacobian
    Returns (terms, c_psi, history) where c_psi are the coefficients of ψ.
    Uses Gauss–Newton on the residual r = log det(Hψ_xy) + ℓ_rhs (up to the constant metric term).
    The XY metric is handled by H_xy = J^{-T} H_uv J^{-1}, so det(H_xy)=det(H_uv)/det(J)^2.
    """
    terms_all = bernstein_terms_deg(deg)
    K = len(terms_all)

    u, v = uv[:, 0], uv[:, 1]
    m = uv.shape[0]

    # --- Extract config values ---
    verbosity = int(config.get("verbosity", 1))
    lam = float(config.get("lam", 1.0e-6))
    iters = int(config.get("iters", 20))
    damping0 = float(config.get("damping", 0.5))
    mu_tether = float(config.get("mu_tether", 0.015))
    gamma_cross = float(config.get("gamma_cross", 32.0))
    s_floor = float(config.get("s_floor", 3e-5))
    det_floor = float(config.get("det_floor", 3e-5))
    max_step_k = float(config.get("max_step_k", 0.12))
    edge_band = float(config.get("edge_band", 0.04))
    edge_gain = float(config.get("edge_gain", 1.5))
    alphas = config.get("alphas", [0.03, 0.10, 0.22, 0.38, 0.55, 0.72, 0.85, 0.92, 0.955, 0.975, 0.987, 0.994, 1.00])
    max_iter_per_stage = int(config.get("max_iter_per_stage", 32))
    mu_polish_scale = float(config.get("mu_polish_scale", 0.02))
    lam_gn_min = float(config.get("lam_gn_min", 3.0e-4))
    allow_beta = bool(config.get("allow_beta", False))
    enforce_d3 = bool(config.get("enforce_d3", True))

    use_neg_ell = bool(config.get("use_neg_ell", True))
    center_rhs = bool(config.get("center_rhs", True))


    damping = damping0

    Huu, Hvv, Huv, HT_Huu, HT_Hvv, HT_Huv, w_edge = build_ma_operators(
        uv, deg, terms_all, edge_band=edge_band, edge_gain=edge_gain
    )

    # Metric transform: H_xy = J^{-T} H_uv J^{-1}
    # For determinants: det(H_xy) = det(H_uv) / det(J)^2. Constant scale factor.
    rhs, log_detJ2, rhs_mean = build_rhs(
        ell_rhs,
        det_j,
        use_minus_ell=use_neg_ell,
        center=center_rhs,
    )
    if verbosity > 0:
        print(f"  MA target centering: log|detJ|^2={log_detJ2:.6f}, rhs_mean(before)={rhs_mean:.6f}")
    beta_cum = 0.0

    print(f"  SPD tether: mu={mu_tether:g}, gamma={gamma_cross:g}; hess-scale=1/(n(n-1))")
    # Notes: lighter mu_tether + looser floors let log-det deviate from 0 more readily.
    # Larger max_step_k and finer homotopy alphas help escape the nearly-constant basin.

    # --- Homotopy continuation: start from constant log-det then morph to rhs ---
    rhs_const = np.full_like(rhs, rhs.mean())

    # Initialize at constant target
    c = ls_init_for(rhs_const, Huu, Hvv, Huv, lam)
    # if enforce_d3:
    #     c = symmetrise_d3(terms_all, c)
    # Evaluate initial residual
    log_det_uv, det_uv, A, B, C = eval_logdet_uv(c, Huu, Hvv, Huv, s_floor, det_floor)
    r = log_det_uv - rhs_const
    rmse0 = float(np.sqrt(np.mean(r * r)))
    if verbosity > 0:
        print(f"  MA init(const): rmse0={rmse0:.6f}")
    hist = [rmse0]
    # SPD/scale diagnostics at init
    det_min, det_med = float(det_uv.min()), float(np.median(det_uv))
    tiny = np.mean(det_uv < det_floor)
    if verbosity > 0:
        print(f"  init det(H_uv): min={det_min:.3e} median={det_med:.3e} tiny_frac<{det_floor:.0e}={tiny:.3f}")
    if rmse0 > 5.0:
        if verbosity > 0:
            print("  init rmse large → re-initialize with stronger ridge")
        c = ls_init_for(np.zeros_like(rhs), Huu, Hvv, Huv, lam)  # s_target=1 baseline
        # if enforce_d3:
        #     c = symmetrise_d3(terms_all, c)
        log_det_uv, det_uv, A, B, C = eval_logdet_uv(c, Huu, Hvv, Huv, s_floor, det_floor)
        r = log_det_uv - rhs_const
        rmse0 = float(np.sqrt(np.mean(r * r)))
        if verbosity > 0:
            print(f"  MA init(retry): rmse0={rmse0:.6f}")
        hist[0] = rmse0

    # Homotopy blend fractions from constant → full target (tunable)
    rmse = rmse0

    for a in alphas:
        rhs_a = (1.0 - a) * rhs_const + a * rhs
        if verbosity > 1:
            print(f"  MA stage α={a:.2f}: target=blend(const,{a:.2f})")
        # Do not let the tether vanish; leave a 10% floor so scale stays controlled.
        mu_stage = mu_tether * ((1.0 - a)**2 + 0.05) * (1.0 + 0.5*rmse)
        # Re-center residual to current target before GN
        log_det_uv, det_uv, A, B, C = eval_logdet_uv(c, Huu, Hvv, Huv, s_floor, det_floor)
        if verbosity > 1:
            print(f"    diag α={a:.2f}: C/√(AB) med={np.median(np.abs(C)/np.sqrt(A*B)):.3f}")
        r = log_det_uv - rhs_a
        if allow_beta:
            beta = float(r.mean())
            r -= beta
            beta_cum += beta
        rmse = float(np.sqrt(np.mean(r * r)))
        hist.append(rmse)
        dmp = damping
        lam_gn = max(lam, lam_gn_min)
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
            # if enforce_d3:
            #     c_new = symmetrise_d3(terms_all, c_new)

            # Evaluate new residual
            log_det_new, det_new, A_new, B_new, C_new = eval_logdet_uv(
                c_new, Huu, Hvv, Huv, s_floor, max(det_floor, 1e-3))
            r_new = log_det_new - rhs_a
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
                if verbosity > 2:
                    print(f"    it {it + 1:02d}: rmse={rmse:.6f} (accepted, damping={dmp})")
                dmp = min(1.0, dmp * 1.2)
                lam_gn = max(lam, lam_gn / 1.5)
            else:
                dmp *= 0.5
                lam_gn = lam_gn * 3.0
                if verbosity > 2:
                    print(f"    it {it + 1:02d}: no improvement, damping→{dmp}, lam→{lam_gn:.1e}")
                if dmp < 1e-3:
                    break
        if verbosity > 1:
            print(f"  stage α={a:.2f}: det min/med = {det_uv.min():.2e}/{np.median(det_uv):.2e}  (β_cum={beta_cum:.3e})")

    # After homotopy, do a final polish against full rhs for any remaining iterations
    remaining = max(128, iters - (len(alphas) + 1) * max_iter_per_stage)  # guarantee at least a few polish steps
    lam_gn = max(lam, lam_gn_min)
    mu_polish = mu_tether * mu_polish_scale   # lighter tether for final fine-tuning
    for it in range(remaining):
        log_det_uv, det_uv, A, B, C = eval_logdet_uv(c, Huu, Hvv, Huv, s_floor, det_floor)
        if verbosity > 2:
            print(f"  polish diag: C/√(AB) med={np.median(np.abs(C)/np.sqrt(A*B)):.3f}")
        r = log_det_uv - rhs
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
        if enforce_d3:
            c_new = symmetrise_d3(terms_all, c_new)

        log_det_new, det_new, A_new, B_new, C_new = eval_logdet_uv(
            c_new, Huu, Hvv, Huv, s_floor, det_floor)
        r_new = log_det_new - rhs
        if allow_beta:
            r_new -= float(r_new.mean())
        rmse_new = float(np.sqrt(np.mean(r_new * r_new)))
        if rmse_new <= rmse:
            c = c_new
            hist[-1] = rmse_new
            if verbosity > 2:
                print(f"  polish it {it + 1:02d}: rmse={rmse_new:.6f} (accepted)")
            lam_gn = max(lam, lam_gn / 1.5)
        else:
            damping *= 0.5
            lam_gn = lam_gn * 3.0
            if verbosity > 2:
                print(f"  polish it {it + 1:02d}: no improvement, damping→{damping}, lam→{lam_gn:.1e}")
            if damping < 1e-3:
                break

    # Final diagnostic: compare MA-implied log-det field to authalic field
    log_det_uv, det_uv, a, b, c_cross = eval_logdet_uv(c, Huu, Hvv, Huv, s_floor, det_floor)
    lhat = log_det_uv - log_detJ2
    lhat_centered = lhat - lhat.mean()
    ell_centered = ell_rhs - ell_rhs.mean()
    std_lhat = float(lhat_centered.std())
    std_ell = float(ell_centered.std())

    if std_lhat < 1e-12 or std_ell < 1e-12:
        # One of the fields is (numerically) constant; correlation is undefined.
        corr_plus = float('nan')
        corr_minus = float('nan')
        if verbosity > 0:
            which = []
            if std_lhat < 1e-12:
                which.append("lhat")
            if std_ell < 1e-12:
                which.append("ell")
            which_str = "/".join(which) if which else "unknown"
            print(f"[diag] corr(lhat, ell): degenerate (nearly-constant field: {which_str})")
    else:
        corr_plus = np.corrcoef(lhat_centered, ell_centered)[0, 1]
        corr_minus = np.corrcoef(lhat_centered, -ell_centered)[0, 1]

    if verbosity > 0:
        if not (np.isnan(corr_plus) or np.isnan(corr_minus)):
            print(f"[diag] corr(lhat, ell)={corr_plus:+.3f}, corr(lhat, -ell)={corr_minus:+.3f}")
        print(f"[diag] std(lhat)={std_lhat:.4f}, std(ell)={std_ell:.4f}")

    return terms_all, c, np.array(hist)


def run(layer, octant, ma_deg, bn_deg, config_name: str):
    global LAPLACIAN_H
    ma_config = load_ma_config(config_name)
    tweak = ma_config.get('tweak', '')
    use_pre_ak = bool(ma_config.get('use_pre_ak', False))
    reg = Registrar()
    b_oct = reg.domain('b_oct')
    sign = b_oct.signs_by_id[octant]
    face = b_oct.signs[sign]
    prj = b_oct.projs[face]
    mode = b_oct.oid_mo[octant]
    q = prj.matrix.T @ prj.orient
    e1_xyz = q[:, 0]  # 3-vector
    e2_xyz = q[:, 1]  # 3-vector

    # --- Load MA input bundle (from prewarp phase) ---
    ma_input_path = Path(f"ma_input_l{layer}_m{mode}_n{bn_deg}_{tweak}.npz")
    ma_in = np.load(ma_input_path, allow_pickle=True)
    uv = ma_in["uv_cent"]
    # p_centroids = ma_in["p_centroids"]  # Projected centroids in 'c_ell' domain
    # j_ak = ma_in["j_centroids"]  # Jacobian of centroids (N_tri, 3, 3)
    rlog = ma_in["ell"]
    rlog_residual = ma_in["ell_resid"]
    terms = ma_in["terms"]
    bn_deg = int(ma_in["degree"])
    c_init = ma_in.get("c_init", None)
    g_inv = ma_in.get("g_inv", {})

    print(f"Loaded MA input: {ma_input_path}")
    print(f"  uv_cent shape={uv.shape}, ell shape={rlog.shape}, terms={len(terms)}, degree={bn_deg}")
    print(f"  ell stats: min={rlog.min():.4f}, max={rlog.max():.4f}, mean={rlog.mean():.4f}, std={rlog.std():.4f}")
    print(f"  ell_resid stats: min={rlog_residual.min():.4f}, "
          f"max={rlog_residual.max():.4f}, "
          f"mean={rlog_residual.mean():.4f}, "
          f"std={rlog_residual.std():.4f}")

    LAPLACIAN_H = float(ma_in["laplacian_h"])

    g_inv_arr = ma_in["g_inv"]
    if np.size(g_inv_arr) == 4:
        g_inv_mat = np.array(g_inv_arr, dtype=float).reshape(2, 2)
        g11 = float(g_inv_mat[0, 0])
        g12 = float(g_inv_mat[0, 1])
        g22 = float(g_inv_mat[1, 1])
        g_mat = np.linalg.inv(g_inv_mat)
        det_j = float(np.sqrt(abs(np.linalg.det(g_mat))))
    else:
        raise ValueError("Missing or invalid g_inv in MA input bundle")

    print(f"  metric: Ginv=[[{g11:.6f},{g12:.6f}],[{g12:.6f},{g22:.6f}]], detJ={det_j:.6f}, h={LAPLACIAN_H}")

    # --- MA solver configuration ---
    # config_name = "default"  # change this to select a different named config from ma_config.yaml
    use_residual = ma_config.get('use_residual')
    use_minus_ell = ma_config.get('use_minus_ell', None)
    if use_minus_ell is not None:
        # Bridge YAML key 'use_minus_ell' to the internal 'use_neg_ell' flag
        ma_config['use_neg_ell'] = bool(use_minus_ell)
    ell_field = rlog_residual if use_residual else rlog
    print(f"[MA] using rhs = {'ell_resid' if use_residual else 'ell'}")

    print(f"[MA] ell_field stats: min={ell_field.min():.4f}, "
          f"max={ell_field.max():.4f}, mean={ell_field.mean():.4f}, "
          f"std={ell_field.std():.4f}")

    ell_mean = np.mean(ell_field)
    ell_to_use = ell_field / ell_mean

    print(f"[MA] centred ell stats: min={ell_to_use.min():.4f}, "
          f"max={ell_to_use.max():.4f}, mean={ell_to_use.mean():.4f}, "
          f"std={ell_to_use.std():.4f}")

    rhs_scale = float(ma_config.get('rhs_scale', 1.0))
    if rhs_scale != 1.0:
        print(f"[MA] applying rhs_scale={rhs_scale:.3g}")
        ell_to_use *= rhs_scale

    centred_ell = ell_field - np.mean(ell_field)
    print(f"[diag] std(ell_orig)     = {ell_field.std():.4f}")
    print(f"[diag] std(centred_ell)  = {centred_ell.std():.4f}")

    # --- Solve Monge–Ampère in Bernstein basis ---
    print("Starting MA solve (discrete collocation, Bernstein basis)...")
    print(f"  MA config '{config_name}':")
    print(f"    lam={ma_config.get('lam', 1.0e-6)}, iters={ma_config.get('iters', 20)}, "
          f"damping={ma_config.get('damping', 0.5)}")
    print(f"    mu_tether={ma_config.get('mu_tether', 0.015)}, "
          f"gamma_cross={ma_config.get('gamma_cross', 32.0)}")
    terms_ma, c_ma, hist = solve_ma_bernstein(uv, ell_to_use, ma_deg, ma_config, det_j)
    print(f"MA done: deg={ma_deg}, iters={len(hist)}, final rmse={hist[-1]:.6f}")

    # Save ψ coefficients and metadata
    np.savez(
        f"ma_psi_x{config_name}_l{layer}_m{mode}_n{ma_deg}_{tweak}.npz",
        terms=np.array(terms_ma, dtype=object),
        c=c_ma,
        depth=layer,
        mode=mode,
        ginv=g_inv,
        ell=rlog,
        ell_resid=rlog_residual,
        config=ma_config,
        config_name=config_name,
    )
    print(f"Saved MA coefficients to ma_psi_x{config_name}_l{layer}_m{mode}_n{ma_deg}_{tweak}.npz")


if __name__ == '__main__':
    octant = 0              # which octant to process
    ma_deg = 16             # MA potential ψ
    bn_deg = 16             # Preconditioned Bernstein degree
    layer = 5               # choose octant hex_layer
    configs = [
        'l5_v0305',
     ]
    for config_name in configs:
        run(layer, octant, ma_deg, bn_deg, config_name)
