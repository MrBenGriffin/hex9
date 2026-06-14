from __future__ import annotations
import numpy as np
from w20_bernstein_fit_phi import bernstein_terms_deg
from w40_solve_monge_ampere import (
    symmetrise_d3,
    solve_ma_bernstein,
    build_ma_operators, build_rhs, load_ma_config,
)


def get_uv(rng, m):
    """Sample m points uniformly in the simplex using an rng."""
    uv = []
    while len(uv) < m:
        u = rng.random()
        v = rng.random()
        if u > 0 and v > 0 and u + v < 1:
            uv.append((u, v))
    return np.array(uv)


def ell_rhs_v0(rng, deg, uv, d3):
    """Compute the right-hand side of the Monge-Ampere equation for toy problem."""
    terms = bernstein_terms_deg(deg)
    K = len(terms)
    # Reuse the same MA operator construction as the main solver
    # Huu, Hvv, Huv are already scaled consistently inside build_ma_operators
    Huu, Hvv, Huv, HT_Huu, HT_Hvv, HT_Huv, w_edge = build_ma_operators(
        uv,
        deg,
        terms,
        edge_band=0.05,
        edge_gain=1.0,
    )
    c_true = 0.1 * rng.standard_normal(K)
    if d3:
        c_true = symmetrise_d3(terms, c_true)

    # 4) true log-det
    s_floor = 3e-5
    det_floor = 3e-5
    a_true = np.maximum(Huu @ c_true, s_floor)
    b_true = np.maximum(Hvv @ c_true, s_floor)
    c_cross_true = Huv @ c_true
    det_uv_true = np.maximum(a_true * b_true - c_cross_true * c_cross_true,
                             det_floor)
    values = {
        "H": Huu,
        "Huu": Huu,
        "Hvv": Hvv,
        "Huv": Huv,
        "HT_Huu": HT_Huu,
        "HT_Hvv": HT_Hvv,
        "HT_Huv": HT_Huv,
        "w_edge": w_edge,
        "det_floor": det_floor,
        "s_floor": s_floor,
        "c_true": c_true,
        "det_uv_true": det_uv_true
    }
    return np.log(det_uv_true), values


def ell_rhs(rng, deg, uv, d3):
    terms = bernstein_terms_deg(deg)
    k = len(terms)

    huu, hvv, huv, ht_huu, ht_hvv, ht_huv, w_edge = build_ma_operators(
        uv,
        deg,
        terms,
        edge_band=0.05,
        edge_gain=0.0,  # for the toy, disable extra edge weighting
    )

    # small true coefficients
    c_true = 0.05 * rng.standard_normal(k)
    if d3:
        c_true = symmetrise_d3(terms, c_true)

    # base SPD metric ~ identity
    s_floor = 1e-3
    det_floor = 1e-3

    a_true = 1.0 + huu @ c_true
    b_true = 1.0 + hvv @ c_true
    c_cross_true = 0.1 * (huv @ c_true)

    # enforce floors
    a_true = np.maximum(a_true, s_floor)
    b_true = np.maximum(b_true, s_floor)
    det_uv_true = a_true * b_true - c_cross_true * c_cross_true
    det_uv_true = np.maximum(det_uv_true, det_floor)

    values = {
        "Huu": huu,
        "Hvv": hvv,
        "Huv": huv,
        "HT_Huu": ht_huu,
        "HT_Hvv": ht_hvv,
        "HT_Huv": ht_huv,
        "w_edge": w_edge,
        "det_floor": det_floor,
        "s_floor": s_floor,
        "c_true": c_true,
        "det_uv_true": det_uv_true,
    }
    # now ell_rhs is a log-det field centred near 0, with good variation
    return np.log(det_uv_true), values


def run_toy(rng, deg=16, m=20000, d3=True):
    print(f"[toy] deg={deg:.4e}, m={m}, d3:{d3}")
    uv = get_uv(rng, m)
    ell_rhs_toy, vals = ell_rhs(rng, deg, uv, d3)
    ma_config = {
        "lam": 1e-4,
        "iters": 60,
        "damping": 0.5,
        "mu_tether": 0.0,       # start with no extra tether
        "gamma_cross": 0.0,
        "alphas": [1.0],        # no homotopy, go straight to full rhs
        "max_iter_per_stage": 80,
        "allow_beta": False,
        "enforce_d3": d3,
        "verbosity": 1,
    }
    terms_ma, c_est, hist = solve_ma_bernstein(uv, ell_rhs_toy, deg, ma_config, 1.0)
    report()

def report(vals, c_est):
    huu = vals["Huu"]
    hvv = vals["Hvv"]
    huv = vals["Huv"]
    s_floor = vals["s_floor"]
    det_floor = vals["det_floor"]
    det_uv_true = vals["det_uv_true"]
    log_det_uv_true = np.log(det_uv_true)
    c_true = vals["c_true"]
    a_est = np.maximum(huu @ c_est, s_floor)
    b_est = np.maximum(hvv @ c_est, s_floor)
    c_cross_est = huv @ c_est
    det_uv_est = np.maximum(a_est * b_est - c_cross_est * c_cross_est, det_floor)
    log_det_uv_est = np.log(det_uv_est)

    diff = log_det_uv_est - log_det_uv_true
    rmse = np.sqrt(np.mean(diff**2))
    corr = np.corrcoef(log_det_uv_est - log_det_uv_est.mean(),
                       log_det_uv_true - log_det_uv_true.mean())[0, 1]
    print(f"[toy] rmse(log_det)={rmse:.4e}, corr={corr:+.3f}")
    print(f"[toy] ||c_true||={np.linalg.norm(c_true):.3f}, ||c_est||={np.linalg.norm(c_est):.3f}")
    coeff_diff = c_est - c_true
    print(f"[toy] ||c_true||={np.linalg.norm(c_true):.3f}, "
          f"||c_est||={np.linalg.norm(c_est):.3f}, "
          f"||Δc||={np.linalg.norm(coeff_diff):.3f}")
    print()


if __name__ == "__main__":
    d_rng = np.random.default_rng(1)
    pts = 20_000
    degree = 16

    uv = get_uv(d_rng, pts)
    ell_rhs, vals = ell_rhs(d_rng, degree, uv, d3=True)

    cfg = load_ma_config("stronger_reg_sym", "toy_config.yaml")
    terms_est, c_est, hist = solve_ma_bernstein(uv, ell_rhs, degree, cfg, det_j=1.0)
    report(vals, c_est)

    cfg = load_ma_config("stronger_reg_asym", "toy_config.yaml")
    terms_est, c_est, hist = solve_ma_bernstein(uv, ell_rhs, degree, cfg, det_j=1.0)
    report(vals, c_est)
