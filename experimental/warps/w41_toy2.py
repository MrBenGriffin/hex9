from __future__ import annotations
import numpy as np

from w20_bernstein_fit_phi import bernstein_terms_deg
from w40_solve_monge_ampere import (
    symmetrise_d3,
    solve_ma_bernstein,
    build_ma_operators, load_ma_config,
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


def project_to_simplex(uv: np.ndarray) -> np.ndarray:
    """Project (u,v) points back into the simplex u>=0, v>=0, u+v<=1.

    We do this via a simple barycentric clip + renormalise, mirroring the
    runtime warp behaviour without depending on the Points class.
    """
    u = uv[..., 0].copy()
    v = uv[..., 1].copy()
    w = 1.0 - u - v

    u = np.clip(u, 0.0, None)
    v = np.clip(v, 0.0, None)
    w = np.clip(w, 0.0, None)

    s = u + v + w
    zero_mask = (s == 0.0)
    if np.any(zero_mask):
        u[zero_mask] = 1.0 / 3.0
        v[zero_mask] = 1.0 / 3.0
        w[zero_mask] = 1.0 / 3.0
        s[zero_mask] = 1.0

    u /= s
    v /= s
    return np.stack([u, v], axis=-1)


def toy_warp_test(uv, vals, terms, c_est, deg, label: str, target_max_disp=1e-3):
    """Toy analogue of the w50 warp experiment.

    We treat the synthetic log-det field as a stand-in "ℓ" and ask:
    for a small warp generated from ψ, does the log-det field move
    towards or away from 0 (uniform) under the same kind of small
    displacement that w50 uses?
    """
    from experimental.algorithms.warp import Warper

    print(f"[toy-warp] config={label}")

    warper = Warper()
    warper.set_values(terms, c_est, deg)
    _, grad = warper.eval_and_grad_uv(uv)

    grad_norm = np.linalg.norm(grad, axis=1)
    gmin, gmed, gmax = grad_norm.min(), np.median(grad_norm), grad_norm.max()
    print(f"[toy-warp] grad_norm: min={gmin:.3e} med={gmed:.3e} max={gmax:.3e}")

    # Geometry-based scale: make max displacement in uv about target_max_disp
    scale_geom = target_max_disp / gmax
    print(f"[toy-warp] scale_geom≈{scale_geom:.3e} for target_max_disp≈{target_max_disp}")

    # Helper to build log-det from operators + coeffs, matching ell_rhs()
    s_floor = vals["s_floor"]
    det_floor = vals["det_floor"]

    def log_det_from(huu, hvv, huv, coeffs):
        a = 1.0 + huu @ coeffs
        b = 1.0 + hvv @ coeffs
        c_cross = 0.1 * (huv @ coeffs)
        a = np.maximum(a, s_floor)
        b = np.maximum(b, s_floor)
        det = a * b - c_cross * c_cross
        det = np.maximum(det, det_floor)
        return np.log(det)

    # Baseline log-det at original uv using the true coefficients
    c_true = vals["c_true"]
    log_det0 = log_det_from(vals["Huu"], vals["Hvv"], vals["Huv"], c_true)
    rmse0 = np.sqrt(np.mean(log_det0 ** 2))

    for factor in [0.25, 0.5, 1.0]:
        scale = factor * scale_geom
        uv_new = uv - scale * grad
        uv_new = project_to_simplex(uv_new)

        huu_new, hvv_new, huv_new, _, _, _, _ = build_ma_operators(
            uv_new,
            deg,
            terms,
            edge_band=0.05,
            edge_gain=0.0,
        )

        log_det1 = log_det_from(huu_new, hvv_new, huv_new, c_true)
        rmse1 = np.sqrt(np.mean(log_det1 ** 2))
        delta = rmse1 - rmse0

        delta_field = log_det1 - log_det0
        corr = np.corrcoef(
            log_det0 - log_det0.mean(),
            delta_field - delta_field.mean(),
        )[0, 1]

        print(
            f"[toy-warp] factor={factor:.2f} scale={scale:.3e} "
            f"RMSE0={rmse0:.6f} RMSE1={rmse1:.6f} ΔRMSE={delta:+.6e} "
            f"Corr(log_det, Δlog)={corr:+.3f}"
        )
    print()


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
    ell_rhs_vals, vals = ell_rhs(d_rng, degree, uv, d3=True)

    cfg = load_ma_config("stronger_reg_sym", "toy_config.yaml")
    terms_est, c_est, hist = solve_ma_bernstein(uv, ell_rhs_vals, degree, cfg, det_j=1.0)
    print("[toy] stronger_reg_sym")
    report(vals, c_est)
    toy_warp_test(uv, vals, terms_est, c_est, degree, label="stronger_reg_sym")

    cfg = load_ma_config("stronger_reg_asym", "toy_config.yaml")
    terms_est, c_est, hist = solve_ma_bernstein(uv, ell_rhs_vals, degree, cfg, det_j=1.0)
    print("[toy] stronger_reg_asym")
    report(vals, c_est)
    toy_warp_test(uv, vals, terms_est, c_est, degree, label="stronger_reg_asym")
