# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
PKDO polynomial fit to the authalic warp displacement field (b_oct space).

Fits a pair of PKDO polynomials (cx, cy) to the Sinkhorn/CT displacement field
and stores them in hhg9/data/{ellipsoid}_pkdo_warp.npz (462 floats at degree 20,
vs ~16 MB for the CT grid). Wired into AuthalicWarp as interp='pkdo'.

CONCLUSION (2026-04-25):
The polynomial warp is an interesting sideshow, not a CT replacement.
CT (piecewise cubic, interpolates all 266k Sinkhorn points exactly) achieves
log-ratio σ ≈ 5×10⁻⁵ and p99 ≈ 0.014%. PKDO at degree 38 (820 coefficients)
gives σ ≈ 8×10⁻⁴ and p99 ≈ 0.07% — about 15× worse — and the gap is structural:
a global polynomial cannot capture the local distortion concentrated near the
6 octahedral poles without Runge-like oscillation everywhere else.

GENUINE USE CASE:
The polynomial does provide a smooth, analytic Jacobian (warp_jacobian_2d), useful
for metric-tensor / area-distortion analysis where a differentiable approximation
matters more than raw authalic fidelity.

PRACTICAL OUTCOME:
- CT (.npz grid) remains the production default; auto-detected by OctahedralBarycentric.
- PKDO coefficients (.npz) loaded as interp='pkdo' when file size is hard-constrained
  (e.g. embedded / no-data-file fallback), accepting the ~15× quality tradeoff.

Last Tested
16 Jun 2026 0.1.3a0 (timeout)
"""

import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge
from scipy.special import eval_jacobi
from hhg9 import Registrar, Points

os.makedirs('output', exist_ok=True)


def pkdo_vandermonde(xy, degree):
    x, y = xy[:, 0], xy[:, 1]

    # 1. Map to Duffy Coordinates
    r = np.sqrt(2) * x + (np.sqrt(6) / 3) * y - 1.0 / 3.0
    s = (-2.0 * np.sqrt(6) / 3) * y - 1.0 / 3.0

    # CRITICAL: Clamp s and calculate 'a' safely
    s_clamped = np.clip(s, -1.0, 1.0 - 1e-15)
    # Clamp 'a' to [-1, 1] to prevent Degree 80 explosion
    a = np.clip(2.0 * (1.0 + r) / (1.0 - s_clamped) - 1.0, -1.0, 1.0)

    N = len(x)
    n_basis = (degree + 1) * (degree + 2) // 2
    V = np.zeros((N, n_basis))

    # 2. Precompute Legendre P_p(a)
    poly_a = [np.ones(N), a]
    for p in range(2, degree + 1):
        # Using the standard recurrence; because |a| <= 1, poly_a stays <= 1.0
        p_val = ((2 * p - 1) * a * poly_a[-1] - (p - 1) * poly_a[-2]) / p
        poly_a.append(p_val)

    col = 0
    for p in range(degree + 1):
        # 3. Duffy weight stays stable because s is clamped
        Jp_weighted = poly_a[p] * ((1.0 - s_clamped) / 2.0) ** p

        alpha = 2 * p + 1
        beta = 0

        # 4. Inner Recurrence for P_q^{(alpha, 0)}(s)
        q_prev2 = np.ones(N)
        q_prev1 = 0.5 * (alpha - beta + (alpha + beta + 2) * s_clamped)

        # Norm for q=0
        norm_0 = np.sqrt((2 * p + 1) * (p + 1) / 2.0)
        V[:, col] = norm_0 * Jp_weighted * q_prev2
        col += 1

        if degree - p >= 1:
            norm_1 = np.sqrt((2 * p + 1) * (p + 2) / 2.0)
            V[:, col] = norm_1 * Jp_weighted * q_prev1
            col += 1

        for q in range(2, degree + 1 - p):
            # Standard Jacobi coefficients
            # We use the float64 safe form to prevent intermediate overflows
            n = q
            apb = alpha + beta
            ns2 = 2 * n + apb

            c = 2 * n * (n + apb) * (ns2 - 2)
            d = (ns2 - 1) * (alpha ** 2 - beta ** 2)
            e = (ns2 - 1) * ns2 * (ns2 - 2)
            f = 2 * (n + alpha - 1) * (n + beta - 1) * ns2

            q_curr = ((d + e * s_clamped) * q_prev1 - f * q_prev2) / c

            norm = np.sqrt((2 * p + 1) * (p + q + 1) / 2.0)
            V[:, col] = norm * Jp_weighted * q_curr

            q_prev2, q_prev1 = q_prev1, q_curr
            col += 1
    return V


def pkdo_vandermonde_easy(xy, degree):
    """Vandermonde matrix for PKDO basis on b_oct triangle.
    Normalization C_{p,q} = sqrt((2p+1)(p+q+1)/2) gives L2-orthonormal functions
    on the reference simplex (area=2).  Must match _pkdo_vandermonde in octahedral_barycentric.py.
    """
    x, y = xy[:, 0], xy[:, 1]
    r = np.sqrt(2) * x + (np.sqrt(6) / 3) * y - 1.0 / 3.0
    s = (-2.0 * np.sqrt(6) / 3) * y - 1.0 / 3.0
    N = len(r)
    n_basis = (degree + 1) * (degree + 2) // 2
    V = np.zeros((N, n_basis))
    s_safe = np.minimum(s, 1.0 - 1e-14)
    a = 2.0 * (1.0 + r) / (1.0 - s_safe) - 1.0
    col = 0
    for p in range(degree + 1):
        Jp = eval_jacobi(p, 0, 0, a)
        hp = ((1.0 - s_safe) / 2.0) ** p
        Jph = Jp * hp
        for q in range(degree + 1 - p):
            Jq = eval_jacobi(q, 2 * p + 1, 0, s)
            norm = np.sqrt((2 * p + 1) * (p + q + 1) / 2.0)
            V[:, col] = norm * Jph * Jq
            col += 1
    return V


def fit_warp_pkdo(src, dst, degree=12, alpha=1e-10):
    """Fit PKDO polynomials to the b_oct warp displacement (dx, dy) separately.

    Ghost augmentation: for mo=1 octants, do() flips y before evaluating the polynomial,
    so the polynomial must cover y up to ~+0.816 (above the equator at ~+0.408).
    Only apex points (src_y < -equator_y) need ghosts — their reflections land above
    the equator and outside the original training domain, avoiding conflicts.
    dx mirrors symmetrically; dy anti-symmetrically.
    """
    # equator_y = src[:, 1].max()          # ≈ +√6/6 ≈ +0.408
    # apex_mask = src[:, 1] < -equator_y  # reflected y lands above equator → new coverage only
    # src_ghost = np.column_stack([src[apex_mask, 0], -src[apex_mask, 1]])
    # dst_ghost = np.column_stack([dst[apex_mask, 0], -dst[apex_mask, 1]])
    # src_aug = np.vstack([src, src_ghost])
    # dst_aug = np.vstack([dst, dst_ghost])
    eps = 1e-9
    x, y = src[:, 0], src[:, 1]

    # Identify boundary points
    is_base = np.abs(y - np.sqrt(6) / 6) < eps
    is_right = np.abs(y - np.sqrt(3) * x + np.sqrt(6) / 3) < eps
    is_left = np.abs(y + np.sqrt(3) * x + np.sqrt(6) / 3) < eps
    is_edge = is_base | is_right | is_left
    verts = (is_base & is_right) | (is_base & is_left) | (is_left & is_right)
    weights = np.ones(len(src))
    weights[is_edge] = 1000000.0  # Or even 10000.0
    weights[verts] = 100000000.0

    src_aug, dst_aug = src, dst
    disp = dst_aug - src_aug
    A = pkdo_vandermonde(src_aug, degree)
    cx = Ridge(alpha=alpha, fit_intercept=False).fit(A, disp[:, 0], sample_weight=weights).coef_
    cy = Ridge(alpha=alpha, fit_intercept=False).fit(A, disp[:, 1], sample_weight=weights).coef_
    return cx, cy


def apply_warp_pkdo(xy, cx, cy, degree):
    """Apply polynomial warp: b_oct → warped b_oct."""
    A = pkdo_vandermonde(xy, degree)
    return xy + np.stack([A @ cx, A @ cy], axis=1)


def warp_jacobian_2d(src, cx, cy, degree, h=1e-6):
    """2D Jacobian determinant of the warp map via finite differences.
    J = det([[1+ddx/dbx, ddx/dby],[ddy/dbx, 1+ddy/dby]])
    For a well-behaved warp J is smooth and positive; Coeff[J]=std/mean watches overfitting.
    """
    bx, by = src[:, 0], src[:, 1]
    def disp_at(pts):
        A = pkdo_vandermonde(pts, degree)
        return A @ cx, A @ cy   # dx, dy

    dx_xp, dy_xp = disp_at(np.column_stack([bx + h, by]))
    dx_xm, dy_xm = disp_at(np.column_stack([bx - h, by]))
    dx_yp, dy_yp = disp_at(np.column_stack([bx, by + h]))
    dx_ym, dy_ym = disp_at(np.column_stack([bx, by - h]))

    ddx_dbx = (dx_xp - dx_xm) / (2 * h)
    ddx_dby = (dx_yp - dx_ym) / (2 * h)
    ddy_dbx = (dy_xp - dy_xm) / (2 * h)
    ddy_dby = (dy_yp - dy_ym) / (2 * h)

    return (1 + ddx_dbx) * (1 + ddy_dby) - ddx_dby * ddy_dbx


def plot_warp_comparison(src, disp_ct, disp_poly, degree, save=True):
    res = disp_ct - disp_poly
    bx, by = src[:, 0], src[:, 1]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=150)
    rows = [
        ([disp_ct[:, 0], disp_ct[:, 1], np.hypot(disp_ct[:, 0], disp_ct[:, 1])],
         ['CT dx', 'CT dy', 'CT |d|'], 'coolwarm'),
        ([res[:, 0], res[:, 1], np.hypot(res[:, 0], res[:, 1])],
         [f'residual dx (deg {degree})', f'residual dy', 'residual |d|'], 'coolwarm'),
    ]
    for row_idx, (data_list, titles, cmap) in enumerate(rows):
        for col_idx, (data, title) in enumerate(zip(data_list, titles)):
            ax = axes[row_idx, col_idx]
            im = ax.tripcolor(bx, by, data, cmap=cmap, shading='flat')
            ax.set_aspect('equal')
            ax.set_title(title)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    if save:
        path = f'output/ex0065c_{degree}_warp_residual.png'
        fig.savefig(path, dpi=300)
        print(f'  Saved: {path}')
        plt.close(fig)
    else:
        plt.show()


if __name__ == '__main__':
    import pathlib
    import hhg9
    from importlib import resources as pkg_resources

    rg = Registrar()
    b_raw = rg.domain('b_raw')
    o_raw = b_raw.projs['NEA']  # AK projection, no warp

    # Load CT warp data directly — this script generates PKDO coefficients from it,
    # so we always need the raw CT arrays regardless of which warp mode is active.
    ell = rg.ellipsoid_name
    pkg = "hhg9.data"
    ct_file = pkg_resources.files(pkg).joinpath(f"{ell}_l5_warp_data.npz")
    if not ct_file.exists():
        ct_file = pkg_resources.files(pkg).joinpath("WGS84_l5_warp_data.npz")
    ct_data = np.load(ct_file)
    src = ct_data['source_pts']
    dst = ct_data['target_pts']
    disp_ct = dst - src

    print(f"Warp grid: {src.shape[0]} points")
    print(f"CT displacement  max|dx|: {np.max(np.abs(disp_ct[:, 0])):.6f}"
          f"  max|dy|: {np.max(np.abs(disp_ct[:, 1])):.6f}  b_oct units")

    # Reference: CT-warped points projected through AK
    ct_c_pts = o_raw.backward(Points(dst.copy(), b_raw, oid=0))

    cx, cy = None, None
    for degree in [80]:  # range(2, 26, 2):
        cx, cy = fit_warp_pkdo(src, dst, degree, alpha=1e-14)  # was 1e-14
        poly_dst = apply_warp_pkdo(src, cx, cy, degree)
        res = disp_ct - (poly_dst - src)

        poly_c_pts = o_raw.backward(Points(poly_dst.copy(), b_raw, oid=0))
        delta_3d = poly_c_pts.coords - ct_c_pts.coords
        max_3d_m = np.max(np.abs(delta_3d))

        J = warp_jacobian_2d(src, cx, cy, degree)
        cj = np.std(J) / np.mean(J)

        n_coeffs = len(cx) + len(cy)
        print(f"\n--- Degree {degree}  ({n_coeffs} coefficients) ---")
        print(f"  b_oct residual  max|dx|: {np.max(np.abs(res[:, 0])):.6e}"
              f"  max|dy|: {np.max(np.abs(res[:, 1])):.6e}")
        print(f"  3D error vs CT: {max_3d_m:.12f} m   Coeff[J]: {cj:.14f}")

        # Plots at last degree
        plot_warp_comparison(src, disp_ct, poly_dst - src, degree)

    # Save PKDO coefficients to hhg9/data/
    save_degree = degree
    cx_save, cy_save = cx, cy
    #cx_save, cy_save = fit_warp_pkdo(src, dst, save_degree, alpha=1e-12)
    data_dir = pathlib.Path(hhg9.__file__).parent / 'data'
    save_path = data_dir / f"{ell}_pkdo_warp.npz"
    np.savez(save_path, cx=cx_save, cy=cy_save, degree=np.array(save_degree))
    print(f"\nSaved PKDO coefficients (degree {save_degree}, {len(cx_save) * 2} floats): {save_path}")
    poly_dst = apply_warp_pkdo(src, cx, cy, degree)
    ct = dict(ct_data)
    ct['source_pts'] = poly_dst
    grid_path = data_dir / f"{ell}_pkdo_grid_{save_degree}.npz"
    np.savez(grid_path, **ct)
    print(f"\nSaved PKDO grid (degree {save_degree}, {len(cx_save) * 2} floats): {grid_path}")
