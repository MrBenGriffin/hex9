# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Single-octant area-deviation map in native b_oct face coordinates.

Renders one octant of the H9 triangular sub-grid in the octahedral face
coordinate space (b_oct: 2D Cartesian, origin at face barycentre). Cells
are coloured by geodesic area deviation from ideal.

All 8 octants are mathematically identical under the transform, so one
octant fully characterises the warp quality. This is also the cleanest
possible representation — b_oct is the native coordinate space; no
reprojection artefacts.

Outputs:
  output/ex0063b_octant_L{depth}.png

Last Tested
2026-04-20
"""

import os
import numpy as np
from matplotlib import pyplot as plt, colors
from sklearn.linear_model import Ridge
import modepy as mp

from hhg9 import Registrar, Points
import numpy.polynomial.chebyshev as cheb


def polyfit_cheby(reg, src, degree=6, use_asymmetry=False, alpha=1.0):
    Z_clough = gather_z(reg, use_asymmetry)
    x_flat, y_flat, z_flat = src[:, 0], src[:, 1], Z_clough

    x_min, x_max = -np.sqrt(2) / 2, np.sqrt(2) / 2
    y_min, y_max = np.sqrt(6) / 6, -np.sqrt(2) * np.sqrt(3) / 3

    x_scaled = 2 * (x_flat - x_min) / (x_max - x_min) - 1
    y_scaled = 2 * (y_flat - y_min) / (y_max - y_min) - 1

    A_full = cheb.chebvander2d(x_scaled, y_scaled, [degree, degree])

    keep_indices = []
    for i in range(degree + 1):
        for j in range(degree + 1):
            idx = i * (degree + 1) + j
            if i + j <= degree:
                if use_asymmetry or (i % 2 == 0):
                    keep_indices.append(idx)

    A_filtered = A_full[:, keep_indices]
    valid_mask = ~np.isnan(z_flat)

    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(A_filtered[valid_mask], z_flat[valid_mask])
    return model.coef_, (x_min, x_max, y_min, y_max)


def eval_poly_cheby(x, y, coeffs, degree, domain, symmetric=True):
    x_min, x_max, y_min, y_max = domain

    # Scale inputs to [-1, 1] using the same domain as the fit
    xs = 2 * (x - x_min) / (x_max - x_min) - 1
    ys = 2 * (y - y_min) / (y_max - y_min) - 1

    # Reconstruct the coefficient grid
    full_coeffs = np.zeros((degree + 1, degree + 1))
    idx = 0
    for i in range(degree + 1):
        for j in range(degree + 1):
            if (symmetric and i % 2 != 0) or (i + j > degree):
                continue
            full_coeffs[i, j] = coeffs[idx]
            idx += 1

    return cheb.chebval2d(xs, ys, full_coeffs)


def boct_to_ref(xy):
    """Affine map: b_oct triangle → PKDO reference simplex (-1,-1),(1,-1),(-1,1).
    V0=(-√2/2,√6/6)→(-1,-1), V1=(√2/2,√6/6)→(1,-1), V2=(0,-√6/3)→(-1,1)
    """
    x, y = xy[:, 0], xy[:, 1]
    r = np.sqrt(2) * x + (np.sqrt(6) / 3) * y - 1/3
    s = (-2 * np.sqrt(6) / 3) * y - 1/3
    return np.array([r, s])  # shape (2, N) for modepy


def polyfit_pkdo(reg, src, degree=6, for_xy=False, alpha=1.0):
    Z_clough = gather_z(reg, for_xy)
    nodes = boct_to_ref(src)
    basis_obj = mp.orthonormal_basis_for_space(mp.PN(2, degree), mp.Simplex(2))
    funcs = list(basis_obj.functions)
    A = mp.vandermonde(funcs, nodes)
    valid = ~np.isnan(Z_clough)
    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(A[valid], Z_clough[valid])
    return model.coef_, funcs


def eval_pkdo(xy, coeffs, funcs):
    nodes = boct_to_ref(xy)
    A = mp.vandermonde(funcs, nodes)
    return A @ coeffs


os.makedirs('output', exist_ok=True)
# ==========================================
# 1. EXACT GEOMETRY DEFINITIONS
# ==========================================
S = np.sqrt(2)  # Edge length
H = S * np.sqrt(3) / 2  # Height of the triangle
Y_BASE = np.sqrt(6) / 6  # -0.408248...
Y_APEX = -2 * H / 3  # +0.8165...
X_MIN, X_MAX = -S / 2, S / 2  # -0.7071 to +0.7071
Y_MIN, Y_MAX = Y_BASE, Y_APEX


# ==========================================
# 3. SET UP GRID & STRICT MASK
# ==========================================


# def cheby_fit(x, y, z, degree=12):
#     # 1. Scale x, y to the [-1, 1] range (REQUIRED for Chebyshev)
#     # If your b_oct coords are already roughly there, just ensure strict scaling
#     x_scaled = (x - np.min(x)) / (np.max(x) - np.min(x)) * 2 - 1
#     y_scaled = (y - np.min(y)) / (np.max(y) - np.min(y)) * 2 - 1
#
#     # 2. Create the 2D Chebyshev Design Matrix
#     # This replaces your 'create_design_matrix'
#     A = cheb.chebvander2d(x_scaled, y_scaled, [degree, degree])
#
#     # 3. Solve (using the same lstsq logic)
#     coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
#     return coeffs, (np.min(x), np.max(x)), (np.min(y), np.max(y))
#
#
# def eval_cheby(x, y, coeffs, degree, x_range, y_range):
#     # Scale inputs the same way
#     x_s = (x - x_range[0]) / (x_range[1] - x_range[0]) * 2 - 1
#     y_s = (y - y_range[0]) / (y_range[1] - y_range[0]) * 2 - 1
#     return cheb.chebval2d(x_s, y_s, coeffs.reshape((degree + 1, degree + 1)))


def print_polynomial(coeffs, degree=6):
    print("\n--- FINAL ANALYTICAL EQUATION ---")
    print("f(x, y) = ")
    term_idx = 0
    equation_terms = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            if i % 2 == 0:  # Only even powers of x
                coeff = coeffs[term_idx]

                # Format the term nicely
                x_str = "" if i == 0 else f"x^{i}" if i > 1 else "x"
                y_str = "" if j == 0 else f"y^{j}" if j > 1 else "y"

                if x_str or y_str:
                    equation_terms.append(f"({coeff:+.6f} * {x_str}{y_str})")
                else:
                    equation_terms.append(f"{coeff:+.6f}")

                term_idx += 1

    # Join and print the equation
    print(" + \n".join(equation_terms).replace("x^", "**").replace("y^", "**"))


def raw_oct(reg, src):
    """
    :param reg:
    :param src:
    :return:
    """
    b_raw = reg.domain('b_raw')
    o_raw = b_raw.projs['NEA']
    b_pts = Points(src.copy(), b_raw, oid=0)  # these are actually b_raw. pts.
    c_pts = o_raw.backward(b_pts)  # native octant (no dst).
    e = reg.ellipsoid
    e_m = [e.a, e.a, e.a * (1 - e.f)]
    return c_pts.coords / e_m     # normalised


def gather_z(reg, for_xy=False):
    idx = 1 if for_xy else 2
    b_oct = reg.domain('b_oct')
    src = b_oct.warp.src
    b_pts = Points(src.copy(), b_oct, oid=0)
    c_pts = reg.project(b_pts, [b_oct, 'c_oct', 'c_ell'])
    raws = raw_oct(reg, src)
    e = reg.ellipsoid
    e_m_val = e.a if idx < 2 else e.a * (1 - e.f)
    base = (c_pts.coords[:, idx] / e_m_val) - raws[:, idx]
    return base


# def to_grid(flat_coeffs, degree, symmetric=True):
#     # 1. Start with an empty square grid of zeros
#     grid = np.zeros((degree + 1, degree + 1))
#
#     # 2. This counter tracks where we are in your 1D flat_coeffs list
#     idx = 0
#
#     # 3. Loop exactly the same way you did during the 'fit'
#     for i in range(degree + 1):
#         for j in range(degree + 1):
#             # If we used this term during the fit, grab it from the list
#             if (symmetric and i % 2 != 0) or (i + j > degree):
#                 continue
#
#             # Place the 1D coefficient into the 2D grid
#             grid[i, j] = flat_coeffs[idx]
#             idx += 1
#     return grid
def coeffs_to_grid(flat_coeffs, degree, use_asymmetry):
    grid = np.zeros((degree + 1, degree + 1))
    ptr = 0
    for i in range(degree + 1):
        for j in range(degree + 1):
            # Use the EXACT same logic as the filter in polyfit
            if i + j <= degree:
                if use_asymmetry or (i % 2 == 0):
                    grid[i, j] = flat_coeffs[ptr]
                    ptr += 1
    return grid


def calculate_jacobian_h9(xy_coef, z_coef, degree, domain, bx, by):
    x_min, x_max, y_min, y_max = domain

    # Chain Rule Scaling Factors
    sx = 2.0 / (x_max - x_min)
    sy = 2.0 / (y_max - y_min)

    c_xy = coeffs_to_grid(xy_coef, degree, use_asymmetry=True)
    c_z = coeffs_to_grid(z_coef, degree, use_asymmetry=False)

    # Derivatives in Chebyshev space
    df_du = cheb.chebder(c_xy, m=1, axis=0)
    df_dv = cheb.chebder(c_xy, m=1, axis=1)
    dh_du = cheb.chebder(c_z, m=1, axis=0)
    dh_dv = cheb.chebder(c_z, m=1, axis=1)

    xs = 2 * (bx - x_min) / (x_max - x_min) - 1
    ys = 2 * (by - y_min) / (y_max - y_min) - 1
    mxs = 2 * (-bx - x_min) / (x_max - x_min) - 1

    # Apply chain rule: d/dx = (d/du) * sx
    # Note: gx(x) = f(-x), so d/dx gx(x) = -f'(-x)
    px = np.stack([
        -cheb.chebval2d(mxs, ys, df_du) * sx,
        cheb.chebval2d(xs, ys, df_du) * sx,
        cheb.chebval2d(xs, ys, dh_du) * sx
    ], axis=1)

    py = np.stack([
        cheb.chebval2d(mxs, ys, df_dv) * sy,
        cheb.chebval2d(xs, ys, df_dv) * sy,
        cheb.chebval2d(xs, ys, dh_dv) * sy
    ], axis=1)

    J = np.linalg.norm(np.cross(px, py), axis=1)
    return J

def do_plot(src, Z_clough, Z_analytical, Z_residual, degree=6, xtra=''):
    x, y = src[:, 0], src[:, 1]

    # Filter out NaNs for the triangulation
    mask = ~np.isnan(Z_clough)
    xf, yf = x[mask], y[mask]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=200, sharey=True)

    titles = ['Original Clough-Tocher', f'Chebyshev Fit (Deg {degree})', 'Residuals (Error)']
    data = [Z_clough[mask], Z_analytical[mask], Z_residual[mask]]
    cmaps = ['gist_ncar', 'gist_ncar', 'coolwarm']

    for i, ax in enumerate(axes):
        # tripcolor is much faster for 260k points than scatter
        # shading='gouraud' provides a smooth interpolation between points
        im = ax.tripcolor(xf, yf, data[i], cmap=cmaps[i], shading='flat')

        ax.set_title(titles[i])
        ax.set_aspect('equal')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    save_path = f"output/ex0065_{degree}_{xtra}_poly.png"
    fig.savefig(save_path, dpi=600)
    print(f'Surface map saved at: {save_path}')
    plt.close(fig)  # Memory management for large grids

def polyfit(reg, degree=6, use_asymmetry=False, plot=False):
    # ==========================================
    # NEW: 2D POLYNOMIAL DESIGN MATRIX
    # ==========================================
    # b_oct = reg.domain('b_oct')
    Z_clough = gather_z(reg, use_asymmetry)

    def create_asymmetric_matrix(x, y, degree=16):
        terms = []
        for i in range(degree + 1):
            for j in range(degree + 1 - i):
                terms.append((x ** i) * (y ** j))
        return np.column_stack(terms)

    def create_design_matrix(x, y, degree=6):
        """
        Creates a matrix of polynomial terms up to 'degree'.
        Enforces horizontal symmetry by only keeping even powers of x.
        """
        terms = []
        for i in range(degree + 1):  # Power of x
            for j in range(degree + 1 - i):  # Power of y
                if i % 2 == 0:  # ONLY even powers of x (x^0, x^2, x^4...)
                    terms.append((x ** i) * (y ** j))
        return np.column_stack(terms)

    # Flatten and filter your data exactly as before
    x_flat = src[:, 0].flatten()
    y_flat = src[:, 1].flatten()
    z_flat = Z_clough.flatten()

    valid_mask = ~np.isnan(z_flat)
    x_valid = x_flat[valid_mask]
    y_valid = y_flat[valid_mask]
    z_valid = z_flat[valid_mask]

    # ==========================================
    # RUN LINEAR LEAST SQUARES (No guesses needed!)
    # ==========================================
    # print(f"Running Polynomial Least Squares...")
    # 1. Build the matrix for the valid data points
    if use_asymmetry:
        A_valid = create_asymmetric_matrix(x_valid, y_valid, degree=degree)
    else:
        A_valid = create_design_matrix(x_valid, y_valid, degree=degree)

    # 2. Solve for the exact mathematical coefficients instantly
    # coeffs, _, _, _ = np.linalg.lstsq(A_valid, z_valid, rcond=None)
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=1e-9)  # Tiny alpha to allow high fit but maintain stability
    model.fit(A_valid, z_valid)
    coeffs = model.coef_
    # ==========================================
    # CALCULATE & PLOT RESULTS
    # ==========================================
    # 3. Apply the coefficients to the full grid to get the analytical surface
    if use_asymmetry:
        A_full = create_asymmetric_matrix(x_flat, y_flat, degree=degree)
    else:
        A_full = create_design_matrix(x_flat, y_flat, degree=degree)

    # A_full = create_design_matrix(x_flat, y_flat, degree=degree)
    Z_analytical = A_full.dot(coeffs)
    Z_residual = Z_clough - Z_analytical

    if plot:
        do_plot(src, Z_clough, Z_analytical, Z_residual)

    # Calculate strict statistics on valid residuals
    valid_residuals = Z_residual[~np.isnan(Z_residual)]
    # print(f"{degree} Residual min:  {np.min(valid_residuals):.12f}")
    # print(f"{degree} Residual max:  {np.max(valid_residuals):.12f}")
    # print(f"{degree} Residual mean: {np.mean(valid_residuals):.12f}")
    print(f"{degree} Residual std:  {np.std(valid_residuals):.12f}")
    # print_polynomial(coeffs, degree=degree)
    return coeffs


def eval_poly(x, y, coeffs, degree, symmetric=True):
    x_pows = np.array([x ** i for i in range(degree + 1)])
    y_pows = np.array([y ** i for i in range(degree + 1)])
    z = np.zeros_like(x)
    idx = 0
    for i in range(degree + 1):
        if symmetric and i % 2 != 0:
            continue
        # Pull the pre-calculated x^i row
        xi = x_pows[i]
        for j in range(degree + 1 - i):
            # This is now just a simple multiply-add of two arrays
            z += coeffs[idx] * xi * y_pows[j]
            idx += 1
    return z


def plot_3d(arr):
    """Display a 3D point cloud using matplotlib"""
    xx, yy, zz = arr.coords[:, 0], arr.coords[:, 1], arr.coords[:, 2]
    cols = arr.samples
    fig = plt.figure(figsize=(10, 10), dpi=200, frameon=False)
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=30, azim=40)
    ax.set_proj_type('ortho')  # FOV = 0 deg
    ax.scatter(xx, yy, zz, marker='.', ec='none', s=1, c=cols)
    ax.set_aspect('equal', adjustable='box')
    # ax.set_axis_off()
    plt.show()
    # fig.savefig(f"output/ex0065_octahedron.png", dpi=200)


def pkdo_jacobian(src, pxy_coef, pxy_funcs, pz_coef, pz_funcs, h=1e-6):
    """Finite-difference Jacobian of the PKDO displacement field [gx, gy, gz]."""
    bx, by = src[:, 0], src[:, 1]
    src_xp = np.column_stack([bx + h, by])
    src_xm = np.column_stack([bx - h, by])
    src_yp = np.column_stack([bx, by + h])
    src_ym = np.column_stack([bx, by - h])

    gy_xp = eval_pkdo(src_xp, pxy_coef, pxy_funcs)
    gy_xm = eval_pkdo(src_xm, pxy_coef, pxy_funcs)
    gy_yp = eval_pkdo(src_yp, pxy_coef, pxy_funcs)
    gy_ym = eval_pkdo(src_ym, pxy_coef, pxy_funcs)

    gz_xp = eval_pkdo(src_xp, pz_coef, pz_funcs)
    gz_xm = eval_pkdo(src_xm, pz_coef, pz_funcs)
    gz_yp = eval_pkdo(src_yp, pz_coef, pz_funcs)
    gz_ym = eval_pkdo(src_ym, pz_coef, pz_funcs)

    # G_X(bx, by) = G_Y(-bx, by)
    gx_xp = eval_pkdo(np.column_stack([-(bx + h), by]), pxy_coef, pxy_funcs)
    gx_xm = eval_pkdo(np.column_stack([-(bx - h), by]), pxy_coef, pxy_funcs)
    gx_yp = eval_pkdo(np.column_stack([-bx, by + h]), pxy_coef, pxy_funcs)
    gx_ym = eval_pkdo(np.column_stack([-bx, by - h]), pxy_coef, pxy_funcs)

    px = np.stack([(gx_xp - gx_xm) / (2*h), (gy_xp - gy_xm) / (2*h), (gz_xp - gz_xm) / (2*h)], axis=1)
    py = np.stack([(gx_yp - gx_ym) / (2*h), (gy_yp - gy_ym) / (2*h), (gz_yp - gz_ym) / (2*h)], axis=1)
    return np.linalg.norm(np.cross(px, py), axis=1)


if __name__ == '__main__':
    rg = Registrar()
    b_raw = rg.domain('b_raw')
    b_oct = rg.domain('b_oct')
    c_oct = rg.domain('c_oct')
    c_ell = rg.domain('c_ell')
    src = b_oct.warp.src  # unwarped grid points
    bx, by = src[:, 0], src[:, 1]
    e = rg.ellipsoid
    e_m = np.array([e.a, e.a, e.a * (1 - e.f)])
    raws = raw_oct(rg, src)

    Z_clough_z = gather_z(rg, for_xy=False)

    g_pts = Points(src.copy(), b_oct, oid=0)
    j_pts = rg.project(g_pts, [b_oct, c_oct, c_ell])

    for degree in [20, 22, 24, 26, 28]:
        # --- Chebyshev ---
        z_coef, z_dom = polyfit_cheby(rg, src, degree, use_asymmetry=False)
        xy_coef, xy_dom = polyfit_cheby(rg, src, degree, use_asymmetry=True)
        Z_cheby = eval_poly_cheby(bx, by, z_coef, degree, z_dom, symmetric=True)
        vr_c = (Z_clough_z - Z_cheby)[~np.isnan(Z_clough_z)]
        jac_c = calculate_jacobian_h9(xy_coef, z_coef, degree, z_dom, bx, by)
        cj_c = np.std(jac_c) / np.mean(jac_c)

        # --- PKDO ---
        pz_coef, pz_funcs = polyfit_pkdo(rg, src, degree, for_xy=False)
        pxy_coef, pxy_funcs = polyfit_pkdo(rg, src, degree, for_xy=True)
        Z_pkdo = eval_pkdo(src, pz_coef, pz_funcs)
        vr_p = (Z_clough_z - Z_pkdo)[~np.isnan(Z_clough_z)]
        jac_p = pkdo_jacobian(src, pxy_coef, pxy_funcs, pz_coef, pz_funcs)
        cj_p = np.std(jac_p) / np.mean(jac_p)

        # PKDO 3D reconstruction error
        gx_p = eval_pkdo(np.column_stack([-bx, by]), pxy_coef, pxy_funcs)
        gy_p = eval_pkdo(src, pxy_coef, pxy_funcs)
        gz_p = eval_pkdo(src, pz_coef, pz_funcs)
        z_pts_p = Points((raws + np.stack([gx_p, gy_p, gz_p], axis=1)) * e_m, c_ell)
        max_err = np.max(np.abs(z_pts_p.coords - j_pts.coords))

        print(f"\n--- Degree {degree} ---")
        print(f"  Chebyshev  Std: {np.std(vr_c):.6f}  Max: {np.max(np.abs(vr_c)):.6f}  Coeff[J]: {cj_c:.4f}")
        print(f"  PKDO       Std: {np.std(vr_p):.6f}  Max: {np.max(np.abs(vr_p)):.6f}  Coeff[J]: {cj_p:.4f}  3D: {max_err:.1f} m")

    # Plots at last degree fitted
    Z_cheby_last = eval_poly_cheby(bx, by, z_coef, degree, z_dom, symmetric=True)
    Z_pkdo_last = eval_pkdo(src, pz_coef, pz_funcs)
    do_plot(src, Z_clough_z, Z_cheby_last, Z_clough_z - Z_cheby_last, degree=degree, xtra='cheby')
    do_plot(src, Z_clough_z, Z_pkdo_last, Z_clough_z - Z_pkdo_last, degree=degree, xtra='pkdo')
    plot_3d(z_pts_p)
    plot_3d(j_pts)
    plot_3d(Points(z_pts_p.coords - j_pts.coords, c_ell))
