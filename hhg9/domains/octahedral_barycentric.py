# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
This is 'b_oct' barycentric xy equilateral.
"""
import numpy as np
from importlib import resources

from hhg9 import Points
from numpy.typing import NDArray
# from scipy.special import eval_jacobi
from hhg9.base.composite import CompositeDomain, ComponentDomain
from hhg9.base.point_format import PointFormat
from hhg9.projections import OctantBary
from hhg9.h9 import H9K, H9O, in_scope
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator, NearestNDInterpolator


# Lateral-edge treatment in AuthalicWarp.do / .undo (see libhex9
# docs/warp-retrain-brief.md; F6 fix 2026-06-11, F6 retrain 2026-06-12).
#
#   'raw' (default) — no edge treatment: the trained field speaks for
#       itself. The F6-retrained WGS84_l5_warp_data.npz is edge-tangent
#       (on-edge vertex deltas slide along the seam, zero normal component)
#       and the loader makes the CT interpolant tangent identically (ghost
#       padding + edge-tangent gradient projection below), so raw mode is
#       exact at the seams: no slivers, no catchment splitting, both octant
#       frames agree about seam points to FP precision.
#   'feather' — pre-retrain workaround: delta scaled by a smoothstep ramp to
#       identity on the edge over line-value LATERAL_EDGE_FEATHER_W. Correct
#       for the OLD zero-pinned field only. DO NOT use with the F6 field:
#       it compresses the field's legitimate km-scale tangential seam slide
#       into the 350 m ramp (massive shear).
#   'bypass'  — the legacy hard identity band (|line value| < 1e-7); the
#       original needle/sliver bug, kept for historic comparison only.
#
# Mode is read from the H9_WARP_EDGE environment variable, same as libhex9.
import os as _os
LATERAL_EDGE_MODE = _os.environ.get("H9_WARP_EDGE", "raw")
if LATERAL_EDGE_MODE not in ("feather", "bypass", "raw"):
    raise ValueError(f"H9_WARP_EDGE must be feather|bypass|raw, got {LATERAL_EDGE_MODE!r}")
LATERAL_EDGE_BYPASS = (LATERAL_EDGE_MODE == "bypass")
LATERAL_EDGE_FEATHER_W = 1e-4   # line-value width ≈ 350 m half-zone


def _edge_feather_scale(xy):
    """Smoothstep ramp per point: 0 exactly on either lateral edge of the
    mode-0 frame (y = ±R3·x − Ẇ) → 1 beyond line-value LATERAL_EDGE_FEATHER_W.
    Width is measured on the line value (2× the perpendicular distance)."""
    R3 = H9K.radical.R3
    Ẇ = H9K.Ẇ
    s = np.ones(xy.shape[0])
    for f in (np.abs(xy[:, 1] - R3 * xy[:, 0] + Ẇ),
              np.abs(xy[:, 1] + R3 * xy[:, 0] + Ẇ)):
        t = np.clip(f / LATERAL_EDGE_FEATHER_W, 0.0, 1.0)
        s *= t * t * (3.0 - 2.0 * t)
    return s


def _jacobi_all(x, n_max, alpha):
    """P_q^(alpha, 0)(x) for q = 0..n_max via 3-term recurrence. Returns (n_max+1, N).
    Vectorised in x; O(n_max) Python iterations with numpy inner ops.
    """
    N = len(x)
    out = np.empty((n_max + 1, N))
    out[0] = 1.0
    if n_max >= 1:
        out[1] = 0.5 * ((alpha + 2) * x + alpha)
    for q in range(2, n_max + 1):
        ab = 2 * q + alpha
        c1 = ab * (ab - 2)
        c2 = alpha * alpha
        c3 = 2 * (q + alpha - 1) * (q - 1) * ab
        c4 = 2 * q * (q + alpha) * (ab - 2)
        out[q] = ((ab - 1) * (c2 + c1 * x) * out[q - 1] - c3 * out[q - 2]) / c4
    return out


def _pkdo_vandermonde(xy, degree):
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


def _o_pkdo_vandermonde(xy, degree):
    """Vandermonde matrix for L2-orthonormal PKDO basis on b_oct triangle.
    C_{p,q} = sqrt((2p+1)(p+q+1)/2).  O(degree) Python loops, all inner ops vectorised.
    """
    x, y = xy[:, 0], xy[:, 1]
    r = np.sqrt(2) * x + (np.sqrt(6) / 3) * y - 1.0 / 3.0
    s = (-2.0 * np.sqrt(6) / 3) * y - 1.0 / 3.0
    N = len(r)
    n_basis = (degree + 1) * (degree + 2) // 2
    V = np.zeros((N, n_basis))
    s_safe = np.minimum(s, 1.0 - 1e-14)
    a = 2.0 * (1.0 + r) / (1.0 - s_safe) - 1.0

    Jp_all = _jacobi_all(a, degree, 0)          # (degree+1, N)  P_p^(0,0)(a)
    half_oms = (1.0 - s_safe) / 2.0             # (N,)
    hp = np.ones(N)                             # (1-s)/2)^p, starts at p=0

    col = 0
    for p in range(degree + 1):
        max_q = degree - p
        Jph = Jp_all[p] * hp                             # (N,)
        Jq_all = _jacobi_all(s, max_q, 2 * p + 1)       # (max_q+1, N)
        norms = np.sqrt((2 * p + 1) * (p + np.arange(max_q + 1) + 1) / 2.0)  # (max_q+1,)
        V[:, col:col + max_q + 1] = (norms[:, None] * Jq_all * Jph).T
        col += max_q + 1
        hp = hp * half_oms                               # accumulate (1-s)/2)^p
    return V


class WarpTolerance:
    """Named Newton-Raphson convergence tolerances (barycentric XY units).
    1 unit ≈ 2.5 × 10⁷ m, so: SUB_MM ≈ 0.25 mm, ROUGH ≈ 25 cm."""
    MACH = 1e-17  # machine ε — exhausts all iterations; for validation
    FINE = 1e-15  # ~0.25 mm  — recommended production default
    NORM = 1e-14  # ~0.25 mm  — recommended production default


class AuthalicWarp:
    def __init__(self, file_name=None, interp='ct', tolerance=WarpTolerance.FINE):
        # Load Data
        self.tolerance = tolerance
        if file_name is None:
            return
        self.file_name = file_name

        if interp == 'pkdo':
            repo = np.load(file_name, allow_pickle=True)
            degree = int(repo['degree'])
            cx, cy = repo['cx'], repo['cy']
            self.fwd_dx = lambda xy, _d=degree, _cx=cx: _pkdo_vandermonde(xy, _d) @ _cx
            self.fwd_dy = lambda xy, _d=degree, _cy=cy: _pkdo_vandermonde(xy, _d) @ _cy
            # NR inverse: identity seed is safe — max displacement ~0.009 b_oct units
            self.inv_linear = lambda pts: pts.copy()
            self.inv_nearest = lambda pts: pts.copy()
            return

        # warp below.
        repo = np.load(file_name, allow_pickle=True)
        self.src = repo['source_pts']  # Regular Grid (a_p)
        self.dst = repo['target_pts']  # Deformed Grid (x_prime)
        repo.close()

        # --- GHOST ROW PADDING (Equator Stabilization) ---
        diff = self.dst - self.src
        Y_EQ = np.sqrt(6.0) / 6.0

        # Grab points within a small margin of the equator (e.g., top 5% of the triangle)
        # Adjust eq_base 0.05 if your grid spacing is wider/narrower.
        # Strict band (exclude y == Y_EQ): the equator-line points reflect onto
        # themselves with identical (dx, dy) — Y-mirror of (dx, 0) is (dx, 0) —
        # so including them generates informationally-redundant duplicates and
        # leaves Qhull's tie-break to pick arbitrary diagonals on rhombi that
        # straddle the equator. Excluding them gives a canonical CT.
        eq_base = 0.05
        delta_y = Y_EQ - self.src[:, 1]
        eq_band_mask = (delta_y > 0.0) & (delta_y < eq_base)

        if np.any(eq_band_mask):
            ghost_src = self.src[eq_band_mask].copy()
            # Mirror Y-coordinate across the equator line
            ghost_src[:, 1] = 2.0 * Y_EQ - ghost_src[:, 1]

            ghost_diff = diff[eq_band_mask].copy()
            # Mirror the Y-displacement (if it stretches North on one side,
            # it stretches South on the other)
            ghost_diff[:, 1] *= -1.0

            # Note: X-displacement (longitudinal shift) remains identical
            # across the equator, so we don't flip diff[:, 0]

            padded_src = np.vstack([self.src, ghost_src])
            padded_diff = np.vstack([diff, ghost_diff])
        else:
            padded_src = self.src
            padded_diff = diff

        # --- LATERAL GHOST PADDING (Seam tangency, F6) ---
        # Same construction as the equatorial ghost row, applied to the two
        # lateral edges: a thin halo of points is reflected across each edge
        # line with the delta's edge-normal component negated (tangential
        # preserved). The trained vertex data is already edge-tangent, but
        # without these ghosts the CT gradient estimation couples the steep
        # near-corner tangential slide into an edge-normal bulge BETWEEN the
        # on-edge vertices (~135 m near the apex, ~48 m cross-frame seam
        # disagreement on the F6 retrain). Mirror-symmetric input makes the
        # interpolant tangent on the edge by construction.
        # Strict band (exclude on-edge points): they reflect onto themselves,
        # and the duplicates leave Qhull's tie-break to pick arbitrary
        # diagonals (see equator note above).
        _R3 = H9K.radical.R3
        _Ẇ = H9K.Ẇ
        GHOST_BAND = 0.05
        lat_src, lat_diff = [padded_src], [padded_diff]
        for sgn in (1.0, -1.0):
            # Edge line: sgn·√3·x − y − Ẇ = 0; unit normal points out of the
            # triangle, signed distance is negative inside.
            n_hat = np.array([sgn * _R3, -1.0]) / 2.0
            s_dist = self.src @ n_hat - _Ẇ / 2.0
            band = (s_dist > -GHOST_BAND) & (s_dist < -1e-12)
            if not np.any(band):
                continue
            g_src = self.src[band] - 2.0 * s_dist[band, None] * n_hat
            d_n = diff[band] @ n_hat
            g_diff = diff[band] - 2.0 * d_n[:, None] * n_hat
            lat_src.append(g_src)
            lat_diff.append(g_diff)

        # --- CORNER ORBIT PADDING (Seam tangency, F6) ---
        # At each triangle corner the two adjacent edge mirrors meet at 60°
        # and generate a dihedral group (3 mirror lines + ±120° rotations).
        # The single-mirror bands above fill only the two wedges adjacent to
        # the corner; the three back wedges stay empty, so the CT gradient
        # estimate at near-corner vertices is still asymmetric — this is
        # where the residual normal bulge lives. Filling the full orbit
        # (rotations ±120° and the third mirror, all about the corner) makes
        # each corner disc exactly dihedral-symmetric. The orbit images are
        # compositions of valid seam continuations, so they are correct
        # field continuations, and each lands in a wedge the bands do not
        # touch (no duplicate points).
        tr_, vf_, vc_ = H9K.limits.TR, H9K.limits.VF, H9K.limits.VC
        c120, s120 = -0.5, _R3 / 2.0
        rot_p = np.array([[c120, -s120], [s120, c120]])   # +120°
        rot_m = rot_p.T                                   # −120°
        # (corner, direction of the adjacent edge line chosen so that the
        # OTHER edge lies at +60°; then +120° rotation of it gives the third
        # mirror line of the dihedral group, never one of the edges)
        corner_edges = (
            (np.array([0.0, vf_]),  np.array([tr_, vc_ - vf_])),   # apex: 60°
            (np.array([tr_, vc_]),  np.array([1.0, 0.0])),         # eq R: 0°
            (np.array([-tr_, vc_]), np.array([tr_, vf_ - vc_])),   # eq L: 120°
        )
        # strict interior: off all three boundary lines
        d_eq = np.abs(self.src[:, 1] - vc_)
        d_lr = np.abs(self.src[:, 1] - _R3 * self.src[:, 0] + _Ẇ)
        d_ll = np.abs(self.src[:, 1] + _R3 * self.src[:, 0] + _Ẇ)
        interior = (d_eq > 1e-12) & (d_lr > 1e-12) & (d_ll > 1e-12)
        for corner, u_a in corner_edges:
            u_a = u_a / np.linalg.norm(u_a)
            d3 = rot_p @ u_a                  # third mirror line direction
            m3 = 2.0 * np.outer(d3, d3) - np.eye(2)
            near = interior & (np.linalg.norm(self.src - corner, axis=1)
                               < GHOST_BAND)
            if not np.any(near):
                continue
            rel = self.src[near] - corner
            for T in (rot_p, rot_m, m3):
                lat_src.append(corner + rel @ T.T)
                lat_diff.append(diff[near] @ T.T)
        padded_src = np.vstack(lat_src)
        padded_diff = np.vstack(lat_diff)

        # 1. Forward Engine (Cubic / Smooth)
        # We model the *displacement* (diff) rather than absolute position for better stability
        diff = self.dst - self.src
        # Nearest-neighbour fallbacks are useful even for Clough-Tocher:
        # CT returns NaN outside the convex hull, and the inverse solver may
        # step infinitesimally outside near seams/boundaries.
        nn_dx = NearestNDInterpolator(self.src, diff[:, 0])
        nn_dy = NearestNDInterpolator(self.src, diff[:, 1])

        if interp != 'linear':
            # tol=1e-12 to converge the Bell-Sibson gradient sweep tightly.
            # scipy's default tol=1e-6 leaves ~600 µm of unconverged "noise"
            # in the CT output (well above the 1 µm ground-accuracy target);
            # tight tol pins the warp at FP-eps and lets the C++ port match
            # Python bit-near-exactly.
            ct_dx = CloughTocher2DInterpolator(padded_src, padded_diff[:, 0],
                                               tol=1e-12, maxiter=2000)
            ct_dy = CloughTocher2DInterpolator(padded_src, padded_diff[:, 1],
                                               tol=1e-12, maxiter=2000)
            # Exposed for the C-port gradient export (see libhex9
            # docs/warp-port-brief-f6-cside.md): .grad carries the final
            # symmetrised, edge-tangent-projected vertex gradients.
            self.ct_dx, self.ct_dy = ct_dx, ct_dy

            # --- EDGE-TANGENT GRADIENT PROJECTION (Seam tangency, F6) ---
            # CT restricted to a triangulation edge is the cubic Hermite of
            # the endpoint values and ALONG-EDGE gradient components only.
            # The on-edge vertex deltas are exactly edge-tangent, so zeroing
            # the along-edge derivative of the delta's edge-normal component
            # at every boundary vertex makes the interpolated delta tangent
            # along the entire edge identically. The ghost padding above
            # improves near-edge gradient quality, but a finite pad cannot
            # make the global gradient estimate exactly symmetric (residual
            # ~1e-5 units of normal bulge near the corners without this).
            # X-mirror symmetrisation of the gradients first makes the two
            # frames' on-edge tangential cubics agree exactly, so adjacent
            # octants concur about where seam points go. C1 continuity is
            # preserved: CT is C1 for any choice of vertex gradients.
            from scipy.spatial import cKDTree
            n_real = len(self.src)
            g_dx = ct_dx.grad   # (n_padded, 1, 2), mutable, used at eval
            g_dy = ct_dy.grad
            _, mir = cKDTree(self.src).query(
                np.column_stack([-self.src[:, 0], self.src[:, 1]]), k=1)
            # dx antisymmetric in x → (∂dx/∂x even, ∂dx/∂y odd);
            # dy symmetric in x   → (∂dy/∂x odd,  ∂dy/∂y even).
            g_dx[:n_real, 0, 0] = 0.5 * (g_dx[:n_real, 0, 0] + g_dx[mir, 0, 0])
            g_dx[:n_real, 0, 1] = 0.5 * (g_dx[:n_real, 0, 1] - g_dx[mir, 0, 1])
            g_dy[:n_real, 0, 0] = 0.5 * (g_dy[:n_real, 0, 0] - g_dy[mir, 0, 0])
            g_dy[:n_real, 0, 1] = 0.5 * (g_dy[:n_real, 0, 1] + g_dy[mir, 0, 1])

            # Boundary lines: (unit normal, unit tangent, on-line mask).
            x_, y_ = self.src[:, 0], self.src[:, 1]
            lines = [
                (np.array([_R3, -1.0]) / 2.0, np.array([1.0, _R3]) / 2.0,
                 np.abs(y_ - _R3 * x_ + _Ẇ) < 1e-9),                # right
                (np.array([-_R3, -1.0]) / 2.0, np.array([1.0, -_R3]) / 2.0,
                 np.abs(y_ + _R3 * x_ + _Ẇ) < 1e-9),                # left
                (np.array([0.0, 1.0]), np.array([1.0, 0.0]),
                 np.abs(y_ - vc_) < 1e-9),                          # equator
            ]
            corner_pts = np.array([[0.0, vf_], [tr_, vc_], [-tr_, vc_]])
            is_corner = np.zeros(n_real, dtype=bool)
            for cp in corner_pts:
                is_corner |= np.linalg.norm(self.src - cp, axis=1) < 1e-9
            for n_hat, t_hat, on in lines:
                idx = np.flatnonzero(on & ~is_corner)
                # s = t̂·∇f_n; subtract s·t̂ from ∇f_n, leaving ∇f_t intact.
                s = (n_hat[0] * (g_dx[idx, 0] @ t_hat)
                     + n_hat[1] * (g_dy[idx, 0] @ t_hat))
                g_dx[idx, 0] -= np.outer(s * n_hat[0], t_hat)
                g_dy[idx, 0] -= np.outer(s * n_hat[1], t_hat)
            # Corners sit on two boundary lines: joint null-space projection
            # of both tangency constraints on (∇dx, ∇dy) ∈ R⁴.
            line_vals = [
                lambda p: p[1] - _R3 * p[0] + _Ẇ,    # right lateral
                lambda p: p[1] + _R3 * p[0] + _Ẇ,    # left lateral
                lambda p: p[1] - vc_,                # equator
            ]
            for cp in corner_pts:
                rows = [np.concatenate([n_hat[0] * t_hat, n_hat[1] * t_hat])
                        for (n_hat, t_hat, _), lv in zip(lines, line_vals)
                        if abs(lv(cp)) < 1e-9]
                ki = np.flatnonzero(
                    np.linalg.norm(self.src - cp, axis=1) < 1e-9)
                if len(rows) < 2 or len(ki) == 0:
                    continue
                A = np.vstack(rows)
                k = ki[0]
                G = np.concatenate([g_dx[k, 0], g_dy[k, 0]])
                G -= A.T @ np.linalg.solve(A @ A.T, A @ G)
                g_dx[k, 0], g_dy[k, 0] = G[:2], G[2:]

            def fwd_dx(xy):
                d = ct_dx(xy)
                # Only fallback if the output is NaN AND the input was a valid coordinate
                m = np.isnan(d) & np.isfinite(xy).all(axis=1)
                if np.any(m):
                    d[m] = nn_dx(xy[m])
                return d

            def fwd_dy(xy):
                d = ct_dy(xy)
                m = np.isnan(d) & np.isfinite(xy).all(axis=1)
                if np.any(m):
                    d[m] = nn_dy(xy[m])
                return d

            self.fwd_dx = fwd_dx
            self.fwd_dy = fwd_dy

        else:
            lin_dx = LinearNDInterpolator(self.src, diff[:, 0])
            lin_dy = LinearNDInterpolator(self.src, diff[:, 1])

            def fwd_dx(xy):
                d = lin_dx(xy)
                m = np.isnan(d) & np.isfinite(xy).all(axis=1)
                if np.any(m):
                    d[m] = nn_dx(xy[m])
                return d

            def fwd_dy(xy):
                d = lin_dy(xy)
                m = np.isnan(d) & np.isfinite(xy).all(axis=1)
                if np.any(m):
                    d[m] = nn_dy(xy[m])
                return d

            self.fwd_dx = fwd_dx
            self.fwd_dy = fwd_dy

        # 2. Inverse Guesser (Linear + Nearest Backup)
        # This provides the "seed" for the solver.
        self.inv_linear = LinearNDInterpolator(self.dst, self.src)
        self.inv_nearest = NearestNDInterpolator(self.dst, self.src)

    def _edge_undo(self, target, n_iter=52):
        """
        1D bisection inverse for b_raw points on a lateral triangle edge.
        The warp is boundary-preserving, so b_oct is on the same lateral edge.
        Parameterise the edge as (t, R3*t - Ẇ) for t ≥ 0 (right) or t ≤ 0 (left).
        Bisection converges to machine precision in ~52 steps.
        Works in the canonical DOWN orientation (mo=0).
        Returns b_oct for each target.
        """
        Ẇ = H9K.Ẇ
        R3 = H9K.radical.R3
        seeds = np.empty_like(target)

        # Find max |x| along each lateral edge in src training data
        src = self.src
        on_right = np.abs(src[:, 1] - R3 * src[:, 0] + Ẇ) < 1e-6
        on_left  = np.abs(src[:, 1] + R3 * src[:, 0] + Ẇ) < 1e-6
        x_max_right = np.max(src[on_right, 0]) if np.any(on_right) else 0.01
        x_max_left  = np.min(src[on_left,  0]) if np.any(on_left)  else -0.01

        for i in range(target.shape[0]):
            tx, ty = target[i, 0], target[i, 1]
            # Determine which lateral edge (right: x ≥ 0, left: x < 0)
            if tx >= 0:
                lo, hi = 0.0, x_max_right
                def do_x(t):
                    return self.do(np.array([[t, R3 * t - Ẇ]]), mo=0)[0, 0]
                t_tgt = tx
            else:
                lo, hi = x_max_left, 0.0
                def do_x(t):
                    return self.do(np.array([[t, -R3 * t - Ẇ]]), mo=0)[0, 0]
                t_tgt = tx

            # 1D bisection: find t s.t. do_x(t) = t_tgt
            for _ in range(n_iter):
                mid = (lo + hi) * 0.5
                if do_x(mid) < t_tgt:
                    lo = mid
                else:
                    hi = mid

            t = (lo + hi) * 0.5
            if tx >= 0:
                seeds[i] = [t, R3 * t - Ẇ]
            else:
                seeds[i] = [t, -R3 * t - Ẇ]

        return seeds

    def do(self, pts, mo=0):
        """ Forward Warp (Precise Cubic) """
        xy = np.array(pts, dtype=np.float64)  # Force Copy
        mode = 1.0 if mo == 0 else -1.0
        xy[:, 1] *= mode

        # Lateral-edge treatment (see LATERAL_EDGE_MODE at module top).
        _R3 = H9K.radical.R3
        _Ẇ  = H9K.Ẇ
        on_right = np.abs(xy[:, 1] - _R3 * xy[:, 0] + _Ẇ) < 1e-7
        on_left  = np.abs(xy[:, 1] + _R3 * xy[:, 0] + _Ẇ) < 1e-7
        edge_mask = on_right | on_left

        # Interpolate displacement
        dx = self.fwd_dx(xy)
        dy = self.fwd_dy(xy)
        if LATERAL_EDGE_MODE == "feather":
            s = _edge_feather_scale(xy)
            # where the ramp is 0 the delta is identically 0 — also guards
            # NaN interpolants exactly on the hull edge (0·NaN otherwise)
            dx = np.where(s > 0.0, dx * s, 0.0)
            dy = np.where(s > 0.0, dy * s, 0.0)

        res = xy + np.stack([dx, dy], axis=1)
        if LATERAL_EDGE_BYPASS and np.any(edge_mask):
            res[edge_mask] = xy[edge_mask]
        res[:, 1] *= mode
        return res

    def set_tolerance(self, tolerance: float):
        """Set the default Newton convergence tolerance for undo()."""
        self.tolerance = tolerance
        return self

    def undo(self, pts, mo=0, iterations=25):
        """Precise Inverse Warp (Newton-Raphson), fixed iteration count."""
        target_all = np.array(pts, dtype=np.float64)
        mode = 1.0 if mo == 0 else -1.0
        target_all[:, 1] *= mode

        # Guard: scipy's KDTree query requires finite inputs.
        finite_mask = np.isfinite(target_all).all(axis=1)
        if not np.all(finite_mask):
            out = np.full_like(target_all, np.nan)
            if not np.any(finite_mask):
                out[:, 1] *= mode
                return out
            target = target_all[finite_mask]
        else:
            out = None
            target = target_all

        # --- LATERAL EDGE DETECTION ---
        # b_raw points on a lateral edge (y − R3·|x| = −Ẇ) must map to b_oct on the
        # same edge (the warp is boundary-preserving). Newton oscillates on these points
        # because its 2D step repeatedly crosses the edge and gets y-clamped back, leaving
        # x wrong. Use 1D bisection along the edge instead — it converges exactly.
        R3 = H9K.radical.R3
        Ẇ = H9K.Ẇ
        on_right_edge = np.abs(target[:, 1] - R3 * target[:, 0] + Ẇ) < 1e-7
        on_left_edge  = np.abs(target[:, 1] + R3 * target[:, 0] + Ẇ) < 1e-7
        edge_mask = on_right_edge | on_left_edge

        # --- COARSE GUESS ---
        # guess = self.inv_nearest(target)
        guess = self.inv_linear(target)
        # inv_linear returns NaN outside its convex hull (e.g. near poles).
        nan_mask = np.isnan(guess[:, 0])
        if np.any(nan_mask):
            tgt_nan = target[nan_mask]
            tgt_nan_finite = np.isfinite(tgt_nan).all(axis=1)
            if np.any(tgt_nan_finite):
                nan_idx = np.flatnonzero(nan_mask)
                tgt_finite = tgt_nan[tgt_nan_finite]
                nan_on_edge = edge_mask[nan_mask][tgt_nan_finite]
                seeds = self.inv_nearest(tgt_finite)
                if LATERAL_EDGE_MODE in ("bypass", "feather") and np.any(nan_on_edge):
                    seeds[nan_on_edge] = tgt_finite[nan_on_edge]  # identity on edge
                guess[nan_idx[tgt_nan_finite]] = seeds

        # Exact-on-edge points: under 'bypass' identity is the forced answer;
        # under 'feather' identity is the EXACT root (the feathered delta is
        # identically zero on the edge), so seed with it and skip Newton.
        # Under 'raw' Newton treats edge points like any other — honestly.
        if LATERAL_EDGE_MODE in ("bypass", "feather") and np.any(edge_mask):
            guess[edge_mask] = target[edge_mask]

        # --- ITERATIVE POLISH ---
        curr = guess.copy()
        edge_done = (edge_mask.copy() if LATERAL_EDGE_MODE in ("bypass", "feather")
                     else np.zeros_like(edge_mask))

        def _eval_d(p):
            """Forward delta at p — the function Newton inverts. The feather
            scale moves with the iterate, so it lives inside the evaluation
            (and therefore inside the finite-difference Jacobian)."""
            dxe = self.fwd_dx(p)
            dye = self.fwd_dy(p)
            if LATERAL_EDGE_MODE == "feather":
                se = _edge_feather_scale(p)
                dxe = np.where(se > 0.0, dxe * se, 0.0)
                dye = np.where(se > 0.0, dye * se, 0.0)
            return dxe, dye

        for _ in range(iterations):
            dx, dy = _eval_d(curr)

            # If the guess drifted off the interpolation hull, snap it back.
            bad_mask = np.isnan(dx)
            if np.any(bad_mask):
                curr[bad_mask] = self.inv_nearest(target[bad_mask])
                dx, dy = _eval_d(curr)

            error = curr + np.stack([dx, dy], axis=1) - target

            # Full Newton step: (I + J_D) · delta = -error, solved via 2×2 Cramer.
            # Two extra evaluations: perturb curr in x then y to estimate J_D.
            _h = 1e-7
            cp_x = curr.copy()
            cp_x[:, 0] += _h
            cp_y = curr.copy()
            cp_y[:, 1] += _h
            dx_px, dy_px = _eval_d(cp_x)
            dx_py, dy_py = _eval_d(cp_y)
            a = 1.0 + (dx_px - dx) / _h  # 1 + ∂dx/∂x
            b = (dx_py - dx) / _h        # ∂dx/∂y
            c = (dy_px - dy) / _h        # ∂dy/∂x
            d = 1.0 + (dy_py - dy) / _h  # 1 + ∂dy/∂y
            det = a * d - b * c
            ex, ey = error[:, 0], error[:, 1]
            safe = np.abs(det) > 1e-12
            denom = np.where(safe, det, 1.0)
            delta_x = np.where(safe, -(d * ex - b * ey) / denom, -ex)
            delta_y = np.where(safe, -(a * ey - c * ex) / denom, -ey)
            if np.any(edge_done):
                delta_x[edge_done] = 0.0
                delta_y[edge_done] = 0.0
            curr += np.stack([delta_x, delta_y], axis=1)

            # Handle out-of-scope or non-finite Newton steps.
            # NaN coordinates fail in_scope (NaN comparisons → False).
            # Near-polar seam points correctly live on the lateral triangle edge;
            # snapping them to inv_nearest (the apex) breaks convergence.
            # Strategy: NaN → inv_nearest fallback; merely out-of-scope → project
            # onto the DOWN triangle boundary so Newton can continue from there.
            ẋ = H9K.radical.R3 * curr[:, 0]
            bad = ~in_scope(ẋ, curr[:, 1], 0)
            if np.any(bad):
                nan_bad = bad & ~np.isfinite(curr).all(axis=1)
                oob_bad = bad & ~nan_bad
                if np.any(nan_bad):
                    curr[nan_bad] = self.inv_nearest(target[nan_bad])
                if np.any(oob_bad):
                    _xd = ẋ[oob_bad]
                    _y = curr[oob_bad, 1]
                    _W = H9K.Ẇ
                    _y = np.maximum(_y, _xd - _W)   # right lateral edge: y - ẋ >= -Ẇ
                    _y = np.maximum(_y, -_xd - _W)  # left lateral edge:  y + ẋ >= -Ẇ
                    _y = np.minimum(_y, H9K.VC)     # ceiling
                    curr[oob_bad, 1] = _y

        curr[:, 1] *= mode

        if out is not None:
            out[finite_mask] = curr
            return out

        return curr


class OctantBarycentric(ComponentDomain):
    """
    This a 2D side of an Octant.
    Validity should be easy enough since we have the 3 points that define it.
    """

    def __init__(self, registrar, dom, oid):
        # sign = H9O.oid_cmp[oid]
        self.oid = oid
        face = H9O.oid_str[oid]
        b_sig = f'{dom.name}:{face}'
        thet = H9O.oid_tht[oid]
        self.mo = H9O.oid_mo[oid]
        # mo_str = 'V' if self.mo == 0 else 'Λ'
        super().__init__(registrar, b_sig, dom,  oid, 2)
        # .th mirrors b_raw's OctantBaryRaw so the warp-free w_oct<->b_oct
        # OctantBraw lift can build an identical orient matrix from this side.
        self.th = (thet % 6) * np.pi / 3.

        edges = H9O.edges_by_id[oid]
        # self.oc = np.array([ed+mo_str for ed in edges], dtype='U3')

    def valid(self, pts: NDArray) -> NDArray:
        """
        Return an array of bools according to the validity criterion
        :param pts: set of 2d Euclidean points
        """
        return in_scope(H9K.radical.R3 * pts[..., 0], pts[..., 1], self.mode)


class OctahedralBarycentric(CompositeDomain):
    """
    Basic octahedral-2d properties and methods.
    In terms of what this does - it now only applies warp to b_raw.
    It sits between b_raw->[b_oct]->n_oct and is also the final address.
    It doesn't do any rotation, nor any matrix work.
    """

    def __init__(self, registrar):
        super().__init__(registrar, 'b_oct', 2)
        self.h9map = {}
        self.h9 = registrar.format('h9')
        self.h9.composite = self

        self.sides = {}
        self.projs = {}

        # ── Warp state (lazy) ────────────────────────────────────────────
        # warp_enabled is the cheap on/off TOGGLE both engines read (no_warp /
        # use_warp flip it). The AuthalicWarp object is built LAZILY on first
        # access to the `warp` property (the CT/Newton build is expensive), so
        # the libhex9 path — which never touches `.warp` — pays no init cost.
        # _warp_spec = (file, method) is what to build when the Python path needs it.
        self.warp_enabled = True
        self._warp_obj = None
        self._warp_spec = None

        # ── Engine state ─────────────────────────────────────────────────
        # Default to the libhex9 backend for g_gcd↔b_oct when it loads;
        # no_lib()/use_lib() flip lib_enabled at runtime. Pure Python stays
        # fully available (blue/green, regression, intermediate inspection).
        try:
            from hhg9.accel import backend as _accel_backend
            self._lib = _accel_backend()
        except Exception:
            self._lib = None
        self.lib_enabled = self._lib is not None

        # c_oct = registrar.domain('c_oct')
        b_raw = registrar.domain('b_raw')   # geometric foundation — owns the matrices

        oid_s = H9O.oid_str
        for oid in range(8):
            face = oid_s[oid]
            ob = OctantBarycentric(registrar, self, oid)
            self.sides[face] = ob
            oc = b_raw.sides[face]
            proj = OctantBary(registrar, face, oc.name, ob.name)
            # proj.matrix = b_raw.projs[face].matrix   # borrow from b_raw, no recomputation
            self.projs[face] = proj

        # self.rot90_idx = b_raw.rot90_idx   # same rotation-quadrant index
        # Warp source: the F6-retrained authalic blob, built LAZILY (see the
        # `warp` property). The legacy pkdo / 'linear' warps are fossils — removed.
        pkg = "hhg9.data"
        layer = 5
        ell = registrar.ellipsoid_name
        data = resources.files(pkg).joinpath(f"{ell}_l{layer}_warp_data.npz")
        if not data.exists():
            data = resources.files(pkg).joinpath(f"WGS84_l{layer}_warp_data.npz")
        self._warp_spec = (data, None)

        registrar.register_bridge(['c_oct', 'b_raw', 'b_oct'])
        registrar.register_bridge(['g_gcd', 'b_raw', 'b_oct'])

        # When libhex9 is present, register the direct g_gcd↔b_oct lib resolver
        # AHEAD of the bridge above (a direct projection beats a bridge in
        # _check_chain). It honours no_lib() by falling through to that exact
        # bridge, so the pure-Python path stays the canonical reference.
        if self._lib is not None:
            from hhg9.projections.gcd_boct_lib import GcdBoctLib
            GcdBoctLib(registrar)

    @property
    def warp(self):
        """The AuthalicWarp, or None when disabled. Built lazily on first use
        (the CT/Newton solver is expensive) so the lib path never pays for it.
        Read live by BrawBoct on the pure-Python path."""
        if not self.warp_enabled or self._warp_spec is None:
            return None
        if self._warp_obj is None:
            file, method = self._warp_spec
            self._warp_obj = AuthalicWarp(file, method)
            self._warp_obj.domain = self
        return self._warp_obj

    @property
    def warp_file(self):
        """The configured warp source WITHOUT building the warp (cheap)."""
        return self._warp_spec[0] if self._warp_spec else None

    def set_warp(self, warp_file=None, method=None):
        """Configure the warp source and enable it (built lazily on first use)."""
        if warp_file is not None:
            self._warp_spec = (warp_file, method)
        self._warp_obj = None        # force rebuild against the new spec
        self.warp_enabled = True
        return self

    def use_warp(self):
        """Enable the warp — AK + warp (b_oct). Symmetric with no_warp()."""
        self.warp_enabled = True
        return self

    def no_warp(self):
        """Disable the warp — AK-only / BRAW (metric & dev comparisons)."""
        self.warp_enabled = False
        self._warp_obj = None
        return self

    def use_lib(self):
        """Use the libhex9 backend for g_gcd↔b_oct (if it loaded)."""
        self.lib_enabled = self._lib is not None
        return self

    def no_lib(self):
        """Use the pure-Python engine (exposes intermediates; blue/green ref)."""
        self.lib_enabled = False
        return self

    def decode(self, addr):
        """Decode octahedral coordinates into a point"""
        return self.h9.revert(addr)


    @classmethod
    def valid(cls, pts: NDArray) -> NDArray:
        """
        Return an array of bools according to the validity criterion
        :param pts: set of 2d Euclidean points
        """
        from hhg9 import Points
        if isinstance(pts, Points):
            _, mode = pts.cm()
            x, y = pts.coords[:, 0], pts.coords[:, 1]
            return in_scope(H9K.radical.R3 * x, y, mode)
        else:
            raise TypeError('pts must be a Points object')

    def register_format(self, af: PointFormat):
        """Decorator to register an AddressFormat for each component."""
        super().register_format(af)
        for side in self.sides:
            self.sides[side].register_format(af)
