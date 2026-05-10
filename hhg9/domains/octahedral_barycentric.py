# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
This is 'b_oct' barycentric xy equilateral.
"""
import numpy as np
from importlib import resources
from numpy.typing import NDArray
# from scipy.special import eval_jacobi
from hhg9.base.composite import CompositeDomain, ComponentDomain
from hhg9.base.point_format import PointFormat
from hhg9.projections import OctantBary
from hhg9.h9 import H9K, H9O, in_scope
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator, NearestNDInterpolator


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
    FINE = 1e-14  # ~0.25 mm  — recommended production default
    # Not much to lose here: AK is the meat grinder.
    # GOOD = 1e-9   # ~25 cm — < 1M
    # OKAY = 1e-6   # ~25 m — < 10M

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

        # --- GHOST ROW PADDING (Equator Stabilization) ---
        diff = self.dst - self.src
        Y_EQ = np.sqrt(6.0) / 6.0

        # Grab points within a small margin of the equator (e.g., top 5% of the triangle)
        # Adjust eq_base 0.05 if your grid spacing is wider/narrower.
        eq_base = 0.05
        eq_band_mask = (Y_EQ - self.src[:, 1]) < eq_base

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
        # --------------------------------------------------

        # 1. Forward Engine (Cubic / Smooth)
        # We model the *displacement* (diff) rather than absolute position for better stability
        diff = self.dst - self.src
        # Nearest-neighbour fallbacks are useful even for Clough-Tocher:
        # CT returns NaN outside the convex hull, and the inverse solver may
        # step infinitesimally outside near seams/boundaries.
        nn_dx = NearestNDInterpolator(self.src, diff[:, 0])
        nn_dy = NearestNDInterpolator(self.src, diff[:, 1])

        if interp != 'linear':
            ct_dx = CloughTocher2DInterpolator(padded_src, padded_diff[:, 0])
            ct_dy = CloughTocher2DInterpolator(padded_src, padded_diff[:, 1])

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

    def do(self, pts, mo=0):
        """ Forward Warp (Precise Cubic) """
        xy = np.array(pts, dtype=np.float64)  # Force Copy
        mode = 1.0 if mo == 0 else -1.0
        xy[:, 1] *= mode

        # Interpolate displacement
        dx = self.fwd_dx(xy)
        dy = self.fwd_dy(xy)

        res = xy + np.stack([dx, dy], axis=1)
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

        # --- COARSE GUESS ---
        guess = self.inv_linear(target)

        nan_mask = np.isnan(guess[:, 0])
        if np.any(nan_mask):
            tgt_nan = target[nan_mask]
            tgt_nan_finite = np.isfinite(tgt_nan).all(axis=1)
            if np.any(tgt_nan_finite):
                nan_idx = np.flatnonzero(nan_mask)
                guess[nan_idx[tgt_nan_finite]] = self.inv_nearest(tgt_nan[tgt_nan_finite])

        # --- ITERATIVE POLISH ---
        curr = guess.copy()

        for _ in range(iterations):
            dx = self.fwd_dx(curr)
            dy = self.fwd_dy(curr)

            # If the guess drifted off the interpolation hull, snap it back.
            bad_mask = np.isnan(dx)
            if np.any(bad_mask):
                curr[bad_mask] = self.inv_nearest(target[bad_mask])
                dx = self.fwd_dx(curr)
                dy = self.fwd_dy(curr)

            error = curr + np.stack([dx, dy], axis=1) - target
            curr -= error

            # Snap any non-finite updates back to a safe seed.
            nonfinite_mask = ~np.isfinite(curr).all(axis=1)
            if np.any(nonfinite_mask):
                curr[nonfinite_mask] = self.inv_nearest(target[nonfinite_mask])

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
        # thet = H9O.oid_tht[oid]
        self.mo = H9O.oid_mo[oid]
        # mo_str = 'V' if self.mo == 0 else 'Λ'
        super().__init__(registrar, b_sig, dom,  oid, 2)
        # self.th = (thet % 6) * np.pi / 3.

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
        self.warp = None

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
        pkg = "hhg9.data"
        layer = 5
        ell = registrar.ellipsoid_name
        pkdo_data = resources.files(pkg).joinpath(f"{ell}_pkdo_warp.npz")
        if pkdo_data.exists():
            self.set_warp(pkdo_data, method='pkdo')
        else:
            data = resources.files(pkg).joinpath(f"{ell}_l{layer}_warp_data.npz")
            if not data.exists():
                data = resources.files(pkg).joinpath(f"WGS84_l{layer}_warp_data.npz")
            self.set_warp(data)
        registrar.register_bridge(['c_oct', 'b_raw', 'b_oct'])
        registrar.register_bridge(['g_gcd', 'b_raw', 'b_oct'])


    def set_warp(self, warp_file=None, method=None):
        """Add a warp method to the domain"""
        self.warp = AuthalicWarp(warp_file, method)
        for proj in self.projs.values():
            proj.warp = self.warp

    def no_warp(self):
        """Remove warp method from the domain"""
        self.warp = None
        for proj in self.projs.values():
            proj.warp = None

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
