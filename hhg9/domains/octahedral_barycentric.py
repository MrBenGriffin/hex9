# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
This is 'b_oct' barycentric xy equilateral.
"""
import numpy as np
from importlib import resources
from numpy.typing import NDArray
from hhg9.base.composite import CompositeDomain, ComponentDomain
from hhg9.base.point_format import PointFormat
from hhg9.projections import OctantBary
from hhg9.h9 import H9K, H9O, in_scope
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator, NearestNDInterpolator


class AuthalicWarp:
    def __init__(self, file_name=None, interp='ct'):
        # Load Data
        if file_name is None:
            return
        self.file_name = file_name
        repo = np.load(file_name, allow_pickle=True)
        self.src = repo['source_pts']  # Regular Grid (a_p)
        self.dst = repo['target_pts']  # Deformed Grid (x_prime)

        # --- GHOST ROW PADDING (Equator Stabilization) ---
        diff = self.dst - self.src
        Y_EQ = np.sqrt(6.0) / 6.0

        # Grab points within a small margin of the equator (e.g., top 5% of the triangle)
        # Adjust 0.05 if your grid spacing is wider/narrower.
        eq_band_mask = (Y_EQ - self.src[:, 1]) < 0.05

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
            # 1. FEED THE PADDED DATA HERE
            ct_dx = CloughTocher2DInterpolator(padded_src, padded_diff[:, 0])
            ct_dy = CloughTocher2DInterpolator(padded_src, padded_diff[:, 1])

            # 2. RESTORE THE NAN WRAPPERS
            def fwd_dx(xy):
                d = ct_dx(xy)
                m = np.isnan(d)
                if np.any(m):
                    d[m] = nn_dx(xy[m])
                return d

            def fwd_dy(xy):
                d = ct_dy(xy)
                m = np.isnan(d)
                if np.any(m):
                    d[m] = nn_dy(xy[m])
                return d

            self.fwd_dx = fwd_dx
            self.fwd_dy = fwd_dy

        else:
            # Linear fallback (does not use padding)
            lin_dx = LinearNDInterpolator(self.src, diff[:, 0])
            lin_dy = LinearNDInterpolator(self.src, diff[:, 1])

            def fwd_dx(xy):
                d = lin_dx(xy)
                m = np.isnan(d)
                if np.any(m):
                    d[m] = nn_dx(xy[m])
                return d

            def fwd_dy(xy):
                d = lin_dy(xy)
                m = np.isnan(d)
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

    def undo(self, pts, mo=0, iterations=30, tolerance=1e-17):
        """
        Precise Inverse Warp (Newton-Raphson).
        1. Guess using Linear Interpolation.
        2. Refine by minimizing |Forward(guess) - target|.
        """
        target_all = np.array(pts, dtype=np.float64)  # Target points we want to find source for
        mode = 1.0 if mo == 0 else -1.0
        target_all[:, 1] *= mode

        # Guard: scipy's KDTree query requires finite inputs.
        finite_mask = np.isfinite(target_all).all(axis=1)
        if not np.all(finite_mask):
            # Process only finite points; propagate NaNs for non-finite rows.
            out = np.full_like(target_all, np.nan)
            if not np.any(finite_mask):
                out[:, 1] *= mode
                return out
            target = target_all[finite_mask]
        else:
            out = None
            target = target_all

        # --- COARSE GUESS ---
        # Try Linear first
        guess = self.inv_linear(target)

        nan_mask = np.isnan(guess[:, 0])
        if np.any(nan_mask):
            # `target[nan_mask]` must be finite for KDTree.
            tgt_nan = target[nan_mask]
            tgt_nan_finite = np.isfinite(tgt_nan).all(axis=1)
            if np.any(tgt_nan_finite):
                guess[nan_mask][tgt_nan_finite] = self.inv_nearest(tgt_nan[tgt_nan_finite])
            # leave any non-finite rows as NaN (they will be propagated)

        # --- ITERATIVE POLISH ---
        # We want to find 'u' such that Forward(u) = target.
        # Function f(u) = Forward(u) - target. We want f(u) = 0.
        # Simple update: u_new = u - (Forward(u) - target)

        curr = guess.copy()

        for i in range(iterations):
            # Run Forward Warp on current guess
            # We can't use self.do() directly because it handles the mode flip.
            # We need the raw internal forward warp.

            # Internal Forward Logic:
            dx = self.fwd_dx(curr)
            dy = self.fwd_dy(curr)

            # If our guess drifted off the map, clamp it (prevents explosion)
            bad_guess = np.isnan(dx)
            if np.any(bad_guess):
                # Reset bad points to the nearest neighbour safety
                curr[bad_guess] = self.inv_nearest(target[bad_guess])
                dx = self.fwd_dx(curr)
                dy = self.fwd_dy(curr)

            # Calculate Residual (Error)
            est = curr + np.stack([dx, dy], axis=1)
            error = est - target

            # Check Convergence (Optional optimization)
            max_err = np.max(np.abs(error))
            if max_err < tolerance:
                break

            # Update (Simple Gradient Descent / Fixed Point)
            # Since the warp is ~1.0 scale (authalic), J is approx Identity.
            # So u_new = u_old - error works surprisingly well.
            curr -= error

            # If any rows become non-finite, snap them back to a safe seed.
            nonfinite = ~np.isfinite(curr).all(axis=1)
            if np.any(nonfinite):
                tgt_nf = target[nonfinite]
                tgt_nf_finite = np.isfinite(tgt_nf).all(axis=1)
                if np.any(tgt_nf_finite):
                    # only reset where the corresponding target is finite
                    idx = np.flatnonzero(nonfinite)
                    curr[idx[tgt_nf_finite]] = self.inv_nearest(tgt_nf[tgt_nf_finite])

        curr[:, 1] *= mode

        if out is not None:
            out[finite_mask] = curr
            out[:, 1] *= mode
            return out

        return curr


class OctantBarycentric(ComponentDomain):
    """
    This a 2D side of an Octant.
    Validity should be easy enough since we have the 3 points that define it.
    """

    def __init__(self, registrar, dom, oid):
        # sign = H9O.oid_cmp[oid]
        face = H9O.oid_str[oid]
        b_sig = f'{dom.name}:{face}'
        thet = H9O.oid_tht[oid]
        self.mo = H9O.oid_mo[oid]
        mo_str = 'V' if self.mo == 0 else 'Λ'
        super().__init__(registrar, b_sig, dom,  oid, 2)
        self.th = (thet % 6) * np.pi / 3.

        edges = H9O.edges_by_id[oid]
        self.oc = np.array([ed+mo_str for ed in edges], dtype='U3')

    def valid(self, pts: NDArray) -> NDArray:
        """
        Return an array of bools according to the validity criterion
        :param pts: set of 2d Euclidean points
        """
        return in_scope(H9K.radical.R3 * pts[..., 0], pts[..., 1], self.mode)


class OctahedralBarycentric(CompositeDomain):
    """
    Basic octahedral-2d properties and methods.
    """

    def __init__(self, registrar):
        super().__init__(registrar, 'b_oct', 2)
        self.h9map = {}
        self.h9 = registrar.format('h9')
        self.h9.composite = self

        self.sides = {}
        self.projs = {}
        # self.warp = None

        c_oct = registrar.domain('c_oct')

        oid_s = H9O.oid_str
        for oid in range(8):
            face = oid_s[oid]
            ob = OctantBarycentric(registrar, self, oid)
            self.sides[face] = ob
            oc = c_oct.sides[face]
            self.projs[face] = OctantBary(registrar, face, oc.name, ob.name)

        # Define base barycentric transformation matrices
        trans = np.array([[-1, 0], [1, -2], [1, 1]])  # Prototype [1,1,1]: Proj Z using √2, √6, √3 resp.
        r90 = np.array([(0, 1), (-1, 0)])  # 90-degree rotation matrix
        mirror_y_neg_x = np.array([(0, 1), (1, 0)])  # Mirror along y = x

        # Compute rotation matrices
        north, south = trans, trans @ mirror_y_neg_x  # South is the mirror of North
        # Loop in 90º rotation order and compute projection matrices for hex_layer and S.
        scale_factors = np.sqrt([2, 6, 3])[:, np.newaxis]
        self.rot90_idx = np.zeros(8, dtype=np.uint8)
        # These are set in order of rotation, starting with NEA
        c_id = H9O.cmp_oid
        o_str = H9O.oid_str

        rot = 0
        sigs = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
        for sig in sigs:
            n_sign = tuple([*sig, 1])
            s_sign = tuple([*sig, -1])
            n_id = c_id[n_sign]
            s_id = c_id[s_sign]
            n_face = o_str[n_id]
            s_face = o_str[s_id]
            self.projs[n_face].matrix = np.column_stack([north, np.ones(3)]) / scale_factors
            self.projs[s_face].matrix = np.column_stack([south, -np.ones(3)]) / scale_factors
            self.rot90_idx[n_id] = rot
            self.rot90_idx[s_id] = rot
            north = north @ r90
            south = south @ r90
            rot = (rot + 1) % 4
        pkg = "hhg9.data"
        data = resources.files(pkg).joinpath("l4_boct_warp_data.npz")
        self.set_warp(data)  # This is the better default.

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

    def _validate_matrices(self):
        valid = True
        for prj in self.projs:
            mtx = self.projs[prj].matrix
            dt = np.linalg.det(mtx)
            if np.abs(1 - dt) > 1e-6:
                valid = False
                print(f'{mtx}: Matrix Determinant is incorrect {dt}')
            dp = np.dot(mtx[0], mtx[1])
            if np.abs(dp) > 1e-15:
                valid = False
                print(f"Dot should be close to zero. R[0] • R[1] = {dp}")
        opposites = {
            'NEA': 'SWP', 'NEP': 'SWA',
            'NWA': 'SEP', 'NWP': 'SEA',
            'SEA': 'NWP', 'SEP': 'NWA',
            'SWA': 'NEP', 'SWP': 'NEA'
        }
        for f1, f2 in opposites.items():
            m1 = self.projs[f'{f1}'].matrix
            m2 = self.projs[f'{f2}'].matrix
            n1 = np.cross(m1[0], m1[1])
            n2 = np.cross(m2[0], m2[1])
            if not np.abs(np.dot(n1, n2) + 1) <= 1e-12:
                valid = False
                print(f"{f1} vs {f2}: {np.dot(n1, n2):.8f}. Should be -1")  # Should be -1
        if valid:
            print('matrices appear to be valid.')

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
