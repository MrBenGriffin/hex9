# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""
AuthalicWarp — edge-preserving authalic correction (inference side).

This is the clean replacement for the AuthalicWarp class in
``hhg9/domains/octahedral_barycentric.py``. Once ``test_warp.py`` confirms a
trained warp round-trips below the 7 nm threshold, this class and the trained
``.npz`` drop straight into that module, and the identity-bypass workarounds in
``do()`` / ``undo()`` are removed.

Key difference from the old class
---------------------------------
The old warp let the lateral octant edges drift: the Clough–Tocher displacement
field had a non-zero (tangential, then bulging) component on the seam, so points
on or near the lateral edges were mapped off the triangle and the Newton inverse
diverged — catastrophically at the apex.

Here the displacement is multiplied by ``config.edge_blend`` — a C1 function that
is identically zero on the two lateral edges. The lateral edges (and the apex,
where they meet) are therefore mapped by the identity *exactly*, independent of
any interpolation overshoot. No edge-detection bypass is needed: the warp simply
*is* the identity there, so ``do()`` and ``undo()`` are correct without special
cases.
"""
import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator, NearestNDInterpolator

from hhg9.h9 import H9K, in_scope
from experimental.sinkhorn import config


class WarpTolerance:
    """Newton-Raphson convergence tolerances (barycentric XY units).
    1 unit ≈ 2.5 × 10⁷ m, so FINE ≈ 0.25 mm."""
    MACH = 1e-17
    FINE = 1e-15
    NORM = 1e-14


class AuthalicWarp:
    """Forward (do) and inverse (undo) authalic warp on the b_oct triangle.

    The warp displacement is interpolated with Clough–Tocher and faded to zero
    on the lateral edges via ``config.edge_blend``.
    """

    def __init__(self, file_name=None, tolerance=WarpTolerance.FINE):
        self.tolerance = tolerance
        self.file_name = file_name
        if file_name is None:
            return

        repo = np.load(file_name, allow_pickle=True)
        self.src = np.asarray(repo['source_pts'], dtype=np.float64)
        self.dst = np.asarray(repo['target_pts'], dtype=np.float64)
        repo.close()

        diff = self.dst - self.src

        # Ghost-row padding across the equatorial (base) edge stabilises the
        # mode-0 / mode-1 boundary. The lateral edges need no padding — the
        # edge-blend already pins them.
        Y_EQ = np.sqrt(6.0) / 6.0
        eq_band = (Y_EQ - self.src[:, 1]) < 0.05
        if np.any(eq_band):
            ghost_src = self.src[eq_band].copy()
            ghost_src[:, 1] = 2.0 * Y_EQ - ghost_src[:, 1]
            ghost_diff = diff[eq_band].copy()
            ghost_diff[:, 1] *= -1.0
            padded_src  = np.vstack([self.src, ghost_src])
            padded_diff = np.vstack([diff, ghost_diff])
        else:
            padded_src, padded_diff = self.src, diff

        self._ct_dx = CloughTocher2DInterpolator(padded_src, padded_diff[:, 0])
        self._ct_dy = CloughTocher2DInterpolator(padded_src, padded_diff[:, 1])
        self._nn_dx = NearestNDInterpolator(self.src, diff[:, 0])
        self._nn_dy = NearestNDInterpolator(self.src, diff[:, 1])

        # Inverse seed: linear map dst→src, nearest-neighbour fallback.
        self.inv_linear  = CloughTocher2DInterpolator(self.dst, self.src)
        self.inv_nearest = NearestNDInterpolator(self.dst, self.src)

    # ── displacement field (edge-blended) ────────────────────────────────────
    def _disp(self, xy):
        """Edge-blended displacement at b_oct query points xy → (N,2)."""
        dx = self._ct_dx(xy)
        dy = self._ct_dy(xy)
        bad = np.isnan(dx) & np.isfinite(xy).all(axis=1)
        if np.any(bad):
            dx[bad] = self._nn_dx(xy[bad])
            dy[bad] = self._nn_dy(xy[bad])
        phi = config.edge_blend(xy)
        return np.stack([phi * dx, phi * dy], axis=1)

    # ── forward ──────────────────────────────────────────────────────────────
    def do(self, pts, mo=0):
        """Forward warp b_raw → b_oct."""
        xy = np.array(pts, dtype=np.float64)
        sign = 1.0 if mo == 0 else -1.0
        xy[:, 1] *= sign
        res = xy + self._disp(xy)
        res[:, 1] *= sign
        return res

    def set_tolerance(self, tolerance: float):
        self.tolerance = tolerance
        return self

    # ── inverse ──────────────────────────────────────────────────────────────
    def undo(self, pts, mo=0, iterations=25):
        """Inverse warp b_oct → b_raw via Newton-Raphson.

        No edge bypass: on the lateral edges the displacement is identically
        zero, so the iteration converges immediately there.
        """
        target_all = np.array(pts, dtype=np.float64)
        sign = 1.0 if mo == 0 else -1.0
        target_all[:, 1] *= sign

        finite = np.isfinite(target_all).all(axis=1)
        if not np.all(finite):
            out = np.full_like(target_all, np.nan)
            if not np.any(finite):
                out[:, 1] *= sign
                return out
            target = target_all[finite]
        else:
            out, target = None, target_all

        # Coarse seed.
        guess = np.asarray(self.inv_linear(target), dtype=np.float64)
        nan = np.isnan(guess[:, 0])
        if np.any(nan):
            fin = np.isfinite(target[nan]).all(axis=1)
            idx = np.flatnonzero(nan)[fin]
            guess[idx] = self.inv_nearest(target[nan][fin])

        curr = guess.copy()
        R3, W_, VC = H9K.radical.R3, H9K.Ẇ, H9K.VC
        _h = 1e-7
        for _ in range(iterations):
            disp = self._disp(curr)
            bad = np.isnan(disp[:, 0])
            if np.any(bad):
                curr[bad] = self.inv_nearest(target[bad])
                disp = self._disp(curr)

            error = curr + disp - target

            d_px = self._disp(curr + [_h, 0.0])
            d_py = self._disp(curr + [0.0, _h])
            a = 1.0 + (d_px[:, 0] - disp[:, 0]) / _h
            b = (d_py[:, 0] - disp[:, 0]) / _h
            c = (d_px[:, 1] - disp[:, 1]) / _h
            d = 1.0 + (d_py[:, 1] - disp[:, 1]) / _h
            det = a * d - b * c
            ex, ey = error[:, 0], error[:, 1]
            safe = np.abs(det) > 1e-12
            denom = np.where(safe, det, 1.0)
            curr[:, 0] += np.where(safe, -(d * ex - b * ey) / denom, -ex)
            curr[:, 1] += np.where(safe, -(a * ey - c * ex) / denom, -ey)

            # Keep iterates inside the octant triangle.
            xd = R3 * curr[:, 0]
            oob = ~in_scope(xd, curr[:, 1], 0)
            if np.any(oob):
                nanb = oob & ~np.isfinite(curr).all(axis=1)
                if np.any(nanb):
                    curr[nanb] = self.inv_nearest(target[nanb])
                proj = oob & ~nanb
                if np.any(proj):
                    y = curr[proj, 1]
                    y = np.maximum(y, xd[proj] - W_)
                    y = np.maximum(y, -xd[proj] - W_)
                    curr[proj, 1] = np.minimum(y, VC)

        curr[:, 1] *= sign
        if out is not None:
            out[finite] = curr
            return out
        return curr
