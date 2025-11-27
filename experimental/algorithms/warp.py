"""
Part of the H9 project
Warp. This uses barycentric simplex co-ordinates to add warp.
The purpose is to improve authalic / equal-area RMSE.
"""
import numpy as np
from pathlib import Path
from math import lgamma


def _all_bernstein_terms(n):
    """Return full list of (i,j) index pairs for degree n on the simplex."""
    terms = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            terms.append((i, j))
    return np.array(terms, dtype=int)


def _log_mult(i, j, k, n):
    return lgamma(n + 1) - (lgamma(i + 1) + lgamma(j + 1) + lgamma(k + 1))


def _prepare_bernstein_terms(terms, n):
    """Precompute per-term exponents and log multinomial coefficients.

    Parameters
    ----------
    terms : array-like, shape (K, 2)
        Integer (i,j) index pairs for Bernstein terms.
    n : int
        Total Bernstein degree.

    Returns
    -------
    i : ndarray, shape (K,)
    j : ndarray, shape (K,)
    k : ndarray, shape (K,)
    log_coeff : ndarray, shape (K,)
        log multinomial coefficients log C(n; i,j,k).
    """
    terms = np.asarray(terms, dtype=int)
    i = terms[:, 0]
    j = terms[:, 1]
    k = n - i - j

    # log C = log(n!) - log(i!) - log(j!) - log(k!)
    log_coeff = np.empty_like(i, dtype=float)
    for idx, (ii, jj, kk) in enumerate(zip(i, j, k)):
        log_coeff[idx] = lgamma(n + 1) - (lgamma(ii + 1) + lgamma(jj + 1) + lgamma(kk + 1))

    return i, j, k, log_coeff


class Warper:
    """
    Provide a warping function for simplex coordinates
    scale : Scalar factor for the displacement field ∇ψ.
    """
    def __init__(self, path=None):
        self.terms = None
        self.c = None
        self.n = None
        self.scale = 1.0
        self._bernstein_pre = None  # (i, j, k, logC)
        if path is None:
            raise ValueError("Must provide path to load warp values.")
            # path = Path(__file__).parent / "ma_psi_xl5_v0305_l5_m0_n16_v0305.npz"
            # self.load_values(path)
        elif isinstance(path, str) or isinstance(path, Path):
            self.load_values(path)
        else:
            self.set_values(*path)

    def eval_and_grad_uv(self, uv, pre=None, eps=1e-12):
        """Bernstein ψ, ∇ψ evaluator on the simplex.

        Evaluate
            ψ(u,v) = Σ c_ij * B_ij^n(u,v),
        and its gradient with respect to (u,v).

        Parameters
        ----------
        uv : array-like, shape (..., 2)
            Simplex coordinates (u,v).
            Total Bernstein degree.
        pre : tuple or None
            Optional precomputed (i, j, k, logC) from `_prepare_bernstein_terms`.
        eps : float
            Clamp value near the simplex boundaries to avoid log(0).
        """
        terms = self.terms
        c = self.c
        n = self.n
        uv = np.asarray(uv, dtype=float)
        u = uv[..., 0]
        v = uv[..., 1]

        # Flatten for vectorised computation, then restore shape at the end
        orig_shape = u.shape
        u_flat = u.ravel()
        v_flat = v.ravel()
        w_flat = 1.0 - u_flat - v_flat

        u_c = np.maximum(u_flat, eps)
        v_c = np.maximum(v_flat, eps)
        w_c = np.maximum(w_flat, eps)

        if pre is not None:
            i, j, k, log_coeff = pre
        else:
            # Lazily prepare and cache Bernstein term data
            if self._bernstein_pre is None:
                self._bernstein_pre = _prepare_bernstein_terms(terms, n)
            i, j, k, log_coeff = self._bernstein_pre

        c = np.asarray(c, dtype=float)

        # Shapes: (M,) for u_c/v_c/w_c, (K,) for i/j/k/log_coeff
        # Broadcast to (M, K)
        u_log = np.log(u_c)[:, None]
        v_log = np.log(v_c)[:, None]
        w_log = np.log(w_c)[:, None]

        i_b = i[None, :]
        j_b = j[None, :]
        k_b = k[None, :]
        log_coeff_b = log_coeff[None, :]

        # Basis values for all points and all terms
        log_basis = log_coeff_b + i_b * u_log + j_b * v_log + k_b * w_log
        basis = np.exp(log_basis)

        # ψ(u,v) = Σ c_t * B_t(u,v)
        psi_flat = basis @ c

        # Gradients via log-derivative trick
        inv_u = 1.0 / u_c
        inv_v = 1.0 / v_c
        inv_w = 1.0 / w_c

        dlog_basis_du = i_b * inv_u[:, None] - k_b * inv_w[:, None]
        dlog_basis_dv = j_b * inv_v[:, None] - k_b * inv_w[:, None]

        dbasis_du = basis * dlog_basis_du
        dbasis_dv = basis * dlog_basis_dv

        dpsi_du_flat = dbasis_du @ c
        dpsi_dv_flat = dbasis_dv @ c

        # Restore original shape
        psi = psi_flat.reshape(orig_shape)
        dpsi_du = dpsi_du_flat.reshape(orig_shape)
        dpsi_dv = dpsi_dv_flat.reshape(orig_shape)

        grad = np.stack([dpsi_du, dpsi_dv], axis=-1)
        return psi, grad

    def warp(self, pts, project_to_simplex=True):
        """Apply warp to uv coordinates.

        Parameters
        ----------
        pts : Points
            Simplex points (u, v) with implicit w = 1 - u - v.
        project_to_simplex : bool
            If True, project warped coordinates back onto the simplex
            (u >= 0, v >= 0, w >= 0, u + v + w = 1) by clipping and
            renormalising barycentric components.
        """
        result = pts.copy()
        uv = result.coords
        _, grad = self.eval_and_grad_uv(uv)

        # Apply uv-metric correction: g_inv corresponds to the equilateral metric
        g_inv = np.array([[2.0 / 3.0, -1.0 / 3.0], [-1.0 / 3.0, 2.0 / 3.0]], dtype=float)
        grad_metric = grad @ g_inv.T  # (N, 2)

        u, v = uv[..., 0], uv[..., 1]
        # assert np.all(u >= -1e-9) and np.all(v >= -1e-9) and np.all(u + v <= 1 + 1e-9), "input not in simplex"

        uv_new = uv - self.scale * grad_metric
        # uv_new = uv + scale * grad
        # uv_new = uv - scale * grad  # -∇ψ

        if project_to_simplex:
            # Convert to (u, v, w), clip to non-negative, renormalise to sum 1
            u = uv_new[..., 0]
            v = uv_new[..., 1]

            w = 1.0 - u - v

            u = np.clip(u, 0.0, None)
            v = np.clip(v, 0.0, None)
            w = np.clip(w, 0.0, None)

            s = u + v + w
            # Avoid division by zero; if s == 0, fall back to equal split
            zero_mask = (s == 0.0)
            if np.any(zero_mask):
                u[zero_mask] = 1.0 / 3.0
                v[zero_mask] = 1.0 / 3.0
                w[zero_mask] = 1.0 / 3.0
                s[zero_mask] = 1.0

            u /= s
            v /= s
            # assert np.all(u >= -1e-9) and np.all(v >= -1e-9) and np.all(u + v <= 1 + 1e-9), "after normalisation input not in simplex"
            uv_new = np.stack([u, v], axis=-1)

        result.coords = uv_new
        return result

    def load_values(self, path):
        """
        sets: (terms, c, n)
        """
        degree = 16
        max_scale = 1.0
        scale = max_scale * 0.46
        z = np.load(path, allow_pickle=True)
        files = set(z.files)
        self.set_values(z['terms'], z['c'], degree, scale)

    def set_values(self, terms, c, n, scale=1.0):
        """
        sets: (terms, c, n, scale)
        """
        self.terms = terms
        self.c = c
        self.n = n
        self.scale = scale
        self._bernstein_pre = _prepare_bernstein_terms(self.terms, self.n)

