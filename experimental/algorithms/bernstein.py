import numpy as np
from math import lgamma


def _log_mult(i, j, k, n):
    return lgamma(n + 1) - (lgamma(i + 1) + lgamma(j + 1) + lgamma(k + 1))


def eval_and_grad_uv(uv, terms, c, n, eps=1e-12):
    """
    Evaluate ψ(u,v) = Σ c_ij * B_ij^n(u,v), and its gradient on the simplex.
    B_ij^n(u,v) = C(n,i,j) * u^i * v^j * (1-u-v)^{k}, with k = n-i-j.
    ∂B/∂u = B * ( i/max(u,eps) - k/max(1-u-v,eps) )
    ∂B/∂v = B * ( j/max(v,eps) - k/max(1-u-v,eps) )
    """
    u = np.asarray(uv[..., 0])
    v = np.asarray(uv[..., 1])
    w = 1.0 - u - v

    u_c = np.maximum(u, eps)
    v_c = np.maximum(v, eps)
    w_c = np.maximum(w, eps)

    # precompute multinomial coefficients in log space for stability
    # log C = log(n!) - log(i!) - log(j!) - log(k!)

    psi = np.zeros_like(u, dtype=float)
    dpsi_du = np.zeros_like(u, dtype=float)
    dpsi_dv = np.zeros_like(u, dtype=float)

    for (ij, coeff) in zip(terms, c):
        i = int(ij[0])
        j = int(ij[1])
        k = n - i - j
        # basis value
        log_basis = _log_mult(i, j, k, n) + i*np.log(u_c) + j*np.log(v_c) + k*np.log(w_c)
        basis = np.exp(log_basis)

        psi += coeff * basis

        # gradient of basis via log-derivative trick
        dlog_basis_du = (i/u_c) - (k/w_c)
        dlog_basis_dv = (j/v_c) - (k/w_c)
        dbasis_du = basis * dlog_basis_du
        dbasis_dv = basis * dlog_basis_dv

        dpsi_du += coeff * dbasis_du
        dpsi_dv += coeff * dbasis_dv
    grad = np.stack([dpsi_du, dpsi_dv], axis=-1)
    return psi, grad
