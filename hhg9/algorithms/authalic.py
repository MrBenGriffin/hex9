# Part of the Hex9 (H9) Project
# Copyright ©2026, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""Geodetic ↔ authalic latitude conversion.

The authalic latitude ξ maps the ellipsoid to the sphere of equal surface
area while preserving area everywhere — the standard datum reduction
(Snyder 3-11..3-13). It is what lets a single sphere-trained equal-area
warp serve every ellipsoid: the ellipsoid-specific part of the chain is
this latitude reparameterisation and nothing else.

PRODUCTION PATH (|n| < 0.01, i.e. every geodetic ellipsoid): Karney's
6th-order Fourier series in the third flattening n = f/(2−f), evaluated
by Clenshaw summation — the same method and coefficients as PROJ's
pj_authalic_lat (src/latitudes.cpp):

    ξ − φ = Σₖ Fₖ·sin(2kφ),  k = 1..6,  Fₖ = polynomial in n (order 6)

and a matching series for the inverse. Computing the small DIFFERENCE
ξ − φ avoids the 1/cosφ conditioning of the closed form entirely; the
n⁷ truncation error (~1e-19 rad for WGS84) is far below float64. See
C. F. F. Karney, "On auxiliary latitudes", Survey Review 56 (2024),
https://doi.org/10.1080/00396265.2023.2217604 (Eqs. A19/A20).

ORACLE / LARGE-n FALLBACK: the exact closed form —

    q(φ)  = (1−e²)·[ sinφ/(1−e²sin²φ) − (1/2e)·ln((1−e·sinφ)/(1+e·sinφ)) ]
    ξ     = arcsin( q(φ) / q(π/2) )

with Newton inversion, both directions switching to a cancellation-free
colatitude form above 45° (measured RT ≤1.5e-15 rad ≈ 9 nm, poles
exact). Exposed as *_exact; used automatically when |n| ≥ 0.01, and by
the test-suite to machine-verify the series. A sphere (e²=0)
short-circuits to the identity in both directions everywhere.
"""
import numpy as np
from numpy.typing import NDArray

# ── Karney 6th-order series coefficients (Survey Review 2024, A19/A20) ──────
# Verbatim from PROJ src/latitudes.cpp (pj_auxlat_coeffs, Maxima-generated):
# each row l gives the Taylor polynomial (ascending powers of n, order
# 6−l−1) whose value, times n^(l+1), is the Fourier coefficient F[l] of
# sin(2(l+1)ζ).

# C[ξ,φ]: forward, ζ = φ (geographic) → η = ξ (authalic)
_C_XI_PHI = (
    (-4.0 / 3.0, -4.0 / 45.0, 88.0 / 315.0, 538.0 / 4725.0,
     20824.0 / 467775.0, -44732.0 / 2837835.0),
    (34.0 / 45.0, 8.0 / 105.0, -2482.0 / 14175.0, -37192.0 / 467775.0,
     -12467764.0 / 212837625.0),
    (-1532.0 / 2835.0, -898.0 / 14175.0, 54968.0 / 467775.0,
     100320856.0 / 1915538625.0),
    (6007.0 / 14175.0, 24496.0 / 467775.0, -5884124.0 / 70945875.0),
    (-23356.0 / 66825.0, -839792.0 / 19348875.0),
    (570284222.0 / 1915538625.0,),
)

# C[φ,ξ]: inverse, ζ = ξ (authalic) → η = φ (geographic)
_C_PHI_XI = (
    (4.0 / 3.0, 4.0 / 45.0, -16.0 / 35.0, -2582.0 / 14175.0,
     60136.0 / 467775.0, 28112932.0 / 212837625.0),
    (46.0 / 45.0, 152.0 / 945.0, -11966.0 / 14175.0, -21016.0 / 51975.0,
     251310128.0 / 638512875.0),
    (3044.0 / 2835.0, 3802.0 / 14175.0, -94388.0 / 66825.0,
     -8797648.0 / 10945935.0),
    (6059.0 / 4725.0, 41072.0 / 93555.0, -1472637812.0 / 638512875.0),
    (768272.0 / 467775.0, 455935736.0 / 638512875.0),
    (4210684958.0 / 1915538625.0,),
)

# PROJ's series-validity cutoff (PROJ_AUTHALIC_SERIES_VALID): full float64
# accuracy for |f| <= 1/150, satisfactory to |f| <= 1/50.
_SERIES_N_CUTOFF = 0.01

_coeff_cache: dict = {}


def _third_flattening(e2: float) -> float:
    """n = (a−b)/(a+b) = f/(2−f) from e²."""
    b_a = np.sqrt(1.0 - e2)
    return (1.0 - b_a) / (1.0 + b_a)


def _series_coeffs(e2: float):
    """(F_forward, F_inverse) — six Fourier coefficients each, or None when
    the series is invalid for this eccentricity (|n| ≥ 0.01)."""
    got = _coeff_cache.get(e2, 0)
    if got != 0:
        return got
    n = _third_flattening(e2)
    if abs(n) >= _SERIES_N_CUTOFF:
        out = None
    else:
        def fill(rows):
            f = np.empty(6)
            d = n
            for l, p in enumerate(rows):
                acc = p[-1]
                for c in p[-2::-1]:       # Horner, ascending powers stored
                    acc = acc * n + c
                f[l] = d * acc
                d *= n
            return f
        out = (fill(_C_XI_PHI), fill(_C_PHI_XI))
    _coeff_cache[e2] = out
    return out


def _clenshaw(szeta: NDArray, czeta: NDArray, F: NDArray) -> NDArray:
    """Σ F[k]·sin((2k+2)ζ) by Clenshaw summation (vectorised;
    mirrors PROJ's pj_clenshaw)."""
    u0 = np.zeros_like(szeta)
    u1 = np.zeros_like(szeta)
    X = 2.0 * (czeta - szeta) * (czeta + szeta)     # 2·cos(2ζ)
    for k in range(5, -1, -1):
        u0, u1 = X * u0 - u1 + F[k], u0
    return 2.0 * szeta * czeta * u0                  # sin(2ζ)·u0


def authalic_q(sphi: NDArray, e2: float) -> NDArray:
    """Snyder's q as a function of sinφ; q(±π/2) = ±q_p."""
    if e2 == 0.0:
        return 2.0 * sphi
    e = np.sqrt(e2)
    es = e * sphi
    return (1.0 - e2) * (sphi / (1.0 - e2 * sphi * sphi)
                         - (0.5 / e) * np.log((1.0 - es) / (1.0 + es)))


def geodetic_to_authalic(phi: NDArray, e2: float) -> NDArray:
    """Geodetic latitude φ (radians) → authalic latitude ξ (radians).

    Karney 6th-order Clenshaw series (production; every geodetic
    ellipsoid); exact closed form when the series is out of range."""
    phi = np.asarray(phi, dtype=np.float64)
    if e2 == 0.0:
        return phi
    F = _series_coeffs(e2)
    if F is None:
        return geodetic_to_authalic_exact(phi, e2)
    return phi + _clenshaw(np.sin(phi), np.cos(phi), F[0])


def authalic_to_geodetic(xi: NDArray, e2: float) -> NDArray:
    """Authalic latitude ξ (radians) → geodetic latitude φ (radians).

    Karney 6th-order Clenshaw series (production; every geodetic
    ellipsoid); exact Newton inversion when the series is out of range."""
    xi = np.asarray(xi, dtype=np.float64)
    if e2 == 0.0:
        return xi
    F = _series_coeffs(e2)
    if F is None:
        return authalic_to_geodetic_exact(xi, e2)
    return xi + _clenshaw(np.sin(xi), np.cos(xi), F[1])


def geodetic_to_authalic_exact(phi: NDArray, e2: float) -> NDArray:
    """Exact closed-form φ → ξ (any eccentricity; series oracle).

    Near the poles arcsin(q/q_p) amplifies q's rounding noise
    by 1/cosξ, so the polar caps use the cancellation-free q-difference:
    1 − sinξ = (q_p − q)/q_p, ξ = π/2 − 2·arcsin(√((q_p − q)/(2q_p))) —
    relative precision in colatitude throughout.
    """
    phi = np.asarray(phi, dtype=np.float64)
    if e2 == 0.0:
        return phi
    qp = authalic_q(np.float64(1.0), e2)
    sgn = np.where(phi < 0.0, -1.0, 1.0)
    aphi = np.abs(phi)
    # The colatitude form is exact to ~1 ULP wherever the colatitude is
    # small-ish; the direct arcsin degrades as 1/cosξ. Crossover at 45°.
    polar = aphi > (np.pi / 4.0)
    xi = np.empty_like(aphi)
    if np.any(~polar):
        x = aphi[~polar]
        xi[~polar] = np.arcsin(np.clip(authalic_q(np.sin(x), e2) / qp,
                                       -1.0, 1.0))
    if np.any(polar):
        u = np.pi / 2.0 - aphi[polar]           # geodetic colatitude
        d = _qp_minus_q(u, e2)                  # q_p − q, full rel. precision
        xi[polar] = np.pi / 2.0 - 2.0 * np.arcsin(
            np.sqrt(np.clip(d / (2.0 * qp), 0.0, 1.0)))
    return sgn * xi


def _qp_minus_q(u: NDArray, e2: float) -> NDArray:
    """q(π/2) − q(π/2 − u), cancellation-free (u = colatitude, radians).

    Direct subtraction loses ~1e-16 absolute in q, which the Newton step
    amplifies by 1/dq ≈ 1/(2·cosφ) — microns of latitude near the poles.
    With w = 1 − sinφ = 2·sin²(u/2) carried exactly and the log via log1p,
    every factor is O(w): the difference has full relative precision.
    """
    e = np.sqrt(e2)
    s = np.cos(u)                       # sinφ
    w = 2.0 * np.sin(0.5 * u) ** 2     # 1 − sinφ, no cancellation
    # (1−e²)[1/(1−e²) − s/(1−e²s²)] reduces to w(1+e²s)/(1−e²s²): the
    # (1−e²) prefactor is already absorbed. Only the log term keeps it.
    term1 = w * (1.0 + e2 * s) / (1.0 - e2 * s * s)
    term2 = -(0.5 / e) * np.log1p(-2.0 * e * w / ((1.0 + e) * (1.0 - e * s)))
    return term1 + (1.0 - e2) * term2


def authalic_to_geodetic_exact(xi: NDArray, e2: float, max_iter: int = 8) -> NDArray:
    """Exact ξ → φ by Newton on q (any eccentricity; series oracle).

    Seeded at φ=ξ. |dξ/dφ − 1| ≲ 3e-3 for WGS84, so the seed
    is already close and convergence is quadratic; the loop exits early on
    a 1e-15 rad step. Near the poles the iteration switches variable to the
    colatitude with a cancellation-free q-difference (see _qp_minus_q), so
    precision is relative in colatitude — no 1/cosφ noise amplification.
    """
    xi = np.asarray(xi, dtype=np.float64)
    if e2 == 0.0:
        return xi
    qp = authalic_q(np.float64(1.0), e2)
    sgn = np.where(xi < 0.0, -1.0, 1.0)
    axi = np.abs(xi)
    # Same 45° crossover as geodetic_to_authalic: the colatitude Newton is
    # noise-free (relative precision in u); the direct form loses 1/cosφ.
    polar = axi > (np.pi / 4.0)

    phi = np.empty_like(axi)

    # ── bulk: Newton on q directly ───────────────────────────────────────
    if np.any(~polar):
        x = axi[~polar]
        q_target = qp * np.sin(x)
        p = x.copy()
        for _ in range(max_iter):
            sphi = np.sin(p)
            dq = 2.0 * (1.0 - e2) * np.cos(p) / (1.0 - e2 * sphi * sphi) ** 2
            step = (q_target - authalic_q(sphi, e2)) / dq
            p = p + step
            if np.max(np.abs(step)) < 1e-15:
                break
        phi[~polar] = p

    # ── polar cap: Newton in colatitude u = π/2 − φ on q_p − q ──────────
    # Target q_p − q_t = q_p(1 − sinξ) = 2·q_p·sin²(v/2), v = π/2 − ξ:
    # exact, no subtraction. d(q_p − q)/du = 2(1−e²)sin u/(1−e²cos²u)².
    if np.any(polar):
        v = np.pi / 2.0 - axi[polar]
        d_target = 2.0 * qp * np.sin(0.5 * v) ** 2
        u = v.copy()                    # seed: colat of ξ
        for _ in range(max_iter):
            su = np.sin(u)
            cu = np.cos(u)
            ddu = 2.0 * (1.0 - e2) * su / (1.0 - e2 * cu * cu) ** 2
            f = _qp_minus_q(u, e2) - d_target
            # u=0 (exact pole) has ddu=0 and f=0: hold it fixed.
            step = np.where(ddu > 0.0, f / np.where(ddu > 0.0, ddu, 1.0), 0.0)
            u = u - step
            if np.max(np.abs(step)) < 1e-18:
                break
        phi[polar] = np.pi / 2.0 - u

    return sgn * phi
