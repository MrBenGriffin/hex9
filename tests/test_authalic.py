# Part of the Hex9 (H9) Project
# Copyright ©2026, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""Authalic latitude: Karney 6th-order Clenshaw series (production path)
machine-verified against the exact closed-form q oracle, plus the
AuthalicGCD boundary projection (c_sph ↔ g_gcd) round-trip."""

import numpy as np
import pytest

from hhg9.algorithms.authalic import (
    geodetic_to_authalic, authalic_to_geodetic,
    geodetic_to_authalic_exact, authalic_to_geodetic_exact,
    _series_coeffs, _third_flattening)

E2_WGS84 = 0.00669437999014132
R_A = 6371007.1809          # WGS84 authalic radius, metres (for nm scaling)

# Dense global sweep + pole-hugging tails (the conditioning hot-spots).
PHI = np.radians(np.concatenate([
    np.linspace(-90.0, 90.0, 100001),
    90.0 - np.geomspace(1e-12, 1.0, 1500),
    -(90.0 - np.geomspace(1e-12, 1.0, 1500)),
]))


def test_series_matches_exact_oracle_both_directions():
    """Series agrees with the closed form to the oracle's own noise floor
    (the oracle's 45° crossover residual, ≤1.5e-15 rad ≈ 9 nm)."""
    xi_s = geodetic_to_authalic(PHI, E2_WGS84)
    xi_e = geodetic_to_authalic_exact(PHI, E2_WGS84)
    assert np.abs(xi_s - xi_e).max() < 1.6e-15
    ph_s = authalic_to_geodetic(xi_s, E2_WGS84)
    ph_e = authalic_to_geodetic_exact(xi_s, E2_WGS84)
    assert np.abs(ph_s - ph_e).max() < 1.6e-15


def test_series_roundtrip_ulp():
    """φ→ξ→φ through the series is ULP-level everywhere (≤3e-16 rad ≈ 2 nm),
    including the pole-hugging tail."""
    back = authalic_to_geodetic(geodetic_to_authalic(PHI, E2_WGS84), E2_WGS84)
    assert np.abs(back - PHI).max() < 3e-16


def test_fixed_points_and_sphere_identity():
    for deg in (90.0, -90.0, 0.0):
        p = np.radians(np.array([deg]))
        assert geodetic_to_authalic(p, E2_WGS84)[0] == p[0]
        assert authalic_to_geodetic(p, E2_WGS84)[0] == p[0]
    assert np.array_equal(geodetic_to_authalic(PHI, 0.0), PHI)
    assert np.array_equal(authalic_to_geodetic(PHI, 0.0), PHI)


def test_leading_term_magnitude():
    """ξ − φ ≈ −(4/3)n·sin2φ: the known ~0.1283° maximum offset for WGS84."""
    xi = geodetic_to_authalic(PHI, E2_WGS84)
    off = np.degrees(np.abs(xi - PHI)).max()
    assert 0.127 < off < 0.130


def test_large_n_falls_back_to_exact():
    """Past PROJ's |n| < 0.01 validity cutoff the dispatchers use the exact
    path; round-trip stays sub-ULP-scale."""
    f = 0.05
    e2 = 2 * f - f * f
    assert _series_coeffs(e2) is None
    assert abs(_third_flattening(e2)) >= 0.01
    p = np.radians(np.linspace(-90, 90, 20001))
    back = authalic_to_geodetic(geodetic_to_authalic(p, e2), e2)
    assert np.abs(back - p).max() < 2e-15


def test_authalic_gcd_projection_roundtrip():
    """AuthalicGCD (c_sph ↔ g_gcd) round-trips to nm on the WGS84 registrar,
    including near-pole points (the atan2/hypot latitude recovery)."""
    from hhg9 import Registrar, Points
    reg = Registrar()          # WGS84; projection reads reg.ellipsoid
    g = reg.domain('g_gcd')
    c_sph = reg.domain('c_sph')
    lat = np.concatenate([np.linspace(-89.0, 89.0, 2001),
                          [89.999, -89.999, 89.9999, -89.9999, 90.0, -90.0]])
    lon = np.linspace(-180.0, 180.0, len(lat))
    pts = np.column_stack([lat, lon])
    sph = reg.project(Points(pts.copy(), g), [g, c_sph])
    # on the unit sphere, exactly
    assert np.abs(np.linalg.norm(sph.coords, axis=1) - 1.0).max() < 1e-15
    back = reg.project(sph, [c_sph, g])
    dlat = np.radians(np.abs(back.coords[:, 0] - lat))
    assert (dlat * R_A).max() < 5e-9    # < 5 nm of latitude
