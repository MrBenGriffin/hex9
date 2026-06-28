# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""
Invariant tests for the pure-numerical ``hhg9.algorithms`` package:

  * distance.py / wgs84.py — geodesy (ECEF round-trip, haversine properties,
    ellipsoid area, geodesic offset round-trip)
  * geometry.py           — winding, point-in-polygon/triangle, ortho basis
  * packing.py / id_packing.py — nibble↔u64 round-trip, LUT composition
  * pickers.py            — random samplers (shape, ranges, determinism,
    on-ellipsoid / unit-sphere membership)
"""
import numpy as np
import pytest

import hhg9.algorithms.distance as D
import hhg9.algorithms.geometry as G
import hhg9.algorithms.wgs84 as W
import hhg9.algorithms.packing as P
import hhg9.algorithms.id_packing as ID
import hhg9.algorithms.pickers as PK


@pytest.fixture(scope="module")
def reg():
    from hhg9 import Registrar
    return Registrar()


# ---------------------------------------------------------------------------
# wgs84.py — ECEF ↔ geodetic
# ---------------------------------------------------------------------------

def test_ecef_latlon_roundtrip():
    lat = np.array([51.5, -33.9, 0.0, 80.0, -45.0])
    lon = np.array([-0.12, 151.2, 0.0, -170.0, 90.0])
    x, y, z = W.latlon_to_ecef(lat, lon)
    la, lo = W.ecef_to_latlon(x, y, z)
    assert np.abs(la - lat).max() < 1e-9
    assert np.abs(lo - lon).max() < 1e-9


def test_latlon_to_ecef_equator_origin():
    x, y, z = W.latlon_to_ecef(np.array([0.0]), np.array([0.0]))
    assert x[0] == pytest.approx(W.A)      # (0,0) sits at +x = semi-major axis
    assert abs(y[0]) < 1e-6 and abs(z[0]) < 1e-6


# ---------------------------------------------------------------------------
# distance.py — haversine & geodesic
# ---------------------------------------------------------------------------

def test_haversine_self_is_zero_and_symmetric():
    p1 = np.array([[51.5, -0.1], [40.0, -74.0]])
    p2 = np.array([[48.8, 2.3], [34.0, 118.0]])
    assert np.allclose(D.haversine(p1, p1), 0.0)
    np.testing.assert_allclose(D.haversine(p1, p2), D.haversine(p2, p1))


def test_haversine_one_degree_at_equator():
    d = D.haversine(np.array([[0.0, 0.0]]), np.array([[0.0, 1.0]]))[0]
    assert d == pytest.approx(np.radians(1.0) * D.R_MEAN, rel=1e-9)


def test_haversine_variants_agree():
    p1 = np.radians(np.array([[51.5, -0.1], [-10.0, 30.0]]))
    p2 = np.radians(np.array([[40.0, -74.0], [10.0, -30.0]]))
    np.testing.assert_allclose(D.haversine_rad(p1, p2),
                               D.great_circle_atan2_rad(p1, p2), rtol=1e-9)


def test_ellipsoid_area_matches_symbolic():
    assert D.ellipsoid_area_wgs84() == pytest.approx(5.10072e14, rel=1e-4)
    symbolic = D.calculate_ellipsoid_area(6378137.0, 298.257223563)
    assert symbolic == pytest.approx(D.ellipsoid_area_wgs84(), rel=1e-9)


def test_geodesic_offset_roundtrip():
    """Offsetting a point by (azimuth, distance) then measuring the geodesic
    distance back recovers the distance."""
    p1 = np.array([51.5, -0.1])
    for azi, dist in [(0.0, 5000.0), (90.0, 12000.0), (210.0, 800.0)]:
        off = np.asarray(D.wgs84_offset(p1, azi, dist)).ravel()
        assert D.ell_distance(p1, off) == pytest.approx(dist, rel=1e-6)


# ---------------------------------------------------------------------------
# geometry.py
# ---------------------------------------------------------------------------

def _signed_area(p):
    p = np.asarray(p)
    return 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])


def test_ensure_cw_orients_clockwise():
    ccw = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)   # signed area > 0
    assert _signed_area(ccw) > 0
    assert _signed_area(G.ensure_cw(ccw)) < 0
    # already-CW stays CW
    cw = ccw[::-1]
    assert _signed_area(G.ensure_cw(cw)) < 0


def test_points_in_or_on_poly():
    poly = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=float)
    pts = np.array([[0.5, 0.5], [2.0, 2.0], [0.0, 0.5]])  # in, out, on-edge
    res = G.points_in_or_on_poly(pts, poly)
    assert res[0] and not res[1] and res[2]


def test_points_in_triangles():
    tris = np.array([[[0, 0], [1, 0], [0, 1]]], dtype=float)
    res = G.points_in_triangles(np.array([[0.2, 0.2], [1.0, 1.0]]), tris)
    assert res[0] and not res[1]


def test_ortho_basis_from_normal_is_orthonormal():
    for n in (np.array([0.0, 0.0, 1.0]), np.array([1.0, 2.0, 3.0])):
        basis = np.asarray(G.ortho_basis_from_normal(n))
        assert basis.shape == (2, 3)
        # rows are unit length, mutually orthogonal, and perpendicular to n
        np.testing.assert_allclose(np.linalg.norm(basis, axis=1), 1.0, atol=1e-9)
        assert abs(basis[0] @ basis[1]) < 1e-9
        np.testing.assert_allclose(basis @ (n / np.linalg.norm(n)), 0.0, atol=1e-9)


# ---------------------------------------------------------------------------
# packing.py / id_packing.py
# ---------------------------------------------------------------------------

def test_u64_pack_layers_roundtrip():
    nb = np.random.default_rng(0).integers(0, 16, size=(4, 16)).astype(np.uint8)
    words = np.asarray(P.u64_pack(nb))
    assert words.shape == (4, 1)
    layers = P.u64_layers(words)
    np.testing.assert_array_equal(layers[:, :16], nb)


def test_ceil_log2():
    assert ID.ceil_log2(1) == 0
    assert ID.ceil_log2(5) == 3
    assert ID.ceil_log2(8) == 3
    assert ID.ceil_log2(9) == 4


def test_select_dtype_widths():
    assert ID.select_dtype(200) == np.uint8
    assert ID.select_dtype(300) == np.uint16
    assert ID.select_dtype(70000) == np.uint32


def test_compose_luts_inverse():
    enc, dec = ID.compose_luts([6, 4, 4])[:2]
    assert enc.shape == (6, 4, 4)
    assert dec.shape == (96, 3)
    # enc[h,p,n] -> code, dec[code] -> (h,p,n): mutually inverse
    for h in range(6):
        for p in range(4):
            for n in range(4):
                code = enc[h, p, n]
                np.testing.assert_array_equal(dec[code], [h, p, n])


# ---------------------------------------------------------------------------
# pickers.py
# ---------------------------------------------------------------------------

def test_gcd_rnd_shape_range_determinism():
    a = PK.gcd_rnd(100, seed=1)
    assert a.shape == (100, 2)
    assert a[:, 0].min() >= -90 and a[:, 0].max() <= 90
    assert a[:, 1].min() >= -180 and a[:, 1].max() <= 180
    np.testing.assert_array_equal(a, PK.gcd_rnd(100, seed=1))


def test_gcd_fibonacci_shape():
    f = PK.gcd_fibonacci(256)
    assert f.shape == (256, 2)
    assert f[:, 0].min() >= -90 and f[:, 0].max() <= 90


def test_sph_rnd_is_unit_sphere():
    s = PK.sph_rnd(200)
    assert s.shape == (200, 3)
    np.testing.assert_allclose(np.linalg.norm(s, axis=1), 1.0, atol=1e-12)


def test_gcd_cap_rnd_count():
    cap = PK.gcd_cap_rnd((51.5, -0.1), 60, 50.0, seed=2)
    assert cap.shape == (60, 2)


def test_ell_rnd_uniform_on_ellipsoid(reg):
    ecef, latlon = PK.ell_rnd_uniform(reg, 300, seed=1, return_latlon=True)
    assert ecef.shape == (300, 3)
    assert latlon.shape == (300, 2)
    ell = reg.domain('c_ell')
    a, b = ell.a, ell.b
    # every point lies on the ellipsoid: x²/a² + y²/a² + z²/b² == 1
    f = ecef[:, 0]**2 / a**2 + ecef[:, 1]**2 / a**2 + ecef[:, 2]**2 / b**2
    np.testing.assert_allclose(f, 1.0, atol=1e-9)
    # deterministic under seed
    ecef2, _ = PK.ell_rnd_uniform(reg, 300, seed=1, return_latlon=True)
    np.testing.assert_array_equal(ecef, ecef2)
