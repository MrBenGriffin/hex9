# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""
Coverage for the geodesic area / densify / ratio helpers in
``hhg9.algorithms.distance`` (the part not covered by test_algorithms.py):

  haversine_from_ref_rad, densify_poly_geodesic_by_step,
  densify_quad_geodesic_by_step, wgs84_ratio, wgs84_angular_ratio,
  wgs84_area (PolygonArea), enu_planar_polygon_area (local ENU planar).
"""
import numpy as np
import pytest

import hhg9.algorithms.distance as D
from hhg9 import Registrar, Points


@pytest.fixture(scope="module")
def reg():
    return Registrar()


# ---------------------------------------------------------------------------
# haversine_from_ref_rad
# ---------------------------------------------------------------------------

def test_haversine_from_ref_matches_pairwise():
    phi0, lam0 = np.radians(51.5), np.radians(-0.1)
    # last target is sub-µdeg away → exercises the small-angle branch
    phis = np.radians(np.array([40.7, -33.9, 51.5000001]))
    lams = np.radians(np.array([-74.0, 151.2, -0.1000001]))
    ref = D.haversine_from_ref_rad(phi0, lam0, phis, lams)
    p1 = np.stack([np.full_like(phis, phi0), np.full_like(lams, lam0)], axis=1)
    p2 = np.stack([phis, lams], axis=1)
    np.testing.assert_allclose(ref, D.haversine_rad(p1, p2), atol=1e-6)


# ---------------------------------------------------------------------------
# densify
# ---------------------------------------------------------------------------

_QUAD = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])


def test_densify_respects_max_step():
    dn = D.densify_poly_geodesic_by_step(_QUAD, max_step_m=20_000.0)
    assert dn.shape[0] > 4
    geo = D.geodesic_wgs84()
    steps = [geo.Inverse(*dn[i], *dn[(i + 1) % len(dn)])["s12"] for i in range(len(dn))]
    assert max(steps) <= 20_000.0 * 1.01


def test_densify_large_step_returns_only_vertices():
    dn = D.densify_poly_geodesic_by_step(_QUAD, max_step_m=1e9)
    assert dn.shape[0] == 4


def test_densify_drops_duplicate_closing_vertex():
    closed = np.vstack([_QUAD, _QUAD[:1]])         # last == first
    dn = D.densify_poly_geodesic_by_step(closed, max_step_m=1e9, closed=True)
    assert dn.shape[0] == 4


def test_densify_bad_shape_raises():
    with pytest.raises(ValueError):
        D.densify_poly_geodesic_by_step(np.zeros((3, 3)))


def test_densify_quad_wrapper_matches():
    a = D.densify_quad_geodesic_by_step(_QUAD, max_step_m=20_000.0)
    b = D.densify_poly_geodesic_by_step(_QUAD, max_step_m=20_000.0, closed=True)
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# ratios
# ---------------------------------------------------------------------------

def test_wgs84_ratio_symmetric_box_near_one():
    # [lon_min, lon_max, lat_min, lat_max] symmetric about the equator
    r = D.wgs84_ratio([-1.0, 1.0, -1.0, 1.0])
    assert r == pytest.approx(1.0, abs=0.02)


def test_wgs84_ratio_zero_width_guard():
    assert D.wgs84_ratio([5.0, 5.0, -1.0, 1.0]) == 1.0


def test_wgs84_angular_ratio():
    assert D.wgs84_angular_ratio([-2.0, 2.0, -1.0, 1.0]) == pytest.approx(0.5)
    assert D.wgs84_angular_ratio([0.0, 0.0, -1.0, 1.0]) == 1.0   # vertical-line guard


# ---------------------------------------------------------------------------
# polygon areas
# ---------------------------------------------------------------------------

def test_wgs84_area_one_degree_cell():
    # 1°×1° cell at the equator ≈ 1.23e10 m² (sign depends on winding)
    quad = np.array([[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]])  # (1,4,2)
    area = D.wgs84_area(reg=Registrar(), pts=quad, shape=4)
    assert abs(area[0]) == pytest.approx(1.23e10, rel=1e-2)


def test_wgs84_area_accepts_points(reg):
    g = reg.domain('g_gcd')
    quad = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
    pts = Points(quad, domain=g)
    area = D.wgs84_area(reg, pts, shape=4)
    assert abs(area[0]) == pytest.approx(1.23e10, rel=1e-2)


def test_enu_area_matches_geodesic_for_small_polygon(reg):
    g = reg.domain('g_gcd')
    quad = np.array([[0.0, 0.0], [0.0, 0.1], [0.1, 0.1], [0.1, 0.0]])
    pts = Points(quad, domain=g)
    enu = D.enu_planar_polygon_area(reg, pts, vertices=4)
    geo = D.wgs84_area(reg, quad.reshape(1, 4, 2), shape=4)
    assert abs(enu[0]) == pytest.approx(abs(geo[0]), rel=1e-4)


def test_enu_area_empty_input(reg):
    assert D.enu_planar_polygon_area(reg, np.zeros((0, 4, 3)), vertices=4).shape == (0,)
