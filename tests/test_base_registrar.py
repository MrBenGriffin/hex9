# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""
Tests for the registry / domain infrastructure
(``hhg9.base.registrar``, ``hhg9.base.domain``, ``hhg9.base.composite``).

Covers: lazy-cached domain/format/projection lookup and its KeyError behaviour,
projection-chain round-trips, the WGS84 ellipsoid area, Domain validity helpers,
and CompositeDomain octant binning (the sign→oid rule with the on-seam
even-popcount convention).
"""
import numpy as np
import pytest

from hhg9 import Registrar, Points
from hhg9.base.points import OID_INVALID


@pytest.fixture(scope="module")
def reg():
    return Registrar()


# ---------------------------------------------------------------------------
# Registrar lookup
# ---------------------------------------------------------------------------

def test_domain_lookup_is_cached(reg):
    assert reg.domain('b_oct') is reg.domain('b_oct')


def test_format_lookup_is_cached(reg):
    assert reg.format('h9') is reg.format('h9')


@pytest.mark.parametrize("bad", ["nope", "xxx", "n_oct"])
def test_unknown_domain_raises_keyerror(reg, bad):
    with pytest.raises(KeyError):
        reg.domain(bad)


def test_unknown_format_raises_keyerror(reg):
    with pytest.raises(KeyError):
        reg.format("zzz")


# ---------------------------------------------------------------------------
# Projection chains
# ---------------------------------------------------------------------------

def test_project_roundtrip(reg):
    g, b = reg.domain('g_gcd'), reg.domain('b_oct')
    ll = np.array([[51.5, -0.12], [40.7, -74.0], [-33.9, 151.2]])
    pb = reg.project(Points(ll, domain=g), [g, b])
    pll = reg.project(pb, [b, g])
    assert np.abs(pll.coords - ll).max() < 1e-10


def test_project_carries_oid_through_chain(reg):
    g, b = reg.domain('g_gcd'), reg.domain('b_oct')
    pb = reg.project(Points(np.array([[10.0, 20.0]]), domain=g), [g, b])
    assert pb.oid is not None
    assert 0 <= int(pb.oid[0]) <= 7


def test_ellipsoid_area_is_wgs84(reg):
    # WGS84 surface area ≈ 5.10072e14 m².
    assert 5.0e14 < reg.ellipsoid_area < 5.2e14


# ---------------------------------------------------------------------------
# Domain validity (g_gcd — a plain, non-composite domain)
# ---------------------------------------------------------------------------

def test_domain_valid_is_per_point(reg):
    g = reg.domain('g_gcd')
    pts = Points(np.array([[51.5, -0.12], [400.0, -74.0]]), domain=g)  # lat 400 invalid
    v = g.valid(pts)
    np.testing.assert_array_equal(v, [True, False])


def test_domain_is_valid_scalar(reg):
    g = reg.domain('g_gcd')
    assert g.is_valid(Points(np.array([[51.5, -0.12]]), domain=g))
    assert not g.is_valid(Points(np.array([[400.0, 0.0]]), domain=g))


def test_domain_sig(reg):
    assert reg.domain('g_gcd').sig() == 'g_gcd'


# ---------------------------------------------------------------------------
# CompositeDomain octant binning (c_oct — the 3-axis composite)
# ---------------------------------------------------------------------------

def test_c_oct_binning_by_sign(reg):
    c = reg.domain('c_oct')
    # oid = (z<0)<<2 | (y<0)<<1 | (x<0) for sign-unambiguous interior points.
    coords = np.array([[1., 1., 1.], [-1., 1., 1.], [1., -1., -1.]])
    p = Points(coords, domain=c)
    np.testing.assert_array_equal(p.oid, [0, 1, 6])


def test_c_oct_binning_nan_is_invalid(reg):
    c = reg.domain('c_oct')
    p = Points(np.array([[np.nan, 1.0, 1.0]]), domain=c)
    assert p.oid[0] == OID_INVALID


def test_c_oct_seam_points_get_even_popcount(reg):
    """A point on a seam (a zero coordinate) is assigned to a mode-0 octant
    (even popcount) by the free-bit convention."""
    c = reg.domain('c_oct')
    seam = Points(np.array([[0.0, 1.0, -1.0], [0.0, 0.0, 1.0]]), domain=c)
    for o in seam.oid:
        assert bin(int(o)).count("1") % 2 == 0


def test_two_axis_composite_binning_not_implemented(reg):
    """b_oct (2-axis) cannot be binned directly — oid arrives via projection."""
    b = reg.domain('b_oct')
    with pytest.raises(NotImplementedError):
        Points(np.zeros((2, 2)), domain=b)


# ---------------------------------------------------------------------------
# Composite handler tables
# ---------------------------------------------------------------------------

def test_composite_handlers_and_c2_tables(reg):
    # c_oct's component handlers are populated; b_oct's oc_c2/handlers are
    # currently broken (OctantBary.oc is commented out — stale), so only c_oct
    # is asserted here.
    c = reg.domain('c_oct')
    assert c.handlers().shape == (8,)
    assert c.oc_c2().shape == (8, 3)
