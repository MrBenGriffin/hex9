# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""
Tests for the domain-aware coordinate container ``hhg9.base.points.Points``.

Mostly pure-Python container behaviour: construction/oid broadcasting, the
octant-id sign utilities (round-trip), class_mode/cm, copy/concat/select/
__getitem__, and bbox.  No projection chain needed except where a domain name
is required (bbox uses a real b_oct domain with an explicit oid to skip binning).
"""
import numpy as np
import pytest

from hhg9 import Registrar, Points
from hhg9.base.points import OID_INVALID


# ---------------------------------------------------------------------------
# Construction & oid handling
# ---------------------------------------------------------------------------

def test_scalar_oid_broadcasts():
    p = Points(np.zeros((5, 2)), oid=np.uint8(3))
    assert p.oid.shape == (5,)
    assert np.all(p.oid == 3)


def test_array_oid_preserved():
    oid = np.array([0, 1, 2, 3], dtype=np.uint8)
    p = Points(np.zeros((4, 2)), oid=oid)
    np.testing.assert_array_equal(p.oid, oid)


def test_bad_oid_shape_raises():
    with pytest.raises(ValueError):
        Points(np.zeros((4, 2)), oid=np.array([1, 2, 3], dtype=np.uint8))


def test_len():
    assert len(Points(np.zeros((7, 2)))) == 7


def test_oids_alias_and_cm_unset():
    p = Points(np.zeros((3, 2)))
    assert p.oids() is None
    assert p.cm() == (None, None)


# ---------------------------------------------------------------------------
# Octant-id sign utilities
# ---------------------------------------------------------------------------

def test_octant_id_invert_calc_roundtrip():
    oc = np.arange(8, dtype=np.uint8)
    comp = Points.invert_octant_ids(oc)
    assert comp.shape == (8, 3)
    assert set(np.unique(comp).tolist()) <= {-1, 1}
    np.testing.assert_array_equal(Points.calc_octant_ids(comp), oc)


def test_class_mode_is_popcount_parity():
    oc = np.arange(8, dtype=np.uint8)
    oid, mode = Points.class_mode(oc)
    expected = np.array([bin(i).count("1") % 2 for i in range(8)], dtype=np.uint8)
    np.testing.assert_array_equal(oid, oc)
    np.testing.assert_array_equal(mode, expected)


def test_cm_returns_oid_and_mode_when_set():
    p = Points(np.zeros((8, 2)), oid=np.arange(8, dtype=np.uint8))
    oid, mode = p.cm()
    np.testing.assert_array_equal(oid, np.arange(8))
    np.testing.assert_array_equal(mode, [bin(i).count("1") % 2 for i in range(8)])


# ---------------------------------------------------------------------------
# copy / select / __getitem__
# ---------------------------------------------------------------------------

def test_copy_is_independent():
    p = Points(np.ones((3, 2)), oid=np.zeros(3, np.uint8), samples=np.arange(3))
    q = p.copy()
    q.coords[0, 0] = 99
    q.oid[0] = 7
    assert p.coords[0, 0] == 1          # original untouched
    assert p.oid[0] == 0


def test_select_filters_all_arrays():
    p = Points(np.arange(8).reshape(4, 2), oid=np.arange(4, dtype=np.uint8),
               samples=np.arange(4))
    mask = np.array([True, False, True, False])
    s = p.select(mask)
    assert len(s) == 2
    np.testing.assert_array_equal(s.oid, [0, 2])
    np.testing.assert_array_equal(s.samples, [0, 2])


def test_select_length_mismatch_raises():
    p = Points(np.zeros((4, 2)))
    with pytest.raises(ValueError):
        p.select(np.array([True, False]))


def test_getitem_returns_singleton_points():
    p = Points(np.arange(8).reshape(4, 2), oid=np.arange(4, dtype=np.uint8))
    one = p[2]
    assert isinstance(one, Points)
    assert len(one) == 1
    np.testing.assert_array_equal(one.coords[0], [4, 5])


def test_getitem_tuple_index_raises():
    p = Points(np.zeros((4, 2)))
    with pytest.raises(TypeError):
        _ = p[1, 0]


# ---------------------------------------------------------------------------
# concat
# ---------------------------------------------------------------------------

def test_concat_combines_points():
    a = Points(np.zeros((2, 2)), oid=np.zeros(2, np.uint8))
    b = Points(np.ones((3, 2)), oid=np.ones(3, np.uint8))
    c = Points.concat([a, b])
    assert len(c) == 5
    np.testing.assert_array_equal(c.oid, [0, 0, 1, 1, 1])


def test_concat_fills_missing_oid_with_sentinel():
    a = Points(np.zeros((2, 2)), oid=np.array([5, 6], np.uint8))
    b = Points(np.ones((2, 2)))                       # no oid
    c = Points.concat([a, b])
    np.testing.assert_array_equal(c.oid, [5, 6, OID_INVALID, OID_INVALID])


def test_concat_empty_raises():
    with pytest.raises(ValueError):
        Points.concat([])


def test_concat_non_points_raises():
    with pytest.raises(TypeError):
        Points.concat([Points(np.zeros((1, 2))), np.zeros((1, 2))])


def test_concat_different_domains_raises(reg_fixture):
    g, b = reg_fixture.domain('g_gcd'), reg_fixture.domain('b_oct')
    pa = Points(np.zeros((1, 2)), domain=g)
    pb = Points(np.zeros((1, 2)), domain=b, oid=np.zeros(1, np.uint8))
    with pytest.raises(ValueError):
        Points.concat([pa, pb])


# ---------------------------------------------------------------------------
# bbox / image / repr
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def reg_fixture():
    return Registrar()


def test_bbox_xy_order(reg_fixture):
    b = reg_fixture.domain('b_oct')
    p = Points(np.array([[0.1, 0.2], [0.3, -0.1], [-0.2, 0.05]]),
               domain=b, oid=np.zeros(3, np.uint8))
    assert p.bbox() == pytest.approx((-0.2, -0.1, 0.3, 0.2))
    # trbl reorders to (top, right, bottom, left)
    assert p.bbox(trbl=True) == pytest.approx((0.2, 0.3, -0.1, -0.2))
    _, diff = p.bbox(return_diff=True)
    assert diff == pytest.approx((0.5, 0.3))


def test_image_shape():
    coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    p = Points(coords, samples=np.arange(4))
    img = p.image((4, 4))
    assert img.shape == (4, 4, 1)


def test_repr_mentions_points():
    assert "Points" in repr(Points(np.zeros((2, 2))))
