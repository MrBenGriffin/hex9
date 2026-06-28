# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""
Invariant tests for the hex-binning layer (``hhg9.h9.binning``), the region/
nibble-space aggregation used by polygon rendering and PostGIS export.

Bound to the public functions (also re-exported via ``polygon.py``):
  hex_reduce, hex_parents, ctr_from_pars, hex_from_pars,
  hex_poly_groups, hex_poly_layer, hh_layer.

Assertions are conservation/shape invariants (every point lands in exactly one
hex; H·6 polygon vertices; scale = 3⁻ᴸ; coarsening monotonicity), not
coordinates.
"""
import numpy as np
import pytest

from hhg9 import Registrar, Points
import hhg9.h9.binning as bn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def reg():
    return Registrar()


@pytest.fixture(scope="module")
def pb(reg):
    """200 b_oct points: a tight London cluster plus a global scatter."""
    g, b = reg.domain('g_gcd'), reg.domain('b_oct')
    rng = np.random.default_rng(2)
    lat = np.r_[51.5 + rng.normal(0, 0.05, 150), rng.uniform(-60, 60, 50)]
    lon = np.r_[-0.12 + rng.normal(0, 0.05, 150), rng.uniform(-150, 150, 50)]
    return reg.project(Points(np.column_stack([lat, lon]), domain=g), [g, b])


LAYER = 8


# ---------------------------------------------------------------------------
# 1. hex_reduce
# ---------------------------------------------------------------------------

def test_hex_reduce_structure(pb):
    n = len(pb)
    hex_num, hex_v, hex_inv, hex_idx = bn.hex_reduce(pb, LAYER)
    assert hex_v.shape == (hex_num, LAYER + 2)
    assert hex_inv.shape == (n,)
    assert hex_inv.min() >= 0 and hex_inv.max() < hex_num   # every point → a hex
    assert hex_idx.shape == (hex_num,)
    assert set(np.unique(hex_inv).tolist()) == set(range(hex_num))  # no empty hex


def test_hex_reduce_coarsening_monotonic(pb):
    counts = [bn.hex_reduce(pb, L)[0] for L in (2, 4, 8)]
    assert counts == sorted(counts)


# ---------------------------------------------------------------------------
# 2. hex_parents / ctr_from_pars / hex_from_pars
# ---------------------------------------------------------------------------

def test_hex_parents_shapes_and_scale(reg, pb):
    b = reg.domain('b_oct')
    hex_num, hex_v, _, _ = bn.hex_reduce(pb, LAYER)
    hex_xy, hex_oid, scale = bn.hex_parents(b, hex_v, hex_num, LAYER)
    assert hex_xy.shape == (hex_num, 2)
    assert hex_oid.shape == (hex_num,)
    assert scale == pytest.approx(3.0 ** -LAYER)


def test_ctr_from_pars_returns_one_centroid_per_hex(reg, pb):
    b = reg.domain('b_oct')
    hex_num, hex_v, _, _ = bn.hex_reduce(pb, LAYER)
    hex_xy, hex_oid, scale = bn.hex_parents(b, hex_v, hex_num, LAYER)
    ctr = bn.ctr_from_pars(b, hex_xy, hex_oid, scale, hex_v[:, -1])
    assert ctr.coords.shape == (hex_num, 2)
    np.testing.assert_array_equal(ctr.oid, hex_oid)


def test_hex_from_pars_has_six_verts_per_hex(reg, pb):
    b = reg.domain('b_oct')
    hex_num, hex_v, _, _ = bn.hex_reduce(pb, LAYER)
    hex_xy, hex_oid, scale = bn.hex_parents(b, hex_v, hex_num, LAYER)
    hx = bn.hex_from_pars(b, hex_xy, hex_oid, scale, hex_v[:, -1])
    assert hx.coords.shape == (hex_num * 6, 2)
    assert hx.oid.shape == (hex_num * 6,)


# ---------------------------------------------------------------------------
# 3. hex_poly_groups
# ---------------------------------------------------------------------------

def test_hex_poly_groups_conserves_population(pb):
    n = len(pb)
    hx_pts, inv_hex, counts, idx = bn.hex_poly_groups(pb, LAYER)
    hex_num = counts.shape[0]
    assert counts.sum() == n                       # every point counted once
    assert hx_pts.coords.shape == (hex_num * 6, 2)
    assert inv_hex.min() >= 0 and inv_hex.max() < hex_num
    assert idx.shape == (hex_num,)


def test_hex_poly_groups_rejects_non_oct_domain(reg, pb):
    g = reg.domain('g_gcd')
    ll = reg.project(pb, [reg.domain('b_oct'), g])
    with pytest.raises(ValueError):
        bn.hex_poly_groups(ll, LAYER)


# ---------------------------------------------------------------------------
# 4. hex_poly_layer
# ---------------------------------------------------------------------------

def test_hex_poly_layer_default_reducer(pb):
    """With no per-point samples the default reducer averages 1s → all-ones."""
    hx_pts, values = bn.hex_poly_layer(pb, LAYER)
    hex_num = hx_pts.coords.shape[0] // 6
    assert values.shape == (hex_num,)
    np.testing.assert_allclose(values, 1.0)


def test_hex_poly_layer_custom_reducer_receives_counts(pb):
    def reducer(inv_hex, counts, pts):
        return counts.astype(float)
    hx_pts, values = bn.hex_poly_layer(pb, LAYER, reducer=reducer)
    _, _, counts, _ = bn.hex_poly_groups(pb, LAYER)
    np.testing.assert_array_equal(values, counts.astype(float))


def test_hex_poly_layer_bad_reducer_shape_raises(pb):
    with pytest.raises(ValueError):
        bn.hex_poly_layer(pb, LAYER, reducer=lambda inv, c, p: np.zeros(3))


# ---------------------------------------------------------------------------
# 5. hh_layer (half-hexagons)
# ---------------------------------------------------------------------------

def test_hh_layer_structure_and_population(pb):
    n = len(pb)
    polygon, pops, inv, oc_poly4 = bn.hh_layer(pb, LAYER)
    num_hh = polygon.shape[0]
    assert polygon.shape == (num_hh, 4, 2)         # 4 verts per half-hex
    assert oc_poly4.shape == (num_hh, 4)
    assert pops.sum() == n                         # population conserved
    assert inv.min() >= 0 and inv.max() < num_hh
