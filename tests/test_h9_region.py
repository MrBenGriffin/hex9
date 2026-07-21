# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""
Behavioural / invariant tests for the PRODUCTION region engine
(``hhg9.h9.region``).

These bind to the public production API — ``xy_regions``, ``regions_xy``,
``region_neighbours`` and the ``H9R`` constant tables — rather than to any
internal helper.  They assert mathematical *invariants* (round-trips,
LUT self-consistency, structural validity, neighbour geometry) so that the
suite survives a refactor of the internals: if a future change preserves the
observable contract the tests stay green, and if it silently breaks the
contract they fail.

This module supersedes the old ``tests/test_h9_engine.py`` (deleted 2026-07-21),
which exercised an in-test *reimplementation* of this engine — the obsolete
"ugc" reference, GridConstants/GridRegions/GridNeighbours — and never imported
hhg9 at all, so it could not have caught a regression in the shipping code.
"""
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from hhg9.h9.region import xy_regions, regions_xy, region_neighbours, H9R
from hhg9.h9.lattice import H9C


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _sample_points(n, seed, spread=0.3):
    """Random in-scope barycentric points + modes (both Up and Down)."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-spread, spread, size=(n, 2))
    mode = rng.integers(0, 2, size=n).astype(np.uint8)
    return xy, mode


# Known regression points carried over from the obsolete reference suite.
# Coordinates are barycentric (x, y); both are Up-mode (mode=1) interior points.
_NAMED_POINTS = {
    "NAΛ centroid":  (-0.210025776447247, -0.020916439797272, 1),
    "SPΛ bad girl":  (+0.105560608419664, -0.134651734661425, 1),
}


# ---------------------------------------------------------------------------
# 1. H9R constant-table structural invariants  (pure — no traversal)
# ---------------------------------------------------------------------------

def test_proto_and_invalid_constants():
    assert H9R.proto_dn == 0x49
    assert H9R.proto_up == 0x16
    assert tuple(int(p) for p in H9R.proto) == (0x49, 0x16)
    assert H9R.invalid_region == 0x5F


def test_ids_are_zero_to_eleven():
    assert list(int(i) for i in H9R.ids) == list(range(12))


def test_supercell_membership_counts():
    # 9 Up cells, 9 Down cells, 12 distinct in-scope cells (3 shared).
    assert len(H9R.downs) == 9
    assert len(H9R.ups) == 9
    assert int(np.asarray(H9R.is_in).sum()) == 12


def test_child_lut_shape():
    assert np.asarray(H9R.child).shape == (2, 3, 3)


def test_child_cells_match_supercells():
    """The 9 children of each mode are exactly that mode's supercell."""
    assert set(np.asarray(H9R.child[0]).reshape(-1).tolist()) == set(H9R.downs.tolist())
    assert set(np.asarray(H9R.child[1]).reshape(-1).tolist()) == set(H9R.ups.tolist())


@pytest.mark.parametrize("mode", [0, 1])
def test_child_cells_unique_within_mode(mode):
    cells = np.asarray(H9R.child[mode]).reshape(-1)
    assert len(set(cells.tolist())) == cells.size, "child cells collide within a mode"


@pytest.mark.parametrize("mode", [0, 1])
def test_child_mcc2_reverse_map(mode):
    """``mcc2`` is the inverse of ``child``: every child cell maps back to the
    c2 wedge that produced it.  This is the production analogue of the old
    ``pc_c1`` reverse-map check."""
    for c2 in range(3):
        for cell in H9R.child[mode, c2]:
            assert H9R.mcc2[mode, int(cell)] == c2, (
                f"mcc2[{mode}, 0x{int(cell):02x}] != {c2}")


# ---------------------------------------------------------------------------
# 2. xy_regions — address structure
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("depth", [1, 4, 12, 34])
def test_address_shape_and_root(depth):
    xy, mode = _sample_points(20, seed=1)
    addr = xy_regions(xy, mode, depth=depth)
    # depth+2 columns: [proto] + depth real layers + 1 trailing slot.
    assert addr.shape == (20, depth + 2)
    assert addr.dtype == np.uint8
    # Root column encodes the initial mode via the proto IDs.
    expected_root = np.where(mode == 1, H9R.proto_up, H9R.proto_dn)
    assert_array_equal(addr[:, 0], expected_root.astype(np.uint8))


def test_each_step_is_a_legal_child_of_its_parent():
    """Every non-invalid region is a canonical child of its parent under the
    parent's mode (structural validity, vectorised over many points)."""
    xy, mode = _sample_points(40, seed=5)
    addr = xy_regions(xy, mode, depth=20)
    mode_of = H9C.mode
    for r in range(addr.shape[0]):
        for i in range(1, addr.shape[1] - 1):
            par, cur = int(addr[r, i - 1]), int(addr[r, i])
            if cur == H9R.invalid_region or par == H9R.invalid_region:
                continue
            pmo = int(mode_of[par])
            c2 = int(H9R.mcc2[pmo, cur])
            assert c2 != H9R.invalid_region, (r, i, hex(par), hex(cur))
            assert cur in H9R.child[pmo, c2], (
                f"row {r} step {i}: 0x{cur:02x} not a child of 0x{par:02x}")


def test_out_of_scope_point_is_invalid():
    """A point far outside the supercell yields no valid in-scope region, and
    canonicalises to the single ``invalid_region`` sentinel regardless of mode.

    Regression: the classifier emits mode-dependent overflow codes (0x0F for an
    Up apex, 0x5F for a Down apex).  Both are real geometric cells, so before
    the canonicalisation fix the 0x0F leak was NOT recognised as invalid by
    ``regions_xy`` and a far Up point decoded to garbage while a far Down point
    correctly decoded to the origin.  Now both collapse to invalid_region.
    """
    xy = np.array([[10.0, 10.0], [100.0, -50.0]])  # one Up apex, one Down apex
    mode = np.array([1, 0], dtype=np.uint8)
    addr = xy_regions(xy, mode, depth=5)
    is_in = np.asarray(H9R.is_in)
    # No real layer (cols 1..depth) classifies as an in-scope cell ...
    assert not is_in[addr[:, 1:-1]].any()
    # ... and every out-of-scope layer is the *same* canonical sentinel.
    assert np.all(addr[:, 1:-1] == H9R.invalid_region)


def test_out_of_scope_decodes_consistently():
    """Both modes of an out-of-scope point must decode identically (to the cell
    origin), not just the Down case.  This is the user-visible symptom of the
    0x0F/0x5F leak."""
    xy = np.array([[10.0, 10.0], [100.0, -50.0]])
    mode = np.array([1, 0], dtype=np.uint8)
    dec = regions_xy(xy_regions(xy, mode, depth=5))
    np.testing.assert_array_equal(dec[:, :2], np.zeros((2, 2)))


# ---------------------------------------------------------------------------
# 3. regions_xy — decode round-trip (the core contract)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("depth,atol", [(34, 1e-12), (20, 1e-7)])
def test_decode_recovers_coordinates(depth, atol):
    """regions_xy ∘ xy_regions recovers (x, y) within the truncation residual,
    and recovers the initial mode exactly."""
    xy, mode = _sample_points(50, seed=7)
    addr = xy_regions(xy, mode, depth=depth)
    dec = regions_xy(addr)
    assert dec.shape == (50, 3)
    np.testing.assert_allclose(dec[:, :2], xy, atol=atol)
    assert_array_equal(dec[:, 2].astype(np.uint8), mode)


@pytest.mark.parametrize("depth", [2, 3, 4, 6])
def test_address_is_idempotent_at_shallow_depth(depth):
    """At shallow depth the decoded representative re-encodes to the *same*
    address — i.e. the address is the canonical label of its cell."""
    xy, mode = _sample_points(30, seed=2, spread=0.25)
    addr = xy_regions(xy, mode, depth=depth)
    dec = regions_xy(addr)
    re_addr = xy_regions(dec[:, :2], dec[:, 2].astype(np.uint8), depth=depth)
    assert_array_equal(re_addr, addr)


@pytest.mark.parametrize("name", list(_NAMED_POINTS))
def test_named_regression_points_roundtrip(name):
    x, y, m = _NAMED_POINTS[name]
    xy = np.array([[x, y]])
    mode = np.array([m], dtype=np.uint8)
    addr = xy_regions(xy, mode, depth=34)
    dec = regions_xy(addr)
    np.testing.assert_allclose(dec[:, :2], xy, atol=1e-12, err_msg=name)
    assert int(dec[0, 2]) == m, name


# ---------------------------------------------------------------------------
# 4. region_neighbours
# ---------------------------------------------------------------------------

def test_region_neighbours_returns_array_and_c2():
    xy, mode = _sample_points(8, seed=3)
    addr = xy_regions(xy, mode, depth=12)
    out = region_neighbours(addr)
    assert isinstance(out, tuple) and len(out) == 2
    nb, c2 = out
    assert nb.shape == addr.shape
    assert nb.dtype == np.uint8
    assert c2.shape == (addr.shape[0],)
    assert np.all((c2 == H9R.invalid_region) | ((c2 >= 0) & (c2 <= 2)))


def test_region_neighbours_root_is_a_proto():
    """The neighbour address must still start at a valid proto root."""
    xy, mode = _sample_points(16, seed=11)
    addr = xy_regions(xy, mode, depth=10)
    nb, _ = region_neighbours(addr)
    roots = set(nb[:, 0].tolist())
    assert roots <= {H9R.proto_up, H9R.proto_dn}


def test_neighbour_is_geometrically_self_consistent():
    """A neighbour address, decoded to coordinates and re-encoded at the same
    depth, must reproduce itself — i.e. the neighbour the LUTs name is the same
    cell geometry names.  Exercises the transform-then-classify path including
    multi-layer mode-flip cascades."""
    xy, mode = _sample_points(24, seed=13)
    depth = 14
    addr = xy_regions(xy, mode, depth=depth)
    nb, _ = region_neighbours(addr)
    dec = regions_xy(nb)
    re = xy_regions(dec[:, :2], dec[:, 2].astype(np.uint8), depth=depth)
    assert_array_equal(re, nb)
