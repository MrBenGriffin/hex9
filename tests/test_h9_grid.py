# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""
Coverage for ``hhg9.h9.grid`` beyond the structural mesh checks in
``test_hex_mesh.py``:

  * ideal area/length tables (``hex_props`` / ``tri_props``) — area conservation
    and the regular-polygon geometric identities,
  * ``densify_step_for_layer``,
  * the antimeridian helpers ``valid_ll`` / ``clipped_ll``,
  * ``HexMesh`` accessor properties (layers / addr / addrs / __getitem__ / repr),
  * ``HexMesh.create_clipped`` behaviour.

All bind to the public surface and assert invariants, not coordinates.
"""
import numpy as np
import pytest

from hhg9.h9.grid import (
    hex_props, tri_props, densify_step_for_layer,
    valid_ll, clipped_ll, HexMesh, h9_postgis_hexagons,
)


@pytest.fixture(scope="module")
def reg():
    from hhg9 import Registrar
    return Registrar()


# ---------------------------------------------------------------------------
# 1. Ideal area / length tables
# ---------------------------------------------------------------------------

def test_hex_props_table_shape():
    table = hex_props(None, 1.0)
    assert table.shape == (64, 5)
    # A single level returns one row of that table.
    np.testing.assert_array_equal(hex_props(0, 1.0), table[0])


@pytest.mark.parametrize("L", [0, 1, 2, 3, 5])
def test_hex_area_conservation(L):
    """12·9^L hexes of the layer-L area tile the whole sphere."""
    area = hex_props(L, 1.0)[0]
    assert 12 * 9 ** L * area == pytest.approx(1.0)


@pytest.mark.parametrize("L", [0, 1, 2, 3, 5])
def test_tri_area_conservation(L):
    """8·9^L triangles of the layer-L area tile the whole sphere."""
    area = tri_props(L, 1.0)[0]
    assert 8 * 9 ** L * area == pytest.approx(1.0)


def test_hex_geometric_identities():
    area, side, inradius, flat_d, point_d = hex_props(2, 1.0)
    assert inradius == pytest.approx((np.sqrt(3) / 2) * side)
    assert flat_d == pytest.approx(2 * inradius)
    assert point_d == pytest.approx(2 * side)
    # area = (3√3/2)·side²
    assert area == pytest.approx((3 * np.sqrt(3) / 2) * side ** 2)


def test_tri_geometric_identities():
    area, side, inradius, height, big_r = tri_props(3, 1.0)
    assert area == pytest.approx((np.sqrt(3) / 4) * side ** 2)
    assert height == pytest.approx((np.sqrt(3) / 2) * side)
    assert inradius == pytest.approx((np.sqrt(3) / 6) * side)
    assert big_r == pytest.approx((np.sqrt(3) / 3) * side)


@pytest.mark.parametrize("kind", ["hex", "tri"])
def test_densify_step_is_scaled_inradius(kind):
    props = tri_props(3) if kind == "tri" else hex_props(3)
    expected = float(props[2]) * 0.9 / 2.0
    assert densify_step_for_layer(3, kind, factor=2.0, safety=0.9) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 2. Antimeridian helpers
# ---------------------------------------------------------------------------

def test_valid_ll_flags():
    arr = np.array([
        [[0, 0], [0, 10], [10, 10]],       # finite, lon span 10° → valid
        [[0, 170], [0, -170], [0, 175]],   # lon span 345° → invalid
        [[0, 0], [np.nan, 5], [1, 1]],     # NaN → invalid
    ], dtype=float)
    np.testing.assert_array_equal(valid_ll(arr), [True, False, False])


def test_clipped_ll_drops_nan_and_splits_dateline():
    arr = np.array([
        [[0, 0], [0, 10], [10, 10]],       # untouched → 3 verts
        [[0, 170], [0, -170], [0, 175]],   # dateline jump → truncated to 1 vert
        [[0, 0], [np.nan, 5], [1, 1]],     # NaN dropped → 2 verts
    ], dtype=float)
    out = clipped_ll(arr)
    assert [len(c) for c in out] == [3, 1, 2]


# ---------------------------------------------------------------------------
# 3. HexMesh accessor properties
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mesh01(reg):
    return HexMesh.create([0, 1], reg)


def test_mesh_layers_and_fine(mesh01):
    assert mesh01.layers == (0, 1)
    assert mesh01.fine == 1


def test_mesh_getitem_counts(mesh01):
    assert mesh01[0].shape == (12, 6)
    assert mesh01[1].shape == (108, 6)        # 12·9
    # .faces is the finest layer
    np.testing.assert_array_equal(mesh01.faces, mesh01[1])


def test_mesh_addr_and_addrs(mesh01):
    assert len(mesh01.addr(0)) == 12
    assert len(mesh01.addr(1)) == 108
    assert len(mesh01.addrs) == 108           # addrs = finest layer addresses


def test_mesh_repr_is_informative(mesh01):
    r = repr(mesh01)
    assert "HexMesh" in r and "hexes" in r


# ---------------------------------------------------------------------------
# 4. create_clipped
# ---------------------------------------------------------------------------

# A lat/lon box over western Europe.
_BIG_POLY = np.array([[30.0, -20.0], [60.0, -20.0], [60.0, 40.0], [30.0, 40.0]])


@pytest.fixture(scope="module")
def clipped_l2(reg):
    return HexMesh.create_clipped(2, _BIG_POLY, reg)


def test_create_clipped_is_nonempty_strict_subset(clipped_l2):
    n = len(clipped_l2.faces)
    assert n > 0
    assert n < 12 * 9 ** 2          # fewer than the full global layer-2 mesh


def test_create_clipped_oids_valid(clipped_l2):
    oids = clipped_l2.pts.oid
    assert not np.any(oids == 255)
    assert np.all(np.isin(oids, np.arange(8)))


def test_create_clipped_finer_layer_has_more_faces(reg):
    n2 = len(HexMesh.create_clipped(2, _BIG_POLY, reg).faces)
    n3 = len(HexMesh.create_clipped(3, _BIG_POLY, reg).faces)
    assert n3 > n2


# ---------------------------------------------------------------------------
# 5. h9_postgis_hexagons
# ---------------------------------------------------------------------------
# Regression: h9_encode/h9_enc return a single list[uuid.UUID] (the self-
# inverting canonical bin) — the old (uuids, adr_bytes) 2-tuple was removed.
# h9_postgis_hexagons previously unpacked `full_uuids, _ = h9_encode(...)` and
# crashed; the call site and the docstrings are now aligned to the single return.

def test_h9_postgis_hexagons_returns_wkt_triples(reg):
    tris = h9_postgis_hexagons(-0.5, 51.0, 0.5, 51.6, 3, reg=reg)
    assert len(tris) > 0
    assert all(len(t) == 3 for t in tris)              # (full_uuid, bin_uuid, wkt)
    assert tris[0][2].startswith("POLYGON")            # third element is the WKT


def test_h9_encode_returns_list_of_uuids(reg):
    """The modern contract: a single list[uuid.UUID], not a tuple or ndarray."""
    import uuid as uuid_mod
    from hhg9.h9.uuid_address import h9_encode
    out = h9_encode([51.5, 40.7], [-0.12, -74.0], reg=reg)
    assert isinstance(out, list)
    assert len(out) == 2
    assert all(isinstance(u, uuid_mod.UUID) for u in out)
