# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""
Round-trip and invariant tests for the UUID address layer
(``hhg9.h9.uuid_address``).

The modern contract (see module docstring): a full H9 UUID is the canonical bin
at ``UUID_DEPTH`` with a single-nibble reversible tail — it is self-inverting,
so ``h9_decode(h9_encode(...))`` recovers lat/lon and no companion bytes exist.

Covered:
  * nibble ↔ int packing
  * h9_encode/h9_decode (lat/lon) and h9_enc/h9_dec/h9_enc_ext (b_oct)
  * h9_bin / h9_bin_pts — layer validation, idempotence, coarsening monotonicity
  * h9_layer inference
  * h9_label / h9_from_label string round-trip
"""
import uuid as uuid_mod

import numpy as np
import pytest

from hhg9 import Registrar, Points
import hhg9.h9.uuid_address as ua


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def reg():
    return Registrar()


_LL = np.array([
    [51.5,  -0.12],
    [40.7, -74.00],
    [-33.9, 151.2],
    [0.0,    0.0],
])


@pytest.fixture(scope="module")
def ll():
    return _LL.copy()


@pytest.fixture(scope="module")
def uuids(reg, ll):
    return ua.h9_encode(ll[:, 0], ll[:, 1], reg=reg)


@pytest.fixture(scope="module")
def cluster_uuids(reg):
    """200 points tightly clustered near London — collapse at coarse layers."""
    rng = np.random.default_rng(1)
    lat = 51.5 + rng.normal(0, 0.01, 200)
    lon = -0.12 + rng.normal(0, 0.01, 200)
    return ua.h9_encode(lat, lon, reg=reg)


# ---------------------------------------------------------------------------
# 1. nibble ↔ int packing
# ---------------------------------------------------------------------------

def test_nibble_int_batch_roundtrip():
    nb = np.random.default_rng(0).integers(0, 16, size=(5, 32)).astype(np.uint8)
    ints = ua.batch_nibbles_to_int(nb)
    np.testing.assert_array_equal(ua._batch_int_to_nibbles(ints, 32), nb)


def test_nibble_int_single_matches_batch():
    nb = np.arange(32, dtype=np.uint8) % 16   # one 32-nibble row
    assert ua._nibbles_to_int(nb) == ua.batch_nibbles_to_int(nb.reshape(1, 32))[0]


def test_uuid_depth_constant():
    assert ua.UUID_DEPTH == 30


# ---------------------------------------------------------------------------
# 2. encode / decode round-trips
# ---------------------------------------------------------------------------

def test_encode_returns_uuid_list(uuids, ll):
    assert isinstance(uuids, list)
    assert len(uuids) == len(ll)
    assert all(isinstance(u, uuid_mod.UUID) for u in uuids)


def test_lat_lon_roundtrip(reg, uuids, ll):
    lat, lon = ua.h9_decode(uuids, reg=reg)
    assert np.abs(lat - ll[:, 0]).max() < 1e-10
    assert np.abs(lon - ll[:, 1]).max() < 1e-10


def test_b_oct_paths_agree(reg, ll, uuids):
    """h9_enc (b_oct) and h9_enc_ext must match the lat/lon wrapper h9_encode."""
    g, b = reg.domain('g_gcd'), reg.domain('b_oct')
    pb = reg.project(Points(ll, domain=g), [g, b])
    oc, mo = pb.cm()
    assert list(ua.h9_enc(pb)) == list(uuids)
    assert list(ua.h9_enc_ext(pb, oc, mo)) == list(uuids)


def test_full_uuid_layer_is_uuid_depth(uuids):
    np.testing.assert_array_equal(ua.h9_layer(uuids), ua.UUID_DEPTH)


# ---------------------------------------------------------------------------
# 3. binning
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("layer", [0, 2, 5, 15, 30])
def test_bin_reports_its_layer(uuids, layer):
    bins = ua.h9_bin(list(uuids), layer)
    np.testing.assert_array_equal(ua.h9_layer(bins), layer)


@pytest.mark.parametrize("layer", [0, 3, 12])
def test_bin_is_idempotent(uuids, layer):
    once = ua.h9_bin(list(uuids), layer)
    twice = ua.h9_bin(list(once), layer)
    assert once == twice


@pytest.mark.parametrize("bad_layer", [-1, 31, 100])
def test_bin_rejects_out_of_range_layer(uuids, bad_layer):
    with pytest.raises(ValueError):
        ua.h9_bin(list(uuids), bad_layer)


def test_bin_coarsening_is_monotonic(cluster_uuids):
    """Finer layers can only split, never merge: |unique bins| is non-decreasing
    in layer, and a tight cluster collapses to a single bin at coarse layers."""
    counts = [len(set(ua.h9_bin(list(cluster_uuids), L))) for L in (0, 2, 5, 10, 20)]
    assert counts == sorted(counts)
    assert counts[0] == 1                 # all in one hex at L0
    assert counts[-1] == len(cluster_uuids)  # all distinct by L20


# ---------------------------------------------------------------------------
# 4. label round-trip
# ---------------------------------------------------------------------------

def test_label_roundtrip(uuids):
    for u in uuids:
        label = ua.h9_label(u)
        assert isinstance(label, str)
        assert ua.h9_from_label(label) == u


# ---------------------------------------------------------------------------
# 5. hierarchy traversal: h9_ancestors / h9_descendants
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bpts(reg, ll):
    """The fixture lat/lons as b_oct Points."""
    g_gcd = reg.domain('g_gcd')
    b_oct = reg.domain('b_oct')
    return reg.project(Points(ll.copy(), g_gcd), [g_gcd, b_oct])


@pytest.mark.parametrize("at_layer,g", [(8, 1), (8, 3), (5, 5)])
def test_ancestors_match_relative_bin(bpts, at_layer, g):
    anc = ua.h9_ancestors(bpts, at_layer, g)
    expect = ua.h9_bin_pts(bpts, at_layer - g)
    assert anc == expect


def test_ancestors_g0_is_anchor(bpts):
    assert ua.h9_ancestors(bpts, 6, 0) == ua.h9_bin_pts(bpts, 6)


@pytest.mark.parametrize("at_layer,gu", [(2, 3), (0, 1), (30, 31)])
def test_ancestors_reject_out_of_range(bpts, at_layer, gu):
    with pytest.raises(ValueError):
        ua.h9_ancestors(bpts, at_layer, gu)


@pytest.mark.parametrize("at_layer,g", [(2, 1), (3, 2)])
def test_descendants_nest_and_layer(reg, bpts, at_layer, g):
    anchors = ua.h9_bin_pts(bpts, at_layer)
    desc = ua.h9_descendants(bpts, at_layer, g, reg=reg)
    target = at_layer + g
    assert len(desc) == len(anchors)
    for anchor, kids in zip(anchors, desc):
        assert kids, "non-empty descendant set expected"
        assert len(set(k.int for k in kids)) == len(kids)        # unique
        assert all(int(ua.h9_layer(k)) == target for k in kids)  # at target layer
        backs = ua.h9_cell_ancestor(list(kids), at_layer, reg=reg)
        assert all(b.int == anchor.int for b in backs)           # every kid nests
        assert len(kids) == 9 ** g                               # exactly 9^g


def test_descendants_g0_is_anchor(reg, bpts):
    anchors = ua.h9_bin_pts(bpts, 4)
    desc = ua.h9_descendants(bpts, 4, 0, reg=reg)
    assert [d[0] for d in desc] == list(anchors)
    assert all(len(d) == 1 for d in desc)


@pytest.mark.parametrize("at_layer,g", [(-1, 1), (5, 30), (10, -1)])
def test_descendants_reject_out_of_range(bpts, at_layer, g):
    with pytest.raises(ValueError):
        ua.h9_descendants(bpts, at_layer, g)


def test_descendants_complete_vs_bruteforce(reg):
    """The descendant set is exactly the target hexes whose canonical
    ancestor is the anchor (mode-0 convention).

    Cross-check against a dense in-anchor sample: every layer-(L+g) hex whose
    canonical ancestor at L equals the anchor must appear, and nothing else
    (samples cover the anchor and its rim, so neighbours' splits are seen
    and correctly excluded)."""
    from hhg9.h9 import H9O
    from hhg9.h9.classifier import in_scope, H9CL
    from hhg9.h9 import H9K
    at_layer, g = 2, 1
    g_gcd, b_oct = reg.domain('g_gcd'), reg.domain('b_oct')
    pt = reg.project(Points(np.array([[51.1789, -1.8262]]), g_gcd), [g_gcd, b_oct])
    anchor = ua.h9_bin_pts(pt, at_layer)[0]

    desc = set(d.int for d in ua.h9_descendants(pt, at_layer, g, reg=reg)[0])

    # brute force: dense grid inside the anchor, keep hexes binning to anchor
    ac = ua.h9_dec([anchor], b_oct).coords[0]
    oc0 = int(pt.cm()[0][0])
    mode0 = np.uint8(H9O.oid_mo[oc0])
    span = 0.80 * 3.0 ** -at_layer
    gx, gy = np.meshgrid(np.linspace(-span, span, 250), np.linspace(-span, span, 250))
    sx, sy = ac[0] + gx.ravel(), ac[1] + gy.ravel()
    md = np.full(sx.shape, mode0)
    ok = in_scope(H9K.R3 * sx, sy, md, H9CL)
    spts = Points(np.column_stack([sx[ok], sy[ok]]), b_oct, oid=np.full(int(ok.sum()), oc0))
    fine = ua.h9_bin_pts(spts, at_layer + g)
    uniq = list({f.int: f for f in fine}.values())
    back = ua.h9_cell_ancestor(uniq, at_layer, reg=reg)
    brute = set(f.int for f, b in zip(uniq, back) if b.int == anchor.int)

    assert desc == brute


# ---------------------------------------------------------------------------
# Canonical cell ancestry (mode-0 convention): h9_cell_parent / h9_cell_ancestor
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def anc_mesh(reg):
    from hhg9.h9.grid import HexMesh
    return HexMesh.create(range(4), reg)


@pytest.mark.parametrize("L", [1, 2, 3])
def test_cell_parent_nine_per_parent(reg, anc_mesh, L):
    """Every layer-(L-1) cell is canonical parent of exactly 9 layer-L cells."""
    from collections import Counter
    up = ua.h9_cell_parent(anc_mesh.addr(L), reg=reg)
    cnt = Counter(u.int for u in up)
    assert len(cnt) == 12 * 9 ** (L - 1)
    assert set(cnt.values()) == {9}


def test_cell_parent_london_known_answer(reg):
    """The §10b worked example: 43585 (interior child carrying the sibling
    lineage spelling) has canonical L3 ancestor 4348 — NOT its lineage cut
    4358, which canonically names a different hexagon elsewhere."""
    u = ua.h9_bin(ua.h9_encode(np.array([52.4365]), np.array([-0.9098])), 4, reg=reg)
    assert ua.h9_label(u[0]) == '43585.1'
    p = ua.h9_cell_parent(u, reg=reg)
    assert ua.h9_label(p[0]) == '4348.2'


def test_cell_ancestor_direct_not_composed(reg, anc_mesh):
    """ancestor is the leaf-reified d_cell relation, NOT parent∘parent.

    The subdivision tree is on d_cells (rep-9, nests exactly); the mode-0
    reification into x_cells happens once, at the leaf, so the ancestor is
    the single deep re-bin of the cell's mode-0 interior point. Composing
    h9_cell_parent level-by-level re-adjudicates splits at every layer and
    decoheres at nested splits (the hexagon grows deep tongues/voids —
    docs/dggs/dggs_nesting.py). Both relations are total and 9-regular
    (81 grandchildren per cell); they differ in membership at exactly the
    nested splits — 1/9 of cells."""
    from collections import Counter
    uu = anc_mesh.addr(3)
    two = ua.h9_cell_ancestor(uu, 1, reg=reg)
    composed = ua.h9_cell_parent(ua.h9_cell_parent(uu, reg=reg), reg=reg)
    assert set(Counter(u.int for u in two).values()) == {81}
    diffs = sum(a.int != b.int for a, b in zip(two, composed))
    assert diffs == len(uu) // 9      # nested splits distinguish the relations


def test_cell_ancestor_address_equals_geometric(reg, anc_mesh):
    """The two doctrine derivations cross-check: the address-space fold
    (production path) equals the geometric mode-0-interior deep re-bin."""
    from hhg9.h9.uuid_address import _mode0_interior_pts
    uu = anc_mesh.addr(3)
    b_oct = reg.domain('b_oct')
    pts = _mode0_interior_pts(list(uu), b_oct)
    for K in (0, 1, 2):
        addr = ua.h9_cell_ancestor(uu, K, reg=reg)
        geo = ua.h9_bin_pts(pts, K)
        assert [a.int for a in addr] == [g.int for g in geo]


def test_cell_ancestor_identity_and_errors(reg, anc_mesh):
    uu = anc_mesh.addr(2)[:5]
    assert ua.h9_cell_ancestor(uu, 2, reg=reg) == list(uu)
    with pytest.raises(ValueError):
        ua.h9_cell_ancestor(uu, 3, reg=reg)
    with pytest.raises(ValueError):
        ua.h9_cell_parent(anc_mesh.addr(0)[:3], reg=reg)
