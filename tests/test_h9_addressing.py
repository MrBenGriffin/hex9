# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""
Round-trip and invariant tests for the address packer layer
(``hhg9.h9.addressing``) — the engine beneath the public ``h9`` format.

Bound to the public packers/unpackers and the ``H9_RA`` region scheme:
  * hex_str_encode / hex_str_decode      (string addresses)
  * hex_digits / hex_decode              (nibble arrays)
  * hex_pack / hex_unpack                (canonical u64 bin, width 14)
  * reg_pack / reg_unpack                (region-based u64)
  * RegionIdScheme self-consistency, tail styles, hex_key, neighbours.

All assertions are invariants (round-trips, LUT inverses, shapes), not literal
addresses, so they survive packer refactors.

Known dead/broken (left for later cleanup, intentionally not tested):
  * addressing._verify_hex_reg — never called and raises (HexLUT not subscriptable).
"""
import numpy as np
import pytest

from hhg9 import Registrar, Points
import hhg9.h9.addressing as adr
from hhg9.h9.lattice import H9C


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
def b_oct(reg):
    return reg.domain('b_oct')


@pytest.fixture(scope="module")
def pb(reg, b_oct, ll):
    g = reg.domain('g_gcd')
    return reg.project(Points(ll, domain=g), [g, b_oct])


def _back(reg, b_oct, pts):
    g = reg.domain('g_gcd')
    return reg.project(pts, [b_oct, g]).coords


# ---------------------------------------------------------------------------
# 1. Enums
# ---------------------------------------------------------------------------

def test_style_enum_members():
    for name in ("HEX", "NUMERIC", "UH64A", "UR64"):
        assert hasattr(adr.Style, name)


def test_tailstyle_members():
    assert hasattr(adr.TailStyle, "reversible")
    assert hasattr(adr.TailStyle, "key")


# ---------------------------------------------------------------------------
# 2. RegionIdScheme (H9_RA) self-consistency
# ---------------------------------------------------------------------------

def test_scheme_rid_cell_inverse():
    sc = adr.H9_RA
    assert sc.r_size == 16
    # 12 valid region ids map to cells, and back, bijectively.
    for r in range(12):
        assert sc.cell2rid[sc.rid2cell[r]] == r


def test_scheme_oob_sentinels():
    sc = adr.H9_RA
    # rid 12..15 are out-of-bounds placeholders (0x5F).
    assert np.all(sc.rid2cell[12:16] == 0x5F)


def test_scheme_modes_match_cell_modes():
    sc = adr.H9_RA
    for r in range(12):
        assert sc.modes[r] == H9C.mode[sc.rid2cell[r]]


def test_scheme_table_shapes():
    sc = adr.H9_RA
    assert tuple(sc.proto) == (0, 1)
    assert np.asarray(sc.c2).shape == (2, 12)
    assert np.asarray(sc.props).shape == (2, 3, 3)


# ---------------------------------------------------------------------------
# 3. hex_str_encode / hex_str_decode  (string addresses)
# ---------------------------------------------------------------------------

def test_hex_str_roundtrip_is_exact(reg, b_oct, pb, ll):
    s = adr.hex_str_encode(pb, 33, adr.TailStyle.reversible, adr.H9_RA)
    assert len(s) == len(ll)
    back = adr.hex_str_decode(s, b_oct)
    assert np.abs(_back(reg, b_oct, back) - ll).max() < 1e-10


def test_hex_str_precision_increases_with_depth(reg, b_oct, pb, ll):
    def err(depth):
        s = adr.hex_str_encode(pb, depth, adr.TailStyle.reversible, adr.H9_RA)
        back = adr.hex_str_decode(s, b_oct)
        return np.abs(_back(reg, b_oct, back) - ll).max()
    assert err(33) < err(20) < err(10)


# ---------------------------------------------------------------------------
# 4. hex_digits / hex_decode  (nibble arrays)
# ---------------------------------------------------------------------------

def test_hex_digits_shape_and_roundtrip(reg, b_oct, pb, ll):
    hd = adr.hex_digits(pb, 33, adr.TailStyle.reversible, adr.H9_RA)
    assert hd.shape == (len(ll), 35)        # depth + 2
    assert hd.dtype == np.uint8
    assert hd.max() <= 0x0F                  # nibbles
    back = adr.hex_decode(hd, b_oct, adr.H9_RA)
    assert np.abs(_back(reg, b_oct, back) - ll).max() < 1e-10


def test_key_and_reversible_tails_differ(pb):
    rev = adr.hex_digits(pb, 14, adr.TailStyle.reversible, adr.H9_RA)
    key = adr.hex_digits(pb, 14, adr.TailStyle.key, adr.H9_RA)
    # Bodies agree; only the tail encoding differs.
    assert not np.array_equal(rev, key)
    assert np.array_equal(rev[:, :-1], key[:, :-1])


# ---------------------------------------------------------------------------
# 5. hex_pack / hex_unpack  (canonical single-u64 bin, width 14)
# ---------------------------------------------------------------------------

def test_hex_pack_unpack_roundtrip(reg, b_oct, pb, ll):
    u = adr.hex_pack(pb, 14, adr.TailStyle.reversible, adr.H9_RA, canonical=True)
    u = np.asarray(u)
    assert u.shape == (len(ll), 1)           # one u64 word at width 14
    back = adr.hex_unpack(u, adr.TailStyle.reversible, reg, adr.H9_RA)
    assert np.abs(_back(reg, b_oct, back) - ll).max() < 1e-3


def test_hex_pack_canonical_is_deterministic(pb):
    a = np.asarray(adr.hex_pack(pb, 14, adr.TailStyle.reversible, adr.H9_RA, canonical=True))
    b = np.asarray(adr.hex_pack(pb, 14, adr.TailStyle.reversible, adr.H9_RA, canonical=True))
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# 6. reg_pack / reg_unpack  (region-based u64)
# ---------------------------------------------------------------------------

def test_reg_pack_unpack_roundtrip(reg, b_oct, pb, ll):
    rp = np.asarray(adr.reg_pack(pb, 14, reg, adr.H9_RA))
    assert rp.shape[0] == len(ll)
    back = adr.reg_unpack(rp, reg, adr.H9_RA)
    assert np.abs(_back(reg, b_oct, back) - ll).max() < 1e-3


# ---------------------------------------------------------------------------
# 7. hex_key — non-mutating key extraction
# ---------------------------------------------------------------------------

def test_hex_key_does_not_mutate_input(pb):
    hd = adr.hex_digits(pb, 14, adr.TailStyle.reversible, adr.H9_RA)
    before = hd.copy()
    k = adr.hex_key(hd, copy=True)
    np.testing.assert_array_equal(hd, before)   # input untouched
    assert k.shape == hd.shape


# ---------------------------------------------------------------------------
# 8. neighbours
# ---------------------------------------------------------------------------

def test_neighbours_returns_points_same_count(pb):
    nb = adr.neighbours(pb, 20, True)
    assert isinstance(nb, Points)
    assert nb.coords.shape == pb.coords.shape
