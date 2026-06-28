# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""
Bit-layout / round-trip tests for the single-nibble tail helpers
(``hhg9.h9.tail``).

The tail nibble layout (shared byte-for-byte with libhex9):
    bit 3      : p_mo  (terminal parent net_mode; 0 for canonical addresses)
    bits 2..1  : p_c2  (terminal c2, 0..2)
    bit 0      : r_mo  (root/octant net_mode)

Key style drops p_mo (bit 3 = 0); after canonicalisation (p_mo == 0) the key
and reversible styles emit identical bytes — an invariant tested below.

Load-bearing note: tail_pack_reversible / tail_unpack_reversible / tail_pack_key
/ tail_key_from_reversible are all called from production; tail_unpack_key is
currently imported but never called (kept here as the public mirror of
tail_pack_key and tested for contract stability).
"""
from itertools import product

import numpy as np
import pytest

from hhg9.h9 import tail as t


_RMO = (0, 1)
_PMO = (0, 1)
_PC2 = (0, 1, 2)


# ---------------------------------------------------------------------------
# Enum
# ---------------------------------------------------------------------------

def test_tailstyle_values():
    assert t.TailStyle.reversible.value == 0
    assert t.TailStyle.key.value == 1
    assert t.TailStyle.none.value == 2


# ---------------------------------------------------------------------------
# Explicit bit layout
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("r_mo,p_mo,p_c2", list(product(_RMO, _PMO, _PC2)))
def test_reversible_bit_layout(r_mo, p_mo, p_c2):
    packed = int(t.tail_pack_reversible(r_mo, p_mo, p_c2))
    assert packed == (p_mo << 3) | (p_c2 << 1) | r_mo


@pytest.mark.parametrize("r_mo,p_c2", list(product(_RMO, _PC2)))
def test_key_bit_layout(r_mo, p_c2):
    # p_mo is dropped: any value must give the same nibble (bit 3 = 0).
    packed0 = int(t.tail_pack_key(r_mo, 0, p_c2))
    packed1 = int(t.tail_pack_key(r_mo, 1, p_c2))
    assert packed0 == packed1 == (p_c2 << 1) | r_mo


# ---------------------------------------------------------------------------
# Round-trips
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("r_mo,p_mo,p_c2", list(product(_RMO, _PMO, _PC2)))
def test_reversible_roundtrip(r_mo, p_mo, p_c2):
    packed = t.tail_pack_reversible(r_mo, p_mo, p_c2)
    out_c2, out_rmo, out_pmo = t.tail_unpack_reversible(packed)
    assert (int(out_c2), int(out_rmo), int(out_pmo)) == (p_c2, r_mo, p_mo)


@pytest.mark.parametrize("r_mo,p_c2", list(product(_RMO, _PC2)))
def test_key_roundtrip(r_mo, p_c2):
    packed = t.tail_pack_key(r_mo, 0, p_c2)
    out_c2, out_rmo = t.tail_unpack_key(packed)
    assert (int(out_c2), int(out_rmo)) == (p_c2, r_mo)


# ---------------------------------------------------------------------------
# Cross-style invariants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("r_mo,p_c2", list(product(_RMO, _PC2)))
def test_canonical_styles_emit_identical_bytes(r_mo, p_c2):
    """With p_mo == 0 (canonical) the reversible and key tails are identical."""
    assert int(t.tail_pack_reversible(r_mo, 0, p_c2)) == int(t.tail_pack_key(r_mo, 0, p_c2))


@pytest.mark.parametrize("r_mo,p_mo,p_c2", list(product(_RMO, _PMO, _PC2)))
def test_key_from_reversible_drops_pmo(r_mo, p_mo, p_c2):
    rev = t.tail_pack_reversible(r_mo, p_mo, p_c2)
    key = t.tail_key_from_reversible(rev)
    assert int(key) == int(rev) & 0x07           # bit 3 cleared
    assert int(key) == int(t.tail_pack_key(r_mo, p_mo, p_c2))


# ---------------------------------------------------------------------------
# Vectorisation
# ---------------------------------------------------------------------------

def test_vectorised_reversible_roundtrip():
    rng = np.random.default_rng(0)
    r = rng.integers(0, 2, 64).astype(np.uint8)
    p = rng.integers(0, 2, 64).astype(np.uint8)
    c = rng.integers(0, 3, 64).astype(np.uint8)
    packed = t.tail_pack_reversible(r, p, c)
    assert packed.shape == (64,)
    c2, rmo, pmo = t.tail_unpack_reversible(packed)
    np.testing.assert_array_equal(c2, c)
    np.testing.assert_array_equal(rmo, r)
    np.testing.assert_array_equal(pmo, p)
