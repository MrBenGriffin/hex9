# Part of the Hex9 (H9) Project
# Copyright ©2026, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""Address-form conversions: x_adr <-> r_adr <-> d_adr.

Three views of one hierarchy path. x_adr is the stored digit chain + tail;
r_adr is the recovered per-level region (t_cell) id chain, whose parity is
the threaded d_cell side ("mode recovery"); d_adr is the digit chain zipped
with those modes — the d_cell address. All conversions are exact table
walks; round trips must be byte-identical for every canonical cell.
"""
import numpy as np
import pytest

from hhg9 import Registrar
from hhg9.h9.grid import HexMesh
from hhg9.h9 import H9O
import hhg9.h9.uuid_address as ua
from hhg9.h9.uuid_address import _batch_int_to_nibbles
from hhg9.h9.addressing import (
    x_adr_to_r_adr, r_adr_to_x_adr, x_adr_to_d_adr, d_adr_to_x_adr,
    d_adr_to_r_adr, r_adr_to_d_adr, H9_RA, HEX_LUTS,
)

OOB = HEX_LUTS.hex_oob


@pytest.fixture(scope="module")
def reg():
    return Registrar()


@pytest.fixture(scope="module")
def mesh(reg):
    return HexMesh.create(range(5), reg)


def x_adr_of(uuids, L):
    """(N, L+2) trimmed body+tail nibble array for layer-L bin UUIDs."""
    nibs = _batch_int_to_nibbles([u.int for u in uuids], n=32)
    return np.column_stack([nibs[:, :L + 1], nibs[:, -1]])


@pytest.mark.parametrize("L", [1, 2, 3, 4])
def test_x_r_roundtrip_global(reg, mesh, L):
    """x_adr -> r_adr -> x_adr is the identity for every layer-L cell."""
    hx = x_adr_of(mesh.addr(L), L)
    oc, r_adr = x_adr_to_r_adr(hx)
    assert r_adr.shape == (len(hx), L + 2)      # [proto, root, L levels]
    assert not np.any(r_adr == OOB)
    back = r_adr_to_x_adr(oc, r_adr)
    assert np.array_equal(back, hx)


@pytest.mark.parametrize("L", [1, 2, 3, 4])
def test_x_d_roundtrip_global(reg, mesh, L):
    """x_adr -> d_adr -> x_adr is the identity for every layer-L cell."""
    hx = x_adr_of(mesh.addr(L), L)
    d_adr = x_adr_to_d_adr(hx)
    assert d_adr.shape == (len(hx), L + 1, 2)   # (root row + L digit rows)
    assert np.array_equal(d_adr[..., 0], hx[:, :-1])       # digits preserved
    back = d_adr_to_x_adr(d_adr)
    assert np.array_equal(back, hx)


@pytest.mark.parametrize("L", [1, 2, 3])
def test_d_r_roundtrip_global(reg, mesh, L):
    """d_adr -> r_adr -> d_adr closes, and matches x_adr_to_r_adr."""
    hx = x_adr_of(mesh.addr(L), L)
    d_adr = x_adr_to_d_adr(hx)
    oc_d, r_from_d = d_adr_to_r_adr(d_adr)
    oc_x, r_from_x = x_adr_to_r_adr(hx)
    assert np.array_equal(oc_d, oc_x)
    assert np.array_equal(r_from_d, r_from_x)
    d_back = r_adr_to_d_adr(oc_d, r_from_d)
    assert np.array_equal(d_back, d_adr)


def test_modes_are_region_parity(reg, mesh):
    """digit i's mode = parity(e_{i-1}): the t_cell the thread occupies one
    scale up carries the side (t_cell intrinsic mode); row 0 carries r_mo."""
    hx = x_adr_of(mesh.addr(3), 3)
    _, r_adr = x_adr_to_r_adr(hx)
    d_adr = x_adr_to_d_adr(hx)
    assert np.array_equal(d_adr[:, 1:, 1], r_adr[:, 1:-1] & 1)
    assert np.array_equal(d_adr[:, 0, 1], r_adr[:, 0])          # r_mo


def test_mode_semantics_nested_split_kat(reg):
    """The d_cell doctrine KAT: 5267.4's thread runs mode-1 through its split
    level, so its L1 container is the lineage parent's neighbour (52.4, not
    the lineage cut 526)."""
    u = ua.h9_from_label('5267.4')
    hx = x_adr_of([u], 3)
    d_adr = x_adr_to_d_adr(hx)
    digits, modes = d_adr[0, :, 0], d_adr[0, :, 1]
    assert list(digits) == [5, 2, 6, 7]
    split_and_mode1 = (digits >= 6) & (digits <= 8) & (modes == 1)
    assert split_and_mode1.any()                            # the hop exists
    anc = ua.h9_cell_ancestor([u], 1, reg=reg)[0]
    assert ua.h9_label(anc) == '52.4'


def test_root_row_is_self_contained(reg, mesh):
    """d_adr row 0 = (root hex, r_mo) recovers the octant without the tail."""
    hx = x_adr_of(mesh.addr(2), 2)
    oc, _ = x_adr_to_r_adr(hx)
    d_adr = x_adr_to_d_adr(hx)
    oc_d = H9O.l0hex_back[d_adr[:, 0, 0], d_adr[:, 0, 1] & 1][:, 0]
    assert np.array_equal(oc, oc_d)


def test_padded_input_matches_trimmed(reg, mesh):
    """x_adr_to_r_adr handles 0x0F-padded (full-width bin) input identically."""
    uuids = mesh.addr(2)
    nibs = _batch_int_to_nibbles([u.int for u in uuids], n=32)
    oc_p, r_p = x_adr_to_r_adr(nibs)                        # padded, tail in last col
    oc_t, r_t = x_adr_to_r_adr(x_adr_of(uuids, 2))
    assert np.array_equal(oc_p, oc_t)
    assert np.array_equal(r_p[:, :4], r_t)
    assert np.all(r_p[:, 4:] == OOB)


def test_mixed_layer_forward_rejected(reg, mesh):
    """r_adr_to_x_adr requires a uniform layer per batch."""
    hx2 = x_adr_of(mesh.addr(2), 2)
    nibs = _batch_int_to_nibbles([u.int for u in mesh.addr(2)[:2]] +
                                 [u.int for u in mesh.addr(3)[:2]], n=32)
    oc, r_adr = x_adr_to_r_adr(nibs)
    with pytest.raises(ValueError):
        r_adr_to_x_adr(oc, r_adr)
    # sanity: the uniform case works
    oc2, r2 = x_adr_to_r_adr(hx2)
    assert r_adr_to_x_adr(oc2, r2).shape == hx2.shape
