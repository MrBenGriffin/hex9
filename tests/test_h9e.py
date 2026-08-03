# Part of the Hex9 (H9) Project
# Copyright ©2026, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""Behavioural tests for the Hex9+E4H PoC (hhg9.h9.e4h, h9e_*).

Invariants under test (design provenance in the module docstring):
  * address stability — decode gives the leaf representative, which
    re-encodes to the SAME address (octant interior, across an octant
    seam, and around a cone point);
  * the count law — an interior host owns exactly 2 * 4^depth leaf
    trapezoids;
  * grammar — labels parse as <h9>E<half><class digits>, invalid
    nibbles rejected;
  * matched pairs — points sampled either side of a fine child's
    diameter agree on the final class digit.
"""
import re

import numpy as np
import pytest

from hhg9 import Registrar
from hhg9.h9.e4h import h9e_decode, h9e_encode, h9e_label, h9e_split

BOX = (19.5, 20.5, 34.5, 35.5)          # octant interior (lat0..1, lon0..1)
LAYER, DEPTH = 6, 2


@pytest.fixture(scope='module')
def reg():
    return Registrar()


def _stable(reg, lats, lons):
    u1 = h9e_encode(lats, lons, LAYER, DEPTH, reg)
    la, lo = h9e_decode(u1, reg)
    u2 = h9e_encode(la, lo, LAYER, DEPTH, reg)
    return sum(a.int == b.int for a, b in zip(u1, u2)), len(u1)


def test_roundtrip_interior(reg):
    rng = np.random.default_rng(9)
    ok, n = _stable(reg, rng.uniform(BOX[0], BOX[1], 200),
                    rng.uniform(BOX[2], BOX[3], 200))
    assert ok == n


def test_roundtrip_equator_seam(reg):
    rng = np.random.default_rng(11)
    ok, n = _stable(reg, rng.uniform(-0.05, 0.05, 120),
                    rng.uniform(34.5, 35.5, 120))
    assert ok == n


def test_roundtrip_cone_point(reg):
    rng = np.random.default_rng(13)
    t = rng.uniform(0, 2 * np.pi, 120)
    r = rng.uniform(0.005, 0.08, 120)
    ok, n = _stable(reg, r * np.cos(t), r * np.sin(t))
    assert ok == n


def test_roundtrip_odd_depth(reg):
    """Depth parity must not matter to the address layer (the boundary
    child structure alternates with depth; the roundtrip must not)."""
    rng = np.random.default_rng(19)
    lats = np.concatenate([rng.uniform(BOX[0], BOX[1], 60),
                           rng.uniform(-0.04, 0.04, 40)])
    lons = np.concatenate([rng.uniform(BOX[2], BOX[3], 60),
                           rng.uniform(-0.04, 0.04, 40)])
    for depth in (1, 3):
        u1 = h9e_encode(lats, lons, LAYER, depth, reg)
        la, lo = h9e_decode(u1, reg)
        u2 = h9e_encode(la, lo, LAYER, depth, reg)
        assert all(a.int == b.int for a, b in zip(u1, u2))


def test_count_law(reg):
    la0, lo0 = 20.0, 35.0
    u0 = h9e_encode([la0], [lo0], LAYER, DEPTH, reg)[0]
    host0 = h9e_split(u0)[0]
    g = np.linspace(-0.11, 0.11, 90)
    gl, gn = np.meshgrid(la0 + g, lo0 + g)
    us = h9e_encode(gl.ravel(), gn.ravel(), LAYER, DEPTH, reg)
    mine = {u.int for u in us if h9e_split(u)[0].int == host0.int}
    assert len(mine) == 2 * 4 ** DEPTH


def test_label_grammar(reg):
    rng = np.random.default_rng(17)
    us = h9e_encode(rng.uniform(BOX[0], BOX[1], 20),
                    rng.uniform(BOX[2], BOX[3], 20), LAYER, DEPTH, reg)
    pat = re.compile(rf'^\d{{{LAYER + 1}}}E[01][0-5]{{{DEPTH}}}$')
    for u in us:
        assert pat.match(h9e_label(u)), h9e_label(u)


def test_split_rejects_plain_h9(reg):
    from hhg9.h9.uuid_address import h9_bin, h9_encode
    u = h9_bin(h9_encode([20.0], [35.0], reg), LAYER, reg)[0]
    with pytest.raises(ValueError):
        h9e_split(u)


def test_depth_budget(reg):
    with pytest.raises(ValueError):
        h9e_encode([20.0], [35.0], layer=6, depth=25, reg=reg)


def test_truncation_is_binning(reg):
    """Address = bin: the depth-1 address of a point is the truncation
    of its depth-2 address (drop the last nibble), for interior, seam
    and cone-point samples alike — the tail is suffix-local, so
    coarser bins are prefixes, with no carries anywhere."""
    from hhg9.h9.uuid_address import (_batch_int_to_nibbles,
                                      batch_nibbles_to_int)
    rng = np.random.default_rng(29)
    lats = np.concatenate([rng.uniform(BOX[0], BOX[1], 60),
                           rng.uniform(-0.04, 0.04, 60)])
    lons = np.concatenate([rng.uniform(BOX[2], BOX[3], 60),
                           rng.uniform(-0.04, 0.04, 60)])
    u2 = h9e_encode(lats, lons, LAYER, 2, reg)
    u1 = h9e_encode(lats, lons, LAYER, 1, reg)
    for a, b in zip(u2, u1):
        nib = _batch_int_to_nibbles([a.int], n=32)[0].copy()
        nib[LAYER + 4] = 0x0F                      # drop the last digit
        trunc = int(batch_nibbles_to_int(nib[None, :])[0])
        assert trunc == b.int


def test_matched_pairs_global(reg):
    """The two halves of EVERY fine hexagon share their final digit —
    same host, cross host, seams and the cone point alike, at even and
    odd depths. Depth 1 is the acid test (every pair is cross-host).
    The five-symbol enumeration (transport note §4d) makes this a
    global invariant; before it, cross-host agreement was 0%%."""
    from hhg9.h9.e4h import h9e_partner_point
    rng = np.random.default_rng(23)
    lats = np.concatenate([rng.uniform(BOX[0], BOX[1], 60),
                           rng.uniform(-0.04, 0.04, 40)])
    lons = np.concatenate([rng.uniform(BOX[2], BOX[3], 60),
                           rng.uniform(-0.04, 0.04, 40)])
    for depth in (1, 2, 3):
        us = h9e_encode(lats, lons, LAYER, depth, reg)
        pla, plo = h9e_partner_point(us, reg)
        pus = h9e_encode(pla, plo, LAYER, depth, reg)
        checked = cross = 0
        for u, pu in zip(us, pus):
            h1, _, d1 = h9e_split(u)
            h2, _, d2 = h9e_split(pu)
            assert pu.int != u.int
            if d1[-1] != 0:
                checked += 1
                cross += h1.int != h2.int
                assert d2[-1] == d1[-1], (h9e_label(u), h9e_label(pu))
        assert checked > 10
        if depth == 1:
            assert cross == checked        # depth 1: all pairs cross-host
