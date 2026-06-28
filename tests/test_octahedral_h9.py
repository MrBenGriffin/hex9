# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""
Tests for the public ``h9`` point format (``hhg9.formats.octahedral_h9``),
reached via ``Registrar.format('h9')``.

This is the user-facing encode/decode surface.  The headline contract is the
``format`` → ``revert`` round-trip in HEX, plus the style-string parsing in
``_select_style``.  Coordinates are checked back in g_gcd (lat/lon degrees).

Notes on style coverage:
  * HEX round-trips to fp precision and is the default.
  * UH64A ('ua') is the canonical sub-metre u64 bin — coarser, so a looser tol.
  * NUMERIC ('i') / UR64 ('r') are exercised for *encoding* only; their
    multi-point revert path packs variable-length words and is not a clean
    round-trip (left as a known gap, see _select_style coverage).
"""
import numpy as np
import pytest

from hhg9 import Registrar, Points
import hhg9.h9.addressing as adr


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def reg():
    return Registrar()


@pytest.fixture(scope="module")
def fmt(reg):
    return reg.format('h9')


# Sample lat/lon (degrees) away from poles/antimeridian seams.
_LL = np.array([
    [51.5,  -0.12],   # London
    [40.7, -74.00],   # New York
    [-33.9, 151.2],   # Sydney
    [0.0,    0.0],    # Gulf of Guinea
    [-22.9, -43.2],   # Rio
])


@pytest.fixture(scope="module")
def ll():
    return _LL.copy()


@pytest.fixture(scope="module")
def pts_b(reg, ll):
    """Sample points projected into b_oct (the domain the format consumes)."""
    g, b = reg.domain('g_gcd'), reg.domain('b_oct')
    return reg.project(Points(ll, domain=g), [g, b])


def _back_to_ll(reg, pts_b):
    g, b = reg.domain('g_gcd'), reg.domain('b_oct')
    return reg.project(pts_b, [b, g]).coords


# ---------------------------------------------------------------------------
# 1. is_valid placeholder
# ---------------------------------------------------------------------------

def test_is_valid_is_permissive(fmt):
    assert fmt.is_valid("anything") is True


# ---------------------------------------------------------------------------
# 2. _select_style parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sub,style,width", [
    ("x33", adr.Style.HEX,    33),
    ("x20", adr.Style.HEX,    20),
    ("25",  adr.Style.HEX,    25),   # bare width → HEX
    ("ua",  adr.Style.UH64A,  14),
    ("uk",  adr.Style.UH64A,  14),   # legacy alias → canonical bin
    ("r",   adr.Style.UR64,   14),
    ("i21", adr.Style.NUMERIC, 21),
    ("i",   adr.Style.NUMERIC, 34),  # no width → default width
    (None,  adr.Style.HEX,    34),   # default
])
def test_select_style(fmt, sub, style, width):
    assert fmt._select_style(sub) == (style, width)


def test_select_style_invalid_width_raises(fmt):
    with pytest.raises(ValueError):
        fmt._select_style("xfoo")


# ---------------------------------------------------------------------------
# 3. HEX round-trip (the headline contract)
# ---------------------------------------------------------------------------

def test_hex_roundtrip_is_exact(reg, fmt, ll, pts_b):
    s = fmt.format(pts_b, None, 'x33')
    back = fmt.revert(s, style=adr.Style.HEX)
    err = np.abs(_back_to_ll(reg, back) - ll).max()
    assert err < 1e-10, f"HEX x33 round-trip error {err} deg"


def test_hex_precision_increases_with_width(reg, fmt, ll, pts_b):
    """Wider addresses encode finer positions: error must be monotone in width."""
    def err(width):
        s = fmt.format(pts_b, None, f'x{width}')
        back = fmt.revert(s, style=adr.Style.HEX)
        return np.abs(_back_to_ll(reg, back) - ll).max()
    e10, e20, e33 = err(10), err(20), err(33)
    assert e33 < e20 < e10


# ---------------------------------------------------------------------------
# 4. UH64A canonical bin round-trip (coarser)
# ---------------------------------------------------------------------------

def test_uh64a_roundtrip_within_tolerance(reg, fmt, ll, pts_b):
    s = fmt.format(pts_b, None, 'ua')
    back = fmt.revert(s, style=adr.Style.UH64A)
    err = np.abs(_back_to_ll(reg, back) - ll).max()
    assert err < 1e-3, f"UH64A round-trip error {err} deg"


def test_uh64a_is_idempotent_bin(fmt, pts_b):
    """Canonical u64 binning: re-encoding the same points yields identical bins."""
    a = fmt.format(pts_b, None, 'ua')
    b = fmt.format(pts_b, None, 'ua')
    assert a == b


# ---------------------------------------------------------------------------
# 5. Output shape: single vs multi-point
# ---------------------------------------------------------------------------

def test_single_point_returns_bare_string(reg, fmt):
    g, b = reg.domain('g_gcd'), reg.domain('b_oct')
    one = reg.project(Points(_LL[:1], domain=g), [g, b])
    s = fmt.format(one, None, 'x33')
    assert isinstance(s, str) and '\n' not in s


def test_multi_point_returns_one_line_each(fmt, pts_b, ll):
    s = fmt.format(pts_b, None, 'x33')
    assert s.count('\n') == len(ll) - 1
    assert len(s.splitlines()) == len(ll)


# ---------------------------------------------------------------------------
# 6. Encode-only smoke for the remaining styles
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sub", ['i21', 'r'])
def test_other_styles_encode_nonempty(fmt, pts_b, ll, sub):
    s = fmt.format(pts_b, None, sub)
    lines = s.splitlines()
    assert len(lines) == len(ll)
    assert all(len(line) > 0 for line in lines)
