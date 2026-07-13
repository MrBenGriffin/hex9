# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
w_oct is the 3D octahedral container reached warp-free from b_oct.

It is the geometric sibling of c_oct: the same octahedron-surface points, but
raised directly from b_oct's already-warped 2D coordinates via b_raw's lift
matrices — no authalic warp is applied on the way in or out. These tests pin the
two properties that make w_oct worth having as its own identity:

  1. b_oct <-> w_oct is an exact, warp-free linear round-trip.
  2. w_oct genuinely holds the *warped* positions (it differs from c_oct wherever
     the warp is non-trivial), and there is deliberately NO path w_oct -> c_ell,
     so a warped-provenance point can never be fed into AK by mistake.
"""
import numpy as np
import pytest

from hhg9 import Registrar, Points


_LL = np.array([
    [51.5,  -0.12],
    [40.7, -74.00],
    [-33.9, 151.2],
    [10.0,  20.00],
    [-22.9, -43.20],
])


@pytest.fixture(scope="module")
def reg():
    return Registrar()


@pytest.fixture(scope="module")
def b_oct_pts(reg):
    """A spread of points landed in b_oct (post-warp), with oid set."""
    src = Points(_LL.copy(), domain=reg.domain('g_gcd'))
    return reg.project(src, ["g_gcd", "b_oct"])


def test_boct_woct_roundtrip_is_exact(reg, b_oct_pts):
    """b_oct -> w_oct -> b_oct is a pure linear lift/flatten: exact to fp noise,
    with none of the warp-inversion error a b_oct -> b_raw -> c_oct route pays."""
    up = reg.project(b_oct_pts, ["b_oct", "w_oct"])
    down = reg.project(up, ["w_oct", "b_oct"])
    assert np.abs(down.coords - b_oct_pts.coords).max() < 1e-12


def test_woct_lies_on_octahedron_surface(reg, b_oct_pts):
    """Lifted points sit on the octahedron faces (sum|xyz| == 1), exactly the
    c_oct validity criterion — warping only moves them within each face plane."""
    up = reg.project(b_oct_pts, ["b_oct", "w_oct"])
    assert np.all(reg.domain('w_oct').valid(up.coords))


def test_woct_carries_oid(reg, b_oct_pts):
    up = reg.project(b_oct_pts, ["b_oct", "w_oct"])
    assert up.oid is not None
    assert np.all((up.oid >= 0) & (up.oid <= 7))


def test_woct_differs_from_coct_by_the_warp(reg):
    """w_oct is reached from the warped plane, c_oct from the unwarped one. With
    the warp enabled they must NOT coincide — proving the warp is being *skipped*,
    not silently reapplied. (If they matched, w_oct would be redundant.)"""
    src = Points(_LL.copy(), domain=reg.domain('g_gcd'))
    w = reg.project(src, ["g_gcd", "b_oct", "w_oct"]).coords
    c = reg.project(src, ["g_gcd", "c_ell", "c_oct"]).coords
    assert np.abs(w - c).max() > 1e-6


def test_no_woct_to_cell_path(reg, b_oct_pts):
    """There must be no route from w_oct into the ellipsoid: AK assumes UNwarped
    c_oct input, so allowing w_oct -> c_ell (directly or via c_oct) would silently
    produce wrong geodetic points. Both must raise."""
    up = reg.project(b_oct_pts, ["b_oct", "w_oct"])
    with pytest.raises((ValueError, KeyError)):
        reg.project(up, ["w_oct", "c_ell"])
    with pytest.raises((ValueError, KeyError)):
        reg.project(up, ["w_oct", "c_oct"])
