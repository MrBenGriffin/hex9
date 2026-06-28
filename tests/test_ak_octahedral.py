# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""
Tests for the AK octahedral projection ``oct_ell`` (c_oct ↔ c_ell),
``hhg9.projections.ak_octahedral``.

forward  : c_oct → c_ell  (Anders Kaseorg analytic map + invariant-vertex handling)
backward : c_ell → c_oct  (find_coords beam search + Newton refinement for
                           near-polar/seam apex-convergence cases)

The near-pole / near-seam round-trip is what drives the Newton-refinement
branch; exact poles are octahedron vertices (ambiguous octant) and are
intentionally not round-tripped.
"""
import numpy as np
import pytest

from hhg9 import Registrar, Points
from hhg9.h9 import H9O


@pytest.fixture(scope="module")
def reg():
    return Registrar()


@pytest.fixture(scope="module")
def proj(reg):
    return reg.projection('oct_ell')


# ---------------------------------------------------------------------------
# Round-trips
# ---------------------------------------------------------------------------

def test_c_oct_c_ell_roundtrip_exact(reg):
    """c_oct → c_ell → c_oct is the analytic forward then its inverse; for
    interior points it is exact to fp precision."""
    g = reg.domain('g_gcd')
    ll = np.array([[10.0, 20.0], [40.7, -74.0], [-33.9, 151.2]])
    co = reg.project(Points(ll, domain=g), ['g_gcd', 'c_ell', 'c_oct'])
    rt = reg.project(co, ['c_oct', 'c_ell', 'c_oct'])
    assert np.abs(rt.coords - co.coords).max() < 1e-12


def test_backward_near_pole_and_seam_newton(reg):
    """Near-polar and near-seam points exercise the Newton refinement in
    backward(); the full g_gcd→c_ell→c_oct→…→g_gcd loop must still recover them."""
    g = reg.domain('g_gcd')
    ll = np.array([
        [89.9, 0.0], [-89.9, 30.0], [89.99, 45.0],
        [0.0, 0.0], [10.0, 20.0], [45.0, 89.99], [-60.0, -120.0],
    ])
    ce = reg.project(Points(ll, domain=g), ['g_gcd', 'c_ell'])
    co = reg.project(ce, ['c_ell', 'c_oct'])           # backward (beam + Newton)
    back = reg.project(co, ['c_oct', 'c_ell', 'g_gcd'])
    assert np.abs(back.coords - ll).max() < 1e-8


# ---------------------------------------------------------------------------
# forward — invariant vertices
# ---------------------------------------------------------------------------

def test_forward_maps_octahedron_vertices_onto_ellipsoid(proj):
    """The 6 octahedron vertices are invariant points; forward must still place
    them on the ellipsoid surface (the `aa` branch)."""
    verts = H9O.oct_vrt.astype(float)
    assert verts.shape == (6, 3)
    oid = (np.arange(6, dtype=np.uint8) % 8)
    pts = Points(verts, domain=proj.c_oct, oid=oid)
    out = proj.forward(pts)
    assert out.coords.shape == (6, 3)
    assert np.all(np.isfinite(out.coords))
    a, b = proj.ab
    f = out.coords[:, 0]**2 / a**2 + out.coords[:, 1]**2 / a**2 + out.coords[:, 2]**2 / b**2
    np.testing.assert_allclose(f, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# config + lazy r_gcd wiring
# ---------------------------------------------------------------------------

def test_projection_axes_and_constants(proj):
    assert proj.fwd_cs.name == 'c_ell'
    assert proj.rev_cs.name == 'c_oct'
    a, b = proj.ab
    assert a > b > 0                              # oblate: semi-minor < semi-major
    assert proj.ALPHA == pytest.approx(3.227806237143884, rel=1e-12)


def test_rad_gcd_lazy_registration(reg, proj):
    r = proj.rad_gcd()
    assert r.name == 'r_gcd'
    assert proj.rad_gcd() is r                    # cached/idempotent
    assert reg.domain('r_gcd') is r
