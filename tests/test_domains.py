# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""
Projection round-trip tests for the ``hhg9.domains`` package.

Domains are the nodes of the projection graph; the forward/backward coordinate
maths lives in (or is reached through) the domain + its registered projection.
Exercising ``Registrar.project`` round-trips therefore covers both the domains
and the projections that wire them.

Core graph:  g_gcd ↔ c_ell ↔ c_oct ↔ b_raw ↔ b_oct ↔ s_oct,  plus g_gcd ↔ r_gcd.
(c_sph / n_pix / p_pix are not on a reachable g_gcd chain — n_pix/p_pix are
parametric pixel domains needing an extent; c_sph is wired to no projection — so
they are out of scope here.)
"""
import numpy as np
import pytest

from hhg9 import Registrar, Points


@pytest.fixture(scope="module")
def reg():
    return Registrar()


_LL = np.array([
    [51.5,  -0.12],
    [40.7, -74.00],
    [-33.9, 151.2],
    [10.0,  20.00],
    [-22.9, -43.20],
])


@pytest.fixture(scope="module")
def src(reg):
    return Points(_LL.copy(), domain=reg.domain('g_gcd'))


# Each domain with the chain that reaches it from g_gcd.
_CHAINS = {
    "b_oct": ["g_gcd", "b_oct"],
    "b_raw": ["g_gcd", "b_raw"],
    "c_ell": ["g_gcd", "c_ell"],
    "r_gcd": ["g_gcd", "r_gcd"],
    "c_oct": ["g_gcd", "c_ell", "c_oct"],
    "s_oct": ["g_gcd", "b_oct", "s_oct"],
    "w_oct": ["g_gcd", "b_oct", "w_oct"],
}

# Octant-carrying domains (those should propagate a valid oid).
_OCTANT_DOMAINS = {"b_oct", "b_raw", "c_oct", "s_oct", "w_oct"}


# ---------------------------------------------------------------------------
# 1. Per-domain round-trips
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", list(_CHAINS))
def test_projection_roundtrip(reg, src, key):
    chain = _CHAINS[key]
    fwd = reg.project(src, chain)
    back = reg.project(fwd, chain[::-1])
    assert np.abs(back.coords - _LL).max() < 1e-10, f"{key} round-trip drifted"


def test_full_pipeline_roundtrip(reg, src):
    """One pass through the entire core chain and back — exercises every core
    domain's forward and backward maps in a single round-trip."""
    chain = ["g_gcd", "c_ell", "c_oct", "b_raw", "b_oct", "s_oct"]
    fwd = reg.project(src, chain)
    back = reg.project(fwd, chain[::-1])
    assert np.abs(back.coords - _LL).max() < 1e-10


# ---------------------------------------------------------------------------
# 2. oid propagation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", sorted(_OCTANT_DOMAINS))
def test_octant_domains_carry_valid_oid(reg, src, key):
    p = reg.project(src, _CHAINS[key])
    assert p.oid is not None
    assert np.all((p.oid >= 0) & (p.oid <= 7))


@pytest.mark.parametrize("key", ["c_ell", "r_gcd"])
def test_non_octant_domains_have_no_oid(reg, src, key):
    assert reg.project(src, _CHAINS[key]).oid is None


# ---------------------------------------------------------------------------
# 3. Domain-specific coordinate invariants
# ---------------------------------------------------------------------------

def test_r_gcd_is_radians_of_g_gcd(reg, src):
    r = reg.project(src, ["g_gcd", "r_gcd"])
    np.testing.assert_allclose(r.coords, np.radians(_LL))


def test_c_ell_points_lie_on_ellipsoid(reg, src):
    ce = reg.project(src, ["g_gcd", "c_ell"])
    assert ce.coords.shape == (len(_LL), 3)
    d = reg.domain('c_ell')
    a, b = d.a, d.b
    f = ce.coords[:, 0]**2 / a**2 + ce.coords[:, 1]**2 / a**2 + ce.coords[:, 2]**2 / b**2
    np.testing.assert_allclose(f, 1.0, atol=1e-9)


def test_b_oct_coords_are_octant_local_bary(reg, src):
    """b_oct coords are octant-local barycentric — bounded within the canonical
    triangle (|x|, |y| comfortably < 1)."""
    pb = reg.project(src, ["g_gcd", "b_oct"])
    assert pb.coords.shape == (len(_LL), 2)
    assert np.abs(pb.coords).max() < 1.5


# ---------------------------------------------------------------------------
# 4. Domain validity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", ["b_oct", "b_raw", "c_ell", "r_gcd"])
def test_projected_points_are_valid_in_their_domain(reg, src, key):
    p = reg.project(src, _CHAINS[key])
    assert reg.domain(key).is_valid(p)


def test_g_gcd_rejects_out_of_range_latitude(reg):
    g = reg.domain('g_gcd')
    bad = Points(np.array([[91.0, 0.0], [0.0, 0.0]]), domain=g)
    np.testing.assert_array_equal(g.valid(bad), [False, True])
