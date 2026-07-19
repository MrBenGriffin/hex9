# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""
Deeper coverage for ``hhg9.base.registrar.Registrar`` beyond the lookup/
project basics in test_base_registrar.py:

  * projection() lazy registration for every projection key
  * set_ellipsoid + the sphere (f=0) area branch + the cache reset
  * register_bridge idempotence / same-endpoint skip
  * project() chain error paths
"""
import numpy as np
import pytest

from hhg9 import Registrar, Points
from hhg9.base.projection import Projection


# ---------------------------------------------------------------------------
# projection() lazy registration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", [
    "ell_gcd", "ell_gcr", "rxd_gcd", "gcd_bry", "gcd_brw", "brw_bct",
    "oct_ell", "pix_gcd",
])
def test_projection_lazy_lookup(key):
    reg = Registrar()
    p = reg.projection(key)
    assert isinstance(p, Projection)
    assert reg.projection(key) is p          # cached


def test_unknown_projection_raises():
    with pytest.raises(KeyError):
        Registrar().projection("zzz")


def test_project_unregistered_chain_raises():
    reg = Registrar()
    g = reg.domain('g_gcd')
    src = Points(np.array([[51.0, 0.0]]), domain=g)
    # g_gcd↔c_sph is now a registered pair (AuthalicGCD, the via-sphere
    # boundary); w_oct is only reachable from b_oct, so this stays invalid.
    _ = reg.domain('w_oct')
    with pytest.raises(ValueError):
        reg.project(src, ['g_gcd', 'w_oct'])     # w_oct has no g_gcd edge


# ---------------------------------------------------------------------------
# Ellipsoid configuration & area
# ---------------------------------------------------------------------------

def test_ellipsoid_area_cached():
    reg = Registrar()
    assert reg._ellipsoid_area is None
    a = reg.ellipsoid_area
    assert reg._ellipsoid_area is not None     # populated
    assert reg.ellipsoid_area == a             # same value second time


def test_set_ellipsoid_sphere_uses_closed_form():
    reg = Registrar()
    reg.set_ellipsoid(a=6_371_000.0, f=0.0, name='Sphere')
    assert reg.ellipsoid_name == 'Sphere'
    assert reg.ellipsoid_area == pytest.approx(4.0 * np.pi * 6_371_000.0**2)


def test_set_ellipsoid_inv_f_resets_cache():
    reg = Registrar()
    _ = reg.ellipsoid_area                      # prime the cache
    reg.set_ellipsoid(a=6_378_137.0, inv_f=298.257223563, name='WGS84b')
    assert reg._ellipsoid_area is None          # cache cleared by set_ellipsoid
    assert 5.0e14 < reg.ellipsoid_area < 5.2e14


def test_set_ellipsoid_requires_f_or_inv_f():
    with pytest.raises(ValueError):
        Registrar().set_ellipsoid(a=6_371_000.0)


# ---------------------------------------------------------------------------
# register_bridge
# ---------------------------------------------------------------------------

def test_register_bridge_skips_same_endpoint():
    reg = Registrar()
    reg.register_bridge(['g_gcd', 'g_gcd'])
    assert len(reg._bridges) == 0


def test_register_bridge_is_bidirectional_and_idempotent():
    reg = Registrar()
    reg.register_bridge(['g_gcd', 'b_raw', 'b_oct'])
    g, b = reg.domain('g_gcd').name, reg.domain('b_oct').name
    assert (g, b) in reg._bridges and (b, g) in reg._bridges
    before = len(reg._bridges)
    reg.register_bridge(['g_gcd', 'b_raw', 'b_oct'])   # again — no-op
    assert len(reg._bridges) == before


# ---------------------------------------------------------------------------
# register_domain / register_format direct
# ---------------------------------------------------------------------------

def test_register_domain_and_format_direct():
    reg = Registrar()
    b = reg.domain('b_oct')
    reg.register_domain(b)                       # idempotent on the same object
    assert reg._domains['b_oct'] is b
    fmt = reg.format('h9')
    reg.register_format(fmt)
    assert reg._formats['h9'] is fmt
