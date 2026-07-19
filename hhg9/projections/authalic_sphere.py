# Part of the Hex9 (H9) Project
# Copyright ©2026, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
AuthalicGCD: c_sph (unit authalic sphere, ECEF-style xyz) ↔ g_gcd
(geodetic lat/lon degrees on the registrar's source ellipsoid).

This is the boundary projection that decouples the datum from the
octahedral engine: the ellipsoid-specific part of a via-sphere chain is
this exact, area-preserving latitude reparameterisation and nothing else —
everything from c_sph inward (AK, beam search, warp) is purely spherical.
"""

import numpy as np
from hhg9 import Points
from hhg9.base.projection import Projection
from hhg9.algorithms.authalic import geodetic_to_authalic, authalic_to_geodetic


class AuthalicGCD(Projection):
    """Unit authalic sphere ↔ geodetic (lat, lon degrees).
    Forward: c_sph xyz → (lat, lon)  [spherical ξ,λ → geodetic φ via Newton]
    Backward: (lat, lon) → c_sph xyz [geodetic φ → authalic ξ, closed form]
    """

    def __init__(self, registrar):
        self._reg = registrar
        super().__init__(registrar, 'aut_gcd', 'c_sph', 'g_gcd')

    @property
    def _e2(self):
        f = self._reg.ellipsoid.f
        return 2.0 * f - f * f

    def forward(self, arr: Points) -> Points:
        """c_sph xyz → geodetic (lat_deg, lon_deg)"""
        x, y, z = arr.coords[:, 0], arr.coords[:, 1], arr.coords[:, 2]
        # arcsin(z) loses colatitude as 1/colat near the poles (z = sinξ
        # quantises colat²/2); hypot(x, y) carries it linearly at full
        # relative precision, so atan2 recovers ξ to ~1 ULP everywhere —
        # the same reason Bowring recovers geodetic φ via atan2 on c_ell.
        xi = np.arctan2(z, np.hypot(x, y))
        lam = np.arctan2(y, x)
        phi = authalic_to_geodetic(xi, self._e2)
        out = np.stack([np.degrees(phi), np.degrees(lam)], axis=-1)
        return Points(out, domain=self.fwd_cs, samples=arr.samples)

    def backward(self, arr: Points) -> Points:
        """Geodetic (lat_deg, lon_deg) → c_sph xyz"""
        phi = np.radians(arr.coords[:, 0])
        lam = np.radians(arr.coords[:, 1])
        xi = geodetic_to_authalic(phi, self._e2)
        cxi = np.cos(xi)
        out = np.stack([cxi * np.cos(lam), cxi * np.sin(lam), np.sin(xi)],
                       axis=-1)
        return Points(out, domain=self.rev_cs, samples=arr.samples)
