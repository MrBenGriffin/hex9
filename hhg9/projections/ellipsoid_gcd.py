# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Projections between WGS84 ECEF (c_ell) and geodetic lat/lon (g_gcd / r_gcd).
Uses pure-numpy Bowring iteration — no pyproj required.
"""

import numpy as np
from hhg9 import Points
from hhg9.base.projection import Projection
from hhg9.algorithms.wgs84 import ecef_to_latlon, latlon_to_ecef


class EllipsoidGCD(Projection):
    """ECEF ↔ geodetic (lat, lon degrees) on WGS84.
    Forward: ECEF → (lat, lon)
    Backward: (lat, lon) → ECEF
    """

    def __init__(self, registrar):
        super().__init__(registrar, 'ell_gcd', 'c_ell', 'g_gcd')

    def forward(self, arr: Points) -> Points:
        """ECEF → geodetic (lat_deg, lon_deg)"""
        x, y, z = arr.coords[:, 0], arr.coords[:, 1], arr.coords[:, 2]
        lat, lon = ecef_to_latlon(x, y, z)
        return Points(np.stack([lat, lon], axis=-1), domain=self.fwd_cs, samples=arr.samples)

    def backward(self, arr: Points) -> Points:
        """Geodetic (lat_deg, lon_deg) → ECEF"""
        lat, lon = arr.coords[:, 0], arr.coords[:, 1]
        x, y, z = latlon_to_ecef(lat, lon)
        return Points(np.stack([x, y, z], axis=-1), domain=self.rev_cs, samples=arr.samples)


class EllipsoidGCDRad(Projection):
    """ECEF ↔ geodetic (lat, lon radians) on WGS84.
    Forward: ECEF → (lat_rad, lon_rad)
    Backward: (lat_rad, lon_rad) → ECEF
    """

    def __init__(self, registrar):
        super().__init__(registrar, 'ell_gcr', 'c_ell', 'r_gcd')
        try:
            registrar.domain('r_gcd')
        except KeyError:
            from hhg9.domains import RadiansGCD
            from hhg9.projections import RGCD_GCD
            RadiansGCD(registrar)
            RGCD_GCD(registrar)

    def forward(self, arr: Points) -> Points:
        """ECEF → geodetic radians (lat_rad, lon_rad)"""
        x, y, z = arr.coords[:, 0], arr.coords[:, 1], arr.coords[:, 2]
        lat, lon = ecef_to_latlon(x, y, z)
        return Points(np.radians(np.stack([lat, lon], axis=-1)), domain=self.fwd_cs, samples=arr.samples)

    def backward(self, arr: Points) -> Points:
        """Geodetic radians (lat_rad, lon_rad) → ECEF"""
        lat = np.degrees(arr.coords[:, 0])
        lon = np.degrees(arr.coords[:, 1])
        x, y, z = latlon_to_ecef(lat, lon)
        return Points(np.stack([x, y, z], axis=-1), domain=self.rev_cs, samples=arr.samples)
