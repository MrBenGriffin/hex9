# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
WGS84 ellipsoid constants and pure-numpy ECEF ↔ geodetic conversion.

Constants are read once from geographiclib.Geodesic.WGS84, which is already
a project dependency. This avoids any dependency on pyproj for basic geodetic
arithmetic.

Geodesic.WGS84 values (IEEE-standard, never change):
    a = 6 378 137.0 m  (semi-major axis)
    f = 1 / 298.257 223 563  (flattening)
"""

import numpy as np
from geographiclib.geodesic import Geodesic

_wgs84 = Geodesic.WGS84
A: float = _wgs84.a          # semi-major axis (m)
F: float = _wgs84.f          # flattening
B: float = A * (1.0 - F)    # semi-minor axis (m)
E2: float = 2.0 * F - F**2  # first eccentricity squared

def ecef_to_latlon(x, y, z, a=A, e2=E2) -> tuple:
    """ECEF (m) → (lat_deg, lon_deg) on the ellipsoid surface.

    Uses Bowring's iterative method; 5 iterations give sub-nanometre accuracy
    for surface points (h ≈ 0).

    Parameters
    ----------
    x, y, z : array-like, metres
    a : float, semi-major axis in metres (default WGS84)
    e2 : float, first eccentricity squared (default WGS84)

    Returns
    -------
    lat_deg, lon_deg : ndarray, degrees
    """
    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    lon = np.degrees(np.arctan2(y, x))
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - e2))
    for _ in range(5):
        sin_lat = np.sin(lat)
        N = a / np.sqrt(1.0 - e2 * sin_lat**2)
        lat = np.arctan2(z + e2 * N * sin_lat, p)
    return np.degrees(lat), lon


def latlon_to_ecef(lat_deg, lon_deg, a=A, e2=E2) -> tuple:
    """(lat_deg, lon_deg) → ECEF (m) on the ellipsoid surface (h = 0).

    Parameters
    ----------
    lat_deg, lon_deg : array-like, degrees
    a : float, semi-major axis in metres (default WGS84)
    e2 : float, first eccentricity squared (default WGS84)

    Returns
    -------
    x, y, z : ndarray, metres
    """
    lat = np.radians(np.asarray(lat_deg))
    lon = np.radians(np.asarray(lon_deg))
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    N = a / np.sqrt(1.0 - e2 * sin_lat**2)
    x = N * cos_lat * np.cos(lon)
    y = N * cos_lat * np.sin(lon)
    z = N * (1.0 - e2) * sin_lat
    return x, y, z
