# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
'r_gcd' Latitude and Longitude GCD in Radians
"""

from numpy.typing import NDArray
from hhg9.base import Points
from hhg9.base.domain import Domain
import numpy as np

EPS = 1e-15  # numeric slack for boundary checks


def _wrap_lon_rad(lon: np.ndarray) -> np.ndarray:
    """Wrap longitude (radians) into [-pi, pi]. Works with any shape."""
    return (lon + np.pi) % (2.0 * np.pi) - np.pi


class RadiansGCD(Domain):
    """Latitude and Longitude GCD in Radians with robust validation and wrapping"""

    def __init__(self, registrar):
        super().__init__(registrar, 'r_gcd', 2)
        self.lat_min = float(-np.pi / 2.0)
        self.lat_max = float(+np.pi / 2.0)
        self.lon_min = float(-np.pi)
        self.lon_max = float(+np.pi)
        self._bounds = (self.lat_min, self.lat_max, self.lon_min, self.lon_max)

    def valid(self, pts: Points) -> NDArray:
        lat = pts.coords[..., 0]
        lon = pts.coords[..., 1]
        finite = np.isfinite(lat) & np.isfinite(lon)
        # Only latitude truly bounded; longitude can always be wrapped
        within_lat = np.abs(lat) <= (self.lat_max + EPS)
        return finite & within_lat

    def adopt(self, pts: NDArray):
        """
        Adopt a (N,2) array of [lat_rad, lon_rad] into this domain.
        - Ensures float64 dtype
        - Wraps longitude to [-pi, pi]
        """
        arr = np.asarray(pts, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"{arr.shape} seems wrong for latitude/longitude array; expected (N,2).")
        arr = arr.copy()
        arr[:, 1] = _wrap_lon_rad(arr[:, 1])
        return Points(arr, self)

    def wrap_longitudes(self, pts: Points) -> Points:
        """Return a new Points with longitudes wrapped into [-pi, pi]."""
        arr = np.asarray(pts.coords, dtype=np.float64).copy()
        arr[..., 1] = _wrap_lon_rad(arr[..., 1])
        return Points(arr, self)
