# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
GeneralGCD is recognised as [latitude,longitude]
"""

import numpy as np
from hhg9.base import Points
from hhg9.base.domain import Domain
from numpy.typing import NDArray


class GeneralGCD(Domain):
    """Latitude and Longitude GCD"""

    def __init__(self, registrar):
        super().__init__(registrar, 'g_gcd', 2)
        self.lat_min, self.lat_max = (-90., 90.)
        self.lon_min, self.lon_max = (-180., 180.)

    def valid(self, pts: Points) -> NDArray:
        lat, lon = pts.coords[..., 0], pts.coords[..., 1]
        return np.logical_and(
            np.abs(lat) <= 90,
            np.abs(lon) <= 180
        )

    def adopt(self, pts: NDArray):
        """
        Take an array and adopt as this domain.
        """
        if pts.ndim == 2 and pts.shape[1] == 2:
            return Points(pts, self)
        raise ValueError(f'{pts.shape} seems wrong for latitude/longitude array.')
