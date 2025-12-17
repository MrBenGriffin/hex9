# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

import numpy as np
from numpy.typing import NDArray
from hhg9 import Points
from hhg9.base.projection import Projection


class RGCD_GCD(Projection):
    """
    LatLon Radians <=> LatLon Degrees projection
    Forward: (r.lat, r.lon) → (d.lat, d.lon)
    Backward: (d.lat, d.lon) → (r.lat, r.lon)
    """

    def __init__(self, registrar):
        super().__init__(registrar, 'rxd_gcd', 'r_gcd', 'g_gcd')

    def forward(self, arr: Points) -> NDArray:
        """R_GCD → G_GCD"""
        degrees = np.degrees(arr.coords)
        return Points(degrees, domain=self.fwd_cs, samples=arr.samples)

    def backward(self, arr: Points) -> NDArray:
        """G_GCD → R_GCD"""
        radians = np.radians(arr.coords)
        return Points(radians, domain=self.rev_cs, samples=arr.samples)
