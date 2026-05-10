# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
EllipsoidCartesian is Hex9 ECEF domain.
"""
import numpy as np
from hhg9.base import Points
from hhg9.base.domain import Domain
from numpy.typing import NDArray
class EllipsoidCartesian(Domain):
    """
    A Domain representing the surface of the reference ellipsoid in Cartesian (ECEF) coordinates.
    Ellipsoid parameters are taken from registrar.ellipsoid (defaults to WGS84).
    """

    def __init__(self, registrar, name='c_ell', a=None, b=None, f=None):
        super().__init__(registrar, name, 3)
        geo = registrar.ellipsoid
        self.f = f if f is not None else geo.f
        self.a = a if a is not None else geo.a
        self.b = b if b is not None else self.a * (1.0 - self.f)
        self.e2 = 2.0 * self.f - self.f ** 2

    def valid(self, pts: Points) -> NDArray:
        """
        Check whether points lie (numerically) on the ellipsoid surface:
        (x/a)^2 + (y/a)^2 + (z/b)^2 ≈ 1
        """
        coords = pts.coords
        x2 = (coords[..., 0] / self.a) ** 2
        y2 = (coords[..., 1] / self.a) ** 2
        z2 = (coords[..., 2] / self.b) ** 2
        ellipsoid_radius = x2 + y2 + z2
        return np.isclose(ellipsoid_radius, 1.0, rtol=1e-9)

    def adopt(self, pts: NDArray) -> Points:
        """
        Wrap raw Cartesian data as Points in this domain.
        Accepts (hex_layer, 3) arrays only.
        """
        if pts.ndim == 2 and pts.shape[1] == 3:
            return Points(pts, domain=self)
        raise ValueError(f"EllipsoidCartesian expects shape (hex_layer, 3), got {pts.shape}")
