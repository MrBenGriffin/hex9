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
from pyproj import CRS


class EllipsoidCartesian(Domain):
    """
    A Domain representing the surface of an ellipsoid in Cartesian (ECEF) coordinates.
    Defaults to epsg=4978
    """

    def __init__(self, registrar, name='c_ell', epsg=4978):
        super().__init__(registrar, name, 3)
        self.crs = CRS.from_epsg(epsg)
        self.a = self.crs.ellipsoid.semi_major_metre # semi-major axis
        self.b = self.crs.ellipsoid.semi_minor_metre
        self.f = 1. / self.crs.ellipsoid.inverse_flattening

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
        Accepts (N, 3) arrays only.
        """
        if pts.ndim == 2 and pts.shape[1] == 3:
            return Points(pts, domain=self)
        raise ValueError(f"EllipsoidCartesian expects shape (N, 3), got {pts.shape}")
