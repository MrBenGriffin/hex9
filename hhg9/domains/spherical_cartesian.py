# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
'c_sph' xyz unit sphere
"""

import numpy as np
from numpy.typing import NDArray
from hhg9.base import Points
from hhg9.base.domain import Domain


class SphericalCartesian(Domain):
    """
    A Domain that represents the surface of a Unit Sphere
    in Cartesian (x,y,z) space.
    """

    def __init__(self, registrar):
        super().__init__(registrar, 'c_sph', 3)

    @classmethod
    def valid(cls, pts: Points) -> NDArray:
        """
        Test that √(util^2+v^2+w^2)=1 (surface of the unit sphere)
        :param pts: set of 3d Euclidean points
        :return: that the points are on the surface of the unit sphere.
        """
        return np.isclose(np.linalg.norm(pts.coords, axis=-1, keepdims=True) - 1.0, np.zeros_like(pts.coords))

    def adopt(self, pts: NDArray):
        """
        Take an array and adopt as this domain.
        """
        if pts.ndim == 2 and pts.shape[1] == 3:
            return Points(pts, self)
        raise ValueError(f'{pts.shape} seems wrong for spherical array.')
