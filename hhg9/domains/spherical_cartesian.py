"""
Part of the H9 project
"""
import numpy as np
from numpy.typing import NDArray
from hhg9.base import Domain, Points


class SphericalCartesian(Domain):
    """
    A Domain that represents the surface of a Unit Sphere
    in Cartesian (x,y,z) space.
    """

    def __init__(self, registrar):
        super().__init__(registrar, 'c_sph')

    @classmethod
    def valid(cls, pts: Points) -> NDArray:
        """
        Test that √(u^2+v^2+w^2)=1 (surface of the unit sphere)
        :param pts: set of 3d Euclidean points
        :return: that the points are on the surface of the unit sphere.
        """
        vals = np.array([pts]) if len(np.array(pts).shape) == 1 else np.array(pts)
        return np.isclose(np.linalg.norm(vals, axis=-1, keepdims=True) - 1.0, np.zeros_like(vals))
