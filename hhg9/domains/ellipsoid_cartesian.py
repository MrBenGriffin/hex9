"""
Part of the H9 project
"""
import numpy as np
from hhg9.base import Domain, Points
from numpy.typing import NDArray


class EllipsoidCartesian(Domain):
    """
    A Domain representing the surface of an ellipsoid in Cartesian (ECEF) coordinates.
    Defaults to WGS84 ellipsoid: a = 6378137.0 m, f = 1/298.257223563.
    """

    def __init__(self, registrar, name='c_ell', a: float = 6378137.0, f: float = 1 / 298.257223563):
        super().__init__(registrar, name, 3)
        self.a = a                  # semi-major axis
        self.f = f                  # flattening
        self.b = a * (1 - f)        # semi-minor axis

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
