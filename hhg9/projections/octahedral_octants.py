"""
Part of the H9 project
"""
import numpy as np
from numpy.typing import NDArray
from hhg9 import Projection


class OctahedralOctants(Projection):
    """
    This is the [Barycentric2D]<->[2DNet] projection.
    """

    def __init__(self, registrar, name, o, n):
        super().__init__(registrar, o.name, n.name)

    def forward(self, pts: NDArray[np.float64]):
        """
        Find octants and then project.
        """
        transformed = pts @ (self.matrix.T @ self.orient)  # These are now in barycentric 3D.
        return np.delete(transformed, 2, -1)

    def backward(self, pts: NDArray[np.float64]):
        """
        Unflatten points of this octant. (inverse of flatten).
        2D points are un-flattened from the Z-Plane.
        """
        pts3 = np.insert(pts, pts.shape[1], self.z_off, axis=1)  # These are now in 3D.
        return pts3 @ (self.matrix.T @ self.orient).T
