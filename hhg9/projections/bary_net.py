"""
Part of the H9 project
"""
import numpy as np
from numpy.typing import NDArray
from hhg9 import Projection, Points


class BaryNet(Projection):
    """
    This is the 2D-Barycentric to 2D-Net projection.
    """
    def __init__(self, registrar, base, o_name, n_name, theta, offset):
        super().__init__(registrar, f'{base}_bn', o_name, n_name)
        self.offset = offset
        self.theta = theta
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        self.matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

    def forward(self, uvw: Points) -> NDArray:
        """
        points from barycentric to grid.
        These should already be 2d.
        """
        xy = uvw[..., -2:]
        xb = xy @ self.matrix + self.offset
        uvw[..., -2:] = xb
        return uvw.view(Points).set_domain(self.fwd_cs)

    def backward(self, uvw: Points) -> NDArray:
        """
        points from grid to barycentric.
        """
        xy = uvw[..., -2:]
        xb = (xy - self.offset) @ self.matrix.T
        uvw[..., -2:] = xb
        return uvw.view(Points).set_domain(self.rev_cs)

