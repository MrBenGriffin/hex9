# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
BaryNet: 2D-Barycentric to 2D-Net projection.
"""
import numpy as np
from numpy.typing import NDArray
from hhg9 import Points
from hhg9.base.projection import Projection


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
        self.c2trans = None

    def forward(self, arr) -> NDArray:
        """
        points from barycentric to grid.
        These should already be 2d.
        """
        xy = arr.coords if isinstance(arr, Points) else arr
        xn = xy @ self.matrix + self.offset
        if isinstance(arr, Points):
            return Points(xn, domain=self.fwd_cs, samples=arr.samples, components=arr.components)
        else:
            return xn

    def backward(self, arr: Points) -> NDArray:
        """
        points from grid to barycentric.
        """
        xy = arr.coords if isinstance(arr, Points) else arr
        xb = (xy - self.offset) @ self.matrix.T
        if isinstance(arr, Points):
            return Points(xb, domain=self.rev_cs, samples=arr.samples, components=arr.components)
        else:
            return xb
