# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Part of the H9 project
"""
import numpy as np
from hhg9 import Points
from hhg9.base.projection import Projection


class OctahedralOctants(Projection):
    """
    This is the [Barycentric2D]<->[2DNet] projection.
    """

    def __init__(self, registrar, name, o, n):
        super().__init__(registrar, o.name, n.name)
        self.matrix = None
        self.orient = None
        self.z_off = None

    def forward(self, arr):
        """
        Find octants and then project.
        """
        xyz = arr.coords if isinstance(arr, Points) else arr
        xyo = xyz @ (self.matrix.T @ self.orient)  # These are now in barycentric 3D.
        xy = np.delete(xyo, 2, -1)  # These are now in barycentric 2D.
        if isinstance(arr, Points):
            return Points(xy, domain=self.fwd_cs, samples=arr.samples, components=arr.components)
        else:
            return xy

    def backward(self, arr):
        """
        Unflatten points of this octant. (inverse of flatten).
        2D points are un-flattened from the Z-Plane.
        """
        xy = arr.coords if isinstance(arr, Points) else arr
        xyz = np.insert(xy, xy.shape[1], self.z_off, axis=1)
        xyo = xyz @ (self.matrix.T @ self.orient).T  # These are now in barycentric 3D.
        if isinstance(arr, Points):
            return Points(xyo, domain=self.rev_cs, samples=arr.samples, components=arr.components)
        else:
            return xyo
