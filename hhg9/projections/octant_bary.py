# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Part of the H9 project
"""
import numpy as np
from hhg9 import Points
from hhg9.base.projection import Projection


class OctantBary(Projection):
    """
    This is the 3D<->2D-Barycentric projection.
    """

    def __init__(self, registrar, base, o_name, b_name):
        super().__init__(registrar, f'{base}_ob', o_name, b_name)
        self.matrix = None
        # self.mode = self.fwd_cs.mode
        # self.h9 = H9Engine()
        # 'V' if sum(np.array(self.sign)+1)/2 % 2 == 1 else 'Λ'
        self.z_off = 1.0 / np.sqrt(3)
        rot_z = self.fwd_cs.th  # -120º As we define NS as apex we need to orient.
        ct, st = np.cos(rot_z), np.sin(rot_z)
        self.orient = np.array([[ct, -st, 0], [st, ct, 0], [0, 0, 1.]])

    def forward(self, arr) -> Points:
        """
        Flatten points of this octant.
        3D points are flattened on the Z-Plane.
        Currently, the domain is merely oct_c.
        This would best be set to the octant.
        Never clamp here - let clamping take place later.
        """
        xyz = arr.coords if isinstance(arr, Points) else arr
        xya = xyz @ (self.matrix.T @ self.orient)  # z should be aligned.
        pts = np.delete(xya, 2, -1)  # These are now in barycentric 2D.
        if isinstance(arr, Points):
            #
            return Points(pts, domain=self.fwd_cs, samples=arr.samples, components=arr.components)
        else:
            return pts

    def backward(self, arr: Points) -> Points:
        """
        Unflatten points of this octant. (inverse of flatten).
        2D points are un-flattened from the Z-Plane.
        """
        xy = arr.coords if isinstance(arr, Points) else arr
        xyz = np.insert(xy, xy.shape[-1], self.z_off, axis=-1)
        xyo = xyz @ (self.matrix.T @ self.orient).T
        if isinstance(arr, Points):
            return Points(xyo, domain=self.rev_cs, samples=arr.samples, components=arr.components)
        else:
            return xyo
