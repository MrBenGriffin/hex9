# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Part of the H9 project
"""
import numpy as np
from hhg9 import Points
from hhg9.base.projection import Projection


class OctantXYUV(Projection):
    """
    This is the 2D-Barycentric<->2D-Simplex projection.
    Should NOT be converting to xyz.
    """

    def __init__(self, registrar, base, o_name, b_name):
        super().__init__(registrar, f'{base}_xyuv', o_name, b_name)
        self.abc = self.fwd_cs.tpt  # Triangle shape ABC
        # a, b, c = self.cols[0], self.cols[1], self.cols[2]
        # Affine form: XY = a*A + b*B + c*C with a+b+c=1.
        # Build the 3x3 "augmented" matrix that maps bary -> XY1 = [x,y,1].
        # [A_x, B_x, C_x]
        # [A_y, B_y, C_y]
        # [1.0, 1.0, 1.0]
        matrix = np.vstack([self.abc.T, np.ones(3, dtype=float)])  # shape (3,3)
        self.uv_matrix = np.linalg.inv(matrix)  # maps XY1 -> bary

    def forward(self, xy: np.ndarray) -> np.ndarray:
        """
        dom is currently assigned by registrar so no need to do here atm.
        Convert points on this octant to barycentric coordinates from b_oct (barycentric xy).
        Normally we do not clip during a projection, but barycentric UV only makes sense between 0 and 1.
        """
        n = xy.shape[0]
        xy1 = np.column_stack([xy, np.ones(n, dtype=float)])  # (N,3)
        bary = xy1 @ self.uv_matrix.T  # (N,3)
        bary = np.clip(bary, 0.0, 1.0)  # or np.clip(bary, -1e-12, 1.0 + 1e-12)
        s = bary.sum(axis=1, keepdims=True)
        s[s == 0.0] = 1.0
        bary /= s
        u = bary[:, 1]
        v = bary[:, 2]
        return np.column_stack([u, v])

    def backward(self, pts: np.ndarray) -> np.ndarray:
        """
        Convert UV coordinates to XY coordinates on this octant.
        """
        uv = np.clip(pts, 0.0, 1.0)
        u = uv[:, 0]
        v = uv[:, 1]
        a = 1.0 - (u + v)
        auv = np.column_stack([a, u, v])
        return auv @ self.abc  # (N,3) @ (3,2) -> (N,2)
