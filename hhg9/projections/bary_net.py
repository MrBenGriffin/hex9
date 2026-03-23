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
from hhg9.h9 import classify_cell, H9K, in_down, in_up
from hhg9.h9.region import H9R


class BaryNet(Projection):
    """
    This is the 2D-Barycentric to 2D-Net projection.

    """
    def __init__(self, registrar, base, o_name, n_name, theta, offset):
        # print(f'> BaryNet: {base}_bn; {o_name} -> {n_name}')
        super().__init__(registrar, f'{base}_bn', o_name, n_name)
        self.offset = offset
        self.theta = theta
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        self.matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        self.c2trans = None

    def set_c2trans(self, c2t):
        """Add a c2-based affine transform to an existing octant affine"""
        self.c2trans = []
        for (mo, rt, gx, gy, tx) in c2t:
            cos_theta = np.cos(tx)
            sin_theta = np.sin(tx)
            mtx = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
            self.c2trans.append([mo, rt, np.array([gx, gy]), mtx])

    def c2_affine(self, c2: int = 0, forward: bool = True):
        """Get the affine transform for a given c2"""
        offs = np.array([0, 0])
        if self.c2trans is not None:
            mo, rt, offs, matr = self.c2trans[c2]
            if forward:
                return self.matrix @ matr, offs
            else:
                return matr.T @ self.matrix.T, -offs
        elif forward:
            return self.matrix, offs
        else:
            return self.matrix.T, -offs

    def forward(self, arr) -> NDArray:
        """
        points from grid to bary centric
        fwd_cs = n_oct, rev_cs=b_oct
        These should already be 2d.
        """
        xy = arr.coords if isinstance(arr, Points) else arr
        if self.c2trans is not None:
            xn = np.full_like(xy, np.nan)
            mode = self.rev_cs.mode  # orientation is still in bary space.
            x, y = xy[:, 0], xy[:, 1]
            ẋ = x * H9K.radical.R3
            cid = classify_cell(ẋ, y)
            c2v = H9R.mcc2[mode, cid]
            for c2 in (0, 1, 2):
                group = c2v == c2
                if np.any(group):
                    matr, off = self.c2_affine(c2)
                    xn[group] = (xy[group] @ matr) + off
        else:
            xn = xy @ self.matrix  # rotation is in net space.
        xn += self.offset
        if isinstance(arr, Points):
            return Points(xn, domain=self.fwd_cs, samples=arr.samples, oid=arr.oid)
        else:
            return xn

    def backward(self, arr: Points) -> NDArray:
        """
        points from grid (xy) to barycentric (xb)
        fwd_cs = n_oct, rev_cs=b_oct
        """
        xy = arr.coords if isinstance(arr, Points) else arr
        xo = (xy - self.offset)
        if self.c2trans is not None:
            xb = np.full_like(xy, np.nan)
            mode = self.rev_cs.mode  # classify in bary (b_oct) space
            for c2 in (0, 1, 2):
                matr, off = self.c2_affine(c2, False)
                xbc = (xo + off) @ matr
                ẋ = H9K.R3 * xbc[:, 0]
                cid = classify_cell(ẋ, xbc[:, 1])
                c2v = H9R.mcc2[mode, cid]
                in_c2 = c2v == c2
                if np.any(in_c2):
                    xb[in_c2] = xbc[in_c2]
        else:
            xb = xo @ self.matrix.T
        if isinstance(arr, Points):
            return Points(xb, domain=self.rev_cs, samples=arr.samples, oid=arr.oid)
        else:
            return xb
