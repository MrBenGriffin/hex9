"""
Part of the H9 project
"""
import numpy as np
from numpy.typing import NDArray
from hhg9 import Projection, Points


class OctantBary(Projection):
    """
    This is the 3D<->2D-Barycentric projection.
    """

    def __init__(self, registrar, base, o_name, b_name):
        super().__init__(registrar, f'{base}_ob', o_name, b_name)
        self.matrix = None
        self.z_off = 1.0 / np.sqrt(3)
        rot_z = -np.pi / 3.  # -120º; As we define NS as apex we need to orient.
        # cos_th, sin_th = np.cos(rot_z), np.sin(rot_z)
        # self.orient = np.array([[cos_th, -sin_th], [sin_th, cos_th]])
        ct, st = np.cos(rot_z), np.sin(rot_z)
        self.orient = np.array([[ct, -st, 0], [st, ct, 0], [0, 0, 1.]])

    def forward(self, pts: NDArray) -> Points:
        """
        Flatten points of this octant.
        3D points are flattened on the Z-Plane.
        Currently, the domain is merely oct_c.
        This would best be set to the octant.
        """
        xyz = pts[:, -3:]
        xya = xyz @ (self.matrix.T @ self.orient)  # z should be aligned.
        sum_z = np.sum(xya[:, -1] - self.z_off).tolist()
        # if sum_z > 1e-14:
        #     print(f'OctantBaryFwd: {self.fwd_cs.name} Points deviating from surface: {sum_z:.2f}')
        xy = np.delete(xya, 2, -1)  # These are now in barycentric 2D.
        ptx = np.delete(pts, -1, -1)  # drop the final value here also.
        ptx[:, -2:] = xy
        return ptx.view(Points).set_domain(self.fwd_cs)

    def backward(self, pts: Points) -> Points:
        """
        Unflatten points of this octant. (inverse of flatten).
        2D points are un-flattened from the Z-Plane.
        """
        # insert a z value.
        pt3 = np.insert(pts, pts.shape[-1], self.z_off, axis=-1)
        # extract the xyz
        if pt3.shape[-1] > 3:
            xyz = pt3[..., -3:]
            pt3[:, -3:] = xyz @ (self.matrix.T @ self.orient).T
        else:
            pt3 = pt3 @ (self.matrix.T @ self.orient).T
        return pt3.view(Points).set_domain(self.rev_cs)
