"""
Part of the H9 project
"""
import numpy as np
from hhg9 import Projection, Points


class OctantBary(Projection):
    """
    This is the 3D<->2D-Barycentric projection.
    """

    def __init__(self, registrar, base, o_name, b_name):
        super().__init__(registrar, f'{base}_ob', o_name, b_name)
        self.matrix = None
        self.ud = self.fwd_cs.ud
        # 'V' if sum(np.array(self.sign)+1)/2 % 2 == 1 else 'Λ'
        self.z_off = 1.0 / np.sqrt(3)
        rot_z = -np.pi / 3.  # -120º; As we define NS as apex we need to orient.
        # rot_z = rot_z if self.ud == 'Λ' else +np.pi / 3
        ct, st = np.cos(rot_z), np.sin(rot_z)
        self.orient = np.array([[ct, -st, 0], [st, ct, 0], [0, 0, 1.]])

    def forward(self, arr) -> Points:
        """
        Flatten points of this octant.
        3D points are flattened on the Z-Plane.
        Currently, the domain is merely oct_c.
        This would best be set to the octant.
        """
        xyz = arr.coords if isinstance(arr, Points) else arr
        xya = xyz @ (self.matrix.T @ self.orient)  # z should be aligned.
        # sum_z = np.sum(xya[:, -1] - self.z_off).tolist()
        # if abs(sum_z) > 1e-1:
        #     raise ValueError(f'OctantBaryFwd: {self.fwd_cs.name} Points deviating from surface: {sum_z:.2f}')
        xy = np.delete(xya, 2, -1)  # These are now in barycentric 2D.
        if isinstance(arr, Points):
            return Points(xy, domain=self.fwd_cs, samples=arr.samples, components=arr.components)
        else:
            return xy

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
