"""
 Part of  h9 by Ben Griffin.
 This file defines a spherical/octahedral projection
"""
from modern.osprojection import OSProjection
import sympy as sp
import numpy as np
from scipy.optimize import root


class CNProjection(OSProjection):
    """
        An Octahedron/Sphere Projection generated via an analytical approximation to a
        conformal projection.
    """
    _e = 1e-40

    @classmethod
    def os(cls, uvw):
        """
        Convert a NDArray of octahedral points projected onto a sphere
        Anders Kaseorg: https://math.stackexchange.com/questions/5016695/
        :param uvw:  An array of Euclidean points on the surface of a unit octahedron.
        :return: UVW on a unit sphere.
        """
        x = np.abs(uvw)
        # return pts @ (self.proj_matrix.T @ self._rz(self.rot_z))  # These are now in barycentric 2D.
        np.delete(values, 2, -1)
        # depict_trx(1.25 * (dtr + np.complex128(gx + 1j * gy)), 'Grid O')

        # o0, d0, b0, o1, d1, b1, o2, d2, b2, o3, d3, b3 = (
        #     -1.86547329, -2.38543150, 2.05896051,
        #     264.53837471, 270.88261948, 401.9071098,
        #     35.66201549, 46.11240162, -23.59702692,
        #     -21.17021298, 142.57135676, -20.84693306
        # )
        # w0 = np.array([[o0, o0, d0], [o0, d0, o0], [d0, o0, o0]])
        # z0 = cls.sigmoid(np.dot(x, w0) + [b0, b0, b0])
        #
        # w1 = np.array([[d1, -o1, o1], [o1, -d1, o1], [o1, -o1, d1]])
        # z1 = np.tanh(np.dot(z0, w1) + [-b1, b1, -b1])
        #
        # w2 = np.array([[o2, o2, -d2], [d2, -o2, -o2], [o2, -d2, o2]])
        # z2 = cls.sigmoid(np.dot(z1, w2) + [b2, b2, b2])
        #
        # w3 = np.array([[o3, o3, d3], [o3, d3, o3], [d3, o3, o3]])
        # dd = np.dot(z2, w3) + [b3, b3, b3]

        dn = dd / (np.linalg.norm(dd, axis=1, keepdims=True))
        an = np.abs(dn)  # octant 0!
        return np.copysign(an, uvw)


    @classmethod
    def so(cls, tsp):
        """
         Projected a spherical point onto the octahedron
         This inverse function using numerical optimization
         :param tsp:  An array of Euclidean points on the surface of a unit sphere.
         :return: UVW on a unit octahedron.
        """

        def find_root(uvw):
            """
            :param uvw: Euclidean point on the unit sphere.
            :return: That spherical point on the octahedron
            """
            result = root(cls._root_fn, np.sign(uvw) * 1. / 3., args=(uvw,), tol=1e-12)
            result.x /= np.linalg.norm(result.x, ord=1)
            return result.x

        return np.apply_along_axis(find_root, -1, tsp)

    @classmethod
    def _root_fn(cls, op, tx):  # octa_point, target_sphere_point
        norm = np.linalg.norm(op, ord=1)
        val = cls.os(np.array([op / norm])) - np.array(tx)
        return val[0]

