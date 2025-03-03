import numpy as np


class OSProjection:
    # Abstract class that manages octahedron<=>sphere co-ordinate projection.

    @classmethod
    def o_valid(cls, pts):  # Constraint: |u|+|v|+|w|=1 (surface of the unit octahedron)
        return np.all(np.apply_along_axis((lambda a: np.abs(np.sum(np.abs(a)) - 1.) < 1e-15), -1, pts))

    @classmethod
    def s_valid(cls, pts):  # Constraint: √(u^2+v^2+w^2)=1 (surface of the unit sphere)
        return np.allclose(np.linalg.norm(pts, axis=-1) - 1.0, np.zeros_like(pts))

    @classmethod
    def os(cls, _):  # octahedron point to spherical point
        # input: array of Euclidean points on the surface of a unit octahedron.
        # output: array of Euclidean points on the surface of a unit sphere.
        return np.array([[0, 0, 0]])

    @classmethod
    def so(cls, _):  # spherical point to octahedron point
        # input: array of Euclidean points on the surface of a unit sphere.
        # output: array of Euclidean points on the surface of a unit octahedron.
        return np.array([[0, 0, 0]])
