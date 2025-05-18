import numpy as np


class OSProjection:
    """Abstract class that manages octahedron<=>sphere co-ordinate projection."""

    @classmethod
    def o_constraint(cls, _pts):
        """Constraint: |u|+|v|+|w|=1 (surface of the unit octahedron)"""
        return np.sum(np.abs(_pts)) - 1  # return delta.

    @classmethod
    def os(cls, _):  # octahedron point to spherical point
        """
        input: array of Euclidean points on the surface of a unit octahedron.
        output: array of Euclidean points on the surface of a unit sphere.
        """
        return np.array([[0, 0, 0]])

    @classmethod
    def so(cls, _):  # spherical point to octahedron point
        """
        input: array of Euclidean points on the surface of a unit sphere.
        output: array of Euclidean points on the surface of a unit octahedron.
        """
        return np.array([[0, 0, 0]])

