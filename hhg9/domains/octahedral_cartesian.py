# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
This is 'c_oct' cartesian octahedral surface xyz
"""

import numpy as np
from numpy.typing import NDArray
from hhg9.h9.constants import H9O
from hhg9.base.composite import CompositeDomain, ComponentDomain
from hhg9.base.point_format import PointFormat


class OctantCartesian(ComponentDomain):
    """
    This a 3D side of an Octahedron.
    """

    def __init__(self, registrar, dom, oid: int):
        face = H9O.oid_str[oid]
        name = f'{dom.name}:{face}'
        super().__init__(registrar, name, dom,  oid, 3)
        self.points = None

    def valid(self, pts: NDArray) -> NDArray:
        """
        Return a boolean mask indicating which 3D points belong to this octant face,
        i.e. their coordinate-wise signs match the face sign triple.
        Accepts a single point shape (3,) or an array of points shape (hex_layer,3).
        """
        vals = np.atleast_2d(np.asarray(pts, dtype=float))
        target = np.array(self.sig(), dtype=float)  # (+/-1, +/-1, +/-1)
        matches = np.sign(vals) == target
        mask = np.all(matches, axis=-1)
        return mask if np.ndim(pts) > 1 else mask[0]


class OctahedralCartesian(CompositeDomain):
    """
    Basic octahedral properties and methods in 3D cartesian space.
    self.signs maps sign triple → face name.
    """

    def __init__(self, registrar: object) -> object:
        super().__init__(registrar, 'c_oct', 3)
        # Define face sides and their properties
        # has no projections.
        for oid in range(8):
            face = H9O.oid_str[oid]
            self.sides[face] = OctantCartesian(registrar, self, oid)

    @classmethod
    def valid(cls, pts: NDArray) -> NDArray:
        """
        Return an array of bools according to the validity criterion
        :param pts: set of 3d Euclidean points
        """
        pbs = np.sum(np.abs(pts), axis=-1) - 1.
        return np.abs(pbs) < 1e-15

    def register_format(self, af: PointFormat):
        """Decorator to register an AddressFormat for each component."""
        super().register_format(af)
        for side in self.sides:
            self.sides[side].register_format(af)
