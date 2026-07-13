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

    def __init__(self, registrar: object, name: str = 'c_oct', link_boct: bool = False) -> object:
        super().__init__(registrar, name, 3)
        # Define face sides and their properties.
        # Plain c_oct has no projections of its own (the AK leg c_oct<->c_ell and
        # the geometric leg c_oct<->b_raw are owned elsewhere). The w_oct variant
        # (link_boct=True) is the SAME 3D container reached warp-free from b_oct:
        # it wires a geometric lift that borrows b_raw's per-face matrices, so
        # b_oct's already-warped 2D coords are raised straight to 3D without ever
        # touching the authalic warp. w_oct deliberately has NO path to c_ell, so
        # nobody can continue a warped-provenance point into AK by mistake.
        for oid in range(8):
            face = H9O.oid_str[oid]
            self.sides[face] = OctantCartesian(registrar, self, oid)
        if link_boct:
            self._link_boct(registrar)

    def _link_boct(self, registrar):
        """Wire the eight warp-free OctantBraw lifts between this (w_oct) 3D domain
        and b_oct, reusing b_raw's geometric matrices verbatim — no recomputation
        and, crucially, no warp. Only the endpoint identities differ from the
        c_oct<->b_raw lifts, which is exactly what keeps the contract honest."""
        from hhg9.projections import OctantBraw
        b_raw = registrar.domain('b_raw')   # owns the per-face lift matrices
        b_oct = registrar.domain('b_oct')
        self.projs = {}
        for oid in range(8):
            face = H9O.oid_str[oid]
            w_side = self.sides[face]
            b_side = b_oct.sides[face]
            proj = OctantBraw(registrar, f'{face}_w', w_side.name, b_side.name)
            proj.matrix = b_raw.projs[face].matrix   # borrow geometry, no recompute
            self.projs[face] = proj

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
