# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
This is 'b_raw' barycentric xy equilateral (no warp).
"""
import numpy as np
from numpy.typing import NDArray
from hhg9.base.composite import CompositeDomain, ComponentDomain
from hhg9.base.point_format import PointFormat
from hhg9.projections import OctantBraw
from hhg9.h9 import H9K, H9O, in_scope


class OctantBaryRaw(ComponentDomain):
    """
    This a 2D side of an Octant.
    Validity should be easy enough since we have the 3 points that define it.
    """

    def __init__(self, registrar, dom, oid):
        face = H9O.oid_str[oid]
        b_sig = f'{dom.name}:{face}'
        thet = H9O.oid_tht[oid]
        self.mo = H9O.oid_mo[oid]
        mo_str = 'V' if self.mo == 0 else 'Λ'
        super().__init__(registrar, b_sig, dom,  oid, 2)
        self.th = (thet % 6) * np.pi / 3.

        edges = H9O.edges_by_id[oid]
        self.oc = np.array([ed+mo_str for ed in edges], dtype='U3')

    def valid(self, pts: NDArray) -> NDArray:
        """
        Return an array of bools according to the validity criterion
        :param pts: set of 2d Euclidean points
        """
        return in_scope(H9K.radical.R3 * pts[..., 0], pts[..., 1], self.mode)


class OctahedralBaryRaw(CompositeDomain):
    """
    Basic octahedral-2d properties and methods.
    No warp.
    """

    def __init__(self, registrar):
        super().__init__(registrar, 'b_raw', 2)
        self.sides = {}
        self.projs = {}
        self.projd = {}

        # BrawBoct

        c_oct = registrar.domain('c_oct')

        oid_s = H9O.oid_str
        for oid in range(8):
            face = oid_s[oid]
            ob = OctantBaryRaw(registrar, self, oid)
            self.sides[face] = ob
            oc = c_oct.sides[face]
            self.projs[face] = OctantBraw(registrar, face, oc.name, ob.name)
            # self.projs[face] = OctantBraw(registrar, face, oc.name, ob.name)
            # self.projd[face] = OctantBraw(registrar, face, oc.name, ob.name)


        # Define base barycentric transformation matrices
        trans = np.array([[-1, 0], [1, -2], [1, 1]])  # Prototype [1,1,1]: Proj Z using √2, √6, √3 resp.
        r90 = np.array([(0, 1), (-1, 0)])  # 90-degree rotation matrix
        mirror_y_neg_x = np.array([(0, 1), (1, 0)])  # Mirror along y = x

        # Compute rotation matrices
        north, south = trans, trans @ mirror_y_neg_x  # South is the mirror of North
        # Loop in 90º rotation order and compute projection matrices for hex_layer and S.
        scale_factors = np.sqrt([2, 6, 3])[:, np.newaxis]
        self.rot90_idx = np.zeros(8, dtype=np.uint8)
        # These are set in order of rotation, starting with NEA
        c_id = H9O.cmp_oid
        o_str = H9O.oid_str

        rot = 0
        sigs = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
        for sig in sigs:
            n_sign = tuple([*sig, 1])
            s_sign = tuple([*sig, -1])
            n_id = c_id[n_sign]
            s_id = c_id[s_sign]
            n_face = o_str[n_id]
            s_face = o_str[s_id]
            self.projs[n_face].matrix = np.column_stack([north, np.ones(3)]) / scale_factors
            self.projs[s_face].matrix = np.column_stack([south, -np.ones(3)]) / scale_factors
            self.rot90_idx[n_id] = rot
            self.rot90_idx[s_id] = rot
            north = north @ r90
            south = south @ r90
            rot = (rot + 1) % 4

    def _validate_matrices(self):
        valid = True
        for prj in self.projs:
            mtx = self.projs[prj].matrix
            dt = np.linalg.det(mtx)
            if np.abs(1 - dt) > 1e-6:
                valid = False
                print(f'{mtx}: Matrix Determinant is incorrect {dt}')
            dp = np.dot(mtx[0], mtx[1])
            if np.abs(dp) > 1e-15:
                valid = False
                print(f"Dot should be close to zero. R[0] • R[1] = {dp}")
        opposites = {
            'NEA': 'SWP', 'NEP': 'SWA',
            'NWA': 'SEP', 'NWP': 'SEA',
            'SEA': 'NWP', 'SEP': 'NWA',
            'SWA': 'NEP', 'SWP': 'NEA'
        }
        for f1, f2 in opposites.items():
            m1 = self.projs[f'{f1}'].matrix
            m2 = self.projs[f'{f2}'].matrix
            n1 = np.cross(m1[0], m1[1])
            n2 = np.cross(m2[0], m2[1])
            if not np.abs(np.dot(n1, n2) + 1) <= 1e-12:
                valid = False
                print(f"{f1} vs {f2}: {np.dot(n1, n2):.8f}. Should be -1")  # Should be -1
        if valid:
            print('matrices appear to be valid.')

    @classmethod
    def valid(cls, pts: NDArray) -> NDArray:
        """
        Return an array of bools according to the validity criterion
        :param pts: set of 2d Euclidean points
        """
        from hhg9 import Points
        if isinstance(pts, Points):
            _, mode = pts.cm()
            x, y = pts.coords[:, 0], pts.coords[:, 1]
            return in_scope(H9K.radical.R3 * x, y, mode)
        else:
            raise TypeError('pts must be a Points object')

    def register_format(self, af: PointFormat):
        """Decorator to register an AddressFormat for each component."""
        super().register_format(af)
        for side in self.sides:
            self.sides[side].register_format(af)
