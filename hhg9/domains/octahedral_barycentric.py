"""
Part of the H9 project
"""
from copy import copy

import numpy as np
from numpy.typing import NDArray

from hhg9 import Points
from hhg9.base import CompositeDomain, ComponentDomain, H9Engine
from hhg9.base.point_format import PointFormat
from hhg9.domains import OctahedralCartesian
from hhg9.projections import OctantBary


class OctantBarycentric(ComponentDomain):
    """
    This a 2D side of an Octant.
    Validity should be easy enough since we have the 3 points that define it.
    """

    def __init__(self, registrar, dom, name: str, sign, cc: tuple):
        super().__init__(registrar, name)
        self.dom = dom
        self.sign = sign
        self.th = (cc[0] % 6) * np.pi / 3.
        self.tr = cc[1]
        self.mode = cc[2]   # 'V' if sum(np.array(self.sign)+1)/2 % 2 == 1 else 'Λ'
        self.geo = {k: v for k, v in zip(self.tr, cc[3])}  # in c1 (orientation) order.

    def _add(self, h9m):
        for k, v in self.geo.items():
            key = f'{v}{self.mode}'
            h9m[key] = (k, self.tr, self.sign, self.name)

    def sig(self) -> tuple:
        return self.sign

    def valid(self, pts: NDArray) -> NDArray:
        """
        Return an array of bools according to the validity criterion
        :param pts: set of 2d Euclidean points
        """
        return H9Engine.in_scope(H9Engine.R3 * pts[..., 0], pts[..., 1], self.mode)


class OctahedralBarycentric(CompositeDomain):
    """
    Basic octahedral-2d properties and methods.
    """

    def __init__(self, registrar, o: OctahedralCartesian):
        super().__init__(registrar, 'b_oct')
        self.sides = {}
        self.projs = {}
        self.signs = {}  # o.signs  # These are used to tie the projection.
        self.h9map = {}
        # Theta is to ensure that each octant has a pole at its apex.
        # The pole is C2:0
        for sign, face in o.signs.items():
            props = {
                # Octahedral Triangle Identities differ.
                # AP EW  NS    θ  V
                (+1, +1, +1): (2, '047', 'V', ('EA', 'NA', 'NE')),  # 'NEA' N:5, E:8, A: 0
                (-1, +1, +1): (5, '085', 'Λ', ('EP', 'NE', 'NP')),  # 'NEP' N:4, E:7, P: 0
                (+1, -1, +1): (5, '085', 'Λ', ('WA', 'NA', 'NW')),  # 'NWA' N:4, W:7, A: 0
                (-1, -1, +1): (2, '047', 'V', ('WP', 'NP', 'NW')),  # 'NWP' N:5, W:8, P: 0
                (+1, +1, -1): (5, '085', 'Λ', ('EA', 'SE', 'SA')),  # 'SEA' S:4, E:7, A: 0
                (-1, +1, -1): (2, '047', 'V', ('EP', 'SP', 'SE')),  # 'SEP' S:5, E:8, P: 0
                (+1, -1, -1): (2, '047', 'V', ('WA', 'SA', 'SW')),  # 'SWA' S:5, W:8, A: 0
                (-1, -1, -1): (5, '085', 'Λ', ('WP', 'SW', 'SP'))   # 'SWP' S:4, W:7, P: 0
            }
            b_sig = f'{self.name}:{face}'
            o_sig = o.sides[face].name
            self.sides[face] = OctantBarycentric(registrar, self, b_sig, sign, props[sign])
            self.sides[face]._add(self.h9map)
            self.projs[face] = OctantBary(registrar, face, o_sig, b_sig)
            self.signs[sign] = face
            self.components[sign] = self.sides[face]

        # Define base barycentric transformation matrices
        trans = np.array([[-1, 0], [1, -2], [1, 1]])  # Prototype [1,1,1]: Proj Z using √2, √6, √3 resp.
        r90 = np.array([(0, 1), (-1, 0)])  # 90-degree rotation matrix
        mirror_y_neg_x = np.array([(0, 1), (1, 0)])  # Mirror along y = -x
        # Compute rotation matrices
        north, south = trans, trans @ mirror_y_neg_x  # South is the mirror of North
        # Loop in 90º rotation order and compute projection matrices for N and S.
        scale_factors = np.sqrt([2, 6, 3])[:, np.newaxis]

        # These are set in order of rotation, starting with NEA
        sigs = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
        for sig in sigs:
            n_sign = tuple([*sig, 1])
            s_sign = tuple([*sig, -1])
            n_face = self.signs[n_sign]
            s_face = self.signs[s_sign]
            self.projs[n_face].matrix = np.column_stack([north, np.ones(3)]) / scale_factors
            self.projs[s_face].matrix = np.column_stack([south, -np.ones(3)]) / scale_factors
            north = north @ r90
            south = south @ r90

        # self._validate_matrices()

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
        raise NotImplementedError

    def binning(self, _pts: Points, sig: tuple = None):
        """
        Given a set of 2D points, bin them according to which side they are on.
        :param _pts: An array of 2D Octahedron/Spherical points
        :param sig face required...
        We do need to override this, because sign means something else here.
        """
        if sig is None:
            rx = True
        raise NotImplementedError('Barycentric Octants use the same dimension for each side.')

    def register_format(self, af: PointFormat):
        """Decorator to register an AddressFormat for each component."""
        super().register_format(af)
        for side in self.sides:
            self.sides[side].register_format(af)
