"""
Part of the H9 project
"""
from copy import copy

import numpy as np
from numpy.typing import NDArray

from hhg9 import Points
from hhg9.base import CompositeDomain
from hhg9.base import ComponentDomain
from hhg9.base.point_format import PointFormat
from hhg9.domains import OctahedralCartesian
from hhg9.projections import OctantBary


class OctantBarycentric(ComponentDomain):
    """
    This a 2D side of an Octant.
    Validity should be easy enough since we have the 3 points that define it.
    """

    def __init__(self, registrar, name: str, sign, cc: tuple):
        super().__init__(registrar, name)
        self.sign = sign
        self.c1 = cc[0]
        self.ud = cc[1]  #'V' if sum(np.array(self.sign)+1)/2 % 2 == 1 else 'Λ'
        self.c2 = cc[2]
        self.cm = cc[3]
        self.m1 = {
            self.c2[0]: self.cm[0],
            self.c2[1]: self.cm[1],
            self.c2[2]: self.cm[2]
        }
        self.geo = name[-3:]

    def sig(self) -> tuple:
        return self.sign

    def valid(self, pts: NDArray) -> NDArray:
        """
        Return an array of bools according to the validity criterion
        :param pts: set of 3d Euclidean points
        """
        raise NotImplementedError


class OctahedralBarycentric(CompositeDomain):
    """
    Basic octahedral-2d properties and methods.
    """

    def __init__(self, registrar, o: OctahedralCartesian):
        super().__init__(registrar, 'b_oct')
        self.sides = {}
        self.projs = {}
        self.signs = {}  # o.signs  # These are used to tie the projection.
        for sign, face in o.signs.items():
            c2 = {
                # AP EW  NS Λ: 021 102 210 // V:201 012 120
                (+1, +1, +1): ('120', 'V', '372', '372',),  # 'NEA': 3x1+0,3x2+1,3x0+2=372
                (-1, +1, +1): ('102', 'Λ', '318', '318',),  # 'NEP': 3x1+0,3x0+1,3x2+2=318
                (+1, -1, +1): ('210', 'Λ', '642', '642',),  # 'NWA': 3x2+0,3x1+1,3x0+2=642
                (-1, -1, +1): ('120', 'V', '372', '147',),  # 'NWP': MAP 120 [372] TO 147
                (+1, +1, -1): ('021', 'Λ', '075', '174',),  # 'SEA': MAP 021 [075] TO 174
                (-1, +1, -1): ('012', 'Λ', '048', '048',),  # 'SEP': 3x0+0,3x1+1,3x2+2=048
                (+1, -1, -1): ('201', 'V', '615', '615',),  # 'SWA': 3x2+0,3x0+1,3x1+2=615
                (-1, -1, -1): ('021', 'Λ', '075', '075',)  # 'SWP': 3x0+0,3x2+1,3x1+2=075
            }
            b_sig = f'{self.name}:{face}'
            o_sig = o.sides[face].name
            self.sides[face] = OctantBarycentric(registrar, b_sig, sign, c2[sign])
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
        sigs = [(1, 1), (-1, 1),  (-1, -1),  (1, -1)]
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
        raise NotImplementedError('Barycentric Octants use the same dimension for each side.')

    def register_format(self, af: PointFormat):
        """Decorator to register an AddressFormat for each component."""
        super().register_format(af)
        for side in self.sides:
            self.sides[side].register_format(af)
