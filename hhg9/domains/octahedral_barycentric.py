"""
Part of the H9 project
"""
from copy import copy

import numpy as np
from numpy.typing import NDArray

from hhg9 import Step
from hhg9.base import CompositeDomain, ComponentDomain, H9Engine
from hhg9.base.h9_engine import Style
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

    def h9_add(self, h9m):
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
        self.h9e = H9Engine()
        # self.octant_index = {
        #     (+1, +1, +1): 0, (-1, +1, +1): 1, (+1, -1, +1): 2, (-1, -1, +1): 3,
        #     (+1, +1, -1): 4, (-1, +1, -1): 5, (+1, -1, -1): 6, (-1, -1, -1): 7
        # }
        self.octant_props = np.array([
            ('047', 'V'),  # index 0
            ('085', 'Λ'),  # index 1
            ('085', 'Λ'),  # index 2
            ('047', 'V'),  # index 3
            ('085', 'Λ'),  # index 4
            ('047', 'V'),  # index 5
            ('047', 'V'),  # index 6
            ('085', 'Λ'),  # index 7
        ], dtype=[('loc', 'U3'), ('mode', 'U1')])
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
            self.sides[face].h9_add(self.h9map)
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

    def decode(self, addr):
        """Decode octahedral coordinates into a point"""
        if addr[:3] in self.h9map.keys():
            rec = self.h9map[addr[:3]]
            return self.h9e.oct_decode(f'{rec[0]}{addr[3:]}', rec[1])

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

    def register_format(self, af: PointFormat):
        """Decorator to register an AddressFormat for each component."""
        super().register_format(af)
        for side in self.sides:
            self.sides[side].register_format(af)

    # def canonical(self, address):
    #     """
    #     Convert a potentially non-canonical hex address into its canonical form,
    #     by reconstructing coordinates and re-encoding from the top down.
    #     """
    #     # Reconstruct (x, y), signs from address
    #     pt = self.decode(address)  # -> Point object, with coords + signs
    #     rec = self.h9map[address[:3]][1]
    #     result = self.h9e.oct_encode(pt, rec)
    #     return result

    def uint64(self, pts, depth=13):
        """
        Convert Point object to uint64 hierarchical address.
        """
        bits = ((- (pts.components - 1)) >> 1).astype(np.uint8)
        oct_n = (bits[:, 0] << 2) | (bits[:, 1] << 1) | bits[:, 2]
        loc_s = self.octant_props[oct_n]
        count = len(pts.coords)
        result = np.zeros(count, dtype=np.uint64)
        for i, ((px, py), oi, (loc, mode)) in enumerate(zip(pts.coords, oct_n, loc_s)):
            step = Step(loc, px, py, Style.U64)
            step, _ = self.h9e.encode_step(step)
            enc = (oi << 4 | step.c1) << 56  # First step is extracted solely to fetch c1 and init position.
            bits = 52
            d_acc = depth - 1
            while bits >= 4 and d_acc > 0:
                last = (d_acc == 1 or bits == 4)
                step, addr = self.h9e.encode_step(step, last)
                enc |= addr << bits
                bits -= 4
                d_acc -= 1
            enc |= ((step.tm << 3) | step.tr) << bits
            bits -= 4
            while bits >= 0:
                enc |= 15 << bits
                bits -= 4
            result[i] = enc
        return result


if __name__ == '__main__':
    from hhg9 import Registrar, H9Engine, Step, Points
    from hhg9.domains import OctahedralCartesian, OctahedralBarycentric
    from hhg9.formats import OctahedralH9
    reg = Registrar()  # Manage Domains & Projections
    c_oct = OctahedralCartesian(reg)  # Cartesian Octahedron (xyz)
    b_oct = OctahedralBarycentric(reg, c_oct)  # 2d Flat for addressing.
    h9 = OctahedralH9()  # formatter.
    b_oct.register_format(h9)
    greenwich = Points(
        np.array([[0.3041404, -0.28970996], [0.3041376, 0.28970996]]), b_oct,
        np.array([(+1, +1, +1), (+1, -1, +1)])
    )
    g_code = f'{greenwich:h9.12}'
    club = Points(
        np.array([[0.30298694022127, 0.2895875423442]]), b_oct,
        np.array([(+1, -1, +1)])
    )

    c_club = f'{club:h9}'
    e_code = b_oct.decode(c_club)
    print(club)
    # First is NAV02676076333V2 -> 'NAV02676076333Λ2'
    # dw = b_oct.canonical('NAV02676076333Λ2')
    # de = b_oct.canonical('NWΛ01686086336V1')
    # print(g_code)
    # print(dw, de)

    club = Points(np.array([[0.30298694, 0.28958754]]), b_oct, np.array([(+1, -1, +1)]))

    for depth in range(2, 16):
        u644 = b_oct.uint64(club, depth)
        hx = f'{u644[0]:016X}'
        print(depth, hx)
