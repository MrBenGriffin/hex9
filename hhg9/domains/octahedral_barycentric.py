# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
This is 'b_oct' barycentric xy equilateral.
"""

import numpy as np
from numpy.typing import NDArray
from hhg9.base.composite import CompositeDomain, ComponentDomain
from hhg9.base.point_format import PointFormat
from hhg9.projections import OctantBary
from hhg9.h9 import H9K, in_scope


class OctantBarycentric(ComponentDomain):
    """
    This a 2D side of an Octant.
    Validity should be easy enough since we have the 3 points that define it.
    """

    def __init__(self, registrar, dom, name: str, sign, cc: tuple):
        super().__init__(registrar, name, dom,  cc[2], sign, 2)
        self.th = (cc[0] % 6) * np.pi / 3.
        self.tr = cc[1]
        self.geo = {k: v for k, v in zip(self.tr, cc[3])}  # in c1 (orientation) order.
        self.reg = cc[4]
        # self.mo = cc[5]
        self._oc = cc[3]
        self.oc = np.array([cc[3][i]+self.mode_str for i in range(3)], dtype='U3')

    def h9_add(self, h9m):
        """compose addresses"""
        for k, v in self.geo.items():
            key = f'{v}{self.mode_str}'
            c2 = np.where(self.oc == key)[0][0]
            h9m[key] = {
                'mode': self.mode,
                'mode_str': self.mode_str,
                'name': self.name,
                'tr': self.tr,
                'component': self._sign,
                'id': self.dom.sign_to_id[self._sign],
                'c2': int(c2)
            }

    def valid(self, pts: NDArray) -> NDArray:
        """
        Return an array of bools according to the validity criterion
        :param pts: set of 2d Euclidean points
        """
        return in_scope(H9K.radical.R3 * pts[..., 0], pts[..., 1], self.mode)


class OctahedralBarycentric(CompositeDomain):
    """
    Basic octahedral-2d properties and methods.
    """

    def __init__(self, registrar):
        o = registrar.domain('c_oct')  # OctahedralCartesian
        super().__init__(registrar, 'b_oct', 2)
        self.sides = {}
        self.projs = {}
        self.signs = {}  # c_oct.signs  # These are used to tie the projection.
        self.h9map = {}
        self.components = {}
        self.h9 = registrar.format('h9')
        self.h9.composite = self
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
        # _components = {}
        props = {
            # Octahedral Triangle Identities differ. 0x16, 0x49
            # Note that OID do not share the id % 2 == mode logic(!)
            # Also, 'c2' is always equator=0; with nb.c2=1 == self.c2 = 2
            # AP EW  NS    θ   loc    V   c2 hexagon region  region mo o_id, c2* ngh  hex_id
            (+1, +1, +1): (2, '047', 'V', ('EA', 'NA', 'NE'), 0x49, 0, 0, (4, 2, 1), (0, 4, 5)),     # 0 'NEA' N:5, E:8, A: 0
            (-1, +1, +1): (5, '085', 'Λ', ('EP', 'NE', 'NP'), 0x16, 1, 1, (5, 0, 3), (1, 5, 7)),     # 1 'NEP' N:4, E:7, P: 0
            (+1, -1, +1): (5, '085', 'Λ', ('WA', 'NW', 'NA'), 0x16, 1, 2, (6, 3, 0), (2, 6, 4)),     # 2 'NWA'
            (-1, -1, +1): (2, '047', 'V', ('WP', 'NP', 'NW'), 0x49, 0, 3, (7, 1, 2), (3, 7, 6)),     # 3 'NWP' N:5, W:8, P: 0
            (+1, +1, -1): (5, '085', 'Λ', ('EA', 'SE', 'SA'), 0x16, 1, 4, (0, 5, 6), (0, 8, 10)),    # 4 'SEA' S:4, E:7, A: 0
            (-1, +1, -1): (2, '047', 'V', ('EP', 'SP', 'SE'), 0x49, 0, 5, (1, 7, 4), (1, 9, 8)),     # 5 'SEP' S:5, E:8, P: 0
            (+1, -1, -1): (2, '047', 'V', ('WA', 'SA', 'SW'), 0x49, 0, 6, (2, 4, 7), (2, 10, 11)),   # 6 'SWA' S:5, W:8, A: 0
            (-1, -1, -1): (5, '085', 'Λ', ('WP', 'SW', 'SP'), 0x16, 1, 7, (3, 6, 5), (3, 11, 9))     # 7 'SWP' S:4, W:7, P: 0
        }
        self.props = props
        self.sign_to_id = {sign: row[6] for sign, row in props.items()}
        # --- Derived lookups by face-id (avoid mixing region codes like 0x49 with face ids) ---

        self.props_by_id = {}
        self.edges_by_id = np.empty((8, 3), dtype='<U2')
        self.l0hex_by_id = np.empty((8, 3), dtype=np.uint8)
        self.l0hex_back = np.full((12, 2, 2), 0x0F, dtype=np.uint8)
        self.signs_by_id = {}

        for sign, row in props.items():
            face_id = row[6]
            face_mode = row[5]
            self.props_by_id[face_id] = (sign, row)
            self.edges_by_id[face_id] = np.array(row[3], dtype='<U2')
            self.signs_by_id[face_id] = sign
            l0_hex = np.array(row[8], dtype=np.uint8)
            self.l0hex_by_id[face_id] = l0_hex
            for c2, hx in enumerate(l0_hex):
                # prev_face, prev_c2 = self.l0hex_back[hx, face_mode]
                # if prev_face != 0xF:
                #     print(f'{hx} overwriting values ({prev_face}, {prev_c2}) with ({face_id}, {c2}) in lut is a key error')
                self.l0hex_back[hx, face_mode] = [face_id, c2]

        for sign, row in props.items():
            o_id = row[6]
            o_mode = row[5]
            l0_hex = np.array(row[8], dtype=np.uint8)
            for c2, hx in enumerate(l0_hex):
                self.l0hex_back[hx, o_mode] = [o_id, c2]

        for sign, face in o.signs.items():
            b_sig = f'{self.name}:{face}'
            o_sig = o.sides[face].name
            self.sides[face] = OctantBarycentric(registrar, self, b_sig, sign, props[sign])
            self.sides[face].h9_add(self.h9map)
            self.projs[face] = OctantBary(registrar, face, o_sig, b_sig)
            self.signs[sign] = face
            self.components[sign] = self.sides[face]

        self.prop_by_id = np.zeros((8,), dtype=np.uint8)

        self.oid_mo = np.zeros((8,), dtype=np.uint8)
        for t in props.values():  # given an octant id, return the mode.
            mode_, idx = t[5], t[6]
            self.oid_mo[idx] = mode_

        self.oid_cp = np.zeros((8, 3), dtype=np.int8)
        for octant, t in props.items():  # given an octant id, return the mode.
            idx = t[6]
            self.oid_cp[idx] = list(octant)

        self.oid_nb = np.zeros((8, 3), dtype=np.int8)
        for t in props.values():  # given an octant id, and c1 return the neigbour.
            idx, c2s = t[6], np.array(list(t[7]))
            self.oid_nb[idx] = c2s

        # --- Edge-level preserve/swap and per-(face,c1) mapping codes ---
        # Preserve edges by LABEL (not by slot)
        preserve_edges = {'EA', 'EP', 'NA', 'WA', 'WP', 'SA'}  # axial edges preserve; diagonals swap

        # Boolean preserve matrix: True where the edge label is in preserve_edges
        self.nb_c2p = np.zeros((8, 3), dtype=bool)
        for f in range(8):
            for i, e in enumerate(self.edges_by_id[f]):
                self.nb_c2p[f, i] = (e in preserve_edges)

        # Mapping code per (face,c1): 0=id for PRESERVE; for SWAP use 1+ci to pick the reflect axis
        # codes: 0=id, 1=reflect across C1=0 axis (horizontal), 2=reflect across C1=1 axis (y=ẋ), 3=reflect across C1=2 axis (y=-ẋ)
        self.nb_c2map = np.zeros((8, 3), dtype=np.uint8)
        swap_code = {
            # reflect across C1=1 axis (y = ẋ)
            'NP': 2, 'SE': 2, 'SP': 2,
            # reflect across C1=2 axis (y = −ẋ)
            'NE': 3, 'NW': 3, 'SW': 3,
        }
        for f in range(8):
            for ci, e in enumerate(self.edges_by_id[f]):
                if self.nb_c2p[f, ci]:
                    self.nb_c2map[f, ci] = 0
                else:
                    try:
                        self.nb_c2map[f, ci] = swap_code[e]
                    except KeyError as ke:
                        raise KeyError(f"Unknown swap edge label '{e}'. Ensure it is in swap_code or preserve_edges.") from ke

        # sanity: all 12 octahedral edge labels must be covered
        # edge_array = self.edges_by_id.reshape(-1)
        all_edge_labels = set(self.edges_by_id.reshape(-1))
        covered = preserve_edges.union(swap_code.keys())
        missing = all_edge_labels - covered
        assert not missing, f"Edge labels missing mapping: {sorted(missing)}"

        # Define base barycentric transformation matrices
        trans = np.array([[-1, 0], [1, -2], [1, 1]])  # Prototype [1,1,1]: Proj Z using √2, √6, √3 resp.
        r90 = np.array([(0, 1), (-1, 0)])  # 90-degree rotation matrix
        mirror_y_neg_x = np.array([(0, 1), (1, 0)])  # Mirror along y = x

        # Compute rotation matrices
        north, south = trans, trans @ mirror_y_neg_x  # South is the mirror of North
        # Loop in 90º rotation order and compute projection matrices for N and S.
        scale_factors = np.sqrt([2, 6, 3])[:, np.newaxis]
        self.rot90_idx = np.zeros(8, dtype=np.uint8)
        # These are set in order of rotation, starting with NEA
        rot = 0
        sigs = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
        for sig in sigs:
            n_sign = tuple([*sig, 1])
            s_sign = tuple([*sig, -1])
            n_face = self.signs[n_sign]
            s_face = self.signs[s_sign]
            self.projs[n_face].matrix = np.column_stack([north, np.ones(3)]) / scale_factors
            self.projs[s_face].matrix = np.column_stack([south, -np.ones(3)]) / scale_factors
            n_id = self.sign_to_id[n_sign]
            s_id = self.sign_to_id[s_sign]
            self.rot90_idx[n_id] = rot
            self.rot90_idx[s_id] = rot
            north = north @ r90
            south = south @ r90
            rot = (rot + 1) % 4

    def decode(self, addr):
        """Decode octahedral coordinates into a point"""
        return self.h9.revert(addr)

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
