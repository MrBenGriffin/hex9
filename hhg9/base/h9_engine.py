"""
    == Calculations ==
    Some of the methods may seem obscure: It is worth reading the documentation, especially
    covering the meaning of C1/C2,UD etc.
    Calculations are done for both an up-pointing triangle and a down-pointing triangle,
    because of the nature of the underlying triangle grid, so it becomes a little onerous
    to transform coordinates (a y-mirror around the origin ?!).
    especially as they repeatedly flip from one orientation to the other.

    == Variations ==
    There are two variations -
    * The planar variation, which uses a uniform hexagon layout at each stage.
    * The octahedral variation, which requires rotation due to the lack of planar transitivity.
    The planar variation is, in general, far more straightforward to intuit, given in planar
    the C1 represents both the orientation of the hexagon *and* it's low-(base3)digit value, whereas
    in the octahedral, we must calculate the hexagon value separate from it's C1.
    (C2 - centrality - is invariant across both variations).

    == Encodings ==
    === Planar ===
    Basic encodings has a long/short structure.
    Easily transposed -
    22035211610266553407865006553346V;
    2Λ2Λ0Λ3Λ5V2V1Λ1Λ6Λ1V0V2V6V6Λ5V5V3V4V0Λ7Λ8Λ6Λ5Λ0V0V6Λ5V5V3V3V4V6V
    the former can be expanded to the latter, however it requires reverse
    calculations especially when dealing with 678.
    Alternative encodings include
    'Extended': 678 for V and GTX for Λ. (`VΛ` convention).
    'HalfHex':  V[abc/def/ghi]; Λ[ABC/DEF/GHI]
                V[012/345/678]; Λ[oiz/eas/gtx]
    === Octahedral ===
    The canonical octahedral addresses is prefixed with a three-character octahedral root hexagon and hint.
     (from which the side of the octahedron can be derived). This value replaces the first digit of the
     planar addresses; also the terminating hexagon rotation is also stored as a part of the suffix.
     This latter allows us to precisely recover a location from the addresses.
     For example, stonehenge (uk) is approximately at NW013502061182541V2, while the statue of liberty (usa)
     has the addresses NA556384535621324Λ0
     """
from dataclasses import dataclass
from enum import Enum, unique
from functools import lru_cache, cache

import numpy as np


@unique
class Style(Enum):
    """
    Various Encoding styles.
    Most work has been done with HEX/FULL.
    However, for hexgrid binning, it may be useful to consider others.
    """
    HEX = 0
    FULL = 1
    EXTENDED = 2
    HALFHEX = 3
    NUMERIC = 4
    CFULL = 5
    U64 = 6


@dataclass
class Step:
    """The full state of an encoding step"""
    # loc: str  # e.g. '850'
    x: float  # cumulative x so far
    y: float  # cumulative y so far
    style: Style = Style.HEX
    s: float = 1.0
    # xa: float = 0.  # offset acc.
    # ya: float = 0.  # offset acc.
    c1: int = None  # add this!
    # tm: int = None  # terminating mode
    tr: int = None  # terminating rotation.


class H9Engine:
    """
    AKA H9 - This is a hierarchic hexagonal grid (HHG) that uses regular tetrahedrons
    """
    # fundamental constants
    H = np.sqrt(6) / 2.
    R3 = np.sqrt(3)
    # All other geometric constants derived from H and R3
    # Done in order to avoid tiny floating point deviations.
    TR = H / R3  #
    W = 2 * TR  # This correctly derives W = sqrt(2)
    ΛC = 2 * H / 3.
    ΛF = -H / 3.
    VC = H / 3.
    VF = -2 * H / 3.
    Ẇ = ΛC
    TL = -TR
    U, V = W / 6., H / 9.
    RH = R3 / 2.  # ratio of height to width. 0.8660

    OFS = {
        # These are subtractive.
        # What to remove from the coordinate, once found! So, for 2V.021 that's at the top!
        (0, 'Λ', '021'): (0, V * 2.),
        (0, 'Λ', '201'): (-U, V),
        (0, 'Λ', '102'): (-U * 2., V * 2.),
        (1, 'Λ', '102'): (U, -V),
        (1, 'Λ', '120'): (U, V),
        (1, 'Λ', '210'): (U * 2., V * 2.),
        (2, 'Λ', '210'): (-U, -V),
        (2, 'Λ', '012'): (0, -V * 2.),
        (2, 'Λ', '021'): (0, -V * 4.),
        (0, 'V', '012'): (0, -V * 2.),
        (0, 'V', '210'): (-U, -V),
        (0, 'V', '120'): (-U * 2., -V * 2.),
        (1, 'V', '201'): (-U, V),
        (1, 'V', '021'): (0, V * 2.),
        (1, 'V', '012'): (0, V * 4.),
        (2, 'V', '120'): (U, V),
        (2, 'V', '102'): (U, -V),
        (2, 'V', '201'): (U * 2., -V * 2.)
    }
    POS = [
        # The co-ordinate to the centre of each sub-triangle
        # The order starts from above the origin, and goes clockwise through the inner set, then the outer set.
        (0, V * 2.), (U, V), (U, -V),     # 0x26, 0x2a, 0x3a
        (0, -V * 2.), (-U, -V), (-U, V),  # 0x39, 0x35, 0x25
        (0, -V * 4.), (-U * 2., -V * 2.), (-U * 2., V * 2.),  # 0x49, 0x34, 0x21
        (0, V * 4.), (U * 2., V * 2.), (U * 2., -V * 2.)   # 0x16, 0x2b, 0x3e
    ]

    @classmethod
    def in_scope(cls, ẋ, y, mode='Λ') -> bool:
        """
        This is a barycentric scope test, for a unit equilateral triangle.
        This expects x to already be pre-calculated as √3(x).
        :param ẋ: `ẋ := √3(x)` on x co-ordinate.
        :param y: y co-ordinate
        :param mode: triangle pointing up/down
        :return: boolean (in scope or not)
        """
        e = 1e-15  # for polygons and edges, best to be a tiny bit safe.
        if mode == 'Λ':  # barycentre at 0, triangle point up.
            return (cls.ΛF <= y) & (y <= cls.ΛC - np.abs(ẋ) + e)
        else:  # barycentre at 0, triangle point down.
            return (cls.VF + np.abs(ẋ) - e <= y) & (y <= cls.VC)

    # @classmethod
    # def get_c1(cls, ẋ, y, mode='Λ'):
    #     """
    #     Given a point in a triangle, identify it's half-hex.
    #     The return value will be 0, 1, 2  for flat/forward/back.
    #     :param ẋ: `ẋ` is a synonym for `√3(x)` of the x coordinate.
    #     :param y: y coordinate
    #     :param mode: triangle pointing up/down
    #     :return: 0,1,2 representing the c1 component of the container triangle
    #     These are checked correct 30th June '25
    #     """
    #     if mode == 'Λ':
    #         if 0 >= y < ẋ:
    #             return 0  # flat
    #         elif ẋ <= y < -ẋ:
    #             return 1  # forward
    #         else:  # 0 <= y > -ẋ
    #             return 2  # back
    #     else:
    #         if 0 <= y > -ẋ:
    #             return 0  # flat
    #         elif 0 > y <= ẋ:
    #             return 1  # forward.
    #         else:  # ẋ < y <= -ẋ
    #             return 2  # back.

    # @classmethod
    # def get_c2(cls, ẋ, y, c1, ud='Λ'):
    #     """
    #     Given a point in a half-hex, identify its triangle
    #     get_c2 works on a half-hexagon, and establishes which of 3
    #     triangles a point is in, returning a 3-digit for that triangle.
    #     For example, our current planar value is 0Λ. (from 021)
    #     The c1 is 0, and the point we are looking
    #     at will be in either 507, 165, or 813 (021, 201, 102).
    #     The same calculation works for 6Λ and 3Λ.
    #     For octahedral (non-planar), we may use an equivalent c1.
    #     For example, in r0, the planar 7Λ is now 6Λ, but it's still c1=1 (Fwd).
    #     We use oc_1 to calculate the c1, and pass it here.
    #     :param c1: (0,1,2) represents the orientation: flat, forward, back.
    #            In planar, hexes 0,3,6 are Flat (0), 1,4,7 are Fwd (1), 2,5,8 are Back (2).
    #     :param ud: [Λ,V] tells us which side of the half-hexagon we are talking about.
    #            This will be Λ for any half-hexagon with two apexes Λ and one V in the middle.
    #     'Λ' and 'V' tells us which of the two half-hexes we are examining.
    #     # c1, ẋ, y to identify the next c2
    #     # return c2 is not determined by the input mode! It will be any one of the six available.
    #     :param ẋ: `ẋ` is a synonym for `√3(x)` of the x coordinate.
    #     :param y: y coordinate
    #     :return: Will be one of ['201', '120', '012', '210', '021', '102']
    #     """
    #     if ud == 'Λ':
    #         if c1 == 0:
    #             if y <= -ẋ:  # alt y <= -ẋ
    #                 return '021'  # √ y <= -ẋ identifies 021
    #             if y <= ẋ - cls.Ẇ:  # alt y <= ẋ - cls.Ẇ
    #                 return '102'  # √ y <= ẋ-ẇ identifies 102
    #             return '201'  # √ y < -ẋ and y > ẋ - ẇ identifies 201
    #         if c1 == 1:
    #             if y >= 0:
    #                 return '102'  # √ y >= 0 identifies 102
    #             if y <= -ẋ - cls.Ẇ:  #
    #                 return '210'  # √ y ≤ -ẋ-ẇ identifies 210
    #             return '120'  # √ y < 0 and `y > -ẋ-ẇ` identifies 120
    #         # c1 == 2
    #         if y <= ẋ:  # alt y <= ẋ
    #             return '210'  # √ `y <= ẋ identifies 210
    #         if y >= cls.H / 3.:
    #             return '021'  # √ y >= h/3 identifies 021
    #         return '012'  # √ y > ẋ and y < h/3 identifies 012
    #     else:  # Now for 'V'
    #         # For points in 0V (flat),
    #         # y >= ẋ identifies 012
    #         # y >= ẇ-ẋ identifies 120
    #         # y < ẋ and y < ẇ-ẋ identifies `210`
    #         if c1 == 0:
    #             if y >= ẋ:  # alt y >= ẋ
    #                 return '012'  # √ y >= ẋ identifies 012
    #             if y >= cls.Ẇ - ẋ:  # alt y >= cls.Ẇ - ẋ
    #                 return '120'  # y >= ẇ-ẋ identifies 120
    #             return '210'  # y < ẋ and y < ẇ-ẋ identifies 210
    #         # For points in `1V` (forward),
    #         # y >= -ẋ identifies 201
    #         # y <= -h/3 identifies 012
    #         # y > -h/3 and y < -ẋ identifies 021
    #         if c1 == 1:
    #             if y >= -ẋ:  # alt y >= -ẋ
    #                 return '201'  # y >= -ẋ identifies 201
    #             if y <= -cls.VC:
    #                 return '012'  # y <= -h/3 identifies 012
    #             return '021'  # y > -h/3 and y < -ẋ identifies 021
    #         # For points in `2V` (back),
    #         # y <= 0 identifies 120
    #         # y >= ẇ+ẋ identifies 201
    #         # y > 0 and y < ẇ+ẋ identifies 102
    #         if y <= 0:
    #             return '120'  # y <= 0 identifies 120
    #         if y >= cls.Ẇ + ẋ:  # alt cls.Ẇ + ẋ:
    #             return '201'  # y >= ẇ+ẋ identifies 201
    #         return '102'  # y > 0 and y < ẇ+ẋ identifies 102

    def __init__(self):
        # These are used to define octahedral enumerations.
        self.region_ids = None
        self.ugc_num_props = 11
        self.num_regions = 96
        self.in_dn, self.in_up, self.mode, self.d_ci, self.u_ci, self.dc0, self.dc1, self.dc2, self.uc0, self.uc1, self.uc2 = range(
            self.ugc_num_props)
        self.invalid_ugc = 0x5F
        self.ugc_lut = {}
        self.ugc_off = {}
        self.ugc_rev = None
        self.uro = {  # (v 0626.2) given a triangle, identify the rotation and mode.
            '085': (0, 'Λ'), '850': (1, 'Λ'), '508': (2, 'Λ'),
            '316': (0, 'Λ'), '163': (1, 'Λ'), '631': (2, 'Λ'),
            '742': (0, 'Λ'), '427': (1, 'Λ'), '274': (2, 'Λ'),
            '047': (0, 'V'), '470': (1, 'V'), '704': (2, 'V'),
            '815': (0, 'V'), '158': (1, 'V'), '581': (2, 'V'),
            '362': (0, 'V'), '623': (1, 'V'), '236': (2, 'V'),
        }
        # self.ocm = {
        #     # Given a triangle, identify the rotation of the hex digit (via index) and it's mode.
        #     '085': ('012012201', 'Λ'),
        #     '850': ('120120012', 'Λ'),
        #     '508': ('201201120', 'Λ'),
        #     '316': ('012012201', 'Λ'),
        #     '163': ('120120012', 'Λ'),
        #     '631': ('201201120', 'Λ'),
        #     '742': ('012012120', 'V'),
        #     '427': ('120120201', 'V'),
        #     '274': ('201201012', 'V'),
        #     '047': ('012012120', 'V'),
        #     '470': ('120120201', 'V'),
        #     '704': ('201201012', 'V'),
        #     '815': ('012012201', 'Λ'),
        #     '158': ('120120012', 'Λ'),
        #     '581': ('201201120', 'Λ'),
        #     '362': ('012012120', 'V'),
        #     '623': ('120120201', 'V'),
        #     '236': ('201201012', 'V'),
        # }
        self.o2p = {}
        self.p2o = {}
        self.oc2 = {}
        self.ard = {}
        self.in_regions = None
        self.in_up_regions = None
        self.in_dn_regions = None
        self.hmc = None
        # self.test = None
        self.rgs = None
        self.xnb = None  # external neighbour regions.
        # self.neighbour_lut = None  # neighbour regions.
        self.child_lut = None
        self.pqc1_lut = None  # given parent, child - find c1
        self.pch = []
        self.next_oc2 = {}
        self._define_luts()
        # self.modev = {
        #     'Λ': [
        #         (0, '021'), (0, '201'), (0, '102'),
        #         (1, '102'), (1, '120'), (1, '210'),
        #         (2, '210'), (2, '012'), (2, '021')],
        #     'V': [
        #         (0, '012'), (0, '210'), (0, '120'),
        #         (1, '201'), (1, '021'), (1, '012'),
        #         (2, '120'), (2, '102'), (2, '201')]
        # }
        # self.o2m = {
        #     '021': 'Λ', '102': 'Λ', '210': 'Λ',
        #     '012': 'V', '120': 'V', '201': 'V',
        # }
        self.poly_hh, _ = self._poly_luts()

    def _define_luts(self):
        """
        Given the class dictionary of octahedral mappings `uro`
        set the octahedral tri to planar tri table and its inverse.
        """
        self.ugc_lut_init()
        for tri, (rot, mode) in self.uro.items():
            # octahedral is indexed by C1 orientation, merely extract the (invariant) C2 upper trit.
            planar_id = ''.join(str(int(d) // 3) for d in tri)
            self.o2p[tri] = (planar_id, rot, mode)
        for tri, (planar_id, rot, mode) in self.o2p.items():
            key = (planar_id, rot, mode)
            self.p2o[key] = tri
            for c1, hx in enumerate(tri):
                self.oc2[(hx, c1, mode)] = tri
        cc2 = {  # Given a mode, return (in c1 order) the three triangles that make each c1.
            'Λ': [('085', '815', '316'), ('508', '581', '631'), ('850', '158', '163')],
            'V': [('047', '742', '362'), ('704', '274', '236'), ('470', '427', '623')]
        }
        for o_mo in ['Λ', 'V']:
            for c1 in [0, 1, 2]:
                chh = cc2[o_mo][c1]
                for cc in chh:
                    _, cm = self.uro[cc]  # got the up/down.
                    for ch in cc:
                        self.next_oc2[(o_mo, c1, ch, cm)] = cc
        # self.cc2_pc2 = {
        #     # ('085', '815', '316')
        #     ('0', '085'): '085',
        #     ('1', '085'): '163',
        #     ('2', '085'): '274',
        #     ('3', '085'): '316',
        #     ('4', '085'): '427',
        #     ('5', '085'): '508',
        #     ('6', '085'): '631',
        #     ('7', '085'): '742',
        #     ('8', '085'): '850',
        #
        #     ('0', '815'): '085',
        #     ('1', '815'): '163',
        #     ('2', '815'): '274',
        #     ('3', '815'): '316',
        #     ('4', '815'): '427',
        #     ('5', '815'): '508',
        #     ('6', '815'): '631',
        #     ('7', '815'): '742',
        #     ('8', '815'): '850',
        #
        #     ('0', '316'): '085',
        #     ('1', '316'): '163',
        #     ('2', '316'): '274',
        #     ('3', '316'): '316',
        #     ('4', '316'): '427',
        #     ('5', '316'): '508',
        #     ('6', '316'): '631',
        #     ('7', '316'): '742',
        #     ('8', '316'): '850',
        #
        #     # ('508', '581', '631')
        #     ('0', '508'): '508',
        #     ('1', '508'): '316',
        #     ('2', '508'): '427',
        #     ('3', '508'): '631',
        #     ('4', '508'): '742',
        #     ('5', '508'): '850',
        #     ('6', '508'): '163',
        #     ('7', '508'): '274',
        #     ('8', '508'): '085',
        #
        #     ('0', '581'): '508',
        #     ('1', '581'): '316',
        #     ('2', '581'): '427',
        #     ('3', '581'): '631',
        #     ('4', '581'): '742',
        #     ('5', '581'): '850',
        #     ('6', '581'): '163',
        #     ('7', '581'): '274',
        #     ('8', '581'): '085',
        #
        #     ('0', '631'): '508',
        #     ('1', '631'): '316',
        #     ('2', '631'): '427',
        #     ('3', '631'): '631',
        #     ('4', '631'): '742',
        #     ('5', '631'): '850',
        #     ('6', '631'): '163',
        #     ('7', '631'): '274',
        #     ('8', '631'): '085',
        #
        #     # ('850', '158', '163')
        #     ('0', '850'): '850',
        #     ('1', '850'): '631',
        #     ('2', '850'): '742',
        #     ('3', '850'): '163',
        #     ('4', '850'): '274',
        #     ('5', '850'): '085',
        #     ('6', '850'): '316',
        #     ('7', '850'): '427',
        #     ('8', '850'): '508',
        #
        #     ('0', '158'): '850',
        #     ('1', '158'): '631',
        #     ('2', '158'): '742',
        #     ('3', '158'): '163',
        #     ('4', '158'): '274',
        #     ('5', '158'): '085',
        #     ('6', '158'): '316',
        #     ('7', '158'): '427',
        #     ('8', '158'): '508',
        #
        #     ('0', '163'): '850',
        #     ('1', '163'): '631',
        #     ('2', '163'): '742',
        #     ('3', '163'): '163',
        #     ('4', '163'): '274',
        #     ('5', '163'): '085',
        #     ('6', '163'): '316',
        #     ('7', '163'): '427',
        #     ('8', '163'): '508',
        #
        #     # ('047', '742', '362')
        #     ('0', '047'): '047',
        #     ('1', '047'): '158',
        #     ('2', '047'): '236',
        #     ('3', '047'): '362',
        #     ('4', '047'): '470',
        #     ('5', '047'): '581',
        #     ('6', '047'): '623',
        #     ('7', '047'): '704',
        #     ('8', '047'): '815',
        #
        #     ('0', '742'): '047',
        #     ('1', '742'): '158',
        #     ('2', '742'): '236',
        #     ('3', '742'): '362',
        #     ('4', '742'): '470',
        #     ('5', '742'): '581',
        #     ('6', '742'): '623',
        #     ('7', '742'): '704',
        #     ('8', '742'): '815',
        #
        #     ('0', '362'): '047',
        #     ('1', '362'): '158',
        #     ('2', '362'): '236',
        #     ('3', '362'): '362',
        #     ('4', '362'): '470',
        #     ('5', '362'): '581',
        #     ('6', '362'): '623',
        #     ('7', '362'): '704',
        #     ('8', '362'): '815',
        #
        #     # ('704', '274', '236')
        #     ('0', '704'): '704',
        #     ('1', '704'): '815',
        #     ('2', '704'): '623',
        #     ('3', '704'): '236',
        #     ('4', '704'): '047',
        #     ('5', '704'): '158',
        #     ('6', '704'): '362',
        #     ('7', '704'): '470',
        #     ('8', '704'): '581',
        #
        #     ('0', '274'): '704',
        #     ('1', '274'): '815',
        #     ('2', '274'): '623',
        #     ('3', '274'): '236',
        #     ('4', '274'): '047',
        #     ('5', '274'): '158',
        #     ('6', '274'): '362',
        #     ('7', '274'): '470',
        #     ('8', '274'): '581',
        #
        #     ('0', '236'): '704',
        #     ('1', '236'): '815',
        #     ('2', '236'): '623',
        #     ('3', '236'): '236',
        #     ('4', '236'): '047',
        #     ('5', '236'): '158',
        #     ('6', '236'): '362',
        #     ('7', '236'): '470',
        #     ('8', '236'): '581',
        #
        #     # ('470', '427', '623')
        #     ('0', '470'): '470',
        #     ('1', '470'): '581',
        #     ('2', '470'): '362',
        #     ('3', '470'): '623',
        #     ('4', '470'): '704',
        #     ('5', '470'): '815',
        #     ('6', '470'): '236',
        #     ('7', '470'): '047',
        #     ('8', '470'): '158',
        #
        #     ('0', '427'): '470',
        #     ('1', '427'): '581',
        #     ('2', '427'): '362',
        #     ('3', '427'): '623',
        #     ('4', '427'): '704',
        #     ('5', '427'): '815',
        #     ('6', '427'): '236',
        #     ('7', '427'): '047',
        #     ('8', '427'): '158',
        #
        #     ('0', '623'): '470',
        #     ('1', '623'): '581',
        #     ('2', '623'): '362',
        #     ('3', '623'): '623',
        #     ('4', '623'): '704',
        #     ('5', '623'): '815',
        #     ('6', '623'): '236',
        #     ('7', '623'): '047',
        #     ('8', '623'): '158',
        # }

    def ugc_lut_init(self):
        """Generate the UGC Lut as a numpy array"""
        """File 'NWA addressing 01' and grid_basis' holds the values here"""

        num_regions = self.num_regions
        # Each region holds a series of (self.ugc_num_props) properties:
        # * in_dn tells us that this region is found in the V mode triangle. (9 regions)
        # * in_up tells us that this region is found in the Λ mode triangle. (9 regions)
        # * mode tells us that this region is itself either a Λ (1) or V (0)
        # * u_ci tells which c1 this region is in (when in Λ mode). (six regions serve both modes - so we keep both)
        # * d_ci tells which c1 this region is in (when in V mode). (six regions serve both modes - so we keep both)
        # * uc012 gives us the three Λ c1 regions that this region serves.
        # * dc012 gives us the three V c1 regions that this region serves
        # up, being 1, is set after down.
        # in_dn, in_up, mode, d_ci, u_ci, dc0, dc1, dc2, uc0, uc1, uc2 = range(self.ugc_num_props)
        self.ugc_lut = np.full((num_regions, self.ugc_num_props), self.invalid_ugc, dtype=np.uint8)
        self.ugc_off = np.full((num_regions, 2), 0., dtype=np.float64)
        # self.in_regions = np.array([0x39, 0x35, 0x25, 0x26, 0x2a, 0x3a, 0x49, 0x34, 0x21, 0x16, 0x2b, 0x3e], dtype=np.uint8)
        self.in_regions = np.array([0x26, 0x2a, 0x3a, 0x39, 0x35, 0x25, 0x49, 0x34, 0x21, 0x16, 0x2b, 0x3e], dtype=np.uint8)
        self.region_ids = np.full(96, self.invalid_ugc, dtype=np.uint8)
        self.ugc_off[self.in_regions] = self.POS
        self.ugc_lut[:, self.in_dn] = 0
        self.ugc_lut[:, self.in_up] = 0
        self.ugc_lut[self.in_regions, self.mode] = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        self.ugc_lut[self.invalid_ugc] = 0  # set nothing for illegal!
        self.in_up_regions = [  # 9 regions serve Λ mode
            0x39, 0x3a, 0x3e,  # c0 ΛVΛ
            0x25, 0x35, 0x34,  # c1 ΛVΛ
            0x2a, 0x26, 0x16,  # c2 ΛVΛ
        ]
        u_cid = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        up_locs = [
            (0, 8, 5), (8, 1, 5), (3, 1, 6),  # 0: 0x39, 0x3a, 0x3e
            (5, 0, 8), (5, 8, 1), (6, 3, 1),  # 1: 0x25, 0x35, 0x3
            (8, 5, 0), (1, 5, 8), (1, 6, 3),  # 2: 0x2a, 0x26, 0x16
        ]
        self.in_dn_regions = [
            0x26, 0x2a, 0x2b,  # c0 VΛV
            0x3a, 0x39, 0x49,  # c1 VΛV
            0x35, 0x25, 0x21,  # c2 VΛV
        ]
        d_cid = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        dn_locs = [
            (0, 4, 7), (7, 4, 2), (3, 6, 2),  # 0: 0x26, 0x2a, 0x2b
            (7, 0, 4), (2, 7, 4), (2, 3, 6),  # 1: 0x3a, 0x39, 0x49
            (4, 7, 0), (4, 2, 7), (6, 2, 3),  # 2: 0x35, 0x25, 0x21
        ]
        self.pch = [
            [  # mode down V
                [  # c0
                    (0, 0x26), (4, 0x26), (7, 0x26),
                    (7, 0x2a), (4, 0x2a), (2, 0x2a),
                    (3, 0x2b), (6, 0x2b), (2, 0x2b),
                ],
                [  # c1
                    (7, 0x3a), (0, 0x3a), (4, 0x3a),
                    (2, 0x39), (7, 0x39), (4, 0x39),
                    (2, 0x49), (3, 0x49), (6, 0x49),
                ],
                [  # c2
                    (4, 0x35), (7, 0x35), (0, 0x35),
                    (4, 0x25), (2, 0x25), (7, 0x25),
                    (6, 0x21), (2, 0x21), (3, 0x21),
                ]
            ], [  # mode up Λ
                [  # c0
                    (0, 0x39), (8, 0x39), (5, 0x39),
                    (8, 0x3a), (1, 0x3a), (5, 0x3a),
                    (3, 0x3e), (1, 0x3e), (6, 0x3e),
                ], [  # c1
                    (5, 0x25), (0, 0x25), (8, 0x25),
                    (5, 0x35), (8, 0x35), (1, 0x35),
                    (6, 0x34), (3, 0x34), (1, 0x34),
                ], [  # c2
                    (8, 0x2a), (5, 0x2a), (0, 0x2a),
                    (1, 0x26), (5, 0x26), (8, 0x26),
                    (1, 0x16), (6, 0x16), (3, 0x16),
                ]
            ]
        ]
        _hmc = {  # Keys(hx,mode,c1 (Λ:=1;V:=0) (hx,mode,c1)=>ch.region
            (0, 1, 0): 0x39, (0, 1, 1): 0x25, (0, 1, 2): 0x2a, (0, 0, 0): 0x26, (0, 0, 1): 0x3a, (0, 0, 2): 0x35,
            (1, 1, 0): 0x16, (1, 1, 1): 0x3e, (1, 1, 2): 0x3a, (1, 0, 0): 0x26, (1, 0, 1): 0x3a, (1, 0, 2): 0x35,
            (2, 1, 0): 0x39, (2, 1, 1): 0x25, (2, 1, 2): 0x2a, (2, 0, 0): 0x49, (2, 0, 1): 0x21, (2, 0, 2): 0x2b,
            (3, 1, 0): 0x3e, (3, 1, 1): 0x34, (3, 1, 2): 0x16, (3, 0, 0): 0x2b, (3, 0, 1): 0x49, (3, 0, 2): 0x21,
            (4, 1, 0): 0x25, (4, 1, 1): 0x2a, (4, 1, 2): 0x39, (4, 0, 0): 0x35, (4, 0, 1): 0x26, (4, 0, 2): 0x3a,
            (5, 1, 0): 0x25, (5, 1, 1): 0x2a, (5, 1, 2): 0x39, (5, 0, 0): 0x35, (5, 0, 1): 0x26, (5, 0, 2): 0x3a,
            (6, 1, 0): 0x34, (6, 1, 1): 0x16, (6, 1, 2): 0x3e, (6, 0, 0): 0x21, (6, 0, 1): 0x2b, (6, 0, 2): 0x49,
            (7, 1, 0): 0x2a, (7, 1, 1): 0x39, (7, 1, 2): 0x25, (7, 0, 0): 0x3a, (7, 0, 1): 0x35, (7, 0, 2): 0x26,
            (8, 1, 0): 0x2a, (8, 1, 1): 0x39, (8, 1, 2): 0x25, (8, 0, 0): 0x3a, (8, 0, 1): 0x35, (8, 0, 2): 0x26,
        }
        self.rgs = np.full([96], 'X', dtype='<U1')
        rgs = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M']
        for i, rg in enumerate(self.in_regions):
            self.rgs[rg] = rgs[i]
            self.ard[rgs[i]] = rg
        self.ard['X'] = self.invalid_ugc
        self.hmc = np.zeros([9, 2, 3], dtype=np.uint8)
        for (ih, im, ic), rg in _hmc.items():
            self.hmc[ih, im, ic] = rg
        # Populate legal regions
        self.ugc_lut[self.in_up_regions, self.in_up] = 1
        self.ugc_lut[self.in_up_regions, self.u_ci] = u_cid
        for lc, rg in zip(up_locs, self.in_up_regions):
            for cx, hx in zip([self.uc0, self.uc1, self.uc2], lc):  # [6,7,8] [0,8,5]
                self.ugc_lut[rg, cx] = hx
        self.ugc_lut[self.in_dn_regions, self.in_dn] = 1
        self.ugc_lut[self.in_dn_regions, self.d_ci] = d_cid
        for lc, rg in zip(dn_locs, self.in_dn_regions):
            for cx, hx in zip([self.dc0, self.dc1, self.dc2], lc):
                self.ugc_lut[rg, cx] = hx

        self.region_ids[self.in_regions] = np.arange(len(self.in_regions), dtype=np.int8)
        self.ugc_rev = np.full((9, 9, num_regions), self.invalid_ugc, dtype=np.uint8)
        for p_reg in self.in_regions:
            p_props = self.ugc_lut[p_reg]
            mode = p_props[2]
            c_hxt = self.pch[mode]
            for a_mode in range(2):  # the ancestral mode value is different from current.
                p_of = self.dc0 + a_mode * 3
                if p_props[p_of] != self.invalid_ugc:
                    p_hxt = p_props[[p_of, p_of + 1, p_of + 2]]  # three hex digits this parent produces
                    for p_c1, p_hx in enumerate(p_hxt):
                        for (c_hx, c_reg) in c_hxt[p_c1]:
                            # self.test[(int(p_hx), c_hx, c_reg)] = p_reg
                            self.ugc_rev[int(p_hx), c_hx, c_reg] = p_reg

        # Now populate the neighbour relations for each region.
        # self.neighbour_lut = np.full((num_regions, 2, 3), self.invalid_ugc, dtype=np.uint8)
        # # Given a region, parent mode, c1, return the neighbour
        # _ngh_dict = {
        #     (0x16, 1): [0x26, 0x2B, 0x21],
        #     (0x21, 1): [0x34, 0x25, 0x16],
        #     (0x25, 1): [0x35, 0x3A, 0x26],
        #     (0x26, 1): [0x16, 0x2A, 0x25],
        #     (0x2A, 1): [0x3A, 0x26, 0x35],
        #     (0x2B, 1): [0x3E, 0x16, 0x2A],
        #     (0x34, 1): [0x21, 0x49, 0x35],
        #     (0x35, 1): [0x25, 0x39, 0x34],
        #     (0x39, 1): [0x26, 0x35, 0x3A],
        #     (0x3A, 1): [0x2A, 0x3E, 0x39],
        #     (0x3E, 1): [0x2B, 0x3A, 0x49],
        #     (0x49, 1): [0x39, 0x34, 0x3E],
        #     (0x16, 0): [0x26, 0x2B, 0x21],
        #     (0x21, 0): [0x34, 0x25, 0x16],
        #     (0x25, 0): [0x35, 0x21, 0x26],
        #     (0x26, 0): [0x39, 0x2A, 0x25],
        #     (0x2A, 0): [0x3A, 0x26, 0x2B],
        #     (0x2B, 0): [0x3E, 0x16, 0x2A],
        #     (0x34, 0): [0x21, 0x49, 0x35],
        #     (0x35, 0): [0x25, 0x39, 0x2A],
        #     (0x39, 0): [0x49, 0x35, 0x3A],
        #     (0x3A, 0): [0x2A, 0x25, 0x39],
        #     (0x3E, 0): [0x2B, 0x3A, 0x49],
        #     (0x49, 0): [0x39, 0x34, 0x3E],
        # }
        # for key, neighbours in _ngh_dict.items():
        #     region_id, mode = key
        #     self.neighbour_lut[region_id, mode] = neighbours

        # Given a mode and c1, find the regions that belong there.
        _chd = {
            (0, 0): [0x26, 0x2A, 0x2B],  # V,C1.0
            (0, 1): [0x3A, 0x39, 0x49],  # V,C1.1
            (0, 2): [0x35, 0x25, 0x21],  # V,C1.2
            (1, 0): [0x39, 0x3A, 0x3E],  # Λ,C1.0
            (1, 1): [0x25, 0x35, 0x34],  # Λ,C1.1
            (1, 2): [0x2A, 0x26, 0x16],  # Λ,C1.2
        }
        self.child_lut = np.zeros((2, 3, 3), dtype=np.uint8)
        for (mode, c1), children in _chd.items():
            self.child_lut[mode, c1] = children

        _pqc1 = {  # this returns the c1 based on parent/child. It could be automated.
            (0x21, 0x26): 0, (0x21, 0x2a): 0, (0x21, 0x2b): 0,
            (0x21, 0x3a): 1, (0x21, 0x39): 1, (0x21, 0x49): 1,
            (0x21, 0x35): 2, (0x21, 0x25): 2, (0x21, 0x21): 2,
            (0x26, 0x26): 0, (0x26, 0x2a): 0, (0x26, 0x2b): 0,
            (0x26, 0x3a): 1, (0x26, 0x39): 1, (0x26, 0x49): 1,
            (0x26, 0x35): 2, (0x26, 0x25): 2, (0x26, 0x21): 2,
            (0x2b, 0x26): 0, (0x2b, 0x2a): 0, (0x2b, 0x2b): 0,
            (0x2b, 0x3a): 1, (0x2b, 0x39): 1, (0x2b, 0x49): 1,
            (0x2b, 0x35): 2, (0x2b, 0x25): 2, (0x2b, 0x21): 2,
            (0x35, 0x26): 0, (0x35, 0x2a): 0, (0x35, 0x2b): 0,
            (0x35, 0x3a): 1, (0x35, 0x39): 1, (0x35, 0x49): 1,
            (0x35, 0x35): 2, (0x35, 0x25): 2, (0x35, 0x21): 2,
            (0x3a, 0x26): 0, (0x3a, 0x2a): 0, (0x3a, 0x2b): 0,
            (0x3a, 0x3a): 1, (0x3a, 0x39): 1, (0x3a, 0x49): 1,
            (0x3a, 0x35): 2, (0x3a, 0x25): 2, (0x3a, 0x21): 2,
            (0x49, 0x26): 0, (0x49, 0x2a): 0, (0x49, 0x2b): 0,
            (0x49, 0x3a): 1, (0x49, 0x39): 1, (0x49, 0x49): 1,
            (0x49, 0x35): 2, (0x49, 0x25): 2, (0x49, 0x21): 2,
            (0x16, 0x39): 0, (0x16, 0x3a): 0, (0x16, 0x3e): 0,
            (0x16, 0x25): 1, (0x16, 0x35): 1, (0x16, 0x34): 1,
            (0x16, 0x2a): 2, (0x16, 0x26): 2, (0x16, 0x16): 2,
            (0x25, 0x39): 0, (0x25, 0x3a): 0, (0x25, 0x3e): 0,
            (0x25, 0x25): 1, (0x25, 0x35): 1, (0x25, 0x34): 1,
            (0x25, 0x2a): 2, (0x25, 0x26): 2, (0x25, 0x16): 2,
            (0x2a, 0x39): 0, (0x2a, 0x3a): 0, (0x2a, 0x3e): 0,
            (0x2a, 0x25): 1, (0x2a, 0x35): 1, (0x2a, 0x34): 1,
            (0x2a, 0x2a): 2, (0x2a, 0x26): 2, (0x2a, 0x16): 2,
            (0x34, 0x39): 0, (0x34, 0x3a): 0, (0x34, 0x3e): 0,
            (0x34, 0x25): 1, (0x34, 0x35): 1, (0x34, 0x34): 1,
            (0x34, 0x2a): 2, (0x34, 0x26): 2, (0x34, 0x16): 2,
            (0x39, 0x39): 0, (0x39, 0x3a): 0, (0x39, 0x3e): 0,
            (0x39, 0x25): 1, (0x39, 0x35): 1, (0x39, 0x34): 1,
            (0x39, 0x2a): 2, (0x39, 0x26): 2, (0x39, 0x16): 2,
            (0x3e, 0x39): 0, (0x3e, 0x3a): 0, (0x3e, 0x3e): 0,
            (0x3e, 0x25): 1, (0x3e, 0x35): 1, (0x3e, 0x34): 1,
            (0x3e, 0x2a): 2, (0x3e, 0x26): 2, (0x3e, 0x16): 2,

        }
        self.pqc1_lut = np.zeros((num_regions, num_regions), dtype=np.uint8)
        for (p_reg, c_reg), c1 in _pqc1.items():
            self.pqc1_lut[p_reg, c_reg] = c1

        self.xnb = np.zeros((2, 3, 2), dtype=np.uint8)
        _data = {
            (0, 0): [0x34, 0x3E], (0, 1): [0x25, 0x34], (0, 2): [0x2A, 0x16],
            (1, 0): [0x26, 0x2B], (1, 1): [0x3A, 0x49], (1, 2): [0x35, 0x21]
        }
        for (mode, c1), children in _data.items():
            self.xnb[mode, c1] = children

    @cache
    def local_offset_lut(self):
        """
        Builds and caches the local/neighbouring offset LUT.
        Give self_mode, parent_mode, c1, sibling-bool return the offset to the neighbour.
        """
        _lut = [
            # self mode = 0
            #   pm c1 sib
            (0, 0, 0, 1, (0, 2)),      # mode 0, c1:0
            (0, 0, 1, 1, (1, -1)),      # k
            (0, 0, 2, 1, (-1, -1)),
            (0, 1, 0, 1, (0, 2)),
            (0, 1, 1, 1, (1, -1)),      # k
            (0, 1, 2, 1, (-1, -1)),
            (0, 0, 0, 0, (0, -2)),      # Not sure this is possible.
            (0, 0, 1, 0, (-1, 1)),      # k
            (0, 0, 2, 0, (-1, -1)),
            (0, 1, 0, 0, (0, 2)),
            (0, 1, 1, 0, (1, -1)),      # k
            (0, 1, 2, 0, (-1, -1)),
            # self mode = 1
            (1, 0, 0, 1, (0, -2)),      # k
            (1, 0, 1, 1, (-1, 1)),      # k
            (1, 0, 2, 1, (1, 1)),       # k
            (1, 1, 0, 1, (0, -2)),
            (1, 1, 1, 1, (-1, 1)),
            (1, 1, 2, 1, (1, 1)),
            (1, 0, 0, 0, (0, -2)),
            (1, 0, 1, 0, (-1, 1)),     # k
            (1, 0, 2, 0, (1, 1)),      # k
            (1, 1, 0, 0, (0, 2)),
            (1, 1, 1, 0, (-1, 1)),
            (1, 1, 2, 0, (1, 1)),
        ]
        mx = np.array([self.W/2, self.H/3])
        local_offset_lut = np.zeros((2, 2, 3, 2, 2), dtype=np.float64)
        for (smo, pmo, c1, sib, oxy) in _lut:
            local_offset_lut[smo, pmo, c1, sib] = oxy * mx
        return local_offset_lut

    @cache
    def neighbour_lut(self):
        """
        Builds and caches a LUT of neighbours according to parent mode
        Give region, parent-mode, parent-c1, return the neighbour.
        """
        # Given a region, parent mode, region-c1, return the neighbour and parent mode.
        # If the parent mode has changed, then the region parent is a neighbour.
        _lut = {
            (0x16, 1): [(0x26, 1), (0x2B, 0), (0x21, 0)],
            (0x21, 1): [(0x5F, 1), (0x5F, 1), (0x5F, 1)],
            (0x25, 1): [(0x35, 1), (0x3A, 0), (0x26, 1)],
            (0x26, 1): [(0x16, 1), (0x2A, 1), (0x25, 1)],
            (0x2A, 1): [(0x3A, 1), (0x26, 1), (0x35, 0)],
            (0x2B, 1): [(0x5F, 1), (0x5F, 1), (0x5F, 1)],
            (0x34, 1): [(0x21, 0), (0x49, 0), (0x35, 1)],
            (0x35, 1): [(0x25, 1), (0x39, 1), (0x34, 1)],
            (0x39, 1): [(0x26, 0), (0x35, 1), (0x3A, 1)],
            (0x3A, 1): [(0x2A, 1), (0x3E, 1), (0x39, 1)],
            (0x3E, 1): [(0x2B, 0), (0x3A, 1), (0x49, 0)],
            (0x49, 1): [(0x5F, 1), (0x5F, 1), (0x5F, 1)],
            (0x16, 0): [(0x5F, 0), (0x5F, 0), (0x5F, 0)],
            (0x21, 0): [(0x34, 1), (0x25, 0), (0x16, 1)],
            (0x25, 0): [(0x35, 0), (0x21, 0), (0x26, 0)],
            (0x26, 0): [(0x39, 1), (0x2A, 0), (0x25, 0)],
            (0x2A, 0): [(0x3A, 0), (0x26, 0), (0x2B, 0)],
            (0x2B, 0): [(0x3E, 1), (0x16, 1), (0x2A, 0)],
            (0x34, 0): [(0x5F, 0), (0x5F, 0), (0x5F, 0)],
            (0x35, 0): [(0x25, 0), (0x39, 0), (0x2A, 1)],
            (0x39, 0): [(0x49, 0), (0x35, 0), (0x3A, 0)],
            (0x3A, 0): [(0x2A, 0), (0x25, 1), (0x39, 0)],
            (0x3E, 0): [(0x5F, 0), (0x5F, 0), (0x5F, 0)],
            (0x49, 0): [(0x39, 0), (0x34, 1), (0x3E, 1)],
        }
        _neighbour_lut = np.full((self.num_regions, 2, 3, 2), self.invalid_ugc, dtype=np.uint8)
        for key, neighbours in _lut.items():
            region_id, mode = key
            _neighbour_lut[region_id, mode] = neighbours
        return _neighbour_lut

    def c1(self, address, layer):
        """
        Given uri addresses(es) and a layer index (where 0 is the root)
        Return the c1 of that layer via its parent.
        """
        if 0 < layer < address.shape[1]:
            par = address[:, layer-1]
            poi = address[:, layer]
            return self.pqc1_lut[par, poi]
        raise IndexError(f'Index must be: 0 < {layer} < {address.shape[1]}')

    def terminate(self, addresses):
        """
        Normalise addresses termination.
        """
        last = addresses[:, -2]
        c1 = self.pqc1_lut[last, addresses[:, -1]]
        mode = self.ugc_lut[last, self.mode]
        addresses[:, -1] = self.child_lut[mode, c1, 2]
        return addresses

    def region_neighbours(self, addresses):
        """Vectorised means to return neighbouring half-hexagon addresses (as regions) via regions."""
        count, layers = addresses.shape
        neighbours = addresses.copy()           # A neighbour may just be a single switch.
        cascading = np.ones(count, dtype=bool)  # Track all the addresses we are managing.
        n_lut = self.neighbour_lut()
        c1 = self.pqc1_lut[addresses[:, -2], addresses[:, -1]]
        for poi in range(layers - 2, -1, -1):
            if not np.any(cascading):
                break
            active = np.where(cascading)[0]
            cur = addresses[:, poi][active]
            par = addresses[:, poi - 1][active]
            pmo = self.ugc_lut[par, self.mode]
            nbm = n_lut[cur, pmo, c1[active]]
            neighbours[:, poi][active] = nbm[:, -2]
            cascading[active] = (nbm[:, 1] != pmo)
        # Normalise terminal and root.
        nmo = self.ugc_lut[neighbours[:, 0], self.mode]
        root = np.where(nmo == 1, 0x16, 0x49)
        neighbours[:, 0] = root
        mode = self.ugc_lut[neighbours[:, -2], self.mode]
        neighbours[:, -1] = self.child_lut[mode, c1, 2]
        return c1, neighbours

    def clamp(self, xx, yy, mode):
        """
        Given arrays of points and modes, clamps them to be within their
        respective barycentric triangles.
        """
        # Create copies to store the final results
        xx_final = xx.copy()
        yy_final = yy.copy()

        ẋ = self.R3 * xx
        eps = 1e-14

        # --- 1. Calculate the Clamped Result for UP Mode Points ---
        up_mask = (mode == 1)
        if np.any(up_mask):
            # Filter to get only the points that are in UP mode
            y_up, ẋ_up = yy[up_mask], ẋ[up_mask]

            # Perform the full clamping logic for the UP case
            y_up_clamped = np.clip(y_up, self.ΛF, self.ΛC)
            max_abs_ẋ = self.ΛC - y_up_clamped
            at_apex = np.isclose(y_up_clamped, self.ΛC, atol=eps)
            max_abs_ẋ = np.where(at_apex, 0.0, max_abs_ẋ)

            ẋ_clamped = np.clip(ẋ_up, -max_abs_ẋ, max_abs_ẋ)

            at_base = np.isclose(y_up_clamped, self.ΛF, atol=eps)
            xc = ẋ_clamped / self.R3

            # Place the final clamped values back into the result arrays using the mask
            ẋ[up_mask] = ẋ_clamped
            xx_final[up_mask] = np.where(at_base, np.sign(xc) * self.TR, xc)
            yy_final[up_mask] = np.where(at_base, self.ΛF, y_up_clamped)

        # --- 2. Calculate the Clamped Result for DOWN Mode Points ---
        down_mask = (mode == 0)
        if np.any(down_mask):
            # Filter to get only the points that are in DOWN mode
            y_down, ẋ_down = yy[down_mask], ẋ[down_mask]

            # Perform the full clamping logic for the DOWN case
            y_down_clamped = np.clip(y_down, self.VF, self.VC)
            max_abs_ẋ = y_down_clamped - self.VF
            at_apex = np.isclose(y_down_clamped, self.VF, atol=eps)
            max_abs_ẋ = np.where(at_apex, 0.0, max_abs_ẋ)

            ẋ_clamped = np.clip(ẋ_down, -max_abs_ẋ, max_abs_ẋ)
            at_base = np.isclose(y_down_clamped, self.VC, atol=eps)
            xc = ẋ_clamped / self.R3

            # Place the final clamped values back into the result arrays
            xx_final[down_mask] = np.where(at_base, np.sign(xc) * self.TR, xc)
            yy_final[down_mask] = np.where(at_base, self.VC, y_down_clamped)
            ẋ[down_mask] = ẋ_clamped

        return xx_final, yy_final, ẋ
    # def clamp(self, xx, yy, mode):
    #     """
    #     Given an array of points, clamp them to be within the barycentric triangle.
    #     This should only be necessary when preparing points for projection to barycentre.
    #     """
    #     ẋ = self.R3 * xx
    #     eps = 1e-14  # A tolerance to detect if we're at a vertex
    #
    #     if mode == 1:  # Clamping for the UP triangle
    #         invalid = (yy < self.ΛF) | (yy > (self.ΛC - np.abs(ẋ)))
    #         if np.any(invalid):
    #             yy = np.clip(yy, self.ΛF, self.ΛC)
    #             max_abs_ẋ = self.ΛC - yy
    #             at_apex = np.isclose(yy, self.ΛC, atol=eps)
    #             max_abs_ẋ = np.where(at_apex, 0.0, max_abs_ẋ)
    #             ẋ = np.clip(ẋ, -max_abs_ẋ, max_abs_ẋ)
    #             at_base = np.isclose(yy, self.ΛF, atol=eps)
    #             xc = ẋ / self.R3
    #             xx = np.where(at_base, np.sign(xc) * self.TR, xc)
    #             yy = np.where(at_base, self.ΛF, yy)
    #     else:  # Clamping for the DOWN triangle
    #         invalid = (yy > self.VC) | (yy < (self.VF + np.abs(ẋ)))
    #         if np.any(invalid):
    #             yc = np.clip(yy, self.VF, self.VC)
    #             max_abs_ẋ = yc - self.VF
    #             at_apex = np.isclose(yy, self.VF, atol=eps)
    #             max_abs_ẋ = np.where(at_apex, 0.0, max_abs_ẋ)
    #             # max_abs_ẋ = np.where(max_abs_ẋ < eps, 0.0, max_abs_ẋ)
    #             ẋ = np.clip(ẋ, -max_abs_ẋ, max_abs_ẋ)
    #             at_base = np.isclose(yc, self.VC, atol=eps)  # Use yc
    #             xc = ẋ / self.R3
    #             xx = np.where(at_base, np.sign(xc) * self.TR, xc)
    #             yy = np.where(at_base, self.VC, yc)  # Use yc
    #     return xx, yy, ẋ

    def region_classification(self, ẋ, y):
        """
        # Grid Classification (Purely geometric) ---
        :param ẋ:       np_array of √3(x)
        :param y:       np_array of y
        :return:        np_array of region_id
        """
        # ẋ, y, self.ẇ ΛC, VC, ΛF, VF are defined
        h_conditions = [  # C0:=0; Horizontal – Flat
            y > self.ΛC,
            y > self.VC,
            y > 0,
            y > self.ΛF,
            y >= self.VF,  # allow floor to be included.
        ]
        h_id = np.select(h_conditions, [0, 1, 2, 3, 4], default=5)
        y_minus_x = y - ẋ
        p_conditions = [  # C0:=1; Positive Slope / Forward
            y_minus_x > self.Ẇ,
            y_minus_x > 0,
            y_minus_x >= -self.Ẇ,
        ]
        p_id = np.select(p_conditions, [0, 1, 2], default=3)
        y_plus_x = y + ẋ

        n_conditions = [  # C0:=2; Negative Slope \ Back
            y_plus_x < -self.Ẇ,
            y_plus_x < 0,
            y_plus_x <= self.Ẇ,
        ]
        n_id = np.select(n_conditions, [0, 1, 2], default=3)
        return h_id << 4 | p_id << 2 | n_id

    def unformat_addresses(self, text_addresses, mode):
        """
        Parses a single or a batch of text addresses into co-ordinates.
        we need to include the root mode!
        """
        addr, t_reg, t_hex = self.unpack_addresses(text_addresses)
        uris = self.ugc_inv(addr, mode, t_reg, t_hex)
        return self.ugc_dec(uris)

    def unpack_addresses(self, text_addresses):
        """
        Parses a single or a batch of text addresses into numerical arrays.

        Args:
            text_addresses (str or list or np.ndarray): A single addresses string or a
                                                        collection of them.
            depth (int): The expected length of the numerical digit part of the addresses.

        Returns:
            tuple: A tuple containing:
                - final_address (np.ndarray): (N, depth) array of hex digits.
                - terminating_region (np.ndarray): (N,) array of terminating region URIs.
                - terminating_hex (np.ndarray): (N,) array of terminating hex digits.
        """
        # 1. Coerce the input into a 1D NumPy array to handle both single and batch cases.
        addresses = np.atleast_1d(text_addresses)

        # 2. Vectorized String Slicing
        # Extract the numerical digit part of the strings.
        hex_digits_str = np.array([list(s[:-2]) for s in addresses])

        # Extract the two-character terminating context.
        terminating_context_str = np.array([(s[-2:-1], s[-1:]) for s in addresses])

        # 3. Vectorized Conversion to Numerical
        # Convert the array of digit characters to integers.
        final_address = hex_digits_str.astype(np.int8)

        # Look up the region character and hex digit from the terminating context.
        # This requires your pre-built CHAR_TO_REGION map.
        region_char = terminating_context_str[..., 0]
        hex_char = terminating_context_str[..., 1]

        # A list comprehension is a fast way to do the character-to-int lookup.
        terminating_region = np.array([self.ard[c] for c in region_char], dtype=np.int16)
        terminating_hex = hex_char.astype(np.int8)

        return final_address, terminating_region, terminating_hex

    def ugc_dec(self, uri_address):
        """
        REVERSE: URI addresses back into (x,y) coordinates and its
        initial mode.
        Inverse of ugc_regions
        """
        num_points, depth = uri_address.shape
        # Initialize x and y with the precise remainder from the encoding process.
        x = np.zeros(num_points, dtype=np.float64)
        y = np.zeros(num_points, dtype=np.float64)

        # Loop backwards from the last layer down to the first REAL layer (index 1),
        # skipping the placeholder root at index 0.
        for i in range(depth - 1, 0, -1):
            region_id = uri_address[:, i]
            valid_mask = (region_id != self.invalid_ugc)

            x /= 3.0
            y /= 3.0

            if np.any(valid_mask):
                valid_ids = region_id[valid_mask]
                off = self.ugc_off[valid_ids]
                x[valid_mask] += off[:, 0]
                y[valid_mask] += off[:, 1]

        # After reconstructing the coordinates, find the initial mode from the root URI.
        initial_mode = self.ugc_lut[uri_address[:, 0], self.mode]

        # Stack all three results into a final (N, 3) array.
        return np.stack([x, y, initial_mode], axis=-1)

    def ugc_regions(self, xy, mode, depth=36):
        """
        Given a vector of Point coords create a set of regions
        """
        num_points = xy.shape[0]
        x = np.copy(xy[:, 0])
        y = np.copy(xy[:, 1])
        addresses = np.full((num_points, depth + 2), self.invalid_ugc, dtype=np.uint8)
        addresses[:, 0] = np.where(mode == 1, 0x16, 0x49)  # These values should come from the octant set.
        for i in range(depth + 1):
            x, y, ẋ = self.clamp(x, y, mode)
            region = self.region_classification(ẋ, y)  # Raw classification
            props = self.ugc_lut[region]
            mode_up = props[:, self.in_up]
            mode_dn = props[:, self.in_dn]
            in_scope = np.where(mode == 1, mode_up, mode_dn)
            region_id = np.where(in_scope, region, self.invalid_ugc)  # Validated ID
            addresses[:, i + 1] = region_id
            off = self.ugc_off[region_id]
            mode = self.ugc_lut[region_id, self.mode]
            x -= off[:, 0]
            y -= off[:, 1]
            x *= 3.
            y *= 3.
        return self.terminate(addresses)

    def ugc_addr(self, xy, mode, depth=32):
        """Given cmp a vector of Point components create a set of addresses
            xy                  # shape (N) — barycentric coordinates (x, y)
            mode                # shape (N) — modes for each octant.
        """
        num_points = xy.shape[0]
        regions = self.ugc_regions(xy, mode, depth)
        addresses = np.zeros((num_points, depth), dtype=np.uint8)
        up_context_initial = np.array([0, 1, 2])  # Root context for UP mode
        dn_context_initial = np.array([0, 1, 2])  # Root context for DOWN mode
        context = np.where(mode[:, np.newaxis] == 1, up_context_initial, dn_context_initial)
        for i in range(depth):
            region_id = regions[:, i+1]
            layer_digit, context, mode = self.ugc_hex_context(region_id, context, mode)
            addresses[:, i] = layer_digit
        term_digit, _, _ = self.ugc_hex_context(regions[:, -1], context, mode)
        return addresses, regions[:, -2], term_digit

    def ugc_hex_context(self, region, context, mode):
        """
        Given a region with parent c1 values and mode,
        return the hex digit, updated context, and new mode
        """
        props = self.ugc_lut[region]
        c1 = np.where(mode == 1, props[:, self.u_ci], props[:, self.d_ci])
        hex_d = np.take_along_axis(context, c1[:, np.newaxis], axis=1).squeeze()
        u_hex = props[:, [self.uc0, self.uc1, self.uc2]]
        d_hex = props[:, [self.dc0, self.dc1, self.dc2]]
        context = np.where(mode[:, np.newaxis] == 1, u_hex, d_hex)
        mode = self.ugc_lut[region][:, self.mode]
        return hex_d, context, mode

    def ugc_inv(self, hex_digits, mode, c_reg, c_hex):
        """
        :param hex_digits:  an array of hex-digit arrays representing grid addresses.
        :param mode:  a vector of modes of each octant.
        :param c_reg:  the terminating child regions of the final digit.
        :param c_hex:  the terminating child digits from that c_reg.
        :return: unpacked arrays the representative regions, that can be converted back to co-ordinates.
        """
        num_points, depth = hex_digits.shape
        uri_address = np.zeros((num_points, depth + 1), dtype=np.uint8)
        uri_address[:, -1] = c_reg  # Seed the last position of the *actual* addresses
        uri_address[:, 0] = np.where(mode == 1, 0x16, 0x49)
        # c_hex = self.child_lut[mode, 1]
        # q_modes = self.ugc_lut[neighbours[:, -2], self.mode]
        # neighbours[:, -1] = self.child_lut[q_modes, c1, 1]
        # p_xp = self.ugc_rev[hex_digits[:, -1], c_hex, c_reg]
        # self.ugc_rev[int(p_hx), c_hx, c_reg]

        # 2. Loop backwards from the second-to-last position of the *actual* addresses
        for i in range(depth - 1, 0, -1):
            p_hex = hex_digits[:, i]

            # The definitive 3-part key lookup
            p_reg = self.ugc_rev[p_hex, c_hex, c_reg]

            # The corrected storage line
            uri_address[:, i] = p_reg

            # Update the state for the next backward step
            c_hex = p_hex
            c_reg = p_reg

        return uri_address

    def addr(self, pts, depth=32, calc_prefix=True):
        """Given a set of Points return their octant-addresses, hex-digits, terminal region, and hex"""
        # from hhg9 import Points
        dom = pts.domain
        oc, mode = pts.cm()
        addr, t_reg, term_hex = self.ugc_addr(pts.coords, mode, depth)
        term_reg = self.rgs[t_reg]
        if calc_prefix:
            oc1 = addr[:, 0]
            handlers = dom.handlers()  # shape (N,), dtype=object
            selected_handlers = handlers[oc]
            get_oc_element = np.frompyfunc(lambda obj, idx: obj.oc[idx], 2, 1)
            pfx = np.array(get_oc_element(selected_handlers, oc1))
            return pfx, addr[:, 1:], term_reg, term_hex
        else:
            return oc, addr, term_reg, term_hex

    def neighbours(self, pts, depth=32):
        """Given a set of points, return their neighbours of a given depth as Points."""
        from hhg9 import Points
        dom = pts.domain
        oc, mode = pts.cm()
        regions = self.ugc_regions(pts.coords, mode, depth)
        c1, reg_neighbours = self.region_neighbours(regions)
        xym = self.ugc_dec(reg_neighbours)
        oob = xym[:, -1] != mode
        nbo = dom.oid_nb[oc[oob], c1[oob]]
        oc[oob] = nbo
        return Points(xym[:, :2], dom, oc)

    def _poly_luts(self):
        """
        Return the half-hex/hexagon coordinates of c1 for the triangle.
        :return: the half-hex coordinates of c1 for the triangle
        """
        u, v = self.U, self.H / 3.
        pts = {
            # Clockwise. 5th pt is half-way along the long part.
            (0, 1): [
                [(-1, -1), (0, 0), (2, 0), (3, -1), (1, -1)],
                [(-1, 1), (0, 0), (-1, -1), (-3, -1), (-2, -0)],
                [(2, 0), (0, 0), (-1, 1), (0, 2), (1, 1)]
            ],
            (0, 0): [
                [(3, 1), (2, 0), (0, 0), (-1, 1), (1, 1)],
                [(0, -2), (-1, -1), (0, 0), (2, 0), (1, -1)],
                [(-3, 1), (-1, 1), (0, 0), (-1, -1), (-2, 0)]
            ],
            (1, 1): [
                [(-1, -1), (0, 0), (2, 0), (3, -1), (2, -2), (0, -2)],
                [(-1, 1), (0, 0), (-1, -1), (-3, -1), (-4, 0), (-3, 1)],
                [(2, 0), (0, 0), (-1, 1), (0, 2), (2, 2), (3, 1)]
            ],
            (1, 0): [
                [(3, 1), (2, 0), (0, 0), (-1, 1), (0, 2), (2, 2)],
                [(0, -2), (-1, -1), (0, 0), (2, 0), (3, -1), (2, -2)],
                [(-3, 1), (-1, 1), (0, 0), (-1, -1), (-3, -1), (-4, 0)]
            ]
        }
        uv = np.array([u, v])
        hh = np.zeros((2, 3, 5, 2), dtype=np.float64)
        hx = np.zeros((2, 3, 6, 2), dtype=np.float64)
        for (kind, mode), c1s in pts.items():
            for c1, poly in enumerate(c1s):
                if kind == 0:
                    hh[mode, c1] = poly * uv
                elif kind == 1:
                    hx[mode, c1] = poly * uv
        return hh, hx

    def enmesh(self, addresses):
        """
        Given a numpy array of URI regions, returns a dictionary of the unique
        nested half-hexagon polygons.
        """
        # Use a dictionary to store unique polygons, keyed by their addresses tuple
        unique_polygons = {}
        polys = []
        num_points, depth = addresses.shape
        for i in range(num_points):
            address_path = addresses[i]
            parent_xy = np.array([0.0, 0.0])
            scale = 1.0
            for j in range(1, depth):
                parent_uri = address_path[j - 1]
                child_uri = address_path[j]
                if child_uri == self.invalid_ugc:
                    break  # Stop processing this path if it becomes invalid
                current_path_key = tuple(address_path[:j + 1])
                if current_path_key not in unique_polygons:
                    parent_mode = self.ugc_lut[parent_uri, self.mode]
                    child_c1 = self.pqc1_lut[parent_uri, child_uri]
                    hh_shape = self.poly_hh[parent_mode, child_c1]
                    polygon = parent_xy[np.newaxis, :] + hh_shape * scale
                    idx = len(polys)
                    polys.append(polygon)
                    unique_polygons[current_path_key] = idx
                parent_xy += self.ugc_off[child_uri] * scale
                scale /= 3.0
        return unique_polygons, np.array(polys)

    @classmethod
    def valid(cls, pts, mode='Λ'):
        """
        syntactic sugar for in_scope()
        :param pts: ndarray of x,y co-ordinates.
        :param mode: triangle pointing up/down
        """
        # ẋ, y = cls.R3 * pts[..., 0], pts[..., 1]
        return cls.in_scope(cls.R3 * pts[..., 0], pts[..., 1], mode)

    # @classmethod
    # def xy_to_h9(cls, pt_i, c2t_i='021'):
    #     """
    #     Within the scope of a c2 triangle, 'c2t' identify the c1 and remaining components.
    #     :param pt_i: 2d coordinate.
    #     :param c2t_i: c2 triangle input
    #     :return: the hex (0...8), the remaining point,and mode/c2 container for the remaining point.
    #     """
    #     ud = 'Λ' if c2t_i in {'021', '102', '210'} else 'V'  # up-triangle/down-triangle
    #     x, y = pt_i[0], pt_i[1]  # This is a point on the plane
    #     ẋ = cls.R3 * x  # We will be using √3x for everything.
    #     if not cls.in_scope(ẋ, y, ud):  # Ensure we are in the equilateral
    #         return None
    #     c1 = cls.get_c1(ẋ, y, ud)  # Identify the c1 lo trit (036 / 147 / 258)
    #     c2 = int(c2t_i[c1])  # Identify the c2 hi trit (012 / 345 / 678)
    #     hx = c2 * 3 + c1  # Fundamental Enumeration: c2*3+c1
    #     c2t_o = cls.get_c2(ẋ, y, c1, ud)  # c1, ẋ, y to identify the next c2t (one of three triangles)
    #     xo, yo = cls.OFS[c1, ud, c2t_o]  # using c1, c2t_o we can find the offset of the next triangle.
    #     pt_o = 3. * (x + xo), 3. * (y + yo)  # update the new coordinates by the offset.
    #     return hx, ud, pt_o, c2t_o  # return the values.

    # @classmethod
    # def _code_pt(cls, style, hx, mode, c1) -> str:
    #     match style:
    #         case Style.U64:
    #             return hx
    #         case Style.HEX:
    #             return f'{hx}'
    #         case Style.NUMERIC:
    #             return f'{hx}'
    #         case Style.FULL:
    #             return f'{hx}{mode}'
    #         case Style.CFULL:
    #             return f'{hx}{c1}{mode}'
    #         case Style.EXTENDED:
    #             ex = {6: 'G', 7: 'T', 8: 'X'}
    #             if mode == 'V' or hx < 6:
    #                 return f'{hx}'
    #             else:
    #                 return f'{ex[hx]}'
    #         case Style.HALFHEX:
    #             fx = {
    #                 0: 'o', 1: 'i', 2: 'z',
    #                 3: 'e', 4: 'a', 5: 's',
    #                 6: 'g', 7: 't', 8: 'x'
    #             }
    #             if mode == 'V':
    #                 return f'{hx}'
    #             else:
    #                 return f'{fx[hx]}'

    # def encode(self, pt, loc='021', _depth=31, style=Style.HEX):
    #     """
    #     *Planar* Barycentric->H9 Encoder.
    #     Given a 2D coordinate and a c2 triangle (one of six), return its addresses.
    #     :param pt: 2d coordinate
    #     :param loc: 'Λ': ['021', '102', '210'], 'V': ['201', '120', '012']
    #     :param rot: rotation 0,1,2 (or None for planar).
    #     :param _depth: Put a limit to the encoding
    #     :param style: Style of encoding being asked for.
    #     :return: The encoded coordinate.
    #     """
    #     result = []
    #     ud = 'Λ' if loc in {'021', '102', '210'} else 'V'  # up-triangle/down-triangle
    #     for d in range(_depth):
    #         vals = self.xy_to_h9(pt, loc)
    #         if not vals:
    #             return None  # Probably a bug: outside triangle bounds.
    #         hx, ud, pt, loc = vals
    #         result.append(self._code_pt(style, hx, ud, hx & 3))
    #     if style == Style.HEX:
    #         result.append(f'{ud}')
    #     return ''.join(result)

    @classmethod
    def h9_to_xy(cls, ud, hx, ch, pt):
        """
        :param ud: ΛV of current environment.
        :param hx: current hex digit
        :param ch: c2 of current triangle.
        :param pt: existing 2d coordinate.
        :return: new c2, and revised 2d coordinate
        """
        x2, y2 = pt  # Extract x2, y2 from pt
        x2 /= 3.
        y2 /= 3.
        c1 = hx % 3
        c2 = {  # Determine c2 from hx
            #        0,     1,     2,     3,     4,     5,     6,     7,     8
            'Λ': ['021', '102', '210', '102', '210', '021', '210', '021', '102'],
            'V': ['012', '201', '120', '120', '012', '201', '201', '120', '012']
        }[ud][hx]
        xo, yo = cls.OFS[c1, ud, ch]  # Retrieve the offsets
        return c2, (x2 - xo, y2 - yo)

    @classmethod
    def decode(cls, addr):
        """
        This is the H9->Barycentric projection.
        Given an addresses string, return its xy coordinates.
        This is the loop part that drives h9_to_xy
        :param addr:
        :return: xy coordinate
        """
        c2i = {  # Determine c2 from hx
            'Λ': ['201', '120', '012'], 'V': ['210', '021', '102']
        }
        _hints = cls.hint(addr)
        pt = (0.0, 0.0)  # Start from the origin
        _addr, _ = cls.un_tail(addr)
        ch = c2i[_hints[-1]][int(_addr[-1]) % 3]
        for hx, ud in zip(reversed(_addr), reversed(_hints)):
            ch, pt = cls.h9_to_xy(ud, int(hx), ch, pt)  # Compute the previous `(x, y)` step
        return pt

    @classmethod
    def un_tail(cls, addr):
        """
        split ΛV from tail of addresses and return both.
        :param addr: Initial HEX format addresses with or without ΛV tail.
        :return: addresses without tail, and ΛV tail.
        """
        if addr[-1] in {'Λ', 'V'}:
            return addr[:-1], addr[-1]
        else:
            # The `VΛ` convention: Assume the final region is `V` if it is undefined.
            return addr, 'V'

    # @classmethod
    # def print_lut(cls):
    #     """
    #     This generates a list of rules used to understand how,
    #     given parent, child and child-UD the parent UD.
    #     :return: printout.
    #     """
    #     fn = (lambda a, b: (a - b) % 3)
    #     for n in range(9):
    #         print(f'?{n}')
    #         g, i = divmod(n, 3)  # g = 0/1/2 for 0..2/3..5/6..8
    #         for p in range(9):
    #             v = [f'{p}X{n}X', f'{p}V{n}X', f'{p}Λ{n}X', f'{p}Y{n}X']
    #             idx = fn((p % 3), i) if g != 1 else fn(i, (p % 3))
    #             rx = v[idx] if g != 2 else v[3] if idx == 0 else v[0]
    #             vl = cls.exp(p, n, 'Λ')
    #             vv = cls.exp(p, n, 'V')
    #             print(f'{p}?{n}X={rx}; {p}{vl}{n}Λ; {p}{vv}{n}V')

    @classmethod
    def exp(cls, par, chd, ud='V'):
        """
        Given a parent & child addresses and child mode, return the parent mode.
        # parental half-hex identity.
        # Eg (0,0,V) as in 00V => V as for (0V0V)
        # VΛ convention: V is default.
        # Hex addresses in base 3 is [C2C1]
        # C2 can be seen as distance from Centre (0,1,2)
        # C1 can be seen as orientation (flat/forward/back for 0,1,2 respectively).
        :param par:
        :param chd:
        :param ud: child mode, one of VΛ
        :return: parent mode
        """
        lut = [ud, 'V', 'Λ', ('Λ' if ud == 'V' else 'V')]
        c2, c1 = divmod(chd, 3)  # unravel base 3 values.
        p_c1 = par % 3  # we only use the c1 value of parent.
        if c2 != 1:
            idx = (p_c1 - c1) % 3
        else:
            idx = (c1 - p_c1) % 3
        return lut[idx] if c2 != 2 else lut[3] if idx == 0 else lut[0]

    @classmethod
    def hint(cls, addr, h=None):
        """
        Given a HEX addresses, return hint string.
        :param addr: eg 520826162014320318416260730241
        :param h: the trailing triangle identity (UD) for the least significant digit.
        :return: The full hint result.
        example
         520826162014320318416260730241
         VVVVVVΛΛVVVVΛVVVVVΛΛΛVVVVΛΛVΛΛ
        """
        result = []
        if h is None:
            addr, h = cls.un_tail(addr)
        chain = reversed(addr)  # VΛ convention: V is default.
        chd = None
        for char in chain:
            par = int(char)
            if chd is not None:
                h = cls.exp(par, chd, h)  # find h of par, given par, chd, h
            chd = par
            result.append(h)
        return ''.join(reversed(result))


if __name__ == '__main__':
    from hhg9 import Points, Registrar
    h9 = H9Engine()
    # various = np.array([
    #     [0.2785587592601327, 0.29386255474543355, 0],   # Stonehenge.
    #     [0.30413319554572815, 0.28972243397424297, 0],  # Greenwich Park West
    #     [0.30413319554572815, -0.28972243397424297, 1],  # Greenwich Park East
    #     [0.302986940221271, 0.2895875423442124, 1],
    #     [0.303014288964195, 0.2896217435874474, 0],
    #     [0.000000000000001, -0.8164965132120238, 0],
    #     [0.292437721410212, 0.2924377012342344, 0],
    #     [0.012345678901234, 0.1234567890123456, 0],
    # ])
