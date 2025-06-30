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
    The canonical octahedral address is prefixed with a three-character octahedral root hexagon and hint.
     (from which the side of the octahedron can be derived). This value replaces the first digit of the
     planar address; also the terminating hexagon rotation is also stored as a part of the suffix.
     This latter allows us to precisely recover a location from the address.
     For example, stonehenge (uk) is approximately at NW013502061182541V2, while the statue of liberty (usa)
     has the address NA556384535621324Λ0
     """
from dataclasses import dataclass
from enum import Enum, unique
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
    loc: str         # e.g. '850'
    x: float         # cumulative x so far
    y: float         # cumulative y so far
    style: Style = Style.HEX
    s: float = 1.0
    xa: float = 0.  # offset acc.
    ya: float = 0.  # offset acc.
    c1: int = None  # add this!
    tm: int = None  # terminating mode
    tr: int = None  # terminating rotation.


class H9Engine:
    """
    AKA H9 - This is a hierarchic hexagonal grid (HHG) that uses regular tetrahedrons
    """
    R3 = np.sqrt(3)
    W = np.sqrt(2)
    H = np.sqrt(6) / 2.
    RH = R3 / 2.     # ratio of height to width. 0.8660
    Ẇ = W * R3 / 3.  # g in grapher. w*√3/3
    ΛC, ΛF = 2 * H / 3., -H / 3.
    VC, VF = H / 3., -2. * H / 3.
    TL, TR = -W/2., W/2.
    U, V = W / 6., H / 9.
    OFS = {
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
        (2, 'V', '201'): ( U * 2., -V * 2.)
    }

    @classmethod
    def in_scope(cls, ẋ, y, ud='Λ') -> bool:
        """
        This is a barycentric scope test, for a unit equilateral triangle.
        This expects x to already be pre-calculated as √3(x).
        :param ẋ: `ẋ := √3(x)` on x co-ordinate.
        :param y: y co-ordinate
        :param ud: triangle pointing up/down
        :return: boolean (in scope or not)
        """
        e = 1e-15  # for polygons and edges, best to be a tiny bit safe.
        if ud == 'Λ':  # barycentre at 0, triangle point up.
            return (cls.ΛF <= y) & (y <= cls.ΛC - np.abs(ẋ) + e)
        else:  # barycentre at 0, triangle point down.
            return (cls.VF + np.abs(ẋ) - e <= y) & (y <= cls.VC)

    @classmethod
    def get_c1(cls, ẋ, y, ud='Λ'):
        """
        Given a point in a triangle, identify it's half-hex.
        The return value will be 0, 1, 2  for flat/forward/back.
        :param ẋ: `ẋ` is a synonym for `√3(x)` of the x coordinate.
        :param y: y coordinate
        :param ud: triangle pointing up/down
        :return: 0,1,2 representing the c1 component of the container triangle
        """
        if ud == 'Λ':
            if 0 >= y < ẋ:
                return 0  # flat
            elif ẋ <= y < -ẋ:
                return 1  # forward
            else:  # 0 <= y > -ẋ
                return 2  # back
        else:
            if 0 <= y > -ẋ:
                return 0  # flat
            elif 0 > y <= ẋ:
                return 1  # forward.
            else:  # ẋ < y <= -ẋ
                return 2  # back.

    @classmethod
    def get_c2(cls, ẋ, y, c1, ud='Λ'):
        """
        Given a point in a half-hex, identify its triangle
        get_c2 works on a half-hexagon, and establishes which of 3
        triangles a point is in, returning a 3-digit for that triangle.
        For example, our current planar value is 0Λ. (from 021)
        The c1 is 0, and the point we are looking
        at will be in either 507, 165, or 813 (021, 201, 102).
        The same calculation works for 6Λ and 3Λ.
        For octahedral (non-planar), we may use an equivalent c1.
        For example, in r0, the planar 7Λ is now 6Λ, but it's still c1=1 (Fwd).
        We use oc_1 to calculate the c1, and pass it here.
        :param c1: (0,1,2) represents the orientation: flat, forward, back.
               In planar, hexes 0,3,6 are Flat (0), 1,4,7 are Fwd (1), 2,5,8 are Back (2).
        :param ud: [Λ,V] tells us which side of the half-hexagon we are talking about.
               This will be Λ for any half-hexagon with two apexes Λ and one V in the middle.
        'Λ' and 'V' tells us which of the two half-hexes we are examining.
        # c1, ẋ, y to identify the next c2
        # return c2 is not determined by the input mode! It will be any one of the six available.
        :param ẋ: `ẋ` is a synonym for `√3(x)` of the x coordinate.
        :param y: y coordinate
        :return: Will be one of ['201', '120', '012', '210', '021', '102']
        """
        if ud == 'Λ':
            if c1 == 0:
                if y <= -ẋ:  # alt y <= -ẋ
                    return '021'  # y <= -ẋ identifies 021
                if y <= ẋ - cls.Ẇ:  # alt y <= ẋ - cls.Ẇ
                    return '102'  # y <= ẋ-ẇ identifies 102
                return '201'  # y < -ẋ and y > ẋ - ẇ identifies 201
            if c1 == 1:
                if y >= 0:
                    return '102'  # y >= 0 identifies 102
                if y <= -ẋ - cls.Ẇ:  #
                    return '210'  # y ≤ -ẋ-ẇ identifies 210
                return '120'  # y < 0 and `y > -ẋ-ẇ` identifies 120
            # c1 == 2
            if y <= ẋ:  # alt y <= ẋ
                return '210'  # `y <= ẋ identifies 210
            if y >= cls.H / 3.:
                return '021'  # y >= h/3 identifies 021
            return '012'  # y > ẋ and y < h/3 identifies 012
        else:  # Now for 'V'
            # For points in 0V (flat),
            # y >= ẋ identifies 012
            # y >= ẇ-ẋ identifies 120
            # y < ẋ and y < ẇ-ẋ identifies `210`
            if c1 == 0:
                if y >= ẋ:  # alt y >= ẋ
                    return '012'  # y >= ẋ identifies 012
                if y >= cls.Ẇ - ẋ:  # alt y >= cls.Ẇ - ẋ
                    return '120'  # y >= ẇ-ẋ identifies 120
                return '210'  # y < ẋ and y < ẇ-ẋ identifies 210
            # For points in `1V` (forward),
            # y >= -ẋ identifies 201
            # y <= -h/3 identifies 012
            # y > -h/3 and y < -ẋ identifies 021
            if c1 == 1:
                if y >= -ẋ:  # alt y >= -ẋ
                    return '201'  # y >= -ẋ identifies 201
                if y <= -cls.VC:
                    return '012'  # y <= -h/3 identifies 012
                return '021'  # y > -h/3 and y < -ẋ identifies 021
            # For points in `2V` (back),
            # y <= 0 identifies 120
            # y >= ẇ+ẋ identifies 201
            # y > 0 and y < ẇ+ẋ identifies 102
            if y <= 0:
                return '120'  # y <= 0 identifies 120
            if y >= cls.Ẇ + ẋ:  # alt cls.Ẇ + ẋ:
                return '201'  # y >= ẇ+ẋ identifies 201
            return '102'  # y > 0 and y < ẇ+ẋ identifies 102

    def __init__(self):
        # These are used to define octahedral enumerations.
        self.uro = {  # (v 0626.2) given a triangle, identify the rotation and mode.
            '085': (0, 'Λ'), '850': (1, 'Λ'), '508': (2, 'Λ'),
            '316': (0, 'Λ'), '163': (1, 'Λ'), '631': (2, 'Λ'),
            '742': (0, 'Λ'), '427': (1, 'Λ'), '274': (2, 'Λ'),
            '047': (0, 'V'), '470': (1, 'V'), '704': (2, 'V'),
            '815': (0, 'V'), '158': (1, 'V'), '581': (2, 'V'),
            '362': (0, 'V'), '623': (1, 'V'), '236': (2, 'V'),
        }
        self.ocm = {
            # Given a triangle, identify the rotation of the hex digit (via index) and it's mode.
            '085': ('012012201', 'Λ'),
            '850': ('120120012', 'Λ'),
            '508': ('201201120', 'Λ'),
            '316': ('012012201', 'Λ'),
            '163': ('120120012', 'Λ'),
            '631': ('201201120', 'Λ'),
            '742': ('012012120', 'V'),
            '427': ('120120201', 'V'),
            '274': ('201201012', 'V'),
            '047': ('012012120', 'V'),
            '470': ('120120201', 'V'),
            '704': ('201201012', 'V'),
            '815': ('012012201', 'Λ'),
            '158': ('120120012', 'Λ'),
            '581': ('201201120', 'Λ'),
            '362': ('012012120', 'V'),
            '623': ('120120201', 'V'),
            '236': ('201201012', 'V'),
        }
        self.o2p = {}
        self.p2o = {}
        self.oc2 = {}
        self.next_oc2 = {}
        self._define_luts()
        self.modev = {
            'Λ': [
                (0, '021'), (0, '201'), (0, '102'),
                (1, '102'), (1, '120'), (1, '210'),
                (2, '210'), (2, '012'), (2, '021')],
            'V': [
                (0, '012'), (0, '210'), (0, '120'),
                (1, '201'), (1, '021'), (1, '012'),
                (2, '120'), (2, '102'), (2, '201')]
        }
        self.o2m = {
            '021': 'Λ', '102': 'Λ', '210': 'Λ',
            '012': 'V', '120': 'V', '201': 'V',
        }

    def _define_luts(self):
        """
        Given the class dictionary of octahedral mappings `uro`
        set the octahedral tri to planar tri table and its inverse.
        """
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
        # The following could be
        # derived - it's just hacked in for the moment.
        self.cc2_pc2 = {
            # ('085', '815', '316')
            ('0', '085'): '085',
            ('1', '085'): '163',
            ('2', '085'): '274',
            ('3', '085'): '316',
            ('4', '085'): '427',
            ('5', '085'): '508',
            ('6', '085'): '631',
            ('7', '085'): '742',
            ('8', '085'): '850',

            ('0', '815'): '085',
            ('1', '815'): '163',
            ('2', '815'): '274',
            ('3', '815'): '316',
            ('4', '815'): '427',
            ('5', '815'): '508',
            ('6', '815'): '631',
            ('7', '815'): '742',
            ('8', '815'): '850',

            ('0', '316'): '085',
            ('1', '316'): '163',
            ('2', '316'): '274',
            ('3', '316'): '316',
            ('4', '316'): '427',
            ('5', '316'): '508',
            ('6', '316'): '631',
            ('7', '316'): '742',
            ('8', '316'): '850',

            # ('508', '581', '631')
            ('0', '508'): '508',
            ('1', '508'): '316',
            ('2', '508'): '427',
            ('3', '508'): '631',
            ('4', '508'): '742',
            ('5', '508'): '850',
            ('6', '508'): '163',
            ('7', '508'): '274',
            ('8', '508'): '085',

            ('0', '581'): '508',
            ('1', '581'): '316',
            ('2', '581'): '427',
            ('3', '581'): '631',
            ('4', '581'): '742',
            ('5', '581'): '850',
            ('6', '581'): '163',
            ('7', '581'): '274',
            ('8', '581'): '085',

            ('0', '631'): '508',
            ('1', '631'): '316',
            ('2', '631'): '427',
            ('3', '631'): '631',
            ('4', '631'): '742',
            ('5', '631'): '850',
            ('6', '631'): '163',
            ('7', '631'): '274',
            ('8', '631'): '085',

            # ('850', '158', '163')
            ('0', '850'): '850',
            ('1', '850'): '631',
            ('2', '850'): '742',
            ('3', '850'): '163',
            ('4', '850'): '274',
            ('5', '850'): '085',
            ('6', '850'): '316',
            ('7', '850'): '427',
            ('8', '850'): '508',

            ('0', '158'): '850',
            ('1', '158'): '631',
            ('2', '158'): '742',
            ('3', '158'): '163',
            ('4', '158'): '274',
            ('5', '158'): '085',
            ('6', '158'): '316',
            ('7', '158'): '427',
            ('8', '158'): '508',

            ('0', '163'): '850',
            ('1', '163'): '631',
            ('2', '163'): '742',
            ('3', '163'): '163',
            ('4', '163'): '274',
            ('5', '163'): '085',
            ('6', '163'): '316',
            ('7', '163'): '427',
            ('8', '163'): '508',

            # ('047', '742', '362')
            ('0', '047'): '047',
            ('1', '047'): '158',
            ('2', '047'): '236',
            ('3', '047'): '362',
            ('4', '047'): '470',
            ('5', '047'): '581',
            ('6', '047'): '623',
            ('7', '047'): '704',
            ('8', '047'): '815',

            ('0', '742'): '047',
            ('1', '742'): '158',
            ('2', '742'): '236',
            ('3', '742'): '362',
            ('4', '742'): '470',
            ('5', '742'): '581',
            ('6', '742'): '623',
            ('7', '742'): '704',
            ('8', '742'): '815',

            ('0', '362'): '047',
            ('1', '362'): '158',
            ('2', '362'): '236',
            ('3', '362'): '362',
            ('4', '362'): '470',
            ('5', '362'): '581',
            ('6', '362'): '623',
            ('7', '362'): '704',
            ('8', '362'): '815',

            # ('704', '274', '236')
            ('0', '704'): '704',
            ('1', '704'): '815',
            ('2', '704'): '623',
            ('3', '704'): '236',
            ('4', '704'): '047',
            ('5', '704'): '158',
            ('6', '704'): '362',
            ('7', '704'): '470',
            ('8', '704'): '581',

            ('0', '274'): '704',
            ('1', '274'): '815',
            ('2', '274'): '623',
            ('3', '274'): '236',
            ('4', '274'): '047',
            ('5', '274'): '158',
            ('6', '274'): '362',
            ('7', '274'): '470',
            ('8', '274'): '581',

            ('0', '236'): '704',
            ('1', '236'): '815',
            ('2', '236'): '623',
            ('3', '236'): '236',
            ('4', '236'): '047',
            ('5', '236'): '158',
            ('6', '236'): '362',
            ('7', '236'): '470',
            ('8', '236'): '581',

            # ('470', '427', '623')
            ('0', '470'): '470',
            ('1', '470'): '581',
            ('2', '470'): '362',
            ('3', '470'): '623',
            ('4', '470'): '704',
            ('5', '470'): '815',
            ('6', '470'): '236',
            ('7', '470'): '047',
            ('8', '470'): '158',

            ('0', '427'): '470',
            ('1', '427'): '581',
            ('2', '427'): '362',
            ('3', '427'): '623',
            ('4', '427'): '704',
            ('5', '427'): '815',
            ('6', '427'): '236',
            ('7', '427'): '047',
            ('8', '427'): '158',

            ('0', '623'): '470',
            ('1', '623'): '581',
            ('2', '623'): '362',
            ('3', '623'): '623',
            ('4', '623'): '704',
            ('5', '623'): '815',
            ('6', '623'): '236',
            ('7', '623'): '047',
            ('8', '623'): '158',
        }

    @classmethod
    def poly(cls, c1=0, mode='Λ', d3=False, hexagon=False) -> np.ndarray:
        """
        Return the half-hex/hexagon coordinates of c1 for the triangle.
        In accordance with Octahedron_Net dimensions.
        :param c1: the c1 required (0,1,2)
        :param mode: up/down
        :param d3: return 3d results.
        :param hexagon: return hexagon polygon.
        :return: the half-hex coordinates of c1 for the triangle
        """
        u, v = cls.U, cls.H / 3.
        pts = {
            # Clockwise. 5th pt is half-way along the long part.
            (False, 'Λ'): [
                [(-1, -1), (0, 0), (2, 0), (3, -1), (1, -1)],
                [(-1, 1), (0, 0), (-1, -1), (-3, -1), (-2, -0)],
                [(2, 0), (0, 0), (-1, 1), (0, 2), (1, 1)]
            ],
            (False, 'V'): [
                [(3, 1), (2, 0), (0, 0), (-1, 1), (1, 1)],
                [(0, -2), (-1, -1), (0, 0), (2, 0), (1, -1)],
                [(-3, 1), (-1, 1), (0, 0), (-1, -1), (-2, 0)]
            ],
            (True, 'Λ'): [
                [(-1, -1), (0, 0), (2, 0), (3, -1), (2, -2), (0, -2)],
                [(-1, 1), (0, 0), (-1, -1), (-3, -1), (-4, 0), (-3, 1)],
                [(2, 0), (0, 0), (-1, 1), (0, 2), (2, 2), (3, 1)]
            ],
            (True, 'V'): [
                [(3, 1), (2, 0), (0, 0), (-1, 1), (0, 2), (2, 2)],
                [(0, -2), (-1, -1), (0, 0), (2, 0), (3, -1), (2, -2)],
                [(-3, 1), (-1, 1), (0, 0), (-1, -1), (-3, -1), (-4, 0)]
            ]
        }
        if not d3:
            return np.array(pts[(hexagon, mode)][c1]) * [u, v]
        else:
            return np.array([[x * u, y * v, 0] for (x, y) in pts[(hexagon, mode)][c1]])

    def enmesh(self, pt, loc='021', _depth=31, single=False, hexagon=False) -> list:
        """
        Given a 2d coordinate and a c2 triangle (one of six), return hierarchy of polygons it belongs to.
        Alternatively just the one at the depth we want.
        :param pt:
        :param loc:
        :param _depth:
        :param single:
        :param hexagon: return hexagon(s) rather than half-hexagon(s).
        :return:
        """
        if loc in self.o2p:
            loc = self.o2p[loc][0]
        result = []
        mx = 1.
        xo, yo = 0., 0.
        for d in range(_depth):
            vals = self.xy_to_h9(pt, loc)
            if not vals:
                break
            hx, mode, pt, loc = vals
            c1 = hx % 3
            if single and d == _depth - 1:
                return (self.poly(c1, mode, False, hexagon) * mx) + [xo, yo]
            else:
                po = (self.poly(c1, mode, False, hexagon) * mx) + [xo, yo]
                result.append(po)
            xd, yd = self.OFS[c1, mode, loc]
            xo, yo = xo - xd * mx, yo - yd * mx
            mx /= 3.
        return np.array(result)

    def poly_step(self, step, hex=False):
        """Return polygon for a given step, correctly scaled and positioned."""
        rot, mode = self.uro[step.loc]
        c1 = self.get_c1(self.R3 * step.x, step.y, mode)
        return (self.poly(c1, mode, hexagon=hex) * step.s) + [step.xa, step.ya]  #

    @classmethod
    def valid(cls, pts, ud='Λ'):
        """
        syntactic sugar for in_scope()
        :param pts: ndarray of x,y co-ordinates.
        :param ud: triangle pointing up/down
        """
        # ẋ, y = cls.R3 * pts[..., 0], pts[..., 1]
        return cls.in_scope(cls.R3 * pts[..., 0], pts[..., 1], ud)

    @classmethod
    def xy_to_h9(cls, pt_i, c2t_i='021'):
        """
        Within the scope of a c2 triangle, 'c2t' identify the c1 and remaining components.
        :param pt_i: 2d coordinate.
        :param c2t_i: c2 triangle input
        :return: the hex (0...8), the remaining point,and mode/c2 container for the remaining point.
        """
        ud = 'Λ' if c2t_i in {'021', '102', '210'} else 'V'  # up-triangle/down-triangle
        x, y = pt_i[0], pt_i[1]  # This is a point on the plane
        ẋ = cls.R3 * x           # We will be using √3x for everything.
        if not cls.in_scope(ẋ, y, ud):  # Ensure we are in the equilateral
            return None
        c1 = cls.get_c1(ẋ, y, ud)  # Identify the c1 lo trit (036 / 147 / 258)
        c2 = int(c2t_i[c1])        # Identify the c2 hi trit (012 / 345 / 678)
        hx = c2 * 3 + c1           # Fundamental Enumeration: c2*3+c1
        c2t_o = cls.get_c2(ẋ, y, c1, ud)  # c1, ẋ, y to identify the next c2t (one of three triangles)
        xo, yo = cls.OFS[c1, ud, c2t_o]  # using c1, c2t_o we can find the offset of the next triangle.
        pt_o = 3. * (x + xo), 3. * (y + yo)  # update the new coordinates by the offset.
        return hx, ud, pt_o, c2t_o  # return the values.

    @classmethod
    def _code_pt(cls, style, hx, mode, c1) -> str:
        match style:
            case Style.U64:
                return hx
            case Style.HEX:
                return f'{hx}'
            case Style.NUMERIC:
                return f'{hx}'
            case Style.FULL:
                return f'{hx}{mode}'
            case Style.CFULL:
                return f'{hx}{c1}{mode}'
            case Style.EXTENDED:
                ex = {6: 'G', 7: 'T', 8: 'X'}
                if mode == 'V' or hx < 6:
                    return f'{hx}'
                else:
                    return f'{ex[hx]}'
            case Style.HALFHEX:
                fx = {
                    0: 'o', 1: 'i', 2: 'z',
                    3: 'e', 4: 'a', 5: 's',
                    6: 'g', 7: 't', 8: 'x'
                }
                if mode == 'V':
                    return f'{hx}'
                else:
                    return f'{fx[hx]}'

    def encode(self, pt, loc='021', _depth=31, style=Style.HEX):
        """
        *Planar* Barycentric->H9 Encoder.
        Given a 2D coordinate and a c2 triangle (one of six), return its address.
        :param pt: 2d coordinate
        :param loc: 'Λ': ['021', '102', '210'], 'V': ['201', '120', '012']
        :param rot: rotation 0,1,2 (or None for planar).
        :param _depth: Put a limit to the encoding
        :param style: Style of encoding being asked for.
        :return: The encoded coordinate.
        """
        result = []
        ud = 'Λ' if loc in {'021', '102', '210'} else 'V'  # up-triangle/down-triangle
        for d in range(_depth):
            vals = self.xy_to_h9(pt, loc)
            if not vals:
                return None  # Probably a bug: outside triangle bounds.
            hx, ud, pt, loc = vals
            result.append(self._code_pt(style, hx, ud, hx & 3))
        if style == Style.HEX:
            result.append(f'{ud}')
        return ''.join(result)

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
        Given an address string, return its xy coordinates.
        This is the loop part that drives h9_to_xy
        :param addr:
        :return: xy coordinate
        """
        c2i = {  # Determine c2 from hx
            'Λ': ['201', '120', '012'], 'V': ['210', '021', '102']
        }
        _hints = cls.hint(addr)
        pt = (0.0, 0.0)  # Start from the origin
        _addr, tail = cls.un_tail(addr)
        ch = c2i[_hints[-1]][int(_addr[-1]) % 3]
        for hx, ud in zip(reversed(_addr), reversed(_hints)):
            ch, pt = cls.h9_to_xy(ud, int(hx), ch, pt)  # Compute the previous `(x, y)` step
        return pt

    @classmethod
    def un_tail(cls, addr):
        """
        split ΛV from tail of address and return both.
        :param addr: Initial HEX format address with or without ΛV tail.
        :return: address without tail, and ΛV tail.
        """
        if addr[-1] in {'Λ', 'V'}:
            return addr[:-1], addr[-1]
        else:
            # The `VΛ` convention: Assume the final region is `V` if it is undefined.
            return addr, 'V'

    @classmethod
    def print_lut(cls):
        """
        This generates a list of rules used to understand how,
        given parent, child and child-UD the parent UD.
        :return: printout.
        """
        fn = (lambda a, b: (a - b) % 3)
        for n in range(9):
            print(f'?{n}')
            g, i = divmod(n, 3)  # g = 0/1/2 for 0..2/3..5/6..8
            for p in range(9):
                v = [f'{p}X{n}X', f'{p}V{n}X', f'{p}Λ{n}X', f'{p}Y{n}X']
                idx = fn((p % 3), i) if g != 1 else fn(i, (p % 3))
                rx = v[idx] if g != 2 else v[3] if idx == 0 else v[0]
                vl = cls.exp(p, n, 'Λ')
                vv = cls.exp(p, n, 'V')
                print(f'{p}?{n}X={rx}; {p}{vl}{n}Λ; {p}{vv}{n}V')

    @classmethod
    def exp(cls, par, chd, ud='V'):
        """
        Given a parent & child address and child mode, return the parent mode.
        # parental half-hex identity.
        # Eg (0,0,V) as in 00V => V as for (0V0V)
        # VΛ convention: V is default.
        # Hex address in base 3 is [C2C1]
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
        Given a HEX address, return hint string.
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

    def _oct_hint(self, path_str, mode, rot):
        """
        From a canonical octahedral address NWV...V2
        Recover Λ/V mode at each level from a hex path and final mode/rotation"""
        modes = [None] * len(path_str)
        modes[-1] = mode
        # Walk backwards
        for i in reversed(range(len(path_str) - 1)):
            hx = path_str[i + 1]
            c2 = self.oc2[(hx, rot, mode)]
            rct, mode = self.ocm[c2]   # rotation/mode contexts for c2.
            hx = int(path_str[i])  # This is the outer (contextual) hex digit.
            rot = int(rct[hx])  # The outer (contextual) rotation
            modes[i] = mode
        return modes

    def _oct_hint_n(self, path_str, root, i_mo, i_ro):
        """
        From a canonical octahedral address NWV...V2
        Recover c2 at each level from: hex path and terminal mode/rotation"""
        cs = [None] * len(path_str)
        cs[0] = (root.index(path_str[0]), root)  # The c1, and the c2, at root.
        # get the c2 for terminating hex.
        i_h = path_str[-1]
        i_c1 = [0, 2, 1][i_ro]  # I should check this, but it looks to be ok.
        i_c2 = self.oc2[(i_h, i_c1, i_mo)]
        cs[-1] = i_c2
        # Walk backwards through the address
        for i in reversed(range(len(path_str) - 1)):
            o_h = path_str[i]               # eg '1'
            i_c2 = self.cc2_pc2[o_h, i_c2]
            cs[i] = i_c2
        return cs

    def oct_decode_n(self, address, root='085'):
        """Recover (x, y) from a full hex path string and final Λ/V mode/rotation."""
        path_str, final_mode, final_rot = address[:-2], address[-2], int(address[-1])
        cs = self._oct_hint(path_str, root, final_mode, final_rot)
        ox, oy, bits = self.oct_decodeo(address, root)
        ddx = []
        s, x, y = 1.0, 0.0, 0.0
        o_c2 = root
        for i in range(len(path_str)-1):
            cx, i_c2 = path_str[i], cs[i]  # in
            o_c1 = i_c2.index(cx)
            opc2, o_r, o_m = self.o2p[o_c2]  # Planar equivalents of o_c2.
            ipc2, i_r, i_m = self.o2p[i_c2]  # Planar equivalents of i_c2.
            d_m = 'V' if o_m == 'Λ' else 'Λ'
            key = o_c1, d_m, ipc2
            # return self.OFS[(i_c1, 'V' if ud == 'Λ' else 'Λ', i_c2)]
            ddx.append((cx, i_c2, (opc2, o_r, o_m), (ipc2, i_r, i_m), *key, (x, y)))
            try:
                dx, dy = self.OFS[key]
                x = x - s*dx
                y = y - s*dy
                s /= 3.0
            except KeyError:
                print(f'key {key} not found in OFS')   # 2 ^ 102
            o_c2 = i_c2
        return x, y

    def oct_decode(self, address, root='085'):
        """Recover (x, y) from a full hex path string and final Λ/V mode/rotation."""
        path_str, final_mode, final_rot = address[:-2], address[-2], int(address[-1])
        modes = self._oct_hint(path_str, final_mode, final_rot)
        # kyx = []
        s, x, y = 1.0, 0.0, 0.0
        oc2 = root
        for i in range(len(path_str)-1):
            ro, mo = self.uro[oc2]  # '085' -> 0,Λ
            cx, cm, nx, nm = path_str[i], modes[i], path_str[i + 1], modes[i + 1]
            c1 = oc2.index(cx)  # '085'(5) = 2
            oc2 = self.next_oc2[(mo, c1, nx, nm)]
            pc2, pr, pm = self.o2p[oc2]  # Planar equivalents of current o_c2.
            key = (c1, mo, pc2)
            dx, dy = self.OFS[key]
            x = x - s*dx
            y = y - s*dy
            # kyx.append((*key, (mo, c1, nx, nm), oc2, (x, y)))
            s /= 3.0
        return x, y

    def branch_step(self, step):
        """Return full set of candidates from step"""
        result = []
        rot, mode = self.uro[step.loc]  # gather rot, half-hex mode of outer hex from triangle
        nxt_scale = step.s / 3.
        for c1, c2 in self.modev[mode]:
            dx, dy = self.OFS[c1, mode, c2]
            dx *= step.s
            dy *= step.s
            x = step.x - dx
            y = step.y - dy
            _rot = [0, 2, 1][c1]
            _mode = self.o2m[c2]
            o_c2 = self.p2o.get((c2, _rot, _mode))  # recover the new inner triangle.
            nxt = Step(o_c2, x, y, step.style)
            nxt.xa, nxt.ya = step.xa - dx, step.ya - dy
            nxt.s = nxt_scale
            result.append(nxt)
        return result

    def encode_step(self, step, last: bool = False):
        """Encode octahedral address, based upon step"""
        ẋ = self.R3 * step.x  # translate x.
        rot, mode = self.uro[step.loc]  # gather rot, half-hex mode of outer hex from triangle
        c1 = self.get_c1(ẋ, step.y, mode)  # Identify inner c1 orientated half-hex of o_c2.
        hx = step.loc[c1]  # get the hex number from index c1 of o_c2.
        tm, tr = None, None
        if step.style == Style.U64:
            result = int(hx)
            if last:
                tm = 1 if mode == 'V' else 0
                tr = rot
        else:
            result = self._code_pt(step.style, hx, mode, c1)
            if last:
                match step.style:
                    case Style.HEX:
                        result = f'{result}{mode}{rot}'

        c2 = self.get_c2(ẋ, step.y, c1, mode)  # Get the inner triangle (0,1,2) in c1.
        dx, dy = self.OFS[c1, mode, c2]  # This gives us the offset.
        x, y = 3. * (step.x + dx), 3. * (step.y + dy)  # Apply offset
        rot = [0, 2, 1][c1]  # get inner rot derived from the c1
        mode = 'Λ' if c2 in {'021', '102', '210'} else 'V'  # inner mode from the c2.
        o_c2 = self.p2o.get((c2, rot, mode))  # recover the new inner triangle.
        nxt = Step(o_c2, x, y, step.style)
        nxt.xa, nxt.ya = step.xa - dx * step.s, step.ya - dy * step.s
        nxt.s = step.s / 3.
        nxt.c1 = c1
        if step.style == Style.U64 and last:
            nxt.tm = tm
            nxt.tr = tr
        return nxt, result

    def oct_encode(self, pt, o_c2='085', depth=30, style=Style.HEX):
        """Octahedral encoding of a point in a triangle"""
        path = []
        suffix = ''
        step = Step(o_c2, pt[0], pt[1], style)
        for i in range(depth):
            step, code = self.encode_step(step, last=i == depth - 1)
            path.append(code)
        path.append(suffix)
        return ''.join(path)


if __name__ == '__main__':
    # Stonehenge: 0.29243772, 0.28113778 293.017 281.458
    h9 = H9Engine()
    shb = 0.29243772, 0.28113778
    sh_loc = '085'
    shf = h9.oct_encode(shb, sh_loc)
    shp = h9.oct_decode(shf, sh_loc)
    print(shb, shf, shp)

    # club = 0.303014288964195, 0.2896217435874474
    # loc = '085'
    # ldn_x, ldn_y = club
    # plx = h9.enmesh(club, loc, 3, False, False)  # This is the set of hexagons.
    # step = Step(loc, ldn_x, ldn_y)
    # for j in range(3):
    #     msh = plx[j]
    #     poly = h9.poly_step(step, False)
    #     print(poly - msh)
    #     step, x = h9.encode_step(step)
