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


@dataclass
class Step:
    """The full state of an encoding step"""
    loc: str         # e.g. '850'
    x: float         # cumulative x so far
    y: float         # cumulative y so far
    style: Style = Style.HEX
    s: float = 1.0


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
    CHIRAL = 1.0  # Chirality is currently fixed (and only tested) to 1.0
    OFS = {
        (0, 'Λ', '021'): (CHIRAL * 0, V * 2.),
        (0, 'Λ', '201'): (CHIRAL * -U, V),
        (0, 'Λ', '102'): (CHIRAL * -U * 2., V * 2.),
        (1, 'Λ', '102'): (CHIRAL * U, -V),
        (1, 'Λ', '120'): (CHIRAL * U, V),
        (1, 'Λ', '210'): (CHIRAL * U * 2., V * 2.),
        (2, 'Λ', '210'): (CHIRAL * -U, -V),
        (2, 'Λ', '012'): (CHIRAL * 0, -V * 2.),
        (2, 'Λ', '021'): (CHIRAL * 0, -V * 4.),
        (0, 'V', '012'): (CHIRAL * 0, -V * 2.),
        (0, 'V', '210'): (CHIRAL * -U, -V),
        (0, 'V', '120'): (CHIRAL * -U * 2., -V * 2.),
        (1, 'V', '201'): (CHIRAL * -U, V),
        (1, 'V', '021'): (CHIRAL * 0, V * 2.),
        (1, 'V', '012'): (CHIRAL * 0, V * 4.),
        (2, 'V', '120'): (CHIRAL * U, V),
        (2, 'V', '102'): (CHIRAL * U, -V),
        (2, 'V', '201'): (CHIRAL * U * 2., -V * 2.)
    }

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

    @classmethod
    def poly(cls, c1=0, mode='Λ', d3=False) -> np.ndarray:
        """
        Return the half-hex coordinates of c1 for the triangle.
        In accordance with Octahedron_Net dimensions.
        Probably correct only for CHIRAL=1
        :param c1: the c1 required (0,1,2)
        :param mode: up/down
        :param d3: return 3d results.
        :return: the half-hex coordinates of c1 for the triangle
        """
        u, v = cls.U, cls.H / 3.
        pts = {
            # Clockwise. 5th pt is half-way along the long part.
            'Λ': [
                [(-1, -1), (0, 0), (2, 0), (3, -1), (1, -1)],
                [(-1, 1), (0, 0), (-1, -1), (-3, -1), (-2, -0)],
                [(2, 0), (0, 0), (-1, 1), (0, 2), (1, 1)]
            ],
            'V': [
                [(3, 1), (2, 0), (0, 0), (-1, 1), (1, 1)],
                [(0, -2), (-1, -1), (0, 0), (2, 0), (1, -1)],
                [(-3, 1), (-1, 1), (0, 0), (-1, -1), (-2, 0)]
            ]
        }
        if not d3:
            return np.array(pts[mode][c1]) * [u, v]
        else:
            return np.array([[x * u, y * v, 0] for (x, y) in pts[mode][c1]])

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
    def valid(cls, pts, ud='Λ'):
        """
        syntactic sugar for in_scope()
        :param pts: ndarray of x,y co-ordinates.
        :param ud: triangle pointing up/down
        """
        # ẋ, y = cls.R3 * pts[..., 0], pts[..., 1]
        return cls.in_scope(cls.R3 * pts[..., 0], pts[..., 1], ud)

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
        if cls.CHIRAL < 0:  # lazy chiral check
            if ud == 'Λ':
                if 0 >= y < -ẋ:
                    return 0  # flat
                elif -ẋ <= y < ẋ:
                    return 1  # forward
                else:  # 0 <= y > ẋ
                    return 2  # back
            else:
                if 0 <= y > ẋ:
                    return 0  # flat
                elif 0 > y <= -ẋ:
                    return 1  # forward.
                else:  # -ẋ < y <= ẋ
                    return 2  # back.
        else:
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
        if cls.CHIRAL < 0:  # lazy chiral check
            if ud == 'Λ':
                if c1 == 0:
                    if y <= ẋ:  # alt y <= -ẋ
                        return '021'  # y <= -ẋ identifies 021
                    if y <= -ẋ - cls.Ẇ:  # alt y <= ẋ - cls.Ẇ
                        return '102'  # y <= ẋ-ẇ identifies 102
                    return '201'  # y < -ẋ and y > ẋ - ẇ identifies 201
                if c1 == 1:
                    if y >= 0:
                        return '102'  # y >= 0 identifies 102
                    if y <= ẋ - cls.Ẇ:  # alt y <= -ẋ - cls.Ẇ
                        return '210'  # y ≤ -ẋ-ẇ identifies 210
                    return '120'  # y < 0 and `y > -ẋ-ẇ` identifies 120
                # c1 == 2
                if y <= -ẋ:  # alt y <= ẋ
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
                    if y >= -ẋ:  # alt y >= ẋ
                        return '012'  # y >= ẋ identifies 012
                    if y >= cls.Ẇ + ẋ:  # alt y >= cls.Ẇ - ẋ
                        return '120'  # y >= ẇ-ẋ identifies 120
                    return '210'  # y < ẋ and y < ẇ-ẋ identifies 210
                # For points in `1V` (forward),
                # y >= -ẋ identifies 201
                # y <= -h/3 identifies 012
                # y > -h/3 and y < -ẋ identifies 021
                if c1 == 1:
                    if y >= ẋ:  # alt y >= -ẋ
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
                if y >= cls.Ẇ - ẋ:  # alt cls.Ẇ + ẋ:
                    return '201'  # y >= ẇ+ẋ identifies 201
                return '102'  # y > 0 and y < ẇ+ẋ identifies 102
        else:
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

    def enmesh(self, pt, loc='021', _depth=31, single=False) -> list:
        """
        Given a 2d coordinate and a c2 triangle (one of six), return hierarchy of polygons it belongs to.
        Alternatively just the one at the depth we want.
        :param pt:
        :param loc:
        :param _depth:
        :param single:
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
            hx, ud, pt, loc = vals
            c1 = hx % 3
            if single and d == _depth - 1:
                return (self.poly(c1, ud) * mx) + [xo, yo]
            else:
                po = (self.poly(c1, ud) * mx) + [xo, yo]
                # ff = cls.valid(po, mode)
                result.append(po)
            xd, yd = self.OFS[c1, ud, loc]
            xo, yo = xo - xd * mx, yo - yd * mx
            mx /= 3.
        return np.array(result)

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

    def oct_decode(self, address, root='085'):
        """Recover (x, y) from a full hex path string and final Λ/V mode/rotation."""
        path_str, final_mode, final_rot = address[:-2], address[-2], int(address[-1])
        modes = self._oct_hint(path_str, final_mode, final_rot)
        s, x, y = 1.0, 0.0, 0.0
        oc2 = root
        for i in range(len(path_str)-1):
            ro, mo = self.uro[oc2]  # '085' -> 0,Λ
            cx, cm, nx, nm = path_str[i], modes[i], path_str[i + 1], modes[i + 1]
            c1 = oc2.index(cx)  # '085'(5) = 2
            oc2 = self.next_oc2[(mo, c1, nx, nm)]
            pc2, pr, pm = self.o2p[oc2]  # Planar equivalents of current o_c2.
            dx, dy = self.OFS[(c1, mo, pc2)]
            x = x - s*dx
            y = y - s*dy
            s /= 3.0
        return x, y

    def branch_step(self, step):
        """Return full set of candidates from step"""
        result = []
        rot, mode = self.uro[step.loc]  # gather rot, half-hex mode of outer hex from triangle
        for c1 in [0, 1, 2]:
            for c2 in ['201', '120', '012', '210', '021', '102']:
                if (c1, mode, c2) in self.OFS:
                    dx, dy = self.OFS[c1, mode, c2]
                    x = step.x - step.s * dx
                    y = step.y - step.s * dy
                    _rot = [0, 2, 1][c1]
                    _mode = 'Λ' if c2 in {'021', '102', '210'} else 'V'  # inner mode from the c2.
                    o_c2 = self.p2o.get((c2, _rot, _mode))  # recover the new inner triangle.
                    result.append(Step(o_c2, x, y, step.style, step.s / 3))
        return result

    def encode_step(self, step, last: bool = False):
        """Encode octahedral address, based upon step"""
        ẋ = self.R3 * step.x  # translate x.
        rot, mode = self.uro[step.loc]  # gather rot, half-hex mode of outer hex from triangle
        c1 = self.get_c1(ẋ, step.y, mode)  # Identify inner c1 orientated half-hex of o_c2.
        hx = step.loc[c1]  # get the hex number from index c1 of o_c2.
        result = self._code_pt(step.style, hx, mode, c1)
        if step.style == Style.HEX and last:
            result = f'{result}{mode}{rot}'
        c2 = self.get_c2(ẋ, step.y, c1, mode)  # Get the inner triangle (0,1,2) in c1.
        dx, dy = self.OFS[c1, mode, c2]  # This gives us the offset.
        x, y = 3. * (step.x + dx), 3. * (step.y + dy)  # Apply offset
        rot = [0, 2, 1][c1]  # get inner rot derived from the c1
        mode = 'Λ' if c2 in {'021', '102', '210'} else 'V'  # inner mode from the c2.
        o_c2 = self.p2o.get((c2, rot, mode))  # recover the new inner triangle.
        return Step(o_c2, x, y, step.style), result

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

