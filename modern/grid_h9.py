from enum import Enum, unique
import numpy as np

# Encodings
# The initial encoding has a long/short structure.
# Easily transposed -
# 22035211610266553407865006553346V;
# 2Λ2Λ0Λ3Λ5V2V1Λ1Λ6Λ1V0V2V6V6Λ5V5V3V4V0Λ7Λ8Λ6Λ5Λ0V0V6Λ5V5V3V3V4V6V
# the former can be expanded to the latter, however it requires reverse
# calculations especially when dealing with 678.
# Therefore, there are two alternative encodings being considered;
# 'Extended': 678 for V and GTX for Λ. (`VΛ` convention).
# 'HalfHex':  V[abc/def/ghi]; Λ[ABC/DEF/GHI]
#             V[012/345/678]; Λ[oiz/eas/gtx]


@unique
class Style(Enum):
    HEX = 0
    FULL = 1
    EXTENDED = 2
    HALFHEX = 3


class GridH9:
    R3 = 3 ** 0.5
    W = 2 ** 0.5
    H = 6 ** 0.5 / 2.
    Ẇ = W * 3 ** 0.5 / 3.  # g in grapher. w*√3/3
    ΛC, ΛF = 2 * H / 3., -H / 3.
    VC, VF = H / 3., -2. * H / 3.
    U, V = W / 6., H / 9.
    CHIRAL = 1  # chirality: 1 or -1
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

    @classmethod
    def poly(cls, c1=0, ud='Λ', d3=False):
        u, v = cls.U, cls.H / 3.
        # Return the half-hex coordinates of c1 for the triangle.
        # This is in accordance with H9Grid dimensions.
        # This is probably correct only for CHIRAL=1
        # d3 = 3d results.
        pts = {
            'Λ': [
                [(-1, -1), (0, 0), (2, 0), (3, -1)],
                [(-1, 1), (0, 0), (-1, -1), (-3, -1)],
                [(2, 0), (0, 0), (-1, 1), (0, 2)]
            ],
            'V': [
                [(3, 1), (2, 0), (0, 0), (-1, 1)],
                [(0, -2), (-1, -1), (0, 0), (2, 0)],
                [(-3, 1), (-1, 1), (0, 0), (-1, -1)]
            ]
        }
        if not d3:
            return np.array(pts[ud][c1]) * [u, v]
        else:
            return np.array([[x * u, y * v, 0] for (x, y) in pts[ud][c1]])

    @classmethod
    def in_scope(cls, ẋ, y, ud='Λ') -> bool:
        if ud == 'Λ':  # `ẋ` is a synonym for `√3(x)`
            # barycentre at 0, triangle point up.
            return cls.ΛF <= y <= cls.ΛC - np.abs(ẋ)
        else:
            # barycentre at 0, triangle point down.
            return cls.VF + np.abs(ẋ) <= y <= cls.VC

    @classmethod
    def get_c1(cls, ẋ, y, ud='Λ'):
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
                else:  #  ẋ < y <= -ẋ
                    return 2  # back.

    @classmethod
    def get_c2(cls, ẋ, y, c1, ud='Λ'):
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
    def xy_to_h9(cls, pt, c2='021'):
        ud = 'Λ' if c2 in {'021', '102', '210'} else 'V'  # up-triangle/down-triangle
        x, y = pt  # This is a point on the plane
        ẋ = cls.R3 * x  # We will be using √3x for everything.
        if not cls.in_scope(ẋ, y, ud):  # Ensure we are in the equilateral
            return None
        c1 = cls.get_c1(ẋ, y, ud)  # Identify which of the three half-hexes we are in: 0,1,2
        hx = int(c2[c1]) * 3 + c1  # c2[c1]*3+c1 index determines the hex number.
        cc2 = cls.get_c2(ẋ, y, c1, ud)  # c1, ẋ, y to identify the next c2 (one of the three c1 triangles)
        xo, yo = cls.OFS[c1, ud, cc2]  # using c1, cc2 we can find the offset of the next triangle.
        x2, y2 = x + xo, y + yo  # update the new coordinates by the offset.
        return hx, ud, (3. * x2, 3. * y2), cc2  # return the values.

    @classmethod
    def encode(cls, pt, loc='021', _depth=32, style=Style.HEX):
        result = []
        ud = 'Λ' if loc in {'021', '102', '210'} else 'V'  # up-triangle/down-triangle
        for d in range(_depth):
            vals = cls.xy_to_h9(pt, loc)
            if not vals:
                return None
                # print('Encode error {pt}, {loc}, {"".join(result)}')
                # break
            # print(f'{vals}')
            hx, ud, pt, loc = vals
            match style:
                case Style.HEX:
                    result.append(f'{hx}')
                case Style.FULL:
                    result.append(f'{hx}{ud}')
                case Style.EXTENDED:
                    ex = {6: 'G', 7: 'T', 8: 'X'}
                    if ud == 'V' or hx < 6:
                        result.append(f'{hx}')
                    else:
                        result.append(f'{ex[hx]}')
                case Style.HALFHEX:
                    fx = {
                        0: 'o', 1: 'i', 2: 'z',
                        3: 'e', 4: 'a', 5: 's',
                        6: 'g', 7: 't', 8: 'x'
                    }
                    if ud == 'V':
                        result.append(f'{hx}')
                    else:
                        result.append(f'{fx[hx]}')
        if style == Style.HEX:
            result.append(f'{ud}')
        return ''.join(result)

    @classmethod
    def enmesh(cls, pt, loc='021', _depth=32, single=False):
        result = []
        ud = 'V'
        mx = 1.
        xo, yo = 0., 0.
        for d in range(_depth):
            vals = cls.xy_to_h9(pt, loc)
            if not vals:
                break
            hx, ud, pt, loc = vals
            c1 = hx % 3
            if single and d == _depth - 1:
                return (cls.poly(c1, ud) * mx) + [xo, yo]
            else:
                result.append((cls.poly(c1, ud) * mx) + [xo, yo])
            xd, yd = cls.OFS[c1, ud, loc]
            xo, yo = xo - xd * mx, yo - yd * mx
            mx /= 3.
        return result

    @classmethod
    def h9_to_xy(cls, ud, hx, ch, pt):
        x2, y2 = pt  # Extract x2, y2 from pt
        x2 /= 3.
        y2 /= 3.
        # 'Λ' hx will determine 0,1,2 only.
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
        c2i = {  # Determine c2 from hx
            #      036,   147,   258,          036,   147,   258,
            'Λ': ['201', '120', '012'], 'V': ['210', '021', '102']
        }
        _hints = cls.hint(addr)
        pt = (0.0, 0.0)  # Start from the origin
        # depth = len(_hints)  #
        _addr, tail = cls.un_tail(addr)
        ch = c2i[_hints[-1]][int(_addr[-1]) % 3]
        for hx, ud in zip(reversed(_addr), reversed(_hints)):
            ch, pt = cls.h9_to_xy(ud, int(hx), ch, pt)  # Compute the previous `(x, y)` step
        return pt

    @classmethod
    def un_tail(cls, addr):  # split ΛV from tail of address and return both,
        if addr[-1] in {'Λ', 'V'}:
            return addr[:-1], addr[-1]
        else:
            # The `VΛ` convention: Assume the final region is `V` if it is undefined.
            return addr, 'V'

    @classmethod
    def print_lut(cls):
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
    def exp(cls, par, chd, hh):
        # VΛ convention: V is default.
        # Hex address in base 3 is [C2C1]
        # C2 can be seen as distance from Centre (0,1,2)
        # C1 can be seen as orientation (–,/,\ for 0,1,2 respectively).
        # Given a parent/child address (with child half-hex VΛ) we calculate the
        # parental half-hex identity.
        # Eg 00V => 0V0V
        # a = 'V', b='Λ'
        lut = [hh, 'V', 'Λ', ('Λ' if hh == 'V' else 'V')]
        c2, c1 = divmod(chd, 3)  # unravel base 3 values.
        p_c1 = par % 3  # we only use the c1 value of parent.
        if c2 != 1:
            idx = (p_c1 - c1) % 3
        else:
            idx = (c1 - p_c1) % 3
        return lut[idx] if c2 != 2 else lut[3] if idx == 0 else lut[0]

    @classmethod
    def hint(cls, addr, h=None):
        # 520826162014320318416260730241
        # VVVVVVΛΛVVVVΛVVVVVΛΛΛVVVVΛΛVΛΛ
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
