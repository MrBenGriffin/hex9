import svg
import math
from itertools import batched
from collections.abc import Sequence
from drawing import Drawing
import matplotlib as mpl
import numpy as np
from hexutil import HexUtil

# h9a is frozen just for ease.

def translate(p, t):
    tx, ty = t
    #  p points, t is translate.
    ps = p if isinstance(p[0], Sequence) else list(batched(p, 2))
    return [(x+tx, y+ty) for x, y in ps]


def rotate(p, r):  # rotate around angle r.
    th = np.radians(r)
    pts = p if isinstance(p[0], Sequence) else batched(p, 2)
    return [(x * np.cos(th) - y * np.sin(th), x * np.sin(th) + y * np.cos(th)) for (x, y) in pts]


class H6:
    @classmethod
    def size_for(cls, hw, hh, rx) -> tuple:
        return math.ceil(hw * 3 * rx + .5*rx), math.ceil(hh * rx * 3.0 * cls.rt3)

    dx = [  # left to right, top to bottom.
        [0.0, 1., 1.], [0.0, 1., 4.], [1.5, 0., 0],    # 6, 7, 13
        [0.0, 3., 3.], [1.5, 2., 2.], [1.5, 2., 5.],   # 0, 1, 2
        [0.0, 3., 0.], [1.5, 4., 1.], [1.5, 4., 4.],   # 5, 4, 3
        [0.0, 5., 2.], [0.0, 5., 5.], [1.5, 6., 3],   # 11, 10, 16
    ]
    a: float       # This is the length of each side of the unit equilateral triangles of a district.
    h: float       # This is the height of a unit equilateral of a district (of which there are 3).
    id_ref: str    # The width of each H9 is 6a, the height is 6h.
    rt3 = math.sqrt(3.)   # the left/right edges have a width of 5a, and top/bottom: 3h
    stroke: float  # the pixel-width/height of H9 are 4.5a / 3h

    def __init__(self, owner: Drawing | None = None, identity: str = 'h6', size: float = 81., stroke: float = 0.5, op=0.9):
        self.owner = owner
        self.opacity = op
        self.id_ref = f'#{identity}'
        self.stroke = stroke
        self.tb, self.lr = [], []
        self.a = float(size)
        self.h = self.a * 0.5 * self.rt3  # height of any equilateral triangle.
        self.a2 = 2. * self.a
        self.a_2 = 0.5 * self.a
        self.scx, self.scy = self.a*3, self.h*6

        # return the district polygons for radius rad.
        a, h = self.a, self.h  # height of any equilateral triangle.
        a_2, a = self.a_2, self.a
        self.pts = [-a, 0., +a, 0., a_2,  h, -a_2, h]
        # self.pts = [0., 0., a2, 0., a_2+a, h, a_2, h]
        self.districts = [translate(rotate(self.pts, rt[2]*60.), (a * rt[0], h * rt[1])) for rt in self.dx]
        self.tx = [translate([0, 0], (a * rt[0], h * (rt[1]-3)))[0] for rt in self.dx]
        self.bboxes = self._bboxes()
        if self.owner is not None:
            self.owner.define(self._sym(identity))
            self.set_offset(self.owner.width, self.owner.height)

        # draw grid.
        # for i in range(12):
        #     for j in range(7):
        #         sqp = [-0.2, -0.2,  0.2, -0.2, 0.2, 0.2, -0.2, 0.2]
        #         t = svg.Translate(j*a, i*h)
        #         px = svg.Polygon(stroke_width=self.stroke, fill='black', fill_opacity=self.opacity, points=sqp, transform=[t])
        #         self.owner.add(px)

        # show bb
        # for i in range(len(self.bboxes)):
        #     t, l, b, r = self.bboxes[i]
        #     px = svg.Polygon(stroke_width=self.stroke, fill_opacity=0.3, points=[t, l, t, r, b, r, b, l])
        #     self.owner.add(px)

    def _sym(self, idx: str):
        pts = [p for xy in self.districts[6] for p in xy]
        return svg.Polygon(id=idx, stroke_width=self.stroke, fill_opacity=self.opacity, points=pts)

    def _bboxes(self):  # set bounding boxes.
        ll, rl, tl, bl = [], [], [], []
        for r in self.districts:
            xs, ys = list(zip(*r))
            ll.append(min(xs))
            rl.append(max(xs))
            tl.append(min(ys))
            bl.append(max(ys))
        return list(zip(ll, tl, rl, bl))

    def set_offset(self, w, h):
        self.scx = w
        self.scy = h
        x = math.ceil(w / (self.a * 3))
        y = math.ceil(h / (self.h * 6))
        self.tb, self.lr = [0, y], [0, x]

    def wxy(self, where) -> tuple:
        x, y = where
        return x * self.a * 3 + self.a, y * self.h * 6

    def points(self) -> list:
        return self.pts

    def set_limits(self, tb, lr):
        self.tb, self.lr = tb, lr

    def get_limits(self) -> tuple:
        x0, x1 = self.lr
        y0, y1 = self.tb
        return range(x0, x1), range(y0, y1), (0, 0)

    def may_place(self, where, district: tuple) -> bool:  # where is in wc - use wxy before if using hex_coords
        x, y = district
        dx = y * 3 + x
        (l, t), (r, b) = translate(self.bboxes[dx], where)
        return -3. <= l and r <= self.scx+3. and -3. <= t and b <= self.scy+3.

    def place_district(self, where, district: tuple, color: str = 'black'):
        if self.may_place(where, district):
            x, y = district
            dx = y * 3 + x
            if self.owner is not None:
                px, py = where
                tx, ty = self.tx[dx]
                rot = self.dx[dx][2]
                r = svg.Rotate(rot * 60., 0, +self.h*3)
                t = svg.Translate(tx+px, ty+py)
                inst = svg.Use(href=self.id_ref, fill=color, transform=[t, r])
                self.owner.add(inst)
            else:
                return translate(self.districts[dx], where)

    def place(self, where, colors):
        wc = self.wxy(where)
        for dy in range(4):
            for dx in range(3):
                self.place_district(wc, (dx, dy), colors[dy*3+dx])


class H9:
    master = 0
    ax_dof = {
        (2, 2, 3): (3, 0, 1),
        (2, 1, 1): (7, 0, 1),
        (2, 1, 2): (7, 1, 0),
        (2, 0, 1): (8, 1, 0),
        (2, 0, 2): (8, 0, 0),
        (2, -1, 3): (3, 0, 0),
        (2, -2, 1): (7, 0, 0),
        (2, -2, 2): (7, 1, 1),

        (1, 2, 3): (2, 0, 1),
        (1, 1, 2): (6, 0, 1),
        (1, 1, 1): (6, 0, 0),
        (1, 0, 3): (1, 0, 0),
        (1, -1, 3): (2, 0, 0),
        (1, -2, 2): (6, 0, 0),
        (1, -2, 1): (6, 0, 1),

        (0, 2, 3): (4, 0, 1),
        (0, 1, 3): (5, 0, 0),
        (0, 0, 3): (0, 0, 0),
        (0, -1, 3): (4, 0, 0),
        (0, -2, 3): (5, 0, 1),

        (-1, 2, 3): (3, 1, 1),
        (-1, 1, 1): (7, 1, 1),
        (-1, 1, 2): (7, 0, 0),
        (-1, 0, 1): (8, 0, 0),
        (-1, 0, 2): (8, 1, 0),
        (-1, -1, 3): (3, 1, 0),
        (-1, -2, 1): (7, 1, 0),
        (-1, -2, 2): (7, 0, 1),

        (-2, 2, 3): (2, 1, 1),
        (-2, 1, 2): (6, 1, 1),
        (-2, 1, 1): (6, 1, 0),
        (-2, 0, 3): (1, 1, 0),
        (-2, -1, 3): (2, 1, 0),
        (-2, -2, 2): (6, 1, 0),
        (-2, -2, 1): (6, 1, 1)
    }
    dx = [  # clockwise, starting \–/
        [-1., 0., 0], [-1., 0., 3.], [0.5, -1.0, 2.], [0.5, -1.0, 5.],  # 0a,0b,1a,1b,
        [0.5, 1., 4.], [0.5, 1., 1.], [2., 0., 0], [2., 0., 3],    # 2a,2b,3a,3b
        [-1., 2., 2], [-1., 2., 5],  [-1., -2, 4], [-1., -2., 1],   # 4a,4b,5a,5b
        [0.5, -3, 0], [0.5, 3., 3], [2., 2., 2], [-2.5, -1.0, 5.],
        [-2.5, 1., 4], [2., -2., 1]
    ]
    lb = ['0a', '0b', '1a', '1b',
          '2a', '2b', '3a', '3b',
          '4a', '4b', '5a', '5b',
          '6a', '6b', '7a', '7b',
          '8a', '8b'
          ]
    lbh = ['0', '0', '1', '1', '2', '2', '3', '3', '4', '4', '5', '5', '6', '6', '7', '7', '8', '8']
    lpt = [
        [-0.85,  0.20], [-0.85, -0.05], [-0.50, -0.80], [-0.36, 0.95],
        [-0.36, -0.80], [-0.50, 0.95], [-0.85, 0.20], [-0.85, -0.05],
        [-0.50, -0.80], [-0.36, 0.95], [-0.36, -0.80], [-0.50, 0.95],
        [-0.85,  0.20], [-0.85, -0.05], [-0.50, -0.80], [-0.36, 0.95],
        [-0.36, -0.80], [-0.50, 0.95]
    ]
    xg = {
        (0, +0, +0): 0,
        (1, +0, -1): 1,
        (1, -1, +0): 2,
        (0, +0, +0): 3,
        (1, +0, -1): 4,
        (1, -1, +0): 5,
        (1, +0, -1): 6,
        (1, -1, +0): 7,
        (1, -1, +0): 8,
    }
    d_ax = [
        (0, 0), (0, 0), (1, 0), (1, 0), (1, -1), (1, -1),
        (2, -1), (2, -1), (0, -1), (0, -1), (0, 1), (0, 1),
        (1, 1), (1, -2),  (2, -2), (-1, 1), (-1, 0), (2, 0)
    ]
    ax_l = {

        (0, 0): ((0, 0), 0),  # 7b
        (0, 1): ((0, 0), 1),
        (0, 2): ((0, 0), 8),

        (0, 1): ((0, 0), 5),
        (1, 1): ((0, 0), 6),
        (2, 1): ((0, 0), 7),

        (2, 0): ((0, 0), 6),
        (2, 1): ((0, 0), 2),
        (2, 2): ((0, 0), 1),
        (2, 3): ((0, 0), 6),

        (3, 0): ((0, 0), 7),  # 7a
        (3, 1): ((0, 0), 3),
        (3, 2): ((0, 0), 8),
        (3, 3): ((0, 1), 7)   # 7a
    }
    ax_d = {

        (0, 0): ((-1, 0), 7),  # 7b
        (0, 1): ((-1, 0), 3),
        (0, 2): ((0, 0), 8),
        (0, 3): ((0, 0), 7),   # 7b

        (1, 0): ((0, -1), 5),
        (1, 1): ((0, 0), 4),
        (1, 2): ((0, 0), 0),
        (1, 3): ((0, 0), 5),

        (2, 0): ((0, 0), 6),
        (2, 1): ((0, 0), 2),
        (2, 2): ((0, 0), 1),
        (2, 3): ((0, 0), 6),

        (3, 0): ((0, 0), 7),  # 7a
        (3, 1): ((0, 0), 3),
        (3, 2): ((0, 0), 8),
        (3, 3): ((0, 1), 7)   # 7a
    }
    # -4, -3
    # This is a proper half-hex symbol not merely a grid.,
    a: float       # This is the length of each side of the unit equilateral triangles of a district.
    h: float       # This is the height of a unit equilateral of a district (of which there are 3).
    id_ref: str    # The width of each H9 is 6a, the height is 6h.
    rt3 = math.sqrt(3.)   # the left/right edges have a width of 5a, and top/bottom: 3h
    stroke: float  # the pixel-width/height of H9 are 4.5a / 3h

    def axial(self, where, b3=False, district=None):
        # currently does not multiply by 3 when district is none.
        # This might be a mistake for usability.
        dq, dr = self._hu.to_axial(where)
        if district is not None:
            di, dj = self.d_ax[district]
            dq, dr = dq * 3 + di, dr * 3 + dj
        if not b3:
            return dq, dr
        return self.b3(dq), self.b3(dr)

    def b3(self, i):
        # return sign, and lowest significant first order of base_3 of an integer.
        # eg b3(17) => +1,22100  (digits depend upon self.hierarchy)
        s, r, i = (+1, '', i) if i >= 0 else (-1, '', -i)
        for x in range(self.hierarchy):
            i, k = divmod(i, 3)
            r += f'{k}'
        return s, r

    @staticmethod
    def addr_bx2(p, c, ab: int = 3) -> int:
        # given parent hex p child hex c as h9 ids (0..8)
        # along with child-half.hex (a=1,/b=2,x=3),
        # return parent's half-hex (a/b/'') character. 1,2
        p_ab_lut = {
            (0, 0): 'X', (0, 1): 'b', (0, 2): 'a',
            (0, 3): 'X', (0, 4): 'a', (0, 5): 'b',
            (0, 6): 'Y', (0, 7): 'X', (0, 8): 'X',
            (1, 0): 'a', (1, 1): 'X', (1, 2): 'b',
            (1, 3): 'b', (1, 4): 'X', (1, 5): 'a',
            (1, 6): 'X', (1, 7): 'Y', (1, 8): 'X',
            (2, 0): 'b', (2, 1): 'a', (2, 2): 'X',
            (2, 3): 'a', (2, 4): 'b', (2, 5): 'X',
            (2, 6): 'X', (2, 7): 'X', (2, 8): 'Y'
        }
        dx = {'a': 1, 'b': 2, 'X': ab, 'Y': 3 if ab == 3 else 3 - ab}
        r = p_ab_lut[(p % 3, c)]
        return dx[r]

    def _sym(self, idx: str):
        pts = [p for xy in self.districts[self.master] for p in xy]
        return svg.Polygon(id=idx, stroke_width=self.stroke, fill_opacity=self.opacity, points=pts)

    def _bboxes(self):  # set bounding boxes.
        ll, rl, tl, bl = [], [], [], []
        for r in self.districts:
            xs, ys = list(zip(*r))
            ll.append(min(xs))
            rl.append(max(xs))
            tl.append(min(ys))
            bl.append(max(ys))
        return list(zip(ll, tl, rl, bl))

    @classmethod
    def hh2r(cls, where) -> tuple:
        # hh: given coordinates - eg (3,3), return h9 coordinates and region.
        ix, iy = where  # this is in hh co-ordinates
        ofx = ix >> 2
        ofy = (iy >> 2) + ((ix & 1) << 1)
        return ofx, ofy

    @staticmethod
    def dx_conf_short(adr: list):
        # df = ['', 'a', 'b', '']
        dq = [f'{a}' for (a, b) in adr[::-1]]
        # fx = df[adr[0][1]] if adr else ''
        return ''.join(dq)

    @staticmethod
    def dx_conf_full(adr: list):
        df = ['', 'a', 'b', '']
        dq = [f'{a}{df[b]}' for (a, b) in adr[::-1]]
        return ''.join(dq)

    @staticmethod
    def dm3(x):
        if x == 0:
            return 0, 0, 0
        sx, ax = int(math.copysign(1, x)), int(abs(x))
        dv, rm = divmod(ax, 3)
        return dv*sx, rm*sx, sx

    def get_hx(self, qx, rx, ab, dc):
        if (qx, rx, ab) in self.ax_dof:
            dp, qno, rno = self.ax_dof[(qx, rx, ab)]
        else:
            dp, qno, rno = self.ax_dof[(qx, rx, 3)]
        abn = self.addr_bx2(dp, dc, ab)
        return dp, abn, qno, rno

    def get_addr(self, where, district=0):
        # ax_dof: Bit 0:1 / Bit 1:2 so a=1,b=2,x=3  // 1 + (district & 1)
        hx, rr = 0, []  # debug vars.
        hx_c, ab = district >> 1, 1 + (district & 1)
        aq, ar = self._hu.to_axial(where)
        rs, qs, qno, rno, result = 0, 0, 0, 0, [(hx_c, ab)]
        for i in range(self.hierarchy+1):  # len(r) and len(q) are the same.
            aq, pqi, qs = self.dm3(aq + qs * qno)
            ar, pri, rs = self.dm3(ar + rs * rno)
            hx_p, abn, qno, rno = self.get_hx(pqi, pri, ab, hx_c)
            result.append((hx_p, abn))
            hx_c, ab = hx_p, abn
            if hx == i:
                rr = self.dx_conf_full([(hx_c, ab)])
        rez = f'{rr}/{self.dx_conf_full(result)}'
        return rez

    def canon(self, addr: str, long=False):
        lt = {'a': 1, 'b': 2, '': 3}
        tl = {1: 'a', 2: 'b', 3: ''}
        p, c = None, None
        ab = ''
        result = []
        for ph in addr[::-1]:
            c = p
            if ph in 'ab':
                ab = lt[ph]
                continue
            p = int(ph)
            if c not in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
                if long:
                    ab, cx = ab, f'{p}{tl[ab]}'
                else:
                    ab, cx = ab, f'{p}'
            else:
                f = self.addr_bx2(p, c, ab)
                if long:
                    ab, cx = f, f'{p}{tl[f]}'
                else:
                    ab, cx = f, f'{p}'
            result.append(cx)
        return ''.join(result[::-1]) + (f'{tl[ab]}' if long else '')

    def label_text(self, where, i):
        ab = 'a' if i & 1 == 0 else 'b'
        dx = f'{i >> 1}{'a' if i & 1 == 0 else 'b'}'
        aq, ar = self.axial(where, False, i)
        # q, r = self.axial(where, True, i)
        # qn, rn = q[::-1], r[::-1]
        # aq = f'{'' if qn[1] > 0 else '-'}{(qn[0])[::-1]}'
        # ar = f'{'' if rn[1] > 0 else '-'}{(rn[0])[::-1]}'
        rx = self.get_addr(where, i)
        # hx = ''.join([f'{i}{d}' for (i, d) in rx])  # + rx[-1][1]
        # hx = ''.join([f'{k}' for (k, _) in rx]) + rx[-1][1]
        # db = f'({qn},{rn}){self.lbh[i]}{ab}'
        # f'{rx} {aq} {ar} {i}/{where}'
        # wx, wy = where -4:
        ad = rx[3:]
        hx = ''.join([f'{j}' for i, j in enumerate(ad) if i & 1 == 0]) + ab
        cx = self.canon(hx, long=False)
        return f'{rx[:3]}{cx}'

    def set_offset(self, w, h):
        self.scx = w
        self.scy = h
        self.ofx = w * 0.5
        self.ofy = h * 0.5
        x = math.ceil((self.ofx + self.a * 3) / self.a45)
        y = math.ceil((self.ofy + self.h3) / self.h6)
        cx, cy = math.floor(w / 4.5 / self.a), math.floor(h / 3.0 / self.h2)
        self.tb, self.lr = [-cy/2, cy/2], [-cx/2, cx/2]
        self.hierarchy = math.ceil(math.sqrt((cx/3.)**2. + (cy/3.)**2.))
        # eg, for 3/4 = 2, so central h9 is at 000

    def __init__(self, owner: Drawing | None = None, identity: str = 'h9', size: float = 81., stroke: float = 0.5, op=0.9):
        self._hu = HexUtil()
        self.owner = owner
        self.opacity = op
        self.id_ref = f'#{identity}'
        self.stroke = stroke
        self.hierarchy = 1
        self.tb, self.lr = [], []
        # self.rt3 = math.sqrt(3.)  # tan(60) == math.sqrt(3.)
        # a hexagon can be seen as made of six equilateral triangles (ht)
        # each of which has a side length which for the hex is the same as it's
        # big radius 'R' (or åa). Each of those is divided into a half-hex
        # Which yields 18 half-hexes to the major hex.
        # The small radius r is the height 'h' of the equilateral triangles.
        # The height of a hex is 2r, and it's width is 2R
        self.a = float(size)
        self.h = self.a * 0.5 * self.rt3  # height of any equilateral triangle.
        self.a2 = 2. * self.a
        self.a3 = 1.5 * self.a
        self.a45 = 4.5 * self.a
        self.ah = 0.5 * self.a
        self.h2 = 2. * self.h
        self.h3 = 3. * self.h
        self.h6 = 6. * self.h
        self.scx, self.scy = self.a*6, self.h*4
        self.ofx, self.ofy = self.scx * 0.5, self.scy * 0.5

        # return the district polygons for radius rad.
        a, h = self.a, self.h  # height of any equilateral triangle.
        ah = self.ah
        # self.addr_lut = {}  # address decomposition lut
        self.pts = [0. - a, 0., a, 0., ah, h, -ah, h]
        self.districts = [translate(rotate(self.pts, rt[2]*60.), (a * rt[0], h * rt[1])) for rt in self.dx]
        self.tx = [translate([0, 0], (a * rt[0], h * rt[1]))[0] for rt in self.dx]
        self.bboxes = self._bboxes()
        # self._set_address_lut()
        if self.owner is not None:
            self.owner.define(self._sym(identity))
            self.set_offset(self.owner.width, self.owner.height)

    def wxy(self, where) -> list:
        xi, yi = where
        return [self.ofx + xi * self.a45, self.ofy + yi * self.h6 - (xi & 1) * self.h3]

    def points(self) -> list:
        return self.pts

    def set_limits(self, tb, lr):
        self.tb, self.lr = tb, lr

    @classmethod
    def size_for(cls, hw, hh, rx) -> tuple:
        return math.ceil(hw * 4.5 * rx), math.ceil(hh * rx * 3.0 * cls.rt3)

    def get_limits(self) -> tuple:
        x0, x1 = int(math.floor(self.lr[0])), int(math.ceil(self.lr[1]))
        y0, y1 = int(math.floor(self.tb[0])), int(math.ceil(self.tb[1]))
        return range(x0, x1+1), range(y0, y1+1), (int(self.ofx / self.a45), int(self.ofy / self.h6))

    def may_place(self, where: list, district: int) -> bool:  # where is in wc - use wxy before if using hex_coords
        (l, t), (r, b) = translate(self.bboxes[district], where)
        return 0 <= l and r <= self.scx and 0 <= t and b <= self.scy

    def place_district(self, where: list, district: int, color: str = 'black', label=None):
        if self.owner is not None:
            px, py = where
            tx, ty = self.tx[district]
            rot = self.dx[district][2]
            r = svg.Rotate(rot * 60., -self.a, 0)
            t = svg.Translate(tx+px+self.a, ty+py)
            inst = svg.Use(href=self.id_ref, fill=color, transform=[t, r])
            self.owner.add(inst)
            if label is not None:
                lx, ly = self.lpt[district]
                self.owner.label(label, self.a/12, tx+px, ty+py, lx*self.a, ly*self.h)
        else:
            return translate(self.districts[district], where)

    def place(self, where, colors):
        wc = self.wxy(where)
        for i in range(len(self.districts)):
            if self.may_place(wc, i):
                label = self.label_text(where, i)
                a = int(label[0]) * 2
                a += 0 if label[1] == 'a' else 1
                label = label[3:]
                x, y = where
                q, r = self.axial(where, False, i)
                label += f';{q},{r}'
                if i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]:
                    self.place_district(wc, i, colors[a], label)
                else:
                    self.place_district(wc, i, colors[a], None)


if __name__ == '__main__':
    cmap = mpl.colormaps['tab20'].resampled(20)
    cols = [mpl.colors.rgb2hex(cmap(i)) for i in range(18)]
    # cmap = mpl.colormaps['plasma'].resampled(30)
    # cols = [mpl.colors.rgb2hex(cmap(i+6)) for i in range(18)]
    radius = 120.
    dim = H9.size_for(4, 4, radius)
    cs = Drawing('test1', dim, False)
    h0 = H9(cs, 'h1', radius, 0, op=0.90)
    h0.hierarchy = 2
    (xx, yy, oo) = h0.get_limits()
    for j in yy:
        for i in xx:
            h0.place([i, j], cols)
    cs.save()
