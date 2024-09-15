# from __future__ import annotations
from typing import List, Any
import svg
import math
from itertools import batched
from collections.abc import Sequence
from drawing import Drawing
import matplotlib as mpl
import numpy as np
from hexutil import HexUtil


def translate(p, t):
    tx, ty = t
    #  p points, t is translate.
    ps = p if isinstance(p[0], Sequence) else list(batched(p, 2))
    return [(x+tx, y+ty) for x, y in ps]


def rotate(p, r, o=(0, 0)):  # rotate around angle r.
    th = np.radians(r)
    ox, oy = o
    pts = p if isinstance(p[0], Sequence) else batched(p, 2)
    return [(
                (x - ox) * np.cos(th) - (y - oy) * np.sin(th) + ox,
                (x - ox) * np.sin(th) + (y - oy) * np.cos(th) + oy
            ) for (x, y) in pts]


class H9Grid:
    # This class could be refactored quite a bit.
    master = 0
    dx = [  # clockwise, starting \–/
        [-1., 0., 0], [-1., 0., 3.], [0.5, -1.0, 2.], [0.5, -1.0, 5.],  # 0a,0b,1a,1b,
        [0.5, 1., 4.], [0.5, 1., 1.], [2., 0., 0], [2., 0., 3],    # 2a,2b,3a,3b
        [-1., 2., 2], [-1., 2., 5],  [-1., -2, 4], [-1., -2., 1],   # 4a,4b,5a,5b
        [0.5, -3, 0], [0.5, 3., 3], [2., 2., 2], [-2.5, -1.0, 5.],
        [-2.5, 1., 4], [2., -2., 1]
    ]
    lpt = [
        [-0.85,  0.20], [-0.85, -0.05], [-0.50, -0.80], [-0.36, 0.95],
        [-0.36, -0.80], [-0.50, 0.95], [-0.85, 0.20], [-0.85, -0.05],
        [-0.50, -0.80], [-0.36, 0.95], [-0.36, -0.80], [-0.50, 0.95],
        [-0.85,  0.20], [-0.85, -0.05], [-0.50, -0.80], [-0.36, 0.95],
        [-0.36, -0.80], [-0.50, 0.95]
    ]
    # This is a proper half-hex symbol not merely a grid.,
    a: float       # This is the length of each side of the unit equilateral triangles of a district.
    h: float       # This is the height of a unit equilateral of a district (of which there are 3).
    id_ref: str    # The width of each H9 is 6a, the height is 6h.
    rt3 = math.sqrt(3.)   # the left/right edges have a width of 5a, and top/bottom: 3h
    stroke: float  # the pixel-width/height of H9 are 4.5a / 3h

    @staticmethod
    def trs(p, trs):
        #  p points, trs is translate/rotate/scale
        (tx, ty), (rot, ox, oy), sc = trs
        th = np.radians(rot)
        pb = p if isinstance(p[0], Sequence) else list(batched(p, 2))
        sp = [(sc * x, sc * y) for (x, y) in pb]
        rp = [((x - ox) * np.cos(th) - (y - oy) * np.sin(th) + ox, (x - ox) * np.sin(th) + (y - oy) * np.cos(th) + oy) for (x, y) in sp]
        tp = [(x + tx, y + ty) for (x, y) in rp]
        return tp

    @staticmethod
    def scale_translate(p, xys):
        tx, ty, sc = xys
        ps = p if isinstance(p[0], Sequence) else list(batched(p, 2))
        return [(sc * x + tx, sc * y + ty) for (x, y) in ps]

    @staticmethod
    def axial(where, district):
        d_ax = [
            (0, 0, 1), (0, 0, 2), (1, 0, 1), (1, 0, 2), (1, -1, 1), (1, -1, 2),
            (2, -1, 1), (2, -1, 2), (0, -1, 1), (0, -1, 2), (0, 1, 1), (0, 1, 2),
            (1, 1, 1), (1, -2, 2), (2, -2, 1), (-1, 1, 2), (-1, 0, 1), (2, 0, 2)
        ]
        x, y, lv = where
        q, r = x, -y - (x - (x & 1)) // 2  # This from offset 'odd-q' to qrs
        i, j, ab = d_ax[district]
        q, r = q * 3 + i, r * 3 + j
        m = 3 ** lv
        return m*q, m*r, ab

    def _sym(self, idx: str):
        pts = [p for xy in self.districts[self.master] for p in xy]
        return svg.Polygon(id=idx, fill_opacity=self.opacity, points=pts)

    def _sym_hex(self, idx: str):
        pts = [p*3. for p in self.hex]
        return svg.Polygon(id=idx, stroke='black', stroke_width=self.hex_stroke, fill_opacity=self.hex_opacity, points=pts)

    def _bboxes(self):  # set bounding boxes.
        ll, rl, tl, bl = [], [], [], []
        for r in self.districts:
            xs, ys = list(zip(*r))
            ll.append(min(xs))
            rl.append(max(xs))
            tl.append(min(ys))
            bl.append(max(ys))
        return list(zip(ll, tl, rl, bl))

    @staticmethod
    def axial_dm(x, denominator):
        if x == 0:
            return 0, 0
        sx, ax = int(math.copysign(1, x)), int(abs(x))
        dv, rm = divmod(ax, denominator)
        return dv*sx, rm*sx

    def d3(self, x):
        return self.axial_dm(x, 3)

    def _label_text(self, where, ab=1, level=0):
        n_dxd = {
            (0, 1): (0, 0, 0), (0, 2): (0, 0, 1), (1, 1): (1, 0, 2), (1, 2): (1, 0, 3),
            (2, 1): (1, -1, 4), (2, 2): (1, -1, 5), (3, 1): (2, -1, 6), (3, 2): (2, -1, 7),
            (4, 1): (0, -1, 8), (4, 2): (0, -1, 9), (5, 1): (0, 1, 10), (5, 2): (0, 1, 11),
            (6, 1): (1, 1, 12), (6, 2): (1, -2, 13), (7, 1): (2, -2, 14), (7, 2): (-1, 1, 15),
            (8, 1): (-1, 0, 16), (8, 2): (2, 0, 17)
        }
        ax3_to_d = {
            (-2, +1): 6, (-2, +2): 2, (-2, -1): 2, (-2, -2): 6, (-2, 0): 1, (-1, +1): 7, (-1, +2): 3,
            (-1, -1): 3, (-1, -2): 7, (-1, 0): 8, (0, +1): 5, (0, +2): 4, (0, -1): 4,
            (0, -2): 5, (0, 0): 0, (+1, +2): 2, (+1, -1): 2, (+1, -2): 6, (+1, 0): 1,
            (+1, +1): 6, (+2, +1): 7, (+2, +2): 3, (+2, -1): 3, (+2, -2): 7, (+2, 0): 8,
        }
        ab_lut = [{}, {
            (0, 0): 1, (0, 1): 2, (0, 2): 1,
            (0, 3): 1, (0, 4): 1, (0, 5): 2,
            (0, 6): 2, (0, 7): 1, (0, 8): 1,
            (1, 0): 1, (1, 1): 1, (1, 2): 2,
            (1, 3): 2, (1, 4): 1, (1, 5): 1,
            (1, 6): 1, (1, 7): 2, (1, 8): 1,
            (2, 0): 2, (2, 1): 1, (2, 2): 1,
            (2, 3): 1, (2, 4): 2, (2, 5): 1,
            (2, 6): 1, (2, 7): 1, (2, 8): 2
        }, {
            (0, 0): 2, (0, 1): 2, (0, 2): 1,
            (0, 3): 2, (0, 4): 1, (0, 5): 2,
            (0, 6): 1, (0, 7): 2, (0, 8): 2,
            (1, 0): 1, (1, 1): 2, (1, 2): 2,
            (1, 3): 2, (1, 4): 2, (1, 5): 1,
            (1, 6): 2, (1, 7): 1, (1, 8): 2,
            (2, 0): 2, (2, 1): 1, (2, 2): 2,
            (2, 3): 1, (2, 4): 2, (2, 5): 2,
            (2, 6): 2, (2, 7): 2, (2, 8): 1
        }]
        od, result = 0, []
        q, r, = where
        if level > 0:
            sc = 3 ** level
            q, qd = self.axial_dm(q, sc)
            r, rd = self.axial_dm(r, sc)
        for i in range(self.hierarchy - level):
            _, qi = self.d3(q)  # we want to keep q, r
            _, ri = self.d3(r)  # the qi,ri of the district-hex.
            nd = ax3_to_d[(qi, ri)]  # convert to district-hex.
            ab = ab_lut[ab][(nd % 3, od)] if i > 0 else ab  # identify the district a/b
            dq, dr, dx = n_dxd[(nd, ab)]  # get offset of district and it's id.
            result.append(dx)  # store district 0..17
            q, _ = self.d3(q - dq)  # divide from district 0
            r, _ = self.d3(r - dr)  # divide from district 0
            od = nd  # keep the old district for calculating next a/b
        rez = ''.join([f'{d >> 1}' for d in result[::-1]]) + ['a', 'b'][result[0] & 1]
        return rez

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
        self.hex_ref = f'#{identity}hex'
        self.stroke = stroke
        self.hex_stroke = 0.2
        self.hex_opacity = 0.05
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
        self.pts = [a, 0., ah, h, -ah, h, 0. - a, 0.]
        self.hex = [a, 0., ah, h, -ah, h, 0. - a, 0., -ah, -h, ah, -h]
        self.districts = [translate(rotate(self.pts, rt[2]*60.), (a * rt[0], h * rt[1])) for rt in self.dx]
        self.tx = [translate([0, 0], (a * rt[0], h * rt[1]))[0] for rt in self.dx]
        self.bboxes = self._bboxes()
        # self._set_address_lut()
        if self.owner is not None:
            self.owner.define(self._sym(identity))
            self.owner.define(self._sym_hex(f'{identity}hex'))
            self.set_offset(self.owner.width, self.owner.height)

    def wxy(self, where) -> list:
        xi, yi, level = where   # record hierarchy. 0 = normal.
        wl = 3 ** level
        # for each level we need to shift a bit left or right.
        # The amount, in a is (3**(level+1)-3)a/2
        # Why?! 3**0-3 = 2 so, this is the amount in ah we shift
        # the 0-district from the centre inside the current level.
        ov = (3**(level+1)-3) * self.ah
        return [
            self.ofx + wl * xi * self.a45 + ov,
            self.ofy + wl * (yi * self.h6 - (xi & 1) * self.h3),
            ov
        ]

    def points(self) -> list:
        return self.pts

    def set_limits(self, tb, lr):
        self.tb, self.lr = tb, lr

    @classmethod
    def size_for(cls, hw, hh, rx) -> tuple:
        # returns results in 'pixels' - not h9 but inner hexes.
        return math.ceil((0 + hw) * 4.5 * rx), math.ceil((0 + hh) * rx * 3.0 * cls.rt3)

    def get_limits(self, over=True) -> tuple:
        x0, x1 = int(math.floor(self.lr[0])) - 2, int(math.floor(self.lr[1]) + 2)
        y0, y1 = int(math.floor(self.tb[0])) - 2, int(math.floor(self.tb[1]) + 2)
        return range(x0, x1), range(y0, y1), (-x0 - 0, -y0 - 0)

    def may_place(self, district: int, xys) -> bool:  # where is in wc - use wxy before if using hex_coords
        (l, t), (r, b) = self.scale_translate(self.bboxes[district], xys)
        return 0 <= l and r <= self.scx and 0 <= t and b <= self.scy

    def place_district(self, where, district: int, color: [str | None], label=None):
        lv = 3 ** where[2]
        (px, py, ov) = self.wxy(where)
        dx, dy = self.tx[district]
        tx = px + dx * lv + self.a * lv  # px + lv * dx + ov
        ty = py + dy * lv
        rot, ox, oy = self.dx[district][2] * 60., -self.a * lv, 0
        if self.owner is not None:
            t = svg.Translate(tx, ty)
            r = svg.Rotate(rot, ox, oy)
            s = svg.Scale(lv)
            if color is not None:
                inst = svg.Use(href=self.id_ref, fill=color, transform=[t, r, s])
            else:
                inst = svg.Use(href=self.id_ref, stroke_width=0.02, stroke="#000000", fill="none", stroke_opacity=0.4, transform=[t, r, s])
            self.owner.add(inst)
            if label is not None:
                lx, ly = self.lpt[district]
                self.owner.label(label, self.a/8 * lv, tx, ty, (lx-1) * self.a * lv, ly * self.h * lv)
        else:
            pts = self.trs(self.districts[self.master], ((tx, ty), (rot, ox, oy), lv))
            return pts

    def label_txt(self, where, district):
        iq, ir, _ = self.axial(where, district)  # where is x, y, level using odd coords.
        return self._label_text((iq, ir), 1 + (district & 1), where[2])

    def place(self, where, colors=None, label=True, ignore_bounds=False):
        wc = self.wxy(where)
        xys = wc[0], wc[1], 3 ** where[2]
        for district in range(len(self.districts)):
            if ignore_bounds or self.may_place(district, xys):
                color = None if colors is None else colors[district]
                if label:
                    label = self.label_txt(where, district)
                    self.place_district(where, district, color, label)
                else:
                    self.place_district(where, district, color, None)

    def place_hex(self, where, color: str = 'black', label=None):
        lv = 3 ** where[2]
        wco = self.wxy(where)
        if self.owner is not None:
            px, py, ov = wco
            t = svg.Translate(px, py)
            s = svg.Scale(lv)
            inst = svg.Use(href=self.hex_ref, stroke=color, fill=None, transform=[t, s])
            self.owner.add(inst)
            if label is not None:
                ls = 3 ** lv
                lx, ly = self.lpt[where[2]]
                self.owner.label(label, ls*self.a/18, px, py, lx*self.a, ly*self.h)
        else:
            return self.scale_translate(self.hex, wco)


class H9District:
    def __init__(self, grid: H9Grid):
        self._grid = grid
        self.parent = None
        self.color = None
        self.tris = None, None, None

    def set_color(self, color=None):
        if color is not None:
            self.color = color

    def place(self, depth=0, ignore_bounds=False):
        pass

class H9Hex:
    districts: list[H9District]

    def __init__(self, grid: H9Grid):
        self._grid = grid
        self.a = grid.a
        self.h = grid.h
        self.parent = None
        self.ab = None, None
        self.colour = None
        self.districts = []
        self.colour = None
        self.wx, self.wy, self.lv = None, None, None
        self.scale = None

    def set_where(self, where):
        self.wx, self.wy, self.lv = where
        self.scale = self.lv ** 3

    # def place_label(self, where, label):
    #     if label is not None:
    #         lx, ly = self._grid.lpt[district]
    #         self._grid.owner.label(label, self._grid.a / 8 * lv, tx, ty, (lx - 1) * self.a * lv, ly * self.h * lv)

    def set_colors(self, colors=None):
        if colors is not None:
            for i, district in enumerate(self.districts):
                district.color = colors[i]

    def place(self, depth=0, colors=None, ignore_bounds=False):
        xys =self.wx, self.wy, self.scale
        if colors is not None:
            for district in enumerate(self.districts):
                pass
        for district in enumerate(self.districts):
            district.place(depth, ignore_bounds)

class H9Tri:
    def __init__(self, grid: H9Grid):
        self._grid = grid
        self.parent = None
        self.color = None
        self.hh = [None, None, None]



    def set_spherical(self, ijk):
        # ijk is a tuple of three cartesian points on a unit sphere in clockwise order.
        # these will be treated as a spherical equilateral triangle.
        pass


if __name__ == '__main__':
    cmap = mpl.colormaps['plasma'].resampled(20)
    cols = [mpl.colors.rgb2hex(cmap(i+1)) for i in range(18)]
    # if we are drawing a sphere, we need to work out depth of resolution we want.
    # eg 1x9^4 = 6561
    # radius 9 = so 2x9x6561
    # measuring in h9, though, it's 3^4 hh width: 81
    radius = 81.
    dim = H9Grid.size_for(30, 20, radius)
    cs = Drawing('test_6', dim, False)
    h0 = H9Grid(cs, 'h0', radius, 20, op=0.90)
    h0.hierarchy = 5
    (xx, yy, oo) = h0.get_limits()

    for j in yy:
        for i in xx:
            h0.place([i, j, 0], cols, label=True, ignore_bounds=False)
    for j in yy:
        for i in xx:
            h0.place([i, j, 1], None, label=False, ignore_bounds=False)
    for j in yy:
        for i in xx:
            h0.place([i, j, 2], None, label=False, ignore_bounds=False)
    # for j in range(0, 2):
    #     for i in range(-1, 1):
    #         h0.place([i, j, 3], None, label=True, ignore_bounds=True)

    cs.save()
