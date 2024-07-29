import svg
import math
from itertools import batched
from collections.abc import Sequence
from drawing import Drawing
import matplotlib as mpl
import numpy as np


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

    # dx = [  # left to right, top to bottom.
    #     [0.5, 1., 1.], [0.5, 1., 4.], [2.0, 0., 0],  # 6, 7, 13
    #     [0.5, 3., 3.], [2.0, 2., 2.], [2.0, 2., 5.],  # 0, 1, 2
    #     [0.5, 3., 0.], [2.0, 4., 1.], [2.0, 4., 4.],  # 5, 4, 3
    #     [0.5, 5., 2.], [0.5, 5., 5.], [2.0, 6., 3]    # 11, 10, 16
    # ]
    dx = [  # left to right, top to bottom.
        [0.0, 1., 1.], [0.0, 1., 4.], [1.5, 0., 0],   # 6, 7, 13
        [0.0, 3., 3.], [1.5, 2., 2.], [1.5, 2., 5.],  # 0, 1, 2
        [0.0, 3., 0.], [1.5, 4., 1.], [1.5, 4., 4.],  # 5, 4, 3
        [0.0, 5., 2.], [0.0, 5., 5.], [1.5, 6., 3]    # 11, 10, 16
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
    dx = [  # clockwise, starting \–/
        [-1., 0., 3.], [0.5, -1.0, 2.], [0.5, -1.0, 5.],
        [0.5, 1., 4.], [0.5, 1., 1.], [-1., 0., 0],
        [-1., -2., 1], [-1., -2, 4], [2., 0., 3], [2., 0., 0],
        [-1., 2., 5], [-1., 2., 2],
        [-2.5, -1.0, 5.], [0.5, -3, 0], [2., -2., 1],
        [2., 2., 2], [0.5, 3., 3], [-2.5, 1., 4]
    ]
    lb = ['0b', '1a', '1b', '2a', '2b', '0a',
          '5b', '5a', '3b', '3a', '4b', '4a',
          '7b', '6a', '8b', '7a', '6b', '8a']
    lpt = [
        [-0.85, -0.05], [-0.53, -0.72], [-0.36, 0.94],   # 012
        [-0.36, -0.72], [-0.46, 0.94], [-0.85, 0.28],   # 345
        [-0.46, 0.94], [-0.36, -0.72], [-0.85, -0.05], [-0.85, 0.28],  # 6789
        [-0.36, 0.94], [-0.48, -0.72],  # 10,11
        [-0.36, 0.94], [-0.85, 0.28], [-0.46, 0.94],  # 12,13,14
        [-0.53, -0.72], [-0.85, -0.05], [-0.36, -0.72],  # 15,16,17
    ]
    # h9_hh = {  # h9_hh: given an address in h9 coordinates, return list of equivalent hh coordinates.
    #     0: (1, 1), 1: (2, 1), 2: (3, 1), 3: (3, 2), 4: (2, 2), 5: (1, 2),
    #     6: (1, 0), 7: (2, 0), 8: (4, 1), 9: (4, 2), 10: (2, 3), 11: (1, 3),
    #     12: (0, 1), 13: (3, 0), 14: (4, 0), 15: (4, 3), 16: (3, 3), 17: (0, 2)
    # }
    # hh_h9 = {  # hh_h9: given an address in hh coordinates, return list of equivalent h9 coordinates.
    #     (0, 1): 12, (0, 2): 17,
    #     (1, 0): 6,  (1, 1): 0, (1, 2): 5, (1, 3): 11,
    #     (2, 0): 7,  (2, 1): 1, (2, 2): 4, (2, 3): 10,
    #     (3, 0): 13, (3, 1): 2, (3, 2): 3, (3, 3): 16,
    #     (4, 0): 14, (4, 1): 8, (4, 2): 9, (4, 3): 15
    # }
    # This is a proper half-hex symbol not merely a grid.,
    a: float       # This is the length of each side of the unit equilateral triangles of a district.
    h: float       # This is the height of a unit equilateral of a district (of which there are 3).
    id_ref: str    # The width of each H9 is 6a, the height is 6h.
    rt3 = math.sqrt(3.)   # the left/right edges have a width of 5a, and top/bottom: 3h
    stroke: float  # the pixel-width/height of H9 are 4.5a / 3h

    def _sym(self, idx: str):
        pts = [p for xy in self.districts[5] for p in xy]
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
        self.ofx = w * 0.5
        self.ofy = h * 0.5
        x = math.ceil((self.ofx + self.a * 3) / self.a45)
        y = math.ceil((self.ofy + self.h3) / self.h6)
        self.tb, self.lr = [-y, y], [-x, x]

    def __init__(self, owner: Drawing | None = None, identity: str = 'h9', size: float = 81., stroke: float = 0.5, op=0.9):
        self.owner = owner
        self.opacity = op
        self.id_ref = f'#{identity}'
        self.stroke = stroke
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
        self.pts = [0. - a, 0., a, 0., ah, h, -ah, h]
        self.districts = [translate(rotate(self.pts, rt[2]*60.), (a * rt[0], h * rt[1])) for rt in self.dx]
        self.tx = [translate([0, 0], (a * rt[0], h * rt[1]))[0] for rt in self.dx]
        self.bboxes = self._bboxes()
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
        x0, x1 = self.lr
        y0, y1 = self.tb
        return range(x0, x1), range(y0, y1), (int(self.ofx / self.a45), int(self.ofy / self.h6))

    def may_place(self, where: list, district: int) -> bool:  # where is in wc - use wxy before if using hex_coords
        (l, t), (r, b) = translate(self.bboxes[district], where)
        return 0 <= l and r <= self.scx and 0 <= t and b <= self.scy

    def place_district(self, where: list, district: int, color: str = 'black'):
        if self.owner is not None:
            px, py = where
            tx, ty = self.tx[district]
            rot = self.dx[district][2]
            r = svg.Rotate(rot * 60., -self.a, 0)
            t = svg.Translate(tx+px+self.a, ty+py)
            inst = svg.Use(href=self.id_ref, fill=color, transform=[t, r])
            self.owner.add(inst)
            dx, dy = self.lpt[district]
            self.owner.label(self.lb[district], 2.5, tx+px, ty+py, dx*self.a, dy*self.h)
        else:
            return translate(self.districts[district], where)

    # @classmethod
    # def h92hh(cls, where) -> list:
    #     # given an address in h9 coordinates, return list of equivalent hh8 coordinates.
    #     # following dict maps region to hh8. Used, for instance, to find colours for an h9
    #     ix, iy = where  # this is in h9 co-ordinates
    #     ofx = ix << 2
    #     ofy = (iy << 2) + ((ix & 1) << 1)
    #     return [(ofx+x, ofy+y) for (x, y) in [cls.h9_hh[k] for k in range(18)]]

    @classmethod
    def hh2r(cls, where) -> tuple:
        # hh: given coordinates, return h9 coordinates and region.
        ix, iy = where  # this is in hh co-ordinates
        ofx = ix >> 2
        ofy = (iy >> 2) + ((ix & 1) << 1)

    def place(self, where, colors):
        wc = self.wxy(where)
        for i in range(len(self.districts)):
            if self.may_place(wc, i):
                self.place_district(wc, i, colors[i])


if __name__ == '__main__':
    cmap = mpl.colormaps['plasma'].resampled(18)
    cols = [mpl.colors.rgb2hex(cmap(i)) for i in range(18)]
    dim = H9.size_for(3, 4, 9)
    cs = Drawing('test9', dim, False)
    h0 = H9(cs, 'h1', 9., 0, op=0.90)
    (xx, yy, oo) = h0.get_limits()
    for j in yy:
        for i in xx:
            h0.place([i, j], cols)
    cs.save()

    # cmap = mpl.colormaps['plasma'].resampled(12)
    # cols = [mpl.colors.rgb2hex(cmap(i)) for i in range(12)]
    # dim = H6.size_for(60, 30, 9)
    # cs = Drawing('test8', dim, False)
    # h0 = H6(cs, 'h1', 9., 0, op=0.90)
    # xx, yy, oo = h0.get_limits()
    # for j in yy:
    #     for i in xx:
    #         h0.place([i, j], cols)
    # cs.save()
