import svg
import math
from drawing import Drawing
import numpy as np
from itertools import batched
from collections.abc import Sequence
from drawing import Drawing


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


if __name__ == '__main__':
    print('test h6 here')
