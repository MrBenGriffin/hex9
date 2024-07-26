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


class H9:
    # This is a proper half-hex symbol not merely a grid.,
    a: float       # This is the length of each side of the unit equilateral triangles of a district.
    h: float       # This is the height of a unit equilateral of a district (of which there are 3).
    id_ref: str    # The width of each H9 is 6a, the height is 6h.
    rt3: float     # the left/right edges have a width of 5a, and top/bottom: 3h
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
        self.rt3 = math.sqrt(3.)  # tan(60) == math.sqrt(3.)
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
        self.dx = [
            # clockwise, starting \–/
            [-1., 0., 3.], [0.5, -1.0, 2.], [0.5, -1.0, 5.],
            [0.5, 1., 4.], [0.5, 1., 1.], [-1., 0., 0],
            [-1., -2., 1], [-1., -2, 4], [2., 0., 3], [2., 0., 0],
            [-1., 2., 5], [-1., 2., 2],
            [-2.5, -1.0, 5.], [0.5, -3, 0], [2., -2., 1],
            [2., 2., 2], [0.5, 3., 3], [-2.5, 1., 4]
        ]
        self.districts = [translate(rotate(self.pts, rt[2]*60.), (a * rt[0], h * rt[1])) for rt in self.dx]
        self.tx = [translate([0, 0], (a * (rt[0]), h * rt[1]))[0] for rt in self.dx]
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

    def get_limits(self) -> tuple:
        x0, x1 = self.lr
        y0, y1 = self.tb
        return range(x0, x1), range(y0, y1)

    def may_place(self, where: list, district: int) -> bool:  # this is in wc. translate to wc if using hex_coords
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
        else:
            return translate(self.districts[district], where)

    def place(self, where: list, colors: list):
        wc = self.wxy(where)
        for i in range(len(self.districts)):
            if self.may_place(wc, i):
                self.place_district(wc, i, colors[i])


if __name__ == '__main__':
    cmap = mpl.colormaps['plasma'].resampled(18)
    cols = [mpl.colors.rgb2hex(cmap(i)) for i in range(18)]

    canvas = Drawing('test7', (900, 900), False)
    h1 = H9(canvas, 'h1', 9., 0, op=0.90)
    (xx, yy) = h1.get_limits()
    for j in yy:
        for i in xx:
            h1.place([i, j], cols)

    canvas.save()
