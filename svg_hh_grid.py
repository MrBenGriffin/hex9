import svg
import math
import matplotlib as mpl
from random import shuffle
from drawing import Drawing
from h9 import H9Grid


class HexUtil:
    @staticmethod
    def to_offs(hc: tuple) -> tuple:
        q, r, s = hc
        x = q
        y = r + (q - (q & 1)) // 2   # This from qrs to offset 'odd-q'
        return x, -y

    @staticmethod
    def to_cube(hc: tuple):
        # This from offset 'odd-q' to qrs
        # col = x, row=y
        x, ny = hc
        y = -ny
        q = x
        r = y - (x - (x & 1)) // 2  # This from offset 'odd-q' to qrs
        s = -q - r
        return q, r, s


class H2:
    a: float
    h: float
    id_ref: str
    rt3: float
    stroke: float

    def _sym(self, idx: str):
        ih = [
            0., 0.,
            self.a2, 0.,
            self.a3, self.h,
            self.ah, self.h,
        ]
        self.wid, self.hgt = self.a2, self.h
        min_xy, self.vx, self.vy = 0. - self.stroke, self.wid + 2.*self.stroke, self.hgt + 2.*self.stroke
        return svg.Polygon(id=idx, stroke_width=self.stroke, fill_opacity=self.opacity, points=ih)

    def __init__(self, owner: Drawing, identity: str = 'h2', size: float = 81., stroke: float = 0.5, op=0.9):
        # size = side_length
        self.owner = owner
        self.opacity = op
        self.id_ref = f'#{identity}'
        self.stroke = stroke
        self.rt3 = math.sqrt(3.)  # tan(60) == math.sqrt(3.)
        # a hexagon can be seen as made of two half-hex
        # The small radius r is the height 'h' of the equilateral triangles.
        # The height of a hex is 2r, and it's width is 2R
        self.a = float(size)
        self.h = self.a * 0.5 * self.rt3  # height of any equilateral triangle.
        self.a2 = 2. * self.a
        self.a3 = 1.5 * self.a
        self.ah = 0.5 * self.a
        self.h2 = 2. * self.h
        self.wid, self.hgt, self.vx, self.vy = 0, 0, 0, 0
        self.owner.define(self._sym(identity))
        self.cx, self.cy = self.vx * 0.5, self.vy * 0.5
        self.ofx, self.ofy = self.owner.width * 0.5-self.a2, self.owner.height * 0.5

    def hh(self, where: list, color: str, district: int):
        districts = [
            [0., 0., 3],
            [0., 0., 0]
        ]
        tx, ty, rot = districts[district]
        tx *= self.a
        ty *= self.h
        xi, yi = where
        x = self.ofx + xi * self.a * 1.5
        y = self.ofy + yi * self.h * 2. - (xi & 1) * self.h
        r = svg.Rotate(rot * 60., self.a, 0)
        t = svg.Translate(x+tx, y+ty)
        inst = svg.Use(href=self.id_ref, fill=color, transform=[t, r])
        self.owner.add(inst)

    def hex(self, where: list, colors: list):
        self.hh(where, colors[i], 0)
        self.hh(where, colors[i], 1)


if __name__ == '__main__':
    cmap = mpl.colormaps['cividis'].resampled(288)
    cols = [mpl.colors.rgb2hex(cmap(i)) for i in range(288)]
    shuffle(cols)
    canvas = Drawing('test7', (600, 600), False)
    h1 = H9Grid(canvas, 'h1', 9., 0, op=0.90)
    h1.set_limits((-5, 4), (-5, 4))
    h1.hierarchy = 4
    level = 0
    for j in range(-5, 5):
        for i in range(-5, 5):
            shuffle(cols)
            h1.place([i, j, level], cols)

    canvas.save()
