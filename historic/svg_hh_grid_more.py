import svg
import math
import matplotlib as mpl
from random import shuffle


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


class Drawing:
    def __init__(self, name: str, size: tuple, bg: bool):
        self.name = name
        self.graph = None
        self.width, self.height = size
        # self.size = size
        self.canvas = svg.SVG(
            viewBox=svg.ViewBoxSpec(0, 0, self.width, self.height),
            width=self.width, height=self.height,
            elements=[]
        )
        self.defs = svg.Defs(elements=[])
        self.canvas.elements.append(self.defs)
        if bg:
            ax, ay, bx, by = [0, 0, self.width, self.height]
            bp = svg.Polygon(fill=f'white', stroke="none")
            bp.points = [(ax, ay), (ax, by), (bx, by), (bx, ay)]
            self.canvas.elements.append(bp)

    def add(self, thing):
        self.canvas.elements.append(thing)

    def define(self, thing):
        self.defs.elements.append(thing)

    def save(self):
        file = open(f"output/{self.name}.svg", "w")
        file.write(self.canvas.as_str())
        file.close()


class SymHH:
    # This is a proper half-hex symbol not merely a grid.,
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

    def __init__(self, owner: Drawing, identity: str = 'sym_hh', size: float = 81., stroke: float = 0.5, op=0.9):
        self.owner = owner
        # self.rgb = rgb
        self.opacity = op
        self.id_ref = f'#{identity}'
        self.stroke = stroke
        # self.font_size = font_size
        self.rt3 = math.sqrt(3.)  # tan(60) == math.sqrt(3.)
        # a hexagon can be seen as made of six equilateral triangles (ht)
        # each of which has a side length which for the hex is the same as it's
        # big radius 'R' (or a). Each of those is divided into a half-hex
        # Which yields 18 half-hexes to the major hex.
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

    def place(self, where: list, color: str, district: int):
        districts = [
            [0., 0., 3], [1.5, -1., 2], [1.5, -1., 5],
            [1.5,  1., 4], [1.5, 1., 1], [0., 0., 0],
            [0., -2., 1], [0., -2, 4], [3., 0., 3],
            [3., 0., 0], [0., 2., 5], [0., 2., 2],
            [-1.5, -1., 5], [1.5, -3, 0], [3., -2., 1],
            [3.,  2., 2], [1.5, 3., 3], [-1.5, 1., 4]
        ]
        tx, ty, rot = districts[district]
        tx *= self.a
        ty *= self.h
        xi, yi = where
        x = self.ofx + xi * self.a * 4.5
        y = self.ofy + yi * self.h * 6. - (xi & 1) * (self.h * 3.)
        r = svg.Rotate(rot * 60., self.a, 0)
        t = svg.Translate(x+tx, y+ty)
        inst = svg.Use(href=self.id_ref, fill=color, transform=[t, r])
        self.owner.add(inst)

    def place_hex(self, where: list, colors: list):
        self.place(where, mpl.colors.rgb2hex(colors[i]), i)


if __name__ == '__main__':
    cmap = mpl.colormaps['turbo'].resampled(1024)
    cols = [cmap(i) for i in range(1024)]
    shuffle(cols)

    canvas = Drawing('test4', (900, 300), False)
    h2 = SymHH(canvas, 'h2', 9, 0, op=0.999)
    for i in range(-9, 9):
        shuffle(cols)
        h2.place_hex([i, -1], cols)
        shuffle(cols)
        h2.place_hex([i, 0], cols)
    canvas.save()
