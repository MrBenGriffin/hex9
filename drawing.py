import svg
import math


class Drawing:
    def __init__(self, name: str, size: tuple, bg: bool):
        self.name = name
        self.graph = None
        self.width, self.height = size
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

    def label(self, text, sz, x1, y1, dx, dy):
        label = svg.Text(x=x1, y=y1, dx=dx, dy=dy, text=text, font_family='monospace', font_size=sz)
        self.canvas.elements.append(label)

    def line(self, x1, y1, x2, y2, w=0.5):
        line = svg.Line(x1=x1, y1=y1, x2=x2, y2=y2, stroke_width=w, stroke="black")
        self.canvas.elements.append(line)

    def save(self):
        file = open(f"output/{self.name}.svg", "w")
        file.write(self.canvas.as_str())
        file.close()


class H2(Drawing):
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
        min_xy, self.vx, self.vy = 0. - self.stroke, self.wid + 2. * self.stroke, self.hgt + 2. * self.stroke
        return svg.Polygon(id=idx, stroke_width=self.stroke, fill_opacity=self.opacity, points=ih)

    def __init__(self, name: str, size: tuple, bg: bool, sz: float = 81.,
                 stroke: float = 0.5, op=0.9, identity: str = 'h2'):
        # size = side_length
        super().__init__(name, size, bg)
        self.opacity = op
        self.id_ref = f'#{identity}'
        self.stroke = stroke
        self.rt3 = math.sqrt(3.)  # tan(60) == math.sqrt(3.)
        # a hexagon can be seen as made of two half-hex
        # The small radius r is the height 'h' of the equilateral triangles.
        # The height of a hex is 2r, and it's width is 2R
        self.a = float(sz)
        self.h = self.a * 0.5 * self.rt3  # height of any equilateral triangle.
        self.a2 = 2. * self.a
        self.a3 = 1.5 * self.a
        self.ah = 0.5 * self.a
        self.a4 = 0.25 * self.a
        self.h2 = 2. * self.h
        self.wid, self.hgt, self.vx, self.vy = 0, 0, 0, 0
        self.define(self._sym(identity))
        self.cx, self.cy = self.vx * 0.5, self.vy * 0.5
        self.ofx, self.ofy = self.a4, self.h2

    def dim(self) -> tuple:
        return self.wid, self.hgt

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
        t = svg.Translate(x + tx, y + ty)
        inst = svg.Use(href=self.id_ref, fill=color, transform=[t, r])
        self.add(inst)

    def hex(self, where: list, col: tuple):
        r, g, b = col
        c = f'#{int(r):02X}{int(g):02X}{int(b):02X}'
        self.hh(where, c, 0)
        self.hh(where, c, 1)

