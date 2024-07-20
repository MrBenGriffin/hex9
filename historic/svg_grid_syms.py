import svg
import math


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

    def line(self, x1, y1, x2, y2, w=0.5):
        line = svg.Line(x1=x1, y1=y1, x2=x2, y2=y2, stroke_width=w, stroke="black")
        self.canvas.elements.append(line)

    def save(self):
        file = open(f"output/{self.name}.svg", "w")
        file.write(self.canvas.as_str())
        file.close()


class SvgHH:
    # While a grid is really a pattern (which we could use with a fill marker),
    # We want to be able to add labels and so on to each fundamental element.
    # Therefore, I believe a symbol is more appropriate in this case.
    a: float
    h: float
    id_ref: str
    rt3: float
    stroke: float

    def _draw(self):
        ra, rh, ah, hh = self.a, self.h, self.ah, self.hh
        # Draw the surrounding polygon.
        hp = [self.cx - ah, self.cy - rh,
              self.cx - ra, self.cy,
              self.cx - ah, self.cy + rh,
              self.cx + ah, self.cy + rh,
              self.cx + ra, self.cy,
              self.cx + ah, self.cy - rh]
        self.graph = self.obj
        hx = svg.Polygon(stroke_width=self.stroke, fill="none", stroke=f'{self.rgb}', stroke_opacity=self.opacity,
                         points=hp)
        self.graph.elements.append(hx)
        self.graph.elements.append(
            svg.Line(x1=self.cx - ra, y1=self.cy, x2=self.cx + ra, y2=self.cy, stroke_width=self.stroke,
                     stroke=f'{self.rgb}', stroke_opacity=self.opacity))
        self.graph.elements.append(
            svg.Line(x1=self.cx - ah, y1=self.cy - rh, x2=self.cx + ah, y2=self.cy + rh, stroke_width=self.stroke,
                     stroke=f'{self.rgb}', stroke_opacity=self.opacity))
        self.graph.elements.append(
            svg.Line(x1=self.cx + ah, y1=self.cy - rh, x2=self.cx - ah, y2=self.cy + rh, stroke_width=self.stroke,
                     stroke=f'{self.rgb}', stroke_opacity=self.opacity))
        # so the hex and the main axis lines are all in.
        # the small hexs are exactly 1/3 length/width of the ah/rh etc/
        sa, sh, = self.a / 3., self.h / 3.
        sx, xt, x2, y2 = sa * 0.5, sa * 1.5, sa * 2., sh * 2.
        # We can draw the inners in different ways.
        ih = [
            self.cx - 2. * sa, self.cy,
            self.cx - xt, self.cy - sh,
            self.cx - sx, self.cy - sh,
            self.cx, self.cy - y2,
            self.cx + sa, self.cy - y2,
            self.cx + xt, self.cy - sh,
            self.cx + sa, self.cy,
            self.cx + xt, self.cy + sh,
            self.cx + sa, self.cy + y2,
            self.cx, self.cy + y2,
            self.cx - sx, self.cy + sh,
            self.cx - xt, self.cy + sh
        ]
        self.graph.elements.append(
            svg.Polygon(stroke_width=self.stroke, fill="none", stroke=f'{self.rgb}', stroke_opacity=self.opacity,
                        points=ih))
        l3 = svg.Line(x1=self.cx + xt, y1=self.cy - sh, x2=self.cx + ra - sx, y2=self.cy - sh, stroke_width=self.stroke,
                      stroke=f'{self.rgb}', stroke_opacity=self.opacity)
        self.graph.elements.append(l3)
        l4 = svg.Line(x1=self.cx + xt, y1=self.cy + sh, x2=self.cx + ra - sx, y2=self.cy + sh, stroke_width=self.stroke,
                      stroke=f'{self.rgb}', stroke_opacity=self.opacity)
        self.graph.elements.append(l4)
        l2 = svg.Line(x1=self.cx - 0, y1=self.cy - y2, x2=self.cx - sx, y2=self.cy - rh, stroke_width=self.stroke,
                      stroke=f'{self.rgb}', stroke_opacity=self.opacity)
        self.graph.elements.append(l2)
        l5 = svg.Line(x1=self.cx - 0, y1=self.cy + y2, x2=self.cx - sx, y2=self.cy + rh, stroke_width=self.stroke,
                      stroke=f'{self.rgb}', stroke_opacity=self.opacity)
        self.graph.elements.append(l5)
        l1 = svg.Line(x1=self.cx - xt, y1=self.cy - sh, x2=self.cx - x2, y2=self.cy - y2, stroke_width=self.stroke,
                      stroke=f'{self.rgb}', stroke_opacity=self.opacity)
        self.graph.elements.append(l1)
        l6 = svg.Line(x1=self.cx - xt, y1=self.cy + sh, x2=self.cx - x2, y2=self.cy + y2, stroke_width=self.stroke,
                      stroke=f'{self.rgb}', stroke_opacity=self.opacity)
        self.graph.elements.append(l6)

    def __init__(self, owner: Drawing, identity: str = 'hh', size: float = 81., stroke: float = 0.5, font_size: int = 9,
                 rgb='#000', op=0.):
        self.owner = owner
        self.id_ref = f'#{identity}'
        self.stroke = stroke
        self.rt3 = math.sqrt(3.)  # tan(60) == math.sqrt(3.)
        # a hexagon can be seen as made of six equilateral triangles (ht)
        # each of which has a side length which for the hex is the same as it's
        # big radius 'R' (or a)
        # The small radius r is the height 'h' of the equilateral triangles.
        # The height of a hex is 2r, and it's width is 2R
        # Here I define R as 81.
        self.font_size = font_size
        self.a = float(size)
        self.rgb = rgb
        self.opacity = op
        self.h = self.a * 0.5 * self.rt3  # height of any equilateral triangle.
        self.wid, self.hgt = 2. * self.a, self.h * 2.
        self.ah, self.hh = self.a * 0.5, self.h * 0.5
        self.aq, self.hq = self.a * 0.25, self.h * 0.25
        self.a3 = 1.5 * self.a
        self.cx, self.cy = self.a + self.stroke, self.h + self.stroke
        self.vx = self.wid + 2. * self.stroke
        self.vy = self.hgt + 2. * self.stroke
        self.ofx, self.ofy = self.owner.width * 0.5 - self.cx, self.owner.height * 0.5 - self.cy
        self.obj = svg.Symbol(
            id=identity,
            preserveAspectRatio=svg.PreserveAspectRatio('xMaxYMax'),
            viewBox=svg.ViewBoxSpec(0, 0, self.vx, self.vy),  # viewBox: <min-x>, <min-y>, <width> and <height>,
            elements=[]
        )
        self._draw()
        self.owner.define(self.obj)

    def place(self, where: list, label: bool):
        xi, yi = where
        x = self.ofx + xi * self.a3
        y = self.ofy + yi * self.hgt + self.h * (xi % 2)
        inst = svg.Use(href=self.id_ref, x=x, y=y, width=self.vx, height=self.vy)
        if label:
            gi = svg.G(elements=[])
            lab = f'{abs(xi):02x}{abs(yi):02x}'
            # lab = f'{abs(yi):04d}'
            lb = svg.Text(fill=f'{self.rgb}', opacity=self.opacity, stroke='none', text=lab, font_family='Courier',
                          text_anchor='middle', font_size=self.font_size, x=x + self.wid * 0.5, y=y + self.hgt - 1)
            gi.elements.append(inst)
            gi.elements.append(lb)
            self.owner.add(gi)
        else:
            self.owner.add(inst)


if __name__ == '__main__':
    # h = HexUtil
    # lc = [(0, 1), (1, 2), (2, 1), (-1, 1), (0, 0), (1, 1), (2, 0), (-1, 0), (0, -1), (1, 0), (2, -1), (1, -1)]
    # cc = [h.to_cube(i) for i in lc]
    # ux = [h.to_offs(i) for i in cc]
    # print(lc)
    # print(ux)
    # print(cc)
    canvas = Drawing('test2', (3200, 6400), True)
    # h1 = SvgHH(canvas, 'h1', 243, 1.2, font_size=10, rgb='#07d624', op=0.5)
    # h2 = SvgHH(canvas, 'h2', 81, 0.6, font_size=8, rgb='#fc0054', op=0.6)
    # h3 = SvgHH(canvas, 'h3', 27, 0.2, font_size=6, rgb='#5a4dff', op=0.7)
    h4 = SvgHH(canvas, 'h4', 9, 0.1, font_size=4, rgb='#000', op=0.8)
    # for i in range(3):
    #     for j in range(6):
    #         h1.place([i, j])
    # for i in range(9):
    #     for j in range(18):
    #         h2.place([i, j])
    # for i in range(27):
    #     for j in range(54):
    #         h3.place([i, j])
    for i in range(81):
        for j in range(162):
            h4.place([i, j], False)

    canvas.save()
