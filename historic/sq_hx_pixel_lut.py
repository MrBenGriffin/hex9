import math
import svg
import json

#This is the old one..
# It's now broken b/c I changed HalfHexFns for normalised..
# have a look at sq_hx_lut.py
from shape_functions import HalfHexFns

# Todo - show both a pixel grid and a hex-grid.
# Calculate the contribution of each hex.px in a square.px. √
# Calculate the contribution of each square.px in a hex.px


class Drawing:
    def __init__(self, wd, ht, hs, qs, hp_x, sp_x, ho, so):
        (
            self.hs, self.qs, self.hps, self.sps, self.hof, self.sof
        ) = hs, qs, hp_x, sp_x, ho, so
        self.w = wd * sp_x + 2. * self.hof
        self.h = ht * sp_x + 2. * self.sof
        self.t_o = hex_side / 25.
        self.t_s = hex_side / 8.
        self.canvas = svg.SVG(
            viewBox=svg.ViewBoxSpec(0, 0, self.w, self.h),
            width=int(self.w), height=int(self.h), elements=[]
        )
        self.bgd()

    def bgd(self):
        bg = svg.Polygon(fill='white', stroke_width=0.1, stroke="black")
        bg.points = [(1, 1), (1, self.h - 1), (self.w - 1, self.h - 1), (self.w - 1, 1), ]
        self.canvas.elements.append(bg)

    def hex(self, pt, id=None):
        # bottom left corner.
        # height = self.qs. hs=hex_side
        xi, yi = pt
        hs = self.hs                        # side width
        fh = self.qs                        # full height
        hh = self.qs * 0.5                  # half height
        ob = self.hs * 0.5                  # offset of hh points
        x = self.hof + self.hps * xi        # hps = 1.5 hs
        y = self.sof + self.sps * yi + (xi % 2) * hh

        tx = svg.Polygon(fill='none', stroke_width=0.1, stroke="black")
        # this is a good hex. and the height is good against the square! The basic offset is excellent.
        tx.points = [(x, y), (x-ob, y+hh), (x, y+fh), (x+hs, y+fh), (x+hs+ob, y+hh), (x+hs, y)]
        self.canvas.elements.append(tx)
        if id:
            idt = svg.Text(fill='black', stroke="none", text=f'h{id}', font_size=self.t_s, x=r - self.t_o*6 + x, y=y + self.sps - self.t_o)
            self.canvas.elements.append(idt)

    def v_mark(self, pt, label, v):
        xi, yi = pt
        vo = self.qs * v
        self.canvas.elements.append(svg.Line(stroke_width=0.1, stroke="red", x1=xi+vo, y1=yi, x2=xi+vo, y2=yi + self.qs))
        # mpv = svg.Polygon(stroke_width=0.2, stroke="green", fill="none")
        # mpv.points = [(xi+vo-rqt, yi), (xi+vo+rqt, yi), (xi+vo+rqt, yi+sq_px), (xi+vo-rqt, yi+sq_px)]
        # self.canvas.elements.append(mpv)

    def v_error(self, pt, label, v):
        xi, yi = pt
        vo = self.qs * v
        self.canvas.elements.append(
            svg.Line(stroke_width=0.5, stroke="red", x1=xi + vo, y1=yi, x2=xi + vo, y2=yi + self.qs))

    def sq(self, pt, id=None, extra=None):
        # bottom left corner.
        xi, yi = pt
        s = self.qs
        x = self.sof + self.sps * xi
        y = self.sof + self.sps * yi
        tx = svg.Polygon(fill='none', stroke_width=0.1, stroke="black")
        tx.points = [(x, y), (x+s, y), (x+s, y+s), (x, y+s)]
        self.canvas.elements.append(tx)
        txt_x = self.t_o*0.75 + x
        txt_y = self.t_o*3.5 + y
        if id:
            idt = svg.Text(fill='black', stroke="none", text=f's{id}', font_size=self.t_s, x=txt_x, y=txt_y)
            self.canvas.elements.append(idt)
        if extra:
            lines = extra
            for v, line in enumerate(lines):
                if isinstance(line, tuple):
                    self.v_mark((x, y), *line)
                    self.canvas.elements.append(
                        svg.Text(fill='black', stroke="none", text=f'{line[0]}: {line[1]:.3}', font_size=self.t_s, x=txt_x,
                                 y=txt_y + self.t_s * (v+1))
                    )
                else:
                    self.canvas.elements.append(
                        svg.Text(fill='black', stroke="none", text=line, font_size=self.t_s, x=txt_x,
                                 y=txt_y + self.t_s * (v+1))
                    )

    def save(self, name: str = 'px_lut_w'):
        f = open(f"{name}.svg", "w")
        f.write(self.canvas.as_str())
        f.close()


def best_fit():
    # The following shows that the best fit (at rank 3) is 0.006819127869120848: (h:112, s:97)
    rdx = {}
    dst = []
    for i in range(2000):
        si, hi = i * sq_px, i * hx_px  # si, hi are current square,hex px
        s = math.floor(hi / sq_px)  # divide by hi by sq_px.
        so = hi - s*sq_px
        h = math.floor(si / hx_px)
        ho = si - h*hx_px
        if (h % 2) == 0:
            ky = math.sqrt(ho * ho + so * so) / hex_height
            dst.append(ky)
            rdx[ky] = s, h, i
    dst.sort()
    for d in dst[:30]:
        print(f'{d}: {rdx[d]}')


def side_contribution(offset: float, left: bool, wax: bool) -> tuple:
    # Used if a square-pixel edge intersects the angled part of a hex
    # The (small radius is the length of every side of a hexagon: normally named lower case 'r')
    # Let's name the angled parts of the hexagon 'hex_side', which has a width of r/2.
    # Return: the left_contribution and the right_contribution 'lc' and 'rc' respectively in units.
    # ... (e.g contributing the entire pixel would be return sq_px) lc+rc are always = hex_side/2. -offset which ..
    # in units is sq_px/(2. * math.sqrt(3.):  (28.867513459481288 - offset)
    # We need to input the offset: measured from the beginning of the hex_side to the sq_pixel_edge.
    # Where 'beginning' is the start of the non-horizontal hex edges.
    # left True:  We want the contribution from the beginning (of the hex_side) to the offset.
    # left False: We want the contribution from the offset to the end (of the hex_side)
    # wax True: '<' hex_side;  False: '>' hex_side.
    # contribution colours are always in order. (For wax, the mc is on the left, for wane, mc is on the right)
    # We need the 0.5 height of the hexagon pixel. We need the √3 tan(60). We need the r/2 measurement.
    # rt3 = math.sqrt(3.)  already defined.
    # rhf = hex_side / 2.  already defined.
    hhf = hex_height * 0.5  # hhf= 0.5 height of the hexagon pixel.
    if offset >= rhf:
        c1 = rqt if left else 0
        c2 = offset-rqt if left else rqt
    else:
        if left:
            px = offset         # px offset value for the right_hand part of the hex_side.
        else:
            px = rhf - offset   # px offset value for the right_hand part of the hex_side
        py = rt3 * px       # the height of the triangle to split between left and right.
        ta = 0.5 * px * py  # area of triangle that is carved.
        ra = px * hhf       # The (rectangular) area that we want to divvy up.
        tx = ta / ra        # the proportion of the triangle to the full area.
        c1 = (tx * px)      # the triangle part of the contribution
        c2 = (px - c1)      # the remainder of the full rectangle.
        # is this returned as a normalised contribution, or as a part of the hex_side?
    return (c1, c2) if left != wax else (c2, c1)


def calc_m0m(o: float) -> list:
    m0, o0, m1 = 0., r, 0.  #
    if o > 0:
        n2 = (o + 2. * r) - sq_px
        flag = '+'
        m0 += o+rqt
        o0 += rqt
        a, b = side_contribution(n2, True, False)
        m1 += a
        o0 += b
    else:
        n2 = sq_px - (o + hx_px)
        flag = '-'
        a, b = side_contribution(-o, False, True)
        m0 += a
        o0 += b
        c, d = side_contribution(n2, True, False)
        o0 += c
        m1 += d

    m0 = m0/sq_px
    h0 = o0/sq_px
    m1 = m1/sq_px
    dfx = 1. - (m0 + h0 + m1)
    if dfx > 0.000001 or m0 < 0. or h0 < 0. or m1 < 0. or h0 > 1:
        print('error')
    # return [f'mhm{flag}', f'offs:{o:.5}', f'm0:{m0:.5}', f'h0:{h0:.5}', f'm1:{m1:.5}', ('m-', m0), ('h-', m0+h0)]  # ('m1', m0+h0+m1)
    return [m0, h0, m1]  # 'mhm'


def calc_0m1(o: float) -> list:
    if o+rqt < 0:
        a, b = side_contribution(rhf+o, True, False)
        z0 = a
        mm = b + r + rqt
        z1 = rqt + sq_px-o-r*2  # add remaining to z1.
    else:
        a, b = side_contribution(sq_px - (o+rhf+r), True, True)
        z0 = o+rqt
        mm = rqt + r + a
        z1 = b
    h0 = z0/sq_px
    m0 = mm/sq_px
    h1 = z1/sq_px
    dfx = 1. - (m0 + h0 + h1)
    assert dfx < 0.00001
    return [h0, m0, h1]  # 'hmh'
    # return [f'h0m0h1', f'offs:{o:.5}', f'r*2:{2. * r:.6}', (f'h0', h0), (f'm0', h0+m0)]  # (f'h1', h0+m0+h1)


def calc_m1(o: float) -> list:
    z = o+rqt
    m = sq_px - z
    h0 = m/sq_px
    m0 = z/sq_px
    dfx = 1. - (m0 + h0)
    assert dfx < 0.00001
    return [m0, h0]  # 'mh'
    # return [f'm0h0', (f'm0', m0)]  # ('h0', m0+h0)


def calc_0m(o: float) -> list:
    z = o+rqt
    m = sq_px - z
    h0 = z/sq_px
    m0 = m/sq_px
    dfx = 1. - (m0 + h0)
    assert dfx < 0.00001
    return [h0, m0]  #'hm'


def st_off(hex_count: int) -> dict:
    sq_table = {}  # hex to square table. For square index i, the hex I am in and how far in I am.
    for hex_idx in range(hex_count+2):
        si, hi = hex_idx * sq_px, hex_idx * hx_px           # si, hi are the square,hex at 'i'
        s = math.floor(hi / sq_px)              # divide by hi by sq_px.
        so = hi - s*sq_px
        s_in_hex, in_h_off = hex_idx-1, hx_px-so
        in_h_toff = in_h_off + rhf   # Offset from left tip of in_hex.
        if in_h_off >= 0.:
            sq_table[s] = [s_in_hex, in_h_toff]
    return sq_table


def do_squares(hexes: int) -> dict:
    # hx_px = hex_side * 1.5 = column width.
    adj = rt3_2
    hhf = HalfHexFns(hex_side)
    lut = st_off(hexes)
    sqd = {}
    for sq, (hidx, sq_x) in lut.items():
        dx = 0-sq_x
        dv = []
        while dx < sq_px:
            dv.append(dx)
            dx += hx_px
        hx_offs = {hidx+i: dv for i, dv in enumerate(dv)}
        sqd[sq] = {}  # a list of hexes, including the one I am in, and their offset from me.
        for idx, offset in hx_offs.items():
            if offset < 0:
                if offset+sq_px > hx_px:
                    fx = hhf.areas([-offset, -offset+sq_px])
                    c = fx[1]
                    # print('eep')
                    # _, c = hhf.area_props_at_x(-offset)
                else:
                    _, c = hhf.areas_at_x(-offset)
                    c /= hhf.a
            else:
                c, _ = hhf.areas_at_x(sq_px - offset)
                c /= hhf.a
            sqd[sq][idx] = c * rt3_2
    for k, i in sqd.items():
        x = sum(i.values())
        if abs(x-1) > 0.00001:
            print(f'line {k} sum is:{x}.  Needs normalising.')
    with open('output/hx_to_sq_lut.json', 'w', encoding='utf-8') as f:
        json.dump(sqd, f, ensure_ascii=False, indent=4)
    return sqd


def fill_st(hex_count: int) -> dict:
    sq_table = {}  # hex to square table. For square index i, the entry hex and the hex contribution.
    # ht = {0: [0, 100.]}  # square to hex table. For hex index i, return the entry square and the square(s) contribution.
    for hex_idx in range(hex_count):
        si, hi = hex_idx * sq_px, hex_idx * hx_px   # si, hi are the square,hex at 'i'
        s = math.floor(hi / sq_px)                  # divide by hi by sq_px.
        so = hi - s*sq_px
        s_in_hex, in_h_off = hex_idx-1, hx_px-so
        in_h_toff = in_h_off + rhf   # Offset from left tip of in_hex.
        s_nx_hex, nx_h_off = hex_idx, so
        sr6 = math.floor(in_h_toff / r6th)
        if in_h_off >= 0.:
            hx_kind = s_in_hex % 2 == 0
            if sr6 < 7. or (sr6 == 7. and sq_px < nx_h_off+hex_side):
                cx = calc_0m(r-in_h_off) if hx_kind else calc_m1(r-in_h_off)
            else:
                cx = calc_0m1(r-in_h_off) if hx_kind else calc_m0m(r-in_h_off)
            sq_table[s] = {s_in_hex+i: v for i, v in enumerate(cx)}
    with open('output/old_hx_to_sq_lut.json', 'w', encoding='utf-8') as f:
        json.dump(sq_table, f, ensure_ascii=False, indent=4)
    return sq_table


def do_hexes(hexes: int, hx_side: float):
    lut = st_off(hexes)
    hx = {}
    for sq in lut.keys():
        hpt = lut[sq]
        hex_id, h_off = hpt[0], hpt[1]
        if hex_id not in hx:
            if h_off > sq_side:
                hx[hex_id] = {sq-1: h_off-sq_side}
                if h_off > hx_px - rhf:
                    hx[hex_id + 1] = {sq: h_off - sq_side}
                hx[hex_id][sq] = h_off
            else:
                hx[hex_id] = {sq: h_off}
        else:
            if h_off > sq_side:
                hx[hex_id][sq-1] = h_off-sq_side
                if h_off > hx_px - rhf:
                    hx[hex_id + 1] = {sq: h_off - sq_side}
            hx[hex_id][sq] = h_off
    # we now have a dict of hex ids, each of which have a dict of squares ids with offset that it comes in.
    h_final = {}
    hhf = HalfHexFns(hx_side)
    for hx, sd in hx.items():
        sq = next(iter(sd)) - 1  # first square value.
        rx = hhf.areas(list(sd.values()))
        h_final[hx] = {i+sq: p for i, p in enumerate(rx)}
    with open('sq_to_hx_lut.json', 'w', encoding='utf-8') as f:
        json.dump(h_final, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    # hex_height = math.sqrt(3.) * hex_side
    # hex_side = 10.
    rt3 = math.sqrt(3.)  # same as tan(60)
    rt3_2 = rt3 / 2.     # used everywhere.
    h_count = 114
    hex_height = 100.
    hex_side = hex_height / math.sqrt(3.)
    r = hex_side
    rhf = hex_side / 2.  # half_short radius / side
    rqt = hex_side / 4.  # quarter radius / side
    r6th = rhf / 3.      # short radius of a hexagon is normally called r.
    hx_px = hex_side * 1.5
    hx_off = hex_side * 0.5
    sq_side = hex_height
    sq_px = sq_side
    sq_off = hex_side * 0.5
    yy = sq_px - hx_px
    st = do_squares(112)
    print(st)
    # s_count = int(hx_px/sq_px * (h_count+1))
    # dx = Drawing(s_count, 3, hex_side, sq_side, hx_px, sq_px, hx_off, sq_off)
    # # print(f'sq_px: {sq_px}, hx_px: {hx_px}, yy:{yy}, r: {hex_side}, r_half:{rhf} r6th:{r6th}')
    # # st = fill_st(h_count)
    # do_hexes(256, hex_side)

    # fo = hx_px/sq_px
    # for i in range(h_count):
    #     if i % 2 == 1:
    #         dx.hex([i, 0], f'{i},0')
    #     for j in range(1, 2):
    #         dx.hex([i, j], f'{i},{j}')
    # for i in range(s_count):
    #     for j in range(1):
    #         dx.sq([i, j+1], f'{i}', st[i])
    # dx.save()
