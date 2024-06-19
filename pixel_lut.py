import math
import numpy as np
import svg
import json
import cv2

# Todo - show both a pixel grid and a hex-grid.
# Calculate the contribution of each hex.px in a sqare.px.
# Calculate the contribution of each square.px in a hex.px


class Drawing:
    def __init__(self, wd, ht, hs, qs, hp_x, sp_x, ho, so):
        (
            self.hs, self.qs, self.hps, self.sps, self.hof, self.sof
        ) = hs, qs, hp_x, sp_x, ho, so
        self.w = wd * sp_x + 2. * self.hof
        self.h = ht * sp_x + 2. * self.sof
        self.canvas = svg.SVG(
            viewBox=svg.ViewBoxSpec(0, 0, self.w, self.h),
            width=int(self.w), height=int(self.h), elements=[]
        )

    def hex(self, pt):
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

    def sq(self, pt):
        # bottom left corner.
        xi, yi = pt
        s = self.qs
        x = self.sof + self.sps * xi
        y = self.sof + self.sps * yi
        tx = svg.Polygon(fill='none', stroke_width=0.1, stroke="black")
        tx.points = [(x, y), (x+s, y), (x+s, y+s), (x, y+s)]
        self.canvas.elements.append(tx)

    def save(self, name: str = 'px_lut_w'):
        f = open(f"{name}.svg", "w")
        f.write(self.canvas.as_str())
        f.close()


if __name__ == '__main__':
    # hex_height = math.sqrt(3.) * hex_side
    # hex_side = 10.
    hex_height = 10.
    hex_side = 10. / math.sqrt(3.)
    hx_px = hex_side * 1.5
    hx_off = hex_side * 0.5
    sq_side = hex_height
    sq_px = sq_side
    sq_off = hex_side * 0.5
    # dx = Drawing(84, 3, hex_side, sq_side, hx_px, sq_px, hx_off, sq_off)

    # hpx = np.arange(start=hx_off, stop=8000., step=hx_px)
    # spx = np.arange(start=sq_off, stop=8000., step=sq_px) 145.492267835785693
    print(f'sq_px: {sq_px}, hx_px: {hx_px}, r: {hex_side}')
    for i in range(100):
        si, hi = i * sq_px, i * hx_px  # si, hi are current square,hex px
        s = math.floor(hi / sq_px)  # divide by hi by sq_px.
        so = hi - s*sq_px
        h = math.floor(si / hx_px)
        ho = si - h*hx_px
        print(i, s, so, h, ho)
        # s = np.searchsorted(hpx, sa)
        # h = np.searchsorted(spx, ha)
        # print(f'[s{i} in h{s} a:{hpx[s]},{sa}]; [h{i} is in s{h}; s{h}:{spx[h]},h{i}:{ha}]')

    # for i in range(84):
    #     for j in range(1):
    #         dx.sq([i, j+1])
    # for i in range(97):
    #     if i % 2 == 1:
    #         dx.hex([i, 0])
    #     for j in range(1, 2):
    #         dx.hex([i, j])
    #
    # dx.save()
