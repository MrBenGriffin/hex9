from numpy import ndarray
import json
import math


class Pixel:
    # This can convert between hex-pixels to square-pixels
    # and square+pixels to hex-pixels.
    def __init__(self):
        self.hr, self.sr = 112, 97
        with (open('assets/sq_to_hx_lut.json', 'r') as infile):
            str_hx_lut = json.load(infile)
            self.hx_lut = {int(i): {int(j): k for j, k in v.items()} for i, v in str_hx_lut.items()}
        with (open('assets/hx_to_sq_lut.json', 'r') as infile):
            str_sq_lut = json.load(infile)
            self.sq_lut = {int(i): {int(j): k for j, k in v.items()} for i, v in str_sq_lut.items()}

    def hex_col(self, img: ndarray, hx, hy) -> tuple:
        height, width = img.shape[:2]
        r, g, b = 0., 0., 0.
        sxo = self.sr * math.floor(hx / self.hr)
        rec = self.hx_lut[hx % self.hr]
        oyy, nv = [[hy, hy - 1], 2.] if hx & 1 == 1 else [[hy], 1.]
        for ofy in oyy:
            for ofx, bit in rec.items():
                x = max(min(width - 1, sxo + ofx), 0)
                y = max(min(height - 1, ofy), 0)
                rd, gd, bd = (img[y, x])
                b += bd * bit
                r += rd * bit
                g += gd * bit
        return r / nv, g / nv, b / nv

    def sq_col(self, img: ndarray, sx, sy) -> tuple:
        height, width = img.shape[:2]
        r, g, b = 0., 0., 0.
        hxo = self.hr * math.floor(sx / self.sr)
        rec = self.sq_lut[sx % self.sr]
        for ofx, bit in rec.items():
            x = max(min(width - 1, hxo + ofx), 0)
            oyy, nv = [[sy, sy + 1], 2.] if x & 1 == 1 else [[sy], 1.]
            for ofy in oyy:
                y = max(min(height - 1, ofy), 0)
                rd, gd, bd = (img[y, x])
                b += bd * bit / nv
                r += rd * bit / nv
                g += gd * bit / nv
        return r, g, b

