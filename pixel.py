from numpy import ndarray
import json
import math


class Pixel:
    # This can convert between hex-pixels to square-pixels
    # and square+pixels to hex-pixels.
    def __init__(self, filename: str | None = None):
        if filename is not None:
            with (open(filename, 'r') as infile):
                data = json.load(infile)
                self.window = data['mod']
                lut = data['lut']
                self.lut = {int(i): {int(j): k for j, k in v.items()} for i, v in lut.items()}


class SQ2HXPixel(Pixel):
    def __init__(self):
        super().__init__('assets/sq_hx_lut.json')

    def col(self, img: ndarray, hx, hy) -> tuple:
        height, width = img.shape[:2]
        r, g, b = 0., 0., 0.
        sxo = self.window * math.floor(hx / self.window)
        rec = self.lut[hx % self.window]
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


class HX2SQPixel(Pixel):
    def __init__(self):
        super().__init__('assets/hx_sq_lut.json')

    def col(self, img: ndarray, sx, sy) -> tuple:
        height, width = img.shape[:2]
        r, g, b = 0., 0., 0.
        hxo = self.window * math.floor(sx / self.window)
        rec = self.lut[sx % self.window]
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


class TR2SQPixel(Pixel):
    def __init__(self):
        super().__init__('assets/tr_sq_lut.json')

    def col(self, img: ndarray, sx, sy) -> tuple:
        height, width = img.shape[:2]
        r, g, b = 0., 0., 0.
        sxo = self.window * math.floor(sx / self.window)
        rec = self.lut[sx % self.window]
        for ofx, bit in rec.items():
            x = max(min(width - 1, sxo + ofx), 0)
            y = max(min(height - 1, sy), 0)
            rd, gd, bd = (img[y, x])
            r += rd * bit
            g += gd * bit
            b += bd * bit
        return r, g, b


class SQ2TRPixel(Pixel):
    def __init__(self):
        super().__init__('assets/sq_tr_lut.json')

    def col(self, img: ndarray, hx, hy) -> tuple:
        height, width = img.shape[:2]
        r, g, b = 0., 0., 0.
        sxo = self.window * math.floor(hx / self.window)
        rec = self.lut[hx % self.window]
        for ofx, bit in rec.items():
            x = max(min(width - 1, sxo + ofx), 0)
            y = max(min(height - 1, hy), 0)
            rd, gd, bd = (img[y, x])
            r += rd * bit
            g += gd * bit
            b += bd * bit
        return r, g, b


class TR2HH8Pixel(Pixel):
    def __init__(self):
        super().__init__(None)
        self.nm = 1./3.
        self.lut = [[
            [(0, 0), (0, 1), (1, 1)], [(1, 0), (2, 0), (2, 1)], [(3, 0), (4, 0), (5, 0)],  # line 0
            [(0, 1), (1, 1), (2, 1)], [(3, 0), (4, 0), (3, 1)], [(5, 0), (5, 1), (4, 1)]   # line 1
            ], [
            [(0, 0), (1, 0), (2, 0)], [(3, 0), (3, 1), (4, 1)], [(4, 0), (5, 0), (5, 1)],  # line 2
            [(0, 0), (1, 0), (0, 1)], [(1, 1), (2, 0), (2, 1)], [(3, 1), (4, 1), (5, 1)]   # line 3
        ]]

        # tr to h9 6x6 tr = [2/3]x1 H9 (12 hh)
        # --------------
        # | 0 1 1 | 0 0 0 \
        # | 0 0 1 | 1 1 2  | ty % 1 == 0
        # | 2 2 2 | 1 2 2 /  define 6 hh px
        # --------------
        # | 0 0 0 | 0 1 1 \
        # | 1 1 2 | 0 0 1  | ty % 1 == 1
        # | 1 2 2 | 2 2 2 /  define 6 hh px.
        # ---------------

    def col(self, img: ndarray, hx, hy) -> tuple:
        # image size will be w/3,h/2
        # given a triangle px img, get hh12 pixel at hh12 coordinates.
        # hh12 rows are 4 to 6 tr rows (hh 0,1 use [0,1]/[1,2], hh 2,3 use [3,4]/[4,5]
        ty = math.floor(hy*1.5)   # [0, 1, 3, 4, 6, 7, 9, 10, 12..]
        tx = 3 * hx
        lut_y_offset = hy & 1   # [0  2  0  2  0  2  0   2   0..]
        lut_x_offset = hx % 3
        rec = self.lut[lut_y_offset][lut_x_offset]
        h, w = img.shape[:2]
        r, g, b = 0., 0., 0.
        for (x, y) in rec:
            ix = min(w-1, tx+x)
            iy = min(h-1, ty+y)
            rd, gd, bd = (img[iy, ix])
            r += rd * self.nm
            g += gd * self.nm
            b += bd * self.nm
        return r, g, b
