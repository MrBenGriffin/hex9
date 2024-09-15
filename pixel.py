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
                self.i_window = data['i_mod']
                self.o_window = data['o_mod']
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
        sxd, sxm = divmod(sx, self.o_window)
        sxo = self.i_window * sxd
        rec = self.lut[sxm]
        for ofx, bit in rec.items():
            x = max(min(width - 1, sxo + ofx), 0)
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
        sxd, sxm = divmod(sx, self.o_window)
        sxo = self.i_window * sxd
        rec = self.lut[sxm]
        for ofx, bit in rec.items():
            x = max(min(width - 1, sxo + ofx), 0)
            y = max(min(height - 1, sy), 0)
            rd, gd, bd = (img[y, x])
            r += rd * bit
            g += gd * bit
            b += bd * bit
        return r, g, b


class SQ2TRPixel(Pixel):
    # convert square pixels to triangle pixels.
    def __init__(self):
        super().__init__('assets/sq_tr_lut.json')

    def col(self, img: ndarray, tx, ty) -> tuple:
        height, width = img.shape[:2]
        r, g, b = 0., 0., 0.
        sxd, sxm = divmod(tx, self.o_window)
        sxo = self.i_window * sxd

        rec = self.lut[sxm]
        for ofx, bit in rec.items():
            x = max(min(width - 1, sxo + ofx), 0)
            y = max(min(height - 1, ty), 0)
            rd, gd, bd = (img[y, x])
            r += rd * bit
            g += gd * bit
            b += bd * bit
        return r, g, b


class TR2H6Pixel(Pixel):
    def __init__(self):
        super().__init__(None)
        self.nm = 1./3.
        self.lut = [
            [(0, 0), (0, 1), (1, 1)], [(1, 0), (2, 0), (2, 1)], [(3, 0), (4, 0), (5, 0)],  # line 0
            [(0, 2), (1, 2), (2, 2)], [(3, 2), (3, 1), (4, 1)], [(4, 2), (5, 2), (5, 1)],  # line 1
            [(0, 3), (1, 3), (2, 3)], [(3, 3), (3, 4), (4, 4)], [(4, 3), (5, 3), (5, 4)],  # line 2
            [(0, 4), (0, 5), (1, 4)], [(1, 5), (2, 4), (2, 5)], [(3, 5), (4, 5), (5, 5)]   # line 3
        ]

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

    def cols(self, img: ndarray, hx, hy, rev=False) -> list:
        # given a triangle px img, get h6 (3x6) half-hex colour group at h6 coordinates.
        # h6 rows are 3 to 6 tr rows (hh 0,1 use [0,1]/[1,2], hh 2,3 use [3,4]/[4,5]
        ty = 6 * hy
        tx = 6 * hx
        result = []
        h, w = img.shape[:2]
        for i in self.lut:
            r, g, b = 0., 0., 0.
            for (x, y) in i:
                ix = min(w-1, tx+x)
                iy = min(h-1, ty+y)
                rd, gd, bd = (img[iy, ix])
                r += rd * self.nm
                g += gd * self.nm
                b += bd * self.nm
            result.append((r, g, b) if not rev else (b, g, r))
        return result


class TR2H9Pixel(Pixel):
    def __init__(self):
        super().__init__(None)
        self.nm = 1./3.
        # for each region, work out the triangle pixel based on a patch 11 across and 6 down.
        # this is in district order. patch top left is above 7b (15).
        self.lut = [
            [(2, 3), (3, 3), (4, 3)],   # district 0a / 0
            [(2, 2), (3, 2), (4, 2)],   # district 0b / 1
            [(5, 2), (5, 1), (6, 1)],   # district 1a / 2
            [(6, 2), (7, 1), (7, 2)],   # district 1b / 3
            [(7, 4), (7, 3), (6, 3)],   # district 2a / 4
            [(5, 3), (5, 4), (6, 4)],   # district 2b / 5
            [(8, 3), (9, 3), (10, 3)],  # district 3a / 6
            [(8, 2), (9, 2), (10, 2)],  # district 3b / 7
            [(2, 4), (2, 5), (3, 4)],   # district 4a / 8
            [(3, 5), (4, 5), (4, 4)],   # district 4b / 9
            [(3, 0), (4, 0), (4, 1)],   # district 5a / 10
            [(2, 0), (2, 1), (3, 1)],   # district 5b / 11
            [(5, 0), (6, 0), (7, 0)],   # district 6a / 12
            [(5, 5), (6, 5), (7, 5)],   # district 6b / 13
            [(8, 5), (8, 4), (9, 4)],   # district 7a / 14
            [(0, 2), (1, 1), (1, 2)],   # district 7b / 15
            [(0, 3), (1, 3), (1, 4)],   # district 8a / 16
            [(8, 0), (8, 1), (9, 1)]    # district 8b / 17
        ]

    def cols(self, img: ndarray, hx, hy, xf=True, rev=False) -> list:
        # given an x, y (in h9 coords), get the colours from an img in triangle coordinates.
        # xf is used for offset adjustment to indicate that x-parity is off.
        parity = hx & 1 if xf else 1 - hx & 1
        tx = hx * 9
        ty = hy * 6 + parity * 3 - 2
        result = []
        h, w = img.shape[:2]
        for i in self.lut:
            r, g, b = 0., 0., 0.
            for (x, y) in i:
                ix = max(0, min(w-1, tx+x))
                iy = max(0, min(h-1, ty+y))
                rd, gd, bd = (img[iy, ix])
                r += rd * self.nm
                g += gd * self.nm
                b += bd * self.nm
            result.append((r, g, b) if not rev else (b, g, r))
        return result

    def set(self, img: ndarray, hx, hy, cols: list):
        tx = hx * 9
        ty = hy * 6 - (hx & 1) * 3
        h, w = img.shape[:2]
        for i, m in enumerate(self.lut):
            for (x, y) in m:
                ix = max(0, min(w-1, tx+x))
                iy = max(0, min(h-1, ty+y))
                r, g, b = cols[i]
                img[iy, ix] = b, g, r

