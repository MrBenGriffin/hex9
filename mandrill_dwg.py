import math
import numpy as np
from pixel import SQ2TRPixel, TR2H9Pixel
from photo import Photo
from h9 import H9Grid
from drawing import Drawing


def convert(src, result, lut):
    h, w = result.shape[:2]
    for y in range(h):
        for x in range(w):
            result[y, x] = lut.col(src, x, y)


if __name__ == '__main__':
    p = Photo()
    pt = Photo()
    p.load('mandrill_64', False)
    sqw, sqh = p.width, p.height

    s2t = SQ2TRPixel()
    t_adj = np.sqrt(3.)
    trw, trh = int(np.ceil(sqw * t_adj)), sqh
    pt.new(trw, trh)
    convert(p.img, pt.img, s2t)

    t2h9 = TR2H9Pixel()
    ptw, pth, radius = int(math.ceil(trw / 9)), int(math.ceil(trh / 6)), 27
    dim = H9Grid.size_for(ptw, pth, radius)
    cs = Drawing('mandrill_64', dim, False)
    h0 = H9Grid(cs, 'h1', radius, 0, op=1.0)
    h0.hierarchy = 4
    rw, rh, (oxf, oyf) = h0.get_limits()
    btz = oxf & 1 == 1
    for wx in rw:      # range(hhw):
        for wy in rh:  # range(hhh):
            cx = t2h9.cols(pt.img, wx+oxf-4, wy+oyf-3,  btz, True)
            cols = [f'#{int(r):02X}{int(g):02X}{int(b):02X}' for (r, g, b) in cx]
            h0.place([wx, wy, 0], cols, False)
    cs.save()
