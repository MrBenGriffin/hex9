import math
import numpy as np
from pixel import SQ2TRPixel, TR2H9Pixel
from photo import Photo
from h9 import H9Grid


def convert(src, result, lut):
    h, w = result.shape[:2]
    for y in range(h):
        for x in range(w):
            result[y, x] = lut.col(src, x, y)


def h9_test(radius=27, side=30):

    p = Photo()
    pt = Photo()
    p.load('world.topo.bathy.200406.3x5400x2700', False)
    sqw, sqh = p.width, p.height

    s2t = SQ2TRPixel()
    t_adj = np.sqrt(3.)
    trw, trh = int(np.ceil(sqw * t_adj)), sqh
    pt.new(trw, trh)
    convert(p.img, pt.img, s2t)

    t2h9 = TR2H9Pixel()
    pt1 = Photo()
    ptw, pth, radius = int(math.ceil(trw / 9)), int(math.ceil(trh / 6)), 9
    dw, dh = H9Grid.size_for(ptw, pth, radius)
    pt1.new(dw, dh)  # pw/ph is in photo-pixels.
    for wx in range(dw):  # stick on a green backdrop
        for wy in range(dh):
            pt1.img[wy, wx] = (0, 128, 0)

    pt1.set_h9(radius)
    rw, rh, (oxf, oyf) = pt1.h9_get_limits()  # should be small
    btz = oxf & 1 == 1
    for wx in rw:      # range(hhw):
        for wy in rh:  # range(hhh):
            cx = t2h9.cols(pt.img, wx+oxf, wy+oyf, btz, True)  # this is in jiggly hex. so wx is %2 sensitive.
            pt1.h9([wx, wy, 0], cx)
    pt1.save('bm_h9')
    # pt1.show('blue marble tr->h9', pause=True)


if __name__ == '__main__':
    h9_test()
