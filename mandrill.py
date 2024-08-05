import math

import numpy as np
from pixel import SQ2TRPixel, TR2H9Pixel
from photo import Photo
from h9 import H9


def convert(src, result, lut):
    h, w = result.shape[:2]
    for y in range(h):
        for x in range(w):
            result[y, x] = lut.col(src, x, y)


def patch_convert(src, result, lut, dim_x, dim_y):
    # dim shows size of patch. eg 6x6
    hd, wd = result.shape[:2]
    h = math.floor(hd / dim_y)
    w = math.floor(wd / dim_x)
    for y in range(h):
        for x in range(w):
            patch = lut.cols(src, x, y)
            for yi in range(dim_y):
                for xi in range(dim_x):
                    result[y*dim_y+yi, x*dim_x+xi] = patch[x, y]


def plot_image(plotter, img):
    hh, hw = img.shape[:2]
    for hy in range(hh):
        for hx in range(hw):
            rgb = img[hy, hx]
            plotter.plot([hx, hy], rgb)
    plotter.show()


if __name__ == '__main__':
    p = Photo()
    pt = Photo()
    p.load('mona_422', False)
    p.show('mandrill original')
    sqw, sqh = p.width, p.height
    # h_adj = 2. / np.sqrt(3.)
    # hxw, hxh = int(np.ceil(sqw * h_adj)), sqh
    # ph.new(hxw, hxh)
    # convert(p.img, ph.img, s2h)
    # # ph.show('mandrill hx')

    s2t = SQ2TRPixel()
    t_adj = np.sqrt(3.)
    trw, trh = int(np.ceil(sqw * t_adj)), sqh
    pt.new(trw, trh)
    convert(p.img, pt.img, s2t)
    pt.show('mandrill tr')

    # # h6 single.
    # pw, ph = phh.h6_size(2, 1, 81)
    # phh.new(pw, ph)
    # phh.set_h6(81)
    # for x in [-1, 0]:
    #     cols = t2h6.cols(pt.img, 9+x, 4, True)
    #     phh.h6([x+1, 0], cols)
    # phh.save('tnc_h6')
    # phh.show('mandrill tr')

    # ph2 = Photo()
    # hhw, hhh = int(trw/2), int(sqh/2)
    # pw, ph = ph2.h6_size(hhw+1, hhh+2, 9)
    # ph2.new(pw, ph)
    # phh.set_h6(9)
    # for y in range(hhh+1):
    #     for x in range(-1, hhw+1):
    #         cols = t2h6.cols(pt.img, x, y, True)
    #         ph2.h6([x, y], cols)
    # ph2.save('tn_h6')
    # ph2.show('mandrill tr->h6')

    # import matplotlib as mpl
    # cmap0 = mpl.colormaps['plasma'].resampled(18)
    # cols0 = [tuple([int(c * 255.) for c in mpl.colors.to_rgb(cmap0(i))]) for i in range(18)]
    # cols0[0] = (255, 255, 255)

    t2h9 = TR2H9Pixel()
    pt0 = pt
    pt1 = Photo()
    ptw, pth, radius = int(math.ceil(trw / 9)), int(math.ceil(trh / 6)), 27
    dw, dh = H9.size_for(ptw, pth, radius)
    pt1.new(dw, dh)  # pw/ph is in photo-pixels.
    for wx in range(dw):
        for wy in range(dh):
            pt1.img[wy, wx] = (0, 255, 0)

    pt1.set_h9(radius)
    rw, rh, (oxf, oyf) = pt1.h9_get_limits()  # should be small
    btz = oxf & 1 == 1
    for wx in rw:      # range(hhw):
        for wy in rh:  # range(hhh):
            cx = t2h9.cols(pt.img, wx+oxf, wy+oyf, btz, True)  # this is in jiggly hex. so wx is %2 sensitive.
            pt1.h9([wx, wy, 0], cx)
    # pt1.save('tr_h9')
    pt1.show('mandrill tr->h9', pause=True)
