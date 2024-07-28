import numpy as np
from pixel import SQ2HXPixel, HX2SQPixel, SQ2TRPixel, TR2SQPixel, TR2H6Pixel
from photo import Photo
import matplotlib as mpl
from h9 import H9


def convert(src, result, lut):
    h, w = result.shape[:2]
    for y in range(h):
        for x in range(w):
            result[y, x] = lut.col(src, x, y)


def plot_image(plotter, img):
    hh, hw = img.shape[:2]
    for hy in range(hh):
        for hx in range(hw):
            rgb = img[hy, hx]
            plotter.plot([hx, hy], rgb)
    plotter.show()


if __name__ == '__main__':
    h2s, s2h, t2s, s2t, t2h6 = HX2SQPixel(), SQ2HXPixel(), TR2SQPixel(), SQ2TRPixel(), TR2H6Pixel()
    p = Photo()
    ph = Photo()
    pt = Photo()
    ps1 = Photo()
    ps2 = Photo()
    phh = Photo()
    p.load('ml_256', False)
    # p.show('mandrill original')
    sqw, sqh = p.width, p.height
    h_adj = 2. / np.sqrt(3.)
    hxw, hxh = int(np.ceil(sqw * h_adj)), sqh
    ph.new(hxw, hxh)
    convert(p.img, ph.img, s2h)
    # ph.show('mandrill hx')

    t_adj = np.sqrt(3.)
    trw, trh = int(np.ceil(sqw * t_adj)), sqh
    pt.new(trw, trh)
    convert(p.img, pt.img, s2t)
    # pt.show('mandrill tr')

    # h6 image.
    hhw, hhh = int(trw/2), int(sqh/2)
    pw, ph = phh.h6_size(hhw+1, hhh+2, 9)
    phh.new(pw, ph)
    for y in range(hhh+1):
        for x in range(-1, hhw+1):
            cols = t2h6.cols(pt.img, x, y, True)
            phh.h6([x, y], cols)

    # convert(pt.img, phh.img, t2h6)
    # phh.show('mandrill tr->h6', pause=True)
    phh.save('mona_h6')

    # pt0 = Photo()
    pt0 = phh
    pt1 = Photo()
    # z = [(255, 255, 255), (0, 0, 0)]
    # pt0.new(200, 200)
    # for i in range(200):
    #     for j in range(200):
    #         pt0.img[j, i] = z[0] if (i + j) % 2 == 0 else z[1]
    # pw, ph = H9.size_for(hhw/5.0, hhh/4.5, 60)
    # pt1.new(pw, ph)  # pw/ph is in photo-pixels.
    # pt1.set_h9(60)
    # rw, rh, (oxf, oyf) = pt1.h9_get_limits()  # should be small
    # ox, oy = -rw[0], -rh[0]
    # for x in rw:  # range(hhw):
    #     for y in rh:  # range(hhh):
    #         h8i = H9.h92hh((x + ox, y + oy))
    #         cx = [pt0.at(xi, yi) for (xi, yi) in h8i]
    #         pt1.h9([x, y], cx)
    # # pt0.show('cols')
    # pt1.show('mandrill tr->hh8->h6', pause=True)

    # pt1.img[0, 0] = (0, 128, 255)
    # h9w, h9h = 1, 1
    # ph9 = Photo()
    # h9w, h9h = int(hhw / 5), int(hhh / 4)
    # pw, ph = ph9.h9_size(h9w, h9h, 27)
    # ph9.new(pw+6, ph + 6)  # 1720
    # ph9.set_h9(27)
    # rw, rh, (oxf, oyf) = ph9.h9_get_limits()
    # ox, oy = -rw[0], -rh[0]
    # for x in rw:  # range(hhw):
    #     for y in rh:  # range(hhh):
    #         h8i = H9.hh8((x+ox, y+oy))
    #         cx = [phh.at(xi, yi) for (xi, yi) in h8i]
    #         ph9.h9([x, y], cx)
    #
    # ph9.show('mandrill hh8->h9', pause=True)



    # ps1.new(sqh, sqw)
    # convert(ph.img, ps1.img, h2s)
    # ps1.show('mandrill h->sq')
    #
    # ps2.new(sqh, sqw)
    # convert(pt.img, ps2.img, t2s)
    # ps2.show('mandrill t->sq', pause=True)
    #
