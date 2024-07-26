import numpy as np
from pixel import SQ2HXPixel, HX2SQPixel, SQ2TRPixel, TR2SQPixel, TR2HH8Pixel
from photo import Photo


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
    h2s, s2h, t2s, s2t, t2hh = HX2SQPixel(), SQ2HXPixel(), TR2SQPixel(), SQ2TRPixel(), TR2HH8Pixel()
    p = Photo()
    ph = Photo()
    pt = Photo()
    ps1 = Photo()
    ps2 = Photo()
    phh = Photo()
    p.load('mandrill_512', False)
    p.show('mandrill original')
    sqw, sqh = p.width, p.height
    h_adj = 2. / np.sqrt(3.)
    hxw, hxh = int(np.ceil(sqw * h_adj)), sqh
    ph.new(hxw, hxh)
    convert(p.img, ph.img, s2h)
    ph.show('mandrill hx')

    t_adj = np.sqrt(3.)
    trw, trh = int(np.ceil(sqw * t_adj)), sqh
    pt.new(trw, trh)
    convert(p.img, pt.img, s2t)
    pt.show('mandrill tr')

    # hh8 image.
    hhw, hhh = int(trw/3), int(trh/2)
    phh.new(hhw, hhh)
    convert(pt.img, phh.img, t2hh)
    phh.show('mandrill tr->hh8')

    ps1.new(sqh, sqw)
    convert(ph.img, ps1.img, h2s)
    ps1.show('mandrill h->sq')

    ps2.new(sqh, sqw)
    convert(pt.img, ps2.img, t2s)
    ps2.show('mandrill t->sq', pause=True)

