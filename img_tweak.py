import numpy as np
import math
from pixel import SQ2HXPixel, HX2SQPixel
from photo import Photo
from drawing import H2
from plot import HexPlotter, SqrPlotter


def hexify(ar, pxl, ph, hp, name: str, h_size: float):
    hw, hh = math.ceil(ph.width*adj), ph.height
    dsize = hw * h_size * 1.5 + h_size, hh * h_size * 1.75
    dhx = H2(name, dsize, False, h_size)
    for hy in range(hh):
        for hx in range(hw):
            rgb = pxl.col(ph.img, hx, hy)
            ar[hy, hx] = rgb
            dhx.hex([hx, hy], rgb)
            hp.plot([hx, hy], rgb)
    dhx.save()


def hex_to_square(sqp, pxl, hex_img):
    hh, hw = hex_img.shape[:2]
    for hy in range(hh):
        for hx in range(hw):
            rgb = pxl.col(hex_img, hx, hy)
            sqp.plot([hx, hy], rgb)
    sqp.show()


def do_globe():
    ph = Photo()
    big = True
    if big:
        img_s = ['north_hemi', 'south_hemi']
        ll = [(-90., 90.), (-180., 180.)]
    # img_s = [f'assets/world.topo.bathy.200407.3x21600x21600.{i}.png' for i in ['A2', 'B2', 'C2', 'D2']]
    # ll = [(-90., 0.), (-180., 180.)]

    # img_s = [f'assets/world.topo.bathy.200407.3x21600x21600.{i}.png' for i in ['A1', 'B1', 'C1', 'D1']]
        # ll = [(0., 90.), (-180., 180.)]
    else:
        img_s = ['assets/world.topo.bathy.200406.3x5400x2700']
        ll = [(-90., 90.), (-180., 180.)]
    ph.load(img_s)
    ph.set_latlon(img_s, *ll)
    # p.draw_grid((255, 255, 255), 2, lat_deg=7.5, lon_deg=7.5)
    ph.save(f'huge_world')


if __name__ == '__main__':
    adj = 2. / np.sqrt(3)
    p = Photo()
    r = Photo()
    h2s, s2h = HX2SQPixel(), SQ2HXPixel()
    p.load('tn', True)  # convert
    hx_dim = p.width, p.height
    hw = math.ceil(p.width*adj)
    hxp = HexPlotter(hx_dim, 0.5)
    sqp = SqrPlotter(hx_dim, 0.5)
    hex_i = np.zeros([p.height, hw,  3])
    hexify(hex_i, s2h, p, hxp, 'res_256', 81.)
    hxp.show()
    hex_to_square(sqp, h2s, hex_i)
    r.adopt(hex_i, True)
    r.show('hex', True)
    done = True


