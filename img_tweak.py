import numpy as np
import math
from pixel import Pixel
from photo import Photo
from drawing import H2
from plot import HexPlot
import matplotlib.pyplot as plt


def hexify(pxl, ph, hp, name: str, h_size: float):
    hw, hh = math.ceil(ph.width*adj), ph.height
    dsize = hw * h_size * 1.5, hh * h_size * 1.75
    dhx = H2(name, dsize, False, h_size)
    for hy in range(hh):
        for hx in range(hw):
            rgb = pxl.hex_col(ph.img, hx, hy)
            dhx.hex([hx, hy], rgb)
            hp.hex([hx, hy], rgb)
    dhx.save()


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
    px = Pixel()
    p.load('tn')
    hx_dim = p.width, p.height
    hxp = HexPlot(hx_dim)
    hexify(px, p, hxp, 'res_256', 81.)
    plt.tight_layout()
    plt.show()
