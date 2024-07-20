import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import numpy as np
import json
import math
import cv2

# This can convert between hex-pixels to square-pixels
# and square+pixels to hex-pixels.

def hex_colour(img, lut, hx, hy) -> tuple:
    height, width = img.shape[:2]
    r, g, b = 0., 0., 0.
    hr, sr = 112, 97  # repeat at x, with sq offset.  15, 13 is also good.
    sxo = sr * math.floor(hx / hr)
    rec = lut[hx % hr]
    oyy, nv = [[hy, hy+1], 2.] if hx % 2 == 0 else [[hy], 1.]
    for ofy in oyy:
        for ofx, bit in rec.items():
            x = max(min(width - 1, sxo + ofx), 0)
            y = max(min(height - 1, ofy), 0)
            rd, gd, bd = (img[y, x])
            b += bd * bit
            r += rd * bit
            g += gd * bit
    return r / nv, g / nv, b / nv


def sq_colour(img, lut, sx, sy) -> tuple:
    height, width = img.shape[:2]
    r, g, b = 0., 0., 0.
    hr, sr = 112, 97  # repeat at x, with sq offset.  15, 13 is also good.
    hxo = hr * math.floor(sx / sr)
    rec = lut[sx % sr]
    for ofx, bit in rec.items():
        x = max(min(width - 1, hxo + ofx), 0)
        oyy, nv = [[sy, sy + 1], 2.] if x % 2 == 1 else [[sy], 1.]
        for ofy in oyy:
            y = max(min(height - 1, ofy), 0)
            rd, gd, bd = (img[y, x])
            b += bd * bit / nv
            r += rd * bit / nv
            g += gd * bit / nv
    return r, g, b


def sq_draw(img):
    sq_h, sq_w = img.shape[:2]
    rx = np.radians(45.)
    plt.axis('off')
    fig, ax = plt.subplots(figsize=(25, 38), dpi=150, layout='tight', frameon=False)
    ax.set_aspect('equal')
    ax.set_xlim([-0.5, sq_w + 0.5])
    ax.set_ylim([-0.5, sq_h + 0.5])
    ax.set_axis_off()
    axs = plt.gca()
    axs.invert_yaxis()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    for py in range(sq_h):
        for px in range(sq_w):
            bf, gf, rf = img[py, px]
            r, g, b = int(math.floor(rf)), int(math.floor(gf)), int(math.floor(bf))
            c = f'#{r:02X}{g:02X}{b:02X}'
            xp = px
            yp = py
            if len(c) == 7:
                hx = RegularPolygon((xp, yp), orientation=rx, numVertices=4, radius=0.717, linewidth=None, facecolor=c, alpha=None, edgecolor=None, aa=True)
                ax.add_patch(hx)
            else:
                print(px, py, c)
    plt.show()


def hex_draw(img):
    hx_h, hx_w = img.shape[:2]
    rx = np.radians(30.)
    xm = np.sqrt(3.) * 0.5  # x multiplier for cartesian x .
    plt.axis('off')
    fig, ax = plt.subplots(figsize=(25, 38), dpi=150, layout='tight', frameon=False)
    ax.set_aspect('equal')
    ax.set_xlim([-0.5, hx_w * xm + 0.5])
    ax.set_ylim([-0.5, hx_h + 1.0])
    ax.set_axis_off()
    axs = plt.gca()
    axs.invert_yaxis()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    for py in range(hxh):
        for px in range(hxw):
            b, g, r = img[py, px]
            c = f'#{int(r):02X}{int(g):02X}{int(b):02X}'
            xp = px * xm
            yp = py if px % 2 == 0 else py - 0.5
            hx = RegularPolygon((xp, yp), numVertices=6, radius=0.568, linewidth=None,
                                orientation=rx, facecolor=c, alpha=None, edgecolor=None, aa=True)
            ax.add_patch(hx)
    plt.show()


if __name__ == '__main__':
    # load and set the lookup table
    with (open('assets/sq_to_hx_lut.json', 'r') as infile):
        str_hx_lut = json.load(infile)
        hx_lut = {int(i): {int(j): k for j, k in v.items()} for i, v in str_hx_lut.items()}
    with (open('assets/hx_to_sq_lut.json', 'r') as infile):
        str_sq_lut = json.load(infile)
        sq_lut = {int(i): {int(j): k for j, k in v.items()} for i, v in str_sq_lut.items()}
    src_img = cv2.imread(f'assets/mandrill_128.png')  # 382 x256
    sqh, sqw = src_img.shape[:2]
    sq_draw(src_img)    # need to use BGR for native opencv.
    adj = 2. / np.sqrt(3)
    hxh, hxw = sqh, int(np.ceil(sqw * adj))
    hex_img = np.zeros([hxh, hxw, 3])
    for y in range(hxh):
        for x in range(hxw):
            hex_img[y, x] = hex_colour(src_img, hx_lut, x, y)
    hex_draw(hex_img)
    # let's convert it back to squares.
    sq_img = np.zeros([sqh, sqw, 3])
    for y in range(sqh):
        for x in range(sqw):
            sq_img[y, x] = sq_colour(hex_img, sq_lut, x, y)
    sq_draw(sq_img)


