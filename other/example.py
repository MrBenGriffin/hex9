import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import numpy as np
import json
import math
import cv2


def hex_colour(img, lut, hr, sr, hx, hy) -> tuple:
    height, width = img.shape[:2]
    r, g, b = 0., 0., 0.
    # hr, sr = 112, 97  # repeat at x, with sq offset.  15, 13 is also good.
    sxo = sr * math.floor(hx / hr)
    rec = lut[hx % hr]
    oyy, nv = [[hy, hy+1], 2.] if hx % 2 == 0 else [[hy], 1.]
    for ofy in oyy:
        for ofx, bit in rec.items():
            x = max(min(width - 1, sxo + ofx), 0)
            y = max(min(height - 1, ofy), 0)
            bd, gd, rd = (img[y, x])
            b += bd * bit
            r += rd * bit
            g += gd * bit
    return r / nv, g / nv, b / nv


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
            r, g, b = img[py, px]
            c = f'#{int(r):02X}{int(g):02X}{int(b):02X}'
            xp = px * xm
            yp = py if px % 2 == 0 else py - 0.5
            hx = RegularPolygon((xp, yp), numVertices=6, radius=0.568, linewidth=None,
                                orientation=rx, facecolor=c, alpha=None, edgecolor=None, aa=True)
            ax.add_patch(hx)
    plt.show()


if __name__ == '__main__':
    # load and set the lookup table
    with (open('../assets/sq_hx_lut.json', 'r') as infile):
        data = json.load(infile)
        i_w = data['i_mod']
        o_w = data['o_mod']
        h_lut = {int(i): {int(j): k for j, k in v.items()} for i, v in data["lut"].items()}
    # read the image to convert
    src_img = cv2.imread(f'../assets/mandrill_64.png')  # 382 x256
    sqh, sqw = src_img.shape[:2]
    # convert the image to hexagonal pixel grid.
    adj = 2. / np.sqrt(3)
    hxh, hxw = sqh, int(np.ceil(sqw * adj))
    hex_img = np.zeros([hxh, hxw, 3])
    for y in range(hxh):
        for x in range(hxw):
            hex_img[y, x] = hex_colour(src_img, h_lut, o_w, i_w, x, y)
    # It's done, so let's draw it.
    hex_draw(hex_img)
