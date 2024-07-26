import math
import json
import numpy as np
from shape_functions import HalfHexFns, ShapeFunctions

'''
This manufactures the LUTs I use for pixel conversion, however I am far from convinced that I still have this right.
The main problem is to do with hexing squares (I'm pretty confident that squaring hexes is done).
This is because hexing squares involves calculating the area of impact of the square onto the hex.
'''

def best_fit(sq_px, hx_px):
    # The following shows that a good fit (at rank 3) is 0.006819127869120848: (h:112, s:97)
    # however, top rank is 0.0004895913533781656: (h:1560, s:1351), ranked 3 at 20,000 (when 112/97 is not-ranked)
    # but 21728, 18817 is even better, and in the top ten for 1,000,000
    rdx = {}
    dst = []
    for i in range(200000):
        si, hi = i * sq_px, i * hx_px  # si, hi are current square,hex px
        s = math.floor(hi / sq_px)  # divide by hi by sq_px.
        so = hi - s*sq_px
        h = math.floor(si / hx_px)
        ho = si - h*hx_px
        if (h % 2) == 0:
            ky = math.sqrt(ho * ho + so * so) / 100.0
            dst.append(ky)
            rdx[ky] = s, h, i
    dst.sort()
    for d in dst[:30]:
        print(f'{d}: {rdx[d]}')


def s_calc(fn, hx, h_val, sbase):  # given a square, calculate the hexagon contributions to it.
    result = {}
    h_id, hbase, adj = int(hx), h_val, 0
    if sbase - hbase < fn.q1:
        hbase -= hx_px
        h_id -= 1
        adj = -1
    for i in range(4):
        h_o = hbase + hx_px * i  # hex-origin
        sl = sbase - h_o
        sr = sl + sq_px
        if sr < 0:
            return result
        if sl < 0:
            contr = fn.areas([sr])
            result[h_id + i] = contr[0] * fn.r3_2
            continue
        contr = fn.areas([sl, sr])
        ctx = 1 if i+adj < 1 else 0
        if contr[ctx] == 0:
            continue
        result[h_id + i] = contr[ctx] * fn.r3_2
    return result


if __name__ == '__main__':

    rt3 = math.sqrt(3.)
    # a3 = 2.0/rt3
    # b3 = rt3/2.0
    # the common part is the height. This really won't survive the local code
    # but is useful for getting an idea about what is going on.
    height = 100.
    hx_height = height   # This is 2.8 the apothem (in-radius)
    sq_height = height
    # shape widths differ, however.
    sq_width = height

    hx_radius = hx_height / rt3  # circum-radius, not apothem
    hx_width = hx_radius * 2.0
    # pixel width implicitly indicate how much of the shape is used in each column.
    sq_px = sq_width
    hx_px = hx_radius * 1.5
    hx_4 = hx_radius * 0.5

    # because the stretch of the hexagon is wider than it's px value,
    # it can have more than 1 square starting in it.
    sq_offset = hx_radius * 0.5
    hx_offset = 0.0
    # but a square cannot have more than 1 hexagon starting in it. It may have none.
    # However, a square always starts in a hexagon, and a hexagon always starts in a square.
    # best_fit(sq_px, hx_px) (1560,  / 1351)
    h_ct = 1562.
    h_fn = HalfHexFns(hx_radius)
    s_fn = ShapeFunctions(sq_width)

    h_sz = h_ct * hx_px + hx_offset
    s_ct = h_ct / (2.0/rt3)
    s_sz = s_ct * sq_px + sq_offset
    if h_sz > h_sz:
        s_sz += hx_px
    else:
        h_sz += sq_px
    # now we have the range we want to go the full route.
    hpx = np.arange(hx_offset, h_sz, step=hx_px)
    spx = np.arange(sq_offset, s_sz, step=sq_px)
    sq_in_hx = np.searchsorted(hpx, spx)
    # sq_in_hx now holds the index values of hpx/spx we always want the idx-1.  This is because it's the index of where it would be placed.
    # for example, given hexes 0-86-173, the squares 28-128-228 would be inserted at 1-2-3, but the parent is at 0,1,2
    s_stuff, h_stuff = {}, {}
    for s_idx, ws_l in enumerate(spx):
        h_idx = sq_in_hx[s_idx] - 1  # for the -1 see above.
        fx = s_calc(h_fn, h_idx, hpx[h_idx], ws_l)
        if abs(sum(fx.values()) - 1.0) > 0.000001:
            print(sum(fx.values()), f'deviates from 1.0 at index {s_idx}')
        s_stuff[s_idx] = fx
    data = {
        'comment': 'keys are the square idx showing the hex column contributions for that row',
        'mod': 1351,
        'lut': s_stuff
    }
    with open('output/hx_sq_lut.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    # Now do the sq_hx lut.
    for h_idx, hs_l in enumerate(hpx):
        fx = {}
        for i, v in s_stuff.items():
            if h_idx in v:
                fx[i] = v[h_idx] / h_fn.r3_2
        h_stuff[h_idx] = fx

    data = {
        'comment': 'keys are the hex idx showing the sq column contributions for that row',
        'mod': 1560,
        'lut': h_stuff
    }
    with open('output/sq_hx_lut.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

