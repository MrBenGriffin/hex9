import math
import json
import numpy as np
from shape_functions import TriangleFns, ShapeFunctions


def best_fit(sq_px, tr_px):
    # The following shows that a good fit (at rank 3) is 0.006819127869120848: (h:112, s:97)
    # however, top rank is 0.0004895913533781656: (h:1560, s:1351), ranked 3 at 20,000 (when 112/97 is not-ranked)
    # but 21728, 18817 is even better, and in the top ten for 1,000,000
    rdx = {}
    dst = []
    for i in range(200000):
        si, hi = i * sq_px, i * tr_px  # si, hi are current square,hex px
        s = math.floor(hi / sq_px)  # divide by hi by sq_px.
        so = hi - s*sq_px
        h = math.floor(si / tr_px)
        ho = si - h*tr_px
        if (h % 2) == 0:
            ky = math.sqrt(ho * ho + so * so) / 100.0
            dst.append(ky)
            rdx[ky] = s, h, i
    dst.sort()
    for d in dst[:30]:
        print(f'{d}: {rdx[d]}')


def s_calc(fn, tx, t_val, sbase):  # given a square, calculate the triangle contributions to it.
    result = {}
    t_id, tbase, adj = int(tx), t_val, 0  # self.s_2
    if sbase - tbase < fn.s_2:
        tbase -= tr_px
        t_id -= 1
        adj = -1
    for i in range(4):
        t_o = tbase + tr_px * i  # tri-origin
        sl = sbase - t_o
        sr = sl + sq_px
        if sr <= 0:
            return result
        if sl <= 0:
            contr = fn.areas([sr])
            result[t_id + i] = contr[0] / fn.r3
            continue
        contr = fn.areas([sl, sr])
        ctx = 1 if i+adj < 1 else 0
        if contr[ctx] == 0:
            continue
        result[t_id + i] = contr[ctx] / fn.r3
    return result


if __name__ == '__main__':

    rt3 = math.sqrt(3.)
    # a3 = 2.0/rt3
    rt3_2 = rt3/2.0
    # the common part is the height. This really won't survive the local code
    # but is useful for getting an idea about what is going on.
    height = 100.
    tr_height = height   # This is 2.8 the apothem (in-radius)
    sq_height = height
    # shape widths differ, however.
    sq_width = height

    tr_radius = tr_height / rt3_2  # circum-radius, not apothem
    tr_width = tr_radius * 1.0
    # pixel width implicitly indicate how much of the shape is used in each column.
    sq_px = sq_width
    tr_px = tr_radius * 0.5

    # because the stretch of the triangle is wider than it's px value,
    # it can have more than 1 square starting in it.
    sq_offset = tr_radius * 0.5  # sq starts on a tr spike.
    tr_offset = 0.0

    # best_fit(sq_px, tr_px)
    # but a square cannot have more than 1 triangle starting in it. It may have none.
    # However, a square always starts in a triangle, and a triangle always starts in a square.
    #  0.0003852084167274113: (390, 4680, 1351)
    # best_fit(sq_px, tr_px) (2340, 1351)
    tr_ct = 2345.
    t_fn = TriangleFns(tr_radius)
    s_fn = ShapeFunctions(sq_width)

    h_sz = tr_ct * tr_px + tr_offset
    s_ct = tr_ct / rt3  # something like that.
    s_sz = s_ct * sq_px + sq_offset
    if h_sz > h_sz:
        s_sz += tr_px
    else:
        h_sz += sq_px
    # now we have the range we want to go the full route.
    tpx = np.arange(tr_offset, h_sz, step=tr_px)
    spx = np.arange(sq_offset, s_sz, step=sq_px)
    sq_in_tr = np.searchsorted(tpx, spx)
    # sq_in_hx now holds the index values of hpx/spx we always want the idx-1.  This is because it's the index of where it would be placed.
    # for example, given hexes 0-86-173, the squares 28-128-228 would be inserted at 1-2-3, but the parent is at 0,1,2
    s_stuff, t_stuff = {}, {}
    for s_idx, ws_l in enumerate(spx):
        t_idx = sq_in_tr[s_idx] - 1  # for the -1 see above.
        fx = s_calc(t_fn, t_idx, tpx[t_idx], ws_l)
        if abs(sum(fx.values()) - 1.0) > 0.000001:
            print(sum(fx.values()), f'deviates from 1.0 at index {s_idx}')
        s_stuff[s_idx] = fx
    data = {
        'comment': 'keys are the square idx showing the tri column contributions for that row',
        'mod': 1351,
        'lut': s_stuff
    }
    with open('output/tr_sq_lut.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    # Now do the sq_hx lut.
    for t_idx, ts_l in enumerate(tpx):
        fx = {}
        for i, v in s_stuff.items():
            if t_idx in v:
                fx[i] = v[t_idx] * t_fn.r3
        t_stuff[t_idx] = fx

    data = {
        'comment': 'keys are the tri idx showing the sq column contributions for that row',
        'mod': 2340,
        'lut': t_stuff
    }
    with open('output/sq_tr_lut.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

