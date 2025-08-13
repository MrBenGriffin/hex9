"""
Part of the H9 project - Visualisation of neighbouring Half-hexagons.
This serves to depict the calculations used to identify neighbour offsets
The neighbourhood logic seemed to be quite elusive, and this is a demonstrator.
Last Tested 05 August 2025 √
"""
import numpy as np
from matplotlib import pyplot as plt
from hhg9 import Registrar, H9Engine
from support import Util
from matplotlib.patches import Polygon


def grid_lines(h9, ax):
    """
    Plots the underlying triangular 'region' grid of the h9 grid.
    """
    xb = np.array([h9.TL*10, h9.TR*10])
    xt = h9.R3 * xb
    h_conditions = [h9.VC * c for c in range(-8, 9)]
    v_conditions = [h9.Ẇ * c for c in range(-8, 9)]
    for h in h_conditions:
        ax.axhline(h, color='gray', linestyle='-', alpha=0.25)
    for h in v_conditions:
        ax.plot(xb, xt + h, color='red', linestyle='-', alpha=0.25)
        ax.plot(xb, h - xt, color='red', linestyle='-', alpha=0.25)


def get_hh(h9e, address, c1):
    """
    Decodes a URI address to find the barycentric vertices of its
    final half-hexagon.
    """
    p_uri = address[:, -2]
    p_mode = h9.ugc_lut[p_uri, h9.mode]
    return h9e.poly_hh[p_mode, c1]


if __name__ == '__main__':
    u = Util()
    reg = Registrar()  # Manage Domains & Projections
    h9 = H9Engine()
    fig = plt.figure(figsize=(14, 17), dpi=300, frameon=False)
    fig.subplots_adjust(top=0.98, bottom=0.02, right=0.98, left=0.02)
    nut = h9.neighbour_lut()
    nof = h9.local_offset_lut()
    rg = [
        # Take IMO:=0, NPM:=0, so if sib=False, IPM:=1 (b/c neighbouring modes switch)
        #                                          (idx  imo, npm, c1, sib, PM, Legal)
        [(0x5F, 0x5F, 0x5F), (0x5F, 0x5F, 0x5F)],  # 00  [0,  0,  0,   0],  1   X
        [(0x3a, 0x3a, 0x2b), (0x3a, 0x2a, 0x2b)],  # 01  [0,  0,  0,   1],  0   √√
        [(0x5F, 0x5F, 0x5F), (0x5F, 0x5F, 0x5F)],  # 02  [0,  0,  1,   0],  1   X
        [(0x21, 0x21, 0x49), (0x21, 0x25, 0x34)],  # 03  [0,  0,  1,   1],  0   √√
        [(0x5F, 0x5F, 0x5F), (0x5F, 0x5F, 0x5F)],  # 04  [0,  0,  2,   0],  1   X
        [(0x3a, 0x3a, 0x35), (0x3a, 0x39, 0x16)],  # 05  [0,  0,  2,   1],  0   √√

        [(0x2b, 0x26, 0x2b), (0x3e, 0x39, 0x3e)],  # 06  [0,  1,  0,   0],  0   √√
        [(0x25, 0x35, 0x2b), (0x25, 0x25, 0x3e)],  # 07  [0,  1,  0,   1],  1   √√
        [(0x21, 0x3a, 0x3a), (0x25, 0x25, 0x25)],  # 08  [0,  1,  1,   0],  0   √√
        [(0x25, 0x35, 0x49), (0x25, 0x39, 0x25)],  # 09  [0,  1,  1,   1],  1   √√
        [(0x3a, 0x35, 0x35), (0x39, 0x2a, 0x16)],  # 10  [0,  1,  2,   0],  0   √√
        [(0x25, 0x35, 0x35), (0x25, 0x34, 0x16)],  # 11  [0,  1,  2,   1],  1   √√

        [(0x25, 0x34, 0x39), (0x35, 0x21, 0x2b)],  # 12  [1,  0,  0,   0],  1   √√
        [(0x35, 0x25, 0x39), (0x35, 0x35, 0x2b)],  # 13  [1,  0,  0,   1],  0   √√
        [(0x25, 0x25, 0x25), (0x3a, 0x3a, 0x49)],  # 14  [1,  0,  1,   0],  1   √√
        [(0x35, 0x25, 0x25), (0x35, 0x21, 0x49)],  # 15  [1,  0,  1,   1],  0   √√
        [(0x25, 0x16, 0x16), (0x26, 0x21, 0x21)],  # 16  [1,  0,  2,   0],  1   √√
        [(0x35, 0x25, 0x2a), (0x35, 0x26, 0x21)],  # 17  [1,  0,  2,   1],  0   √√

        [(0x5F, 0x5F, 0x5F), (0x5F, 0x5F, 0x5F)],  # 18  [1,  1,  0,   0],  0   X
        [(0x25, 0x25, 0x3e), (0x25, 0x35, 0x3e)],  # 19  [1,  1,  0,   1],  1   √√
        [(0x5F, 0x5F, 0x5F), (0x5F, 0x5F, 0x5F)],  # 20  [1,  1,  1,   0],  0   X
        [(0x25, 0x2a, 0x34), (0x25, 0x26, 0x49)],  # 21  [1,  1,  1,   1],  1   √√
        [(0x5F, 0x5F, 0x5F), (0x5F, 0x5F, 0x5F)],  # 22  [1,  1,  2,   0],  0   X
        [(0x25, 0x39, 0x16), (0x25, 0x3a, 0x21)],  # 23  [1,  1,  2,   1],  1   √√

    ]
    for depth, (_reg, _ngh) in enumerate(rg):
    # for _imo in range(2):
    #     for _npm in range(2):
    #         for _c1 in range(3):
    #             for _sib in range(2):
    #                 depth = _sib+_c1*2+_npm*6+_imo*12
    #                 if depth < len(rg):
        if _reg[1] == 0x5F:
            continue  # illegal combination
        poi = np.array([_reg])
        ngh = np.array([_ngh])
        poi_region = poi[:, 1]  #
        par_region = poi[:, 0]  #
        c1 = h9.c1(poi, 2)  # c1 of POI via its child.
        pmo = h9.ugc_lut[par_region, h9.mode]  # parent mode
        imo = h9.ugc_lut[poi_region, h9.mode]  # poi mode
        ngh_region = ngh[:, 1]  #
        npr_region = ngh[:, 0]  #
        nbm = h9.ugc_lut[ngh_region, h9.mode]
        npm = h9.ugc_lut[npr_region, h9.mode]
        sib = np.array(npm == pmo, dtype=np.uint8)
        n_off = nof[imo, npm, c1, sib]  # This returns the offset via modes, c1, bool.
        original_hh = get_hh(h9, poi, c1)
        neighbor_hh = get_hh(h9, ngh, c1) + n_off
        ax = fig.add_subplot(5, 5, depth+1)  # (*nrows*, *ncols*, *index*)
        ax.set_xlim(h9.TL * 3.2, h9.TR * 3.2)  # Use TL/TR with a 10% margin
        ax.set_ylim(h9.VF * 3.2, h9.ΛC * 3.2)  # Use VF/ΛC with a 10% margin
        ax.set_aspect('equal', adjustable='box')
        grid_lines(h9, ax)
        r_col = 'black' if pmo[0] == 1 else 'blue'
        l_col = 'yellow' if imo[0] == 1 else 'cyan'
        n_col = 'green' if sib[0] else 'red'
        ax.add_patch(Polygon(original_hh[0], linewidth=3, edgecolor=l_col, facecolor=r_col, alpha=0.95))
        ax.add_patch(Polygon(neighbor_hh[0], linewidth=3, edgecolor=n_col, facecolor=n_col, alpha=0.35))
        ax.set_title(f"{depth}: {imo}{npm}{c1}{sib}")
    plt.show()
