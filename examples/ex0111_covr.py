# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Visualisation of neighbouring Half-hexagons.
This serves to depict the calculations used to identify neighbour offsets.
However, it's out of date and is probably not really doing what it was used for.
The region list would be better expressed as a set of legal regions (rather than cells),
and use the correct semantics for neighbour management (which are now resolved).

Last Tested
13 Mar 2026 0.1.1a1 (passed)
26 Dec 2025 0.1.0a4 (passed - but meaningless)
16 Dec 2025 0.1.0a3 (passed - but meaningless)
29 Aug 2025 (failed)
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from hhg9.h9 import H9K, H9P, H9C, H9R
import hhg9.h9.region as rgn


def grid_lines(axes):
    """
    Plots the underlying triangular 'region' grid of the h9e grid.
    """
    xb = np.array([H9K.limits.TL * 20, H9K.limits.TR * 20])
    xt = H9K.radical.R3 * xb
    h_conditions = [H9K.limits.VC * c for c in range(-19, 19)]
    v_conditions = [H9K.derived.Ẇ * c for c in range(-19, 19)]
    for h in h_conditions:
        axes.axhline(h, linewidth=0.25, color='red', linestyle='-', alpha=0.15)
    for h in v_conditions:
        axes.plot(xb, xt + h, linewidth=0.25, color='red', linestyle='-', alpha=0.15)
        axes.plot(xb, h - xt, linewidth=0.25, color='red', linestyle='-', alpha=0.15)


def set_axis(mfig, lev):
    """Axis template"""
    ax = mfig.add_subplot(6, 6, lev)  # (*nrows*, *ncols*, *index*)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(H9K.limits.TL * 3.25, H9K.limits.TR * 3.25)  # Use TL/TR with a 10% margin
    ax.set_ylim(H9K.limits.VF * 3.25, H9K.limits.ΛC * 3.25)  # Use VF/ΛC with a 10% margin
    ax.set_axis_off()
    grid_lines(ax)
    return ax



def run():
    """Do the work"""
    # initialise figure.
    fig = plt.figure(figsize=(14, 17), dpi=300, frameon=False)
    fig.subplots_adjust(top=0.98, bottom=0.02, right=0.98, left=0.02)
    hh_polys = H9P.hh  # (mode, c2) numpy array of half-hexagons.
    sc_polys = H9P.sv
    n_offs = H9R.loc_offs
    offs = H9C.off_xy
    #
    rg = [
        # Take IMO:=0, NPM:=0, so if sib=False, IPM:=1 (b/c neighbouring modes switch)
        #                                          (idx  imo, npm, c2, sib, PM, Legal)
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
    for level, (_reg, _ngh) in enumerate(rg):
        if _reg[1] == 0x5F:
            continue  # illegal combination
        slf = np.array([_reg], dtype=np.uint8)
        nbr = np.array([_ngh], dtype=np.uint8)
        ax = set_axis(fig, level)
        poi = slf[:, 1][0]            # This is the region whose neighbour we want.
        pmo = H9C.mode[slf[:, 0]][0]
        c2 = rgn.H9R.mcc2[pmo, poi]
        ax.add_patch(Polygon(3*sc_polys[pmo], linewidth=0.5, facecolor='yellow', alpha=0.50))
        smo = H9C.mode[poi]          # We will need its mode (smo=self_mode)
        pxy = 3*offs[poi]            # Offset to the point within the supercell (offs are in sc.scope).
        txy = sc_polys[smo]+pxy      # Get the region triangle at normal size.
        ax.add_patch(Polygon(txy, linewidth=0.5, facecolor='orange', alpha=0.50))  # orange is the new black.
        hhp = hh_polys[smo, c2]+pxy  # Now we can draw the half-hex related to the c2 - this is the POI half-hex.
        ax.add_patch(Polygon(hhp, linewidth=0.5, edgecolor='black', facecolor='green', alpha=0.7))  # Self
        neighbour = nbr[0]           # unpack the array response. neighbour is same structure as rgx
        npa, nme = neighbour[0], neighbour[1]  # we won't use the nc2 indicator here.
        pmn = H9C.mode[npa]          # the mode of the neighbours parent (used for external checking).
        nmo = H9C.mode[nme]          # the mode of the neighbour region.
        nhp = hh_polys[nmo, c2]      # We can paint in the neighbour hhex.
        fc = 'blue' if pmn == pmo else 'red'  # Paint if it's internal to parent, or external.
        xy = n_offs[smo, pmo, c2, 1]  # rgn.H9R.loc_offs provides offsets for neighbours (visualisation only).
        ax.add_patch(Polygon((nhp+xy+pxy), linewidth=0.5, facecolor=fc))
        ax.set_title(f"{level}; C2:{c2}; R:{poi:02x}; N:{nme:02x}")
    fig.savefig(f"output/ex0111.jpg", dpi=400)
    print(f'fig saved at output/ex0111.jpg')


if __name__ == '__main__':
    run()
