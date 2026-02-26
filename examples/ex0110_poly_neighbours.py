# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Visualisation of neighbouring Half-hexagons.
This serves to depict the calculations used to identify neighbours.
It also shows how we can tell if the neighbour is outside the region's
own parent region (as a supercell). Values are shown as cell-ids (hex)

Last Tested
26 December 2025 0.1.0a4 (passed)
16 December 2025 0.1.0a3 (passed)
20 October 2025 (passed)
"""
import json
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from hhg9 import Points, Registrar
from hhg9.h9 import H9K, H9P, H9C, H9R
import hhg9.h9.region as rgn


def json_load(path):
    """Load json file"""
    with (open(path) as infile):
        obj = json.load(infile)
        infile.close()
    return obj


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
    reg = Registrar()  # Manage Domains & Projections
    locs = json_load('../assets/locations.json')
    region = locs['NWA']
    spot = region['Stonehenge']
    g_gcd = reg.domain('g_gcd')
    b_oct = reg.domain('b_oct')
    b_oct.set_warp('src/l4_polished.npz')

    ll0 = Points(np.array([spot]), g_gcd)
    bc0 = reg.project(ll0, [g_gcd, b_oct])  # spherical cart
    co_, mode = bc0.cm()
    n_offs = H9R.loc_offs
    offs = H9C.off_xy
    it = rgn.xy_regions_iter(bc0.coords, mode=mode, depth=35)
    for ev in it:
        if ev.phase == 'pre' and ev.i > 0:
            level = ev.i                # current iteration (>0) level will start at 2.
            ax = set_axis(fig, level)
            c2i = int(ev.cid[0])         # c2 indicator (next region).
            rgx = [int(a) for a in ev.addresses[:, ev.i-1: ev.i+1][0]]
            rgx.append(c2i)              # moving window with ci being the c2-indicator.
            pmo = H9C.mode[rgx[0]]       # supercell: parent mode. par_region
            ax.add_patch(Polygon(3*sc_polys[pmo], linewidth=0.5, facecolor='yellow', alpha=0.50))
            poi = rgx[1]                 # This is the region whose neighbour we want.
            smo = H9C.mode[poi]          # We will need it's mode (smo=self_mode)
            pxy = 3*offs[poi]            # Offset to the point within the supercell (offs are in sc.scope).
            txy = sc_polys[smo]+pxy      # Get the region triangle at normal size.
            ax.add_patch(Polygon(txy, linewidth=0.5, facecolor='orange', alpha=0.50))  # orange is the new black.
            c2 = rgn.H9R.mcc2[smo, c2i]  # Now grab the c2 of the current region - which neighbour?!
            hhp = hh_polys[smo, c2]+pxy  # Now we can draw the half-hex related to the c2 - this is the POI half-hex.
            ax.add_patch(Polygon(hhp, linewidth=0.5, edgecolor='black', facecolor='green', alpha=0.7))  # Self
            nbr, nc2 = rgn.region_neighbours(np.array([rgx]))  # region_neighbours returns the nbr stack and c2.
            neighbour = nbr[0]           # unpack the array response. neighbour is same structure as rgx
            npa, nme = neighbour[0], neighbour[1]  # we won't use the nc2 indicator here.
            pmn = H9C.mode[npa]          # the mode of the neighbours parent (used for external checking).
            nmo = H9C.mode[nme]          # the mode of the neighbour region.
            nhp = hh_polys[nmo, c2]      # We can paint in the neighbour hhex.
            fc = 'blue' if pmn == pmo else 'red'  # Paint if it's internal to parent, or external.
            xy = n_offs[smo, pmo, c2, 1]  # rgn.H9R.loc_offs provides offsets for neighbours (visualisation only).
            ax.add_patch(Polygon((nhp+xy+pxy), linewidth=0.5, facecolor=fc))
            ax.set_title(f"{level}; C2:{c2}; R:{c2i:02x}; N:{nme:02x}")
    fig.savefig(f"output/ex0110.jpg", dpi=400)
    print(f'fig saved at output/ex0110.jpg')


if __name__ == '__main__':
    run()
