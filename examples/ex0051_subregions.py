# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Part of the H9 project
Use an address to resolve the region list and show the
region identity for each of the first 36 layers.
This helps to identify if/where any issues may arise.
13 Mar 2026 0.1.1a1 (passed)
28 Feb 2026 0.1.1a1 (passed)
26 Dec 2025 0.1.0a4 (passed)
16 Dec 2025 0.1.0a3 (passed)
25 Nov 2025 (passed)
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from hhg9 import Points, Registrar
import hhg9.h9.region as rgn
from hhg9.h9 import H9C, H9P
from hhg9.h9.region import xy_regions_iter


def run():
    """Do the work"""
    reg = Registrar()  # Manage Domains & Projections

    # set up the polygon references.
    tr0 = H9P.tx[0].reshape(-1, 3, 2)  # 9 mode 0 triangles
    tr1 = H9P.tx[1].reshape(-1, 3, 2)  # 9 mode 1 triangles
    rtx0 = {rg: tx for rg, tx in zip([0x26, 0x2a, 0x2b, 0x3a, 0x39, 0x49, 0x35, 0x25, 0x21], tr0)}
    rtx1 = {rg: tx for rg, tx in zip([0x39, 0x3a, 0x3e, 0x25, 0x35, 0x34, 0x2a, 0x26, 0x16], tr1)}

    # initialise figure.
    fig = plt.figure(figsize=(14, 17), dpi=300, frameon=False)
    fig.subplots_adjust(top=0.98, bottom=0.02, right=0.98, left=0.02)

    # Great Pyramid.
    poi = [29.979189320439783, 31.134207865680587]
    ll0 = Points(np.array([poi]), 'g_gcd')
    bc0 = reg.project(ll0, ['b_oct'])  # spherical cart
    co_, mode = bc0.cm()
    erg = rgn.xy_regions(bc0.coords, mode, depth=36)[0][1:]
    hxe = [f'{e:02x}' for e in erg]
    print(' '.join(hxe))
    irg = []
    it = xy_regions_iter(bc0.coords, mode=mode, depth=35)
    for ev in it:
        if ev.phase == 'pre':
            layer = ev.i
            ax = fig.add_subplot(6, 6, layer+1)  # (*nrows*, *ncols*, *index*)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlim(-0.75, 0.75)
            ax.set_ylim(-0.84, 0.84)
            ax.set_axis_off()
            par = ev.addresses[0, layer]
            pmo = H9C.mode[par]
            region = ev.cid[0]
            cmo = H9C.mode[region]
            irg.append(int(region))

            txp, txe = (rtx0, rtx1) if pmo == 0 else (rtx1, rtx0)
            eg = erg[layer]
            for rg, tx in txp.items():
                if rg == region:
                    color = 'green'
                else:
                    color = 'none'
                ax.add_patch(Polygon(tx, linewidth=0.5, edgecolor='black', facecolor=color, alpha=0.95))
                txl = np.mean(tx, axis=0)
                ax.text(txl[0], txl[1], f'{rg:02x}', color='k', fontsize=14, ha='center', va='center')
            if eg not in txp and eg in txe:
                tx = txe[eg]
                ax.add_patch(Polygon(tx, linewidth=0.5, edgecolor='red', facecolor='pink', alpha=0.95))
            ax.set_title(f"L{layer+1}: 0x{par:02x}->0x{region:02x}")
    fig.savefig(f'output/ex0046.png', dpi=300, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)


if __name__ == '__main__':
    run()
