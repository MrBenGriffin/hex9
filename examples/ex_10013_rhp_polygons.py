# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Test the rhombus net forward projection by rendering the 24 half-hex polygons
(8 faces × 3 c2) directly from H9P.hh via per-c2 affines, then round-tripping
them back to barycentric space and reporting the max error.

Run this *before* worrying about the pixel-grid renderer (ex_10012) — it
confirms that:
  1. Each of the 24 net-space quads lands in the right place in the rhombus layout.
  2. BaryNet.backward() recovers the original barycentric vertices.

The plot is saved to output/ex_10013_rhp_polygons.svg (vector, easy to zoom).

Last Tested
08 March 2026
"""
import numpy as np
import matplotlib

from hhg9.h9.uuid_address import h9_enc, h9_bin

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from hhg9 import Registrar, Points
from hhg9.h9 import H9O, H9P, uuid_address

# Colours per octant (8 faces)
FACE_COLOURS = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231',
    '#911eb4', '#42d4f4', '#f032e6', '#bfef45',
]
FACE_ALPHA = [0.6, 0.4, 0.25]   # c2=0 most opaque, 1 and 2 lighter


def run(*, save_svg=True, print_errors=True):
    # nets.net_layouts['rhombus'] = RHOMBUS_LAYOUT
    reg = Registrar()
    b_oct = reg.domain('b_oct')
    n_oct = reg.domain('n_oct:butterfly:0500')

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_aspect('equal')
    ax.set_title('Net — 24 half-hex quads (8 faces × 3 c2)')

    max_err = 0.0
    legend_patches = []
    poly_eps = 1 - 1e-16

    for name, sdom in n_oct.sides.items():
        oid   = sdom.oid
        bmo   = int(H9O.oid_mo[oid])   # barycentric net_mode
        colour = FACE_COLOURS[oid]
        label  = H9O.oid_str[oid]
        for c2 in range(3):
            tetra = H9P.hh[bmo, c2].copy()
            tct = np.atleast_2d(np.mean(tetra, axis=0))
            bp = Points(tct, domain=b_oct, oid=oid)
            bv = Points(tetra, domain=b_oct, oid=oid)
            mm = h9_enc(bp)
            mi = h9_bin(mm, 0, reg)
            nv = reg.project(bv, [b_oct, n_oct])
            net_verts = nv.coords
            bm = np.mean(net_verts, axis=0)
            print(f'{label} c2={c2} mean={bm}')

            # ── forward render ──────────────────────────────────────────────
            poly = MplPolygon(net_verts, closed=True,
                              facecolor=colour, alpha=FACE_ALPHA[c2],
                              edgecolor='k', linewidth=0.4)
            ax.add_patch(poly)

            # label centroid with face name + c2
            cx, cy = net_verts.mean(axis=0)

            ax.text(cx, cy, f'{label}\nc2={c2}', ha='center', va='center', fontsize=4, color='k')

            # ── round-trip: net → bary ──────────────────────────────────────
            pts_bary = reg.project(nv, [n_oct, b_oct])
            err = float(np.max(np.abs(pts_bary.coords - bv.coords)))
            max_err = max(max_err, err)

        legend_patches.append(
            mpatches.Patch(color=colour, label=label)
        )

    ax.legend(handles=legend_patches, fontsize=7, loc='upper right')
    ax.autoscale_view()
    ax.set_xlabel('U (lattice units)')
    ax.set_ylabel('V (lattice units)')

    if save_svg:
        path = 'output/ex_10013_rhp_polygons.svg'
        fig.savefig(path, bbox_inches='tight')
        print(f'Saved {path}')
    else:
        path = 'output/ex_10013_rhp_polygons.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'Saved {path}')

    plt.close(fig)

    if print_errors:
        print(f'Max round-trip error (net → bary): {max_err:.2e}')
        if max_err < 1e-10:
            print('  ✓ Round-trip OK (numerical noise only)')
        else:
            print('  ✗ Round-trip error is large — check affines / c2trans values')


if __name__ == '__main__':
    run(save_svg=True)
