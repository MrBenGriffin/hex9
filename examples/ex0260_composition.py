# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Composition frame + dynamic local net + backdrop.

Companion to ex0260_polygrid.py.  Where ex0260_polygrid drives HexMesh and the
local net by hand to show addressing, this renders the SAME non-convex
Britain–Belgium region through the high-level *composition frame*:

    Compositor(reg, b_oct, n_oct, specs, local=True).run(polygon_n)

`local=True` places every hex on a dynamic local net (n_oct.local_layout),
hinged from a fundamental octant so the two-octant region lies flat and seam
continuous.  A Blue-Marble backdrop is rasterised in the *same* local-net frame
with ``make_local_backdrop`` (the local analogue of ``make_backdrop``) so the
imagery aligns with the hex overlay, and both are drawn by ``plot_hex``.

Run ``ex0262_greenwich_seam`` / ``ex0261_polygrid_mesh`` for the geographic
(g_gcd) rendering of the same engine; this is the octant-native (regular-hex)
view with a backdrop.

Last Tested
-----------
28 Jun 2026 0.1.3a0 (new; composition frame, local net, Blue Marble backdrop)
"""
import numpy as np
from matplotlib import image

from hhg9 import Points, Registrar
from hhg9.rendering.composition import (
    LayerSpec, Compositor, make_local_backdrop, estimate_backdrop_size)
from hhg9.rendering.render import plot_hex

# [birmingham-bristol-london-belgium] NON-CONVEX polygon, [lat, lon]; spans octant 0 & 2.
EU = np.array([
    [51.084464524954795, -3.4462861844929585],
    [52.68223344923537, -2.189458562294672],
    [52.60858425701964, -1.432594162686184],
    [51.648685168766754, -2.0537941887799422],
    [51.52446468552721, -0.8970768988122543],
    [51.82555423461282, -0.747132064927554],
    [51.684114370958646, 0.5309691381848917],
    [51.40435883726121, 0.47384729670500597],
    [51.29190823165293, 5.4859054058927266],
    [50.72448838777691, 5.4044368907702074],
])

# NON-CONVEX C-band over all four northern octants (0,1,2,3), OPEN over the
# Pacific so it does not enclose the north pole -> unfolds tear-free.  [lat, lon].
C_BAND = np.array([
    [68, -170], [68, -90], [68, 0], [68, 90], [68, 115],
    [42, 115], [42, 90], [42, 0], [42, -90], [42, -170],
])


if __name__ == '__main__':
    rg = Registrar()
    g_gcd = rg.domain('g_gcd')
    b_oct = rg.domain('b_oct')
    n_oct = rg.domain('n_oct:butterfly')

    # --- Blue Marble plate-carrée sampler (b_oct centroids/pixels -> RGB) ---
    bm = image.imread('src/bm_3600x1800.png')[:, :, :3]   # (H, W, 3) float
    img_h, img_w = bm.shape[:2]

    def bm_sampler(pts: Points) -> np.ndarray:
        gcd = rg.project(pts, [b_oct, g_gcd])             # (lat, lon)
        lat, lon = gcd.coords[:, 0], gcd.coords[:, 1]
        col = (((lon + 180.0) / 360.0) * img_w).astype(np.int32) % img_w
        row = np.clip((((90.0 - lat) / 180.0) * img_h).astype(np.int32), 0, img_h - 1)
        return bm[row, col]

    def run_case(polygon, specs, name, target_m, title, desc, north=True):
        polygon_n = rg.project(Points(polygon, g_gcd), [g_gcd, b_oct, n_oct])
        comp = Compositor(rg, b_oct, n_oct, specs, local=True)
        layers = comp.run(polygon_n)
        layout = comp.layout.layout
        cut = '' if comp.local_cut < 1e-9 else \
            f' — wraps a cone vertex (cut_residual {comp.local_cut:.2f})'
        print(f'{name}: octants {sorted(layout)}, '
              f'residual {comp.local_residual:.2e}, '
              f'layers {[(cl.spec.level, cl.count) for cl in layers]}, '
              f'dropped(open-seam straddlers) {comp.local_dropped}{cut}')

        allv = np.vstack([cl.verts.coords for cl in layers])
        lt, rt = float(allv[:, 0].min()), float(allv[:, 0].max())
        bt, tp = float(allv[:, 1].min()), float(allv[:, 1].max())
        bbox = (tp, lt, bt, rt)

        px_w, px_h = estimate_backdrop_size(bbox, target_m=target_m)
        print(f'  backdrop {px_w}×{px_h}')
        backdrop = make_local_backdrop(rg, b_oct, layout, bbox, px_w, px_h, bm_sampler)

        # North in the local-net frame at the region centroid (+0.5° lat).
        north_dir = None
        if north:
            c_ll = polygon.mean(axis=0)
            p0 = rg.project(Points(np.array([c_ll]), g_gcd), [g_gcd, b_oct])
            pn = rg.project(Points(np.array([[c_ll[0] + 0.5, c_ll[1]]]), g_gcd),
                            [g_gcd, b_oct])
            xy0 = n_oct.place(p0.coords, p0.oid, layout)[0]
            xyn = n_oct.place(pn.coords, pn.oid, layout)[0]
            north_dir = xyn - xy0

        plot_hex(
            layers, save_path=f'output/{name}.png', bbox=bbox, backdrop=backdrop,
            draw_bbox=False, north_dir=north_dir, show_north=north,
            title=title, description=desc, dpi=200,
        )

    # Two-octant region (Britain–Belgium), fine layers + north arrow.
    run_case(
        EU,
        [LayerSpec(level=4, kind='outline', style={'lw': 1.2, 'ec': '#ffffff80'}),
         LayerSpec(level=5, kind='outline', style={'lw': 0.5, 'ec': '#ffffff80'}),
         LayerSpec(level=6, kind='outline', reference=True, threshold=2,
                   style={'lw': 0.3, 'ec': '#ffffff80'})],
        'ex0260_composition', target_m=300,
        title='Britain–Belgium — local-net composition',
        desc='Blue Marble backdrop, L4–L6 hex outlines (dynamic local net)')

    # Four-octant C-band, coarse layers; no single north (curved 4-octant net).
    run_case(
        C_BAND,
        [LayerSpec(level=1, kind='outline', style={'lw': 1.4, 'ec': '#ffffff80'}),
         LayerSpec(level=2, kind='outline', style={'lw': 0.7, 'ec': '#ffffff80'}),
         LayerSpec(level=3, kind='outline', reference=True,
                   style={'lw': 0.3, 'ec': '#ffffff80'})],
        'ex0260_composition_cband', target_m=12000,
        title='Northern C-band — four-octant local-net composition',
        desc='Blue Marble backdrop, L1–L3 hex outlines (dynamic local net, 4 octants)',
        north=False)
