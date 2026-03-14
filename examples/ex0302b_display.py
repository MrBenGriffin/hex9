# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
import os

import numpy as np
from osgeo import gdal
from hhg9 import Registrar, Points
from hhg9.geo.gdal import wkt_geotiff_meta, make_geotiff_sampler, estimate_backdrop_size
from hhg9.rendering.composition import make_backdrop, LayerSpec, Compositor
from hhg9.rendering.render import plot_hex

if __name__ == '__main__':
    gdal.UseExceptions()
    rg = Registrar()
    g_gcd = rg.domain('g_gcd')
    c_ell = rg.domain('c_ell')
    b_oct = rg.domain('b_oct')
    c_oct = rg.domain('c_oct')
    p_pix = rg.domain('p_pix')
    n_oct = rg.domain(f'n_oct:mortar')

    base = 'output/ex0302_overlay.tif'
    if not os.path.exists(base):
        print(f'{base} needs to exist for this example to run.')
        raise SystemExit(0)

    # Stream pixels — no array in memory
    ds, bbox_wkt, wkt = wkt_geotiff_meta(base, reg=rg, name='g_bs')
    sampler = make_geotiff_sampler(ds, rg, [n_oct, b_oct, c_oct, c_ell, wkt])
    top, left, bottom, right = bbox_wkt
    tif_corners = Points(np.array([[left, top], [right, top], [right, bottom], [left, bottom]]), wkt)
    corners_n = rg.project(tif_corners, [wkt, c_ell, c_oct, b_oct, n_oct])
    nlt = float(corners_n.coords[:, 0].min())
    nrt = float(corners_n.coords[:, 0].max())
    nbt = float(corners_n.coords[:, 1].min())
    ntp = float(corners_n.coords[:, 1].max())
    bbox_n = (ntp, nlt, nbt, nrt)
    px_w, px_h = estimate_backdrop_size(ds, bbox_n, rg, [n_oct, b_oct, c_oct, c_ell, wkt])
    backdrop = make_backdrop(rg, b_oct, n_oct, bbox_n, px_w, px_h, sampler)

    specs = [
        LayerSpec(level=4, kind='outline', reference=True,
                  style={'lw': 0.6, 'ec': '#ffffffa0'}),
        LayerSpec(level=5, kind='outline', labels=True, threshold=2,
                  style={'lw': 0.3, 'ec': '#ffffff60', 'label_size': 4,
                         'label_color': '#ffffffc0'}),
    ]
    print('Running compositor')
    composed = Compositor(rg, b_oct, n_oct, specs).run(corners_n)

    print('Plotting')
    plot_hex(
        composed,
        save_path='output/ex0302a_gb_backdrop.png',
        bbox=bbox_n,
        backdrop=backdrop,
        draw_bbox=False,
        title='Great Britain — RT backdrop',
        description='L4 / L5 hex outlines',
    )


