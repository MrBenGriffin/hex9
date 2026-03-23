# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Showing 3 layers of hexagons:  land-usage, hill-shading, coarse over-grid.
Uses the Compositor / LayerSpec pipeline from hhg9.rendering.composition.

Source callables sample GeoTIFFs at hex centroids (single lookup per hex).
This is efficient for fine layers (L12 ≈ 10m) where each hex covers only
a small number of raster pixels; any per-raster-pixel variance within a hex
is negligible at that scale.

Last Tested
13 Mar 2026 0.1.1a1 (passed)
26 Dec 2026 0.1.0a4 (passed)
"""
from pathlib import Path

import numpy as np
import math
from hhg9 import Points, Registrar
from hhg9.algorithms.distance import wgs84_offset, densify_poly_geodesic_by_step
from osgeo import gdal
from hhg9.geo.gdal import Wkt, Wkt_4978, sample_gdal
from hhg9.rendering.composition import LayerSpec, Compositor
from hhg9.rendering.render import plot_hex


if __name__ == '__main__':
    print('Initialising')
    gdal.UseExceptions()
    rg = Registrar()

    repo_file = 'output/ex0310/repo.npz'
    file_to_use = Path(repo_file)
    if not file_to_use.exists():
        print(f'{repo_file} needs to exist for this example to run. Run ex0310_stonehenge_1.py first')
        raise SystemExit(0)

    repo = np.load(file_to_use, allow_pickle=True)
    name = str(repo['name'])
    n_oct_layout = str(repo['noct'])
    ll_spot = repo['spot']
    layers = repo['layers']
    print(f'Loaded {name} with {len(layers)} layers, using `{n_oct_layout}`')
    print(f'ex0310_stonehenge_2.py is not yet implemented!')
    raise SystemExit(0)

    g_gcd = rg.domain('g_gcd')
    c_ell = rg.domain('c_ell')
    b_oct = rg.domain('b_oct')
    c_oct = rg.domain('c_oct')
    n_oct = rg.domain(n_oct_layout)

    for level in layers:
        layer = level['layer']
        print(f'Layer {layer}')
        tif_file = level['tif']
        zoom_size = level['zoom']
        hex_n = Points(level['hex_n_coords'], n_oct, oid=level['hex_n_cmp'])
        bbox_n = hex_n.bbox(trbl=True)
        (ntp, nlt, nbt, nrt) = bbox_n

        # Fill polygon: axis-aligned rectangle in n_oct derived from the bbox.
        fill_n = Points(np.array([[nlt, nbt], [nrt, nbt], [nrt, ntp], [nlt, ntp]]), n_oct)

        ds = gdal.Open(tif_file)
        l_wkt = Wkt(rg, 'l_wkt', ds.GetProjection())
        Wkt_4978(rg, l_wkt)



    # Project boundary polygon into n_oct to establish display bounds
    # ntt = rg.project(q_dens, [g_gcd, b_oct, n_oct])
    # nlt = float(ntt.coords[:, 0].min())
    # nrt = float(ntt.coords[:, 0].max())
    # nbt = float(ntt.coords[:, 1].min())
    # ntp = float(ntt.coords[:, 1].max())
    # bbox_n = (ntp, nlt, nbt, nrt)

    # Local north direction in n_oct: project bbox centre ±Δlat through the full chain
    _ctr_n = Points(np.array([[(nlt + nrt) / 2, (nbt + ntp) / 2]]), n_oct)
    _ctr_gcd = rg.project(_ctr_n, [n_oct, b_oct, c_oct, c_ell, g_gcd])
    _lat, _lon = float(_ctr_gcd.coords[0, 0]), float(_ctr_gcd.coords[0, 1])
    _north_n = rg.project(Points(np.array([[_lat + 0.5, _lon]]), g_gcd), [g_gcd, b_oct, n_oct])
    north_dir = _north_n.coords[0] - _ctr_n.coords[0]

    # Fill polygon: axis-aligned rectangle in n_oct derived from the bbox.
    # This gives a clean rectangular fill (matching the original tri_grid_clipped
    # approach) rather than the warped-parallelogram shape of the geodesic quad.
    # Swap `fill_n` for `ntt` below to clip to the true polygon boundary instead.
    fill_n = Points(
        np.array([[nlt, nbt], [nrt, nbt], [nrt, ntp], [nlt, ntp]]),
        n_oct,
    )

    for layer in [12]:
        print(f'Layer {layer}')

        # Source callables receive hex centroids as Points in b_oct.
        # Each closes over its own dataset + WKT domain.
        def land_source(ctrs: Points) -> np.ndarray:
            smp = rg.project(ctrs, [b_oct, c_oct, c_ell, l_wkt])
            vals, _, _ = sample_gdal(ds, smp.coords)
            return vals.astype(np.float64)

        def shade_source(ctrs: Points) -> np.ndarray:
            smp = rg.project(ctrs, [b_oct, c_oct, c_ell, s_wkt])
            vals, _, valid = sample_gdal(hs_ds, smp.coords)
            basis = vals.astype(np.float64)
            nd = hs_ds.GetRasterBand(1).GetNoDataValue()
            if nd is not None:
                basis[vals == nd] = np.nan
            else:
                basis[vals == 0] = np.nan
            result = np.clip(basis / np.nanmax(basis), 0.0, 1.0)
            result[~valid] = np.nan
            return result

        # ``lw``, ``ec``, ``fc``, ``label_color``, ``label_size``
        specs = [
            LayerSpec(level=layer, kind='fill', source=land_source, lut=nlcd_lut),
            LayerSpec(level=layer + 2, kind='fill', source=shade_source, style={'lw': 0.0, 'lc': '#ffffffff'}),
            LayerSpec(level=layer - 1, kind='outline', threshold=2, reference=True),
            # LayerSpec(level=layer - 1, kind='outline', labels='layer', threshold=2,
            #           style={'label_size': 4, 'ec': '#00000040', 'label_color': '#ffffff60'}),
            LayerSpec(level=layer - 2, kind='outline', labels=True, threshold=2, reference=True),
            LayerSpec(level=layer - 3, kind='outline', labels=True, threshold=2, style={'lw': 1.0, 'label_size': 7}),
        ]

        composed = Compositor(rg, b_oct, n_oct, specs).run(fill_n)

        print(f'{layer}: Plotting')
        title = 'Emerald Bay: Tahoe trailhead centre'
        plot_hex(composed, save_path=f'output/ex0252_sn_B{size_str}L{layer:02d}_{focus}.png',
                 bbox=bbox_n, title=title, north_dir=north_dir)
