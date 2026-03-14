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
import numpy as np
import math
import json
from hhg9 import Points, Registrar
from hhg9.algorithms.distance import wgs84_offset, densify_quad_geodesic_by_step
from osgeo import gdal
from hhg9.geo.gdal import Wkt, Wkt_4978, sample_gdal
from hhg9.rendering.composition import LayerSpec, Compositor
from hhg9.rendering.render import plot_hex


def create_nlcd_lut() -> np.ndarray:
    """Load NLCD colour table.  Returns (256, 3) uint8 RGB array."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    with open('../assets/nlcd_legend.json') as fp:
        data = json.load(fp)
    for item in data['nlcd_legend']:
        idx = item['id']
        hv = item['color'].lstrip('#')
        lut[idx] = [int(hv[i:i + 2], 16) for i in (0, 2, 4)]
    return lut
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('Initialising')
    gdal.UseExceptions()
    rg = Registrar()
    g_gcd = rg.domain('g_gcd')
    c_ell = rg.domain('c_ell')
    b_oct = rg.domain('b_oct')
    c_oct = rg.domain('c_oct')
    n_oct = rg.domain('n_oct:diamonds')  # diamonds keeps north pole north

    nlcd_lut = create_nlcd_lut()

    print('Loading geotiffs')
    # Land-usage (NLCD 2024, cropped to Sierra Nevada)
    src_file = '../experimental/personal/src/Annual_NLCD_LndCov_2024_CU_C1V1_crop.tif'
    ds = gdal.Open(src_file)
    l_wkt = Wkt(rg, 'l_wkt', ds.GetProjection())
    Wkt_4978(rg, l_wkt)

    # Hillshade
    hs_ds = gdal.Open('../experimental/personal/src/CA_SierraNevada_hs.tif')
    s_wkt = Wkt(rg, 's_wkt', hs_ds.GetProjection())
    Wkt_4978(rg, s_wkt)

    print('Building area polygon')
    focus = 'CaSN'
    size, size_str = 1_000, '1k'  # metres half-width each side
    # Emerald Bay / Tahoe trailhead centre
    centre = np.atleast_2d([38.9517340730661, -120.1119544660363])
    diag = (size / 2.0) * math.sqrt(2.0)
    az = np.array([45.0, 135.0, 225.0, 315.0])
    corners = wgs84_offset(
        np.repeat(centre, 4, axis=0),
        np.tile(az, 1),
        np.full(4, diag),
    )
    qx = densify_quad_geodesic_by_step(corners, max_step_m=50)
    q_dens = Points(qx, g_gcd)

    # Project boundary polygon into n_oct to establish display bounds
    ntt = rg.project(q_dens, [g_gcd, b_oct, n_oct])
    nlt = float(ntt.coords[:, 0].min())
    nrt = float(ntt.coords[:, 0].max())
    nbt = float(ntt.coords[:, 1].min())
    ntp = float(ntt.coords[:, 1].max())
    bbox_n = (ntp, nlt, nbt, nrt)

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
