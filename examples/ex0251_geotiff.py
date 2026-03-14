# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Chiginagak volcano
Showing 3 layers of hexagons: land-usage, hill-shading, geology (labels)
Uses the Compositor / LayerSpec pipeline from hhg9.rendering.composition.
Source callables sample GeoTIFFs at hex centroids (single lookup per hex).

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
        lut[idx] = [int(hv[i:i+2], 16) for i in (0, 2, 4)]
    return lut


def create_label_lut(file_name: str) -> dict:
    """Load a label look-up table from a JSON file.  Returns {int: value}."""
    with open(file_name) as fp:
        lut = json.load(fp)
    return {int(k): v for k, v in lut.items()}


if __name__ == '__main__':
    print('Initialising')
    gdal.UseExceptions()
    rg = Registrar()
    g_gcd = rg.domain('g_gcd')
    c_ell = rg.domain('c_ell')
    b_oct = rg.domain('b_oct')
    c_oct = rg.domain('c_oct')
    n_oct = rg.domain('n_oct:diamonds')   # keeps north pole north

    nlcd_lut = create_nlcd_lut()
    rock_lut = create_label_lut('../experimental/personal/src/chiginagak_geology.json')

    print('Loading geotiffs')
    # Land-usage (NLCD 2016, Alaska)
    src_file = '../experimental/personal/src/NLCD_2016_Land_Cover_AK_20200724.img'
    ds = gdal.Open(src_file)
    g_wkt = Wkt(rg, 'g_wkt', ds.GetProjection())
    Wkt_4978(rg, g_wkt)

    # Hillshade and geology share the same CRS as the NLCD
    hs_ds = gdal.Open('../experimental/personal/src/hs_md.tif')
    rocks = gdal.Open('../experimental/personal/src/chiginagak_geology.tif')

    print('Building area polygon')
    focus = 'APNPr'
    size, size_str = 7_500.0, '7_5k'    # metres half-width each side
    # Chiginagak volcano centre
    centre = np.atleast_2d([57.13330344238441, -156.98699788901348])
    diag = (size / 2.0) * math.sqrt(2.0)
    az = np.array([45.0, 135.0, 225.0, 315.0], dtype=np.float64)
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
    # Swap `fill_n` for `ntt` to clip to the true geodesic polygon boundary instead.
    fill_n = Points(
        np.array([[nlt, nbt], [nrt, nbt], [nrt, ntp], [nlt, ntp]]),
        n_oct,
    )

    for layer in [10]:
        print(f'Layer {layer}')

        # Source callables receive hex centroids as Points in b_oct.
        # All three datasets share g_wkt (same Alaska CRS).
        def land_source(ctrs: Points) -> np.ndarray:
            smp = rg.project(ctrs, [b_oct, c_oct, c_ell, g_wkt])
            vals, _, _ = sample_gdal(ds, smp.coords)
            return vals.astype(np.float64)

        def shade_source(ctrs: Points) -> np.ndarray:
            smp = rg.project(ctrs, [b_oct, c_oct, c_ell, g_wkt])
            vals, _, valid = sample_gdal(hs_ds, smp.coords)
            # Mask raster nodata (often 0 for hillshade files outside DEM extent)
            nd = hs_ds.GetRasterBand(1).GetNoDataValue()
            basis = vals.astype(np.float64)
            if nd is not None:
                print(f'Hillshade nodata: {nd}')
                basis[vals == nd] = np.nan
            else:
                basis[vals == 0] = np.nan
            max_val = np.nanmax(basis)
            print(f'Hillshade max: {max_val}')
            # Normalise uint8 hillshade (0=shadowed, 255=lit) to [0,1]
            result = np.clip(basis / max_val, 0.0, 1.0)
            # Mask out-of-bounds pixels
            result[~valid] = np.nan
            return result

        def rock_source(ctrs: Points) -> np.ndarray:
            smp = rg.project(ctrs, [b_oct, c_oct, c_ell, g_wkt])
            vals, _, _ = sample_gdal(rocks, smp.coords)
            return vals.astype(np.float64)

        specs = [
            LayerSpec(level=layer+0, kind='fill', source=land_source, lut=nlcd_lut),
            LayerSpec(level=layer+2, kind='fill', source=shade_source, style={'lw': 0.0, 'lc': '#ffffffff'}),
            LayerSpec(level=layer-1, kind='fill', source=rock_source, lut=rock_lut, reference=True),
            LayerSpec(level=layer-2, kind='outline', reference=True),
            LayerSpec(level=layer-3, kind='outline', labels=True, threshold=2, style={'label_size': 6}),
        ]

        composed = Compositor(rg, b_oct, n_oct, specs).run(fill_n)

        print(f'{layer}: Plotting')
        title = 'Chiginagak Volcano, Alaska'
        plot_hex(composed, save_path=f'output/ex0251_whx_{size_str}L{layer:02d}{focus}.png',
                 bbox=bbox_n, title=title, north_dir=north_dir)
