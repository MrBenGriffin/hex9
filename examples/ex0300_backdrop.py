# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
OSM plate-carrée backdrop with hex outline layers.

Demonstrates :func:`~hhg9.rendering.composition.make_backdrop` using a regional
plate-carrée image produced by ex0075_osm_mesh.py (OSM tiles via Cartopy).

Because the source is a regular lat/lon grid, a direct pixel lookup is used
instead of a KDTree — faster and exact for this case.

Region driven by the ex0075 extents file; run ex0075 first to generate the
cached image.

Last Tested
13 Mar 2026 0.1.1a1 (passed)
"""
import numpy as np
from matplotlib import image
from hhg9 import Registrar, Points
from hhg9.algorithms.distance import densify_quad_geodesic_by_step
from hhg9.rendering.composition import LayerSpec, Compositor, make_backdrop, estimate_backdrop_size
from hhg9.rendering.render import plot_hex


def osm_sampler(ctrs: Points) -> np.ndarray:
    """Direct pixel lookup in the regional plate-carrée. Out-of-zone → black."""
    gcd = rg.project(ctrs, [b_oct, c_oct, c_ell, g_gcd])
    lats, lons = gcd.coords[:, 0], gcd.coords[:, 1]
    cols = np.floor((lons - lon_min) / (lon_max - lon_min) * img_w).astype(np.int32)
    rows = np.floor((lat_max - lats) / (lat_max - lat_min) * img_h).astype(np.int32)
    valid = (cols >= 0) & (cols < img_w) & (rows >= 0) & (rows < img_h)
    cols = np.clip(cols, 0, img_w - 1)
    rows = np.clip(rows, 0, img_h - 1)
    result = osm_img[rows, cols].copy()
    result[~valid] = 0.0
    return result  # (N, 3) float32


if __name__ == '__main__':
    print('Initialising')
    rg = Registrar()
    g_gcd = rg.domain('g_gcd')
    c_ell = rg.domain('c_ell')
    b_oct = rg.domain('b_oct')
    c_oct = rg.domain('c_oct')
    n_oct = rg.domain('n_oct:butterfly:2000')

    # --- Load OSM regional plate-carrée (produced by ex0075) ---
    print('Loading OSM image')
    osm_img = image.imread('output/ex0075.png')           # (H, W, 3|4) float32
    osm_img = osm_img[:, :, :3]                           # drop alpha if present
    img_h, img_w = osm_img.shape[:2]

    zone = np.load('output/ex0075_extents.npy')           # [lon_min, lon_max, lat_min, lat_max]
    lon_min, lon_max, lat_min, lat_max = zone

    # --- Focus region: use the OSM zone extents directly ---
    print('Building region')
    corners = np.array([
        [lat_min, lon_min], [lat_min, lon_max],
        [lat_max, lon_max], [lat_max, lon_min],
    ])
    qx = densify_quad_geodesic_by_step(corners, max_step_m=50_000)

    ntt = rg.project(Points(qx, g_gcd), [g_gcd, b_oct, n_oct])
    nlt = float(ntt.coords[:, 0].min())
    nrt = float(ntt.coords[:, 0].max())
    nbt = float(ntt.coords[:, 1].min())
    ntp = float(ntt.coords[:, 1].max())
    bbox_n = (ntp, nlt, nbt, nrt)

    fill_n = Points(
        np.array([[nlt, nbt], [nrt, nbt], [nrt, ntp], [nlt, ntp]]),
        n_oct
    )

    # Local north direction in n_oct
    _ctr_n = Points(np.array([[(nlt + nrt) / 2, (nbt + ntp) / 2]]), n_oct)
    _ctr_gcd = rg.project(_ctr_n, [n_oct, b_oct, g_gcd])
    _lat, _lon = float(_ctr_gcd.coords[0, 0]), float(_ctr_gcd.coords[0, 1])
    _north_n = rg.project(Points(np.array([[_lat + 0.5, _lon]]), g_gcd),
                          [g_gcd, b_oct, n_oct])
    north_dir = _north_n.coords[0] - _ctr_n.coords[0]

    # --- Pixel-resolution backdrop ---
    px_w, px_h = estimate_backdrop_size(bbox_n, target_m=175)  # 200 m/px
    print(f'Building backdrop ({px_w} × {px_h})')
    backdrop = make_backdrop(rg, b_oct, n_oct, bbox_n, px_w, px_h, osm_sampler)

    # --- Hex layers: coarse outlines only ---
    specs = [
        LayerSpec(level=2, kind='outline', style={'lw': 0.6, 'ec': '#ffffff80'}),
        LayerSpec(level=3, kind='outline', style={'lw': 0.2, 'ec': '#ffffff80'}),
        LayerSpec(level=4, kind='outline', reference=True, labels=True, threshold=2,
                  style={'lw': 0.4, 'ec': '#ffffff80', 'label_size': 4,
                         'label_color': '#ffffffc0'}),
        LayerSpec(level=5, kind='outline', style={'lw': 0.2, 'ec': '#ffffff80'}),
    ]
    print('Running compositor')
    composed = Compositor(rg, b_oct, n_oct, specs).run(fill_n)

    print('Plotting')
    plot_hex(
        composed,
        save_path='output/ex0300_gb_backdrop.png',
        bbox=bbox_n,
        backdrop=backdrop,
        draw_bbox=False,
        north_dir=north_dir,
        title='Great Britain — OSM backdrop',
        description='L2 ... L5 hex outlines',
    )
