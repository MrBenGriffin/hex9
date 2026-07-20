# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Demonstrates the direct-RGB fill path: the source callable returns an
``(H, 3)`` float array produced by applying a matplotlib colormap to a
scalar field.  The renderer recognises the 2-D ``values`` array and uses
the colours directly as per-hex face colours (no LUT lookup required).

Last Tested
13 Mar 2026 0.1.1a1 (passed)
"""
import numpy as np
from matplotlib import image, colormaps
from scipy.spatial import KDTree

from hhg9 import Registrar, Points
from hhg9.algorithms.distance import (densify_quad_geodesic_by_step, haversine_from_ref_rad)
from hhg9.rendering.composition import LayerSpec, Compositor, make_backdrop, estimate_backdrop_size
from hhg9.rendering.render import plot_hex

# Reference location: Stonehenge, Wiltshire, UK
REF_LAT, REF_LON = 51.1789, -1.8262
MAX_DIST_KM = 100.0          # distance at which colormap saturates


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
    p_pix = rg.domain('p_pix')
    pc_px = p_pix.adopt(osm_img, extent=(lon_min, lat_min, lon_max, lat_max))  # (xmin,ymin,xmax,ymax)
    pc_gcd = rg.project(pc_px, [p_pix, g_gcd])

    ref_lat_r = np.radians(REF_LAT)
    ref_lon_r = np.radians(REF_LON)
    cmap = colormaps['plasma']

    gt, gr, gb, gl = pc_gcd.bbox(trbl=True)  # (lat_max, lon_max, lat_min, lon_min)
    quad = np.array([[gb, gl], [gb, gr], [gt, gr], [gt, gl]])  # (lat, lon) order for g_gcd
    qx = densify_quad_geodesic_by_step(quad, max_step_m=10_000)

    ntt = rg.project(Points(qx, g_gcd), [g_gcd, b_oct, n_oct])
    nlt = float(ntt.coords[:, 0].min())
    nrt = float(ntt.coords[:, 0].max())
    nbt = float(ntt.coords[:, 1].min())
    ntp = float(ntt.coords[:, 1].max())
    bbox_n = (ntp, nlt, nbt, nrt)    # (top, left, bottom, right) for make_backdrop/plot_hex
    corners = np.array([[nlt, nbt], [nrt, nbt], [nrt, ntp], [nlt, ntp]])
    fill_n = Points(corners, n_oct)
    px_w, px_h = estimate_backdrop_size(bbox_n, 300)

    # Local north direction in n_oct
    _ctr_n = Points(np.array([[(nlt + nrt) / 2, (nbt + ntp) / 2]]), n_oct)
    _ctr_gcd = rg.project(_ctr_n, [n_oct, b_oct, c_oct, c_ell, g_gcd])
    _lat, _lon = float(_ctr_gcd.coords[0, 0]), float(_ctr_gcd.coords[0, 1])
    _north_n = rg.project(Points(np.array([[_lat + 0.5, _lon]]), g_gcd),
                          [g_gcd, b_oct, n_oct])
    north_dir = _north_n.coords[0] - _ctr_n.coords[0]

    src = KDTree(pc_gcd.coords)

    def dist_source(ctrs: Points) -> np.ndarray:
        """Per-hex RGB: haversine distance from Stonehenge → plasma colormap."""
        gcd = rg.project(ctrs, [b_oct, c_oct, c_ell, g_gcd])
        dists_m = haversine_from_ref_rad(
            ref_lat_r, ref_lon_r,
            np.radians(gcd.coords[:, 0]),
            np.radians(gcd.coords[:, 1]),
        )
        t = dists_m / (MAX_DIST_KM * 1_000)
        rx = np.full([dists_m.shape[0], 3], np.nan)
        ok = t < 1.0
        rx[ok] = cmap(t)[ok, :3]
        return rx

    def bm_sampler(ctrs: Points) -> np.ndarray:
        """Sample blue marble RGB at pixel grid points."""
        gcd = rg.project(ctrs, [b_oct, c_oct, c_ell, g_gcd])
        _, idx = src.query(gcd.coords, workers=-1)
        return pc_px.samples[idx]          # (N, 3) float32

    # --- UK pixel backdrop ---
    print(f'Building backdrop ({px_w}×{px_h})')
    backdrop = make_backdrop(rg, b_oct, n_oct, bbox_n, px_w, px_h, bm_sampler)

    # --- Hex layers ---
    layer = 5
    specs = [
        LayerSpec(level=layer+3, kind='fill', source=bm_sampler,
                  style={'alpha': 0.75}),
        # Per-hex heatmap fill (direct RGB from colormap)
        LayerSpec(level=layer, kind='fill', source=dist_source,
                  style={'alpha': 0.20}),
        # Coarser outline layer to segment the heatmap visually
        LayerSpec(level=layer - 1, kind='outline', reference=True,
                  style={'lw': 0.1, 'ec': '#ffffff50'}),
        # Coarsest labels for geographic orientation
        LayerSpec(level=layer - 2, kind='outline', labels=True, threshold=2,
                  style={'lw': 0.05, 'ec': '#ffffff80', 'label_size': 3,
                         'label_color': '#ffffffd0'}),
    ]
    print(f'Running compositor (layer {layer})')
    composed = Compositor(rg, b_oct, n_oct, specs).run(fill_n)

    print('Plotting')
    plot_hex(
        composed,
        save_path='output/ex0301_stonehenge_heatmap.png',
        bbox=bbox_n,
        # backdrop=backdrop,
        draw_bbox=False,
        north_dir=north_dir,
        rotate_north=True,
        title='Distance from Stonehenge',
        description=f'plasma colormap · max {MAX_DIST_KM:.0f} km',
    )
