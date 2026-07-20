# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Hex grid over satellite imagery, anywhere on Earth.

Give it a latitude/longitude and a window size and it renders nested Hex9
layers over Esri World Imagery. Nothing is stored locally beyond a tile cache,
so unlike ex0251/ex0252 — which composite specific public-domain rasters over
fixed places — this one runs anywhere without obtaining any data first.

It needs a network connection on first run. Tiles are cached under
``wmts_cache/``, so re-runs of the same view are offline and fast.

Pipeline (all shared library code — this example is only configuration):
    make_wmts_source   imagery sampler, from hhg9.rendering.imagery
    make_backdrop      the same sampler rasterised per pixel
    LayerSpec/Compositor   the hex layers
    plot_hex           draws it, rotates north up, and renders the credit

The imagery credit travels on the sampler (``.attribution``) and is collected
by ``credits_of``, so the Esri attribution cannot be lost by editing the title.

Note on the fixed net: the window is placed into a preselected ``n_oct``
layout, so a region straddling an octant seam (the equator, or 0/90/180/270°
longitude) will not frame correctly — see examples/INDEX.md. Keep the window
inside one octant, or use the dynamic local net of ex0260.

Last Tested
20 Jul 2026 0.1.3a0 (passed — rewritten on Compositor + Esri backdrop)
16 Jun 2026 0.1.3a0 (passed) 300.6s — pre-Compositor version, local NLCD
"""
import os
import math
import numpy as np

from hhg9 import Points, Registrar
from hhg9.algorithms.distance import wgs84_offset, densify_quad_geodesic_by_step
from hhg9.rendering.imagery import make_wmts_source, credits_of
from hhg9.rendering.composition import (LayerSpec, Compositor, make_backdrop,
                                        estimate_backdrop_size)
from hhg9.rendering.render import plot_hex

HERE = os.path.dirname(os.path.abspath(__file__))

ESRI_WMTS = ('https://services.arcgisonline.com/arcgis/rest/services/'
             'World_Imagery/MapServer/WMTS/1.0.0/WMTSCapabilities.xml')
ESRI_CREDIT = 'Esri World Imagery · Esri, Maxar, Earthstar Geographics'

# Somewhere recognisable, and comfortably inside one octant.
places = {
    'GGB': ('Golden Gate Bridge, CA', (37.8199, -122.4783), 'diamonds'),
    'AAK': ('Aniakchak, AK', (57.1255, -156.8763), 'diamonds'),
    'NWO': ('New Orleans, LA', (29.9511,  -90.0715), 'diamonds'),
    'YEL': ('Yellowstone, WY', (44.4280, -110.5885), 'diamonds'),
    'EVG': ('Everglades, FL', (25.4687,  -80.4776), 'diamonds'),
    'STN': ('Stonehenge, UK', (51.1789,   -1.8262), 'butterfly:0500'),
    'SHB': ('Shibuya, Tokyo', (35.6595,  139.7005), 'diamonds'),

}
FOCUS = 'STN'
NAME, CENTRE, FLAVOUR = places[FOCUS]
SIZE_M = 2_000.0          # full width and height of the window, in metres
BACKDROP_M_PER_PX = 0.5   # backdrop ground resolution; Esri imagery is ~0.3–0.6 m


def demo_out(name: str) -> str:
    """Resolve an output path beside this script, whatever the working directory."""
    out = os.path.join(HERE, 'output')
    os.makedirs(out, exist_ok=True)
    return os.path.join(out, name)


if __name__ == '__main__':
    print('Initialising')
    rg = Registrar()
    g_gcd = rg.domain('g_gcd')
    c_ell = rg.domain('c_ell')
    b_oct = rg.domain('b_oct')
    c_oct = rg.domain('c_oct')
    n_oct = rg.domain(f'n_oct:{FLAVOUR}')   # diamonds keeps north pole north

    print('Building imagery source')
    imagery = make_wmts_source(ESRI_WMTS, 'World_Imagery', rg, b_oct,
                               cache_dir=os.path.join(HERE, 'wmts_cache'),
                               attribution=ESRI_CREDIT)

    print('Building area polygon')
    centre = np.atleast_2d(CENTRE)
    diag = (SIZE_M / 2.0) * math.sqrt(2.0)          # centre → corner
    az = np.array([45.0, 135.0, 225.0, 315.0], dtype=np.float64)
    corners = wgs84_offset(np.repeat(centre, 4, axis=0), az, np.full(4, diag))
    qx = densify_quad_geodesic_by_step(corners, max_step_m=25)
    q_dens = Points(qx, g_gcd)

    # Display bounds in n_oct
    ntt = rg.project(q_dens, [g_gcd, b_oct, n_oct])
    nlt = float(ntt.coords[:, 0].min())
    nrt = float(ntt.coords[:, 0].max())
    nbt = float(ntt.coords[:, 1].min())
    ntp = float(ntt.coords[:, 1].max())
    bbox_n = (ntp, nlt, nbt, nrt)
    fill_n = Points(np.array([[nlt, nbt], [nrt, nbt], [nrt, ntp], [nlt, ntp]]), n_oct)

    # Local north, from a small northward step through the full chain
    lat, lon = CENTRE
    ctr_n = rg.project(Points(np.array([[lat, lon]]), g_gcd), [g_gcd, b_oct, n_oct])
    nth_n = rg.project(Points(np.array([[lat + 0.01, lon]]), g_gcd), [g_gcd, b_oct, n_oct])
    north_dir = nth_n.coords[0] - ctr_n.coords[0]

    px_w, px_h = estimate_backdrop_size(bbox_n, BACKDROP_M_PER_PX)
    print(f'Sampling backdrop ({px_w}×{px_h})')
    backdrop = make_backdrop(rg, b_oct, n_oct, bbox_n, px_w, px_h, imagery)

    # Three nested levels, coarse labelled. Aperture 9 means each step is 3×
    # finer across, so three levels is about all that stays legible at once:
    # here L9 ≈ 356 m across (~6 per window), L10 ≈ 119 m, L11 ≈ 40 m. Going a
    # level finer (L12 ≈ 13 m) puts 34k hexes over 2 km and buries the imagery.
    specs = [
        LayerSpec(level=9, kind='outline', labels=True, threshold=2, reference=True,
                  style={'lw': 0.9, 'ec': '#ffffffd0', 'label_size': 20}),
        LayerSpec(level=10, kind='outline',
                  style={'lw': 0.6, 'ec': '#ffffff90', 'label_size': 9}),
        LayerSpec(level=11, kind='outline',
                  style={'lw': 0.3, 'ec': '#ffffff45'}),
    ]
    print('Running compositor')
    composed = Compositor(rg, b_oct, n_oct, specs).run(fill_n)
    print('  layers:', [(c.spec.level, c.count) for c in composed])

    print('Plotting')
    plot_hex(
        composed,
        save_path=demo_out(f'ex0250_imagery_{FOCUS}_{int(SIZE_M)}m.png'),
        bbox=bbox_n,
        backdrop=backdrop,
        draw_bbox=False,
        north_dir=north_dir,
        rotate_north=True,
        credits=credits_of(imagery),      # backdrop credit — no layer source carries it
        title=f'{NAME}',
        description=f'{CENTRE[0]:.4f}, {CENTRE[1]:.4f} · {SIZE_M:.0f} m · L9–L11',
        dpi=200,
    )
