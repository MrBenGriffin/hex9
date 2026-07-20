# Part of the Hex9 (H9) Project
# Copyright ©2026, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Greenwich Park — a real polygon, across the prime meridian.

The Compositor end to end on the hard case: a 93-vertex non-convex boundary
that straddles 0° longitude, which is an octant seam. A fixed n_oct net cannot
frame that (the region has no single b_oct plot), so this uses the **dynamic
local net** — octants hinged flat from a fundamental — and lets every other
part of the pipeline fall out of it:

    Compositor(..., local=True)   fitted unfolding, tear-free across the seam
    nest (default)                coarse layers closed over OWNERSHIP ancestry,
                                  so no fine hex is drawn without its parent
    make_local_backdrop           imagery sampled under the fitted layout
    make_wmts_source              Esri World Imagery (network on first run)
    rotate_north                  scene quarter-turned so north is within ±45°
    credits                       attribution carried on the sampler itself

The polygon is imported from ex0262 rather than duplicated — that example draws
the same boundary at a single fine layer without imagery.

Ownership, not lineage: the coarse cell containing a fine cell is *not* its
address prefix on the hexagon band (they differ for roughly a sixth of cells),
so the nest is closed with ``x_adr_cell_ancestor``.  See INDEX.md.

Needs a network connection on first run; tiles cache under ``wmts_cache/``.

Last Tested
20 Jul 2026 0.1.3a0 (passed)
"""
import os
import numpy as np

from hhg9 import Registrar, Points
from hhg9.rendering.composition import (LayerSpec, Compositor,
                                        make_local_backdrop,
                                        estimate_backdrop_size)
from hhg9.rendering.imagery import make_wmts_source, credits_of
from hhg9.rendering.render import plot_hex

from ex0262_greenwich_seam import GREENWICH

HERE = os.path.dirname(os.path.abspath(__file__))

ESRI_WMTS = ('https://services.arcgisonline.com/arcgis/rest/services/'
             'World_Imagery/MapServer/WMTS/1.0.0/WMTSCapabilities.xml')
ESRI_CREDIT = 'Esri World Imagery · Esri, Maxar, Earthstar Geographics'

LEVELS = (11, 12, 13)      # L11 ≈ 22.8 m/side, L13 ≈ 2.5 m/side
BACKDROP_M_PER_PX = 0.3    # Esri imagery is ~0.3 m here


def demo_out(name: str) -> str:
    """Resolve an output path beside this script, whatever the working directory."""
    out = os.path.join(HERE, 'output')
    os.makedirs(out, exist_ok=True)
    return os.path.join(out, name)


if __name__ == '__main__':
    print('Initialising')
    rg = Registrar()
    g_gcd = rg.domain('g_gcd')
    b_oct = rg.domain('b_oct')
    n_oct = rg.domain('n_oct:diamonds')

    print('Building imagery source')
    imagery = make_wmts_source(ESRI_WMTS, 'World_Imagery', rg, b_oct,
                               cache_dir=os.path.join(HERE, 'wmts_cache'),
                               attribution=ESRI_CREDIT)

    specs = [
        LayerSpec(level=LEVELS[0], kind='outline', labels=True, reference=True,
                  style={'lw': 2.4, 'ec': '#ffcc00d0', 'label_size': 6,
                         'label_color': '#ffcc00'}),
        LayerSpec(level=LEVELS[1], kind='outline',
                  style={'lw': 0.8, 'ec': '#66ccff90'}),
        LayerSpec(level=LEVELS[2], kind='outline',
                  style={'lw': 0.2, 'ec': '#ffffff40'}),
    ]

    # The polygon goes in as Points — create_clipped projects it, so the caller
    # need not convert. local=True fits the net to the region; nest defaults on.
    print('Running compositor (local net)')
    comp = Compositor(rg, b_oct, n_oct, specs, local=True)
    layers = comp.run(Points(GREENWICH, g_gcd))
    print('  counts:', {cl.spec.level: cl.count for cl in layers})
    print(f'  octants {sorted(set(np.concatenate([cl.oids for cl in layers]).tolist()))}'
          f' · seam residual {comp.local_residual:.1e} · dropped {comp.local_dropped}')

    # Frame the placed geometry (the local net has no global bbox to inherit).
    allv = np.vstack([np.asarray(cl.verts.coords) for cl in layers])
    lt, rt = float(allv[:, 0].min()), float(allv[:, 0].max())
    bt, tp = float(allv[:, 1].min()), float(allv[:, 1].max())
    pad = 0.02 * max(rt - lt, tp - bt)
    bbox = (tp + pad, lt - pad, bt - pad, rt + pad)

    px_w, px_h = estimate_backdrop_size(bbox, BACKDROP_M_PER_PX)
    print(f'Sampling backdrop ({px_w}×{px_h})')
    backdrop = make_local_backdrop(rg, b_oct, comp.layout.layout,
                                   bbox, px_w, px_h, imagery)

    # Local north: a small northward step, placed through the SAME layout.
    lat, lon = float(GREENWICH[:, 0].mean()), float(GREENWICH[:, 1].mean())
    c0 = rg.project(Points(np.array([[lat, lon]]), g_gcd), [g_gcd, b_oct])
    c1 = rg.project(Points(np.array([[lat + 0.002, lon]]), g_gcd), [g_gcd, b_oct])
    p0 = n_oct.place(c0.coords, c0.oid, comp.layout.layout)[0]
    p1 = n_oct.place(c1.coords, c1.oid, comp.layout.layout)[0]

    print('Plotting')
    plot_hex(
        layers,
        save_path=demo_out('ex0254_greenwich.png'),
        bbox=bbox,
        backdrop=backdrop,
        draw_bbox=False,
        north_dir=p1 - p0,
        rotate_north=True,
        credits=credits_of(imagery),
        title='Greenwich Park — Hex9 L11–L13 across the prime meridian',
        description='local net · ownership nest · Esri imagery · north auto-rotated',
        dpi=150,
    )
