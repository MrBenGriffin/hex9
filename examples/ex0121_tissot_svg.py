# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Tissot indicatrix SVG — vector projection map.

Projects Natural Earth 50 m land polygons, a 30° graticule, and Tissot
indicatrices through the H9 octahedral projection into the chosen net
layout.  Outputs a true vector SVG with the H9 hex grid overlaid.

Layers (bottom to top):
  1. Land polygons     — filled, no stroke
  1b. Lake polygons    — filled SEA_FILL (Caspian, Great Lakes, etc.)
  2. Graticule         — thin grey lines
  3. Tissot circles    — red outlines, seam-split into arcs
  4. Hex L1            — green bold outlines
  5. Hex L2            — green fine outlines

Outputs:
  output/ex0121_tissot_{flavour}.svg

Usage:
  python ex0121_tissot_svg.py
  python ex0121_tissot_svg.py --flavour mortar
  python ex0121_tissot_svg.py --flavour butterfly:0500

Last Tested
28 Jun 2026 0.1.3a0 (passed) — octant-clip fills; butterfly clean, rhombus
            (c2-split) fills guarded/skipped pending c2/cell clip.
16 Jun 2026 0.1.3a0 (passed) 158.4s
20 Apr 2026
"""

import argparse
import os
from pathlib import Path

import numpy as np
from svgelements import SVG, Group, Polygon, Polyline, Matrix

from hhg9 import Registrar
from hhg9.h9 import H9O, H9P, H9K
from hhg9.h9.grid import HexMesh
from hhg9.geo.vector import (load_shape_polygons, graticule_paths,
                             tissot_circle, project_path, split_path_at_seams,
                             clip_polygon_to_octants, is_c2_split)

os.makedirs('output', exist_ok=True)

# ── Configuration ────────────────────────────────────────────────────────────

# FLAVOUR = 'rhombus'  # 'butterfly:0500'
FLAVOUR = 'butterfly:0500'      # 'butterfly:0500'
PIXELS = 1200  # triangle side in pixels → controls SVG canvas size

LAND_FILE = os.path.join(os.path.dirname(__file__), '../preparatory/landmasses/ne_50m_land.shp')
LAKE_FILE = os.path.join(os.path.dirname(__file__), '../preparatory/landmasses/ne_50m_lakes.shp')
SEAS_FILE = os.path.join(os.path.dirname(__file__), '../preparatory/landmasses/ne_50m_ocean.shp')

# Graticule
GRAT_STEP = 10  # degrees between parallels / meridians

# Tissot circles — placed at 30° graticule intersections
TISSOT_LATS = list(range(-60, 61, 30))  # -60 … 60
TISSOT_LONS = list(range(-180, 181, 30))  # -180 … 180
TISSOT_R_M = 500_000  # circle radius in metres (≈ 4.5° great-circle)
TISSOT_PTS = 72  # vertices per circle before projection

# Densification
STEP_LAND_M = 50_000  # max geodesic step for land rings (m)
STEP_GRAT_M = 100_000  # max geodesic step for graticule lines (m)
STEP_TISS_M = 20_000  # max geodesic step for Tissot arcs (m)

# Seam-split threshold — W/9 safe for steps above; W/27 for finer densification
SEAM_THRESH = H9K.derived.W / 9

# SVG style
LAND_FILL = '#d8cfc0'
LAND_STROKE = 'none'
SEA_FILL = '#b8d4e8'
GRAT_STROKE = '#888888'
GRAT_WIDTH = 0.4
TISS_STROKE = 'none'
TISS_FILL = '#cc2200'
TISS_WIDTH = 0.8
HEX_STROKE = '#1a6b1a'
HEX_W1 = 0.9  # L1 linewidth
HEX_W2 = 0.6  # L2 linewidth
HEX_W3 = 0.3  # L3 linewidth


# ── Helpers ──────────────────────────────────────────────────────────────────

def _poly(pts_scaled, **style):
    """SVG Polygon from (N, 2) array in pixel coords."""
    return Polygon(*pts_scaled.tolist(), **style)


def _line(pts_scaled, **style):
    """SVG Polyline from (N, 2) array in pixel coords."""
    return Polyline(*pts_scaled.tolist(), **style)


def _scale_flip(segs, scale, img_h):
    """Scale n_oct coords to pixels and flip y for SVG."""
    return [(s * scale) * np.array([1, -1]) + np.array([0, img_h])
            for s in segs]


# ── Main render ──────────────────────────────────────────────────────────────

def run(flavour: str = FLAVOUR, pixels: int = PIXELS) -> None:
    from hhg9.h9.region import recover_stats_reset, recover_stats_report

    reg = Registrar()
    n_oct = reg.domain(f'n_oct:{flavour}')
    b_oct = reg.domain('b_oct')
    recover_stats_reset()

    img_w, img_h = n_oct.image_dims(pixels=pixels)
    scale = img_w / n_oct.wi

    def proj(latlon, step_m, closed=False):
        return project_path(latlon, reg, n_oct,
                            max_step_m=step_m, closed=closed,
                            threshold=SEAM_THRESH)

    # ── SVG canvas ───────────────────────────────────────────────────────────
    canvas = SVG(width=img_w, height=img_h,
                 viewBox=f'0 0 {img_w} {img_h}')
    flip = Matrix(1, 0, 0, -1, 0, img_h)

    g_land = Group(id='land')
    g_lakes = Group(id='lakes')
    g_seas = Group(id='seas')
    g_grat = Group(id='graticule')
    g_tiss = Group(id='tissot')
    g_hex1 = Group(id='hex_l1')
    g_hex2 = Group(id='hex_l2')
    g_hex3 = Group(id='hex_l3')

    # Octant-clip fills are only correct for simple (non-c2-split) layouts.
    # c2-split layouts (e.g. rhombus) have seams *inside* an octant, so the
    # octant is not a fine enough clip fundamental — skip fills there until a
    # c2/cell-level clip exists, rather than emit broken fan artifacts.
    fills_ok = not is_c2_split(n_oct)
    if not fills_ok:
        print(f'WARNING: layout "{flavour}" is c2-split — octant-clip fills not '
              'yet supported (needs c2/cell clip). Skipping land/ocean/lake/tissot fills.')

    def fill_group(shp_file, grp, fill):
        """Octant-clip each ring, project, and append closed fills to grp."""
        for ring in load_shape_polygons(shp_file):
            for sub in clip_polygon_to_octants(ring):
                for seg in proj(sub, STEP_LAND_M, closed=True):
                    grp.append(_poly(seg * scale,
                                     fill=fill, stroke=LAND_STROKE,
                                     stroke_width=0))

    # ── 1a/b/c. Land / ocean / lake polygons ──────────────────────────────────
    if fills_ok:
        print('Land polygons...')
        fill_group(LAND_FILE, g_land, LAND_FILL)
        fill_group(SEAS_FILE, g_seas, SEA_FILL)
        fill_group(LAKE_FILE, g_lakes, SEA_FILL)
    # ── 2. Graticule ─────────────────────────────────────────────────────────
    print('Graticule...')
    for path in graticule_paths(GRAT_STEP, GRAT_STEP):
        for seg in proj(path, STEP_GRAT_M, closed=False):
            pts = seg * scale
            g_grat.append(_line(pts,
                                fill='none', stroke=GRAT_STROKE,
                                stroke_width=GRAT_WIDTH))

    # ── 3. Tissot circles ────────────────────────────────────────────────────
    if fills_ok:
        print('Tissot circles...')
        for lat in TISSOT_LATS:
            for lon in TISSOT_LONS:
                circle = tissot_circle(lat, lon, TISSOT_R_M, TISSOT_PTS)
                # octant-clip so seam-straddling circles close cleanly per octant
                for sub in clip_polygon_to_octants(circle):
                    for seg in proj(sub, STEP_TISS_M, closed=True):
                        g_tiss.append(_poly(seg * scale,
                                            fill=TISS_FILL, stroke=TISS_STROKE,
                                            stroke_width=TISS_WIDTH, alpha=0.25))

    # ── 4 & 5. Hex overlay ───────────────────────────────────────────────────
    print('Hex overlay...')
    mesh = HexMesh.create([1, 2, 3], reg)
    n_pts = reg.project(mesh.pts, [b_oct, n_oct])
    face_edge = H9K.derived.W
    for layer, grp, lw in [(1, g_hex1, HEX_W1), (2, g_hex2, HEX_W2), (3, g_hex3, HEX_W3)]:
        hexes = n_pts.coords[mesh[layer]].reshape(-1, 6, 2)
        edges = np.roll(hexes, -1, axis=1) - hexes
        max_edge = np.hypot(edges[:, :, 0], edges[:, :, 1]).max(axis=1)
        valid = max_edge <= (face_edge / (3 ** layer))
        for hex_pts in hexes[valid]:
            pts = hex_pts * scale
            grp.append(_poly(pts,
                             fill='none', stroke=HEX_STROKE,
                             stroke_width=lw,
                             vector_effect='non-scaling-stroke'))

    # ── Assemble, flip y, write ──────────────────────────────────────────────
    for grp in [g_land, g_seas, g_lakes, g_grat, g_tiss, g_hex1, g_hex2, g_hex3]:
        grp *= flip
        canvas.append(grp)

    safe = flavour.replace(':', '_')
    f_name = f'output/ex0121_tissot_50_warp_file_{safe}.svg'
    canvas.write_xml(f_name)
    print(f'Saved: {f_name}')
    recover_stats_report()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--flavour', default=FLAVOUR)
    parser.add_argument('--pixels', type=int, default=PIXELS)
    args = parser.parse_args()
    run(args.flavour, args.pixels)
