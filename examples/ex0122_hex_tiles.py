# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Printable Hex9 hexagon tiles — the octahedral "jigsaw", at any level.

A tile is one H9 hexagon: the 1–2 octant faces it spans, developed flat via the
net-independent OctahedralNet.local_layout, with land/ocean clipped to it as a
true vector SVG — printable at any scale, for cutting out and stitching/tabbing
into a flat jigsaw or a 3D octahedral solid.  Level 0 = the 12 edge-centred base
cells; level 1 = their 108 children, and so on (no hexagon ever crosses more than
one octahedron edge, so a tile never wraps a cone vertex).

  * Geometry: hhg9.geo.vector.tile_frame + project_fill_to_tile (holes preserved).
  * Land cuts cleanly to the hexagon; ocean carries continents as holes; the
    hexagon outline is drawn *external* to the cell so its inner edge is the cut.
  * The H9 subgrid (3 levels below the tile — progressively finer + fainter) is
    overlaid and clipped, so cells straddling the edge read as half-hexes.
  * Tiles are named by their H9 hex address (e.g. 4, 42), tied to the binning,
    and (default) oriented North-up; --label / --sublabels print the addresses.
  * CVD-safe palette: pale gold land / clear blue sea (blue↔gold luminance axis).

Outputs:
  output/hextiles/ex0122_{address}.svg          # ex0122_4.svg, ex0122_42.svg, …

Usage:
  python ex0122_hex_tiles.py                      # all 108 L1 tiles
  python ex0122_hex_tiles.py --level 0            # all 12 L0 tiles
  python ex0122_hex_tiles.py --tiles 42           # the single L1 hexagon '42'
  python ex0122_hex_tiles.py --tiles 430 431 432 --label --sublabels   # a labelled UK jigsaw (L2)
  python ex0122_hex_tiles.py --tiles 42 --no-north                     # keep the native develop

Last Tested
-----------
29 Jun 2026 0.1.3a0 (tiles by hex address via --tiles; North-up + --label/--sublabels;
            ne_10m land+islands+lakes; seam-switch + geo_bbox clip)
"""
import argparse
import os

os.environ.setdefault('SHAPE_RESTORE_SHX', 'YES')      # rebuild a missing .shx sidecar

import numpy as np
from shapely.affinity import translate, rotate
from shapely.geometry import Point

from hhg9 import Registrar
from hhg9.geo.vector import (load_shape_polygons, tile_frame, project_fill_to_tile,
                             tile_subgrid, tile_labels)
from hhg9.geo.naturalearth import ne_layer

os.makedirs('output/hextiles', exist_ok=True)

HERE = os.path.dirname(__file__)
# Sea is the hexagon backdrop, so no ocean polygon is needed (it would be sea
# under land — redundant); the Caspian/endorheic seas are excluded from `land`
# so they read as backdrop sea.  Land + minor islands are gold; lakes are sea.
# Natural Earth layers (public domain); downloaded to examples/src/landmasses
# on first use, prompting before it reaches the network.
LAND_FILES = [ne_layer(n, '10m') for n in ('land', 'minor_islands')]
LAKE_FILE = ne_layer('lakes', '10m')

TILE = 2
PIXELS = 1000           # long edge of the tile artwork, in px
STEP_M = 50_000         # geodesic densification step (m)

# CVD-safe palette: blue↔gold luminance axis (pale gold land / clear blue sea),
# far apart in lightness, no warm-on-warm or blue-green.
LAND_FILL = '#efdca0'   # pale gold
SEA_FILL = '#4f86c6'    # medium blue
EDGE_STROKE = '#222222'
EDGE_WIDTH = 1.2        # fine cut line, drawn external to the hexagon
GRID_STROKE = '#222222'
# H9 subgrid 3 layers below the tile, progressively finer + fainter
# (tile_level+1 boldest → +3 finest): (stroke width, opacity).
GRID_STYLE = [(1.3, 0.55), (0.7, 0.38), (0.35, 0.22)]
PAD = 24.0
# Address labels: dark text + white halo reads on both land and sea (CVD-safe,
# luminance-only).  Sizes are a fraction of the tile artwork's long edge.
LABEL_COLOR = '#222222'
LABEL_FRAC = 0.085      # tile address height / pixels
SUB_FRAC = 0.024        # +1 child address height / pixels


def _ring_d(coords, scale, oy):
    """SVG path data for one ring: scale to px and flip y."""
    c = (np.asarray(coords) * scale) * np.array([1, -1]) + np.array([0, oy])
    return 'M ' + ' L '.join(f'{x:.3f},{y:.3f}' for x, y in c) + ' Z'


def _path(poly, scale, oy, **attrs):
    """SVG <path> for a shapely polygon (exterior + holes, evenodd fill)."""
    d = _ring_d(poly.exterior.coords, scale, oy)
    for hole in poly.interiors:
        d += ' ' + _ring_d(hole.coords, scale, oy)
    a = ' '.join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
    return f'<path d="{d}" {a}/>'


def _text(x, y, s, size):
    """SVG centred address label with a white halo (reads on land or sea)."""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size:.1f}" text-anchor="middle" '
            f'dominant-baseline="central" font-family="sans-serif" font-weight="bold" '
            f'fill="{LABEL_COLOR}" stroke="#ffffff" stroke-width="{max(0.6, size * 0.1):.1f}" '
            f'paint-order="stroke">{s}</text>')


def run(tile=TILE, level: int = 0, pixels: int = PIXELS,
        label: bool = False, sublabels: bool = False, north: bool = True) -> None:
    reg = Registrar()
    frame = tile_frame(reg, tile, level)        # tile: mesh index or hex address ('42')
    hexagon = frame['hexagon']
    print(f'tile {frame["address"]} (L{level}): octant faces {frame["octants"]}')

    def fill(srcs):
        return [p for g in srcs
                for p in project_fill_to_tile(g, reg, frame, max_step_m=STEP_M)]

    land = [p for f in LAND_FILES for p in fill(load_shape_polygons(f))]
    lakes = fill(load_shape_polygons(LAKE_FILE))      # painted (sea) over land
    sub_layers = [level + 1 + i for i in range(len(GRID_STYLE))]   # 3 levels below
    subgrid = tile_subgrid(reg, frame, layers=tuple(sub_layers))
    labels = tile_labels(reg, frame, sub=sublabels) if (label or sublabels) else []
    print(f'  land pieces {len(land)}, lake pieces {len(lakes)}, '
          f'subgrid cells {sum(len(v) for v in subgrid.values())}')

    # North-up: rotate every shape about the hexagon centre so geographic North
    # is vertical — nicer for reading labels and assembling a physical jigsaw.
    if north and abs(frame['north_deg']) > 1e-6:
        ang, ctr = frame['north_deg'], hexagon.centroid
        rot = lambda g: rotate(g, ang, origin=ctr)
        hexagon = rot(hexagon)
        land = [rot(p) for p in land]
        lakes = [rot(p) for p in lakes]
        subgrid = {L: [rot(p) for p in cs] for L, cs in subgrid.items()}
        labels = [(a, np.asarray(rot(Point(c)).coords)[0], d) for a, c, d in labels]

    # Layout: recentre to the hexagon bbox, scale to `pixels`, pad.
    minx, miny, maxx, maxy = hexagon.bounds
    scale = pixels / max(maxx - minx, maxy - miny)
    sh = lambda p: translate(p, xoff=-minx, yoff=-miny)
    w = (maxx - minx) * scale + 2 * PAD
    h = (maxy - miny) * scale + 2 * PAD
    oy = h - 2 * PAD

    # External cut outline: buffer the hexagon outward by half the stroke width
    # (mitre joins keep the corners sharp) so the stroke's *inner* edge is the
    # true hexagon — i.e. you cut along the inside of the printed line.
    half = (EDGE_WIDTH / 2) / scale
    outline = sh(hexagon).buffer(half, join_style=2, mitre_limit=10)
    hex_d = _ring_d(sh(hexagon).exterior.coords, scale, oy)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.0f}" '
             f'height="{h:.0f}" viewBox="0 0 {w:.0f} {h:.0f}">',
             f'<defs><clipPath id="tilehex"><path d="{hex_d}"/></clipPath></defs>',
             f'<g transform="translate({PAD},{PAD})">',
             _path(sh(hexagon), scale, oy, fill=SEA_FILL, stroke='none')]   # sea backdrop
    parts += [_path(sh(p), scale, oy, fill=LAND_FILL, stroke='none') for p in land]
    parts += [_path(sh(p), scale, oy, fill=SEA_FILL, stroke='none') for p in lakes]

    # H9 subgrid, clipped to the hexagon (straddling cells become half-hexes).
    parts.append('<g clip-path="url(#tilehex)" fill="none" '
                 f'stroke="{GRID_STROKE}" stroke-linejoin="round">')
    for lay, (lw, op) in zip(sub_layers, GRID_STYLE):
        parts.append(f'<g stroke-width="{lw}" stroke-opacity="{op}">')
        for cell in subgrid.get(lay, []):
            parts.append(f'<path d="{_ring_d(sh(cell).exterior.coords, scale, oy)}"/>')
        parts.append('</g>')
    parts.append('</g>')

    parts.append(_path(outline, scale, oy, fill='none',
                       stroke=EDGE_STROKE, stroke_width=EDGE_WIDTH))

    # Address labels on top (children first, tile last so it stays readable).
    for a, c, d in sorted(labels, key=lambda t: -t[2]):
        if (d == 0 and not label) or (d == 1 and not sublabels):
            continue
        lx, ly = (c[0] - minx) * scale, -(c[1] - miny) * scale + oy
        parts.append(_text(lx, ly, a, pixels * (LABEL_FRAC if d == 0 else SUB_FRAC)))
    parts += ['</g>', '</svg>']

    out = os.path.join('output/hextiles', f'ex0122_{frame["address"]}.svg')
    with open(out, 'w') as f:
        f.write('\n'.join(parts))
    print(f'Saved: {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--level', type=int, default=1,
                    help='render ALL tiles at this level (0 → 12, 1 → 108); ignored if --tiles given')
    ap.add_argument('--tiles', nargs='+', default=None, metavar='ADDR',
                    help="hex addresses to render, e.g. --tiles 430 431 432 (a UK jigsaw); "
                         "level is inferred per address (len-1: '4'→L0, '42'→L1, '430'→L2)")
    ap.add_argument('--pixels', type=int, default=PIXELS)
    ap.add_argument('--label', action='store_true', help="print the tile's hex address at its centre")
    ap.add_argument('--sublabels', action='store_true',
                    help='also print the +1 child cell addresses (smaller)')
    ap.add_argument('--north', action=argparse.BooleanOptionalAction, default=True,
                    help='orient each tile North-up (default; --no-north keeps the native develop)')
    args = ap.parse_args()
    kw = dict(pixels=args.pixels, label=args.label, sublabels=args.sublabels, north=args.north)
    if args.tiles is not None:
        for addr in args.tiles:
            run(addr, level=len(addr) - 1, **kw)             # level from address length
    else:
        for tile in range(12 * 9 ** args.level):   # 12, 108, 972, …  (mesh index)
            run(tile, args.level, **kw)
