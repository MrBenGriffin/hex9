# Part of the Hex9 (H9) Project
# Copyright ©2026, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Engraving data for the 3D-printed hinged Hex9 tile (h9_hinged_hex.scad).

Reuses the ex0122 tile pipeline (tile_frame / project_fill_to_tile /
tile_subgrid) to emit, for one L0 tile, land edges + L1 cell boundaries as
pre-buffered stroke polygons in millimetres, hinge-on-X aligned for the SCAD
model. Editorial cut: land edges make it a planet, L1 shows the hierarchy;
graticule and deeper grids are below nozzle resolution at true L16 scale
(1 mm ~ 45 km).

Alignment contract with h9_hinged_hex.scad: the hexagon is centred at the
origin with its long diagonal — the octahedron-edge fold — exactly along X,
vertices at (±side, 0), side scaled so the printed hexagon has the
authalic-ideal area of a cell at --level. Coordinates are emitted as seen
from OUTSIDE the assembled solid (the ex0122/paper-map view); the SCAD
mirrors them in X when cutting the underside so they read correctly on the
printed outer face.

Per tile this writes THREE files:
  engrave_<addr>.scad — the data (pure assignments; opening it alone gives
                        "No top level geometry", which is expected)
  tile_<addr>.scad    — an openable per-tile model: includes the model, then
                        the tile data, whose assignments override the model's
                        engrave_data.scad include (last assignment wins —
                        verified byte-identical against the copy route)
  engrave_data.scad   — the current selection for h9_hinged_hex.scad itself
                        (single-tile runs and --use update it; OpenSCAD cannot
                        parameterise an include path, so selection is a copy)

Open tile_<addr>.scad for a quick look or CLI render of a specific tile; open
h9_hinged_hex.scad (+ --use to pick the tile) when you want the customizer
panel, which only scans the main file.

Usage (from anywhere; data paths are package-relative):
  python3 gen_engrave.py --tile 4            # L0 tile by hex address '0'..'b'
  python3 gen_engrave.py --tile all          # generate all 12 tiles' files
  python3 gen_engrave.py --use 7             # switch engrave_data.scad to tile 7
  python3 gen_engrave.py --tile all --render # + batch STLs into output/
  python3 gen_engrave.py --tile 4 --lakes    # also engrave lake shorelines
  python3 gen_engrave.py --list              # addresses + geographic extents

--render finds OpenSCAD on PATH or in /Applications and renders each
tile_<addr>.scad, so a batch render never touches the current selection.
"""
import argparse
import math
import os
import shutil
import subprocess
from pathlib import Path

os.environ.setdefault('SHAPE_RESTORE_SHX', 'YES')

import numpy as np
from shapely.affinity import rotate, scale as ascale, translate
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from hhg9 import Points, Registrar
from hhg9.h9 import H9O
from hhg9.geo.naturalearth import ne_layer
from hhg9.geo.vector import (load_shape_polygons, project_fill_to_tile,
                             tile_frame, tile_subgrid, _switch_seam_oids)

HERE = Path(__file__).resolve().parent
R_AUTH = 6371007.1810          # WGS84 authalic radius, m (matches the SCAD)
STEP_M = 50_000                # geodesic densification step

LAND_W = 0.5                   # engraved stroke widths, mm
GRID_W = 0.8
SIMPLIFY_MM = 0.12             # pre-buffer line simplification
POLY_SIMPLIFY_MM = 0.06        # post-buffer polygon simplification
RIM_EPS_MM = 0.3               # strip boundary segments lying on the hexagon rim
MIN_AREA_MM2 = 0.4             # cull unprintable specks


def side_mm(level: int) -> float:
    """Regular-hex side (mm) with the authalic-ideal area of a level-L cell."""
    area = 4 * math.pi * R_AUTH ** 2 / (12 * 9 ** level)
    return 1000 * math.sqrt(2 * area / (3 * math.sqrt(3)))


def fold_line_points(reg, frame):
    """Two points on the tile's fold line (the shared octahedron edge), in
    frame coordinates.

    The two octants' ECEF signs (H9O.oid_cmp) agree on exactly two axes; the
    shared edge is the quarter great-circle between those two cardinal
    vertices. Two interior points of that arc are projected through the same
    seam-aware path the fills use (they lie exactly on the seam)."""
    octs = frame['octants']
    s0, s1 = H9O.oid_cmp[octs[0]], H9O.oid_cmp[octs[1]]
    axes = np.where(s0 == s1)[0]
    va, vb = (np.eye(3)[a] * float(s0[a]) for a in axes)
    latlon = []
    for t in (0.3, 0.7):                       # slerp on the 90-degree arc
        v = va * math.cos(math.radians(90 * t)) + vb * math.sin(math.radians(90 * t))
        latlon.append([math.degrees(math.asin(v[2])),
                       math.degrees(math.atan2(v[1], v[0]))])
    pb = reg.project(Points(np.asarray(latlon), 'g_gcd'), ['g_gcd', 'b_oct'])
    coords, oids = _switch_seam_oids(pb.coords.copy(), np.asarray(pb.oid).copy(), octs)
    xy = frame['n_oct'].place(coords, oids, frame['layout'])
    assert np.isfinite(xy).all(), 'fold-line points failed to place'
    return xy


def hinge_transform(frame, xy_fold, s_target):
    """Affine (as a closure) putting the fold on X, centre at origin, side =
    s_target mm. Two-step: the projected fold points pick WHICH long diagonal;
    the hexagon's own vertices then set the angle exactly."""
    hexagon = frame['hexagon']
    c = hexagon.centroid
    theta = math.degrees(math.atan2(*(xy_fold[1] - xy_fold[0])[::-1]))

    def apply(g, ang):
        g = rotate(g, -ang, origin=c)
        g = translate(g, xoff=-c.x, yoff=-c.y)
        return g

    hx = apply(hexagon, theta)
    verts = np.asarray(hx.exterior.coords)[:-1]
    diag = verts[np.argsort(np.abs(verts[:, 1]))[:2]]        # the fold vertices
    fine = math.degrees(math.atan2(*(diag[1] - diag[0])[::-1])) % 180.0
    fine = fine - 180.0 if fine > 90.0 else fine             # smallest correction
    ang = theta + fine
    side = np.abs(apply(hexagon, ang).exterior.coords[0][0])  # vertex 0 hits (±side, 0)?
    hx = apply(hexagon, ang)
    verts = np.asarray(hx.exterior.coords)[:-1]
    diag = verts[np.argsort(np.abs(verts[:, 1]))[:2]]
    assert np.abs(diag[:, 1]).max() < 1e-6 * s_target, 'fold diagonal not on X'
    side = np.abs(diag[:, 0]).mean()
    k = s_target / side

    def to_mm(g):
        return ascale(apply(g, ang), xfact=k, yfact=k, origin=(0, 0))
    return to_mm


def stroke_layers(reg, frame, to_mm, lakes: bool, detail: str):
    """Land-edge and L1-boundary stroke polygons, in mm, clipped to the hex."""
    hex_mm = to_mm(frame['hexagon'])
    rim = hex_mm.boundary.buffer(RIM_EPS_MM)

    land_files = [ne_layer('land', detail)]
    if detail == '10m':
        land_files.append(ne_layer('minor_islands', detail))
    if lakes:
        land_files.append(ne_layer('lakes', detail))
    pieces = [p for f in land_files for g in load_shape_polygons(f)
              for p in project_fill_to_tile(g, reg, frame, max_step_m=STEP_M)]
    land_lines = unary_union([to_mm(p).boundary for p in pieces])
    land_lines = land_lines.simplify(SIMPLIFY_MM).difference(rim)

    cells = tile_subgrid(reg, frame, layers=(frame['level'] + 1,))
    grid_lines = unary_union([to_mm(c).exterior for cs in cells.values() for c in cs])
    grid_lines = grid_lines.simplify(SIMPLIFY_MM).intersection(hex_mm).difference(rim)

    strokes = unary_union([land_lines.buffer(LAND_W / 2),
                           grid_lines.buffer(GRID_W / 2)])
    strokes = strokes.intersection(hex_mm).simplify(POLY_SIMPLIFY_MM)
    polys = [p for p in getattr(strokes, 'geoms', [strokes])
             if isinstance(p, Polygon) and p.area >= MIN_AREA_MM2]
    return MultiPolygon(polys)


def emit_scad(path: Path, frame, level: int, s_mm: float, mp: MultiPolygon):
    def ring(r):
        return '[' + ','.join(f'[{x:.2f},{y:.2f}]' for x, y in r.coords[:-1]) + ']'
    rows = []
    for p in mp.geoms:
        rows.append('[' + ','.join([ring(p.exterior)] + [ring(i) for i in p.interiors]) + ']')
    npts = sum(len(p.exterior.coords) + sum(len(i.coords) for i in p.interiors)
               for p in mp.geoms)
    with open(path, 'w') as f:
        f.write(f'// generated by gen_engrave.py — Hex9 L0 tile "{frame["address"]}", '
                f'scaled for level {level} (side {s_mm:.4f} mm)\n'
                f'// land edges + L1 cell boundaries, outside view; '
                f'{len(rows)} polygons, {npts} points\n'
                f'engrave_address = "{frame["address"]}";\n'
                f'engrave_side = {s_mm:.4f};\n'
                f'engrave_polys = [\n' + ',\n'.join(rows) + '\n];\n')
    return len(rows), npts


def write_wrapper(addr: str) -> Path:
    """Openable per-tile model: the tile data include comes AFTER the model
    include, so its assignments override the engrave_data.scad ones (OpenSCAD
    last-assignment-wins). Customizer callers should use h9_hinged_hex.scad
    instead — the customizer only scans the main file."""
    out = HERE / f'tile_{addr}.scad'
    out.write_text(
        f"// generated by gen_engrave.py — open or render THIS file for tile '{addr}'.\n"
        f'include <h9_hinged_hex.scad>\n'
        f'include <engrave_{addr}.scad>\n')
    return out


def build_tile(reg, addr, level: int, lakes: bool, detail: str) -> Path:
    """Generate engrave_<addr>.scad + tile_<addr>.scad for one L0 tile."""
    frame = tile_frame(reg, addr, 0)
    s_mm = side_mm(level)
    to_mm = hinge_transform(frame, fold_line_points(reg, frame), s_mm)
    mp = stroke_layers(reg, frame, to_mm, lakes, detail)
    out = HERE / f'engrave_{frame["address"]}.scad'
    n, npts = emit_scad(out, frame, level, s_mm, mp)
    wrapper = write_wrapper(frame['address'])
    print(f"tile '{frame['address']}' octants {frame['octants']}: "
          f'{out.name} + {wrapper.name}, {n} stroke polygons, {npts} points')
    return out


def find_openscad():
    exe = shutil.which('openscad') or '/Applications/OpenSCAD.app/Contents/MacOS/OpenSCAD'
    return exe if Path(exe).exists() else None


def render_stl(exe, addr: str, level: int, mode='flat') -> Path:
    """Render one tile's wrapper to output/ (independent of the selection)."""
    outdir = HERE / 'output'
    outdir.mkdir(exist_ok=True)
    out = outdir / f'hex9_{addr}_L{level}_{mode}.stl'
    r = subprocess.run([exe, '-D', f'mode="{mode}"', '-D', f'level={level}',
                        '-o', str(out), str(HERE / f'tile_{addr}.scad')],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f'OpenSCAD failed on tile {addr}:\n{r.stderr[-2000:]}')
    print(f'  rendered {out.relative_to(HERE)}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tile', default='4', help="L0 hex address '0'..'b', or 'all'")
    ap.add_argument('--use', metavar='ADDR',
                    help='switch engrave_data.scad to an existing engrave_<ADDR>.scad and exit')
    ap.add_argument('--level', type=int, default=16, help='true-scale level (SCAD default 16)')
    ap.add_argument('--lakes', action='store_true', help='engrave lake shorelines too')
    ap.add_argument('--detail', default='50m', choices=('10m', '50m', '110m'))
    ap.add_argument('--render', action='store_true',
                    help='also render each generated tile to output/*.stl via OpenSCAD')
    ap.add_argument('--mode', default='flat', choices=('flat', 'folded'),
                    help='SCAD mode for --render')
    ap.add_argument('--list', action='store_true', help='list tiles + extents and exit')
    args = ap.parse_args()

    if args.use is not None:
        src = HERE / f'engrave_{args.use}.scad'
        if not src.exists():
            raise SystemExit(f'{src.name} not found — generate it first: '
                             f'gen_engrave.py --tile {args.use}')
        shutil.copy(src, HERE / 'engrave_data.scad')
        print(f'engrave_data.scad <- {src.name}  ({src.read_text().splitlines()[0][3:]})')
        return

    reg = Registrar()
    if args.list:
        for i in range(12):
            fr = tile_frame(reg, i, 0)
            bb = fr['geo_bbox']
            ext = 'antimeridian' if bb is None else \
                f'lon {bb[0]:+7.1f}..{bb[2]:+7.1f}  lat {bb[1]:+6.1f}..{bb[3]:+6.1f}'
            print(f"  '{fr['address']}'  octants {fr['octants']}  {ext}")
        return

    exe = find_openscad() if args.render else None
    if args.render and exe is None:
        raise SystemExit('--render: OpenSCAD not found on PATH or in /Applications')

    addrs = [format(i, 'x') for i in range(12)] if args.tile == 'all' else [args.tile]
    for addr in addrs:
        stamped = build_tile(reg, addr, args.level, args.lakes, args.detail)
        if args.tile != 'all':
            shutil.copy(stamped, HERE / 'engrave_data.scad')
            print(f'engrave_data.scad <- {stamped.name}')
        if args.render:
            render_stl(exe, addr, args.level, args.mode)
    if args.tile == 'all':
        print("engrave_data.scad left unchanged — switch with: gen_engrave.py --use <addr>")


if __name__ == '__main__':
    main()
