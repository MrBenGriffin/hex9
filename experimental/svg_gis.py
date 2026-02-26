# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import argparse
import json
from math import hypot
from typing import Iterable, List, Tuple

# from svg import Polyline, Polygon
from svgelements import SVG, Path, Polyline, Polygon
from shapely.geometry import LineString, mapping
from shapely.ops import linemerge


def affine_viewbox_to_bbox(lines, viewbox, bbox):
    vb_x, vb_y, vb_w, vb_h = viewbox
    lon_min, lat_min, lon_max, lat_max = bbox

    sx = (lon_max - lon_min) / vb_w
    sy = (lat_max - lat_min) / vb_h

    out = []
    for ls in lines:
        pts = []
        for x, y in ls.coords:
            lon = lon_min + (x - vb_x) * sx
            lat = lat_max - (y - vb_y) * sy  # invert SVG y-down
            pts.append((lon, lat))
        out.append(LineString(pts))
    return out


# def affine_to_bbox(lines, viewbox, bbox):
#     lon_min, lat_min, lon_max, lat_max = bbox
#     x0, y0, x1, y1 = affine_viewbox_to_bbox(viewbox)
#     sx = (lon_max - lon_min) / (x1 - x0)
#     sy = (lat_max - lat_min) / (y1 - y0)
#
#     out = []
#     for ls in lines:
#         pts = []
#         for x, y in ls.coords:
#             lon = lon_min + (x - x0) * sx
#             lat = lat_max - (y - y0) * sy  # invert y
#             pts.append((lon, lat))
#         out.append(LineString(pts))
#     return out

def path_length_approx(p: Path, samples: int = 50) -> float:
    # quick length approximation (svgelements has .length() but this is stable across versions)
    pts = [p.point(i / samples) for i in range(samples + 1)]
    return sum(hypot(pts[i+1].x - pts[i].x, pts[i+1].y - pts[i].y) for i in range(samples))


def sample_path(p: Path, max_step: float, min_samples: int = 50) -> List[Tuple[float, float]]:
    # choose samples so chord length ~ max_step (in SVG units)
    L = max(path_length_approx(p, samples=min_samples), 1e-9)
    n = max(min_samples, int(L / max_step) + 1)
    pts = [p.point(i / n) for i in range(n + 1)]
    # drop consecutive duplicates
    out = []
    last = None
    for q in pts:
        xy = (float(q.x), float(q.y))
        if last is None or (abs(xy[0] - last[0]) > 1e-9 or abs(xy[1] - last[1]) > 1e-9):
            out.append(xy)
            last = xy
    return out


def svg_to_lines(svg, max_step: float, flip_y: bool) -> List[LineString]:

    # Determine SVG height for y-flip (if absent, we leave as-is)
    svg_h = None
    if hasattr(svg, "height") and svg.height is not None:
        try:
            svg_h = float(svg.height)
        except Exception:
            svg_h = None
    if svg_h is None and getattr(svg, "viewbox", None):
        try:
            svg_h = float(svg.viewbox[3])
        except Exception:
            svg_h = None

    lines: List[LineString] = []

    for elem in svg.elements():
        p: Path = Path()
        if isinstance(elem, Path):
            p = elem
        elif isinstance(elem, Polygon):
            pts = list(elem.points)
            p.move(pts[0])
            for xy in pts[1:]:
                p.line(xy)
            p.line(pts[0])
        elif isinstance(elem, Polyline):
            pts = list(elem.points)
            p.move(pts[0])
            for xy in pts[1:]:
                p.line(xy)
        else:
            continue

        pts = sample_path(p, max_step=max_step)
        if len(pts) < 2:
            continue

        if flip_y and svg_h is not None:
            pts = [(x, svg_h - y) for (x, y) in pts]  # SVG y-down -> y-up

        ls = LineString(pts)
        if not ls.is_empty and ls.length > 0:
            lines.append(ls)

    # Merge connected segments where possible (common with Illustrator exports)
    if lines:
        merged = linemerge(lines)
        if merged.geom_type == "LineString":
            return [merged]
        if merged.geom_type == "MultiLineString":
            return list(merged.geoms)

    return lines


def write_geojson(lines: Iterable[LineString], out_path: str) -> None:
    feats = []
    for i, ls in enumerate(lines, start=1):
        feats.append({
            "type": "Feature",
            "properties": {"id": i},
            "geometry": mapping(ls),
        })
    fc = {"type": "FeatureCollection", "features": feats}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fc, f, indent=2)


def run(_args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--svg", type=str, default='coast.svg', help="input SVG filepath")
    ap.add_argument("--geojson", type=str, default='coast.json', help="output GeoJSON filepath")
    ap.add_argument("--max_step", type=float, default=1.0, help="max segment length in SVG units (smaller = more vertices)")
    ap.add_argument("--flip_y", action='store_true', help="flip Y axis using SVG height/viewBox so Y increases upward")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
                    help="Interpret SVG coords as Plate Carree over this bbox; outputs RFC7946 lon/lat GeoJSON.")
    args = ap.parse_args(args=_args)

    svg = SVG.parse(args.svg)
    lines = svg_to_lines(svg, max_step=args.max_step, flip_y=args.flip_y)
    if not lines:
        raise SystemExit("No paths found in SVG (or all were empty).")

    if args.bbox:
        vb = getattr(svg, "viewbox", None)
        if not vb:
            raise SystemExit("SVG has no viewBox; add one or provide explicit bounds.")
        vb_x = float(getattr(vb, "x"))
        vb_y = float(getattr(vb, "y"))
        vb_w = float(getattr(vb, "width"))
        vb_h = float(getattr(vb, "height"))
        lines = affine_viewbox_to_bbox(lines, (vb_x, vb_y, vb_w, vb_h), args.bbox)

    write_geojson(lines, args.geojson)
    print(f"Wrote {len(lines)} line(s) to {args.geojson}")


if __name__ == "__main__":
    _args = {
        "svg": "mountains.svg",
        "geojson": "mountains.json",
        "bbox": [15.0, 19.56, 30.05, 33.60]
    }
    vals = []
    for k, v in _args.items():
        vals.append(f"--{k}")
        if isinstance(v, list):
            for vi in v:
                vals.append(f'{vi}')
        else:
            vals.append(f'{v}')
    run(vals)
