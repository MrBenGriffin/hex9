# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Showing 2 layers of hexagons:  land-usage, hill-shading
Last Tested
26 December 2026 0.1.0a4 (passed)
"""

import numpy as np
from hhg9 import Points, Registrar
from hhg9.algorithms.distance import wgs84_area
from hhg9.h9.addressing import hex_layer, TailStyle
from geographiclib.geodesic import Geodesic

geod = Geodesic.WGS84


def densify_poly_geodesic_by_step(poly_latlon: np.ndarray,
                                  max_step_m: float = 2_000.0,
                                  max_per_edge: int = 512,
                                  closed: bool = True) -> np.ndarray:
    """
    poly_latlon: (m,2) in (lat, lon), ordered along the boundary.
    Returns boundary samples (lat, lon).

    Notes:
      - Samples include each edge start, exclude each edge end (avoids duplicate vertices).
      - If closed and last vertex repeats first, the duplicate is ignored.
    """
    poly = np.asarray(poly_latlon, dtype=np.float64)
    if poly.ndim != 2 or poly.shape[1] != 2:
        raise ValueError(f"poly_latlon must have shape (m,2); got {poly.shape}")
    if poly.shape[0] < 2:
        return poly.copy()

    if closed and np.allclose(poly[0], poly[-1]):
        poly = poly[:-1]

    m = poly.shape[0]
    last = m if closed else (m - 1)

    out = []
    for i in range(last):
        lat1, lon1 = poly[i]
        lat2, lon2 = poly[(i + 1) % m]

        inv = geod.Inverse(float(lat1), float(lon1), float(lat2), float(lon2))
        s12 = float(inv["s12"])
        line = geod.Line(float(lat1), float(lon1), float(inv["azi1"]))

        nseg = int(np.ceil(s12 / max_step_m)) if max_step_m > 0 else 1
        nseg = max(1, min(nseg, max_per_edge))

        s = np.linspace(0.0, s12, nseg, endpoint=False)
        for si in s:
            r = line.Position(float(si))
            out.append((r["lat2"], r["lon2"]))

    return np.array(out, dtype=np.float64)


def calculate_intersections(poly_grid, v):
    """
    Finds u-coordinates where the horizontal line at v crosses polygon edges.
    poly_grid: (m, 2) array of polygon vertices in scaled UV space.
    v: The current scanline (integer row index).
    """
    # 1. Define segments (p1 to p2)
    p1 = poly_grid
    p2 = np.roll(poly_grid, -1, axis=0)

    # 2. Find edges that cross the scanline v
    # We use (p1y <= v < p2y) or (p2y <= v < p1y) to handle vertices correctly
    # and avoid double-counting horizontal edges.
    mask = ((p1[:, 1] <= v) & (p2[:, 1] > v)) | ((p2[:, 1] <= v) & (p1[:, 1] > v))

    if not np.any(mask):
        return np.array([])

    # 3. Filter segments
    e1 = p1[mask]
    e2 = p2[mask]

    # 4. Linear Interpolation to find the u-coordinate
    # u = x1 + (v - y1) * (x2 - x1) / (y2 - y1)
    u_ints = e1[:, 0] + (v - e1[:, 1]) * (e2[:, 0] - e1[:, 0]) / (e2[:, 1] - e1[:, 1])

    # 5. Return sorted intersections for pairing
    return np.sort(u_ints)


def scanline_h9_sheet(poly_n, level):
    # sn is grid spacing factor
    sn = np.sqrt(2) * (3 ** -(level + 1))

    # 1. Scale polygon to integer grid space (UV Skew Space)
    x, y = poly_n.coords[:, 0], poly_n.coords[:, 1]
    # Standard Axial/Skew transform for hexagonal grids
    v_skew = (y * 2 / np.sqrt(3)) / sn
    u_skew = (x / sn) - (v_skew / 2)

    poly_grid = np.stack([u_skew, v_skew], axis=1)

    # 2. Get Row Range
    v_min = int(np.floor(poly_grid[:, 1].min()))
    v_max = int(np.ceil(poly_grid[:, 1].max()))

    u_acc, v_acc = [], []

    for v in range(v_min, v_max + 1):
        u_ints = calculate_intersections(poly_grid, v)

        # Standard even-odd fill pairing
        for i in range(0, len(u_ints), 2):
            if i + 1 >= len(u_ints): break
            u_start = int(np.ceil(u_ints[i]))
            u_end = int(np.floor(u_ints[i + 1]))

            if u_end >= u_start:
                u_range = np.arange(u_start, u_end + 1)
                u_acc.append(u_range)
                v_acc.append(np.full_like(u_range, v))

    if not u_acc:
        # Fallback to centroid if no points found
        avg = np.mean(poly_grid, axis=0)
        u_final, v_final = np.array([np.round(avg[0])]), np.array([np.round(avg[1])])
    else:
        u_final = np.concatenate(u_acc)
        v_final = np.concatenate(v_acc)

    y_n = (v_final - 0.333) * sn * (np.sqrt(3) / 2)
    x_n = (u_final + (v_final / 2)) * sn

    uv = np.stack([x_n, y_n], axis=1)
    result = Points(uv, domain=poly_n.domain, components=poly_n.components[0])
    # print(f'Layer: {level}; Points: {len(result)}')
    return result


if __name__ == '__main__':
    # WGS84 ellipsoid surface area (m^2). Using helper keeps this aligned with the rest of the stack.
    # earth = float(wgs84_area())
    earth = 510_065_621_724_154.6
    hex_0 = earth / 12
    l_hex = hex_0
    h_areas = np.zeros((64,), dtype=np.float64)
    for i in range(64):
        h_areas[i] = l_hex
        l_hex /= 9

    tests = {
        'LAX, CA': [
            (-118.418990325877004, 33.9333814840111),
            (-118.419070754935007, 33.933926959066604),
            (-118.382766218171994, 33.9376406125145),
            (-118.382686017115006, 33.937095114458899),
        ],
        'Utqiagvik, AK':
        [
            [-156.738405694211991, 71.284672766744194],
            [-156.798810301001993, 71.284662977615298],
            [-156.798811581733986, 71.285072736580005],
            [-156.738405700945009, 71.285082525709299],
            [-156.738405694211991, 71.284672766744194]
         ],
        'North Pole':
        [
            [1, 89.9],
            [1.15, 89.9],
            [1.15, 89.85],
            [1, 89.85],
        ]
    }

    rg = Registrar()
    b_oct = rg.domain('b_oct')
    # b_oct.set_warp('src/eb2_model_L5.npz')  # GPT
    # b_oct.set_warp('src/l4_polished.npz')   # Ghosts

    g_gcd = rg.domain('g_gcd')

    for name, values in tests.items():
        coords = np.array(values, dtype=np.float64)[:, ::-1]
        gcd_pts = Points(coords, g_gcd)  # expects (lat,lon) in g_gcd
        gcd_area = wgs84_area(rg, gcd_pts, coords.shape[0])
        print(f'\n{name}: WGS84 area: {float(gcd_area[0])}m²')
        h9_pts = rg.project(gcd_pts, [g_gcd, b_oct])  # or your chosen path
        for layer in range(9, 16):
            b_pts = scanline_h9_sheet(h9_pts, layer)
            h_key = hex_layer(b_pts, layer=layer, tail_style=TailStyle.key)
            hex_k = np.unique(h_key, axis=0)
            hexes = hex_k.shape[0]
            one_hex = h_areas[layer]
            area = hexes * one_hex
            print(f'Layer {layer} {one_hex}m² / hex; {hexes} estimated area {area}m²')
