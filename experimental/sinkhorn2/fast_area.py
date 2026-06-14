# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""
Vectorised ellipsoidal triangle areas for the warp-training loop.

hhg9.algorithms.distance.wgs84_area loops over polygons calling geographiclib's
PolygonArea — single-threaded, with ~tens of thousands of calls per feedback /
gradient iteration. On macOS that single-core path dominates the run time.

This module replaces it with a fully vectorised authalic-sphere computation:

    geodetic latitude  ->  authalic latitude        (exact, via the q-function)
    (authalic lat, lon) ->  unit vectors on a sphere of radius R_q
    triangle solid angle via the signed triple-product formula
    area = R_q^2 * |Omega|

For the small (~1-2 deg) L4/L5 triangles this matches geodesic polygon area to
well under 1 ppm — far below the loop's MAE ~1e-3 working precision — and any
residual bias is position-smooth, so it cancels in ratios = areas/mean(areas).
Typically 100-1000x faster than the geographiclib loop.

`fast_wgs84_area` is a drop-in for `wgs84_area` (triangles, shape=3).
"""
import numpy as np


def _authalic_lat(lat_rad: np.ndarray, e2: float):
    """Geodetic -> authalic latitude (exact). Returns (xi, q_p)."""
    e = np.sqrt(e2)
    s = np.sin(lat_rad)
    q = (1.0 - e2) * (s / (1.0 - e2 * s * s)
                      - (1.0 / (2.0 * e)) * np.log((1.0 - e * s) / (1.0 + e * s)))
    q_p = (1.0 - e2) * (1.0 / (1.0 - e2)
                        - (1.0 / (2.0 * e)) * np.log((1.0 - e) / (1.0 + e)))
    return np.arcsin(np.clip(q / q_p, -1.0, 1.0)), q_p


def triangle_areas(lat_lon_deg, a: float, f: float) -> np.ndarray:
    """Areas (m^2) of triangles given as an (n, 3, 2) lat/lon array in degrees."""
    e2 = f * (2.0 - f)
    ll = np.deg2rad(np.asarray(lat_lon_deg, dtype=np.float64))
    lat, lon = ll[..., 0], ll[..., 1]
    if e2 > 0.0:
        xi, q_p = _authalic_lat(lat, e2)
        r_q = a * np.sqrt(q_p / 2.0)
    else:
        xi, r_q = lat, a
    cx = np.cos(xi)
    v = np.stack([cx * np.cos(lon), cx * np.sin(lon), np.sin(xi)], axis=-1)
    A, B, C = v[:, 0], v[:, 1], v[:, 2]
    triple = np.einsum('ij,ij->i', A, np.cross(B, C))
    denom = (1.0 + np.einsum('ij,ij->i', A, B)
             + np.einsum('ij,ij->i', B, C)
             + np.einsum('ij,ij->i', C, A))
    omega = 2.0 * np.arctan2(triple, denom)
    return np.abs(omega) * r_q * r_q


def fast_wgs84_area(reg, pts, shape: int = 3) -> np.ndarray:
    """Drop-in replacement for hhg9.algorithms.distance.wgs84_area (triangles)."""
    from hhg9 import Points
    geo = reg.ellipsoid
    if isinstance(pts, Points):
        dom = pts.domain
        wp = pts if dom.name == 'g_gcd' else reg.project(pts, [dom, 'g_gcd'])
        lat_lon = wp.coords.reshape((-1, shape, 2))
    else:
        lat_lon = np.asarray(pts).reshape((-1, shape, 2))
    return triangle_areas(lat_lon, geo.a, geo.f)
