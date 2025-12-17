# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Distance calculations for Hex9 (H9) project.
"""

import numpy as np
from geographiclib.geodesic import Geodesic
from geographiclib.polygonarea import PolygonArea
from functools import cache

R_MEAN = 6371008.8  # mean Earth radius in meters (IUGG)


def haversine(p1, p2, R: float = R_MEAN):
    """Great-circle distance on a sphere using the haversine formula.
    Accepts degrees; converts once, then defers to the radians variant.
    p1, p2: array-like (..., 2) as [lat_deg, lon_deg]
    Returns distance in meters (float or ndarray).
    """
    p1r = np.radians(p1)
    p2r = np.radians(p2)
    return haversine_rad(p1r, p2r, R)

def haversine_rad(p1_rad, p2_rad, R: float = R_MEAN):
    """Haversine distance with inputs in radians. Vectorised over leading dims.
    p1_rad, p2_rad: (..., 2) arrays [lat_rad, lon_rad]
    Returns meters.
    """
    lat1 = p1_rad[..., 0]
    lon1 = p1_rad[..., 1]
    lat2 = p2_rad[..., 0]
    lon2 = p2_rad[..., 1]
    d_lat = lat2 - lat1
    d_lon = lon2 - lon1
    sin_dlat2 = np.sin(d_lat * 0.5)
    sin_dlon2 = np.sin(d_lon * 0.5)
    a = sin_dlat2 * sin_dlat2 + np.cos(lat1) * np.cos(lat2) * sin_dlon2 * sin_dlon2
    # c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arcsin(np.sqrt(a))
    return R * c


def haversine_from_ref_rad(phi0: float, lam0: float, phis, lams, R: float = R_MEAN, small_eps: float = 1e-8):
    """Fast distances from a single reference (phi0,lam0) [radians] to many targets.
    Uses an exact haversine for general cases, and a small-angle approximation
    when both |Δφ| and |Δλ| are tiny to save trig calls.
    phis, lams: arrays of target lat/lon in radians.
    Returns meters (np.ndarray).
    """
    sphi0 = np.sin(phi0)
    cphi0 = np.cos(phi0)
    dphi = phis - phi0
    dlam = lams - lam0
    small = (np.abs(dphi) < small_eps) & (np.abs(dlam) < small_eps)
    # exact
    sin_dphi2 = np.sin(dphi * 0.5)
    sin_dlam2 = np.sin(dlam * 0.5)
    a_exact = sin_dphi2 * sin_dphi2 + cphi0 * np.cos(phis) * sin_dlam2 * sin_dlam2
    c_exact = 2.0 * np.arctan2(np.sqrt(a_exact), np.sqrt(1.0 - a_exact))
    # small-angle: Δσ^2 ≈ (Δφ)^2 + (cosφ0)^2 (Δλ)^2
    c_approx = np.sqrt(dphi * dphi + (cphi0 * cphi0) * (dlam * dlam))
    c = np.where(small, c_approx, c_exact)
    return R * c


def great_circle_atan2_rad(p1_rad, p2_rad, R: float = R_MEAN):
    """Robust great-circle distance via atan2(||util×v||, util·v) with radian inputs.
    p1_rad, p2_rad: (..., 2) arrays [lat_rad, lon_rad]
    Returns meters.
    """
    lat1 = p1_rad[..., 0]
    lon1 = p1_rad[..., 1]
    lat2 = p2_rad[..., 0]
    lon2 = p2_rad[..., 1]
    c1 = np.cos(lat1)
    c2 = np.cos(lat2)
    u1 = np.stack([c1 * np.cos(lon1), c1 * np.sin(lon1), np.sin(lat1)], axis=-1)
    u2 = np.stack([c2 * np.cos(lon2), c2 * np.sin(lon2), np.sin(lat2)], axis=-1)
    dot = np.sum(u1 * u2, axis=-1)
    cross = np.linalg.norm(np.cross(u1, u2), axis=-1)
    ang = np.arctan2(cross, np.clip(dot, -1.0, 1.0))
    return R * ang


@cache
def geodesic_wgs84():
    """
    Create and cache a single instance of the Geodesic object.
    """
    return Geodesic.WGS84


def _wgs84_distance(lat1, lon1, lat2, lon2):
    """
    Create and cache a single instance of the vectorised inverse
    limited to just DISTANCE.
    """
    geo = geodesic_wgs84()
    return geo.Inverse(lat1, lon1, lat2, lon2, Geodesic.DISTANCE)['s12']


def wgs84(p1, p2):
    """
    Calculates the geodesic distance between a batch of reference points
    and their corresponding sets of candidate points in a vectorized way.
    """
    lat1, lon1 = p1[..., 0], p1[..., 1]
    lat2, lon2 = p2[..., 0], p2[..., 1]

    result = np.vectorize(_wgs84_distance)(lat1, lon1, lat2, lon2)
    return result


def wgs84_ratio(bounds):
    """
    Calculates the precise geographic aspect ratio for a bounding box
    using geodesic calculations.

    Args:
        bounds (list or tuple): Bounding box in the format
                                [lon_min, lon_max, lat_min, lat_max].
    """
    lon_min, lon_max, lat_min, lat_max = bounds
    avg_lon = (lon_min + lon_max) / 2
    avg_lat = (lat_min + lat_max) / 2
    p1_lat, p2_lat = np.array([[lat_min, avg_lon]]), np.array([[lat_max, avg_lon]])
    p1_lon, p2_lon = np.array([[avg_lat, lon_min]]), np.array([[avg_lat, lon_max]])
    height_m = wgs84(p1_lat, p2_lat)[0]
    width_m = wgs84(p1_lon, p2_lon)[0]
    if width_m == 0:
        return 1.0
    return height_m / width_m


def wgs84_angular_ratio(bounds):
    """
    Calculates the simple angular aspect ratio (Δlat / Δlon).
    """
    lon_min, lon_max, lat_min, lat_max = bounds
    delta_lat = lat_max - lat_min
    delta_lon = lon_max - lon_min
    if delta_lon == 0:
        return 1.0  # Avoid division by zero for a vertical line
    return delta_lat / delta_lon


def wgs84_area(reg, pts, shape=6):
    """
    Given a set of polygon points, calculate the areas of the polygons
    Polygons are defined in hhg following CW rules.
    """
    from hhg9 import Points
    geo = geodesic_wgs84()

    wp = None
    if isinstance(pts, Points):
        dom = pts.domain
        if dom.name != 'g_gcd':
            try:
                wp = reg.project(pts, [dom, 'g_gcd'])
            except ValueError as e:
                msg = e.args[0]
                raise ValueError(f'wgs84_polygon_area: {msg} while attempting [{dom.name} => g_gcd]')
        else:
            wp = pts.copy()
    lat_lon_pts = wp.coords.reshape((-1, shape, 2))
    areas = np.empty(len(lat_lon_pts))
    for i, poly in enumerate(lat_lon_pts):
        lat, lon = poly[:, 0], poly[:, 1]
        pa = PolygonArea(earth=geo)  # new instance per polygon
        for lat, lon in zip(lat, lon):
            pa.AddPoint(lat=lat, lon=lon)
        edges, perimeter, area = pa.Compute(reverse=True)  # number, perimeter, area
        areas[i] = area
    return areas


def enu_planar_polygon_area(reg, pts, vertices: int = 6):
    """
    Fast local-planar (ENU) area estimate on WGS84.

    Inputs:
        reg: registrar; must support reg.project(..., ['c_ell','g_gcd']) and ['*','c_ell']
        pts: hhg9.Points in any domain or ndarray (..., vertices, 3) of ECEF xyz
        vertices: number of vertices per polygon (default 6)

    Returns:
        areas: ndarray shape (n_polys,) with areas in square meters.
    """
    from hhg9 import Points

    # --- ensure ECEF xyz (c_ell) and shape (n, V, 3) ---
    if isinstance(pts, Points):
        dom = pts.domain
        if dom.name != 'c_ell':
            try:
                xyz = reg.project(pts, [dom, 'c_ell']).coords
            except ValueError as e:
                msg = e.args[0]
                raise ValueError(f'enu_planar_polygon_area: {msg} while attempting [{dom.name} => c_ell]')
        else:
            xyz = pts.coords
    else:
        xyz = np.asarray(pts)

    xyz = xyz.reshape((-1, vertices, 3))
    n = xyz.shape[0]
    if n == 0:
        return np.empty((0,), dtype=float)

    # --- geodetic centroid (for stable, “central” ENU frames) ---
    ctr_xyz = xyz.mean(axis=1)  # (n,3)
    ctr_pts = Points(ctr_xyz, domain='c_ell')
    try:
        ctr_ll = reg.project(ctr_pts, ['c_ell', 'g_gcd']).coords  # degrees; (n,2)
    except ValueError as e:
        msg = e.args[0]
        raise ValueError(f'enu_planar_polygon_area: {msg} while attempting [c_ell => g_gcd]')

    lat = np.radians(ctr_ll[:, 0])  # (n,)
    lon = np.radians(ctr_ll[:, 1])

    # --- ENU basis at each centroid ---
    # east = [-sinλ,  cosλ, 0]
    # north= [-sinφ cosλ, -sinφ sinλ, cosφ]
    # up   = [ cosφ cosλ,  cosφ sinλ, sinφ]
    slat, clat = np.sin(lat), np.cos(lat)
    slon, clon = np.sin(lon), np.cos(lon)

    e_vec = np.stack([-slon,  clon, np.zeros_like(lon)], axis=1)                 # (n,3)
    n_vec = np.stack([-slat * clon, -slat * slon,  clat], axis=1)                # (n,3)
    u_vec = np.stack([ clat * clon,  clat * slon,  slat], axis=1)                # (n,3)

    # rotation matrix from ECEF to ENU: rows are basis vectors
    r_ecef2enu = np.stack([e_vec, n_vec, u_vec], axis=1)                         # (n,3,3)

    # --- map vertices to ENU around centroid ---
    delta = xyz - ctr_xyz[:, None, :]                                            # (n,V,3)
    enu = np.einsum('nij,nvj->nvi', r_ecef2enu, delta)                           # (n,V,3)
    x = enu[..., 0]
    y = enu[..., 1]

    # --- shoelace per polygon ---
    x_next = np.roll(x, -1, axis=1)
    y_next = np.roll(y, -1, axis=1)
    areas = 0.5 * np.abs(np.sum(x * y_next - y * x_next, axis=1))                # (n,)

    # handle degenerates (all points same or collinear)
    deg = ~np.isfinite(areas)
    if deg.any():
        areas[deg] = 0.0

    return areas