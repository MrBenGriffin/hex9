"""
Part of the H9 project
"""
import numpy as np
from geographiclib.geodesic import Geodesic
from functools import lru_cache


def haversine(p1, p2):
    """ Use haversine - which will be fine for root-finding. """
    lat1, lon1 = p1[..., 0], p1[..., 1]
    lat2, lon2 = p2[..., 0], p2[..., 1]
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    d_lon = lon2_rad - lon1_rad
    d_lat = lat2_rad - lat1_rad
    # Haversine formula
    a = np.sin(d_lat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(d_lon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return 6371.0 * c


@lru_cache(maxsize=None)
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
        geo (Geodesic, optional): A pre-instantiated Geodesic object.
                                  Defaults to WGS84.
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
