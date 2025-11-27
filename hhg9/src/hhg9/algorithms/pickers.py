# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""Pickers for sampling points on the sphere and ellipsoids."""

import numpy as np
from pyproj import Geod


def ell_rnd_quick(reg, n_points):
    """
    Generates random points on the ellipsoid surface eg (ECEF).
    NOTE: This uses axis scaling of uniformly sampled spherical directions.
    It is *not* area-uniform on the ellipsoid surface; it’s suitable for
    quick tests but not for strict uniform sampling. For geodesically
    uniform sampling on the sphere (GCD), use `gcd_rnd`. If you need
    area-uniform sampling on the ellipsoid surface, implement a dedicated
    sampler (e.g., rejection via surface normal density).

    Args:
        reg (Registrar): The project registrar (used to obtain the ellipsoid params).
        n_points (int): The number of points to generate.

    Returns:
        np.ndarray: An (n_points, 3) array of ECEF equivalent (x, y, z) on the ellipsoid.
    """
    ell = reg.domain('c_ell')
    # ellipsoid parameters in meters as set by c_ell.
    a = ell.a
    b = ell.b
    xyz = sph_rnd(n_points)
    x = a * xyz[:, 0]
    y = a * xyz[:, 1]
    z = b * xyz[:, 2]
    return np.stack([x, y, z], axis=1)


def ell_rnd_uniform(reg, n_points, seed=None, return_latlon=False, degrees=True):
    """
    Generate points area-uniformly on an oblate ellipsoid surface using acceptance–rejection sampling
    on the parametric latitude φ.

    The ellipsoid is parameterized by semi-axes 'a' (x,y) and 'b' (z), with
    x = a * cosφ * cosλ,
    y = a * cosφ * sinλ,
    z = b * sinφ,
    where φ ∈ [-π/2, π/2] is the parametric latitude and λ ∈ [-π, π) is longitude.

    The surface element area is
        dA = a² cosφ sqrt(1 - e² sin²φ) dφ dλ,
    where e² = 1 - (b²/a²) is the eccentricity squared.

    Sampling λ uniformly in [-π, π) and sinφ = u uniformly in [-1,1] yields a candidate density proportional to cosφ,
    which is not uniform on the ellipsoid surface. To correct this, we use acceptance–rejection with acceptance
    weight w = sqrt(1 - e² u²).

    The acceptance rate is ≥ sqrt(1 - e²), which is about 99.6% for WGS84 ellipsoid, making this method very efficient.

    Note: The returned latitudes (if requested) are parametric latitudes, NOT geodetic latitudes.

    Args:
        reg (Registrar): The project registrar (used to obtain the ellipsoid params).
        n_points (int): Number of points to generate.
        seed (int|None): Optional RNG seed for reproducibility.
        return_latlon (bool): If True, also return parametric latitude and longitude arrays.
        degrees (bool): If True, lat/lon are returned in degrees; else in radians.

    Returns:
        np.ndarray: If return_latlon is False, returns (n_points, 3) ECEF coordinates.
                    If True, returns a tuple ((n_points,3) ECEF array, (n_points,2) parametric lat/lon array).
    """
    ell = reg.domain('c_ell')
    a = ell.a
    b = ell.b
    e2 = 1.0 - (b * b) / (a * a)

    # Handle near-sphere case by falling back to spherical sampler
    if e2 <= 0 or e2 < 1e-16:
        # Use spherical sampling scaled to ellipsoid
        xyz = sph_rnd(n_points)
        x = a * xyz[:, 0]
        y = a * xyz[:, 1]
        z = b * xyz[:, 2]
        ecef = np.stack([x, y, z], axis=1)
        if return_latlon:
            s = xyz[:, 2]  # sinφ parametric latitude
            φ = np.arcsin(s)
            λ = np.arctan2(xyz[:, 1], xyz[:, 0])
            if degrees:
                latlon = np.column_stack((np.degrees(φ), np.degrees(λ)))
            else:
                latlon = np.column_stack((φ, λ))
            return ecef, latlon
        return ecef

    if seed is not None:
        rng = np.random.default_rng(seed)
        use_numpy_rng = False
    else:
        rng = None
        use_numpy_rng = True  # fallback to np.random

    accepted_s = []
    accepted_lambda = []

    batch_size = max(1000, n_points * 2)  # generate in batches

    while len(accepted_s) < n_points:
        if rng is not None:
            u = rng.uniform(-1.0, 1.0, batch_size)
            lambdas = rng.uniform(-np.pi, np.pi, batch_size)
            r = rng.uniform(0.0, 1.0, batch_size)
        else:
            u = np.random.uniform(-1.0, 1.0, batch_size)
            lambdas = np.random.uniform(-np.pi, np.pi, batch_size)
            r = np.random.uniform(0.0, 1.0, batch_size)

        w = np.sqrt(1.0 - e2 * u * u)
        accept_mask = r < w
        accepted_s.append(u[accept_mask])
        accepted_lambda.append(lambdas[accept_mask])

    s_all = np.concatenate(accepted_s)[:n_points]
    lambda_all = np.concatenate(accepted_lambda)[:n_points]

    cos_phi = np.sqrt(1.0 - s_all * s_all)
    x = a * cos_phi * np.cos(lambda_all)
    y = a * cos_phi * np.sin(lambda_all)
    z = b * s_all

    ecef = np.column_stack((x, y, z))

    if return_latlon:
        φ = np.arcsin(s_all)  # parametric latitude
        λ = lambda_all
        if degrees:
            latlon = np.column_stack((np.degrees(φ), np.degrees(λ)))
        else:
            latlon = np.column_stack((φ, λ))
        return ecef, latlon

    return ecef


def sph_rnd(n):
    """
    Generate n random spherical points
    returned in Euclidean (x,y,z)
    https://mathworld.wolfram.com/SpherePointPicking.html
    """
    θ = np.random.uniform(0, 2. * np.pi, n)
    u = np.random.uniform(-1., 1., n)
    x = ((1. - u ** 2.) ** 0.5) * np.cos(θ)
    y = ((1. - u ** 2.) ** 0.5) * np.sin(θ)
    z = u
    return np.stack([x, y, z], axis=1)


def gcd_fibonacci(n):
    """
    Generates n uniformly distributed (lat, lon) points on a sphere
    using the Fibonacci spiral method.

    Parameters:
    - n (int): The number of points to generate.

    Returns:
    - numpy.ndarray: An (n, 2) array where each row is [latitude, longitude]
                     in WGS84 degrees (-90 to 90, -180 to 180).
    """

    # Create an array of n indices
    i = np.arange(n, dtype=float) + 0.5

    # Calculate z-coordinate (cos(colatitude))
    # This creates evenly spaced z-coordinates from just under 1 to just over -1
    z = 1.0 - (2.0 * i / n)

    # Calculate latitude in degrees from z
    # lat = arcsin(z)
    latitude = np.arcsin(z) * 180.0 / np.pi

    # Calculate longitude based on the golden angle
    # Golden angle = pi * (3 - sqrt(5))
    golden_angle_rad = np.pi * (3. - np.sqrt(5.))
    longitude_rad = golden_angle_rad * i

    # Convert longitude to degrees and wrap to [-180, 180]
    longitude = np.degrees(longitude_rad)
    longitude = np.mod(longitude + 180.0, 360.0) - 180.0

    # Stack into an (n, 2) array
    return np.stack([latitude, longitude], axis=-1)


def gcd_rnd(n, seed=None, degrees=True):
    """
    Sample `n` points uniformly on the unit sphere in geodesic (lat, lon).
    This is area-uniform on the sphere: lon ~ U[-180, 180), sin(lat) ~ U[-1, 1].

    Args:
        n (int): number of points
        seed (int|None): optional RNG seed for reproducibility
        degrees (bool): return degrees (default) or radians

    Returns:
        np.ndarray: shape (n, 2): [lat, lon]
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
        u = rng.uniform(-1.0, 1.0, n)
        lon = rng.uniform(-np.pi, np.pi, n)
    else:
        u = np.random.uniform(-1.0, 1.0, n)
        lon = np.random.uniform(-np.pi, np.pi, n)
    lat = np.arcsin(u)
    if degrees:
        return np.column_stack((np.degrees(lat), np.degrees(lon)))
    return np.column_stack((lat, lon))


def gcd_cap_rnd(center_latlon, n, radius_km, seed=None, degrees=True, Rm=6371008.8):
    """
    Sample `n` points uniformly within a spherical cap of angular radius
    `radius_km / Rm` around `center_latlon` (geodesic uniform, on the sphere).

    Args:
        center_latlon (array-like): [lat, lon] of cap center (degrees by default)
        n (int): number of points
        radius_km (float): cap radius in kilometres
        seed (int|None): RNG seed
        degrees (bool): inputs/outputs in degrees (default). If False, use radians.
        Rm (float): mean Earth radius in metres for angular conversion

    Returns:
        np.ndarray: shape (n, 2) [lat, lon] sampled uniformly within the cap
    """
    if degrees:
        lat0 = np.radians(center_latlon[0])
        lon0 = np.radians(center_latlon[1])
    else:
        lat0 = float(center_latlon[0])
        lon0 = float(center_latlon[1])

    alpha_max = (radius_km * 1000.0) / Rm  # angular radius (rad)

    # Uniform in cap area ⇒ cos(alpha) ~ U[cos(alpha_max), 1]
    if seed is not None:
        rng = np.random.default_rng(seed)
        u = rng.random(n)
        bearing = rng.uniform(0.0, 2.0 * np.pi, n)
    else:
        u = np.random.random(n)
        bearing = np.random.uniform(0.0, 2.0 * np.pi, n)

    cos_a = 1.0 - u * (1.0 - np.cos(alpha_max))
    sin_a = np.sqrt(np.maximum(0.0, 1.0 - cos_a * cos_a))

    sin_lat0 = np.sin(lat0)
    cos_lat0 = np.cos(lat0)

    lat = np.arcsin(sin_lat0 * cos_a + cos_lat0 * sin_a * np.cos(bearing))
    lon = lon0 + np.arctan2(np.sin(bearing) * sin_a * cos_lat0,
                            cos_a - sin_lat0 * np.sin(lat))

    # normalize lon to [-pi, pi)
    lon = (lon + np.pi) % (2.0 * np.pi) - np.pi
    if degrees:
        return np.column_stack((np.degrees(lat), np.degrees(lon)))
    return np.column_stack((lat, lon))


def geodesic_cap_rnd_ecef(reg, center_lat, center_lon, n, radius_m, seed=None):
    """
    Samples points uniformly on a GEODESIC cap.
    Return n GCD Points
    """
    from hhg9 import Points

    # Define the WGS84 ellipsoid for pyproj
    g_gcd = reg.domain('g_gcd')
    geod = Geod(ellps='WGS84')

    rng = np.random.default_rng(seed)  # seed=None is legal
    azimuths = rng.uniform(0.0, 360.0, n)
    distances = radius_m * np.sqrt(rng.random(n))

    end_lon, end_lat, _ = geod.fwd(
        lons=np.full(n, center_lon),
        lats=np.full(n, center_lat),
        az=azimuths,
        dist=distances
    )
    # g_gcd uses lat/lon order.
    return Points(np.stack([end_lat, end_lon], axis=1), g_gcd)

