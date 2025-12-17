# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

import numpy as np
from numpy.typing import NDArray
from hhg9 import Points
from hhg9.base.projection import Projection
from pyproj import CRS, Transformer


class EllipsoidGCD(Projection):
    """
    Ellipsoidal XYZ <=> LatLon projection using WGS84.
    Forward: ECEF → (lat, lon)
    Backward: (lat, lon) → ECEF
    """

    def __init__(self, registrar):
        super().__init__(registrar, 'ell_gcd', 'c_ell', 'g_gcd')
        # Define coordinate systems
        c_ell = registrar.domain('c_ell')
        crs_geodetic = CRS.from_epsg(4326)  # WGS84 lat/lon
        # pyproj Transformers
        self.to_geodetic = Transformer.from_crs(c_ell.crs, crs_geodetic, always_xy=True)
        self.to_ecef = Transformer.from_crs(crs_geodetic, c_ell.crs, always_xy=True)

    def forward(self, arr: Points) -> NDArray:
        """ECEF → Geodetic (lat, lon)"""
        x, y, z = arr.coords[..., 0], arr.coords[..., 1], arr.coords[..., 2]
        lon, lat, _ = self.to_geodetic.transform(x, y, z)
        latlon = np.stack([lat, lon], axis=-1)
        return Points(latlon, domain=self.fwd_cs, samples=arr.samples)

    def backward(self, arr: Points) -> NDArray:
        """Geodetic (lat, lon) → ECEF"""
        lat, lon = arr.coords[..., 0], arr.coords[..., 1]
        heights = np.zeros_like(lat)
        x, y, z = self.to_ecef.transform(lon, lat, heights)
        xyz = np.stack([x, y, z], axis=-1)
        return Points(xyz, domain=self.rev_cs, samples=arr.samples)


# Use EllipsoidGCDRad when you want g_gcd in radians (r_gcd) to avoid per-candidate deg↔rad churn in root-finding.
# The degree-based EllipsoidGCD remains for human-facing I/O and compatibility.
class EllipsoidGCDRad(Projection):
    """
    Ellipsoidal XYZ ↔ Geodetic (lat, lon in radians) using WGS84.
    Forward: ECEF → (lat_rad, lon_rad)
    Backward: (lat_rad, lon_rad) → ECEF
    """
    def __init__(self, registrar):
        super().__init__(registrar, 'ell_gcr', 'c_ell', 'r_gcd')
        # Ensure r_gcd domain and deg↔rad projections exist (lazy/idempotent)
        try:
            registrar.domain('r_gcd')
        except Exception:
            from hhg9.domains import RadiansGCD
            from hhg9.projections import RGCD_GCD
            RadiansGCD(registrar)
            RGCD_GCD(registrar)
        # Reuse the pyproj transformers from the degree-based class
        crs_geodetic = CRS.from_epsg(4326)
        crs_ecef = CRS.from_epsg(4978)
        self.to_geodetic = Transformer.from_crs(crs_ecef, crs_geodetic, always_xy=True)
        self.to_ecef = Transformer.from_crs(crs_geodetic, crs_ecef, always_xy=True)

    def forward(self, arr: Points) -> NDArray:
        """ECEF → Geodetic radians (lat, lon in rad)"""
        x, y, z = arr.coords[..., 0], arr.coords[..., 1], arr.coords[..., 2]
        lon_deg, lat_deg, _ = self.to_geodetic.transform(x, y, z)
        latlon_rad = np.radians(np.stack([lat_deg, lon_deg], axis=-1))
        return Points(latlon_rad, domain=self.fwd_cs, samples=arr.samples)

    def backward(self, arr: Points) -> NDArray:
        """Geodetic radians (lat, lon in rad) → ECEF"""
        lat_deg = np.degrees(arr.coords[..., 0])
        lon_deg = np.degrees(arr.coords[..., 1])
        heights = np.zeros_like(lat_deg)
        x, y, z = self.to_ecef.transform(lon_deg, lat_deg, heights)
        xyz = np.stack([x, y, z], axis=-1)
        return Points(xyz, domain=self.rev_cs, samples=arr.samples)


if __name__ == '__main__':
    from support import Display, Photo
    from hhg9 import Registrar
    from hhg9.domains import GeneralGCD, PlatePixel, EllipsoidCartesian
    from hhg9.projections import PlatePixelGCD

    reg = Registrar()
    p_pix = PlatePixel(reg)
    e_sph = EllipsoidCartesian(reg)
    g_gen = GeneralGCD(reg)      # (xyz)
    ppg = PlatePixelGCD(reg)
    prj = EllipsoidGCD(reg)   # EllipsoidalGCD<->Cartesian (xyz)

    ps = Photo()

    ps.load('../../preparatory/world1350x675.png')
    pc_px = p_pix.adopt(ps.img)
    gg_ll = reg.project(pc_px, [p_pix, g_gen])
    gg_el = reg.project(gg_ll, [g_gen, e_sph])

    d = Display()  # simple support display class
    d.show_pts_3d(gg_el)
