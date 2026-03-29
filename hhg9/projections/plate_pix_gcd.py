# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Part of the H9 project
"""
import numpy as np
from numpy.typing import NDArray
from hhg9 import Points
from hhg9.base.projection import Projection


class PlatePixelGCD(Projection):
    """PlatePixel ↔ GCD.

    After the `PlatePixel` refactor, points in the `p_pix` domain typically carry
    *world* coordinates (x,y) derived from `extent` and/or `origin/pixel_size`.

    For Plate Carrée-style rasters, that world frame is usually (lon, lat).

    This projection therefore supports two modes:
    - **World net_mode**: `p_pix` coords are already (lon,lat) (recommended).
    - **Legacy index net_mode**: `p_pix` coords are (px,py) pixel indices and we use
      lookup tables created by `set_dim`.
    """

    def __init__(self, registrar):
        super().__init__(registrar, 'pix_gcd', 'p_pix', 'g_gcd')
        self.p_hgt = None
        self.p_wid = None
        self.lon = None
        self.lat = None

    def set_dim(self, pts: Points = None, bounds: object = None) -> object:
        """Set dimensions / lookup tables for legacy index net_mode."""
        if pts is None:
            raise ValueError('set_dim requires pts for legacy index net_mode.')

        if self.p_hgt is None or pts is not None:
            px, py = pts.coords[:, 0], pts.coords[:, 1]
            self.p_hgt = np.uint32(np.max(py) + 1)
            self.p_wid = np.uint32(np.max(px) + 1)

        if bounds is None:
            self.lon = np.linspace(-180.0, 180.0, self.p_wid)
            self.lat = np.linspace(-90.0, 90.0, self.p_hgt)
        else:
            # Bounds: (lon_min, lon_max, lat_min, lat_max)
            lon_min, lon_max, lat_min, lat_max = bounds
            self.lon = np.linspace(float(lon_min), float(lon_max), self.p_wid)
            self.lat = np.linspace(float(lat_min), float(lat_max), self.p_hgt)

    def forward(self, pts: Points) -> NDArray:
        """Convert p_pix → g_gcd.

        World net_mode (preferred): if the `p_pix` domain carries an `extent`, we
        interpret p_pix coords as (lon, lat) already.

        Legacy index net_mode: if no extent is available, interpret coords as
        (px,py) pixel indices and map via lookup tables.
        """
        dom = getattr(pts, 'domain', None)
        extent = getattr(dom, 'extent', None)

        # World net_mode: coords already represent (lon,lat)
        if extent is not None:
            lo = pts.coords[:, 0].astype(np.float64)
            la = pts.coords[:, 1].astype(np.float64)
            ret = np.stack([la, lo], axis=-1)
            return Points(ret, domain=self.fwd_cs, samples=pts.samples)

        # Legacy index net_mode
        if self.lon is None:
            self.set_dim(pts)

        px = pts.coords[:, 0].astype(np.uint32)
        py = pts.coords[:, 1].astype(np.uint32)
        lo = self.lon[px]  # longitude is <w--e>
        la = self.lat[py]  # latitude is <s--n> depending on how indices were defined
        ret = np.stack([la, lo], dtype=np.float64, axis=-1)
        return Points(ret, domain=self.fwd_cs, samples=pts.samples)

    def backward(self, pts: Points) -> NDArray:
        """Convert g_gcd → p_pix.

        World net_mode (preferred): if the reverse domain (`p_pix`) carries an
        `extent`, return world coords (lon,lat) in that frame.

        Legacy index net_mode: if no extent is available, return (px,py) indices via
        lookup tables.
        """
        # Coordinates are in (lat, lon) order from GCD domain
        la = pts.coords[:, 0].astype(np.float64)
        lo = pts.coords[:, 1].astype(np.float64)

        # Try to detect whether the reverse domain carries world extent.
        # The reverse CS is `p_pix` for this projection.
        dom = getattr(self, 'rev_cs', None)
        extent = getattr(dom, 'extent', None)

        # World net_mode: return (lon,lat) directly.
        if extent is not None:
            ret = np.stack([lo, la], axis=-1)
            return Points(ret, domain=self.rev_cs, samples=pts.samples)

        # Legacy index net_mode
        if self.lon is None:
            self.set_dim(Points(np.stack([lo, la], axis=-1), domain=self.fwd_cs, samples=pts.samples))

        px = np.searchsorted(self.lon, lo)
        py = np.searchsorted(self.lat, la)
        ret = np.stack([px, py], axis=-1)
        return Points(ret, domain=self.rev_cs, samples=pts.samples)
