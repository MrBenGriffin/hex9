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
    """
        Convert Plate Carrée pixel to latitude/longitude
    """

    def __init__(self, registrar):
        super().__init__(registrar, 'pix_gcd', 'p_pix', 'g_gcd')
        self.e = 1e-100
        self.p_hgt = None
        self.p_wid = None
        self.lon = None
        self.lat = None

    def set_dim(self, pts: Points = None, bounds: object = None) -> object:
        """Set dimensions for plate."""
        if self.p_hgt is None or pts is not None:
            px, py = pts.coords[:, 0], pts.coords[:, 1]
            self.p_hgt = np.uint32(np.max(py)+1)
            self.p_wid = np.uint32(np.max(px)+1)
        if bounds is None:
            self.lon = np.linspace(-180+self.e, 180-self.e, self.p_wid)
            self.lat = np.linspace(-90, 90, self.p_hgt)
        else:
            # Bounds: (lon_min, lon_max, lat_min, lat_max)
            lon_min, lon_max, lat_min, lat_max = bounds
            self.lon = np.linspace(lon_min+self.e, lon_max-self.e, self.p_wid)
            self.lat = np.linspace(lat_min, lat_max, self.p_hgt)

    def forward(self, pts: Points) -> NDArray:
        """
        INPUT:  Plate Carrée coordinates (origin bottom-left)
        OUTPUT: GCD
        """
        if self.lon is None:
            self.set_dim(pts)
        px, py = pts.coords[:, 0].astype(np.uint32), pts.coords[:, 1].astype(np.uint32)
        lx = self.lon[px]  # longitude is <w--e>
        ly = self.lat[py]  # latitude is <n--s>
        ret = np.stack([ly, lx], dtype=np.float64, axis=-1)
        return Points(ret, domain=self.fwd_cs, samples=pts.samples)

    def backward(self, pts: Points) -> NDArray:
        """
        Convert (lat, lon) to pixel (x, y) coordinates using the
        lookup tables created by set_dim.
        """
        if self.lon is None:
            self.set_dim(pts)

        # Coordinates are in (lat, lon) order from GCD domain
        la = pts.coords[:, 0].astype(np.float64)
        lo = pts.coords[:, 1].astype(np.float64)

        # Use the pre-calculated lookup tables from set_dim.
        # Do NOT recalculate them here. These are correctly bounded to your map.
        px = np.searchsorted(self.lon, lo)
        py = np.searchsorted(self.lat, la)

        # Return pixel coordinates in (x, y) order
        ret = np.stack([px, py], axis=-1)
        return Points(ret, domain=self.rev_cs, samples=pts.samples)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    from hhg9 import Registrar
    from hhg9.domains import EllipsoidCartesian, GeneralGCD, PlatePixel

    reg = Registrar()
    p_pix = PlatePixel(reg)
    c_sph = EllipsoidCartesian(reg)  # Cartesian Spherical (xyz)
    g_sph = GeneralGCD(reg)  # Cartesian Spherical (xyz)
    pg = PlatePixelGCD(reg)

    # # Create dummy image of shape (18, 36, 3)
    h, w = 1800, 3600
    img = np.zeros((h, w, 1))
    p0 = p_pix.adopt(img)  # Shape: (648, 5)
    # p0 = np.array([[i, 1799] for i in range(3600)])

    # Project to lat/lon and back
    l1 = reg.project(p0, [p_pix, g_sph])
    p1 = reg.project(l1, [g_sph, p_pix])

    # Compute pixel round-trip error
    original_px = np.array(p0.coords, dtype=np.uint64)
    projects_px = np.array(p1.coords, dtype=np.uint64)

    px_error = np.linalg.norm(original_px - projects_px, axis=1)
    p1.samples = px_error
    p2 = p_pix.image(p1)
    plt.imshow(p2, origin='lower')
    plt.show()

    plt.imshow(p2, cmap='hot', origin='lower')
    plt.colorbar(label="Pixel error (L2 norm)")
    plt.title("Round-trip pixel error (Plate Carrée)")
    plt.show()

    error_img = px_error.reshape(h, w)
    plt.imshow(error_img, cmap='hot', origin='lower')
    plt.colorbar(label="Pixel error (L2 norm)")
    plt.title("Round-trip pixel error (Plate Carrée)")
    plt.show()
