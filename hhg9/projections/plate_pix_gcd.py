"""
Part of the H9 project
"""
import numpy as np
from numpy.typing import NDArray
from hhg9 import Projection, Points


class PlatePixelGCD(Projection):
    """
        Convert Plate Carrée pixel to latitude/longitude
    """

    def __init__(self, registrar):
        super().__init__(registrar, 'pg_plt', 'p_plt', 'g_sph')
        self.e = 1e-50
        self.p_hgt = None
        self.p_wid = None
        self.lon = None
        self.lat = None

    def set_dim(self, pts: Points):
        """Set dimensions for plate."""
        px, py = pts.coords[:, 0], pts.coords[:, 1]
        self.p_hgt = np.uint32(np.max(py)+1)
        self.p_wid = np.uint32(np.max(px)+1)
        self.lon = np.linspace(-180+self.e, 180-self.e, self.p_wid)
        self.lat = np.linspace(-90, 90, self.p_hgt)


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
        Convert (lat, lon) to pixel (x, y) coordinates using plate carrée.
        Plate Carrée coordinates (origin bottom-left)
        """
        if self.lon is None:
            raise TypeError(f"Points need to have x/y dimensions set")

        la = pts.coords[:, 0].astype(np.float64)
        lo = pts.coords[:, 1].astype(np.float64)

        px = np.searchsorted(self.lon, lo)
        py = np.searchsorted(self.lat, la)

        ret = np.stack([px, py], axis=-1)
        return Points(ret, domain=self.rev_cs, samples=pts.samples)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    from support import Photo
    from hhg9 import Registrar
    from hhg9.domains import SphericalCartesian, SphericalGCD, PlatePixel

    reg = Registrar()
    p_plt = PlatePixel(reg)
    c_sph = SphericalCartesian(reg)  # Cartesian Spherical (xyz)
    g_sph = SphericalGCD(reg)  # Cartesian Spherical (xyz)
    pg = PlatePixelGCD(reg)

    # # Create dummy image of shape (18, 36, 3)
    h, w = 1800, 3600
    img = np.zeros((h, w, 1))
    p0 = p_plt.adopt(img)  # Shape: (648, 5)
    # p0 = np.array([[i, 1799] for i in range(3600)])

    # Project to lat/lon and back
    l1 = reg.project(p0, [p_plt, g_sph])
    p1 = reg.project(l1, [g_sph, p_plt])

    # Compute pixel round-trip error
    original_px = np.array(p0.coords, dtype=np.uint64)
    projects_px = np.array(p1.coords, dtype=np.uint64)
    dv = np.unique(projects_px-original_px, axis=0)

    px_error = np.linalg.norm(original_px - projects_px, axis=1)
    p1.samples = px_error
    p2 = p_plt.image(p1)
    plt.imshow(p2, origin='lower')
    plt.show()

    plt.imshow(p2, cmap='hot', origin='lower')
    plt.colorbar(label="Pixel error (L2 norm)")
    plt.title("Round-trip pixel error (Plate Carrée)")
    plt.show()

    # Analyze errors
    error_img = px_error.reshape(h, w)
    # import matplotlib.pyplot as plt
    #
    plt.imshow(error_img, cmap='hot', origin='lower')
    plt.colorbar(label="Pixel error (L2 norm)")
    plt.title("Round-trip pixel error (Plate Carrée)")
    plt.show()
