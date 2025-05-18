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

    def forward(self, pix: Points) -> NDArray:
        """
        INPUT:  Plate Carrée coordinates (origin bottom-left)
        OUTPUT: GCD
        """
        px = pix[:, -2].astype(np.uint32)
        py = pix[:, -1].astype(np.uint32)

        # lat_min, lat_max = self.fwd_cs.lat_min, self.fwd_cs.lat_max
        # lon_min, lon_max = self.fwd_cs.lon_min, self.fwd_cs.lon_max
        p_hgt = self.rev_cs.height
        p_wid = self.rev_cs.width

        lon = np.linspace(-180+self.e, 180-self.e, p_wid)
        lat = np.linspace(-90, 90, p_hgt)
        # j = max(px)
        lx = np.array([lon[x] for x in px])
        ly = np.array([lat[y] for y in py])
        # lon2d, lat2d = np.meshgrid(lon, lat)
        # Normalize pixel coordinates
        # lat = lat_min + (lat_max - lat_min) * (py / (p_hgt - self.e))
        # lon = lon_min + (lon_max - lon_min) * (px / (p_wid - self.e))
        # lon_range = (lon_max - lon_min) % 360.
        # if lon_range == 0 and lon_max != lon_min:
        #     lon_range = 360.
        #
        # lon = lon_min + lon_range * (px / (p_wid - 1))
        # # lon = (lon + 180.) % 360. - 180.
        ret = np.array(pix, dtype=np.float64)
        # ret = pix.copy()
        ret[:, -2] = ly
        ret[:, -1] = lx
        return ret.view(Points).set_domain(self.fwd_cs)

    def backward(self, pts: Points) -> NDArray:
        """
        Convert (lat, lon) to pixel (x, y) coordinates using plate carrée.
        Plate Carrée coordinates (origin bottom-left)
        """
        p_hgt = self.rev_cs.height
        p_wid = self.rev_cs.width

        lon = np.linspace(-180+self.e, 180-self.e, p_wid)
        lat = np.linspace(-90, 90, p_hgt)

        la = pts[:, -2].astype(np.float64)
        lo = pts[:, -1].astype(np.float64)

        px = np.searchsorted(lon, lo)
        py = np.searchsorted(lat, la)


        # lat_min, lat_max = self.fwd_cs.lat_min, self.fwd_cs.lat_max
        # lon_min, lon_max = self.fwd_cs.lon_min, self.fwd_cs.lon_max
        # p_hgt = self.rev_cs.height
        # p_wid = self.rev_cs.width

        # Handle longitude wraparound
        # lon_range = (lon_max - lon_min) % 360.
        # if lon_range == 0 and lon_max != lon_min:
        #     lon_range = 360.
        #
        # delta_lon = lon - lon_min
        # px = delta_lon / lon_range * (p_wid - 1)
        # px = (lon - lon_min) / (lon_max - lon_min) * (p_wid - self.e)
        # py = (lat - lat_min) / (lat_max - lat_min) * (p_hgt - self.e)

        ret = pts.copy()
        ret[:, -2] = px
        ret[:, -1] = py
        return ret.view(Points).set_domain(self.rev_cs)


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

    # p_plt.height = 1800
    # p_plt.width = 3600

    # l0 = pg.forward(p0)
    # p1 = pg.backward(l0)

    # Project to lat/lon and back
    l1 = reg.project(p0, [p_plt, g_sph])
    p1 = reg.project(l1, [g_sph, p_plt])

    # Compute pixel round-trip error
    original_px = p0[:, -2:]
    projects_px = np.array(p1[:, -2:], dtype=np.uint64)
    dv = np.unique(projects_px-original_px, axis=0)
    # p0y_count = np.unique(original_px[:, -2]).shape[0]
    # p1y_count = np.unique(projects_px[:, -2]).shape[0]
    # p0x_count = np.unique(original_px[:, -1]).shape[0]
    # p1x_count = np.unique(projects_px[:, -1]).shape[0]

    px_error = np.linalg.norm(original_px - projects_px, axis=1)
    p1[:, 0] = px_error
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
