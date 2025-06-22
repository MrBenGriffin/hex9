import numpy as np
from numpy.typing import NDArray
from hhg9 import Projection, Points
from pyproj import CRS, Transformer

from hhg9.projections import PlatePixelGCD


class EllipsoidGCD(Projection):
    """
    Ellipsoidal XYZ <=> LatLon projection using WGS84.
    Forward: ECEF → (lat, lon)
    Backward: (lat, lon) → ECEF
    """

    def __init__(self, registrar):
        super().__init__(registrar, 'ell_gcd', 'c_ell', 'g_gcd')
        # Define coordinate systems
        crs_geodetic = CRS.from_epsg(4326)  # WGS84 lat/lon
        crs_ecef = CRS.from_epsg(4978)      # WGS84 ECEF (x, y, z)

        # pyproj Transformers
        self.to_geodetic = Transformer.from_crs(crs_ecef, crs_geodetic, always_xy=True)
        self.to_ecef = Transformer.from_crs(crs_geodetic, crs_ecef, always_xy=True)

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


if __name__ == '__main__':
    from support import Util, Display, Photo
    from hhg9 import Registrar
    from hhg9.domains import GeneralGCD, PlatePixel, EllipsoidCartesian

    reg = Registrar()
    p_plt = PlatePixel(reg)
    e_sph = EllipsoidCartesian(reg)
    g_gen = GeneralGCD(reg)      # (xyz)
    ppg = PlatePixelGCD(reg)
    prj = EllipsoidGCD(reg)   # EllipsoidalGCD<->Cartesian (xyz)

    ps = Photo()
    ps.load('../../preparatory/world1350x675.png')
    # use plate_carrée pc_px domain to adopt the ps.image (shape [675,1350,4], with RGBA)
    pc_px = p_plt.adopt(ps.img)
    gg_ll = reg.project(pc_px, [p_plt, g_gen])

    gg_el = reg.project(gg_ll, [g_gen, e_sph])

    d = Display()  # simple support display class
    d.show_pts_3d(gg_el)
