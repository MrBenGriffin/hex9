"""
Part of the H9 project
"""
import numpy as np
from numpy.typing import NDArray
from hhg9 import Points
from hhg9.base.projection import Projection


# class SphericalGCD(Projection):
#     """
#         Spherical XYZ<=>LatLon projection (spherical)
#         This is alternative 'SP'
#     """
#
#     def __init__(self, registrar):
#         super().__init__(registrar, 'SP', 'c_sph', 'g_gcd')
#
#     def forward(self, arr: Points) -> NDArray:
#         """Standard Cartesian->GCD projection."""
#         px, py, pz = arr.coords[..., 0], arr.coords[..., 1], arr.coords[..., 2]
#         lat = np.degrees(np.arctan2(pz, np.sqrt(px ** 2. + py ** 2.)))
#         lon = np.degrees(np.arctan2(py, px))
#         result = np.stack([lat, lon], axis=1)
#         return Points(result, domain=self.fwd_cs, samples=arr.samples)
#
#     def backward(self, arr: Points) -> NDArray:
#         """Standard GCD->Cartesian projection."""
#         phi, theta = np.radians(arr.coords[..., 0]), np.radians(arr.coords[..., 1])
#         x = (np.cos(phi) * np.cos(theta)).reshape(-1,)
#         y = (np.cos(phi) * np.sin(theta)).reshape(-1,)
#         z = np.sin(phi).reshape(-1,)  # z is 'up'
#         result = np.stack([x, y, z], axis=-1)
#         return Points(result, domain=self.rev_cs, samples=arr.samples)


if __name__ == '__main__':
    # from support import Util, Display, Photo
    from hhg9 import Registrar
    # from hhg9.domains import SphericalCartesian, GeneralGCD, PlatePixel
    #
    reg = Registrar()
    # p_pix = PlatePixel(reg)
    # c_sph = SphericalCartesian(reg)  # Cartesian Spherical (xyz)
    # g_sph = GeneralGCD(reg)  # Cartesian Spherical (xyz)
    # cg = GeneralGCD(reg)
    #
    # ps = Photo()
    # ps.load('../../preparatory/world1350x675.png')
    # # use plate_carrée pc_px domain to adopt the ps.image (shape [675,1350,4], with RGBA)
    # pc_px = p_pix.adopt(ps.img)
    # sp_ll = reg.project(pc_px, [p_pix, g_sph])
    #
    # d = Display()  # simple support display class
    # util = Util()
    # s1 = util.sph_rnd(1000000)
    # g1 = reg.project(s1, [c_sph, g_sph])
    # s2 = reg.project(g1, [g_sph, c_sph])
    # er = np.linalg.norm(s2-s1, axis=-1, keepdims=True)
    # ur = np.sort(np.unique(er))[::-1]
    # print(ur[0])
