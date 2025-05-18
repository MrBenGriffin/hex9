"""
Part of the H9 project
"""
import numpy as np
from numpy.typing import NDArray
from hhg9 import Projection, Points


class CartesianGCD(Projection):
    """
        Standard XYZ<=>LatLon projection (spherical)
        This is alternative 'SP'
    """

    def __init__(self, registrar):
        super().__init__(registrar, 'SP', 'c_sph', 'g_sph')

    def forward(self, arr: Points) -> NDArray:
        """Standard Cartesian->GCD projection."""
        pts = np.array([arr]) if len(arr.shape) == 1 else arr
        x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
        lat = np.degrees(np.arctan2(z, np.sqrt(x ** 2. + y ** 2.)))
        lon = np.degrees(np.arctan2(y, x))
        result = np.stack([lat, lon], axis=1)
        return result.view(Points).set_domain(self.fwd_cs)

    def backward(self, arr: Points) -> NDArray:
        """Standard GCD->Cartesian projection."""
        phi, theta = np.radians(arr[..., -2]), np.radians(arr[..., -1])
        x = (np.cos(phi) * np.cos(theta)).reshape(-1, 1)
        y = (np.cos(phi) * np.sin(theta)).reshape(-1, 1)
        z = np.sin(phi).reshape(-1, 1)  # z is 'up'
        if arr.shape[-1] > 2:
            smp = np.delete(arr, [-2, -1], -1)
            result = np.hstack([smp, x, y, z])
        else:
            result = np.hstack([x, y, z])
        return result.view(Points).set_domain(self.rev_cs)


if __name__ == '__main__':
    from support import Util, Display, Photo
    from hhg9 import Registrar
    from hhg9.domains import SphericalCartesian, SphericalGCD, PlatePixel

    reg = Registrar()
    p_plt = PlatePixel(reg)
    c_sph = SphericalCartesian(reg)  # Cartesian Spherical (xyz)
    g_sph = SphericalGCD(reg)  # Cartesian Spherical (xyz)
    cg = CartesianGCD(reg)

    ps = Photo()
    ps.load('../preparatory/world1350x675.png')
    # use plate_carrée pc_px domain to adopt the ps.image (shape [675,1350,4], with RGBA)
    pc_px = p_plt.adopt(ps.img)
    sp_ll = reg.project(pc_px, [p_plt, g_sph])

    d = Display()  # simple support display class
    u = Util()
    s1 = u.sph_rnd(1000000)
    g1 = reg.project(s1, [c_sph, g_sph])
    s2 = reg.project(g1, [g_sph, c_sph])
    er = np.linalg.norm(s2-s1, axis=-1, keepdims=True)
    ur = np.sort(np.unique(er))[::-1]
    print(ur[0])
