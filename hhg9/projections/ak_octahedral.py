"""
Part of the H9 project
"""
import os

import numpy as np
from geographiclib.geodesic import Geodesic
from numpy.typing import NDArray
from hhg9 import Projection, Points, H9Engine, Step
from hhg9.formats import OctahedralH9


class AKOctahedral:
    """
        An Octahedron/(Sphere/Elliptical) Projection generated via an analytical approximation to a
        force-directed dataset. Approximation designer: Anders Kaseorg
    """
    ALPHA = 3.227806237143884260376580  # 𝛂 - vis. Kaseorg.

    def __init__(self, reg, oct_dom, sp_norm_fn):
        self.h9 = OctahedralH9()  # formatter.
        self.h9e = H9Engine()
        self.reg = reg
        self.b_oct = reg.domain('b_oct')
        self.c_oct = reg.domain('c_oct')
        self.c_ell = reg.domain('c_ell')
        self.g_gcd = reg.domain('g_gcd')
        self.vertices = np.array(list(oct_dom.vertices.values()))
        self.geo = Geodesic.WGS84
        self._e = 1e-200
        self.tol = 1e-40

        # Level 0 hex diameter ≈ 5362 km (area of Earth / 12 hexes)
        self.diameters = 5362177 / (3 ** np.arange(32))  # in meters
        self.accuracy = 28  # accuracy is sub mm - based on hexagon layers. over 31 hits limits.
        self.sp_norm_fn = sp_norm_fn

    def _invariants(self, v):
        """Return invariant points: Those which are on the vertices themselves"""
        diff = np.abs(v[..., None, :] - self.vertices[..., :, :])  # shape (?, 8, 3)
        matches = np.all(diff < self.tol, axis=-1)  # shape (1000, 8)
        return np.array(np.any(matches, axis=-1))  # indices of v

    def _core(self, uvw: NDArray) -> NDArray:
        """
        Vectorized core projection: maps points from the unit octahedron to the unit sphere.
        Handles edge cases where one coordinate is near zero (i.e., edge of the octant).
        """
        uvw = np.asarray(uvw)
        α = self.ALPHA
        e = self._e
        tol = self.tol

        t_uvw = np.tan((np.pi * uvw + e) * 0.5)
        xu, xv, xw = t_uvw[..., -3], t_uvw[..., -2], t_uvw[..., -1]
        u2, v2, w2 = xu ** 2, xv ** 2, xw ** 2

        # Default calculation
        y0p = np.asarray(xu * (v2 + w2 + α * w2 * v2) ** 0.25)
        y1p = np.asarray(xv * (u2 + w2 + α * u2 * w2) ** 0.25)
        y2p = np.asarray(xw * (u2 + v2 + α * u2 * v2) ** 0.25)
        pv = np.stack([y0p, y1p, y2p], axis=-1)
        return self.sp_norm_fn(pv)

    def _geo_distance(self, p1, p2):
        return self.geo.Inverse(p1[0], p1[1], p2[0], p2[1])['s12']

    def reverse(self, uvw):
        """
        Given spheroidal coordinates, return octahedral.
        This uses branch and bound strategy via the H9 Grid.
        """
        # vt = self._invariants(uvw.coords)
        """Reverse function for a given value."""
        rll = self.reg.project(uvw, [self.c_ell, self.g_gcd])
        for idx, (ref_ll, s) in enumerate(zip(rll.coords, uvw.components)):
            dom = self.b_oct.components[tuple(s)]
            steps = [Step(dom.tr, 0, 0)]
            beam_width = 6
            candidates = [(steps[0], np.inf)]  # tuple of (hex, score)
            for i in range(self.accuracy):
                next_candidates = []
                for stp, _ in candidates:
                    cs = self.h9e.branch_step(stp)  # Next level hexes
                    vx = np.array([[c.x, c.y] for c in cs])
                    bvx = dom.adopt(vx)
                    grx = self.reg.project(bvx, [self.b_oct, self.c_oct, self.c_ell, self.g_gcd])
                    for gdx, latlon in enumerate(grx.coords):
                        dist = self._geo_distance(latlon, ref_ll)
                        next_candidates.append((cs[gdx], dist))

                # Sort and prune
                next_candidates.sort(key=lambda x: x[1])
                candidates = next_candidates[:beam_width]

                # Pick best to display at this level

            best = candidates[0][0]
            bary = dom.adopt(np.array([[best.x, best.y]]))
            grx = self.reg.project(bary, [self.b_oct, self.c_oct])
            uvw.coords[idx] = grx.coords
        return uvw

    def forward(self, uvw):
        """given octahedral coordinates, return spheroidal."""
        aa = self._invariants(uvw)
        trx = self._core(uvw[~aa])
        uvw[~aa] = trx
        return uvw

    def set_accuracy(self, meters):
        """
        Set the level such that the hex diameter is ≤ desired accuracy in meters.
        """
        idx = np.searchsorted(self.diameters[::-1], meters, side='right')
        self.accuracy = len(self.diameters) - idx
        return self.accuracy


class AKOctahedralEllipsoid(Projection):
    """
        An Octahedron/Ellipsoid Projection generated via an analytical approximation to a
        force-directed dataset. Approximation designer: Anders Kaseorg
    """
    def __init__(self, registrar):
        super().__init__(registrar, 'ake', 'c_oct', 'c_ell')
        self.ab = 6378137.0, 6356752.3142
        self.ab2 = 1., (self.ab[1] / self.ab[0]) ** 2
        self.ak = AKOctahedral(registrar, self.rev_cs, self.normalise)

    def set_accuracy(self, meters):
        """Control amount of work needed in the reverse."""
        return self.ak.set_accuracy(meters)

    def normalise(self, p):
        """Normalise result to elliptical coordinates"""
        xx, yy, zz = p[..., 0], p[..., 1], p[..., 2]
        a2, b2 = self.ab2
        n = np.sqrt((xx**2 + yy**2) / a2 + zz**2 / b2)
        return np.stack([xx / n, yy / n, zz / n], axis=-1)

    def forward(self, arr: Points) -> NDArray:
        """
        Convert a NDArray of octahedral points projected onto WGS84 Ellipsoid
        Anders Kaseorg: https://math.stackexchange.com/questions/5016695/
        :param pts:  An array of Euclidean points on the surface of a unit octahedron.
        :return: UVW on WGS84 Ellipsoid
        """
        xyz = arr.coords
        res = self.ab[0] * np.copysign(self.ak.forward(xyz), xyz)
        return Points(res, domain=self.fwd_cs, samples=arr.samples, components=arr.components)

    def backward(self, arr: Points) -> NDArray:
        """
         Project WGS84 Ellipsoids onto the octahedron
         This inverse function using numerical optimization
         :param pts:  An array of Euclidean points on the surface of the WGS84 Ellipsoid.
         :return: UVW on a unit octahedron.
        """
        uvw = arr.copy()
        if uvw.components is None:
            self.rev_cs.binning(uvw)
        self.ak.reverse(uvw)
        uvw.domain = self.rev_cs
        return uvw



class AKOctahedralSpherical(Projection):
    """
        An Octahedron/Sphere Projection generated via an analytical approximation to a
        force-directed dataset. Approximation designer: Anders Kaseorg
    """
    def __init__(self, registrar):
        super().__init__(registrar, 'ake', 'c_oct', 'c_sph')
        self.ak = AKOctahedral(registrar, self.rev_cs, self.normalise)

    def normalise(self, pv):
        """Normalise result to spherical coordinates"""
        return pv / np.linalg.norm(pv, axis=-1, keepdims=True)

    # def jac(self):
    #     a = AKApproximation.ALPHA
    #     u, v, w = sp.symbols('u v w')  # Define symbolic variables for inputs
    #     tan_u = sp.tan(sp.pi * u / 2)
    #     tan_v = sp.tan(sp.pi * v / 2)
    #     tan_w = sp.tan(sp.pi * w / 2)
    #
    #     u2 = tan_u ** 2
    #     v2 = tan_v ** 2
    #     w2 = tan_w ** 2
    #
    #     y0p = tan_u * (v2 + w2 + a * w2 * v2) ** 0.25
    #     y1p = tan_v * (u2 + w2 + a * u2 * w2) ** 0.25
    #     y2p = tan_w * (u2 + v2 + a * u2 * v2) ** 0.25
    #
    #     # Combine outputs into a vector
    #     y = sp.Matrix([y0p, y1p, y2p])
    #
    #     # Normalize the vector (divide by its magnitude)
    #     norm = sp.sqrt(y[0] ** 2 + y[1] ** 2 + y[2] ** 2)
    #     y_normalized = y / norm
    #
    #     variables = [u, v, w]
    #     jacobian = y_normalized.jacobian(variables)
    #     return sp.lambdify(sp.Matrix(variables), jacobian, modules=['numpy'])

    def forward(self, arr: Points) -> NDArray:
        """
        Convert a NDArray of octahedral points projected onto a sphere
        Anders Kaseorg: https://math.stackexchange.com/questions/5016695/
        :param pts:  An array of Euclidean points on the surface of a unit octahedron.
        :return: UVW on a unit sphere.
        """
        xyz = arr.coords
        uvw = xyz / (np.linalg.norm(xyz, ord=1, axis=-1, keepdims=True))
        res = np.copysign(self.ak.forward(uvw), xyz)
        return Points(res, domain=self.fwd_cs, samples=arr.samples, components=arr.components)

    def backward(self, arr: Points) -> NDArray:
        """
         Projected a spherical point onto the octahedron
         This inverse function using numerical optimization
         :param pts:  An array of Euclidean points on the surface of a unit sphere.
         :return: UVW on a unit octahedron.
        """
        xyz = arr.coords
        res = np.copysign(self.ak.reverse(xyz), arr.coords)
        return Points(res, domain=self.rev_cs, samples=arr.samples, components=arr.components)


if __name__ == '__main__':
    from support import Util, Display
    from hhg9 import Registrar
    from hhg9.domains import EllipsoidCartesian, OctahedralCartesian, OctahedralBarycentric
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)

    lon, lat = -0.1278, 51.5074  # London
    x, y, z = transformer.transform(lon, lat, 0.0)  # Height = 0
    ellipsoid_point = np.array([[x, y, z]])

    reg = Registrar()
    c_ell = EllipsoidCartesian(reg)             # Cartesian Spherical (xyz)
    c_oct = OctahedralCartesian(reg)            # Cartesian Octahedron (xyz)
    ake = AKOctahedralEllipsoid(reg)
    test = c_ell.adopt(ellipsoid_point)
    wgs = ake.backward(test.copy())
    round_trip = ake.forward(wgs.copy())
    print("Ellipsoid residual:", np.linalg.norm(test.coords - round_trip.coords))  # Should be small

    reg = Registrar()
    c_ell = EllipsoidCartesian(reg)             # Cartesian Spherical (xyz)
    b_oct = OctahedralBarycentric(reg, c_oct)   # Barycentric Octahedron (xyz)
    ak = AKOctahedralEllipsoid(reg)

    d = Display()  # 0.044711 simple support display class
    u = Util()

    # x = np.array([
    #     [0.98, 0.01, 0.01],
    #     [-0.98, 0.01, 0.01],
    #     [0.01, 0.98, 0.01],
    #     [0.01, 0.01, 0.98],
    #     [-0.39269128, -0.72864642, 0.56113097],
    #     [-0.55192923, -0.16413949, 0.81757713],
    #     [-0.85376977, -0.39657625, 0.33734916],
    #     [0.62657288, -0.01997740, 0.77910675],
    #     [0.84136576, -0.26177117, 0.47284195],
    #     [0.23109285, -0.88246783, 0.40969088],
    #     [0.74143701, 0.44786834, 0.49968501],
    #     [-0.61293061, 0.53795414, 0.57872395],
    #     [0.78307239, 0.26161829, -0.56422823],
    #     [0.34584769, 0.23629583, -0.90804937],
    #     [0.42566755, 0.54205007, -0.72456115],
    #     [0.77058801, 0.62823186, -0.10732586],
    #     [0.63195672, 0.64372129, -0.43157110],
    #     [0.22731009, 0.60741974, -0.76116449],
    #     [-0.34232075, 0.93020867, -0.13239463],
    #     [-0.35661942, 0.82387026, -0.44052285],
    #     [-0.67290360, 0.13900220, -0.72655291],
    #     [-0.84038944, 0.51935069, -0.15498534],
    #     [-0.29382126, -0.84010879, -0.45594549],
    #     [0.33112712, -0.52420282, -0.78458029],
    #     [-0.34232075, -0.93020867, -0.13239463],
    #     [-0.55192923, -0.16413949, 0.81757713]
    # ])
    # x = u.oct0_ce_biased(25000, 0.25)

    x = u.oct_rnd(25000)
    x = np.abs(x)
    x = x / (np.linalg.norm(x, ord=1, axis=1, keepdims=True))
    ox = c_oct.adopt(x)
    fd = ak.forward(ox)
    bk = ak.backward(fd)
    rt = bk.coords - ox.coords
    zk = abs(rt) / np.linalg.norm(rt)
    mx = np.max(rt)
    zk *= 10000.
    oc = c_oct.adopt(x)
    bc = reg.project(oc, [c_oct, b_oct])
    bc.samples = zk
    d.show_pts_2d(bc, label=f'{mx}', clip=True)
