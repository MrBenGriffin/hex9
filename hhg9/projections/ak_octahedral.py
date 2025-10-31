"""
Part of the H9 project
"""
from functools import cache
import numpy as np
from numpy.typing import NDArray
from hhg9 import Projection, Points
from hhg9.algorithms import find_coords, haversine_rad
from pyproj import CRS
from hhg9.h9 import H9C

class AKOctahedralEllipsoid(Projection):
    """
        An Octahedron/Ellipsoid Projection generated via an analytical approximation to a
        force-directed dataset. Approximation designer: Anders Kaseorg
    """
    ALPHA = 3.227806237143884260376580  # 𝛂 - vis. Kaseorg.

    def __init__(self, registrar):
        self.reg = registrar
        super().__init__(self.reg, 'oct_ell', 'c_oct', 'c_ell')
        crs_ecef = CRS.from_epsg(4978)  # WGS84 ECEF (x, y, z)
        ecef_e = crs_ecef.ellipsoid
        self.ab = ecef_e.semi_major_metre, ecef_e.semi_minor_metre
        self.ab2 = 1., (self.ab[1] / self.ab[0]) ** 2

        self.b_oct = reg.domain('b_oct')
        self.c_oct = reg.domain('c_oct')
        self.c_ell = reg.domain('c_ell')
        self.g_gcd = reg.domain('g_gcd')
        self.vertices = np.array(list(self.rev_cs.vertices.values()))
        self._e = 1e-200
        self.tol = 1e-40

        # Level 0 hex diameter ≈ 5362 km (area of Earth / 12 hexes)
        self.diameters = 5362177 / (3 ** np.arange(38))  # in meters
        self.accuracy = 34  # accuracy is nanometres.

    @cache
    def rad_gcd(self):
        """Return the radians GCD domain, registering it (and its deg⟷rad projection) on first use.
        This is lazy and idempotent: repeated calls reuse the same registrar entries.
        """
        # Imports kept local to avoid import cycles at module import time
        from hhg9.domains import RadiansGCD
        from hhg9.projections import RGCD_GCD

        reg = self.reg
        try:
            return reg.domain('r_gcd')
        except Exception:
            # not yet registered; fall through to create & wire projections
            pass

        r_gcd = RadiansGCD(reg)  # registers the domain
        RGCD_GCD(reg)  # registers deg↔rad projections (idempotent if already present)
        return r_gcd

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

        t_uvw = np.tan((np.pi * uvw + e) * 0.5)
        xu, xv, xw = t_uvw[..., -3], t_uvw[..., -2], t_uvw[..., -1]
        u2, v2, w2 = xu ** 2, xv ** 2, xw ** 2

        # Default calculation
        y0p = np.asarray(xu * (v2 + w2 + α * w2 * v2) ** 0.25)
        y1p = np.asarray(xv * (u2 + w2 + α * u2 * w2) ** 0.25)
        y2p = np.asarray(xw * (u2 + v2 + α * u2 * v2) ** 0.25)
        pv = np.stack([y0p, y1p, y2p], axis=-1)
        return self.normalise(pv)

    def set_accuracy(self, meters):
        """
        Set the level such that the hex diameter is ≤ desired accuracy in meters.
        """
        idx = np.searchsorted(self.diameters[::-1], meters, side='right')
        self.accuracy = len(self.diameters) - idx
        return self.accuracy

    def normalise(self, p):
        """Normalise result to elliptical coordinates"""
        xx, yy, zz = p[..., 0], p[..., 1], p[..., 2]
        a2, b2 = self.ab2
        n = np.sqrt((xx ** 2 + yy ** 2) / a2 + zz ** 2 / b2)
        return np.stack([xx / n, yy / n, zz / n], axis=-1)

    def forward(self, arr: Points) -> Points:
        """
        Convert a NDArray of octahedral points projected onto WGS84 Ellipsoid
        Anders Kaseorg: https://math.stackexchange.com/questions/5016695/
        :param pts:  An array of Euclidean points on the surface of a unit octahedron.
        :return: Points UVW on WGS84 Ellipsoid
        """
        xyz = arr.coords
        sgn = np.sign(xyz)
        # Run core forward on a copy so we don't lose the original signs
        uvw = np.asarray(xyz, dtype=np.float64).copy()  # avoid mutating caller
        aa = self._invariants(uvw)
        if np.any(~aa):
            trx = self._core(uvw[~aa])
            uvw[~aa] = trx
        core_abs = np.abs(uvw)
        # Apply signs: components with sign==0 remain exactly 0
        res = self.ab[0] * (core_abs * sgn)
        return Points(res, domain=self.fwd_cs, samples=arr.samples, components=arr.components)

    def backward(self, arr: Points) -> Points:
        r_gcd = self.rad_gcd()
        if arr.components is None:
            self.rev_cs.binning(arr)  # We need the octant identity for each point.
        uvw = arr.copy()
        cmp = uvw.components
        # cmp = uvw.components[:, np.newaxis, :]  # use this for referring to the points' octant identity.
        rll = self.reg.project(uvw, [self.c_ell, self.g_gcd, r_gcd])  # Project to give us GCD reference values.
        ref = rll.coords  # reference addresses
        _, oct_m = uvw.cm()  # we want their modes.

        def fwd(xy, octants):
            """Project contender xy (in barycentric) to GCD"""
            coords = Points(xy.reshape(-1, 2), self.b_oct, octants.reshape(-1, 3))
            grx = self.reg.project(coords, [self.b_oct, self.c_oct, self.c_ell, self.g_gcd, r_gcd])
            return grx.coords.reshape(xy.shape)

        found, _ = find_coords(ref, oct_m, cmp, H9C, fwd, haversine_rad, self.accuracy, beam_width=6)
        bpt = Points(found, self.b_oct, uvw.components)
        return self.reg.project(bpt, [self.b_oct, self.rev_cs])  # rev_cs = c_oct


if __name__ == '__main__':
    from matplotlib import image, pyplot as plt
    # from support import Util
    from hhg9 import Registrar
    from hhg9.domains import RadiansGCD, EllipsoidCartesian, OctahedralCartesian, OctahedralBarycentric, GeneralGCD
    from hhg9.projections import EllipsoidGCD, RGCD_GCD
    from hhg9.algorithms import wgs84
    from hhg9.algorithms.pickers import gcd_rnd

    reg = Registrar()
    g_gcd = GeneralGCD(reg)  # GCD Domain (latitude/longitude)
    c_ell = EllipsoidCartesian(reg)  # Cartesian Spherical (xyz)
    c_oct = OctahedralCartesian(reg)  # Cartesian Octahedron (xyz)
    b_oct = OctahedralBarycentric(reg, c_oct)  # Barycentric Octahedron (xyz)
    ake = AKOctahedralEllipsoid(reg)
    ake.set_accuracy(0.000000000001)
    EllipsoidGCD(reg)  # (g_gcd c_ell) Project (GCD <=> Geodesic Cartesian)

    london = np.array([[51.50744520, -0.1278120321]])  # London Latitude/Longitude
    ldn = g_gcd.adopt(london)
    t_oct = reg.project(ldn, [g_gcd, c_ell, c_oct])
    r_ldn = reg.project(t_oct, [c_oct, c_ell, g_gcd])
    delta = wgs84(ldn.coords[0], r_ldn.coords[0]) * 1e+9
    print("1nm Accuracy: Ellipsoid residual in nanometres:", delta)  # Should be small

    ake.set_accuracy(1000.0)
    t_oct = reg.project(ldn, [g_gcd, c_ell, c_oct])
    r_ldn = reg.project(t_oct, [c_oct, c_ell, g_gcd])
    delta = wgs84(ldn.coords[0], r_ldn.coords[0]) * 0.001
    print("1km Accuracy: Ellipsoid residual in kilometres:", delta)  # Should be small

    ake.set_accuracy(10000.0)
    wgs = gcd_rnd(25000)
    w_pts = Points(wgs, domain=g_gcd)
    ocs = reg.project(w_pts, [g_gcd, c_ell, c_oct])

    # dv = bk.coords - ox.coords
    # dd = np.linalg.norm(dv, axis=1)
    px = ocs.coords
    fig = plt.figure(figsize=(10, 10), dpi=100, frameon=False)
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(px[:, 0], px[:, 1], px[:, 2], marker='.', s=5.0)
    ax.set_aspect('equal', adjustable='box')
    plt.show()

    bak = reg.project(ocs, [c_oct, c_ell])
    px = bak.coords
    fig = plt.figure(figsize=(10, 10), dpi=100, frameon=False)
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(px[:, 0], px[:, 1], px[:, 2], marker='.', s=5.0)
    ax.set_aspect('equal', adjustable='box')
    plt.show()
