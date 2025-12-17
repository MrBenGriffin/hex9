# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
AK Octahedral Projection 'oct_ell'
"""
from functools import cache
import numpy as np
from numpy.typing import NDArray
from hhg9 import Points
from hhg9.base.projection import Projection
from hhg9.algorithms.distance import haversine_rad
from hhg9.algorithms import find_coords
from pyproj import CRS
from hhg9.h9 import H9C, H9K


class AKOctahedralEllipsoid(Projection):
    """
        An Octahedron/Ellipsoid Projection generated via an analytical approximation to a
        force-directed dataset. Approximation designer: Anders Kaseorg
    """
    ALPHA = 3.227806237143884260376580  # 𝛂 - vis. Kaseorg.

    def __init__(self, registrar, name='oct_ell'):
        self.reg = registrar
        super().__init__(self.reg, name, 'c_oct', 'c_ell')
        crs_ecef = CRS.from_epsg(4978)  # WGS84 ECEF (x, y, z)
        ecef_e = crs_ecef.ellipsoid
        self.ab = ecef_e.semi_major_metre, ecef_e.semi_minor_metre
        self.ab2 = 1., (self.ab[1] / self.ab[0]) ** 2

        self.b_oct = self.reg.domain('b_oct')
        self.c_oct = self.reg.domain('c_oct')
        self.c_ell = self.reg.domain('c_ell')
        self.g_gcd = self.reg.domain('g_gcd')
        self.vertices = np.array(list(self.rev_cs.vertices.values()))
        self._e = 1e-200
        self.tol = 1e-15

        # Level 0 area ≈ 5362 km (area of Earth / 12 hexes)
        earth = 510_065_621_724_154.6
        self.hex_0 = earth / 12
        self.tri_0 = earth / 8
        l_hex = self.hex_0
        l_tri = self.tri_0
        self.h_areas = np.zeros((64,), dtype=np.float64)
        self.t_areas = np.zeros((64,), dtype=np.float64)
        for i in range(64):
            self.h_areas[i] = l_hex
            self.t_areas[i] = l_tri
            l_hex /= 9
            l_tri /= 9
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

    def _core_raw(self, uvw: NDArray) -> NDArray:
        """
        Vectorized core projection: maps points from the unit octahedron to the unit sphere.
        Handles edge cases where one coordinate is near zero (i.e., edge of the octant).
        Returns the un-normalised projection vector.
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
        return pv

    def _core(self, uvw: NDArray) -> NDArray:
        pv = self._core_raw(uvw)
        return self.normalise(pv)

    def get_accuracy(self, layer):
        """
        Get accuracy in m2 from a given level.
        """
        if layer < 0:
            raise ValueError("get_accuracy: layer must be >= 0")
        h_area = self.h_areas[-1]
        t_area = self.t_areas[-1]
        if layer < len(self.h_areas):
            h_area = self.h_areas[layer]
            t_area = self.t_areas[layer]
        # side = np.sqrt((2*area)/(3*H9K.R3))
        return h_area, t_area

    def set_accuracy(self, m2):
        """
        Set the level such that the hex area is ≤ desired accuracy in m2.
        """
        idx = np.searchsorted(self.h_areas[::-1], m2, side='right')
        self.accuracy = len(self.h_areas) - idx
        return self.accuracy

    def normalise(self, p):
        """Normalise result to elliptical coordinates"""
        xx, yy, zz = p[..., 0], p[..., 1], p[..., 2]
        a2, b2 = self.ab2
        n = np.sqrt((xx ** 2 + yy ** 2) / a2 + zz ** 2 / b2)
        return np.stack([xx / n, yy / n, zz / n], axis=-1)

    def forward(self, arr: Points) -> Points:
        """
        c_oct->c_ell projection
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
        if np.any(aa):
            # map them to the ellipsoid as well
            uvw[aa] = self.normalise(uvw[aa])
        core_abs = np.abs(uvw)
        # Apply signs: components with sign==0 remain exactly 0
        res = self.ab[0] * (core_abs * sgn)
        return Points(res, domain=self.fwd_cs, samples=arr.samples, components=arr.components)

    def backward(self, arr: Points) -> Points:
        """
        c_ell->c_oct projection using find_coords rootfinding algorithm.
        """
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

        found, _ = find_coords(ref, oct_m, cmp, H9C, fwd, haversine_rad, self.accuracy+2, beam_width=6)
        bpt = Points(found, self.b_oct, uvw.components)
        return self.reg.project(bpt, [self.b_oct, self.rev_cs])  # rev_cs = c_oct

    def _core_jacobian(self, uvw):
        """
        :param uvw:
        :return:
        """
        alpha = self.ALPHA
        # match the epsilon shift used in _core to avoid tiny inconsistencies
        t = np.tan((np.pi * uvw + self._e) * 0.5)
        s = (np.pi * 0.5) * (1 + t * t)  # d/du tan(πu/2)

        u, v, w = t[..., 0], t[..., 1], t[..., 2]
        u2, v2, w2 = u * u, v * v, w * w

        def f_pow(b, c):
            # (b + c + alpha*b*c) ** 1/4
            return (b + c + alpha * b * c) ** 0.25

        def df_db(b, c):
            return 0.25 * (b + c + alpha * b * c) ** (-0.75) * (1 + alpha * c)

        def df_dc(b, c):
            return 0.25 * (b + c + alpha * b * c) ** (-0.75) * (1 + alpha * b)

        g0 = f_pow(v2, w2)
        g1 = f_pow(u2, w2)
        g2 = f_pow(u2, v2)

        # partials for y0, y1, y2 w.r.t. (u, v, w)
        dy0_du = s[..., 0] * g0
        dy0_dv = u * s[..., 1] * df_db(v2, w2) * 2 * v
        dy0_dw = u * s[..., 2] * df_dc(v2, w2) * 2 * w

        dy1_du = v * s[..., 0] * df_db(u2, w2) * 2 * u
        dy1_dv = s[..., 1] * g1
        dy1_dw = v * s[..., 2] * df_dc(u2, w2) * 2 * w

        dy2_du = w * s[..., 0] * df_db(u2, v2) * 2 * u
        dy2_dv = w * s[..., 1] * df_dc(u2, v2) * 2 * v
        dy2_dw = s[..., 2] * g2

        j = np.empty(uvw.shape[:-1] + (3, 3), dtype=uvw.dtype)
        j[..., 0, 0] = dy0_du
        j[..., 0, 1] = dy0_dv
        j[..., 0, 2] = dy0_dw
        j[..., 1, 0] = dy1_du
        j[..., 1, 1] = dy1_dv
        j[..., 1, 2] = dy1_dw
        j[..., 2, 0] = dy2_du
        j[..., 2, 1] = dy2_dv
        j[..., 2, 2] = dy2_dw
        return j

    def _norm_jacobian(self, p):
        a2, b2 = self.ab2
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        n = np.sqrt((x ** 2 + y ** 2) / a2 + z ** 2 / b2)

        # Mp (ellipsoidal metric times p)
        mp = np.stack([x / a2, y / a2, z / b2], axis=-1)  # (...,3)

        eye = np.eye(3)
        eye = np.broadcast_to(eye, p.shape[:-1] + (3, 3))  # (...,3,3)

        # make n broadcastable
        n2 = n[..., None, None]
        mp_scaled = mp / (n[..., None] ** 2)

        # Df = (I - p ⊗ (Mp)/n**2) / n
        outer = p[..., :, None] * mp_scaled[..., None, :]  # (...,3,3)
        return (eye - outer) / n2

    def jacobian(self, uvw):
        """
        :param uvw:
        :return: jacobian of the forward projection
        """
        uvw = np.asarray(uvw, dtype=np.float64, copy=True)
        aa = self._invariants(uvw)
        # Invariant points: match the forward() behaviour x ↦ a * x
        # i.e. take J ≈ a * I (this is a convention, but consistent with the code path)
        if np.any(aa):
            a = uvw[aa]
            a[a == 0] = np.finfo(np.float64).tiny
            c = np.sign(a)/np.sqrt(3)
            adj = a + 1e-15 * (c - a)
            uvw[aa] = adj

        # Non-invariant points: current analytic Jacobian
        # if np.any(~aa):
        u = uvw
        y_raw = self._core_raw(u)
        y = self.normalise(y_raw)
        j_core = self._core_jacobian(u)
        j_norm = self._norm_jacobian(y_raw)
        j_sub = j_norm @ j_core

        sgn = np.sign(u)
        dabs = np.sign(y) * sgn
        j_sub = dabs[..., :, None] * j_sub

        j = self.ab[0] * j_sub

        return j


