"""
Part of the H9 project
"""
import os

import numpy as np
import sympy as sp
from numpy.typing import NDArray
from hhg9 import Projection, Points
from scipy.optimize import root
import warnings
import json
from scipy.special import erf


class AKOctahedralSpherical(Projection):
    """
        An Octahedron/Sphere Projection generated via an analytical approximation to a
        force-directed dataset. Approximation designer: Anders Kaseorg
        cf. other/akre.py for attempts to improve minimum roundtrip.
    """

    def __init__(self, registrar):
        super().__init__(registrar, 'ak', 'c_oct', 'c_sph')
        self.vertices = np.array(list(self.rev_cs.vertices.values()))
        self._ALPHA = 3.227806237143884260376580  # 𝛂 - vis. Kaseorg.
        self._e = 1e-200
        self.tol = 1e-40
        self._jac_fn = None
        self.weights = None

    def _axis_aligned(self, v):
        diff = np.abs(v[..., None, :] - self.vertices[..., :, :])  # shape (?, 8, 3)
        matches = np.all(diff < self.tol, axis=-1)  # shape (1000, 8)
        return np.array(np.any(matches, axis=-1))  # indices of v

    def _core(self, uvw: NDArray) -> NDArray:
        """
        Vectorized core projection: maps points from the unit octahedron to the unit sphere.
        Handles edge cases where one coordinate is near zero (i.e., edge of the octant).
        """
        uvw = np.asarray(uvw)
        α = self._ALPHA
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
        return pv / np.linalg.norm(pv, axis=-1, keepdims=True)

    def _root(self, t, g):

        def _octnorm(v):
            return v / (np.linalg.norm(v, ord=1, keepdims=True))

        def _fwd(v):
            return self._core(v)
            # return self._core_mp([mp.mpf(i) for i in v])

        def _residual(x0):
            return _fwd(x0) - t

        def _jac(vx):
            return self._jac_fn(*vx)

        rt = root(_residual, x0=g, jac=_jac, tol=self.tol)
        rx = _octnorm(rt.x)
        gv = np.linalg.norm(_fwd(g) - t, axis=-1)
        gx = np.linalg.norm(_fwd(rx) - t, axis=-1)
        return rx if gx < gv else g

    def _rev_do(self, uvw):
        # import warnings
        # warnings.filterwarnings("error")
        if self._jac_fn is None:
            self._set_jac()

        vt = self._axis_aligned(uvw)
        """Reverse function for a given value."""

        ctr = uvw[~vt]
        if ctr.any():
            gss = self.rev_guess(ctr)
            rts = np.array([self._root(v, g) for v, g in zip(ctr, gss)])
            rx = rts / (np.linalg.norm(rts, ord=1, axis=-1, keepdims=True))
            uvw[~vt] = rx
        return uvw

    def _fwd_do(self, uvw):
        aa = self._axis_aligned(uvw)
        trx = self._core(uvw[~aa])
        uvw[~aa] = trx
        return uvw

    def _set_jac(self):
        if self._jac_fn:
            return
        a = self._ALPHA
        u, v, w = sp.symbols('u v w')  # Define symbolic variables for inputs
        tan_u = sp.tan(sp.pi * u / 2)
        tan_v = sp.tan(sp.pi * v / 2)
        tan_w = sp.tan(sp.pi * w / 2)
        # y0p, y1p, y2p = None, None, None
        # if d < sp.Abs(u) - self.tol:
        #     xv_m = tan_v
        #     xw_m = tan_w
        #     y0p = 0.0
        #     y1p = xv_m * (a * xw_m ** 2 + 1) ** 0.25
        #     y2p = xw_m * (a * xv_m ** 2 + 1) ** 0.25
        #
        # if sp.Abs(v) < self.tol:
        #     xu_m = tan_u
        #     xw_m = tan_w
        #     y0p = xu_m * (a * xw_m ** 2 + 1) ** 0.25
        #     y1p = 0.0
        #     y2p = xw_m * (a * xu_m ** 2 + 1) ** 0.25
        #
        # if sp.Abs(w) < self.tol:
        #     xu_m = tan_u
        #     xv_m = tan_v
        #     y0p = xu_m * (a * xv_m ** 2 + 1) ** 0.25
        #     y1p = xv_m * (a * xu_m ** 2 + 1) ** 0.25
        #     y2p = 0.0
        #
        # if y0p is None:
        u2 = tan_u ** 2
        v2 = tan_v ** 2
        w2 = tan_w ** 2

        y0p = tan_u * (v2 + w2 + a * w2 * v2) ** 0.25
        y1p = tan_v * (u2 + w2 + a * u2 * w2) ** 0.25
        y2p = tan_w * (u2 + v2 + a * u2 * v2) ** 0.25

        # Combine outputs into a vector
        y = sp.Matrix([y0p, y1p, y2p])

        # Normalize the vector (divide by its magnitude)
        norm = sp.sqrt(y[0] ** 2 + y[1] ** 2 + y[2] ** 2)
        y_normalized = y / norm

        variables = [u, v, w]
        jacobian = y_normalized.jacobian(variables)
        self._jac_fn = sp.lambdify(sp.Matrix(variables), jacobian, modules=['numpy'])

    def rev_guess(self, xyz: Points) -> NDArray:
        """
        Guess octahedral based on trained weights.
        Load weights and biases from JSON file
        """
        if self.weights is None:
            # {os.path.dirname(__file__)}/ak9.json'
            with open(f'{os.path.dirname(__file__)}/ak9big.json') as f:
                self.weights = json.load(f)

        a = np.abs(xyz)
        for i, (w, bss) in enumerate(self.weights.values()):
            wts = np.array(w)           # shape (in_dim, out_dim)
            bss = np.array(bss)         # shape (out_dim,)
            z = np.dot(a, wts) + bss    # shape (batch_size, out_dim)

            # Apply activation
            if i == 0 or i == 3:  # tanh on first and last layers
                a = np.tanh(z)
            elif i == 1 or i == 2:  # gelu on middle layers
                a = 0.5 * z * (1 + erf(z / np.sqrt(2)))
            else:
                a = z
        return np.copysign(a, xyz)

    def forward(self, pts: Points) -> NDArray:
        """
        Convert a NDArray of octahedral points projected onto a sphere
        Anders Kaseorg: https://math.stackexchange.com/questions/5016695/
        :param pts:  An array of Euclidean points on the surface of a unit octahedron.
        :return: UVW on a unit sphere.
        """
        pts = pts / (np.linalg.norm(pts, ord=1, axis=-1, keepdims=True))
        res = pts.copy()
        xyz = np.array(pts[..., -3:])
        res[..., -3:] = np.copysign(self._fwd_do(xyz), res)
        return res.view(Points).set_domain(self.fwd_cs)

    def backward(self, pts: Points) -> NDArray:
        """
         Projected a spherical point onto the octahedron
         This inverse function using numerical optimization
         :param pts:  An array of Euclidean points on the surface of a unit sphere.
         :return: UVW on a unit octahedron.
        """
        res = pts.copy()
        xyz = np.array(pts[..., -3:])
        res[..., -3:] = np.copysign(self._rev_do(xyz), xyz)
        return res.view(Points).set_domain(self.rev_cs)


if __name__ == '__main__':
    from support import Util, Display
    from hhg9 import Registrar
    from hhg9.domains import SphericalCartesian, OctahedralCartesian, OctahedralBarycentric

    reg = Registrar()
    c_sph = SphericalCartesian(reg)             # Cartesian Spherical (xyz)
    c_oct = OctahedralCartesian(reg)            # Cartesian Octahedron (xyz)
    b_oct = OctahedralBarycentric(reg, c_oct)   # Barycentric Octahedron (xyz)
    ak = AKOctahedralSpherical(reg)

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
    fd = ak.forward(x)
    bk = ak.backward(fd)
    rt = bk - x
    zk = abs(rt) / np.linalg.norm(rt)
    mx = np.max(zk)
    zk *= 1000.
    oc = c_oct.adopt(x)
    bc = reg.project(oc, [c_oct, b_oct])
    bc = np.insert(bc, 0, zk[:, 0], axis=1)
    bc = np.insert(bc, 0, zk[:, 1], axis=1)
    bc = np.insert(bc, 0, zk[:, 2], axis=1)
    d.show_pts_2d(bc, label=f'{mx}')
