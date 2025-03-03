from modern.osprojection import OSProjection
import sympy as sp
import numpy as np
from scipy.optimize import root


class AK(OSProjection):
    # This is concerned with generating the actual point values
    # for a sphere.
    _ALPHA = 3.227806237143884260376580641604959964752197265625  # 𝛂 - vis. Kaseorg.
    _jac_fn = None
    _e = 1e-20

    @classmethod
    def os(cls, uvw):
        # Convert a np.array of octahedral points projected onto a sphere
        # Anders Kaseorg: https://math.stackexchange.com/questions/5016695/
        # input:  uvw is an array of Euclidean points on the surface of a unit octahedron.
        # output: UVW on a unit sphere.
        t_uvw = np.tan((np.pi * uvw + cls._e) * 0.5)
        xu, xv, xw = t_uvw[..., 0], t_uvw[..., 1], t_uvw[..., 2]
        u2, v2, w2 = xu ** 2., xv ** 2., xw ** 2.
        y0p = xu * (v2 + w2 + cls._ALPHA * w2 * v2) ** 0.25
        y1p = xv * (u2 + w2 + cls._ALPHA * u2 * w2) ** 0.25
        y2p = xw * (u2 + v2 + cls._ALPHA * u2 * v2) ** 0.25
        pv = np.stack([y0p, y1p, y2p], axis=-1)
        np.seterr(invalid='ignore')
        _rx = pv / np.linalg.norm(pv, axis=-1, keepdims=True)
        return _rx

    @classmethod
    def so(cls, tsp):
        # Projected a spherical point onto the octahedron
        # This inverse function using numerical optimization
        # input:  uvw is an array of Euclidean points on the surface of a unit sphere.
        # output: UVW on a unit octahedron.
        if not cls._jac_fn:
            cls._set_jac()

        def wrapped_jac(x, _):
            return cls._jac_fn(*x)

        def find_root(uvw):
            result = root(
                cls._root_fn,
                np.sign(uvw) * 1. / 3.,  # initial_guess,
                args=(uvw,),
                jac=wrapped_jac,
                method='hybr', tol=1e-12
            )
            result.x /= np.linalg.norm(result.x, ord=1)
            return result.x

        return np.apply_along_axis(find_root, -1, tsp)

    @classmethod
    def _root_fn(cls, op, tx):  # octa_point, target_sphere_point
        norm = np.linalg.norm(op, ord=1)
        val = cls.os(np.array([op / norm])) - np.array(tx)
        return val[0]

    @classmethod
    def _set_jac(cls):
        if cls._jac_fn:
            return
        u, v, w = sp.symbols('u v w')  # Define symbolic variables for inputs
        tan_u = sp.tan(sp.pi * u / 2)
        tan_v = sp.tan(sp.pi * v / 2)
        tan_w = sp.tan(sp.pi * w / 2)

        u2 = tan_u ** 2
        v2 = tan_v ** 2
        w2 = tan_w ** 2

        y0p = tan_u * (v2 + w2 + cls._ALPHA * w2 * v2) ** 0.25
        y1p = tan_v * (u2 + w2 + cls._ALPHA * u2 * w2) ** 0.25
        y2p = tan_w * (u2 + v2 + cls._ALPHA * u2 * v2) ** 0.25

        # Combine outputs into a vector
        y = sp.Matrix([y0p, y1p, y2p])

        # Normalize the vector (divide by its magnitude)
        norm = sp.sqrt(y[0] ** 2 + y[1] ** 2 + y[2] ** 2)
        y_normalized = y / norm

        variables = [u, v, w]
        jacobian = y_normalized.jacobian(variables)
        cls._jac_fn = sp.lambdify(sp.Matrix(variables), jacobian, modules=['numpy'])
