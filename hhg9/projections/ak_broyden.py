"""
Part of the H9 project
"""
import time

import numpy as np
# import sympy as sp
from numpy.typing import NDArray
from hhg9 import Projection, Points
# from scipy.optimize import root, least_squares
# from joblib import Parallel, delayed
from jax import numpy as jnp, vmap, lax
from jaxopt import Broyden
# from functools import partial


class AKOctahedralSpherical(Projection):
    """
        An Octahedron/Sphere Projection generated via an analytical approximation to a
        force-directed dataset. Approximation designer: Anders Kaseorg
    """

    def __init__(self, registrar):
        super().__init__(registrar, 'ak', 'c_oct', 'c_sph')
        self._ALPHA = 3.227806237143884260376580  # 𝛂 - vis. Kaseorg.
        self._const = np.array(list(self.rev_cs.vertices.values()))
        self._e = 1e-50
        self.tol = 1e-12
        self.batched_fwd_fn = vmap(self._fwd_do)
        self.batched_rev_fn = vmap(self._rev_do)
        # Warmup functions (serve no purpose because batches are done on size).
        # _ = self.forward(np.array([[0., 0., 1.]]))
        # _ = self.backward(np.array([[0., 0., 1.]]))
        initialised = True

    def _axis_aligned(self, v):
        """Check to see if there's any invariants..."""
        axis_vectors = jnp.array([
            [1., 0., 0.],
            [-1., 0., 0.],
            [0., 1., 0.],
            [0., -1., 0.],
            [0., 0., 1.],
            [0., 0., -1.]
        ])
        dists = jnp.linalg.norm(v - axis_vectors, axis=1)
        return jnp.any(dists < self.tol)

    def _fwd_fn(self, uvw):
        t_uvw = jnp.tan((jnp.pi * uvw + self._e) * 0.5)
        xu, xv, xw = t_uvw[0], t_uvw[1], t_uvw[2]
        u2, v2, w2 = xu ** 2, xv ** 2, xw ** 2
        y0p = xu * (v2 + w2 + self._ALPHA * w2 * v2) ** 0.25
        y1p = xv * (u2 + w2 + self._ALPHA * u2 * w2) ** 0.25
        y2p = xw * (u2 + v2 + self._ALPHA * u2 * v2) ** 0.25
        pv = jnp.array([y0p, y1p, y2p])
        return pv / jnp.linalg.norm(pv)

    def _root_fn(self, uvw, guess):
        """Reverse function for a given value."""

        def residual(x):
            """The residual function: What is left as we get close to the answer."""
            return self._fwd_fn(x) - uvw

        solver = Broyden(fun=residual, tol=self.tol)
        result = solver.run(guess)
        return result.params  # This is the uvw that maps to target_vec

    def _rev_do(self, uvw, guess):
        return lax.cond(
            self._axis_aligned(uvw),
            lambda _: uvw,
            lambda _: self._root_fn(uvw, guess),
            operand=None
        )

    def _fwd_do(self, uvw):
        return lax.cond(
            self._axis_aligned(uvw),
            lambda _: uvw,
            lambda _: self._fwd_fn(uvw),
            operand=None
        )

    # def _core(self, uvw):
    #     t_uvw = np.tan((np.pi * np.array(uvw) + self._e) * 0.5)
    #     xu, xv, xw = t_uvw[..., 0], t_uvw[..., 1], t_uvw[..., 2]
    #     u2, v2, w2 = xu ** 2., xv ** 2., xw ** 2.
    #     y0p = xu * (v2 + w2 + self._ALPHA * w2 * v2) ** 0.25
    #     y1p = xv * (u2 + w2 + self._ALPHA * u2 * w2) ** 0.25
    #     y2p = xw * (u2 + v2 + self._ALPHA * u2 * v2) ** 0.25
    #     pv = np.stack([y0p, y1p, y2p], axis=-1)
    #     np.seterr(invalid='ignore')
    #     return pv / np.linalg.norm(pv, axis=-1, keepdims=True)

    def forward(self, pts: Points, local: bool = False) -> NDArray:
        """
        Convert a NDArray of octahedral points projected onto a sphere
        Anders Kaseorg: https://math.stackexchange.com/questions/5016695/
        :param pts:  An array of Euclidean points on the surface of a unit octahedron.
        :param local: If True, this is not to be converted to Points.
        :return: UVW on a unit sphere.
        """
        # t_uvw = np.tan((np.pi * uvw + self._e) * 0.5)
        # xu, xv, xw = t_uvw[..., 0], t_uvw[..., 1], t_uvw[..., 2]
        # u2, v2, w2 = xu ** 2., xv ** 2., xw ** 2.
        # y0p = xu * (v2 + w2 + self._ALPHA * w2 * v2) ** 0.25
        # y1p = xv * (u2 + w2 + self._ALPHA * u2 * w2) ** 0.25
        # y2p = xw * (u2 + v2 + self._ALPHA * u2 * v2) ** 0.25
        # pv = np.stack([y0p, y1p, y2p], axis=-1)
        # np.seterr(invalid='ignore')
        # _rx = pv / np.linalg.norm(pv, axis=-1, keepdims=True)

        res = pts.copy()
        uvw = np.array(pts[:, -3:])
        results = self.batched_fwd_fn(uvw)
        res[:, -3:] = results
        return res.view(Points).set_domain(self.fwd_cs)

    # def _find_root(self, uvw, guess):
    #     if np.any(np.isclose(self._const, uvw, rtol=1e-16)):
    #         return uvw
    # 24.287523984909058 with nn. NO jac. tol=1e-16
    # result = root(lambda x: self._core(x) - uvw, guess, tol=1e-16)
    # least_squares: 36 no jac.
    # result = least_squares(lambda x: self._core(x) - uvw, guess, bounds=(0, 1.0), xtol=1e-12)
    # return result.x / np.linalg.norm(result.x, ord=1)

    @classmethod
    def rev_guess_nn(cls, xyz: Points) -> NDArray:
        """
        # :param xyz:  An array of Euclidean points on the surface of a sphere.
        # :return: UVW first guess of point on octahedron
        """
        pi = xyz
        w01, w02, w03, w04, w05, w06, w07, w08, w09, b00, b01, b02 = (
            -0.36413011148295343, 0.22358146659332434, 0.3538029088297118,
            -0.3620350206545561, -0.3606362246540251, -0.2214257702409059,
            0.2183981326500946, -0.36086071093734123, 0.35262707975240376,
            -0.7057535968308752, -0.7268977986430195, 0.7615682941562207,
        )
        w11, w12, w13, w14, w15, w16, w17, w18, w19, b10, b11, b12 = (
            0.483728593487979, -0.8394899014332871, 0.4827232281974734,
            -0.8622822597395994, 0.49269330190278804, 0.4857011567731579,
            -0.5215142284274863, -0.538240779892749, 0.9221633944844435,
            0.2506275204116232, 0.30744994939215275, 0.14771423882379717,
        )
        w21, w22, w23, w24, w25, w26, w27, w28, w29, b20, b21, b22 = (
            -1.5061539866142977, 0.7285011686470936, 0.7734282169198589,
            0.7403975405606282, 0.7295381215019191, -1.4742112857906202,
            0.7381024365646078, -1.5011881679421537, 0.7588293149375611,
            0.33854177313354794, 0.3124257420327817, 0.3508013220611576,
        )
        w0 = np.array([[w01, w02, w03], [w04, w05, w06], [w07, w08, w09]])
        z0 = np.tanh(np.dot(pi, w0) + [b00, b01, b02])
        w1 = np.array([[w11, w12, w13], [w14, w15, w16], [w17, w18, w19]])
        z1 = np.tanh(np.dot(z0, w1) + [b10, b11, b12])
        w2 = np.array([[w21, w22, w23], [w24, w25, w26], [w27, w28, w29]])
        z2 = np.dot(z1, w2) + [b20, b21, b22]
        po = z2
        return po / (np.linalg.norm(po, ord=1, axis=1, keepdims=True))

    @classmethod
    def rev_guess(cls, xyz: Points) -> NDArray:
        """
        # :param xyz:  An array of Euclidean points on the surface of a sphere.
        # :return: UVW first guess of point on octahedron
        """
        x0, y0, z0 = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        x1 = np.arcsinh(np.tanh(np.arctanh(x0) * 0.5489857) * (1.155511 - abs(np.tanh((np.arcsin(z0 * y0) * np.cosh(np.tan((np.cos(x0 * 2.161049) * np.cosh(x0)) * 1.1334274))) / 0.2227472) * 0.18363458)))
        y1 = np.tanh(np.tan(np.arcsinh(np.cos(np.cos(0.6649576 - np.arcsin(np.arctan(np.arctan(np.tanh(np.tanh(np.tanh(np.tanh(np.tanh(abs(np.tanh(np.tanh(np.tanh(np.arcsinh(np.tan(np.arcsin(z0))) * np.arctanh(x0))))))))))))))) * np.arcsinh(np.arctanh(y0) / 1.0951235))))
        z1 = np.arcsinh(np.cos(np.arctan(np.cbrt((np.arctanh(y0) * np.arctanh(x0)) * 0.8195135) * np.cos(np.arctanh(np.arcsin(np.arctanh(y0) * np.arctanh(x0))) * 0.65252346))) * (np.tanh(np.arctanh(z0) * -0.56760114) / -0.87257904))
        uvw = np.stack([x1, y1, z1], axis=-1)
        return uvw / (np.linalg.norm(uvw, ord=1, axis=1, keepdims=True))

    def _chunked_rev_fn(self, targets, guesses, chunk_size=1000):
        outputs = []
        for i in range(0, len(targets), chunk_size):
            t_chunk = targets[i:i + chunk_size]
            g_chunk = guesses[i:i + chunk_size]
            out_chunk = self.batched_rev_fn(t_chunk, g_chunk).block_until_ready()
            outputs.append(np.array(out_chunk))  # Or keep in JAX form
        return np.concatenate(outputs, axis=0)

    def backward(self, tsp: Points) -> NDArray:
        """
         Projected a spherical point onto the octahedron
         This inverse function using numerical optimization
         :param tsp:  An array of Euclidean points on the surface of a unit sphere.
         :return: UVW on a unit octahedron.
        """

        xyz = np.array(tsp[:, -3:])
        # sig = xyz.copy()
        # oz0 = np.abs(xyz)  # octant 0!
        gss = self.rev_guess(xyz)
        targets = jnp.array(xyz)  # shape [N, 3], output unit vectors from forward pass
        guesses = jnp.array(gss)  # shape [N, 3], NN guesses or initial estimate
        # a = time.time()
        uvw = self._chunked_rev_fn(targets, guesses)  # shape [N, 3]
        # print(time.time() - a)
        # xyz = np.copysign(uvw, sig)
        res = tsp.copy()
        res[:, -3:] = uvw
        return res.view(Points).set_domain(self.rev_cs)


if __name__ == '__main__':
    from support import Util, Display
    from hhg9 import Registrar
    from hhg9.domains import SphericalCartesian, OctahedralCartesian

    reg = Registrar()
    c_sph = SphericalCartesian(reg)  # Cartesian Spherical (xyz)
    c_oct = OctahedralCartesian(reg)  # Cartesian Octahedron (xyz)
    ak = AKOctahedralSpherical(reg)

    """
    Load a photo, convert to PlatePixel points, 
    show it, then convert back and save.
    """
    d = Display()  # simple support display class
    u = Util()
    o = u.oct_rnd(2000)
    s = ak.forward(o, True)
    d.show_pts_3d(s)
    z = ak.rev_guess(s)
    d.show_pts_3d(o-z)
    w = np.array(ak.backward(s))
    d.show_pts_3d(o-w)
    d.show_pts_3d(w-z)

    # from hhg9 import Registrar
    # from hhg9.domains import SphericalCartesian, OctahedralCartesian
    #
    # reg = Registrar()
    # c_sph = SphericalCartesian(reg)  # Cartesian Spherical (xyz)
    # c_oct = OctahedralCartesian(reg)  # Cartesian Octahedron (xyz)
    # ak = AKOctahedralSpherical(reg)
