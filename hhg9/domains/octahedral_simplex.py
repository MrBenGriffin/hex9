# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
This is 's_oct' barycentric simplex uv
"""

import numpy as np
from numpy.typing import NDArray
from hhg9.base.composite import CompositeDomain, ComponentDomain


class OctantSimplex(ComponentDomain):
    """
    This a 2D side of an Octant.
    Validity should be easy enough since we have the 3 points that define it.
    """

    def __init__(self, registrar, dom, name: str, sign, t_pt):
        super().__init__(registrar, name, dom, None, sign, 2)
        self.tpt = t_pt

    def valid(self, pts: NDArray) -> NDArray:
        """
        Return an array of bools according to the validity criterion
        :param pts: set of UV Simplex points
        """
        uv = np.atleast_2d(np.asarray(pts, dtype=float))
        u, v = uv[..., 0], uv[..., 1]
        return (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0)


class OctahedralSimplex(CompositeDomain):
    """
    Basic octahedral-simplex properties and methods.
    """

    def __init__(self, registrar):
        from hhg9.projections import OctantXYUV
        from hhg9.h9 import H9K

        super().__init__(registrar, 's_oct', 2)
        c_oct = registrar.domain('c_oct')
        b_oct = registrar.domain('b_oct')
        self.oid_mo = b_oct.oid_mo
        self.signs = {}
        self.sides = {}
        self.projs = {}

        k = H9K.limits
        self.tx = np.array([
            [[k.TL, k.VC], [k.TR, k.VC], [0.0, k.VF]],
            [[k.TL, k.ΛF], [k.TR, k.ΛF], [0.0, k.ΛC]]
        ])

        for sign, face in c_oct.signs.items():
            s_sig = f'{self.name}:{face}'
            b_sig = b_oct.sides[face].name
            oid = b_oct.sign_to_id[sign]
            o_mode = b_oct.oid_mo[oid]
            t_pt = self.tx[o_mode]
            self.sides[face] = OctantSimplex(registrar, self, s_sig, sign, t_pt)
            self.projs[face] = OctantXYUV(registrar, face, b_sig, s_sig)
            self.signs[sign] = face
            self.components[sign] = self.sides[face]

    @classmethod
    def clamp(cls, pts, eps=0.0):
        """
        Project coords into the simplex {u>=0, v>=0, u+v<=1} in barycentric space.

        eps > 0 slightly pushes points away from the edges by enforcing
        a,b,c >= eps before renormalisation.
        """
        uv = pts.coords
        u = uv[..., 0]
        v = uv[..., 1]
        w = 1.0 - u - v

        # Stack as (..., 3)
        lam = np.stack([u, v, w], axis=-1)

        # Clamp barycentric components from below
        lam_clipped = np.maximum(lam, eps)

        # Renormalise to sum to 1 (avoid division by 0)
        sums = lam_clipped.sum(axis=-1, keepdims=True)
        # If everything hit eps and sum==0 by some weird NaN or inf case, bail out to uniform
        good = np.isfinite(sums) & (sums > 0)
        sums = np.where(good, sums, 1.0)
        lam_norm = lam_clipped / sums

        # Return (u,v) again
        u_new = lam_norm[..., 0]
        v_new = lam_norm[..., 1]
        pts.coords = np.stack([u_new, v_new], axis=-1)

    @classmethod
    def valid(cls, pts: NDArray) -> NDArray:
        """
        Return an array of bools according to the validity criterion
        :param pts: set of 2d Euclidean points
        """
        uv = np.atleast_2d(np.asarray(pts, dtype=float))
        u, v = uv[..., 0], uv[..., 1]
        return (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0)
