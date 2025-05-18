"""
Part of the H9 project
"""
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root
from hhg9 import (Registrar, Points, Projection)
from hhg9.domains import (SphericalGCD,
    SphericalCartesian, OctahedralCartesian, OctahedralBarycentric,
    OctahedralNet)
from hhg9.projections import AKOctahedralSpherical, CartesianGCD
from hhg9.formats import DMS, DecimalDegrees

from support import Util, Display

# c_oct.register_format(DecimalDegrees())
class CNOctahedralSpherical(Projection):
    """
        An Octahedral/Spherical Projection generated via conformal
    """

    def __init__(self, registrar):
        super().__init__(registrar, 'cn', 'octahedral', 'csp')

    @classmethod
    def sigmoid(cls, x):
        """NN sigmoid"""
        return 1 / (1 + np.exp(-x))

    def forward(self, uv: Points) -> NDArray:
        """
        # Convert a NDArray of barycentric points projected onto a sphere
        # :param uv:  An array of Euclidean points on the surface of a unit octahedron.
        # :return: UVW on a unit sphere.
        """

        # pz = ((np.abs(uv) - 1 / 3) / 1.25) + (1, 0, 0)
        pz = np.abs(uv)

        o0, d0, b0, o1, d1, b1, o2, d2, b2, o3, d3, b3 = (
            -1.86547329, -2.38543150, 2.05896051,
            264.53837471, 270.88261948, 401.9071098,
            35.66201549, 46.11240162, -23.59702692,
            -21.17021298, 142.57135676, -20.84693306
        )
        w0 = np.array([[o0, o0, d0], [o0, d0, o0], [d0, o0, o0]])
        z0 = self.sigmoid(np.dot(pz, w0) + [b0, b0, b0])

        w1 = np.array([[d1, -o1, o1], [o1, -d1, o1], [o1, -o1, d1]])
        z1 = np.tanh(np.dot(z0, w1) + [-b1, b1, -b1])

        w2 = np.array([[o2, o2, -d2], [d2, -o2, -o2], [o2, -d2, o2]])
        z2 = self.sigmoid(np.dot(z1, w2) + [b2, b2, b2])

        w3 = np.array([[o3, o3, d3], [o3, d3, o3], [d3, o3, o3]])
        dd = np.dot(z2, w3) + [b3, b3, b3]

        dn = dd / (np.linalg.norm(dd, axis=1, keepdims=True))
        pp = np.abs(dn)
        return np.copysign(pp, uv)  # octant 0!


    def backward(self, tsp: Points) -> NDArray:
        """
         Projected a spherical point onto the octahedron
         This inverse function using numerical optimization
         :param tsp:  An array of Euclidean points on the surface of a unit sphere.
         :return: UVW on a unit octahedron.
        """

        def find_root(uvw):
            """
            :param uvw: Euclidean point on the unit sphere.
            :return: That spherical point on the octahedron
            """
            result = root(self._root_fn, np.sign(uvw) * 1. / 3., args=(uvw,), tol=1e-12)
            result.x /= np.linalg.norm(result.x, ord=1)
            return result.x

        return np.apply_along_axis(find_root, -1, tsp)

    def _root_fn(self, op, tx):  # octa_point, target_sphere_point
        norm = np.linalg.norm(op, ord=1)
        val = self.forward(np.array([op / norm])) - np.array(tx)
        return val[0]


def net_map(reg, u, bar, net):
    """
       Generate a random F,
       let each barycentric adopt it,
       project onto the net map.
    """
    pts = {}
    eff = u.tri_eff(4000)
    for sign in bar.signs:
        # sgn = bar.projs[bar.signs[sign]]  # Barycentric adapter.
        # pts[sign] = sgn.adopt(eff)
        pts[sign] = bar.binning(eff, sign)
    stf = reg.project(pts, [bar, net])
    npts = net.where_valid(stf)
    Display.show_pts_2d(npts, net.glx, net.gly)
    # v2 = reg.project(npts, [net, g_sph])  # net to Octahedral; keep dict if there is one.
    # Display.show_global(v2)


if __name__ == '__main__':
    # Before going TODO too much, know that Octant/Octahedron are *pretty* good in octahedron.py
    # TODO: Add h9 Formatter and global variant (eg NAV3234.122)
    reg = Registrar()  # Manage coordinate sets & projections
    # Coordinate sets
    g_sph = SphericalGCD(reg)  # 3D Spherical (2-value Polar)
    c_sph = SphericalCartesian(reg)  # 3D Spherical (3-value Euclidean)
    c_oct = OctahedralCartesian(reg)  # 3D Octahedral (surface) coordinate set
    b_oct = OctahedralBarycentric(reg, c_oct)  # 2d Flat coordinate set.
    n_oct = OctahedralNet(reg, c_oct, b_oct)  # 2d Flat coordinate set.
    # Projections
    es = CartesianGCD(reg)              # crt<=>gcd 3D to Polar
    ak = AKOctahedralSpherical(reg)     # crt <=> hed 3D
    # cn = CNOctahedralSpherical(reg)   # crt <=> hed 3D  -- This isn't really working correctly, yet!

    reg.register_projection('chain', [g_sph, c_sph, c_oct])  # latlon to octahedral via spherical_euclidean
    reg.register_projection('chain', [c_oct, b_oct, n_oct])  # octahedral to net via barycentric.
    reg.register_projection('chain', [g_sph, c_oct, n_oct])  # latlon to octahedral_net via octahedral

    # Formatters
    g_sph.register_format(DMS())
    c_oct.register_format(DecimalDegrees())
    c_sph.register_format(DecimalDegrees())

    u = Util()
    rng = np.random.default_rng()
    net_map(reg, u, b_oct, n_oct)


    # pts = []
    # for sign in bry.signs:
    #     # sgn = bry.sides[bry.signs[sign]]                # Barycentric adapter.
    #     # eff = u.tri_eff(5000).view(PosArray).cs(sgn)    # Marked as Barycentric.NEA. (didn't work!)
    #     sgn = bry.projs[bry.signs[sign]]                  # Barycentric adapter.
    #     eff = sgn.adopt(u.tri_eff(5000))                  # Adopted into hed.
    #     # ho1 = reg.project(eff, [bry, hed])              # convert NEA to Octahedral
    #     # ho2 = reg.project(eff, [hed, bry])              # Octahedral to Barycentric.
    #     # ho3 = reg.project(ho2, [bry, net])              # Barycentric to Net
    #     # ho3 = reg.project(eff, [bry, net])              # Barycentric to Net DIRECT (works fine).
    #     pts.append(eff)
    # # Display.show_pts_2d(np.vstack(pts), net.glx, net.gly)

    # eff = u.d2_3(u.tri_eff(5000), 1./np.sqrt(3)).view(PosArray).cs(cx.name)
    # ho1 = reg.project(eff, [bry, hed])
    # ntx = reg.project(eff, [ho1, net])

    # populate some random gcd (latitude/longitude) values.
    arr = np.array([[30.26256, 20.08669], [40.26256, -170.08669]]).view(Points).set_domain(g_sph)
    # gcd to octahedral
    xyz = reg.project(arr, [g_sph, c_oct])
    # octahedral to gcd
    bar = reg.project(xyz, [c_oct, g_sph])
    print(f'{xyz:decimal}')
    print(f'{arr:dms}')
    print(f'{bar:dms}')
    bim = reg.project(bar, [g_sph, c_oct, b_oct])
    print(f'{bim}')
