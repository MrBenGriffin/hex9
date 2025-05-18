from typing import Sequence
import numpy as np
from numpy.typing import NDArray


class Octahedron:
    """Octahedron covers the basic properties and methods used that are
    governed by a unit octahedron. However, I think it may be supplanted
    by Octahedral (Projection)"""
    r2 = 2 ** 0.5
    r3 = 3 ** 0.5
    r6 = r2 * r3
    i2 = 1.0 / r2
    i3 = 1.0 / r3
    i6 = 1.0 / r6
    i26 = 2. * i6

    def __init__(self):
        i2 = self.i2
        i3 = self.i3
        i6 = self.i6
        i26 = self.i26

        # Vertex dictionary
        self.vertices = {
            'N': (0.0, 0.0, +1.0), 'S': (0.0, 0.0, -1.0),
            'E': (0.0, +1.0, 0.0), 'W': (0.0, -1.0, 0.0),
            'A': (+1.0, 0.0, 0.0), 'P': (-1.0, 0.0, 0.0),
        }

        # These rotate a 3D face to the (Z) plane
        # We will also use the associated keys for the octants.
        self.matrices = {
            'NEA': np.array([[-i2, 0, i2], [i6, -i26, i6], [i3, i3, i3]]),
            'NEP': np.array([[0, -i2, i2], [i26, i6, i6], [-i3, i3, i3]]),
            'NWA': np.array([[0, i2, i2], [-i26, -i6, i6], [i3, -i3, i3]]),
            'NWP': np.array([[i2, 0, i2], [-i6, i26, i6], [-i3, -i3, i3]]),
            'SEA': np.array([[0, -i2, -i2], [-i26, i6, -i6], [i3, i3, -i3]]),
            'SEP': np.array([[i2, 0, -i2], [-i6, -i26, -i6], [-i3, i3, -i3]]),
            'SWA': np.array([[-i2, 0, -i2], [i6, i26, -i6], [i3, -i3, -i3]]),
            'SWP': np.array([[0, i2, -i2], [i26, -i6, -i6], [-i3, -i3, -i3]]),
        }

        self.faces = {}
        # Populate each octant.
        for face, matrix in self.matrices.items():
            octant = Octant(face)
            self.faces[face] = octant
            octant.proj_matrix = matrix
            octant.vertices = [self.vertices[i] for i in face]
            octant.apex = octant.vertices[0]  # needs each name to be Nx or Sx
            octant.signs = np.sum(octant.vertices, axis=0)

        self.pt_signs = {tuple(v.signs): k for k, v in self.faces.items()}

    def _validate_matrices(self):
        for face, octant in self.faces.items():
            mtx = octant.proj_matrix
            dt = np.linalg.det(mtx)
            if np.abs(1 - dt) > 1e-6:
                print(f'{mtx}: Matrix Determinant is incorrect {dt}')
            dp = np.dot(mtx[0], mtx[1])
            if np.abs(dp) > 1e-15:
                print(f"Dot should be close to zero. R[0] • R[1] = {dp}")
        opposites = {
            'NEA': 'SWP', 'NEP': 'SWA',
            'NWA': 'SEP', 'NWP': 'SEA',
            'SEA': 'NWP', 'SEP': 'NWA',
            'SWA': 'NEP', 'SWP': 'NEA'
        }
        for f1, f2 in opposites.items():
            m1 = self.faces[f1].proj_matrix
            m2 = self.faces[f2].proj_matrix
            n1 = np.cross(m1[0], m1[1])
            n2 = np.cross(m2[0], m2[1])
            if not np.abs(np.dot(n1, n2) + 1) <= 1e-12:
                print(f"{f1} vs {f2}: {np.dot(n1, n2):.8f}. Should be -1")  # Should be -1

    def pt_face(self, uvw):
        """
         Which face does this point belong to?
         Calculated by testing the polarity/sign of each dimension.
         This does *not* test that the points are on the surface of the octahedron.
         That is done with pt_valid / pts_valid
        """
        if np.all(uvw):  # not = 0..
            key = np.sign(uvw)
        else:
            dx = np.mean(uvw, keepdims=True) * 1E-100
            key = np.sign(uvw - dx)
        return self.pt_signs[tuple(key.astype(int))]

    def pts_faces(self, uvw_s):
        """ Return face keys from np_array of 3d Octahedron/Spherical points"""
        return np.apply_along_axis(self.pt_face, -1, np.array(uvw_s))

    @classmethod
    def pt_valid(cls, pt: NDArray[np.float64] | Sequence[float]):
        """ Constraint: |u|+|v|+|w|=1 (surface of the unit octahedron)"""
        return np.abs(np.sum(np.abs(pt)) - 1.) < 1e-15

    def pts_valid(self, uvw_s: NDArray[np.float64]):
        """
        Test that all points in an array are ON the octahedron surface.
        :param uvw_s: numpy array of 3d Octahedron/Spherical points
        :return: boolean representing validity.
        """
        return np.all(np.apply_along_axis(self.pt_valid, -1, uvw_s))

    @classmethod
    def where_valid(cls, uvw_s: NDArray[np.float64]):
        """
        Returns those points in an array which are ON the octahedron surface.
        :param uvw_s: numpy array of 3d Octahedron/Spherical points
        :return: boolean representing validity.
        """
        val = np.abs(np.sum(np.abs(uvw_s), axis=-1) - 1.) < 1e-15
        return uvw_s[val]

    def bin_points(self, uvw_s: NDArray[np.float64]):
        """
        Given a set of 3D points, identify those which are valid
        and then bin them according to which side they are on.
        :param uvw_s:
        """
        points = self.where_valid(uvw_s)
        face = self.pts_faces(points)
        fn, fsz = np.unique(face, return_counts=True)
        bins = {k: np.zeros((s, 3)) for k, s in zip(fn, fsz)}
        acc = {k: 0 for k in fn}
        for pt, fc in zip(points, face):
            a = acc[fc]
            acc[fc] += 1
            bins[fc][a] = pt
        return bins


class Octant:
    """ Octant: Each of eight parts into which a space or solid body is divided by
        three planes which intersect (especially at right angles) at a single point.
    """
    def __init__(self, name):
        self.rot_z = -np.pi / 3.  # -120º; As we define NS as apex we need to orient.
        self.name = name
        self.vertices = None
        self.apex = None
        self.proj_matrix = None  # matrix to convert 3d to 2d
        self.signs = None  # Used for Point Identity

    @classmethod
    def _rz(cls, theta):
        ct, st = np.cos(theta), np.sin(theta)
        return np.array([[ct, -st, 0], [st, ct, 0], [0, 0, 1.]])

    @classmethod
    def _d3_2(cls, values):
        return np.delete(values, 2, -1)

    @classmethod
    def _d2_3(cls, values, val):
        return np.insert(values, values.shape[1], val, axis=1)

    def adopt(self, pts: NDArray[np.float64]):
        """
        Coerce points to this octant.
        3D points are coerced by adopting the sign of the plane.
        2D points are turned to 3d and oriented according to the apex (N/S).
        Validity of points are not established here.
        """
        if pts.shape[-1] == 2:  # Seems like a reasonable way of identifying 2D/3D.
            pts3 = self._d2_3(pts, Octahedron.i3)  # These are now in 3D.
            return pts3 @ (self.proj_matrix.T @ self._rz(self.rot_z)).T
        return np.copysign(pts, self.signs)  # opt is corrected.

    def flatten(self, pts: NDArray[np.float64]):
        """
        Flatten points of this octant.
        3D points are flattened on the Z-Plane.
        2D points could be restricted - currently just left as is.
        """
        if pts.shape[-1] == 3:  # Seems like a reasonable way of identifying 2D/3D.
            return self._d3_2(pts @ (self.proj_matrix.T @ self._rz(self.rot_z)))  # These are now in barycentric 2D.
        return pts  # currently just left as is.

    def unflatten(self, pts: NDArray[np.float64]):
        """
        Unflatten points of this octant. (inverse of flatten).
        2D points are un-flattened from the Z-Plane.
        3D points are - currently just left as is.
        """
        if pts.shape[-1] == 2:  # Seems like a reasonable way of identifying 2D/3D.
            pts3 = self._d2_3(pts, Octahedron.i3)  # These are now in 3D.
            return pts3 @ (self.proj_matrix.T @ self._rz(self.rot_z)).T
        return pts  # currently just left as is.


# def frac(a, b, f):  # linear fraction from a->b
#     na, nb = np.array(a), np.array(b)
#     return na + (nb - na) * f  # 0.3333 fraction of a->b


if __name__ == '__main__':
    from modern.util import Util
    from modern.display import Display
    # from modern.cn_projection import CNProjection
    from modern.ak_projection import AKProjection
    o = Octahedron()
    u = Util()
    ak = AKProjection()
    # cn = CNProjection()

    eff = u.tri_eff(10000)
    effs = {f: octant.adopt(eff) for f, octant in o.faces.items()}
    pts = np.vstack(list(effs.values()))
    sph = ak.os(pts)
    # sph = cn.os(pts)

    Display.show_pts_3d(sph, (-1.1, 1.1), (-1.1, 1.1), (-1.1, 1.1))
