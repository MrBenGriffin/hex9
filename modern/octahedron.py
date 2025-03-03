import numpy as np
from modern.util import Util


class Octahedron:
    r2 = 2 ** 0.5
    r3 = 3 ** 0.5
    r6 = 6 ** 0.5  # r2 * r3
    i2 = 1.0 / r2
    i3 = 1.0 / r3
    i6 = 1.0 / r6
    i26 = 2. * i6

    ov = {
        'N': (0.0, 0.0, +1.0), 'S': (0.0, 0.0, -1.0),  # NS 0 1 (z is vertical)       +90, -90 lat
        'E': (0.0, +1.0, 0.0), 'W': (0.0, -1.0, 0.0),  # EW 2 3 (y is left to right)  +90, -90 lon
        'A': (+1.0, 0.0, 0.0), 'P': (-1.0, 0.0, 0.0),  # AP 4 5 (x is front to back)  0, 180 lon
    }
    matrices = {
        'NEA': np.array([[-i2, 0, i2], [i6, -i26, i6], [i3, i3, i3]]),
        'NEP': np.array([[0, -i2, i2], [i26, i6, i6], [-i3, i3, i3]]),
        'NWA': np.array([[0, i2, i2], [-i26, -i6, i6], [i3, -i3, i3]]),
        'NWP': np.array([[i2, 0, i2], [-i6, i26, i6], [-i3, -i3, i3]]),
        'SEA': np.array([[0, -i2, -i2], [-i26, i6, -i6], [i3, i3, -i3]]),
        'SEP': np.array([[i2, 0, -i2], [-i6, -i26, -i6], [-i3, i3, -i3]]),
        'SWA': np.array([[-i2, 0, -i2], [i6, i26, -i6], [i3, -i3, -i3]]),
        'SWP': np.array([[0, i2, -i2], [i26, -i6, -i6], [-i3, -i3, -i3]]),
    }

    def __init__(self):
        for m in self.matrices:
            mtx = self.matrices[m]
            dt = np.linalg.det(mtx)
            if np.abs(1 - dt) > 1e-6:
                print(f'{m}: Matrix Determinant is incorrect {dt}')
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
            n1 = np.cross(self.matrices[f1][0], self.matrices[f1][1])
            n2 = np.cross(self.matrices[f2][0], self.matrices[f2][1])
            if not np.abs(np.dot(n1, n2) + 1) <= 1e-12:
                print(f"{f1} vs {f2}: {np.dot(n1, n2):.8f}. Should be -1")  # Should be -1
        self.signs = dict()
        for face in self.matrices:
            self.signs[face] = np.sum([self.ov[v] for v in face], axis=0)
        self.pt_signs = {tuple(v): k for k, v in self.signs.items()}

    def pt_face(self, uvw):  # face.key from 3d Octahedron/Spherical point(s)
        return self.pt_signs[tuple(np.sign(uvw).astype(int).tolist())]

    def pts_faces(self, uvw):  # face.keys from 3d Octahedron/Spherical points
        pts = uvw if isinstance(uvw, np.ndarray) else np.array(uvw)
        return np.apply_along_axis(self.pt_face, -1, pts)


def frac(a, b, f):  # linear fraction from a->b
    na, nb = np.array(a), np.array(b)
    return na + (nb - na) * f  # 0.3333 fraction of a->b


if __name__ == '__main__':
    from modern.ak import AK
    o = Octahedron()
    u = Util()
    ak = AK()
    rf = {}
    for face in o.matrices:
        rf[face] = []
        fx = [o.ov[v] for v in face]
        mx = np.mean(fx, axis=0)
        rf[face].append(u.xyz_ll(ak.os(np.array([frac(f, mx, 0.20) for f in fx]))))
    print(repr(rf))
