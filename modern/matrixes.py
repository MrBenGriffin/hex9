import numpy as np
from enum import Enum, IntFlag

from modern.util import Util

# Constants
i2 = 1.0 / np.sqrt(2)
i3 = 1.0 / np.sqrt(3)
i6 = 1.0 / np.sqrt(6)
i26 = 2.0 * i6


def compute_rotation_matrix(face_vertices):
    """Computes the rotation matrix that aligns a given octahedron face with the XY-plane."""
    v1, v2, v3 = np.array(face_vertices)  # Extract vertices

    # Compute normal (cross product of two edges)
    normal = np.cross(v1 - v2, v1 - v3)
    normal = normal / np.linalg.norm(normal)  # Normalize
    # Compute first basis vector (edge in the plane)
    u = v1 - v2
    u = u / np.linalg.norm(u)
    # Compute second basis vector (perpendicular to both u and normal)
    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)
    # Construct the rotation matrix
    R = np.vstack([u, v, normal])
    return R


op = {
    'N': 'S', 'E': 'W', 'A': 'P',
    'S': 'N', 'W': 'E', 'P': 'A'
}

ov = {
    'N': (0.0, 0.0, +1.0), 'S': (0.0, 0.0, -1.0),  # NS 0 1 (z is vertical)       +90, -90 lat
    'E': (0.0, +1.0, 0.0), 'W': (0.0, -1.0, 0.0),  # EW 2 3 (y is left to right)  +90, -90 lon
    'A': (+1.0, 0.0, 0.0), 'P': (-1.0, 0.0, 0.0),  # AP 4 5 (x is front to back)  0, 180 lon
}

# winding?
keys = ['NWA', 'NWP', 'NEA', 'NEP', 'SWA', 'SWP', 'SEA', 'SEP']
ud = {
    'NEA': 'V', 'SWP': 'Λ',
    'NEP': 'Λ', 'SWA': 'V',
    'NWA': 'Λ', 'SEP': 'V',
    'NWP': 'V', 'SEA': 'Λ'
}

opf = {'NWA': 'SEP', 'NWP': 'SEA', 'NEA': 'SWP', 'NEP': 'SWA'}


def plane_to_side(r_mat, pts):  # given a side rotate to Z
    return pts @ r_mat.T  # rotate and adjust.


if __name__ == '__main__':
    faces = {}
    for ab in [(0, 1, 2), (0, 2, 1)]:
        print(ab)

        for face in keys:
            faces[face] = np.array([ov[face[i]] for i in ab])

        matrices = {face: compute_rotation_matrix(faces[face]) for face in keys}
        print(matrices.__repr__())

        for m in matrices:
            mtx = matrices[m]
            dt = np.linalg.det(mtx)
            if np.abs(1 - dt) > 1e-15:
                print(f'{m}: Matrix Determinant is incorrect {dt}')
            dp = np.dot(mtx[0], mtx[1])
            if np.abs(dp) > 1e-15:
                print(f"Dot should be close to zero. R[0] • R[1] = {dp}")
            nm = np.cross(mtx[0], mtx[1])
            print(f"{m} normal: {nm}")
            print(np.dot(mtx[0], mtx[1]), np.dot(mtx[0], mtx[2]), np.dot(mtx[1], mtx[2]))
        print('\ntesting opposites')
        for f1, f2 in opf.items():
            n1 = np.cross(matrices[f1][0], matrices[f1][1])
            n2 = np.cross(matrices[f2][0], matrices[f2][1])
            ok = np.abs(np.dot(n1, n2) + 1) <= 1e-15
            if not ok:
                print(f"{f1} vs {f2}: {np.dot(n1, n2):.6f}. Should be -1")  # Should be -1
