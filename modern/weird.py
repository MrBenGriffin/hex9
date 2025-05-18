'''Tough stuff to get a conformal projection via elliptical functions'''
import numpy as np
import scipy.special as sp
from scipy.optimize import fsolve

# Define vertices in (x, y, z) format
vertices_nea = np.array([
    [0, 0, 1],  # North Pole
    [0, 1, 0],  # Equator-East
    [1, 0, 0],  # Equator-Anterior
])


def stereographic_projection(x, y, z):
    """Map (x, y, z) sphere coordinates to complex plane via stereographic projection."""
    if np.isclose(z, 1):  # Special case: North Pole
        return np.inf  # Convention: maps to infinity
    return ((x + 1j) * y) / (1 - z)


def mobius_transform(z, a, b, c, d):
    """Applies a Möbius transformation: w = (a*z + b) / (c*z + d), handling infinity explicitly."""
    if np.isinf(z):  # Explicitly check for infinity
        return a / c if c != 0 else np.inf
    denominator = (c * z + d)
    return (a * z + b) / denominator if denominator != 0 else np.inf


if __name__ == '__main__':
    # Apply stereographic projection to each vertex
    stereo_points = np.array([stereographic_projection(*v) for v in vertices_nea])

    stereo_points = np.array([np.inf, 0 + 1j, 0 + 0j])  # (North Pole, Edge, Edge)
    print("stereo_points", stereo_points)


    def mobius_transform(z, a, b, c, d):
        """Applies a Möbius transformation: w = (a*z + b) / (c*z + d), handling infinity explicitly."""
        if np.isinf(z):  # Explicitly check for infinity
            return a / c if c != 0 else np.inf
        denominator = c * z + d
        # print("(a * z + b)", (a * z + b))
        # print("denominator", denominator)
        return (a * z + b) / denominator if denominator != 0 else np.inf


    # Möbius matrix for correct orientation
    A = np.array([[1, 0], [1, 1]])  # Properly centers the barycentric mapping
    a, b, c, d = A.flatten()
    print("a, b, c, d:", a, b, c, d)


    # def schwarz_christoffel_mapping(z, _a, _b, k):
    #     """ Schwarz-Christoffel transformation using elliptic functions """
    #     # A, B, k = params
    #     bz = _b * z
    #     vx = sp.ellipj(bz, k)
    #     return _a * vx[0]  # Jacobi sn function


    def check_equilateral(points):
        """Checks if the given complex points form an equilateral triangle."""
        d1 = np.abs(points[1] - points[0])  # Distance between p1 and p2
        d2 = np.abs(points[2] - points[1])  # Distance between p2 and p3
        d3 = np.abs(points[0] - points[2])  # Distance between p3 and p1
        print(f"Side lengths: {d1:.6f}, {d2:.6f}, {d3:.6f}")
        return np.allclose([d1, d2, d3], d1)  # Check if all are equal


    print(f"\n\nStereo Points: {stereo_points}")
    print(f"Möbius a,b,c,d: {a, b, c, d}")
    mapped_points = np.array([mobius_transform(z, a, b, c, d) for z in stereo_points])
    expected_triangle = np.array([1 + 0j, -0.5 + 0.866j, -0.5 - 0.866j])
    print(f"Expected: {expected_triangle}")
    print(f"Actual (after Möbius Transformation): {mapped_points}")

    good = check_equilateral(mapped_points)

    # Transformed triangle points (complex numbers)
    mapped_points = np.array([1 + 0j, -0.5 + 0.866j, -0.5 - 0.866j])
    # Convert to 3D (z=0)
    P = np.array([[p.real, p.imag, 0] for p in mapped_points])

    # Compute edge vectors
    v1 = P[1] - P[0]
    v2 = P[2] - P[0]

    # Compute cross product
    normal = np.cross(v1, v2)

    # Normalize the normal
    normal /= np.linalg.norm(normal)

    print("Computed Normal:", normal)
