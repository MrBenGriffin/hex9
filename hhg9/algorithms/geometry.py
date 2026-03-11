# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Useful Geometry functions
"""
import numpy as np


def inside_convex_polygon_cw(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Vectorised point-in-convex-polygon test.
    pts: (N,2), poly: (M,2) in CW order. Returns bool mask."""
    mask = np.ones(len(pts), dtype=bool)
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ab = b - a
        ap = pts - a
        mask &= (ab[0] * ap[:, 1] - ab[1] * ap[:, 0]) <= 0
    return mask


# def inside_triangle_cw(pts: np.ndarray, tri: np.ndarray) -> np.ndarray:
#     """Vectorised point-in-triangle test.
#     points: (hex_layer,2), tri: (3,2) in CW order. Returns mask bool."""
#     # Edges
#     a, b, c = tri[0], tri[1], tri[2]
#     ab = b - a
#     bc = c - b
#     ca = a - c
#     ap = pts - a
#     bp = pts - b
#     cp = pts - c
#     cross1 = ab[0] * ap[:, 1] - ab[1] * ap[:, 0]
#     cross2 = bc[0] * bp[:, 1] - bc[1] * bp[:, 0]
#     cross3 = ca[0] * cp[:, 1] - ca[1] * cp[:, 0]
#     return (cross1 <= 0) & (cross2 <= 0) & (cross3 <= 0)


def ortho_basis_from_normal(n):
    """Return two unit vectors (u, v) that form an orthonormal basis of the
    tangent plane given a unit normal vector n (shape (3,)).
    """
    nx, ny, nz = n
    # pick a vector not parallel to n
    if abs(nx) < 0.5:
        a = np.array([1.0, 0.0, 0.0])
    else:
        a = np.array([0.0, 1.0, 0.0])
    u = np.cross(n, a)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    v /= np.linalg.norm(v)
    return u, v


def ellipsoid_f_grad(p, a, b):
    """Implicit surface function f(x,y,z)=x^2/a^2 + y^2/a^2 + z^2/b^2 - 1
    and its gradient at tri_points (shape (3,)).
    Returns (f, gradF).
    """
    x, y, z = p
    inv_a2 = 1.0 / (a * a)
    inv_b2 = 1.0 / (b * b)
    f = x * x * inv_a2 + y * y * inv_a2 + z * z * inv_b2 - 1.0
    grad = np.array([2.0 * x * inv_a2, 2.0 * y * inv_a2, 2.0 * z * inv_b2])
    return f, grad
