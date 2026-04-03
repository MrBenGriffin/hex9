# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Useful Geometry functions
"""
import numpy as np


def ensure_cw(outline):
    """Ensure that outline is cw."""
    signed_area = 0.0

    for i in range(len(outline)):
        x1, y1 = outline[i]
        x2, y2 = outline[(i + 1) % len(outline)]

        # The Shoelace edge formula
        signed_area += (x2 - x1) * (y2 + y1)

    # signed_area > 0 means it is Clockwise (CW).
    # signed_area < 0 means it is Counter-Clockwise (CCW).
    # In a Y-down coordinate space (like SVGs) the inequality would be reversed.
    if signed_area < 0:
        # It's CCW, so reverse the list to make it CW
        return outline[::-1]

    # It's already CW, return as-is
    return outline


def points_in_or_on_poly(pts, poly, tol=1e-5):
    """
    Return if a points is inside or on a polygon.
    Points should be prefiltered with a bounding box if there are many of them!
    """
    x = pts[:, 0]
    y = pts[:, 1]

    inside = np.zeros(len(pts), dtype=bool)
    on_boundary = np.zeros(len(pts), dtype=bool)
    n = len(poly)

    for i in range(n):
        p1x, p1y = poly[i]
        p2x, p2y = poly[(i + 1) % n]

        dx = p2x - p1x
        dy = p2y - p1y
        length_sq = dx ** 2 + dy ** 2

        # -----------------------------------------------------
        # 1. BOUNDARY CHECK (Distance to line segment)
        # -----------------------------------------------------
        if length_sq > 0:
            # Find the projection scalar 't' of the point onto the infinite line
            t = ((x - p1x) * dx + (y - p1y) * dy) / length_sq

            # Clamp 't' between 0 and 1 so we only check the physical segment
            t_clamped = np.clip(t, 0.0, 1.0)

            # Find the closest physical point on the segment
            proj_x = p1x + t_clamped * dx
            proj_y = p1y + t_clamped * dy

            # Calculate distance squared and flag points within tolerance
            dist_sq = (x - proj_x) ** 2 + (y - proj_y) ** 2
            on_boundary |= (dist_sq <= tol ** 2)

        # -----------------------------------------------------
        # 2. INSIDE CHECK (Standard Ray Casting)
        # -----------------------------------------------------
        y_cond = (p1y > y) != (p2y > y)
        if dy != 0:
            x_intersect = dx * (y - p1y) / dy + p1x
            inside ^= y_cond & (x < x_intersect)

    # The point is valid if it is strictly inside OR sitting on the boundary
    return inside | on_boundary


def inside_convex_polygon_cw(pts: np.ndarray, poly: np.ndarray,
                             tol: float = 0.0) -> np.ndarray:
    """Vectorised point-in-convex-polygon test.
    pts:  (N, 2), poly: (M, 2) in CW order. Returns bool mask.
    tol:  Tolerance in distance units.  When > 0, each edge is expanded
          outward by *tol*, so boundary pixels are claimed by both adjacent
          polygons, eliminating 1-pixel tears at shared edges.  The
          cross-product threshold per edge becomes ``tol * |ab|``.
    """
    mask = np.ones(len(pts), dtype=bool)
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ab = b - a
        ap = pts - a
        cross = ab[0] * ap[:, 1] - ab[1] * ap[:, 0]
        threshold = tol * np.sqrt(ab[0] ** 2 + ab[1] ** 2) if tol > 0.0 else 0.0
        mask &= cross <= threshold
    return mask


def points_in_triangles(pts, triangles):
    """Find points within a mesh with a normalized boundary artefacts."""
    # A static epsilon is perfectly safe for coordinates bound by ~1.0
    eps = 1e-13

    # 1. Bounding Box Filter
    min_val = np.min(triangles, axis=(0, 1))
    max_val = np.max(triangles, axis=(0, 1))

    bbox_mask = np.all((pts >= min_val) & (pts <= max_val), axis=1)
    surviving_pts = pts[bbox_mask]

    if len(surviving_pts) == 0:
        return np.zeros(len(pts), dtype=bool)

    inside_any = np.zeros(len(surviving_pts), dtype=bool)

    # 2. Vectorized Triangle Tests
    for tri in triangles:
        ta, tb, tc = tri[0], tri[1], tri[2]
        v0, v1, v2 = tb - ta, tc - tb, ta - tc
        p0, p1, p2 = surviving_pts - ta, surviving_pts - tb, surviving_pts - tc

        cross0 = v0[0] * p0[:, 1] - v0[1] * p0[:, 0]
        cross1 = v1[0] * p1[:, 1] - v1[1] * p1[:, 0]
        cross2 = v2[0] * p2[:, 1] - v2[1] * p2[:, 0]

        # Using a static epsilon for the boundary check
        has_neg = (cross0 < -eps) | (cross1 < -eps) | (cross2 < -eps)
        has_pos = (cross0 > eps) | (cross1 > eps) | (cross2 > eps)

        inside_this_tri = ~(has_neg & has_pos)
        inside_any |= inside_this_tri

    # 3. Reconstruct final array
    final_mask = np.zeros(len(pts), dtype=bool)
    final_mask[bbox_mask] = inside_any

    return final_mask


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
