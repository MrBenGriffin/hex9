# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Grid Methods - eg for composing rectilinear pixel grids for projected sampling.
"""
import numpy as np
from hhg9.h9 import H9K
from hhg9.h9.classifier import in_scope


def fit(pts, img_w, img_h):
    """fit/scale pnts to the grid size"""
    x = pts.coords[:, 0]
    y = pts.coords[:, 1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    nx = x_max - x_min
    ny = y_max - y_min
    sx = (img_w - 1) / nx if nx > 0 else 1.0
    sy = (img_h - 1) / ny if ny > 0 else 1.0
    pvx = np.rint((x - x_min) * sx).astype(int)
    pvy = (img_h - 1) - np.rint((y - y_min) * sy).astype(int)
    pvx = np.clip(pvx, 0, img_w - 1)
    pvy = np.clip(pvy, 0, img_h - 1)
    return pvx, pvy


def sq_grid(scale: float = 1000, mode: int = 1):
    """
    Generate a rectilinear grid of points within a square centered at (cx, cy) in pixel space.
    """
    wid = max(1, int(round(scale)))
    hgt = max(1, int(np.ceil(scale * H9K.derived.RH)))
    fl, cl = (H9K.limits.ΛF, H9K.limits.ΛC) if mode == 1 else (H9K.limits.VC, H9K.limits.VF)
    xl = np.linspace(H9K.limits.TL, H9K.limits.TR, num=wid)
    yl = np.linspace(fl, cl,     num=hgt)

    # IMPORTANT: one meshgrid for everything; keep indexing="xy"
    xx, yy = np.meshgrid(xl, yl, indexing="xy")                   # shapes (hgt, wid)
    rec    = np.stack((xx.ravel(), yy.ravel()), axis=1)     # N = hgt*wid

    ii, jj = np.meshgrid(np.arange(wid), np.arange(hgt), indexing="xy")
    px_idx = ii.ravel().astype(np.int32)                        # x (col)
    py_idx = (hgt - 1 - jj.ravel()).astype(np.int32)            # y (row), flipped once

    # Edge-inclusive mask
    dx = (H9K.limits.TR - H9K.limits.TL) / max(wid - 1, 1)
    dy = (cl - fl) / max(hgt - 1, 1)
    pix = 1 - max(abs(dx), abs(dy))
    trx = in_scope(H9K.radical.R3 * rec[:, 0] * pix, rec[:, 1] * pix, mode)
    return wid, hgt, rec, trx, px_idx, py_idx


def sq_grid_vx(scale: float = 1000, mode: int = 0):
    """
    Return a rectilinear grid of points within an equilateral triangle centered
    at (cx, cy) in pixel space conforming to the barycentric projection of the side of a unit octahedron.
    When calling this for a net, remember to use the net's ΛV not the barycentric!

    """
    wid = int(scale)
    if wid < 1:
        wid = 1
    hgt = max(1, int(round(scale * H9K.derived.RH)))

    # generate a covering rectangle (inclusive of endpoints)
    fl, cl = (H9K.limits.ΛF, H9K.limits.ΛC) if mode == 1 else (H9K.limits.VF, H9K.limits.VC)
    yl = np.linspace(fl, cl, num=hgt).astype(np.float64)
    xl = np.linspace(H9K.limits.TL, H9K.limits.TR, num=wid)
    xx, yy = np.meshgrid(xl, yl)
    rec = np.stack((xx.ravel(), yy.ravel()), axis=1, dtype=np.float64)
    mo = np.full(rec.shape[0], mode, dtype=np.uint8)

    # Integer pixel indices for each sample (column x, row y)
    jj, ii = np.meshgrid(np.arange(hgt), np.arange(wid), indexing='ij')
    pix_x = ii.ravel()
    pix_y = (hgt - 1 - jj.ravel())  # flip Y once, globally

    # Robust edge inclusion: evaluate validity on a slightly in-set copy so that
    # points that lie numerically on the sloped edges are treated as inside.
    # Pixel size in coordinate space
    dx = (H9K.limits.TR - H9K.limits.TL) / max(wid - 1, 1)
    dy = (cl - fl) / max(hgt - 1, 1)
    pix = max(abs(dx), abs(dy))
    tx = rec[:, 0] * H9K.radical.R3
    trx = in_scope(tx, rec[:, 1], mo)  # ~half a pixel in coord space
    return wid, hgt, rec, trx, pix_x, pix_y


def _cross2d(ab, ap):
    """in_convex_poly Helper: 2D cross for batched points vs one edge"""
    return ap[..., 0] * ab[1] - ap[..., 1] * ab[0]


def in_convex_poly(points, poly):
    """
    Vectorized check if each point in `points` is inside the convex polygon.

    Parameters:
        points: (n, 2) NumPy array of n points to test.
        poly:   (m, 2) array-like of vertices in CW or CCW order.

    Returns:
        Boolean mask of length n indicating whether each point is inside.
    """
    poly = np.asarray(poly, dtype=float)
    points = np.atleast_2d(points).astype(float)
    npts = points.shape[0]
    m = poly.shape[0]
    if m < 3:
        return np.zeros(npts, dtype=bool)

    # Use polygon centroid to determine the interior side relative to edges.
    # This avoids ambiguity from winding (CW vs CCW) and axis conventions.
    ref = poly.mean(axis=0)
    a0 = poly[0]
    b0 = poly[1 % m]
    ab0 = b0 - a0
    s = np.sign(_cross2d(ab0, ref - a0))
    if s == 0:
        s = 1.0

    inside = np.ones(npts, dtype=bool)
    eps = 1e-12
    for i in range(m):
        a = poly[i]
        b = poly[(i + 1) % m]
        ab = b - a
        cp = _cross2d(ab, points - a)  # (N,)
        # Keep points on the same side as the centroid (within tolerance)
        inside &= (s * cp >= -eps)
        if not inside.any():
            # early exit if all points are already outside
            return inside
    return inside


def qa_grid(quad, scale: float = 1000):
    """
    Return a rectilinear grid of points within a quadrilateral.
    Also returns the mask and scales.

    Notes:
        - Y is sampled from max→min so row 0 corresponds to the top of the quad (image-like).
        - The returned scales are (sx, sy) in pixels-per-unit for X and Y respectively,
          derived from the chosen integer width/height and the quad spans.
    """
    quad = np.asarray(quad, dtype=float)
    minx = float(quad[..., 0].min())
    maxx = float(quad[..., 0].max())
    miny = float(quad[..., 1].min())
    maxy = float(quad[..., 1].max())
    w = maxx - minx
    h = maxy - miny
    if w <= 0 or h <= 0:
        # Degenerate quad; return empty structures
        return 0, 0, np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=bool), (
        np.array([minx, miny, maxx, maxy]), (1.0, 1.0))

    wid = int(scale)
    if wid < 1:
        wid = 1
    hgt = max(1, int(round((h / w) * wid)))

    yl = np.linspace(maxy, miny, num=hgt)
    xl = np.linspace(minx, maxx, num=wid)
    xx, yy = np.meshgrid(xl, yl)
    rec = np.stack((xx.ravel(), yy.ravel()), axis=1)
    trx = in_convex_poly(rec, quad)

    # Pixel-per-unit scales for X and Y
    sx = (wid - 1) / w if wid > 1 else 1.0
    sy = (hgt - 1) / h if hgt > 1 else 1.0
    return wid, hgt, rec, trx, (np.array([minx, miny, maxx, maxy]), (sx, sy))
