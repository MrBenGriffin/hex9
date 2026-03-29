# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
base/h9/classifier.py
H9 barycentric classifier
This module defines the barycentric classifier LUTs, and freezes them.
It also defines the methods used to identify a cell by its coordinate,
and provides inclusion/exclusion methods for supercell boundaries.
By convention classifier uses ẋ to mean √3 · x, where √3 = H9K.R3

Coordinates for the equilateral triangle tiling of the plane can come in different formats.
Here we are only interested in a small area, defined by just enough boundaries to identify
the 12 cells (triangles) that compose the two supercells (each have nine, but six cells are shared).

Therefore, we use a fourfold coordinate scheme,
 * M is the net_mode (polarity of cell),
 * H is the horizontal band it belongs to.
 * P is the positive slope band it belongs to.
 * hex_layer is the negative slope band it belongs to.
 The origin is fixed to the barycentre of the supercell.

Barycentric Cell LUTs
After classifying a point into horizontal and slope tiers (h_id, p_id, n_id),
we pack these integers into a compact cell ID using a fixed bit layout.
This mapping is bijective; a matching decode[cell_id] → (h_id, p_id, n_id) recovers the tiers.
The encode/decode tables are precomputed once to avoid drift and to centralise the convention.
lattice.py defines the [h,tri_points,d,m] <-> [u,v] luts.

Included here are the core functions concerning the identification of a given barycentric coordinate with
its cell, as defined by the LUTs
"""
from dataclasses import dataclass
import numpy as np
from typing import Tuple
from numpy.typing import NDArray
from .protocols import H9ConstLike, H9ClassifierLike, BaryLoc, BaryPlc


@dataclass(frozen=True, slots=True)
class H9Classifier:
    """Packed thresholds used by cell classification; precomputed to avoid drift/rebuilds."""
    h_levels: NDArray[np.float64]  # h_levels: horizontal tiers (y vs constants): [ΛC, VC, 0, ΛF, VF]
    p_levels: NDArray[np.float64]  # p_levels: positive slope (y - ẋ) vs [+Ẇ, 0, -Ẇ]
    n_levels: NDArray[np.float64]  # n_levels: negative slope (y + ẋ) vs [-Ẇ, 0, +Ẇ]
    mode_0_lim: Tuple[float, float]  # (VF barycentric floor, VC barycentric ceiling)
    mode_1_lim: Tuple[float, float]  # (ΛF barycentric floor, ΛC barycentric ceiling)
    mode_lim: NDArray[np.float64]  # both mode_0_lim, mode_1_lim
    encode: NDArray[np.uint8]  # shape (6,4,4)
    decode: NDArray[np.uint8]  # shape (96,3)
    eps: np.floating


def h9_classifier(h9k: H9ConstLike = None) -> H9Classifier:
    """
    H9Classifier factory method.
    :return: H9Classifier
    """
    if h9k is None:
        from hhg9.h9.constants import H9K
        h9k = H9K
    from hhg9.algorithms.id_packing import compose_luts

    ẇ = h9k.Ẇ
    h_levels = np.array([h9k.ΛC, h9k.VC, 0.0, h9k.ΛF, h9k.VF], dtype=np.float64)
    p_levels = np.array([+ẇ, 0.0, -ẇ], dtype=np.float64)
    n_levels = -p_levels  # the +ve/-ve slopes are mirrored along y=0

    h_count = h_levels.shape[0]
    s_count = p_levels.shape[0]  # slope count = p_levels and n_levels.

    # The count of bands across any axis is the number of thresholds + 1
    h_bands = h_count + 1
    s_bands = s_count + 1

    mode_0_lim = h9k.VF, h9k.VC
    mode_1_lim = h9k.ΛF, h9k.ΛC
    mode_lim = np.array([mode_0_lim, mode_1_lim], dtype=np.float64)

    encode, decode, *_ = compose_luts([h_bands, s_bands, s_bands])  # id lut = 96 size.
    eps = np.finfo(np.float64).eps
    return H9Classifier(h_levels, p_levels, n_levels, mode_0_lim, mode_1_lim, mode_lim, encode, decode, eps)


H9CL = h9_classifier()  # singleton for typical use


def in_up(ẋ, y, h9c: H9ClassifierLike = H9CL) -> NDArray[bool]:
    """Is (ẋ, y) inside supercell 1?
    :param ẋ: `ẋ := h9k.R3*(x)` on x co-ordinate.
    :param y: y co-ordinate
    :param h9c: H9Classifier
    :return: boolean (in scope or not)
    """
    # Supercell-1 (up) is bounded by:
    #   y >= ΛF  (flat floor)
    #   y - ẋ <= +Ẇ
    #   y + ẋ <= +Ẇ
    min_y, _ = h9c.mode_1_lim
    w = float(h9c.p_levels[0])  # +Ẇ
    scale = np.maximum(1.0, np.maximum(np.abs(ẋ), np.abs(y)))
    tol = 256 * h9c.eps * scale
    return (y >= (min_y - tol)) & ((y - ẋ) <= (w + tol)) & ((y + ẋ) <= (w + tol))


def in_down(ẋ, y, h9c: H9ClassifierLike = H9CL) -> NDArray[bool]:
    """Is (ẋ, y) inside supercell 0?
    :param ẋ: `ẋ := h9k.R3*(x)` on x co-ordinate.
    :param y: y co_ordinate
    :param h9c: H9Classifier
    :return: boolean (in scope or not)
    """
    # Supercell-0 (down) is bounded by:
    #   y <= VC  (flat ceiling)
    #   y - ẋ >= -Ẇ
    #   y + ẋ >= -Ẇ
    _, max_y = h9c.mode_0_lim
    w = float(h9c.p_levels[0])  # +Ẇ
    scale = np.maximum(1.0, np.maximum(np.abs(ẋ), np.abs(y)))
    tol = 256 * h9c.eps * scale
    return (y <= (max_y + tol)) & ((y - ẋ) >= (-w - tol)) & ((y + ẋ) >= (-w - tol))


def in_scope(ẋ, y, mode=1, h9c: H9ClassifierLike = H9CL) -> NDArray[bool]:
    """Vectorised scope test supporting array-valued net_mode.
    Parameters broadcast like NumPy: ẋ, y, and net_mode may be scalars or arrays.
    Returns a boolean array of the broadcast shape.
    """
    up = in_up(ẋ, y, h9c)
    dn = in_down(ẋ, y, h9c)
    mode_arr = np.asarray(mode)
    return np.where(mode_arr == 1, up, dn)


def in_scope_xym(xym, h9c: H9ClassifierLike = H9CL) -> NDArray[bool]:
    """Convenience vectorised scope test for packed barycentric triplets.
    Expects xym[..., 0] = ẋ (√3·x), xym[..., 1] = y, xym[..., 2] = net_mode {0,1}.
    Returns a boolean mask matching xym[..., 0].shape.
    """
    xym = np.asarray(xym)
    ẋ = xym[..., 0]
    y = xym[..., 1]
    mode = xym[..., 2]
    return in_scope(ẋ, y, mode, h9c)


def location(ẋ, y, mode=0, detailed: bool = False, h9c: H9ClassifierLike = H9CL):
    """
    Classify the location of (ẋ, y) as internal / edge / vertex / external
    with respect to supercell boundaries, using barycentric inclusion.

    Returns:
      - detailed=False: BaryLoc.{EXT, INT, EDG, VTX}
      - detailed=True : BaryPlc.{EXT, INT, EG0, EG1, EG2, VXL, VXR, VXA}
    """

    # Tolerance: scale by float64 ULP at the local magnitude.
    # (Avoid rtol-based isclose; we want tight, geometry-consistent predicates.)
    ulp_k = 128  # ~128 ulps at |x|~1 => ~2.8e-14

    ẋ = np.atleast_1d(np.asarray(ẋ))
    y = np.atleast_1d(np.asarray(y))
    n = y.shape[0]

    mode_arr = np.asarray(mode, dtype=np.uint8)
    if mode_arr.ndim == 0:
        mode_arr = np.full(n, int(mode_arr), dtype=np.uint8)
    else:
        mode_arr = np.atleast_1d(mode_arr).astype(np.uint8, copy=False)

    is_up = (mode_arr == 1)
    ups = np.flatnonzero(is_up)
    dns = np.flatnonzero(~is_up)

    # Partitioned coordinates
    ẋu, yu = ẋ[ups], y[ups]
    ẋd, yd = ẋ[dns], y[dns]

    # Residuals to the three half-space constraints (partition-sized).
    # Using residuals prevents treating points on the *infinite* edge line (but outside the
    # triangle segment) as “in-scope”. A point is in-scope iff all residuals >= -tol.

    # Per-partition tolerances (scaled by local magnitude).
    scale_u = np.maximum(1.0, np.maximum(np.abs(ẋu), np.abs(yu)))
    scale_d = np.maximum(1.0, np.maximum(np.abs(ẋd), np.abs(yd)))
    tol_u = ulp_k * h9c.eps * scale_u
    tol_d = ulp_k * h9c.eps * scale_d

    # Mode-specific limits (synonyms are fine; we use h9c to keep a single source of truth).
    min_u, _ = h9c.mode_1_lim   # (ΛF, ΛC)
    _, max_d = h9c.mode_0_lim   # (VF, VC)

    # Diagonal boundary constant (±Ẇ) in classifier units.
    w = float(h9c.p_levels[0])  # +Ẇ

    # Up triangle constraints:
    #   r0: y >= ΛF
    #   r1: y - ẋ <= +Ẇ
    #   r2: y + ẋ <= +Ẇ
    r0_u = yu - min_u
    r1_u = w - (yu - ẋu)
    r2_u = w - (yu + ẋu)

    # Down triangle constraints:
    #   r0: y <= VC
    #   r1: y - ẋ >= -Ẇ
    #   r2: y + ẋ >= -Ẇ
    r0_d = max_d - yd
    r1_d = (yd - ẋd) + w
    r2_d = (yd + ẋd) + w

    # In-scope within tolerance
    in_u_eff = np.minimum.reduce([r0_u, r1_u, r2_u]) >= -tol_u
    in_d_eff = np.minimum.reduce([r0_d, r1_d, r2_d]) >= -tol_d

    # On-edge within tolerance (only meaningful when also in-scope)
    on0_u = in_u_eff & (np.abs(r0_u) <= tol_u)
    on1_u = in_u_eff & (np.abs(r1_u) <= tol_u)
    on2_u = in_u_eff & (np.abs(r2_u) <= tol_u)

    on0_d = in_d_eff & (np.abs(r0_d) <= tol_d)
    on1_d = in_d_eff & (np.abs(r1_d) <= tol_d)
    on2_d = in_d_eff & (np.abs(r2_d) <= tol_d)

    # close counts (partition-sized)
    close_u = on0_u.astype(int) + on1_u.astype(int) + on2_u.astype(int)
    close_d = on0_d.astype(int) + on1_d.astype(int) + on2_d.astype(int)

    # ---- Lift everything back to full length ----
    in_eff = np.zeros(n, dtype=bool)
    close = np.zeros(n, dtype=int)

    on0 = np.zeros(n, dtype=bool)
    on1 = np.zeros(n, dtype=bool)
    on2 = np.zeros(n, dtype=bool)

    in_eff[ups] = in_u_eff
    in_eff[dns] = in_d_eff

    close[ups] = close_u
    close[dns] = close_d

    on0[ups] = on0_u
    on0[dns] = on0_d
    on1[ups] = on1_u
    on1[dns] = on1_d
    on2[ups] = on2_u
    on2[dns] = on2_d

    # ---- Assign results using full-length masks ----
    if detailed:
        result = np.full(n, BaryPlc.EXT, dtype=int)

        # Internal: inside and not within eps of any boundary
        result[in_eff & (close == 0)] = BaryPlc.INT

        # Edges (override INT where appropriate)
        result[in_eff & on0] = BaryPlc.EG0
        result[in_eff & on1] = BaryPlc.EG1
        result[in_eff & on2] = BaryPlc.EG2

        # Vertices (override edges)
        result[in_eff & on1 & on2] = BaryPlc.VXA  # apex: forward/backward
        result[in_eff & on0 & on1] = BaryPlc.VXL  # left:  flat/forward
        result[in_eff & on0 & on2] = BaryPlc.VXR  # right: flat/backward

    else:
        result = np.full(n, BaryLoc.EXT, dtype=int)
        result[in_eff & (close == 0)] = BaryLoc.INT
        result[in_eff & (close == 1)] = BaryLoc.EDG
        result[in_eff & (close >= 2)] = BaryLoc.VTX

    return result


def classify_cell(ẋ, y, h9c: H9ClassifierLike = H9CL) -> NDArray[np.uint8]:
    """
    return the cell id (uint8 in 0x00–0x5F), given (ẋ, y) and constants

    :param ẋ: np.ndarray;  √3-scaled x-coordinates (classifier frame)
    :param y: np.ndarray;  y-coordinates in classifier frame
    :param h9c: H9GConst; H9 constants bundle (defaults to global H9GC)
    :return: np.ndarray;  Array of cell IDs (nybble: (h_id << 4) | (p_id << 2) | n_id)

    Classifies points into cell IDs based on position in the barycentric
    classifier plane, where ẋ := √3·x; (See ascii art).
    """
    (h0, h1, h2, h3, h4) = h9c.h_levels
    (p0, p1, p2) = h9c.p_levels
    (n0, n1, n2) = h9c.n_levels
    encode = h9c.encode

    y = y.astype(np.longdouble, copy=False)
    ẋ = ẋ.astype(np.longdouble, copy=False)

    # Derived slopes
    ymx = y - ẋ  # forward (positive) slope boundaries.
    ypx = y + ẋ  # backward (negative) slope boundaries.

    # Horizontal tiers (C2 := 0) we need the two >= for up/down
    h_conditions = [y > h0, y > h1, y > h2, y >= h3, y >= h4]
    h_id = np.select(h_conditions, [0, 1, 2, 3, 4], default=5)

    # Positive slope tiers (C2 := 1) we need the two >= for up/down
    p_conditions = [ymx > p0, ymx > p1, ymx >= p2]
    p_id = np.select(p_conditions, [0, 1, 2], default=3)

    # Negative slope tiers (C2 := 2) we need the two >= for up/down
    n_conditions = [ypx < n0, ypx < n1, ypx <= n2]
    n_id = np.select(n_conditions, [0, 1, 2], default=3)

    # Encode the result into the same format. Ranges 0..0x5f (see ascii reference above).
    return encode[h_id, p_id, n_id]  # h_id << 4 | p_id << 2 | n_id


def classify_mode_cell(ẋ, y, m, h9c: H9ClassifierLike = H9CL) -> NDArray[np.uint8]:
    """
    return the cell id (uint8 in 0x00–0x5F), given (ẋ, y, net_mode) and constants
    The difference is miniscule but might affect some rare edge cases.

    :param ẋ: np.ndarray;  √3-scaled x-coordinates (classifier frame)
    :param y: np.ndarray;  y-coordinates in classifier frame
    :param m: np.ndarray;  net_mode of supercell
    :param h9c: H9GConst; H9 constants bundle (defaults to global H9GC)
    :return: np.ndarray;  Array of cell IDs (nybble: (h_id << 4) | (p_id << 2) | n_id)

    Classifies points into cell IDs based on position in the barycentric
    classifier plane, where ẋ := √3·x; (See ascii art).
    """
    (h0, h1, h2, h3, h4) = h9c.h_levels
    (p0, p1, p2) = h9c.p_levels
    (n0, n1, n2) = h9c.n_levels
    encode = h9c.encode

    y = y.astype(np.longdouble, copy=False)
    ẋ = ẋ.astype(np.longdouble, copy=False)

    # Derived slopes
    ymx = y - ẋ  # forward (positive) slope boundaries.
    ypx = y + ẋ  # backward (negative) slope boundaries.

    m1 = (m == 1)
    m0 = (m == 0)

    h_id = np.full(y.shape, 5, dtype=np.uint8)
    # Horizontal tiers (C2 := 0) we need the three >= for up/down
    #                 [h9k.ΛC, h9k.VC, 0.0, h9k.ΛF, h9k.VF]
    h_m0_conditions = [y[m0] > h0, y[m0] >= h1, y[m0] >= h2, y[m0] > h3, y[m0] > h4]
    h_m1_conditions = [y[m1] > h0, y[m1] > h1, y[m1] >= h2, y[m1] > h3, y[m1] > h4]
    h_id[m0] = np.select(h_m0_conditions, [0, 1, 2, 3, 4], default=5)
    h_id[m1] = np.select(h_m1_conditions, [0, 1, 2, 3, 4], default=5)

    p_id = np.full(y.shape, 3, dtype=np.uint8)
    # Positive slope tiers (C2 := 1) we need the two >= for up/down
    p_m0_conditions = [ymx[m0] >= p0, ymx[m0] >= p1, ymx[m0] >= p2]
    p_m1_conditions = [ymx[m1] > p0, ymx[m1] > p1, ymx[m1] > p2]
    p_id[m0] = np.select(p_m0_conditions, [0, 1, 2], default=3)
    p_id[m1] = np.select(p_m1_conditions, [0, 1, 2], default=3)

    # Negative slope tiers (C2 := 2) we need the two >= for up/down
    n_id = np.full(y.shape, 3, dtype=np.uint8)
    n_m0_conditions = [ypx[m0] < n0, ypx[m0] <= n1, ypx[m0] < n2]
    n_m1_conditions = [ypx[m1] <= n0, ypx[m1] <= n1, ypx[m1] <= n2]
    n_id[m0] = np.select(n_m0_conditions, [0, 1, 2], default=3)
    n_id[m1] = np.select(n_m1_conditions, [0, 1, 2], default=3)

    # Encode the result into the same format. Ranges 0..0x5f (see ascii reference above).
    return encode[h_id, p_id, n_id]  # h_id << 4 | p_id << 2 | n_id


def classify_cell_uv(U, V, h9c: H9ClassifierLike = H9CL) -> NDArray[np.uint8]:
    """
    Classify using lattice-domain residuals (U, V), not metric (ẋ, y).
    Inputs:
      U = ẋ * sqrt(6)   (integer for barycentres / decoded points)
      V = y * 3*sqrt(6)
    This removes radicals from thresholds so we compare against exact integers:
      A := V + 3U   ↔ (y + ẋ) vs [-Ẇ, 0, +Ẇ]   → [-6, 0, +6]
      B := V - 3U   ↔ (y - ẋ) vs [+Ẇ, 0, -Ẇ]   → [+6, 0, -6]
      C := V        ↔  y      vs tiers         → [-6, -3, 0, +3, +6]
    Returns:
      uint8 cell IDs via the same encode LUT.
    """
    u = np.asarray(U)
    v = np.asarray(V)
    a = v + 3 * u  # y + ẋ   scaled by 3√6
    b = v - 3 * u  # y - ẋ   scaled by 3√6
    c = v  # y       scaled by 3√6
    # Horizontal tiers (ΛC, VC, 0, ΛF, VF) with the same strict/inclusive sides as before
    h_id = np.select([c > 6, c > 3, c > 0, c >= -3, c >= -6], [0, 1, 2, 3, 4], default=5)
    # Positive-slope family (y-ẋ): [+6, 0, -6]
    p_id = np.select([b > 6, b > 0, b >= -6], [0, 1, 2], default=3)
    # Negative-slope family (y+ẋ): [-6, 0, +6]
    n_id = np.select([a < -6, a < 0, a <= 6], [0, 1, 2], default=3)

    return h9c.encode[h_id, p_id, n_id]
