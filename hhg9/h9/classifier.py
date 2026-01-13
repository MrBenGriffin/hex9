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
 * M is the mode (polarity of cell),
 * H is the horizontal band it belongs to.
 * P is the positive slope band it belongs to.
 * layer is the negative slope band it belongs to.
 The origin is fixed to the barycentre of the supercell.

Barycentric Cell LUTs
After classifying a point into horizontal and slope tiers (h_id, p_id, n_id),
we pack these integers into a compact cell ID using a fixed bit layout.
This mapping is bijective; a matching decode[cell_id] → (h_id, p_id, n_id) recovers the tiers.
The encode/decode tables are precomputed once to avoid drift and to centralise the convention.
lattice.py defines the [h,p,d,m] <-> [u,v] luts.

Included here are the core functions concerning the identification of a given barycentric coordinate with
its cell, as defined by the LUTs
"""
from dataclasses import dataclass
import numpy as np
from typing import Tuple
from numpy.typing import NDArray
from .protocols import H9ConstLike, H9ClassifierLike, BaryLoc


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
    min_y, max_y = h9c.mode_1_lim
    return (min_y <= y) & (y <= max_y - np.abs(ẋ))


def in_down(ẋ, y, h9c: H9ClassifierLike = H9CL) -> NDArray[bool]:
    """Is (ẋ, y) inside supercell 0?
    :param ẋ: `ẋ := h9k.R3*(x)` on x co-ordinate.
    :param y: y co_ordinate
    :param h9c: H9Classifier
    :return: boolean (in scope or not)
    """
    min_y, max_y = h9c.mode_0_lim
    return (min_y + np.abs(ẋ) <= y) & (y <= max_y)


def in_scope(ẋ, y, mode=1, h9c: H9ClassifierLike = H9CL) -> NDArray[bool]:
    """Vectorised scope test supporting array-valued mode.
    Parameters broadcast like NumPy: ẋ, y, and mode may be scalars or arrays.
    Returns a boolean array of the broadcast shape.
    """
    up = in_up(ẋ, y, h9c)
    dn = in_down(ẋ, y, h9c)
    mode_arr = np.asarray(mode)
    return np.where(mode_arr == 1, up, dn)


def in_scope_xym(xym, h9c: H9ClassifierLike = H9CL) -> NDArray[bool]:
    """Convenience vectorised scope test for packed barycentric triplets.
    Expects xym[..., 0] = ẋ (√3·x), xym[..., 1] = y, xym[..., 2] = mode {0,1}.
    Returns a boolean mask matching xym[..., 0].shape.
    """
    xym = np.asarray(xym)
    ẋ = xym[..., 0]
    y = xym[..., 1]
    mode = xym[..., 2]
    return in_scope(ẋ, y, mode, h9c)


def location(ẋ, y, mode=0, h9c: H9ClassifierLike = H9CL):
    """
    Classify the location of (ẋ, y) as "internal", "edge", "vertex", or "external"
    with respect to supercell boundaries, using barycentric inclusion.
    Vectorized for scalar or array input, returns array of strings.
    """
    from hhg9.h9 import H9K
    # NOTE: some boundary constants (e.g. VC/ΛF) can be ~0, in which case np.isclose
    # relies almost entirely on `atol`. This needs to be comfortably above float64
    # epsilon-scale noise and consistent with the mesh de-dup tolerance.
    a_eps = 2e-14
    r_eps = 1e-12
    if not isinstance(mode, np.ndarray):
        mode = np.full(ẋ.shape[0], mode, dtype=np.uint8)
    ups = np.flatnonzero(mode)
    dns = np.flatnonzero(~mode)
    ẋu, yu = ẋ[ups], y[ups]
    ẋd, yd = ẋ[dns], y[dns]
    in_d, in_u = in_down(ẋd, yd, h9c), in_up(ẋu, yu, h9c)

    # Solve C2=0 (flat edge)
    on0_d = np.isclose(yd, H9K.limits.VC, rtol=r_eps, atol=a_eps)  # uppermost point is flat
    on0_u = np.isclose(yu, H9K.limits.ΛF, rtol=r_eps, atol=a_eps)  # lowermost point is flat
    # Solve C2=1 (forward edge)
    ẇ = H9K.derived.Ẇ
    on1_d = np.isclose(yd - ẋd, -ẇ, rtol=r_eps, atol=a_eps)  # y-ẋ == -ẇ
    on1_u = np.isclose(yu - ẋu,  ẇ, rtol=r_eps, atol=a_eps)  # y-ẋ == ẇ
    # Solve C2=1 (backward edge)
    on2_d = np.isclose(yd + ẋd, -ẇ, rtol=r_eps, atol=a_eps)  # y+ẋ == -ẇ
    on2_u = np.isclose(yu + ẋu,  ẇ, rtol=r_eps, atol=a_eps)  # y+ẋ == ẇ
    close_d = on0_d.astype(int) + on1_d.astype(int) + on2_d.astype(int)
    close_u = on0_u.astype(int) + on1_u.astype(int) + on2_u.astype(int)

    # If a point is within tolerance of a boundary line, treat it as "inside" for
    # classification purposes. This prevents epsilon-scale drift from turning true
    # edge/vertex points into EXT.
    in_d_eff = in_d | (close_d > 0)
    in_u_eff = in_u | (close_u > 0)

    result = np.full(ẋ.shape, BaryLoc.EXT, dtype=int)
    result[in_d_eff & (close_d == 2)] = BaryLoc.VTX  # Vertex: two or more boundaries within eps
    result[in_d_eff & (close_d == 1)] = BaryLoc.EDG  # Edge: exactly one boundary within eps
    result[in_d_eff & (close_d == 0)] = BaryLoc.INT  # Internal: inside and no boundary within eps
    result[in_u_eff & (close_u == 2)] = BaryLoc.VTX  # Vertex: two or more boundaries within eps
    result[in_u_eff & (close_u == 1)] = BaryLoc.EDG  # Edge: exactly one boundary within eps
    result[in_u_eff & (close_u == 0)] = BaryLoc.INT  # Internal: inside and no boundary within eps
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
    return the cell id (uint8 in 0x00–0x5F), given (ẋ, y, mode) and constants
    The difference is miniscule but might affect some rare edge cases.

    :param ẋ: np.ndarray;  √3-scaled x-coordinates (classifier frame)
    :param y: np.ndarray;  y-coordinates in classifier frame
    :param m: np.ndarray;  mode of supercell
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
    # Horizontal tiers (C2 := 0) we need the two >= for up/down
    #                 [h9k.ΛC, h9k.VC, 0.0, h9k.ΛF, h9k.VF]
    h_m0_conditions = [y[m0] > h0, y[m0] > h1, y[m0] > h2, y[m0] > h3, y[m0] >= h4]
    h_m1_conditions = [y[m1] > h0, y[m1] > h1, y[m1] > h2, y[m1] >= h3, y[m1] > h4]
    h_id[m0] = np.select(h_m0_conditions, [0, 1, 2, 3, 4], default=5)
    h_id[m1] = np.select(h_m1_conditions, [0, 1, 2, 3, 4], default=5)

    # Positive slope tiers (C2 := 1) we need the two >= for up/down
    p_conditions = [ymx > p0, ymx > p1, ymx >= p2]
    p_id = np.select(p_conditions, [0, 1, 2], default=3)

    # Negative slope tiers (C2 := 2) we need the two >= for up/down
    n_conditions = [ypx < n0, ypx < n1, ypx <= n2]
    n_id = np.select(n_conditions, [0, 1, 2], default=3)

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
