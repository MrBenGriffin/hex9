# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
H9 Region Management.

This module defines the 12 legal layer **Regions** and manages layer/interlayer relations.
It handles the transition between the raw barycentric coordinates of a face and its
hierarchical addresses.

**Core Responsibilities:**

* **Traversing:** Converting between Barycentric (x, y) coordinates and hierarchical Region IDs.
* **Clamping:** Ensuring points stay within valid triangle bounds during float operations.
* **Neighborhoods:** Calculating adjacent regions for hex-binning and seamless traversal.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Literal, Optional, Generator
from dataclasses import dataclass

from hhg9.h9.protocols import H9ConstLike, H9ClassifierLike, H9CellLike, H9RegionLike


@dataclass(frozen=True, slots=True)
class H9Region:
    """
    Immutable container for Region Constants and LUTs.

    Attributes:
        invalid_region (int): Marker for invalid regions (e.g., 0x5F, becoming 0x0F).
        proto (NDArray[np.uint8]): Virtual prototypes [Down=0x49, Up=0x16].
        proto_up (int): Virtual Up prototype ID.
        proto_dn (int): Virtual Down prototype ID.
        ids (NDArray[np.uint8]): List of valid region IDs (0-11).
        is_in (NDArray[bool]): Boolean array indicating in-scope status for cells.
        downs (NDArray[np.uint8]): Array of cells belonging to the Down supercell.
        ups (NDArray[np.uint8]): Array of cells belonging to the Up supercell.
        child (NDArray[np.uint8]): Child transitions LUT (shape 12, 3).
        mcc2 (NDArray[np.uint8]): Mapping [super_cell_mode, cell] -> c2.
        cmc2n (NDArray[np.uint8]): Mapping [cell, super_cell_mode, c2] -> neighbour.
        loc_offs (NDArray[np.uint8]): Location offsets [cell_mode, sc_mode, c2, sibling] -> neighbour.
    """
    invalid_region: int
    proto: NDArray[np.uint8]
    proto_up: int
    proto_dn: int
    ids: NDArray[np.uint8]
    is_in: NDArray[bool]
    downs: NDArray[np.uint8]
    ups: NDArray[np.uint8]
    child: NDArray[np.uint8]
    mcc2: NDArray[np.uint8]
    cmc2n: NDArray[np.uint8]
    loc_offs: NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class H9Context:
    """Dependency injection container for lazy loading to avoid cycles."""
    k: H9ConstLike
    cl: H9ClassifierLike
    c: H9CellLike
    r: H9RegionLike


# --- StepEvent dataclasses for per-layer introspection ---
@dataclass(slots=True)
class StepEvent:
    """Event data emitted during hierarchical traversal steps."""
    i: int  #: Current layer index.
    phase: Literal['pre', 'post']  #: Phase of the step (before or after update).
    addresses: np.ndarray  #: Current address buffer.
    pmo: np.ndarray  #: Parent mode array.
    cid: np.ndarray  #: Current cell ID array.
    bad: np.ndarray  #: Boolean mask of invalid/bad points.
    y: np.ndarray  #: Current y coordinates.
    ẋ: np.ndarray  #: Current scaled x coordinates.


@dataclass(slots=True)
class StepEventXY(StepEvent):
    """Specialized StepEvent for XY space."""
    space: Literal['xy'] = 'xy'


def near_ulps(x: NDArray[np.float64], target: float, k: float = 4.0) -> NDArray[bool]:
    """
    Checks if values in x are within `k` ULPs (Units in Last Place) of target.

    Args:
        x: Input array.
        target: Comparison target.
        k: Tolerance multiplier.
    """
    x = np.asarray(x, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    ulp = np.maximum(np.spacing(target), np.finfo(np.float64).eps)
    return np.abs(x - target) <= k * ulp


def _mode_child_c2_builder(cells: np.ndarray, c2: np.ndarray, mode: np.ndarray, bad: int, t_size: int = 96) -> NDArray[
    np.uint8]:
    """
    Builds the per-mode child -> c2 map.

    Returns:
        NDArray[np.uint8]: Shape (2, t_size). For a given mode (0/1) and child cell ID,
        gives the c2 index (0, 1, 2) or 'bad'.
    """
    m_sh, c_sh, k_sh = c2.shape  # gather sizes of modes, c2s, cells.
    mo_c2_of_cell = np.full((m_sh, t_size), bad, dtype=np.uint8)  # generate full lut
    c2_ids_flat = np.repeat(np.arange(c_sh, dtype=np.uint8), k_sh)  # keep c2s tied to corresponding cells
    for m in range(m_sh):
        cells_flat = c2[m].reshape(-1)  # flatten the cells for this mode.
        mo_c2_of_cell[m, cells_flat] = c2_ids_flat  # assign corresponding c2s to those cells.
    return mo_c2_of_cell


def _neighbour_builder(size: int, invalid_region: int) -> NDArray[np.uint8]:
    """
    Constructs the LUT for neighbor identity.

    Returns:
        NDArray[np.uint8]: LUT mapping [region, super_region_mode, c2] -> [neighbour, neighbour_parent_mode].
    """
    lut = np.full((size, 2, 3, 2), invalid_region, dtype=np.uint8)
    _ngh_dict = {
        (0x21, 0): [[0x34, 1], [0x25, 0], [0x16, 1]],  # top left vertex - 2 have another parent.  101
        (0x25, 0): [[0x35, 0], [0x21, 0], [0x26, 0]],  # up mode - all neighbours in same parent.  000
        (0x26, 0): [[0x39, 1], [0x2A, 0], [0x25, 0]],  # mid-edge - 1 neighbour has another parent 100
        (0x2A, 0): [[0x3A, 0], [0x26, 0], [0x2B, 0]],  # up mode - all neighbours in same parent.  000
        (0x2B, 0): [[0x3E, 1], [0x16, 1], [0x2A, 0]],  # top right vertex - 2 have another parent. 110
        (0x35, 0): [[0x25, 0], [0x39, 0], [0x2A, 1]],  # mid-edge - 1 neighbour has another parent 001
        (0x39, 0): [[0x49, 0], [0x35, 0], [0x3A, 0]],  # up mode - all neighbours in same parent.  000
        (0x3A, 0): [[0x2A, 0], [0x25, 1], [0x39, 0]],  # mid-edge - 1 neighbour has another parent 010
        (0x49, 0): [[0x39, 0], [0x34, 1], [0x3E, 1]],  # bottom vertex - 2 have another parent.    011
        (0x16, 1): [[0x26, 1], [0x2B, 0], [0x21, 0]],  # top vertex - 2 have another parent.       100
        (0x25, 1): [[0x35, 1], [0x3A, 0], [0x26, 1]],  # mid-edge - 1 neighbour has another parent 101
        (0x26, 1): [[0x16, 1], [0x2A, 1], [0x25, 1]],  # dn mode - all neighbours in same parent.  111
        (0x2A, 1): [[0x3A, 1], [0x26, 1], [0x35, 0]],  # mid-edge - 1 neighbour has another parent 110
        (0x34, 1): [[0x21, 0], [0x49, 0], [0x35, 1]],  # bot-left vertex - 2 have another parent.  001
        (0x35, 1): [[0x25, 1], [0x39, 1], [0x34, 1]],  # dn mode - all neighbours in same parent.  111
        (0x39, 1): [[0x26, 0], [0x35, 1], [0x3A, 1]],  # mid-edge - 1 neighbour has another parent 011
        (0x3A, 1): [[0x2A, 1], [0x3E, 1], [0x39, 1]],  # dn mode - all neighbours in same parent.  111
        (0x3E, 1): [[0x2B, 0], [0x3A, 1], [0x49, 0]],  # bot-right vertex - 2 have another parent. 010
    }
    for key, c2_neighbours in _ngh_dict.items():
        region_id, mode = key
        lut[region_id, mode] = c2_neighbours
    return lut


def _local_offset_lut_builder() -> NDArray[np.float64]:
    """
    Builds the local/neighbouring offset LUT.

    Given [self_mode, parent_mode, c2, sibling-bool], returns the offset to the neighbour.
    """
    from hhg9.h9.constants import H9K
    _lut = [
        # self mode = 0
        #   pm c2 sib
        (0, 0, 0, 1, (0, 2)),  # mode 0, c2:0
        (0, 0, 1, 1, (1, -1)),  # k
        (0, 0, 2, 1, (-1, -1)),
        (0, 1, 0, 1, (0, 2)),
        (0, 1, 1, 1, (1, -1)),  # k
        (0, 1, 2, 1, (-1, -1)),
        (0, 0, 0, 0, (0, -2)),  # Not sure this is possible.
        (0, 0, 1, 0, (-1, 1)),  # k
        (0, 0, 2, 0, (-1, -1)),
        (0, 1, 0, 0, (0, 2)),
        (0, 1, 1, 0, (1, -1)),  # k
        (0, 1, 2, 0, (-1, -1)),
        # self mode = 1
        (1, 0, 0, 1, (0, -2)),  # k
        (1, 0, 1, 1, (-1, 1)),  # k
        (1, 0, 2, 1, (1, 1)),  # k
        (1, 1, 0, 1, (0, -2)),
        (1, 1, 1, 1, (-1, 1)),
        (1, 1, 2, 1, (1, 1)),
        (1, 0, 0, 0, (0, -2)),
        (1, 0, 1, 0, (-1, 1)),  # k
        (1, 0, 2, 0, (1, 1)),  # k
        (1, 1, 0, 0, (0, 2)),
        (1, 1, 1, 0, (-1, 1)),
        (1, 1, 2, 0, (1, 1)),
    ]
    mx = np.array([H9K.lattice.U * 3, H9K.lattice.V * 3])  # This is a 'big' lattice
    lut = np.zeros((2, 2, 3, 2, 2), dtype=np.float64)
    for (smo, pmo, c2, sib, oxy) in _lut:
        lut[smo, pmo, c2, sib] = oxy * mx
    return lut


def region_constants(h9c: Optional[H9CellLike] = None) -> H9Region:
    """Factory to instantiate the H9Region singleton."""
    if h9c is None:
        from hhg9.h9.lattice import H9C
        h9c = H9C

    bad = h9c.count - 1
    mo_ch_c2_lut = _mode_child_c2_builder(h9c.in_scope, h9c.c2, h9c.mode, bad, h9c.count)
    is_in_scope = h9c.in_dn | h9c.in_up
    proto = np.array([0x49, 0x16], dtype=np.uint8)
    nb_lut = _neighbour_builder(h9c.count, bad)
    nb_offs = _local_offset_lut_builder()

    return H9Region(
        invalid_region=bad,
        proto=proto,
        proto_up=proto[1],
        proto_dn=proto[0],
        ids=np.array(range(12)),
        is_in=is_in_scope,
        downs=h9c.downs,  # array of region in down supercell
        ups=h9c.ups,  # array of region in up supercell
        child=h9c.c2,
        mcc2=mo_ch_c2_lut,
        cmc2n=nb_lut,
        loc_offs=nb_offs
    )


H9R = region_constants()


def default_ctx() -> H9Context:
    """Returns the default context with lazy imports."""
    from hhg9.h9.constants import H9K
    from hhg9.h9.classifier import H9CL
    from hhg9.h9.lattice import H9C
    # from hhg9.h9.region     import H9R
    return H9Context(k=H9K, cl=H9CL, c=H9C, r=H9R)


H9CTX = default_ctx()


def ulp_nudge(z: NDArray[np.float64], unit: float) -> NDArray[np.float64]:
    """
    Single-ULP nudge of z toward the nearest integer multiple of `unit`.

    Ensures floating point stability during grid traversal steps.
    Works elementwise and returns a new array.
    """
    target = np.rint(np.divide(z, unit, out=np.zeros_like(z), where=(unit != 0))) * unit
    return np.nextafter(z, np.where(unit != 0, target, z))


def at_vertices(ẋ: NDArray[np.float64], y: NDArray[np.float64], mode: NDArray[int], ulps: float, h9k: H9ConstLike) -> \
NDArray[bool]:
    """
    Return mask for points at the barycentric supercell vertices.

    There are six vertices total, with three active for any given mode (Up/Down).
    """
    ẋv = h9k.limits.TR * h9k.radical.R3
    result = np.zeros_like(mode, dtype=bool)
    pt_dns = (mode == 0)
    pt_ups = (mode == 1)
    central = near_ulps(ẋ, 0, k=ulps)
    result[pt_ups] = (
            (near_ulps(y[pt_ups], h9k.limits.ΛC, k=ulps) & central[pt_ups]) |
            (near_ulps(y[pt_ups], h9k.limits.ΛF, k=ulps) & near_ulps(np.abs(ẋ[pt_ups]), ẋv, k=ulps))
    )
    result[pt_dns] = (
            (near_ulps(y[pt_dns], h9k.limits.VF, k=ulps) & central[pt_dns]) |
            (near_ulps(y[pt_dns], h9k.limits.VC, k=ulps) & near_ulps(np.abs(ẋ[pt_dns]), ẋv, k=ulps))
    )
    return result


def hard_clamp(ẋ: NDArray[np.float64], y: NDArray[np.float64], mode: NDArray[int], ulps: float, h9k: H9ConstLike):
    """
    Strictly clamps points to be within their respective barycentric triangles.

    Used as a fallback when points drift out of bounds during recursive steps.
    """
    ẋ_final = ẋ.copy()
    y_final = y.copy()

    # --- Calculate the Clamped Result for UP Mode Points ---
    up_mask = (mode == 1)
    if np.any(up_mask):
        y_up, ẋ_up = y[up_mask], ẋ[up_mask]

        at_apex = near_ulps(y_up, h9k.limits.ΛC, k=ulps)
        at_base = near_ulps(y_up, h9k.limits.ΛF, k=ulps)
        if np.any(at_base | at_apex):
            y_up_clamped = np.clip(y_up, h9k.limits.ΛF, h9k.limits.ΛC)

            max_abs_ẋ = h9k.limits.ΛC - y_up_clamped
            max_abs_ẋ = np.where(at_apex, 0.0, max_abs_ẋ)

            ẋ_clamped = np.clip(ẋ_up, -max_abs_ẋ, max_abs_ẋ)

            ẋ_final[up_mask] = ẋ_clamped
            y_final[up_mask] = np.where(at_base, h9k.limits.ΛF, y_up_clamped)

    # --- 2. Calculate the Clamped Result for DOWN Mode Points ---
    down_mask = (mode == 0)
    if np.any(down_mask):
        y_down, ẋ_down = y[down_mask], ẋ[down_mask]

        at_apex = near_ulps(y_down, h9k.limits.VF, k=ulps)
        at_base = near_ulps(y_down, h9k.limits.VC, k=ulps)
        if np.any(at_base | at_apex):
            y_down_clamped = np.clip(y_down, h9k.limits.VF, h9k.limits.VC)

            max_abs_ẋ = y_down_clamped - h9k.limits.VF
            max_abs_ẋ = np.where(at_apex, 0.0, max_abs_ẋ)
            ẋ_clamped = np.clip(ẋ_down, -max_abs_ẋ, max_abs_ẋ)

            ẋ_final[down_mask] = ẋ_clamped
            y_final[down_mask] = np.where(at_base, h9k.limits.VC, y_down_clamped)

    return ẋ_final, y_final


def soft_clamp(ẋ: NDArray[np.float64], y: NDArray[np.float64], mode: NDArray[int], ulps: float, h9k: H9ConstLike):
    """
    Clamps points with minimal modifications.

    Only snaps points that are BOTH near a boundary AND out-of-bounds.
    Valid points are never modified.
    """
    ẋ_final = ẋ.copy()
    y_final = y.copy()

    # --- Process UP Mode Points ---
    up_mask = (mode == 1)
    if np.any(up_mask):
        up_indices = np.flatnonzero(up_mask)
        y_up, ẋ_up = y[up_indices], ẋ[up_indices]

        is_near_apex = near_ulps(y_up, h9k.limits.ΛC, k=ulps)
        is_near_base = near_ulps(y_up, h9k.limits.ΛF, k=ulps)
        is_in_middle = ~is_near_apex & ~is_near_base

        # --- Apex Group (UP) ---
        oob_apex_mask = is_near_apex & (y_up > h9k.limits.ΛC)
        if np.any(oob_apex_mask):
            indices_to_fix = up_indices[oob_apex_mask]
            y_final[indices_to_fix] = h9k.limits.ΛC
            ẋ_final[indices_to_fix] = 0.0

        # --- Base Group (UP) ---
        oob_base_mask = is_near_base & (y_up < h9k.limits.ΛF)
        if np.any(oob_base_mask):
            indices_to_fix = up_indices[oob_base_mask]
            y_final[indices_to_fix] = h9k.limits.ΛF
            max_ẋ_at_base = h9k.limits.ΛC - h9k.limits.ΛF
            ẋ_final[indices_to_fix] = np.clip(ẋ[indices_to_fix], -max_ẋ_at_base, max_ẋ_at_base)

        # --- Middle Group (UP) ---
        middle_indices = up_indices[is_in_middle]
        if middle_indices.size > 0:
            y_mid, ẋ_mid = y[middle_indices], ẋ[middle_indices]
            max_abs_ẋ = h9k.limits.ΛC - y_mid
            to_correct_right = (ẋ_mid > max_abs_ẋ) & near_ulps(ẋ_mid, max_abs_ẋ, k=ulps)
            if np.any(to_correct_right):
                ẋ_mid[to_correct_right] = max_abs_ẋ[to_correct_right]
            to_correct_left = (ẋ_mid < -max_abs_ẋ) & near_ulps(ẋ_mid, -max_abs_ẋ, k=ulps)
            if np.any(to_correct_left):
                ẋ_mid[to_correct_left] = -max_abs_ẋ[to_correct_left]
            ẋ_final[middle_indices] = ẋ_mid

    # --- Process DOWN Mode Points (Symmetric Logic) ---
    down_mask = (mode == 0)
    if np.any(down_mask):
        down_indices = np.flatnonzero(down_mask)
        y_down, ẋ_down = y[down_indices], ẋ[down_indices]

        is_near_apex = near_ulps(y_down, h9k.limits.VF, k=ulps)
        is_near_base = near_ulps(y_down, h9k.limits.VC, k=ulps)
        is_in_middle = ~is_near_apex & ~is_near_base

        # --- Apex Group (DOWN) ---
        oob_apex_mask = is_near_apex & (y_down < h9k.limits.VF)
        if np.any(oob_apex_mask):
            indices_to_fix = down_indices[oob_apex_mask]
            y_final[indices_to_fix] = h9k.limits.VF
            ẋ_final[indices_to_fix] = 0.0

        # --- Base Group (DOWN) ---
        oob_base_mask = is_near_base & (y_down > h9k.limits.VC)
        if np.any(oob_base_mask):
            indices_to_fix = down_indices[oob_base_mask]
            y_final[indices_to_fix] = h9k.limits.VC
            max_ẋ_at_base = h9k.limits.VC - h9k.limits.VF
            ẋ_final[indices_to_fix] = np.clip(ẋ[indices_to_fix], -max_ẋ_at_base, max_ẋ_at_base)

        # --- Middle Group (DOWN) ---
        middle_indices = down_indices[is_in_middle]
        if middle_indices.size > 0:
            y_mid, ẋ_mid = y[middle_indices], ẋ[middle_indices]
            max_abs_ẋ = y_mid - h9k.limits.VF
            to_correct_right = (ẋ_mid > max_abs_ẋ) & near_ulps(ẋ_mid, max_abs_ẋ, k=ulps)
            if np.any(to_correct_right):
                ẋ_mid[to_correct_right] = max_abs_ẋ[to_correct_right]
            to_correct_left = (ẋ_mid < -max_abs_ẋ) & near_ulps(ẋ_mid, -max_abs_ẋ, k=ulps)
            if np.any(to_correct_left):
                ẋ_mid[to_correct_left] = -max_abs_ẋ[to_correct_left]
            ẋ_final[middle_indices] = ẋ_mid

    return ẋ_final, y_final


def _recover(cid: np.ndarray,
             ẋ: np.ndarray,
             y: np.ndarray,
             p_mo: np.ndarray,
             bad: np.ndarray,
             h9cl: H9ClassifierLike,
             h9k: H9ConstLike,
             ) -> np.ndarray:
    """
    Common 'bad' squashing routine for steppers.

    If points fall off the grid due to floating point drift during traversal,
    this attempts to recover them by nudging them back to valid lattice points.

    Strategies tried in order:
    1. Single-ULP nudge of y toward nearest :math:`kV`.
    2. Single-ULP nudge of ẋ toward nearest :math:`m\\ddot{U}`.
    3. Soft clamps with increasing tolerance (1.0, 2.0, 4.0 ULPs).
    4. Hard clamp (10.0 ULPs).
    """
    from hhg9.h9.classifier import in_scope, classify_mode_cell
    to_fix = np.flatnonzero(bad)

    if to_fix.size == 0:
        return cid

    lat_v = h9k.lattice.V
    lat_ü = h9k.lattice.Ü

    strategies = [
        (lambda ẋ_s, y_s, **_: (ẋ_s, ulp_nudge(y_s, lat_v))),  # 1. Nudge y
        (lambda ẋ_s, y_s, **_: (ulp_nudge(ẋ_s, lat_ü), y_s)),  # 2. Nudge ẋ
        (lambda ẋ_s, y_s, m_s: soft_clamp(ẋ_s, y_s, m_s, 0.5, h9k)),  # Clamp 0.5
        (lambda ẋ_s, y_s, m_s: soft_clamp(ẋ_s, y_s, m_s, 1.0, h9k)),  # Clamp 1.0
        (lambda ẋ_s, y_s, m_s: soft_clamp(ẋ_s, y_s, m_s, 2.0, h9k)),  # Clamp 2.0
        (lambda ẋ_s, y_s, m_s: soft_clamp(ẋ_s, y_s, m_s, 4.0, h9k)),  # Clamp 4.0
        (lambda ẋ_s, y_s, m_s: hard_clamp(ẋ_s, y_s, m_s, 10.0, h9k)),  # Clamp 10.0
    ]

    for fix_int, fixer in enumerate(strategies):
        if to_fix.size == 0:
            break

        m_sub = p_mo[to_fix]
        ẋ_sub = ẋ[to_fix]
        y_sub = y[to_fix]
        ẋ_loc, y_loc = fixer(ẋ_sub, y_sub, m_s=m_sub)
        ok = in_scope(ẋ_loc, y_loc, m_sub, h9cl)
        if np.any(ok):
            ok_idx = to_fix[ok]
            cid[ok_idx] = classify_mode_cell(ẋ_loc[ok], y_loc[ok], m_sub[ok], h9cl)
            ẋ[ok_idx] = ẋ_loc[ok]
            y[ok_idx] = y_loc[ok]
            to_fix = to_fix[~ok]
    return cid


def xy_regions_iter(xy: NDArray[np.float64], mode: NDArray[int] = None, depth: int = 36, ctx: H9Context = None) -> \
Generator[StepEventXY, None, NDArray[np.uint8]]:
    """
    Generator that mirrors `xy_regions` but yields per-layer StepEvents.

    Useful for debugging or visualizing the traversal process step-by-step.

    Yields:
        StepEventXY: Data about the current step (pre/post update).

    Returns:
        NDArray: The final addresses (available via StopIteration value).
    """
    import hhg9.h9.classifier as clf
    ctx = ctx or default_ctx()
    h9k, h9cl, h9c, cr = ctx.k, ctx.cl, ctx.c, ctx.r
    invalid_region = cr.invalid_region
    proto_up, proto_dn = cr.proto_up, cr.proto_dn
    addresses = np.full((xy.shape[0], depth + 2), invalid_region, dtype=np.uint8)
    addresses[:, 0] = np.where(mode == 1, proto_up, proto_dn)
    pid = addresses[:, 0]
    offs_ẋ = h9c.off_ẋy[:, 0].astype(np.float64, copy=False)
    offs_y = h9c.off_ẋy[:, 1].astype(np.float64, copy=False)
    x = np.array(xy[:, 0], copy=True)
    y = np.array(xy[:, 1], copy=True)
    ẋ = x * h9k.radical.R3

    p_mo = h9c.mode[pid]
    bad = ~clf.in_scope(ẋ, y, p_mo, h9cl)

    if np.any(bad):
        # Pre-amble adjustment: clamp/snap only rows that are out-of-scope
        ẋ[bad] = ulp_nudge(ẋ[bad], h9k.lattice.Ü)
        y[bad] = ulp_nudge(y[bad], h9k.lattice.V)

    for i in range(depth + 1):
        cid = clf.classify_mode_cell(ẋ, y, p_mo, h9cl)
        bad = ~h9c.in_mode[p_mo, cid]
        yield StepEventXY(i=i, phase='pre', addresses=addresses, pmo=p_mo, cid=cid, bad=bad, y=y, ẋ=ẋ)
        cid = _recover(cid, ẋ, y, p_mo, bad, h9cl, h9k)
        addresses[:, i + 1] = cid
        ẋ = (ẋ - offs_ẋ[cid]) * 3.0
        y = (y - offs_y[cid]) * 3.0
        p_mo = h9c.mode[cid]
        yield StepEventXY(i=i, phase='post', addresses=addresses, pmo=p_mo, cid=cid, bad=bad, y=y, ẋ=ẋ)
    return addresses


def xy_regions(xy: NDArray[np.float64], mode: NDArray[int] = None, depth: int = 36, ctx: H9Context = None) -> NDArray[
    np.uint8]:
    """
    Convert barycentric (x, y) vectors into layered region IDs.

    This is the primary method for converting geometric points into H9 addresses.

    Args:
        xy: Array of (x, y) coordinates.
        mode: Array of initial modes (0 or 1).
        depth: Recursion depth. This is probably out; hex depth is 2 when this is 1 (But returns 3 values).  Which isn't correct.

    Returns:
        NDArray[np.uint8]: Array of region IDs for each layer.
    """
    ctx = ctx or default_ctx()
    it = xy_regions_iter(xy, mode=mode, depth=depth, ctx=ctx)
    addresses = None
    for ev in it:
        addresses = ev.addresses
    return addresses


def regions_xy(uri_address: NDArray[np.uint8], ctx: H9Context = None) -> NDArray[np.float64]:
    """
    REVERSE: Convert Region IDs back into (x, y) coordinates.

    Reconstructs coordinates in :math:`(\\dot{x}, y)` space using ``off_ẋy``,
    then converts back to standard (x, y).

    Args:
        uri_address: Array of Region IDs (shape: num_points, depth).

    Returns:
        NDArray[np.float64]: Stack of [x, y, initial_mode].
    """
    ctx = ctx or default_ctx()
    h9k, h9cl, h9c, cr = ctx.k, ctx.cl, ctx.c, ctx.r
    num_points, depth = uri_address.shape

    offs_ẋ = h9c.off_ẋy[:, 0].astype(np.float64, copy=False)
    offs_y = h9c.off_ẋy[:, 1].astype(np.float64, copy=False)

    ẋ = np.zeros(num_points, dtype=np.float64)
    y = np.zeros(num_points, dtype=np.float64)

    # Walk from the last *real* layer down to 1
    for i in range(depth - 1, 0, -1):
        cid = uri_address[:, i]
        valid = (cid != cr.invalid_region)
        if not np.any(valid):
            continue
        ẋ[valid] = ẋ[valid] / 3.0 + offs_ẋ[cid[valid]]
        y[valid] = y[valid] / 3.0 + offs_y[cid[valid]]

    x = ẋ * (h9k.U / h9k.Ü)
    initial_mode = (uri_address[:, 0] == cr.proto_up).astype(np.uint8)
    return np.stack([x, y, initial_mode], axis=-1)


def region_neighbours(addresses: NDArray[np.uint8], ctx: H9Context = None):
    """
    Calculates the neighboring regions for a list of addresses.

    Used primarily for hex-binning to find adjacent cells.

    Logic:
    * **P (-3):** Ante-penultimate region (Parent).
    * **I (-2):** Penultimate region (Point of Interest).
    * **C (-1):** Terminal region (Determines C2).

    Returns:
        tuple: (neighbor_address_array, c2_indices)
    """
    ctx = ctx or default_ctx()
    h9k, h9cl, h9c, h9r = ctx.k, ctx.cl, ctx.c, ctx.r

    count, layers = addresses.shape
    nb_array = addresses.copy()
    xl = layers

    # Identify P, I and their modes
    par_index = -3 if xl > 2 else -2
    p_region = addresses[:, par_index]  # P
    cur = addresses[:, -2]  # I
    imo = h9c.mode[cur]  # I.m

    # Base C2 lookup from terminal C
    term = addresses[:, -1]  # C
    c2 = h9r.mcc2[imo, term]  # may be invalid_region
    bad_val = h9r.invalid_region
    bad = (c2 == bad_val)

    if np.any(bad):
        # Geometric fallback: choose C2 group closest to C.
        off_xy = h9c.off_xy
        for i in np.flatnonzero(bad):
            mode_i = imo[i]
            t = term[i]
            if t == bad_val:
                c2[i] = 0
                continue
            xy_t = off_xy[t]
            cells_mode = h9c.c2[mode_i]
            cand_ids = cells_mode.reshape(-1)
            d2 = np.sum((off_xy[cand_ids] - xy_t) ** 2, axis=1)
            best_flat = int(np.argmin(d2))
            k = cells_mode.shape[1]
            best_c2 = best_flat // k
            c2[i] = np.uint8(best_c2)

    pmo = h9c.mode[p_region]
    nb_m = h9r.cmc2n[cur, pmo, c2]  # neighbour + its parent_mode
    nbr = nb_m[:, 0]
    pmn = nb_m[:, 1]

    nbm = h9c.mode[nbr]
    trm = h9r.child[nbm, c2, 2]

    # Update terminal and I
    nb_array[:, -1] = trm
    nb_array[:, -2] = nbr

    # Cascade up the hierarchy where parent mode changes
    cascading = (pmn != pmo)
    for poi in range(layers - 3, -1, -1):
        if not np.any(cascading):
            break
        active = np.where(cascading)[0]
        c2a = c2[active]
        cur = addresses[active, poi]
        par = addresses[active, poi - 1]
        pmo = h9c.mode[par]
        nb_region, nb_parent_mode = h9r.cmc2n[cur, pmo, c2a].T
        nb_array[active, poi] = nb_region
        hop = nb_parent_mode != pmo
        cascading[active] = hop

    # Normalise root proto
    nmo = h9c.mode[nb_array[:, 0]]
    nb_array[:, 0] = h9r.proto[nmo]
    return nb_array, c2
