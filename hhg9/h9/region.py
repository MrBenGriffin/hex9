"""
Part of the H9 project
Author: Ben
base/h9/region.py

In `region.py` we define 12 legal layer **regions** and single **invalid region*.
We manage the layer/interlayer relations and derived methods.
The transition between barycentric coordinates of a face and it's layers is managed by the cells.py facade.
Right now layer-ids == cell-ids.
"""
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from hhg9.h9.protocols import H9ConstLike, H9ClassifierLike, H9CellLike, H9RegionLike
from typing import Literal
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class H9Region:
    """Region Constants"""
    invalid_region: int          # 0x5f will become 0x0f
    proto: NDArray[np.uint8]     # virtual up (0x16 to new proto)
    proto_up: int                # virtual up (0x16 to new proto)
    proto_dn: int                # virtual dn (0x49 to new proto)
    ids: NDArray[np.uint8]       # region_ids
    is_in: NDArray[bool]         # ids length indicating in_scope
    downs: NDArray[np.uint8]     # array of cells in down supercell
    ups: NDArray[np.uint8]       # array of cells in up supercell
    child: NDArray[np.uint8]     # (12,3)  uint8  (child transitions)
    mcc2: NDArray[np.uint8]      # [super-cell-mode,cell]->c2
    cmc2n: NDArray[np.uint8]     # [cell,super-cell-mode,c2]->neighbour
    loc_offs: NDArray[np.uint8]  # [cell_mode, super-cell-mode, c2, sibling]->neighbour


@dataclass(frozen=True, slots=True)
class H9Context:
    """H9 Context"""
    k: H9ConstLike
    cl: H9ClassifierLike
    c: H9CellLike
    r: H9RegionLike


# --- StepEvent dataclasses for per-layer introspection ---
@dataclass(slots=True)
class StepEvent:
    i: int
    phase: Literal['pre', 'post']
    addresses: np.ndarray
    pmo: np.ndarray
    cid: np.ndarray
    bad: np.ndarray
    y: np.ndarray
    ẋ: np.ndarray


@dataclass(slots=True)
class StepEventXY(StepEvent):
    space: Literal['xy'] = 'xy'


def near_ulps(x, target, k=4.0):
    """Returns when something is close to a specific pt"""
    x = np.asarray(x, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    ulp = np.maximum(np.spacing(target), np.finfo(np.float64).eps)
    return np.abs(x - target) <= k * ulp


def _mode_child_c2_builder(cells: np.ndarray, c2: np.ndarray, mode: np.ndarray, bad: int, t_size: int = 96) -> NDArray[np.uint8]:
    """
    Build per-mode child→c2 map
    Typical Hex9 use: K == 3 → shape (2, 3, 3).
    Returns
    mo_c2_of_cell : np.ndarray
        int8 array, shape (2, table_size). For a given mode m ∈ {0,1} and child cell id, gives
        c2 ∈ {0,1,2}, or bad if that cell is not a C2-child for that mode.

    c2 has a shape of modes/c2s/cells
    This function requires that there are the same number of cells under each c2 for each mode.
    Under a given supercell mode, which cells can be found in which c2?
    """
    m_sh, c_sh, k_sh = c2.shape   # gather sizes of modes, c2s, cells.
    mo_c2_of_cell = np.full((m_sh, t_size), bad, dtype=np.uint8)  # generate full lut
    c2_ids_flat = np.repeat(np.arange(c_sh, dtype=np.uint8), k_sh)  # keep c2s tied to corresponding cells
    for m in range(m_sh):
        cells_flat = c2[m].reshape(-1)   # flatten the cells for this mode.
        mo_c2_of_cell[m, cells_flat] = c2_ids_flat  # assign corresponding c2s to those cells.
    return mo_c2_of_cell


def _neighbour_builder(size, invalid_region) -> NDArray[np.uint8]:
    """
    Construct LUT for neighbour identity
    :param size number of cells.
    :param invalid_region constant.
    :return: LUT (Given a region, super_region_mode, c2, return [neighbour, neighbour-super_region_mode)
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


def _local_offset_lut_builder():
    """
    Builds and caches the local/neighbouring offset LUT.
    Give self_mode, parent_mode, c2, sibling-bool return the offset to the neighbour.
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


def region_constants(h9c: H9CellLike = None) -> H9Region:
    """Instantiate Region Const & return it"""
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
        ids=np.arange(h9c.count),
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
    """Lazy imports to avoid cycles"""
    from hhg9.h9.constants  import H9K
    from hhg9.h9.classifier import H9CL
    from hhg9.h9.lattice    import H9C
    # from hhg9.h9.region     import H9R
    return H9Context(k=H9K, cl=H9CL, c=H9C, r=H9R)


H9CTX = default_ctx()


def ulp_nudge(z: np.ndarray, unit: float) -> np.ndarray:
    """
    Single-ULP nudge of z toward the nearest integer multiple of `unit`.
    Works elementwise and returns a new array; safe for masked slices like z[idx].
    """
    target = np.rint(np.divide(z, unit, out=np.zeros_like(z), where=(unit != 0))) * unit
    return np.nextafter(z, np.where(unit != 0, target, z))


def at_vertices(ẋ, y, mode, ulps, h9k: H9ConstLike):
    """
    Return mask for points at the barycentric supercell vertices.
    There are six vertices, with three at any given mode.
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


def hard_clamp(ẋ, y, mode, ulps, h9k: H9ConstLike):
    """
    Given arrays of points and modes, clamps them to be within their
    respective barycentric triangles.
    """
    ẋ_final = ẋ.copy()
    y_final = y.copy()

    # --- Calculate the Clamped Result for UP Mode Points ---
    up_mask = (mode == 1)
    if np.any(up_mask):
        y_up, ẋ_up = y[up_mask], ẋ[up_mask]

        # CHANGED: Check for nearness on the ORIGINAL y_up values, *before* clipping.
        at_apex = near_ulps(y_up, h9k.limits.ΛC, k=ulps)
        at_base = near_ulps(y_up, h9k.limits.ΛF, k=ulps)
        if np.any(at_base | at_apex):
            # Now, perform the clipping. This is still necessary for points far out of bounds.
            y_up_clamped = np.clip(y_up, h9k.limits.ΛF, h9k.limits.ΛC)

            # Calculate the ẋ limit based on the clamped y...
            max_abs_ẋ = h9k.limits.ΛC - y_up_clamped
            # ...but force it to zero if the original point was already near the apex.
            max_abs_ẋ = np.where(at_apex, 0.0, max_abs_ẋ)

            ẋ_clamped = np.clip(ẋ_up, -max_abs_ẋ, max_abs_ẋ)

            # Place the final values back, snapping to the base if the original was near it.
            ẋ_final[up_mask] = ẋ_clamped
            y_final[up_mask] = np.where(at_base, h9k.limits.ΛF, y_up_clamped)

    # --- 2. Calculate the Clamped Result for DOWN Mode Points ---
    down_mask = (mode == 0)
    if np.any(down_mask):
        y_down, ẋ_down = y[down_mask], ẋ[down_mask]

        # CHANGED: Check for nearness on the ORIGINAL y_down values.
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


def soft_clamp(ẋ, y, mode, ulps, h9k: H9ConstLike):
    """
    Clamps points with minimal modifications, only snapping points that are
    both near a boundary and out-of-bounds. Valid points are never modified.
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
        # Correct only points that are near the apex AND out-of-bounds (y > ΛC)
        oob_apex_mask = is_near_apex & (y_up > h9k.limits.ΛC)
        if np.any(oob_apex_mask):
            indices_to_fix = up_indices[oob_apex_mask]
            y_final[indices_to_fix] = h9k.limits.ΛC
            ẋ_final[indices_to_fix] = 0.0

        # --- Base Group (UP) ---
        # Correct only points near the base AND out-of-bounds (y < ΛF)
        oob_base_mask = is_near_base & (y_up < h9k.limits.ΛF)
        if np.any(oob_base_mask):
            indices_to_fix = up_indices[oob_base_mask]
            y_final[indices_to_fix] = h9k.limits.ΛF
            max_ẋ_at_base = h9k.limits.ΛC - h9k.limits.ΛF
            ẋ_final[indices_to_fix] = np.clip(ẋ[indices_to_fix], -max_ẋ_at_base, max_ẋ_at_base)

        # --- Middle Group (UP) ---
        # Logic from before, which already respects the "only modify OOB" rule for ẋ
        middle_indices = up_indices[is_in_middle]
        if middle_indices.size > 0:
            # (Same logic as before, which is correct)
            y_mid, ẋ_mid = y[middle_indices], ẋ[middle_indices]
            max_abs_ẋ = h9k.limits.ΛC - y_mid
            # Correct right side
            to_correct_right = (ẋ_mid > max_abs_ẋ) & near_ulps(ẋ_mid, max_abs_ẋ, k=ulps)
            if np.any(to_correct_right):
                ẋ_mid[to_correct_right] = max_abs_ẋ[to_correct_right]
            # Correct left side
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
        # Correct only points near the apex AND out-of-bounds (y < VF)
        oob_apex_mask = is_near_apex & (y_down < h9k.limits.VF)
        if np.any(oob_apex_mask):
            indices_to_fix = down_indices[oob_apex_mask]
            y_final[indices_to_fix] = h9k.limits.VF
            ẋ_final[indices_to_fix] = 0.0

        # --- Base Group (DOWN) ---
        # Correct only points near the base AND out-of-bounds (y > VC)
        oob_base_mask = is_near_base & (y_down > h9k.limits.VC)
        if np.any(oob_base_mask):
            indices_to_fix = down_indices[oob_base_mask]
            y_final[indices_to_fix] = h9k.limits.VC
            max_ẋ_at_base = h9k.limits.VC - h9k.limits.VF
            ẋ_final[indices_to_fix] = np.clip(ẋ[indices_to_fix], -max_ẋ_at_base, max_ẋ_at_base)

        # --- Middle Group (DOWN) ---
        middle_indices = down_indices[is_in_middle]
        if middle_indices.size > 0:
            # (Same logic as before, which is correct)
            y_mid, ẋ_mid = y[middle_indices], ẋ[middle_indices]
            max_abs_ẋ = y_mid - h9k.limits.VF
            # Correct right side
            to_correct_right = (ẋ_mid > max_abs_ẋ) & near_ulps(ẋ_mid, max_abs_ẋ, k=ulps)
            if np.any(to_correct_right):
                ẋ_mid[to_correct_right] = max_abs_ẋ[to_correct_right]
            # Correct left side
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
    """Common 'bad' squashing routine for xy/xyu steppers.
    Returns (cid_updated, bad_next), where bad_next is a full-length mask of still-bad rows.
    Modifies ẋ,y to adjust for forward cascade.
    Steps:
      1) single-ULP nudge of y toward nearest k·V
      2) single-ULP nudge of ẋ toward nearest m·Ü
      3) 1-ulps clamp to grid.
      4) 5-ulps clamp to grid.
    """
    from hhg9.h9.classifier import in_scope, classify_mode_cell
    to_fix = np.flatnonzero(bad)

    # Run recovery only when there are bad rows; otherwise, return early.
    if to_fix.size == 0:
        return cid  # nothing to do

    lat_v = h9k.lattice.V
    lat_ü = h9k.lattice.Ü

    # Each strategy is a lambda that takes subsets and returns (ẋ_loc, y_loc).
    strategies = [
        (lambda ẋ_s, y_s, **_: (ẋ_s, ulp_nudge(y_s, lat_v))),  # 1. Nudge y
        (lambda ẋ_s, y_s, **_: (ulp_nudge(ẋ_s, lat_ü), y_s)),  # 2. Nudge ẋ
        (lambda ẋ_s, y_s, m_s: soft_clamp(ẋ_s, y_s, m_s, 1.0, h9k)),  # Clamp to grid
        (lambda ẋ_s, y_s, m_s: soft_clamp(ẋ_s, y_s, m_s, 2.0, h9k)),  # Clamp to grid
        (lambda ẋ_s, y_s, m_s: soft_clamp(ẋ_s, y_s, m_s, 4.0, h9k)),  # Clamp to grid
        (lambda ẋ_s, y_s, m_s: hard_clamp(ẋ_s, y_s, m_s, 10.0, h9k)),  # Clamp to grid
    ]

    # Loop through each strategy until all bad rows are fixed.
    for fix_int, fixer in enumerate(strategies):
        if to_fix.size == 0:
            break  # Stop if all rows have been recovered.

        m_sub = p_mo[to_fix]  # Select the data for the still-bad rows
        ẋ_sub = ẋ[to_fix]
        y_sub = y[to_fix]
        ẋ_loc, y_loc = fixer(ẋ_sub, y_sub, m_s=m_sub)  # Apply current recovery strategy
        ok = in_scope(ẋ_loc, y_loc, m_sub, h9cl)  # Check which of the attempted fixes were successful
        if np.any(ok):
            # print(f'fixed {np.sum(ok.astype(np.uint8))} at strategy {fix_int + 1}')
            ok_idx = to_fix[ok]  # Get the original indices of the newly fixed rows
            cid[ok_idx] = classify_mode_cell(ẋ_loc[ok], y_loc[ok], m_sub[ok], h9cl)  # update cls.
            ẋ[ok_idx] = ẋ_loc[ok]
            y[ok_idx] = y_loc[ok]
            to_fix = to_fix[~ok]
    return cid


# --- Iterator variant: per-layer event generator for xy stepper ---
def xy_regions_iter(xy, mode=None, depth=36, ctx: H9Context = None):
    """Generator that mirrors xy_regions but yields per-layer StepEventXY.
    Yields two events per layer (pre/post) and returns the final addresses at StopIteration.
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
    # recip_r3 = (h9k.U / h9k.Ü)

    p_mo = h9c.mode[pid]
    bad = ~clf.in_scope(ẋ, y, p_mo, h9cl)
    # Pre-amble adjustment for initial outliers: clamp/snap only rows that are out-of-scope
    if np.any(bad):
        # Clamp to the current face bounds (mode-aware) and resnap on the lattice
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


def xy_regions(xy, mode=None, depth=36, ctx: H9Context = None) -> NDArray[np.uint8]:
    """
    Given a vector of barycentric coords create a set of layered regions
    Must indicate the layer 0 mode.
    """
    # Use the iterator but materialize the result for compatibility.
    ctx = ctx or default_ctx()
    it = xy_regions_iter(xy, mode=mode, depth=depth, ctx=ctx)
    addresses = None
    for ev in it:
        pass
    addresses = ev.addresses if 'ev' in locals() else None
    return addresses


def regions_xy(uri_address, ctx: H9Context = None) -> NDArray[np.float64]:
    """
    REVERSE: URI addresses back into (x,y) coordinates and initial mode.
    Inverse of xy_regions/xy_regions_iter.
    We reconstruct in (ẋ,y) space using off_ẋy, then convert ẋ→x via 1/√3.
    """
    ctx = ctx or default_ctx()
    h9k, h9cl, h9c, cr = ctx.k, ctx.cl, ctx.c, ctx.r
    num_points, depth = uri_address.shape

    # Work in ẋ,y space; start at the origin and walk backward: ẋ = ẋ/3 + off_ẋ; y = y/3 + off_y
    offs_ẋ = h9c.off_ẋy[:, 0].astype(np.float64, copy=False)
    offs_y = h9c.off_ẋy[:, 1].astype(np.float64, copy=False)

    ẋ = np.zeros(num_points, dtype=np.float64)
    y = np.zeros(num_points, dtype=np.float64)

    # Walk from the last *real* layer down to 1 (skip proto at index 0)
    for i in range(depth - 1, 0, -1):
        cid = uri_address[:, i]
        # If any rows carry invalid_region (e.g., shorter depth), skip those
        valid = (cid != cr.invalid_region)
        if not np.any(valid):
            continue
        ẋ[valid] = ẋ[valid] / 3.0 + offs_ẋ[cid[valid]]
        y[valid] = y[valid] / 3.0 + offs_y[cid[valid]]

    # Convert back to x from ẋ
    x = ẋ * (h9k.U / h9k.Ü)

    # Initial mode: prefer the first real child if present; fallback to proto nibble
    # first = uri_address[:, 1]
    # has_first = (first != cr.invalid_region)
    # mode_from_first = np.where(has_first, H9CTX.c.mode[first], 0).astype(np.uint8)
    # mode_from_proto = (uri_address[:, 0] == cr.proto_up).astype(np.uint8)
    # initial_mode = np.where(has_first, mode_from_first, mode_from_proto)
    initial_mode = (uri_address[:, 0] == cr.proto_up).astype(np.uint8)
    return np.stack([x, y, initial_mode], axis=-1)


def rn_diagnostic(cur, pmo, c2, nbr, pmn):
    """Diagnostic to check the region neighbours algorithm."""
    same_parent = (pmn == pmo)
    idx = np.flatnonzero(same_parent)
    if idx.size:
        bad = []
        for i in idx:
            ok_back = False
            for c2r in (0, 1, 2):
                nb2, pm2 = H9R.cmc2n[nbr[i], pmo[i], c2r]
                if nb2 == cur[i]:
                    ok_back = True
                    break
            if not ok_back:
                bad.append((int(cur[i]), int(pmo[i]), int(c2[i]),
                            int(nbr[i]), int(pmn[i])))
        if bad:
            # aggregate to see pattern
            import collections
            cnt = collections.Counter((b[0], b[1]) for b in bad)
            print("cmc2n inverse failures (cur,pmo)->count:", cnt)
            # print a few concrete examples
            print("examples:", bad[:10])


def region_neighbours(addresses, ctx: H9Context = None):
    """
    Given a region list/hierarchy,
    the 'P' -3 ante-penultimate region is the operating mode
    the 'I' -2 penultimate region as the point of interest, (it's mode and...)
    the 'C' -1 terminal region is the C2 determinate.
    """
    ctx = ctx or default_ctx()
    h9k, h9cl, h9c, h9r = ctx.k, ctx.cl, ctx.c, ctx.r

    count, layers = addresses.shape
    nb_array = addresses.copy()
    xl = layers

    # Identify P, I and their modes
    par_index = -3 if xl > 2 else -2
    p_region = addresses[:, par_index]      # P
    cur = addresses[:, -2]                  # I
    imo = h9c.mode[cur]                     # I.m

    # Base C2 lookup from terminal C
    term = addresses[:, -1]                 # C
    c2 = h9r.mcc2[imo, term]                # may be invalid_region
    bad_val = h9r.invalid_region
    bad = (c2 == bad_val)

    if np.any(bad):
        # Geometric fallback for rows where C isn't a recognised C2 child:
        # choose the C2 group whose member cells are closest (in xy) to C.
        off_xy = h9c.off_xy                 # (cell_count, 2)
        for i in np.flatnonzero(bad):
            mode_i = imo[i]
            t = term[i]
            # If even C is the global invalid marker, just give it some default
            if t == bad_val:
                c2[i] = 0
                continue
            xy_t = off_xy[t]                # (2,)
            cells_mode = h9c.c2[mode_i]     # shape (3, 3): cell ids per C2 group
            cand_ids = cells_mode.reshape(-1)
            d2 = np.sum((off_xy[cand_ids] - xy_t) ** 2, axis=1)
            best_flat = int(np.argmin(d2))
            k = cells_mode.shape[1]         # usually 3
            best_c2 = best_flat // k        # 0,1,2
            c2[i] = np.uint8(best_c2)

    # Now c2 ∈ {0,1,2} for all rows
    pmo = h9c.mode[p_region]
    nb_m = h9r.cmc2n[cur, pmo, c2]          # neighbour + its parent_mode
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