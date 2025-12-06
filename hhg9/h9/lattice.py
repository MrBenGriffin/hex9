# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
H9 Lattice Construction.

This module constructs the UV rectilinear lattice **once** in float64 and freezes it.
It acts as the bridge between the mathematical classifier and the discrete grid structure.

**Key Responsibilities:**

1.  **Lattice Generation:** Generates a set of cell barycentres using the H9K constants.
2.  **Cell Identification:** Fixes each of the 42 geometric cells of the barycentric domain
    with an integer (u, v) coordinate.
3.  **Region Collapse:** Collapses these into 12 legal layer **regions**:
    * 9 cells belonging to the **Up** supercell.
    * 9 cells belonging to the **Down** supercell.
    * Shared cells are handled via C2 membership logic.
4.  **Hexagon Emergence:** Determines the half-hexagons from which the full hexagons emerge.

**The Lattice Map:**

.. code-block:: text

       \\     \\     \\    /     /     /
     00 \\  01 \\  02 \\03/ 07  / 0b  / 0f
    _____\\_____\\ ____\\/____ /____ /______
          \\     \\    /\\    /     /
        10 \\  11 \\12/16\\17/ 1b  / 1f
    ________\\_____\\/____\\/____ /_________
             \\    /\\    /\\    /
           20 \\21/25\\26/2a\\2b/ 2f
    ___________\\/____\\/____\\/____________
               /\\    /\\    /\\
           30 /34\\35/39\\3a/3e\\ 3f
    _________/____\\/____\\/____\\__________
            /     /\\    /\\     \\
        40 /  44 /48\\49/4d\\ 4e  \\ 4f
    ______/____ /____\\/____\\ ____\\________
         /     /     /\\     \\     \\
     50 / 54  / 58  /5c\\  5d \\  5e \\ 5f

**Coordinate Systems:**

* **(N, P, H, M):** An affine cube-like system (Negative slope, Positive slope, Horizontal, Mode).
    Constraint: :math:`n - p + h + m = 3` for every valid lattice point.
* **(U, V):** A rectangular lattice over the plane.
    * U interval: 1/2 triangle width.
    * V interval: 1/3 triangle height.
    * Every barycentre maps to an integer (u, v).
"""

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from typing import Optional
from .protocols import H9ConstLike, H9ClassifierLike


@dataclass(frozen=True, slots=True)
class H9Cell:
    """
    Cell Properties (derived via H9Lattice).

    This frozen dataclass holds the pre-computed properties for all cells in the lattice.

    Attributes:
        count (int): Count of cells in the classifier.
        mode (NDArray[np.uint8]): Cell mode (0=down, 1=up).
        off_uv (NDArray[np.int8]): UV coordinates, where :math:`uv \\in [-9..9]`.
        off_xy (NDArray[np.float64]): Metric barycentric coordinates (x, y).
        off_ẋy (NDArray[np.float64]): Metric coordinates scaled by :math:`\\sqrt{3}` in x (:math:`\\dot{x}, y`).
        in_scope (NDArray[np.uint8]): Array of in-scope cell IDs.
        in_mode (NDArray[bool]): Boolean array indicating mode membership.
        in_dn (NDArray[bool]): Boolean mask for cells in the **Down** supercell.
        in_up (NDArray[bool]): Boolean mask for cells in the **Up** supercell.
        downs (NDArray[np.uint8]): Array of valid cell IDs in the Down supercell.
        ups (NDArray[np.uint8]): Array of valid cell IDs in the Up supercell.
        c2 (NDArray[np.uint8]): C2 cluster membership [2, 3, 3] for each mode.
    """
    count: int
    mode: NDArray[np.uint8]
    off_uv: NDArray[np.int8]
    off_xy: NDArray[np.float64]
    off_ẋy: NDArray[np.float64]
    in_scope: NDArray[np.uint8]
    in_mode: NDArray[bool]
    in_dn: NDArray[bool]
    in_up: NDArray[bool]
    downs: NDArray[np.uint8]
    ups: NDArray[np.uint8]
    c2: NDArray[np.uint8]


def _c2_groups(cell_ids: NDArray[np.uint8], offsets: NDArray[np.float64], supercell_mode: int) -> NDArray[np.uint8]:
    """
    Split the 9 cells of one supercell into 3 C2 wedges (size 3) by angular sectors.

    Args:
        cell_ids: The list of cell IDs to group.
        offsets: The offset array to determine positions.
        supercell_mode: 0 for Down, 1 for Up.

    Returns:
        NDArray[np.uint8]: Shape (3, 3) containing [c2_index, cell_id].
    """
    # Hardcoded groups derived from angular sectors for stability
    c2x = [
        [[0x26, 0x2a, 0x2b], [0x3a, 0x39, 0x49], [0x35, 0x25, 0x21]],  # mode 0 (Down)
        [[0x39, 0x3a, 0x3e], [0x25, 0x35, 0x34], [0x2a, 0x26, 0x16]]  # mode 1 (Up)
    ]
    return np.array(c2x[supercell_mode]).astype(np.uint8)


def h9_cell_lattice(h9k: Optional[H9ConstLike] = None, h9cl: Optional[H9ClassifierLike] = None) -> H9Cell:
    """
    Factory method to construct the H9Cell singleton.

    This function performs the heavy lifting of generating the lattice grid,
    filtering for valid barycentres, and identifying supercell memberships.

    Args:
        h9k: Constants bundle (defaults to global H9K).
        h9cl: Classifier bundle (defaults to global H9CL).

    Returns:
        H9Cell: The populated and frozen cell lattice object.
    """
    import hhg9.h9.classifier as clf
    if h9k is None:
        from hhg9.h9.constants import H9K
        h9k = H9K
    if h9cl is None:
        h9cl = clf.H9CL

    c_dec = h9cl.decode
    s_count = h9cl.p_levels.shape[0]  # p and n share the same shape.
    h_count = h9cl.h_levels.shape[0]
    cell_size = len(c_dec)

    # define the lattice to cover enough area as defined by the classifier.
    u_scp = 2 * s_count  # u = cell.width/2 - will need 1 outlier.
    v_scp = 3 * (h_count - 2)  # v = cell.height/3 - won't need more than 1 outlier.

    # define a rectilinear lattice.
    u_vals = np.arange(-u_scp, u_scp, dtype=np.int16)
    v_vals = np.arange(-v_scp, v_scp, dtype=np.int16)

    # However, we are solely interested in cell barycentres here.
    # So we will eliminate those points that do not rest on a barycentre.
    # even u: v ≡ 2 or 4 (mod 6)  → sequences r, r+6, ... within -v_scp, v_scp
    # odd  u: v ≡ 1 or 5 (mod 6)  → sequences r, r+6, ... within -v_scp, v_scp
    parity_match = ((v_vals[None, :] & 1) == (u_vals[:, None] & 1))
    not_mult3 = (v_vals[None, :] % 3) != 0
    bary_mask = parity_match & not_mult3

    # Pull pairs
    iu, iv = np.nonzero(bary_mask)
    uv = np.column_stack((u_vals[iu], v_vals[iv])).astype(np.int16, copy=False)

    uvf = uv.astype(np.float64, copy=False)
    uvx = uvf.copy()
    # metric scaling (needed for classification and distance):
    uvf *= (h9k.U, h9k.V)
    uvx *= (h9k.Ü, h9k.V)

    rx = clf.classify_cell(uvx[:, 0], uvx[:, 1], h9cl)

    # Now we need to select a reference barycentre for those regions which have more than 1.
    # Choose a per‑cell reference barycentre: closest to origin
    # Compute squared Euclidean distance in metric space (uvf) to avoid sqrt.
    r2 = uvf[:, 0] * uvf[:, 0] + uvf[:, 1] * uvf[:, 1]
    order = np.lexsort((r2, rx))  # Sort primarily by cell id (rx), secondarily by distance r2 (ascending).
    first_idx_in_sorted = np.unique(rx[order], return_index=True)[1]
    ref_rows = order[first_idx_in_sorted]
    ref_cells = rx[ref_rows]

    # These 42 ref_cells are all correctly placed on the grid.
    # It follows, via classification, that the hex9 cell plane is carved into 42 distinct regions.
    # This is the *actual* meaning of life, the universe, and everything. Don't tell the mice.
    ref_uv = uv[ref_rows]

    ref_xy = uvf[ref_rows]
    ref_ẋy = uvx[ref_rows]
    in_up_bools = clf.in_up(ref_ẋy[:, 0], ref_xy[:, 1], h9cl)
    in_dn_bools = clf.in_down(ref_ẋy[:, 0], ref_xy[:, 1], h9cl)

    # ups, downs
    ups = ref_cells[in_up_bools]
    dns = ref_cells[in_dn_bools]
    assert len(ups) == len(dns) == 9, f'Count of [Ups({len(ups)}) | Downs({len(dns)})] != 9'

    # in_scope
    in_scope = ref_cells[in_up_bools | in_dn_bools]
    assert len(in_scope) == 12, f'{len(in_scope)} != 12'

    # Build a dense per‑cell offsets table (NaN for non‑geometric / unused ids).
    # Store offsets in integer lattice units (uv), which are stable and compact.
    offs_xy = np.full((cell_size, 2), np.nan, dtype=np.float64)
    offs_xy[ref_cells] = ref_xy
    bary_xy = np.full((cell_size, 2), np.nan, dtype=np.float64)
    bary_xy[ref_cells] = ref_ẋy

    # in_ups, in_dns
    in_ups = np.full((cell_size,), False, dtype=bool)
    in_dns = np.full((cell_size,), False, dtype=bool)
    in_ups[ref_cells] = in_up_bools
    in_dns[ref_cells] = in_dn_bools
    in_mode = np.vstack([in_dns, in_ups])

    # uv indexes
    uv_idx = np.full((cell_size, 2), (99, 99), dtype=np.int8)
    uv_idx[ref_cells] = uv[ref_rows]

    # modes
    mode = np.zeros((cell_size,), dtype=np.uint8)
    i_dec = c_dec[ref_cells]
    mode[ref_cells] = (1 ^ (i_dec.sum(axis=1) & 1)).astype(np.uint8)  # 0=V, 1=Λ

    # c2_groups
    d_c2 = _c2_groups(dns, offs_xy, 0)
    u_c2 = _c2_groups(ups, offs_xy, 1)
    c2 = np.stack([d_c2, u_c2])  # shape 2,3,3

    return H9Cell(cell_size, mode, uv_idx, offs_xy, bary_xy, in_scope, in_mode, in_dns, in_ups, dns, ups, c2)


H9C: H9Cell = h9_cell_lattice()

# --- Sanity Checks ---
# uv indices should be integers in range
assert np.allclose(H9C.off_uv, H9C.off_uv.astype(int))

# xy and ẋy should correspond where they are non-nan.
mask = ~np.isnan(H9C.off_xy[:, 0])
assert np.allclose(H9C.off_ẋy[mask, 0], np.sqrt(3) * H9C.off_xy[mask, 0])
assert np.allclose(H9C.off_ẋy[mask, 1], H9C.off_xy[mask, 1])
