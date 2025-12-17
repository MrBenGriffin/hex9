# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Checks cells, c2 via angular sectors.
This is possibly not used - but does demonstrate the geometric relations.
16 December 2025 0.1.0a3 (passed)
"""
import numpy as np
from hhg9.h9 import H9C


def c2_groups(cell_ids, offsets, supercell_mode):
    """
    Split the 9 cells of one supercell into 3 C2 wedges (size 3) by angular sectors.
    Returns (N,2): [c2 in {0,1,2}, cell_id]
    """
    # nudge pushes non-collinear cells to align with favoured group.
    # base aligns labels so C2=0 matches canon: Λ→base=0, V→base=2
    nudge, base = (1, 0) if supercell_mode else (-1, 2)
    pts = offsets[cell_ids].astype(np.float64)  # (N,2)
    theta6 = np.arctan2(pts[:, 1], pts[:, 0]) / (np.pi/3)  # 60° units, (half-integer)
    sector = np.rint(6.5 + theta6).astype(np.int32) % 6    # 30° shift; 6.5 + half-integer -> 0...5
    counts = np.bincount(sector, minlength=6)  # find singletons
    singleton_mask = counts[sector] == 1  # gather singleton mask
    sector[singleton_mask] += nudge  # merge singletons (will % 6 next)
    c2 = (((base - sector) % 6) >> 1).astype(np.int8)  # align to C2 canon, %6 and 0..2
    return np.column_stack((c2, cell_ids.astype(np.int16)))


if __name__ == '__main__':
    h9c = H9C
    # For V (down):   c2 = (u_idx - v_idx) % 3
    # For Λ (up):     c2 = (u_idx - v_idx + 1) % 3
    ups = h9c.ups
    dns = h9c.downs
    offsets = h9c.off_xy
    cell_uv = h9c.off_uv
    uvu = cell_uv[ups]
    uvd = cell_uv[dns]
    u_groups = c2_groups(ups, offsets, 1)
    for c2, cell in u_groups:
        print(c2, f'{cell:02x}')
    d_groups = c2_groups(dns, offsets, 0)
    for c2, cell in d_groups:
        print(c2, f'{cell:02x}')
