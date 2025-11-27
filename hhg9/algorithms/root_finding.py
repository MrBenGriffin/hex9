# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
find_coords is a general root finding algorithm for finding coordinates
on the ellipsoidal surface of an equilateral polyhedron projection.
"""
import numpy as np
from hhg9.h9.protocols import H9CellLike


def find_coords(target_rll, initial_mode, target_octants, h9c: H9CellLike,
                projector_func, distance_func, depth=34, beam_width=5):
    """
    Vectorised beam search in barycentric xy space.
    target_rll: (N,2) lat/lon (or whatever the projector uses)
    initial_mode: (N,) int {0,1}
    target_octants: (N,3) face-signs for projection
    h9c: H9Cell instance providing mode, offsets, and child region info
    projector_func: (XY,(...))->(M,2) projects bary XY to target space
    distance_func: ((M,2),(M,2))->(M,) distances in target space
    offset_kind: which offset flavour to use ("xy", "xbar_y"/"ẋy"/"xby", or "uv")
    """
    num_pts = target_rll.shape[0]  # N
    off = h9c.off_xy

    # child region id lists per supercell mode (shape (9,))
    up_children = h9c.ups
    down_children = h9c.downs
    mode_lut = h9c.mode

    # Roots by mode: up→0x16, down→0x49
    root_uris = np.where(initial_mode == 1, 0x16, 0x49).astype(np.uint8)

    # Beam state
    best_paths = root_uris[:, None, None]  # (N,1,1)
    best_xy = np.zeros((num_pts, 1, 2), dtype=np.float64)  # (N,1,2)

    for i in range(depth):
        k = best_paths.shape[1]  # current beam width
        last = best_paths[:, :, -1]  # (N,k)
        par_mode = mode_lut[last]  # (N,k)

        # Select children per parent mode
        children = np.where(par_mode[..., None] == 1, up_children, down_children)  # (N,k,9)

        # Expand candidate URIs and coords
        cand_uris = children.reshape(num_pts, k * 9)  # (N,k*9)
        scale = (1.0 / 3.0) ** i
        par_xy = np.repeat(best_xy, 9, axis=1)  # (N,k*9,2)
        offs = off[cand_uris]  # (N,k*9,2)
        cand_xy = par_xy + offs * scale  # (N,k*9,2)

        # Project once: flatten → project → reshape
        nk9 = num_pts * (k * 9)
        flat_xy = cand_xy.reshape(nk9, 2)
        flat_oct = np.repeat(target_octants, k * 9, axis=0)  # (nk9,3)
        proj = projector_func(flat_xy, flat_oct).reshape(num_pts, k * 9, 2)

        # Distances to targets
        d = distance_func(proj, target_rll[:, None, :])  # (N,k*9)

        # Top-k via arg-partition, then stable tie-break by URI
        idx_k = np.argpartition(d, beam_width - 1, axis=1)[:, :beam_width]  # (N,k)
        # Gather those distances/uris for stable sort
        d_k = np.take_along_axis(d, idx_k, axis=1)  # (N,k)
        u_k = np.take_along_axis(cand_uris, idx_k, axis=1)  # (N,k)
        # Stable rank on (distance, uri)
        order = np.lexsort((u_k, d_k))  # (N,k)
        top_idx = np.take_along_axis(idx_k, order, axis=1)  # (N,k)

        # Keep winners
        win_uris = np.take_along_axis(cand_uris, top_idx, axis=1)  # (N,k)
        win_xy = np.take_along_axis(cand_xy, top_idx[..., None], axis=1)  # (N,k,2)

        # Extend paths: pick corresponding parent slices
        parent_sel = top_idx // 9  # (N,k)
        parent_paths = np.take_along_axis(best_paths, parent_sel[..., None], axis=1)  # (N,k,depth_so_far)
        best_paths = np.concatenate([parent_paths, win_uris[..., None]], axis=2)  # (N,k,depth_so_far+1)
        best_xy = win_xy

    # Take the first beam entry as the result
    return best_xy[:, 0, :], best_paths[:, 0, :]
