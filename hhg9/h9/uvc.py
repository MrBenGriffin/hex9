# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Together (u, v, c2) uniquely identifies a hexagon at a given layer.

Forward:  hex_digits_to_uvc(hx)         -> (u, v, c2)
Inverse:  uvc_to_hex_digits(u,v,c2,c_mo,r_mo,layer) -> body digits
Step:     hex_step(hx, target_layer, direction)      -> new hx

Disambiguation rule (proved by exhaustive LUT check):
  Every UV collision is a (p_mo=0, p_mo=1) pair.
  Propagating r_mo forward from the root resolves all ambiguities in O(layer) time.

Direction convention (6 axes, 0-indexed):
  0 / 3:  ±c2=0 axis   (perpendicular to c2=0 edge)
  1 / 4:  ±c2=1 axis
  2 / 5:  ±c2=2 axis
"""
from hhg9.h9 import HEX_LUTS, H9_RA
import numpy as np

from hhg9.h9.tail import tail_unpack_reversible, tail_pack_reversible


# ---------------------------------------------------------------------------
# LUT: net_mode-0 UV for each (digit, c_mo, c2) context
# ---------------------------------------------------------------------------

def _build_hex_uvc_lut() -> np.ndarray:
    """Shape (9, 2, 3, 2): net_mode-0 UV for each valid (digit, c_mo, c2)."""
    from hhg9.h9.lattice import H9C
    hr = HEX_LUTS.hex_reg
    oob = HEX_LUTS.hex_oob
    lut = np.full((9, 2, 3, 2), 99, dtype=np.int8)
    for d in range(9):
        for c_mo in range(2):
            for c2 in range(3):
                e = hr[d, c_mo, c2]
                if e[0] == oob:
                    continue
                c_reg = int(e[0])
                cell = H9_RA.rid2cell[c_reg]  # actual cell, not net_mode-0 normalised
                lut[d, c_mo, c2] = H9C.off_uv[cell]
    return lut


_LUT = _build_hex_uvc_lut()  # (9, 2, 3, 2)


# ---------------------------------------------------------------------------
# Reverse map:  (c_mo, c2, u%3)  ->  [(digit, p_mo, p_c2, uv), ...]
# Used in the inverse pass.
# ---------------------------------------------------------------------------

def _build_reverse_map() -> dict:
    """(c_mo, c2, u_mod3) -> list of (digit, p_mo, p_c2, uv_tuple)"""
    hr = HEX_LUTS.hex_reg
    oob = HEX_LUTS.hex_oob
    rev = {}
    for d in range(9):
        for c_mo in range(2):
            for c2 in range(3):
                uv = _LUT[d, c_mo, c2]
                if uv[0] == 99:
                    continue
                e = hr[d, c_mo, c2]
                if e[0] == oob:
                    continue
                u_mod3 = int(uv[0]) % 3
                key = (c_mo, c2, u_mod3)
                rev.setdefault(key, []).append(
                    (d, int(e[1]), int(e[2]), (int(uv[0]), int(uv[1])))
                )
    return rev


_REV = _build_reverse_map()


# ---------------------------------------------------------------------------
# Direction vectors in UV space
# Derived at init from H9R.cmc2n (same-parent neighbours) + H9C.off_uv.
# Cell 0x25 is interior to both net_mode-0 and net_mode-1 supercells; all three
# c2-direction neighbours in a net_mode-0 parent are same-parent.
# ---------------------------------------------------------------------------

def _build_dir_uvs() -> np.ndarray:
    """Return shape (6, 3, 2) int32: direction vectors indexed by (axis, source_c2_class, uv).

    The UV lattice has three sublattices (one per c2 class 0/1/2).  The direction
    vector for a given axis differs between sublattices, so each of the 3 c2 classes
    needs its own reference cell from H9C.c2[net_mode=0].
    Directions 0-2 = +c2=0,1,2 axes;  3-5 = their negatives.
    """
    from hhg9.h9.lattice import H9C
    from hhg9.h9.region import H9R
    dirs = np.zeros((6, 3, 2), dtype=np.int32)
    for c2_class in range(3):
        ref = int(H9C.c2[0, c2_class, 0])  # one net_mode-0 reference cell per c2 class
        ref_uv = H9C.off_uv[ref].astype(np.int32)
        for c2_dir in range(3):
            nbr_cell, _nbr_pmo = H9R.cmc2n[ref, 0, c2_dir]
            nbr_uv = H9C.off_uv[nbr_cell].astype(np.int32)
            dirs[c2_dir, c2_class] = nbr_uv - ref_uv
            dirs[c2_dir + 3, c2_class] = ref_uv - nbr_uv
    return dirs


_DIR_UV = _build_dir_uvs()  # (6, 3, 2)


# ---------------------------------------------------------------------------
# Forward: hex address -> (u, v, c2)
# ---------------------------------------------------------------------------
def hex_digits_to_uvc(
        hx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Forward: hex address (body + reversible tail) -> (u, v, c2).
    Args:
        hx: (N, layer+1) uint8 — body digits + reversible tail column.

    Returns:
        u, v : (N,) int32
        c2   : (N,) uint8   (leaf c2 from tail)
    """
    hx = np.asarray(hx, dtype=np.uint8)
    body = hx[:, :-1]
    tail = hx[:, -1]

    sz, layer = body.shape
    c_mo, c2, _r_mo, _tail_h = tail_unpack_reversible(tail)

    hr = HEX_LUTS.hex_reg
    # Walk leaf -> root, collecting net_mode-0 UV at each layer.
    uv_stack = []
    c_mo_t = c_mo.copy()
    c2_t = c2.copy()
    for i in range(layer - 1, 1, -1):  # stop at 2; body[:,1] is the proto sentinel (0x0F), not a digit
        d = body[:, i]
        uv = _LUT[d, c_mo_t, c2_t].astype(np.int32)  # (N, 2)
        uv_stack.append(uv)
        rmc = hr[d, c_mo_t, c2_t]
        c_mo_t = rmc[:, 1]
        c2_t = rmc[:, 2]

    # Accumulate root -> leaf:  qr = 3 * qr + uv
    qr = np.zeros((sz, 2), dtype=np.int32)
    for uv in reversed(uv_stack):
        qr = 3 * qr + uv

    return qr[:, 0], qr[:, 1], c2


# ---------------------------------------------------------------------------
# Inverse: (u, v, c2, c_mo, r_mo) -> body digits
# ---------------------------------------------------------------------------
def uvc_to_hex_digits(
        u: np.ndarray,
        v: np.ndarray,
        c2: np.ndarray,
        c_mo: np.ndarray,
        r_mo: np.ndarray,
        layer: int,
) -> np.ndarray:
    """
    Inverse: (u, v, c2, c_mo, r_mo) -> hex body digits (no tail).

    Uses a leaf->root DFS.  At each step, candidates are those digits whose
    net_mode-0 UV satisfies (qr - uv) % 3 == 0 (both components).  Ambiguous
    pairs (same UV, p_mo=0 vs p_mo=1) are resolved because only one branch
    reaches the root with c_mo == r_mo.

    body[:, 0] (the root hex digit, encoded in the octant) is left as 0xFF.

    Args:
        u, v  : (N,) int32
        c2    : (N,) uint8   leaf c2 from tail
        c_mo  : (N,) uint8   leaf parent-net_mode from tail
        r_mo  : (N,) uint8   root net_mode from tail
        layer : int           number of body columns

    Returns:
        body: (N, layer) uint8   body[n, 0] = 0xFF (root hex not recovered here)
        :param layer:
        :param r_mo:
        :param c_mo:
        :param c2:
        :param v:
        :param u:
    """
    u = np.asarray(u, dtype=np.int32)
    v = np.asarray(v, dtype=np.int32)
    c2 = np.asarray(c2, dtype=np.uint8)
    c_mo = np.asarray(c_mo, dtype=np.uint8)
    r_mo = np.asarray(r_mo, dtype=np.uint8)
    sz = u.shape[0]

    body = np.full((sz, layer), 0xFF, dtype=np.uint8)

    for n in range(sz):
        qr_root = np.array([u[n], v[n]], dtype=np.int32)

        # DFS: (remaining_levels, qr, c_mo_cur, c2_cur, digits_leaf_to_root)
        stack = [(layer - 1, qr_root.copy(), int(c_mo[n]), int(c2[n]), [])]
        found = None

        while stack and found is None:
            lvl, qr_cur, cm, c2v, digs = stack.pop()

            if lvl == 0:
                # Reached root: qr must be exactly zero AND c_mo must match r_mo
                if cm == int(r_mo[n]) and qr_cur[0] == 0 and qr_cur[1] == 0:
                    found = digs  # leaf-to-root order
                continue

            u_mod3 = int(qr_cur[0]) % 3
            if u_mod3 < 0:
                u_mod3 += 3

            for d, p_mo, p_c2, uv in _REV.get((cm, c2v, u_mod3), []):
                rem = qr_cur - np.array(uv, dtype=np.int32)
                if rem[0] % 3 == 0 and rem[1] % 3 == 0:
                    stack.append((lvl - 1, rem // 3, p_mo, p_c2, digs + [d]))

        if found is not None:
            # found is leaf->root; fill body root->leaf (skip body[:,0])
            for k, d in enumerate(reversed(found)):
                body[n, k + 1] = d
        else:
            print(f'  [WARN] pt {n}: no valid path found')
    return body


# ---------------------------------------------------------------------------
# Single-point DFS that also returns leaf tail context (p_mo, c2, h).
# Used by hex_step to rebuild the reversible tail after navigation.
# ---------------------------------------------------------------------------

def _solve_uvc(u_val: int, v_val: int, r_mo_val: int, layer: int):
    """
    DFS over all valid leaf (c_mo, c2) starting states for a single point.

    Returns (digits_leaf_to_root, leaf_p_mo, leaf_c2, leaf_h) or None.
      digits_leaf_to_root : list of ints, length layer-1 (body[1..layer-1])
      leaf_p_mo : parent-net_mode of the terminal region
      leaf_c2   : c2 of the terminal region
      leaf_h    : region ID of the terminal region (for tail)
    """
    hr = HEX_LUTS.hex_reg
    oob = HEX_LUTS.hex_oob
    qr_root = np.array([u_val, v_val], dtype=np.int32)

    # Enumerate all valid leaf (c_mo, c2) starting states from _REV
    u_mod3 = int(u_val) % 3
    if u_mod3 < 0:
        u_mod3 += 3

    starts = set()
    for cm in range(2):
        for c2v in range(3):
            if (cm, c2v, u_mod3) in _REV:
                starts.add((cm, c2v))

    for leaf_cm, leaf_c2v in starts:
        stack = [(layer - 1, qr_root.copy(), leaf_cm, leaf_c2v, [])]
        while stack:
            lvl, qr_cur, cm, c2v, digs = stack.pop()
            if lvl == 0:
                if cm == r_mo_val and qr_cur[0] == 0 and qr_cur[1] == 0:
                    # Recover terminal region h from leaf digit + leaf context
                    leaf_digit = digs[0]
                    e = hr[leaf_digit, leaf_cm, leaf_c2v]
                    leaf_h = int(e[0]) if e[0] != oob else 0
                    return digs, leaf_cm, leaf_c2v, leaf_h
                continue
            u_m3 = int(qr_cur[0]) % 3
            if u_m3 < 0:
                u_m3 += 3
            for d, p_mo, p_c2, uv in _REV.get((cm, c2v, u_m3), []):
                rem = qr_cur - np.array(uv, dtype=np.int32)
                if rem[0] % 3 == 0 and rem[1] % 3 == 0:
                    stack.append((lvl - 1, rem // 3, p_mo, p_c2, digs + [d]))
    return None


# ---------------------------------------------------------------------------
# hex_step: navigate one hex in a given direction at a given layer
# ---------------------------------------------------------------------------

def hex_step(
        hx: np.ndarray,
        target_layer: int,
        direction: int,
) -> np.ndarray:
    """
    Step each hex address one position in `direction` at `target_layer`.

    Args:
        hx:           (N, depth+1) uint8 — body digits + reversible tail column.
        target_layer: Body column index to step at (1 = coarsest, depth-1 = finest).
        direction:    0-5.  0-2 = +c2=0,1,2 axes;  3-5 = their negatives.

    Returns:
        New hex address array, same shape as hx.  body[:,0] (octant hex) is
        unchanged; carries propagate upward through body[:,1..depth-1] only.
        Returns None in the row if no valid neighbour was found (octant boundary).
    """
    if not (1 <= target_layer <= (hx.shape[1] - 2)):
        raise ValueError(f"target_layer must be 1..depth-1, got {target_layer}")
    if not (0 <= direction <= 5):
        raise ValueError(f"direction must be 0-5, got {direction}")

    hx = np.asarray(hx, dtype=np.uint8)
    body = hx[:, :-1]
    tail_col = hx[:, -1]
    depth = body.shape[1]           # number of body columns (includes layer-0)
    layer = depth                   # alias used by uvc functions

    u_all, v_all, c2_all = hex_digits_to_uvc(hx)
    p_mo_all, _c2t, r_mo_all, _h = tail_unpack_reversible(tail_col)

    # Place value of column `target_layer` = 3^(depth-1-target_layer)
    scale = 3 ** (depth - 1 - target_layer)

    result = hx.copy()
    for n in range(hx.shape[0]):
        c2_val = int(c2_all[n])  # c2 class of leaf cell (= target layer when target_layer == depth-1)
        du, dv = int(_DIR_UV[direction, c2_val, 0]), int(_DIR_UV[direction, c2_val, 1])
        u_new = int(u_all[n]) + du * scale
        v_new = int(v_all[n]) + dv * scale
        r_mo_val = int(r_mo_all[n])

        found = _solve_uvc(u_new, v_new, r_mo_val, depth - 1)
        if found is None:
            print(f'  [WARN] hex_step: pt {n} has no neighbour (octant boundary?)')
            continue

        digs, leaf_cm, leaf_c2v, leaf_h = found
        # Fill body columns 2..depth-1 (skip col 0 = root hex, col 1 = proto sentinel 0x0F)
        for k, d in enumerate(reversed(digs)):
            result[n, k + 2] = d
        # Rebuild tail
        new_tail = tail_pack_reversible(
            np.uint8(leaf_cm), np.uint8(leaf_c2v),
            np.uint8(r_mo_val), np.uint8(leaf_h),
        )
        result[n, -1] = new_tail

    return result
