# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Together (u, v, c2) uniquely identifies a hexagon at a given layer.

Forward:  hex_digits_to_uvc(hx)         -> (u, v, c2)
Inverse:  uvc_to_hex_digits(u,v,c2,c_mo,r_mo,layer) -> body digits

Disambiguation rule (proved by exhaustive LUT check):
  Every UV collision is a (p_mo=0, p_mo=1) pair.
  Propagating r_mo forward from the root resolves all ambiguities in O(layer) time.
"""
from hhg9.h9 import HEX_LUTS, H9_RA
import numpy as np

from hhg9.h9.tail import tail_unpack_reversible


# ---------------------------------------------------------------------------
# LUT: mode-0 UV for each (digit, c_mo, c2) context
# ---------------------------------------------------------------------------

def _build_hex_uvc_lut() -> np.ndarray:
    """Shape (9, 2, 3, 2): mode-0 UV for each valid (digit, c_mo, c2)."""
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
                cell = H9_RA.rid2cell[c_reg]  # actual cell, not mode-0 normalised
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
    # Walk leaf -> root, collecting mode-0 UV at each level.
    uv_stack = []
    c_mo_t = c_mo.copy()
    c2_t = c2.copy()
    for i in range(layer - 1, 0, -1):
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
    mode-0 UV satisfies (qr - uv) % 3 == 0 (both components).  Ambiguous
    pairs (same UV, p_mo=0 vs p_mo=1) are resolved because only one branch
    reaches the root with c_mo == r_mo.

    body[:, 0] (the root hex digit, encoded in the octant) is left as 0xFF.

    Args:
        u, v  : (N,) int32
        c2    : (N,) uint8   leaf c2 from tail
        c_mo  : (N,) uint8   leaf parent-mode from tail
        r_mo  : (N,) uint8   root mode from tail
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
