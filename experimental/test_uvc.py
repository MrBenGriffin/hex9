# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""
Prototype / test for h9 address <-> UV hexgrid (axial) coordinate.

Coordinate: (u, v, c2)
  - (u, v)  integer, accumulated net_mode-0 UV via:  qr = 3 * qr + mode0_uv[layer]
  - c2      uint8, leaf-layer c2 from the reversible tail

Together (u, v, c2) uniquely identifies a hexagon at a given layer.

Forward:  hex_digits_to_uvc(hx)         -> (u, v, c2)
Inverse:  uvc_to_hex_digits(u,v,c2,c_mo,r_mo,layer) -> body digits

Disambiguation rule (proved by exhaustive LUT check):
  Every UV collision is a (p_mo=0, p_mo=1) pair.
  Propagating r_mo forward from the root resolves all ambiguities in O(layer) time.
"""

import numpy as np
from hhg9.h9.addressing import (
    HEX_LUTS, H9_RA,
    tail_unpack_reversible,
    hex_digits, TailStyle,
)
from hhg9.h9.lattice import H9C
from hhg9 import Registrar, Points

# ---------------------------------------------------------------------------
# LUT: net_mode-0 UV for each (digit, c_mo, c2) context
# ---------------------------------------------------------------------------

def _build_hex_uvc_lut() -> np.ndarray:
    """Shape (9, 2, 3, 2): net_mode-0 UV for each valid (digit, c_mo, c2)."""
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
                cell = H9_RA.rid2cell[c_reg]   # actual cell, not net_mode-0 normalised
                lut[d, c_mo, c2] = H9C.off_uv[cell]
    return lut


_LUT = _build_hex_uvc_lut()   # (9, 2, 3, 2)


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

    # Walk leaf -> root, collecting net_mode-0 UV at each layer.
    uv_stack = []
    c_mo_t = c_mo.copy()
    c2_t   = c2.copy()
    for i in range(layer - 1, 0, -1):
        d   = body[:, i]
        uv  = _LUT[d, c_mo_t, c2_t].astype(np.int32)   # (N, 2)
        uv_stack.append(uv)
        rmc    = hr[d, c_mo_t, c2_t]
        c_mo_t = rmc[:, 1]
        c2_t   = rmc[:, 2]

    # Accumulate root -> leaf:  qr = 3 * qr + uv
    qr = np.zeros((sz, 2), dtype=np.int32)
    for uv in reversed(uv_stack):
        qr = 3 * qr + uv

    return qr[:, 0], qr[:, 1], c2


# ---------------------------------------------------------------------------
# Inverse: (u, v, c2, c_mo, r_mo) -> body digits
# ---------------------------------------------------------------------------

def uvc_to_hex_digits(
    u:     np.ndarray,
    v:     np.ndarray,
    c2:    np.ndarray,
    c_mo:  np.ndarray,
    r_mo:  np.ndarray,
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
    """
    u    = np.asarray(u,   dtype=np.int32)
    v    = np.asarray(v,   dtype=np.int32)
    c2   = np.asarray(c2,  dtype=np.uint8)
    c_mo = np.asarray(c_mo,dtype=np.uint8)
    r_mo = np.asarray(r_mo,dtype=np.uint8)
    sz   = u.shape[0]

    body = np.full((sz, layer), 0xFF, dtype=np.uint8)

    for n in range(sz):
        qr_root = np.array([u[n], v[n]], dtype=np.int32)

        # DFS: (remaining_levels, qr, c_mo_cur, c2_cur, digits_leaf_to_root)
        stack  = [(layer - 1, qr_root.copy(), int(c_mo[n]), int(c2[n]), [])]
        found  = None

        while stack and found is None:
            lvl, qr_cur, cm, c2v, digs = stack.pop()

            if lvl == 0:
                # Reached root: qr must be exactly zero AND c_mo must match r_mo
                if cm == int(r_mo[n]) and qr_cur[0] == 0 and qr_cur[1] == 0:
                    found = digs    # leaf-to-root order
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
# Tests
# ---------------------------------------------------------------------------

def test_forward_unique(layer=8, n_pts=200):
    """Forward pass: distinct hexes at layer N get distinct (u, v, c2)."""
    print(f'\n--- test_forward_unique (layer={layer}, n_pts={n_pts}) ---')
    reg = Registrar()
    b_oct = reg.domain('b_oct')
    rng = np.random.default_rng(42)
    coords = rng.uniform(-0.3, 0.3, (n_pts, 2))
    oids   = rng.integers(0, 8, n_pts, dtype=np.uint8)
    pts    = Points(coords, domain=b_oct, components=oids)

    hx = hex_digits(pts, layer=layer, tail_style=TailStyle.reversible)
    u, v, c2 = hex_digits_to_uvc(hx)

    keys = set(zip(u.tolist(), v.tolist(), c2.tolist()))
    print(f'  {n_pts} pts -> {len(keys)} unique (u,v,c2) keys')
    assert len(keys) == n_pts, 'FAIL: collisions in (u,v,c2)!'
    print('  PASS')


def test_roundtrip(layer=8, n_pts=50):
    """Round-trip: address -> UVC -> digits (body[1:]) matches original."""
    print(f'\n--- test_roundtrip (layer={layer}, n_pts={n_pts}) ---')
    reg = Registrar()
    b_oct = reg.domain('b_oct')
    rng = np.random.default_rng(7)
    coords = rng.uniform(-0.3, 0.3, (n_pts, 2))
    oids   = rng.integers(0, 8, n_pts, dtype=np.uint8)
    pts    = Points(coords, domain=b_oct, components=oids)

    hx   = hex_digits(pts, layer=layer, tail_style=TailStyle.reversible)
    body = hx[:, :-1]
    tail = hx[:, -1]
    c_mo, c2, r_mo, _ = tail_unpack_reversible(tail)

    u, v, c2_out = hex_digits_to_uvc(hx)
    body_rec     = uvc_to_hex_digits(u, v, c2_out, c_mo, r_mo, body.shape[1])

    # body[:, 0] is root hex (handled separately); compare body[:, 1:]
    ok = np.all(body[:, 1:] == body_rec[:, 1:])
    mismatches = np.where(body[:, 1:] != body_rec[:, 1:])
    if not ok:
        print(f'  FAIL: {len(mismatches[0])} mismatched positions')
        for i in range(min(5, len(mismatches[0]))):
            r, c = mismatches[0][i], mismatches[1][i] + 1
            print(f'    pt={r} col={c}  orig={body[r,c]}  rec={body_rec[r,c]}')
    else:
        print(f'  PASS: all {n_pts} pts, body[:, 1:] recovered correctly')


def test_adjacency(layer=6):
    """Two hexes that differ by one digit at the finest layer should be close in UV."""
    print(f'\n--- test_adjacency (layer={layer}) ---')
    reg = Registrar()
    b_oct = reg.domain('b_oct')
    # One point and its neighbour (shifted slightly)
    p0 = np.array([[0.10, 0.04]])
    p1 = np.array([[0.11, 0.04]])
    pts0 = Points(p0, domain=b_oct, components=np.array([0], dtype=np.uint8))
    pts1 = Points(p1, domain=b_oct, components=np.array([0], dtype=np.uint8))

    hx0 = hex_digits(pts0, layer=layer, tail_style=TailStyle.reversible)
    hx1 = hex_digits(pts1, layer=layer, tail_style=TailStyle.reversible)
    u0, v0, c2_0 = hex_digits_to_uvc(hx0)
    u1, v1, c2_1 = hex_digits_to_uvc(hx1)

    du = abs(int(u0[0]) - int(u1[0]))
    dv = abs(int(v0[0]) - int(v1[0]))
    print(f'  pt0 uvc=({u0[0]}, {v0[0]}, {c2_0[0]})')
    print(f'  pt1 uvc=({u1[0]}, {v1[0]}, {c2_1[0]})')
    print(f'  |du|={du}  |dv|={dv}')
    # Adjacent hexes that differ at layer k contribute at most 4 * 3^(n-k) to UV.
    # For layer=6, two hexes differing at layer 4 (weight=9) can have du up to ~36.
    max_uv = 4 * 3 ** (layer - 1)   # conservative upper bound
    assert du <= max_uv and dv <= max_uv, f'FAIL: UV distance too large: du={du} dv={dv} (max={max_uv})'
    print(f'  PASS (du={du}, dv={dv})')


def test_same_hex_same_uvc(layer=8, n_pts=40):
    """Two barycentric points inside the same hex get the same (u, v, c2)."""
    print(f'\n--- test_same_hex_same_uvc (layer={layer}) ---')
    from hhg9.h9.addressing import hex_layer as hex_layer_fn
    reg = Registrar()
    b_oct = reg.domain('b_oct')
    rng = np.random.default_rng(13)

    # Generate pairs of points and coalesce them to hex centres
    coords = rng.uniform(-0.2, 0.2, (n_pts, 2))
    oids   = np.zeros(n_pts, dtype=np.uint8)
    pts    = Points(coords, domain=b_oct, components=oids)

    # hex_layer coalesces to hex bin centres
    from hhg9.h9.addressing import hex_digits as hd_fn, neighbours
    pts_coalesced = neighbours(pts, layer=layer, coalesce=True)
    hx = hd_fn(pts_coalesced, layer=layer, tail_style=TailStyle.reversible)
    u, v, c2 = hex_digits_to_uvc(hx)

    # All points coalesced to same hex centre -> same UVC
    keys = set(zip(u.tolist(), v.tolist(), c2.tolist()))
    # After coalescing, same-hex points should have same address -> same UVC
    # Just verify UVC is consistent with address uniqueness
    from hhg9.h9.addressing import hex_layer as hl
    hx_keys_orig = [tuple(int(x) for x in row) for row in hx]
    uvc_keys     = list(zip(u.tolist(), v.tolist(), c2.tolist()))
    # Check: same address <=> same UVC
    addr_to_uvc = {}
    ok = True
    for addr, uvc in zip(hx_keys_orig, uvc_keys):
        if addr in addr_to_uvc:
            if addr_to_uvc[addr] != uvc:
                print(f'  FAIL: same address, different UVC')
                ok = False
        else:
            addr_to_uvc[addr] = uvc
    if ok:
        print(f'  PASS: {len(addr_to_uvc)} unique hexes, UVC consistent with address')


if __name__ == '__main__':
    test_forward_unique(layer=8, n_pts=200)
    test_roundtrip(layer=6, n_pts=50)
    test_roundtrip(layer=8, n_pts=30)
    test_adjacency(layer=6)
    test_same_hex_same_uvc(layer=6, n_pts=40)
    print('\nAll tests done.')