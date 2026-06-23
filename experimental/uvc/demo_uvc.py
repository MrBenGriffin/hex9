# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""
Demo: H9 address <-> UVC axial hexgrid coordinate.

(u, v) is the accumulated net_mode-aware UV lattice coordinate — each layer
contributes off_uv[actual_cell] * 3^(depth - layer).  c2 is the leaf
half-hex selector from the reversible tail.

Demonstrates:
  1. Forward: hex address → (u, v, c2)
  2. Inverse: (u, v, c2) + tail metadata → hex digits
  3. Spatial ordering: nearby points → nearby UV
  4. A small UV neighbourhood grid
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
# Build LUTs (same as test_uvc.py)
# ---------------------------------------------------------------------------

def _build_hex_uvc_lut() -> np.ndarray:
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


def _build_reverse_map(lut) -> dict:
    hr = HEX_LUTS.hex_reg
    oob = HEX_LUTS.hex_oob
    rev = {}
    for d in range(9):
        for c_mo in range(2):
            for c2 in range(3):
                uv = lut[d, c_mo, c2]
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


_LUT = _build_hex_uvc_lut()
_REV = _build_reverse_map(_LUT)


# ---------------------------------------------------------------------------
# Forward: hx array (N, layer+2) → (u, v, c2)
# ---------------------------------------------------------------------------

def hex_to_uvc(hx: np.ndarray):
    hx = np.asarray(hx, dtype=np.uint8)
    body = hx[:, :-1]        # (N, layer+1): octant col 0, digits cols 1..layer
    tail = hx[:, -1]

    sz, n_cols = body.shape
    c_mo, c2, _r_mo, _ = tail_unpack_reversible(tail)
    hr = HEX_LUTS.hex_reg

    c_mo_t, c2_t = c_mo.copy(), c2.copy()
    uv_stack = []
    for i in range(n_cols - 1, 0, -1):
        d  = body[:, i]
        uv = _LUT[d, c_mo_t, c2_t].astype(np.int32)
        uv_stack.append(uv)
        rmc   = hr[d, c_mo_t, c2_t]
        c_mo_t = rmc[:, 1]
        c2_t   = rmc[:, 2]

    qr = np.zeros((sz, 2), dtype=np.int32)
    for uv in reversed(uv_stack):
        qr = 3 * qr + uv

    return qr[:, 0], qr[:, 1], c2


# ---------------------------------------------------------------------------
# Inverse: (u, v, c2, c_mo, r_mo, n_body_cols) → body digits
# ---------------------------------------------------------------------------

def uvc_to_hex(u, v, c2, c_mo, r_mo, n_body_cols: int) -> np.ndarray:
    u    = np.asarray(u,    dtype=np.int32)
    v    = np.asarray(v,    dtype=np.int32)
    c2   = np.asarray(c2,   dtype=np.uint8)
    c_mo = np.asarray(c_mo, dtype=np.uint8)
    r_mo = np.asarray(r_mo, dtype=np.uint8)
    sz   = u.shape[0]

    body = np.full((sz, n_body_cols), 0xFF, dtype=np.uint8)

    for n in range(sz):
        qr_root = np.array([u[n], v[n]], dtype=np.int32)
        stack  = [(n_body_cols - 1, qr_root.copy(), int(c_mo[n]), int(c2[n]), [])]
        found  = None

        while stack and found is None:
            lvl, qr_cur, cm, c2v, digs = stack.pop()

            if lvl == 0:
                if cm == int(r_mo[n]) and qr_cur[0] == 0 and qr_cur[1] == 0:
                    found = digs
                continue

            u_mod3 = int(qr_cur[0]) % 3
            for d, p_mo, p_c2, uv in _REV.get((cm, c2v, u_mod3), []):
                rem = qr_cur - np.array(uv, dtype=np.int32)
                if rem[0] % 3 == 0 and rem[1] % 3 == 0:
                    stack.append((lvl - 1, rem // 3, p_mo, p_c2, digs + [d]))

        if found is not None:
            for k, d in enumerate(reversed(found)):
                body[n, k + 1] = d
        else:
            print(f'  [WARN] no path found for pt {n}')

    return body


# ---------------------------------------------------------------------------
# Demo 1 – print (u, v, c2) for a handful of addresses
# ---------------------------------------------------------------------------

def demo_coordinates(layer=6, n_pts=8, seed=42):
    print(f'\n{"="*60}')
    print(f'Demo 1: H9 address -> (u, v, c2)  [layer={layer}]')
    print('='*60)

    reg = Registrar()
    b_oct = reg.domain('b_oct')
    rng = np.random.default_rng(seed)
    coords = rng.uniform(-0.3, 0.3, (n_pts, 2))
    oids   = rng.integers(0, 8, n_pts, dtype=np.uint8)
    pts    = Points(coords, domain=b_oct, components=oids)
    hx     = hex_digits(pts, layer=layer, tail_style=TailStyle.reversible)

    u, v, c2 = hex_to_uvc(hx)

    print(f'{"oct":>3} {"body digits":>20}  =>  {"u":>8} {"v":>8} {"c2":>2}')
    print('-'*55)
    for i in range(n_pts):
        body_str = ' '.join(str(d) for d in hx[i, 1:-1])
        print(f'{hx[i,0]:>3}  {body_str:>20}   {u[i]:>8} {v[i]:>8} {c2[i]:>2}')


# ---------------------------------------------------------------------------
# Demo 2 – spatial ordering: nearby lon/lat → nearby (u, v)
# ---------------------------------------------------------------------------

def demo_spatial_ordering(layer=7):
    print(f'\n{"="*60}')
    print(f'Demo 2: spatial locality in UV  [layer={layer}]')
    print('='*60)

    reg = Registrar()
    g_gcd = reg.domain('g_gcd')
    b_oct = reg.domain('b_oct')

    # A small 3×3 grid of lon/lat points around London
    lons = np.array([-0.2, 0.0, 0.2])
    lats = np.array([51.4, 51.5, 51.6])
    gx, gy = np.meshgrid(lons, lats)
    lonlat = np.column_stack([gx.ravel(), gy.ravel()])

    from hhg9 import Points
    geo_pts  = Points(lonlat, domain=g_gcd)
    bary_pts = reg.project(geo_pts, [g_gcd, b_oct])
    hx       = hex_digits(bary_pts, layer=layer, tail_style=TailStyle.reversible)
    u, v, c2 = hex_to_uvc(hx)

    print(f'{"lon":>6} {"lat":>6}  ->  {"u":>7} {"v":>7} {"c2":>2}')
    print('-'*40)
    for i in range(len(lonlat)):
        print(f'{lonlat[i,0]:>6.2f} {lonlat[i,1]:>6.2f}   {u[i]:>7} {v[i]:>7} {c2[i]:>2}')

    # UV range
    print(f'\n  u range: {u.min()} .. {u.max()}  (span {u.max()-u.min()})')
    print(f'  v range: {v.min()} .. {v.max()}  (span {v.max()-v.min()})')
    # Note: London (lon≈0) sits on an octant boundary.  Points west fall in one
    # octant (v≈-2177) and points east fall in the adjacent octant (v≈+2186),
    # so v is not continuous across the boundary — by design.
    # The u axis (N-S) stays compact across all 9 points.
    max_uv = 3**layer - 1   # theoretical max accumulated UV magnitude
    print(f'  Octant UV span at layer {layer}: ±{max_uv}  '
          f'(v jump = one full octant diameter)')


# ---------------------------------------------------------------------------
# Demo 3 – round-trip verification
# ---------------------------------------------------------------------------

def demo_roundtrip(layer=7, n_pts=20):
    print(f'\n{"="*60}')
    print(f'Demo 3: round-trip  address -> UVC -> address  [layer={layer}]')
    print('='*60)

    reg = Registrar()
    b_oct = reg.domain('b_oct')
    rng = np.random.default_rng(99)
    coords = rng.uniform(-0.25, 0.25, (n_pts, 2))
    oids   = rng.integers(0, 8, n_pts, dtype=np.uint8)
    pts    = Points(coords, domain=b_oct, components=oids)

    hx   = hex_digits(pts, layer=layer, tail_style=TailStyle.reversible)
    body = hx[:, :-1]
    tail = hx[:, -1]
    c_mo, c2, r_mo, _ = tail_unpack_reversible(tail)

    u, v, c2_out = hex_to_uvc(hx)
    body_rec = uvc_to_hex(u, v, c2_out, c_mo, r_mo, body.shape[1])

    ok = np.all(body[:, 1:] == body_rec[:, 1:])
    if ok:
        print(f'  PASS: all {n_pts} addresses recovered exactly from (u, v, c2).')
    else:
        mis = np.where(body[:, 1:] != body_rec[:, 1:])
        print(f'  FAIL: {len(mis[0])} mismatches')


# ---------------------------------------------------------------------------
# Demo 4 – UV neighbourhood: show which digits are adjacent in UV
# ---------------------------------------------------------------------------

def demo_uv_neighbourhood(layer=5, seed=7):
    print(f'\n{"="*60}')
    print(f'Demo 4: UV neighbourhood  [layer={layer}]')
    print('='*60)
    print('  One seed hex and its UV-nearest neighbours from a random sample.')

    reg = Registrar()
    b_oct = reg.domain('b_oct')
    rng = np.random.default_rng(seed)
    coords = rng.uniform(-0.1, 0.1, (500, 2))
    oids   = np.zeros(500, dtype=np.uint8)      # single octant for clarity
    pts    = Points(coords, domain=b_oct, components=oids)

    hx     = hex_digits(pts, layer=layer, tail_style=TailStyle.reversible)
    u, v, c2 = hex_to_uvc(hx)

    # Pick first hex as reference
    u0, v0 = int(u[0]), int(v[0])
    du = np.abs(u - u0).astype(np.int32)
    dv = np.abs(v - v0).astype(np.int32)
    dist = du + dv

    order = np.argsort(dist)[:8]
    print(f'\n  Reference: (u={u0}, v={v0})')
    print(f'  {"u":>7} {"v":>7} {"du":>4} {"dv":>4}  body[1:]')
    print('  ' + '-'*50)
    for i in order:
        body_str = ' '.join(str(d) for d in hx[i, 1:-1])
        print(f'  {u[i]:>7} {v[i]:>7} {du[i]:>4} {dv[i]:>4}  {body_str}')


if __name__ == '__main__':
    demo_coordinates(layer=6)
    demo_spatial_ordering(layer=7)
    demo_roundtrip(layer=7, n_pts=30)
    demo_uv_neighbourhood(layer=5)
    print('\nDone.')
