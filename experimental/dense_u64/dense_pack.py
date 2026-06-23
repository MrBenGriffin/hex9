# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
EXPERIMENTAL: dense (mixed-radix) 64-bit H9 address packing.

The production uint64 format (hhg9.h9.addressing.hex_pack / 'ua') packs one
hex *nibble* (4 bits) per digit. But the digits are NOT base-16:

    L0 (root) : 0..11   (base 12)
    L1..Ln    : 0..8    (base  9)   <- wastes ~0.83 bits/digit in a nibble
    tail      : 0..13   (single reversible nibble: (p_mo<<3)|(p_c2<<1)|r_mo)

Nibble packing therefore tops out at L14 in 64 bits (15 body nibbles + 1 tail
nibble = 64 bits, ~0.7 m representative). Packing the digits in their true
mixed radix instead reclaims the wasted bits and reaches deeper:

    value = root                       # base 12
    for d in region_digits: value = value*9 + d
    value = value*16 + tail            # tail kept in the low nibble (lossless)

Bit budget (tail in low nibble): 12 * 9^D * 16 < 2^64  ->  D <= 17.

    depth (D region digits)   layer   approx representative resolution
        13                     L13      ~2.2  m   (old nibble 'ua')
        14                     L14      ~0.7  m   (current nibble 'ua')
        15                     L15      ~0.24 m
        17                     L17      ~0.027 m  (densest single u64)

The cost is readability: a dense u64 is no longer a string of hierarchical hex
digits, so you cannot eyeball the hierarchy or truncate to a parent by chopping
characters. See README.md for the bin/sentinel trade-off this implies.

This module is self-contained and read-only with respect to the library: it
reuses hex_digits (encode) and hex_digits_reg (decode) so the recovered cell is
identical to the production path — only the bit packing differs.
"""
from __future__ import annotations

import numpy as np
from hhg9 import Registrar, Points
from hhg9.h9.addressing import hex_digits, hex_digits_reg, H9_RA
from hhg9.h9.tail import TailStyle
import hhg9.h9.region as rg

DEFAULT_DEPTH = 15   # L15, ~0.24 m; bump up to 17 for ~27 mm
MAX_DEPTH = 17       # 12 * 9^17 * 16 < 2^64


def dense_pack(pts: Points, depth: int = DEFAULT_DEPTH, scheme=H9_RA) -> np.ndarray:
    """Pack b_oct Points into a single dense uint64 each (mixed-radix)."""
    if not (1 <= depth <= MAX_DEPTH):
        raise ValueError(f"depth must be 1..{MAX_DEPTH}, got {depth}")
    hx = np.asarray(hex_digits(pts, layer=depth, tail_style=TailStyle.reversible,
                               scheme=scheme), dtype=np.uint8)
    body = hx[:, :-1]                       # (N, depth+1): root + depth region digits
    tail = hx[:, -1].astype(np.uint64)
    if body.shape[1] != depth + 1:
        raise ValueError(f"expected {depth + 1} body digits, got {body.shape[1]}")
    if body[:, 0].max() > 11 or (depth and body[:, 1:].max() > 8):
        raise ValueError("digit out of range (root must be 0..11, regions 0..8)")
    v = body[:, 0].astype(np.uint64)        # root, base 12
    for j in range(1, depth + 1):
        v = v * np.uint64(9) + body[:, j].astype(np.uint64)   # region digits, base 9
    return (v << np.uint64(4)) | (tail & np.uint64(0x0F))     # tail in low nibble


def dense_unpack(words, depth: int = DEFAULT_DEPTH, reg=None, scheme=H9_RA) -> Points:
    """Invert dense_pack back to b_oct Points (fixed `depth`)."""
    if reg is None:
        reg = Registrar()
    dom = reg.domain('b_oct')
    v = np.asarray(words, dtype=np.uint64).reshape(-1).copy()
    tail = (v & np.uint64(0x0F)).astype(np.uint8)
    v = v >> np.uint64(4)
    region = np.zeros((v.shape[0], depth), dtype=np.uint8)
    for j in range(depth - 1, -1, -1):
        region[:, j] = (v % np.uint64(9)).astype(np.uint8)
        v = v // np.uint64(9)
    root = v.astype(np.uint8)               # 0..11
    hx = np.column_stack([root, region, tail])
    oc, cells = hex_digits_reg(dom, hx[:, :-1], tail=hx[:, -1], scheme=scheme)
    xy = rg.regions_xy(cells)
    return Points(xy[:, :2], domain=dom, oid=oc)


if __name__ == '__main__':
    from hhg9.algorithms.distance import wgs84

    reg = Registrar()
    b_oct = reg.domain('b_oct'); g_gcd = reg.domain('g_gcd')
    lat = np.linspace(-85, 85, 40); lon = np.linspace(-179, 179, 40)
    LA, LO = np.meshgrid(lat, lon)
    pos = Points(np.column_stack([LA.ravel(), LO.ravel()]), g_gcd)
    bry = reg.project(pos, [g_gcd, b_oct])

    print(f"sample N={len(bry)}   (2^64 = {2**64})")
    print(f"{'depth':>5} {'max u64':>20} {'fits':>5} {'max err (m)':>12}")
    for D in (13, 14, 15, 16, 17):
        w = dense_pack(bry, D)
        gcd = reg.project(dense_unpack(w, D, reg), [b_oct, g_gcd])
        err = wgs84(pos.coords, gcd.coords).max()
        print(f"{D:>5} {int(w.max()):>20} {str(int(w.max()) < 2**64):>5} {err:>12.4f}")
