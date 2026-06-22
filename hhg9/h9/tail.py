# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""
H9 Hex Address Tail class and helper functions.
There  are two main tail styles - reversible and key
They serve two separate purposes:
1.  **Reversible** tail is used to encode the geometry of the hex.
    It is invertible and can be used to reconstruct the geometry of the hex.
    One can convert a reversible address to latitude/longitude coordinates.
    However, there are multiple ways to encode the same geometry.
2.  **Key** tail is used to encode the geometry of the hex.
    It is not invertible and cannot be used to reconstruct the geometry of the hex.
    Key tails are used to guarantee uniqueness of the hex, for binning.
    This is needed because the reversible tail is not unique enough for some applications.
"""
from __future__ import annotations
from enum import unique, Enum
import numpy as np
from numpy.typing import NDArray


# --- Hex Address TailStyle enum and helper ---
@unique
class TailStyle(Enum):
    """How the final tail byte is encoded."""
    reversible = 0  # full metadata tail (invertible)
    key = 1  # short tail (binning/grouping)
    none = 2  # omit tail entirely


# ---------- Hex Address Tail packing helpers ----------------------------------------
# Single-nibble tail layout (matches libhex9 key_tail, so packed UUIDs are
# byte-compatible across the Python and C implementations):
#     bit 3 : p_mo  (parent/terminal net_mode; 0 for canonical addresses)
#     bits 2..1 : p_c2 (terminal c2, 0..2 — the c2=3 codepoint is unused)
#     bit 0 : r_mo  (root/octant net_mode)
# The reversible style stores p_mo; the key style drops it (bit 3 = 0, a spare).
# After canonicalisation p_mo is always 0, so the two styles emit identical bytes.

def tail_pack_reversible(
        r_mo: NDArray[np.uint8] | np.uint8,  # root region net_mode      -> bit 0
        p_mo: NDArray[np.uint8] | np.uint8,  # terminal parent net_mode  -> bit 3
        p_c2: NDArray[np.uint8] | np.uint8,  # terminal c2 (0..2)        -> bits 1..2
) -> NDArray[np.uint8]:
    """Pack the reversible tail nibble: (p_mo << 3) | (p_c2 << 1) | r_mo."""
    r_mo = np.asarray(r_mo, dtype=np.uint8)
    p_mo = np.asarray(p_mo, dtype=np.uint8)
    p_c2 = np.asarray(p_c2, dtype=np.uint8)
    return (((p_mo & 0x01) << 3) | ((p_c2 & 0x03) << 1) | (r_mo & 0x01)).astype(np.uint8)


def tail_unpack_reversible(short_tail: NDArray[np.uint8] | np.uint8):
    """Unpack the reversible tail nibble -> (p_c2, r_mo, p_mo). Mirror of pack."""
    short_tail = np.asarray(short_tail, dtype=np.uint8)
    p_c2 = ((short_tail >> 1) & 0x03).astype(np.uint8)
    r_mo = (short_tail & 0x01).astype(np.uint8)
    p_mo = ((short_tail >> 3) & 0x01).astype(np.uint8)
    return p_c2, r_mo, p_mo


def tail_pack_key(
        r_mo: NDArray[np.uint8] | np.uint8,  # root region net_mode  -> bit 0
        p_mo: NDArray[np.uint8] | np.uint8,  # unused for keys (bit 3 stays 0)
        p_c2: NDArray[np.uint8] | np.uint8,  # terminal c2 (0..2)    -> bits 1..2
) -> NDArray[np.uint8]:
    """Pack the binning key tail nibble: (p_c2 << 1) | r_mo (p_mo dropped, bit 3 = 0)."""
    r_mo = np.asarray(r_mo, dtype=np.uint8)
    p_c2 = np.asarray(p_c2, dtype=np.uint8)
    return (((p_c2 & 0x03) << 1) | (r_mo & 0x01)).astype(np.uint8)


def tail_unpack_key(short_tail: NDArray[np.uint8] | np.uint8):
    """Unpack the key tail nibble -> (p_c2, r_mo). Mirror of tail_pack_key."""
    short_tail = np.asarray(short_tail, dtype=np.uint8)
    p_c2 = ((short_tail >> 1) & 0x03).astype(np.uint8)
    r_mo = (short_tail & 0x01).astype(np.uint8)
    return p_c2, r_mo


def tail_key_from_reversible(tail_ids: NDArray[np.uint8] | np.uint8) -> NDArray[np.uint8]:
    """Derive the key tail from a reversible tail: drop p_mo (bit 3)."""
    tail_ids = np.asarray(tail_ids, dtype=np.uint8)
    return (tail_ids & 0x07).astype(np.uint8)
