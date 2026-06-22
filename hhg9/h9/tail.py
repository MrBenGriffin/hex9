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
def tail_pack_reversible(
        r_mo: NDArray[np.uint8] | np.uint8,  # root region net_mode
        p_mo: NDArray[np.uint8] | np.uint8,  # [0,1]   terminating hex net_mode of parent region
        p_c2: NDArray[np.uint8] | np.uint8,  # terminating hex c2 of parent region
) -> NDArray[np.uint8]:
    """Pack reversible tail into one uint8 nibble.

    Layout: high nibble = 0xF (bin/key sentinel — h slot, never a real 0..11
    value), low nibble = (p_c2 << 1) | r_mo. Matches uuid_address.py
    docstring's "F: bin" placement and the C++ bin emission.
    """
    r_mo = np.asarray(r_mo, dtype=np.uint8)
    p_mo = np.asarray(p_mo, dtype=np.uint8)
    p_c2 = np.asarray(p_c2, dtype=np.uint8)
    tail = (((p_c2 & 0x03) << 2) | ((r_mo & 0x01) << 1) | (p_mo & 0x01)).astype(np.uint8)
    return tail

def tail_unpack_reversible(short_tail: NDArray[np.uint8] | np.uint8):
    """Unpack reversible tail

    Layout must match `tail_pack_key`:
      bits 7..4: sentinel (0xF) # old
      bits 3:    p_mo # new
      bits 2..1: p_c2
      bit  0:    r_mo
    """
    # Must mirror tail_pack_key: tail = (p_c2 << 2) | (r_mo << 1) | p_mo
    short_tail = np.asarray(short_tail, dtype=np.uint8)
    p_c2 = ((short_tail >> 2) & 0x03).astype(np.uint8)
    r_mo = ((short_tail >> 1) & 0x01).astype(np.uint8)
    p_mo = (short_tail & 0x01).astype(np.uint8)
    return p_c2, r_mo, p_mo


def tail_pack_key(
        r_mo: NDArray[np.uint8] | np.uint8,  # root region net_mode
        p_mo: NDArray[np.uint8] | np.uint8,  # [0,1]   terminating hex net_mode of parent region
        p_c2: NDArray[np.uint8] | np.uint8,  # terminating hex c2 of parent region
) -> NDArray[np.uint8]:
    """Pack key tail (binning-safe) into one uint8 byte.

    Layout: high nibble = 0xF (bin/key sentinel — h slot, never a real 0..11
    value), low nibble = (p_c2 << 1) | r_mo. Matches uuid_address.py
    docstring's "F: bin" placement and the C++ bin emission.
    """
    r_mo = np.asarray(r_mo, dtype=np.uint8)
    # p_mo = np.asarray(p_mo, dtype=np.uint8)
    p_c2 = np.asarray(p_c2, dtype=np.uint8)
    tail = (((p_c2 & 0x03) << 2) | ((r_mo & 0x01) << 1)).astype(np.uint8)
    return tail


def tail_unpack_key(short_tail: NDArray[np.uint8] | np.uint8):
    """Unpack key tail into (p_c2, r_mo).

    Layout must match `tail_pack_key`:
      bits 7..4: sentinel (0xF) # old
      bits 3:    p_mo # new
      bits 2..1: p_c2
      bit  0:    r_mo
    """
    # Must mirror tail_pack_key: tail = (p_c2 << 2) | (r_mo << 1) | p_mo
    short_tail = np.asarray(short_tail, dtype=np.uint8)
    p_c2 = ((short_tail >> 2) & 0x03).astype(np.uint8)
    r_mo = ((short_tail >> 1) & 0x01).astype(np.uint8)
    # p_mo = (short_tail & 0x01).astype(np.uint8)
    return p_c2, r_mo


def tail_key_from_reversible(tail_ids: NDArray[np.uint8] | np.uint8) -> NDArray[np.uint8]:
    """Derive key tail from reversible tail without recomputing geometry."""
    tail_ids = np.asarray(tail_ids, dtype=np.uint8)
    return (tail_ids & 0x0E).astype(np.uint8)
