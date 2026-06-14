# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
H9 UUID Address Encoding.

Provides a stable 128-bit (UUID) representation of a hex9 address at maximum
practical depth, plus a companion byte that enables exact round-trip to lat/lon.

Layout (32 nibbles = 128 bits):
    nibbles 0..29 : hex body (L0..L29) — hierarchical hex digits
    byte 15       : tail byte, split as nibble 30 (high) and nibble 31 (low)
      bits [7..4] : h    — terminal region id (0..11). 0xF = bin, 0xE = key
      bit  [3]    : p_mo — parent's mode (or 0 if key)
      bits [2..1] : p_c2 — parent's c2 orientation (0..2)
      bit  [0]    : r_mo — root region mode
    bin preserves the low nibble (p_mo, p_c2, r_mo) of a key UUID.

    h    = ((adr_byte >> 4) & 0x0F).astype(np.uint8)
    p_mo = ((adr_byte >> 3) & 0x01).astype(np.uint8)
    p_c2 = ((adr_byte >> 1) & 0x03).astype(np.uint8)
    r_mo =  (adr_byte       & 0x01).astype(np.uint8)

The UUID alone is sufficient for spatial indexing and hexbin at any layer.
The (UUID, adr) pair enables exact reconstruction of the b_oct coordinate
and therefore round-trip to lat/lon.

Public API:
    h9_encode(lats, lons)              -> (uuids, adr_bytes)
    h9_decode(uuids)                   -> (lats, lons)
    h9_bin(uuids, layer)               -> uuids
"""

from __future__ import annotations

import uuid as uuid_mod
import numpy as np
from numpy.typing import NDArray
from typing import TYPE_CHECKING

from hhg9.h9.region import xy_regions

if TYPE_CHECKING:
    from hhg9 import Points, Registrar

from hhg9.h9.addressing import hex_digits, hex_decode, H9_RA, hex_digits_reg, reg_hex_digits
from hhg9.h9.tail import TailStyle, tail_pack_reversible, tail_unpack_reversible
from hhg9.h9.protocols import RegionAddressLike

# Public API also includes h9_postgis_hexagons (defined at bottom of module).

# The layer parameter passed to hex_digits that yields 32 nibbles:
#   xy_regions(depth=30) -> addresses shape (N, 32)
#   reg_hex_digits sees cols=32, depth=31 -> bdy has 30 cols (L0...L29)
#   key pack: 30 body nibbles + 2 key_tail nibbles = 32 nibbles = 128 bits
UUID_DEPTH: int = 29


# ---------- 128-bit packing helpers ----------------------------------------

def _nibbles_to_int(nibbles: NDArray[np.uint8]) -> int:
    """Pack a (32,) uint8 nibble row into a Python int, MSB-first."""
    hi = nibbles[0::2].astype(np.uint8)
    lo = nibbles[1::2].astype(np.uint8)
    return int.from_bytes(bytes((hi << 4) | lo), 'big')


def batch_nibbles_to_int(nibbles: NDArray[np.uint8]) -> list[int]:
    """Pack (N, 32) uint8 nibbles into N Python ints, MSB-first."""

    # Force 1D arrays of shape (32,) into 2D arrays of shape (1, 32)
    nibbles = nibbles.reshape(-1, 32)
    hi = nibbles[:, 0::2].astype(np.uint8)
    lo = nibbles[:, 1::2].astype(np.uint8)
    byte_arr = (hi << 4) | lo  # (N, 16)

    return [int.from_bytes(bytes(row), 'big') for row in byte_arr]


def _batch_int_to_nibbles(values: list[int], n: int = 32) -> NDArray[np.uint8]:
    """Unpack N Python ints into (N, n) uint8 nibbles, MSB-first."""
    n_bytes = n // 2  # n is always even (32)
    result = np.zeros((len(values), n), dtype=np.uint8)
    for i, val in enumerate(values):
        b = val.to_bytes(16, 'big')[-n_bytes:]  # rightmost n_bytes of 128-bit int
        for j, byte in enumerate(b):
            result[i, 2 * j] = (byte >> 4) & 0xF
            result[i, 2 * j + 1] = byte & 0xF
    return result


# ---------- Core encode/decode ---------------------------------------------
#     dom = pts.domain
#     oc, mo = pts.cm()
#     cx = rg.xy_regions(pts.coords, mo, layer)  # regions are length 2+'depth'
#     return reg_hex_digits(cx, oc, dom, tail_style, scheme=scheme)
def h9_enc(
        b_pts,
        scheme: RegionAddressLike = H9_RA,
) -> list[uuid_mod.UUID]:
    """
    Encode b_oct Points to H9 UUID addresses.

    The caller is responsible for projecting to b_oct beforehand.
    For a lat/lon convenience wrapper see h9_encode().

    Parameters
    ----------
    b_pts  : Points in b_oct domain
    scheme : RegionAddressLike (normally H9_RA)

    Returns
    -------
    uuids     : list[uuid.UUID]  — 128-bit key addresses, one per point
    """
    hx = hex_digits(b_pts, layer=UUID_DEPTH, tail_style=TailStyle.reversible, scheme=scheme)
    body = hx[:, :-1]   # (N, 30): L0...L29 as nibble values
    tail_byte = hx[:, -1]
    tail_n = np.stack([(tail_byte >> 4) & 0x0F, tail_byte & 0x0F], axis=1)  # (N, 2)
    uuid_nibbles = np.concatenate([body, tail_n], axis=1)
    return np.array([uuid_mod.UUID(int=v) for v in batch_nibbles_to_int(uuid_nibbles)])


def h9_enc_ext(b_pts, oc, mo) -> list[uuid_mod.UUID]:
    """
    Encode b_oct Points to H9 UUID addresses.
    The caller is responsible for projecting to b_oct beforehand.
    For a lat/lon convenience wrapper see h9_encode().

    Parameters
    ----------
    b_pts  : Points in b_oct domain
    scheme : RegionAddressLike (normally H9_RA)
    Returns
    -------
    uuids     : list[uuid.UUID]  — 128-bit key addresses, one per point
    """
    from hhg9.h9.region import xy_regions
    xy_r = xy_regions(b_pts.coords, mo, UUID_DEPTH)  # regions are length 2+'depth'
    hx = reg_hex_digits(xy_r, oc, b_pts.domain)   # TailStyle.reversible = default.
    body = hx[:, :-1]   # (N, 30): L0...L29 as nibble values
    tail_byte = hx[:, -1]
    tail_n = np.stack([(tail_byte >> 4) & 0x0F, tail_byte & 0x0F], axis=1)  # (N, 2)
    uuid_nibbles = np.concatenate([body, tail_n], axis=1)
    return np.array([uuid_mod.UUID(int=v) for v in batch_nibbles_to_int(uuid_nibbles)])


def h9_dec(
        uuids: list[uuid_mod.UUID],
        b_oct,
        scheme: RegionAddressLike = H9_RA,
) -> 'Points':
    """
    Decode H9 (UUID, adr) pairs back to b_oct Points.

    The caller is responsible for projecting onward from b_oct.
    For a lat/lon convenience wrapper see h9_decode().

    Parameters
    ----------
    uuids     : list[uuid.UUID] — key UUIDs from h9_enc
    adr_bytes : array-like uint8 — companion adr bytes from h9_enc
    b_oct     : b_oct domain instance
    scheme    : RegionAddressLike (normally H9_RA)

    Returns
    -------
    Points in b_oct domain
    """
    import hhg9.h9.region as rg
    from hhg9 import Points
    uuid_ints = [u.int for u in uuids]
    uuid_nibbles = _batch_int_to_nibbles(uuid_ints, n=32)   # (N, 32)
    key_nibbles = uuid_nibbles[:, -2:]
    key_tail = (key_nibbles[:, 0] << 4) + key_nibbles[:, 1]
    body = uuid_nibbles[:, :-2]                           # (N, 30) nibble values
    oc, cells = hex_digits_reg(b_oct, body, tail=key_tail)
    xy_m = rg.regions_xy(cells)
    return Points(xy_m[:, :2], domain=b_oct, oid=oc)


def h9_encode(
        lats,
        lons,
        reg=None,
        scheme: RegionAddressLike = H9_RA,
) -> tuple[list[uuid_mod.UUID], NDArray[np.uint8]]:
    """
    Encode geographic coordinates to H9 UUID addresses.

    Convenience wrapper: projects lat/lon → b_oct then calls h9_enc().
    Prefer calling h9_enc() directly when b_oct Points are already available.

    Parameters
    ----------
    lats : array-like, degrees, WGS84 geodetic latitude
    lons : array-like, degrees, WGS84 geodetic longitude
    reg  : Registrar (created if None)
    scheme : RegionAddressLike (normally H9_RA)

    Returns
    -------
    uuids : list[uuid.UUID]  — 128-bit key addresses, one per point
    adr_bytes : NDArray[uint8] — companion byte enabling round-trip decode,
                packed as (p_mo << 4) | h
    """
    from hhg9 import Registrar, Points

    lats = np.asarray(lats, dtype=np.float64).ravel()
    lons = np.asarray(lons, dtype=np.float64).ravel()
    if lats.shape != lons.shape:
        raise ValueError("lats and lons must have the same length")

    if reg is None:
        reg = Registrar()

    g_gcd = reg.domain('g_gcd')
    b_oct = reg.domain('b_oct')

    coords = np.column_stack([lats, lons])
    b_pts = reg.project(Points(coords, g_gcd), [g_gcd, b_oct])
    return h9_enc(b_pts, scheme=scheme)


def h9_decode(
        uuids: list[uuid_mod.UUID],
        reg=None,
        scheme: RegionAddressLike = H9_RA,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Decode H9 UUIDs back to geographic coordinates.

    Convenience wrapper: calls h9_dec() then projects b_oct → lat/lon.
    Prefer calling h9_dec() directly when b_oct output is sufficient.

    Parameters
    ----------
    uuids  : list[uuid.UUID] — key UUIDs from h9_encode / h9_enc
    reg    : Registrar (created if None)
    scheme : RegionAddressLike (normally H9_RA)

    Returns
    -------
    lats : NDArray[float64], geodetic latitude  (degrees)
    lons : NDArray[float64], geodetic longitude (degrees)
    """
    from hhg9 import Registrar

    if reg is None:
        reg = Registrar()

    b_oct = reg.domain('b_oct')
    g_gcd = reg.domain('g_gcd')

    b_pts = h9_dec(uuids, b_oct, scheme=scheme)
    g_pts = reg.project(b_pts, [b_oct, g_gcd])

    lats = g_pts.coords[:, 0]
    lons = g_pts.coords[:, 1]
    return lats, lons


def h9_bin_pts(b_pts: Points, layer: int):
    """Given a b_oct Points, return the layer-L bin UUIDs."""
    # hex_dig = hex_digits(pts, layer=layer, tail_style=TailStyle.key)
    # dom = b_pts.domain
    oc, mo = b_pts.cm()
    xy_r = xy_regions(b_pts.coords, mo, layer)  # regions are length 2+'depth'
    hx = reg_hex_digits(xy_r, oc, b_pts.domain, TailStyle.key)   # TailStyle.reversible = default.
    body = hx[:, :-1]   # (N, 30): L0...L29 as nibble values
    tail_byte = hx[:, -1]
    # tail_n = np.stack([(tail_byte >> 4) & 0x0F, tail_byte & 0x0F], axis=1)  # (N, 2)
    uuid_nibs = np.full((len(body), 32), 0x0F, dtype=np.uint8)
    uuid_nibs[:, :layer + 1] = body                      # body at layer L
    uuid_nibs[:, -1] = tail_byte & 0x0F                  # key always at nibble 31
    uuid_nibs[:, -2] = (tail_byte >> 4) & 0x0F           # key always at nibble 30
    return np.array([uuid_mod.UUID(int=v) for v in batch_nibbles_to_int(uuid_nibs)])


def h9_bin(
        uuids: list[uuid_mod.UUID],
        layer: int,
        reg=None,
        scheme: RegionAddressLike = H9_RA,
) -> list[uuid_mod.UUID]:
    """
    Return the layer-L bin UUID for each H9 UUID address.

    The bin UUID has the same 128-bit format: nibbles 0..layer carry the
    hierarchical bin key; nibbles layer+1..30 are 0x0F (OOB sentinel, unambiguous
    since valid digits are 0-11 at L0 and 0-8 at L1+); nibble 31 carries the key
    tail. Bin UUIDs at the same layer can be compared directly for equality and sorted.

    The adr companion byte is NOT required — the key UUID alone contains sufficient
    information to identify the correct bin at any layer <= UUID_DEPTH.

    Parameters
    ----------
    uuids : list[uuid.UUID] — key UUIDs from h9_encode
    layer : int — target hex layer (0..UUID_DEPTH)
    reg   : Registrar (created if None)

    Returns
    -------
    list[uuid.UUID] — bin-key UUIDs at the requested layer
    """
    if not (0 <= layer <= UUID_DEPTH):
        raise ValueError(f"layer must be in 0..{UUID_DEPTH}, got {layer}")
    if reg is None:
        from hhg9 import Registrar
        reg = Registrar()
    b_oct = reg.domain('b_oct')
    pts = h9_dec(uuids, b_oct)
    return h9_bin_pts(pts, layer)


# ---------- Human-readable label format ------------------------------------
# Format:  '<body>.<key>'
#   body : hex chars for each address nibble, stopping before the first 0x0F sentinel
#   key  : single hex char for the key nibble at position 31
# Example: UUID for a layer-5 hex → '32343.2'
# Full-depth (layer 30) addresses have no sentinel so all 31 body nibbles appear.

def h9_label(u: uuid_mod.UUID) -> str:
    """Convert an H9 UUID to a human-readable label string.

    Format: '<body>.<key>'  e.g. '32343.2'
    The body is the hex-digit address up to (not including) the first 0x0F
    sentinel nibble.  The key nibble (nibble 31) follows the dot.
    """
    nibs = _batch_int_to_nibbles([u.int], n=32)[0]          # (32,) uint8
    body_nibs = nibs[:31]
    sentinels = np.where(body_nibs == 0x0F)[0]
    stop = int(sentinels[0]) if len(sentinels) else 31       # full-depth → all 31
    body_str = ''.join(format(int(v), 'x') for v in body_nibs[:stop])
    return f'{body_str}.{format(int(nibs[31]), "x")}'


def h9_from_label(label: str) -> uuid_mod.UUID:
    """Roundtrip: convert an h9_label string back to a UUID.

    Accepts the '<body>.<key>' format produced by h9_label().
    """
    body_str, key_str = label.split('.')
    body_nibs = [int(c, 16) for c in body_str]
    L = len(body_nibs) - 1                                   # layer index
    nibs = np.zeros(32, dtype=np.uint8)
    nibs[:len(body_nibs)] = body_nibs
    if L + 1 < 31:
        nibs[L + 1:31] = 0x0F                               # restore OOB sentinel
    nibs[31] = int(key_str, 16)
    return uuid_mod.UUID(int=_nibbles_to_int(nibs))
