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
#   WAS 29
UUID_DEPTH: int = 30

# Hexagon centroid offset in lattice units, indexed [mode][c2] (see polygon.py).
# Mode 0: c2 0,1,2 -> (1,1),(1,-1),(-2,0); mode 1 is the same set rotated.
# Used by h9_dec to seed regions_xy at the cell centroid (scaled by Ü).
_HEX_CENTROID = np.array([[(1, 1), (1, -1), (-2, 0)],
                          [(1, -1), (-2, 0), (1, 1)]], dtype=np.float64)


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

def _coalesce_bin(coords, oc, mo, dom, layer, scheme: RegionAddressLike = H9_RA):
    """Canonical layer-L key for each point — the single source of truth shared by
    binning AND encoding (a full UUID is just the canonical bin at UUID_DEPTH).

    Coalesces the half-hex triangles into their canonical full hexagon in
    region-space: the three half-hexes meeting at a vertex (the mode-1 parent
    "splits") share one binning hexagon. After the fold every cell has a mode-0
    terminal parent (p_mo == 0), so the (c2, r_mo) key tail alone identifies it —
    which is exactly why address == bin and bins are invertible.
    """
    from hhg9.h9 import H9O, H9C, H9K
    from hhg9.h9.region import region_neighbours
    from hhg9.h9.classifier import location
    from hhg9.h9.protocols import BaryLoc
    oc = np.asarray(oc).copy()
    x, y = coords[:, 0], coords[:, 1]
    regions = xy_regions(coords, mo, layer)                     # (N, layer+2)
    # Folding is a property of the *cell* (a mode-1 half-hex), independent of
    # where the point sits, so EDGE/VERTEX points fold too; only out-of-scope
    # (EXT/UDF) points are left untouched.
    locs = location(H9K.R3 * x, y, mo)
    active = ((locs != BaryLoc.EXT) & (locs != BaryLoc.UDF) & (H9C.mode[regions[:, -2]] == 1))
    if np.any(active):
        idx = np.flatnonzero(active)
        nbr, c2 = region_neighbours(regions[idx])
        hopped = regions[idx, 0] != nbr[:, 0]                   # octant-spanning fold
        regions[idx[~hopped]] = nbr[~hopped]
        if np.any(hopped):
            hidx = idx[hopped]
            oc_h = H9O.oid_nb[oc[hidx], c2[hopped]]             # neighbour octant
            flipped = np.column_stack([x[hidx], -y[hidx]])      # seam = inverted y-axis
            regions[hidx] = xy_regions(flipped, H9O.oid_mo[oc_h], layer)
            oc[hidx] = oc_h
    hx = reg_hex_digits(regions, oc, dom, TailStyle.key, scheme=scheme)
    body = hx[:, :-1]
    uuid_nibs = np.full((len(body), 32), 0x0F, dtype=np.uint8)  # nibbles layer+1..30 = 0xF
    uuid_nibs[:, :layer + 1] = body                            # body L0..L_layer
    uuid_nibs[:, -1] = hx[:, -1] & 0x0F                        # key tail at nibble 31
    return np.array([uuid_mod.UUID(int=v) for v in batch_nibbles_to_int(uuid_nibs)])


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
    uuids     : list[uuid.UUID]  — 128-bit canonical addresses, one per point

    A full UUID is the canonical bin at UUID_DEPTH: encode == bin(., UUID_DEPTH),
    so every address is invertible and h9_bin(addr, L) is a pure truncation.
    """
    oc, mo = b_pts.cm()
    return _coalesce_bin(b_pts.coords, oc, mo, b_pts.domain, UUID_DEPTH, scheme=scheme)


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
    uuids     : list[uuid.UUID]  — 128-bit canonical addresses, one per point
    """
    return _coalesce_bin(b_pts.coords, oc, mo, b_pts.domain, UUID_DEPTH)


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
    from hhg9.h9 import H9K
    from hhg9.h9.tail import tail_unpack_reversible
    uuid_ints = [u.int for u in uuids]
    uuid_nibbles = _batch_int_to_nibbles(uuid_ints, n=32)   # (N, 32)
    # Single-nibble tail at [31]; body at [0..30]. hex_digits_reg is layer-aware
    # (skips 0xF sentinels), so this decodes full addresses and truncated bins alike.
    key_tail = uuid_nibbles[:, -1]
    # Decode to the cell CENTROID, not the cell origin: omit the "3" vertex proxy
    # and seed regions_xy at the hexagon centroid (in lattice units, scaled by Ü).
    # Centroid offset by (mode, c2) — see polygon.py; the seed divides by 3 per
    # layer so it lands at the cell's own scale (incl. octant-sized L0).
    c2, _r_mo, c_mo = tail_unpack_reversible(key_tail)
    cent = _HEX_CENTROID[c_mo.astype(np.intp), c2.astype(np.intp)] * H9K.Ü
    oc, cells = hex_digits_reg(b_oct, uuid_nibbles[:, :-1], tail=key_tail, place_terminal=False)
    xy_m = rg.regions_xy(cells, seed=cent)
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
    """Given a b_oct Points, return the layer-L bin UUIDs.

    Shares the canonical coalesce with h9_enc (a full address is just the bin at
    UUID_DEPTH); see _coalesce_bin. L0 is handled directly because
    region_neighbours degenerates at the root.
    """
    oc, mo = b_pts.cm()
    if layer == 0:
        # L0 is the degenerate root for region_neighbours (no parent to cascade,
        # halves span octant *edges*). But the root-hex nibble is already a clean
        # bijection to the 12 bent hexagons, so canonicalise straight to the
        # mode-0 octant rep (r_mo=0): each physical hexagon -> exactly one key.
        from hhg9.h9 import H9O, H9R
        cx = xy_regions(b_pts.coords, mo, 0)
        c2 = H9R.mcc2[mo, cx[:, 1]]
        c2 = np.where(c2 == 0x5F, np.uint8(0), c2)
        root_hex = H9O.l0hex_by_id[oc, c2]
        c2_canon = H9O.l0hex_back[root_hex, 0][:, 1]            # mode-0 rep c2
        tail = ((c2_canon & 0x03) << 1).astype(np.uint8)        # (p_c2<<1)|r_mo; r_mo=0, p_mo=0
        uuid_nibs = np.full((len(root_hex), 32), 0x0F, dtype=np.uint8)
        uuid_nibs[:, 0] = root_hex
        uuid_nibs[:, -1] = tail & 0x0F
        return np.array([uuid_mod.UUID(int=v) for v in batch_nibbles_to_int(uuid_nibs)])

    return _coalesce_bin(b_pts.coords, oc, mo, b_pts.domain, layer)


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

def h9_layer(uuids):
    """The hierarchical layer of an H9 UUID/bin, derived by counting 0xF sentinels.

    Body digits are 0..11 (L0) / 0..8 (L1+) and never 0xF, so a 0xF nibble is
    unambiguously a sentinel. The layer is the index of the deepest non-sentinel
    body nibble (nibbles 0..30): a full address has none -> UUID_DEPTH; a bin
    truncated to L carries sentinels in nibbles L+1..30 -> L.

    Accepts a single uuid.UUID (returns int) or an iterable (returns an int array).
    """
    single = isinstance(uuids, uuid_mod.UUID)
    items = [uuids] if single else list(uuids)
    body = _batch_int_to_nibbles([u.int for u in items], n=32)[:, :UUID_DEPTH + 1]
    is_real = body != 0x0F
    layer = UUID_DEPTH - np.argmax(is_real[:, ::-1], axis=1)
    return int(layer[0]) if single else layer


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
