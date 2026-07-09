# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
H9 UUID Address Encoding.

Provides a stable 128-bit (UUID) representation of a hex9 address at maximum
practical depth, plus a companion byte that enables exact round-trip to lat/lon.

Layout (32 nibbles = 128 bits):
    nibbles 0..30 : hex body (L0..L30) — hierarchical hex digits
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

A full UUID is the canonical bin at UUID_DEPTH and carries a single-nibble
reversible tail, so the UUID *alone* is sufficient for spatial indexing, hexbin
at any layer, AND exact round-trip to lat/lon — it is self-inverting. (Earlier
revisions returned a companion ``adr`` byte for reconstruction; that is no
longer needed and has been removed.)

Public API:
    h9_encode(lats, lons)                       -> uuids
    h9_decode(uuids)                            -> (lats, lons)
    h9_bin(uuids, layer)                        -> uuids
    h9_ancestors(b_pts, at_layer, gens_up)     -> uuids   (one per point)
    h9_descendants(b_pts, at_layer, gens_down) -> list of uuid lists
"""

from __future__ import annotations

import uuid as uuid_mod
import numpy as np
from numpy.typing import NDArray
from typing import TYPE_CHECKING

from hhg9.h9.region import xy_regions

if TYPE_CHECKING:
    from hhg9 import Points, Registrar

from hhg9.h9.addressing import hex_digits, hex_decode, H9_RA, hex_digits_reg, reg_hex_digits, canonicalise
from hhg9.h9.tail import TailStyle, tail_pack_reversible, tail_unpack_reversible
from hhg9.h9.protocols import RegionAddressLike

# Public API also includes h9_postgis_hexagons (defined at bottom of module).

# The layer parameter passed to hex_digits that yields 32 nibbles:
#   xy_regions(depth=30) -> addresses shape (N, 32)
#   reg_hex_digits sees cols=32, depth=31 -> bdy has 31 cols (L0...L30)
#   key pack: 30 body nibbles + 2 key_tail nibbles = 32 nibbles = 128 bits
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
    # The half-hex -> mode-0 fold is the packer-agnostic canonicaliser; a full
    # UUID is just the canonical bin at UUID_DEPTH. Shared with every other packer.
    regions, oc = canonicalise(coords, oc, mo, dom, layer, scheme=scheme)
    hx = reg_hex_digits(regions, oc, dom, TailStyle.key, scheme=scheme)
    body = hx[:, :-1]
    uuid_nibs = np.full((len(body), 32), 0x0F, dtype=np.uint8)  # nibbles layer+1..30 = 0xF
    uuid_nibs[:, :layer + 1] = body                            # body L0..L_layer
    uuid_nibs[:, -1] = hx[:, -1] & 0x0F                        # key tail at nibble 31
    return [uuid_mod.UUID(int=v) for v in batch_nibbles_to_int(uuid_nibs)]


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
    so every address is invertible.
    """
    oc, mo = b_pts.cm()
    return _coalesce_bin(b_pts.coords, oc, mo, b_pts.domain, UUID_DEPTH, scheme=scheme)


def h9_enc_ext(b_pts, oc, mo, depth=UUID_DEPTH) -> list[uuid_mod.UUID]:
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
    :param depth: unsigned int defaulting to UUID_DEPTH
    """
    return _coalesce_bin(b_pts.coords, oc, mo, b_pts.domain, depth)


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
    uuids     : list[uuid.UUID] — key UUIDs from h9_enc (self-inverting)
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
) -> list[uuid_mod.UUID]:
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
    uuids : list[uuid.UUID]  — 128-bit canonical addresses, one per point.
            Each UUID is the canonical bin at UUID_DEPTH and is self-inverting,
            so no companion adr bytes are returned (see module docstring).
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


def h9_bin_pts(b_pts: Points, layer: int) -> list[uuid_mod.UUID]:
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
        return [uuid_mod.UUID(int=v) for v in batch_nibbles_to_int(uuid_nibs)]

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


# ---------- Hierarchy traversal (relative generations) ---------------------
# These express the H9 "direct per-layer contract": descendants(H, L+i) is not an
# iterated parent/child walk over lineage digits, it is the set of layer-(L+i)
# hexagons whose canonical ancestor (mode-0 convention) at layer L is H. See
# docs/h3/dggs_nesting.py for the motivation. Input is a b_oct Points; the anchor
# is the layer-`at_layer` bin of each point.

def h9_ancestors(
        b_pts: 'Points',
        at_layer: int,
        generations_up: int,
        reg=None,
) -> list[uuid_mod.UUID]:
    """Coarser-layer ancestor bin of each point, ``generations_up`` layers up.

    A point lies in exactly one hexagon per layer, so the ancestor is just the
    coarser bin: ``h9_bin_pts(b_pts, at_layer - generations_up)``. Returns one
    UUID per input point.

    Parameters
    ----------
    b_pts : Points in b_oct domain
    at_layer : int — the layer at which each point's anchor hexagon is taken
    generations_up : int — how many layers coarser to go (>= 0)
    reg : unused (kept for signature symmetry with h9_descendants)
    """
    target = at_layer - generations_up
    if not (0 <= target <= at_layer <= UUID_DEPTH):
        raise ValueError(
            f"need 0 <= at_layer - generations_up <= at_layer <= {UUID_DEPTH}; "
            f"got at_layer={at_layer}, generations_up={generations_up}")
    return h9_bin_pts(b_pts, target)


# ---------- Canonical cell ancestry (mode-0 convention) ----------------------
# h9_bin / h9_bin_pts / h9_ancestors answer the POINT question: which layer-K
# cell contains this point. The functions below answer the CELL question:
# which single layer-K cell is the canonical ancestor of this cell. The two
# differ at split cells (x_dig 6..8), which geometrically straddle two
# parents; the canonical owner is the one containing the cell's mode-0
# d_cell (the §10b mode-0 convention). Binning a split cell's decoded
# centroid cannot answer this — the centroid lies exactly on the straddled
# boundary — so we decode to a point strictly interior to the mode-0 half:
# the centroid nudged toward the cell origin (libhex9's full_id_from_cell
# applies the same cure to grid identity UUIDs).
#
# DOCTRINE — the tree is on d_cells, not x_cells. An x_cell's territory at
# the next layer is 18 d_cells (12 complete = its 6 interior children, plus
# its own 6 split halves), and the d_cell tree nests EXACTLY (rep-9
# rep-tile). A cell's mode-0 d_cell therefore lies wholly inside one cell
# at EVERY coarser layer, so canonical ancestry at any depth is the
# leaf-reified d_cell relation — the mode-0 reification of d_cells into
# x_cells happens once, at the leaf. Naively composing x_cell parents
# level-by-level re-adjudicates the splits at every layer and the hexagon
# decoheres (deep tongues/voids; see docs/dggs/dggs_nesting.py).
# Two equivalent derivations, byte-identical globally (L1..L4, every
# target): the ADDRESS-SPACE fold (addressing.x_adr_cell_ancestor — the
# production path, exact at any depth) and the GEOMETRIC deep re-bin of a
# mode-0-interior point (_mode0_interior_pts + h9_bin_pts — kept as the
# cross-check oracle; its nudge margins scale 3^-L so it degrades in
# doubles near L25+). Verified: exactly 9 canonical children per parent,
# every cell, layer pairs L1..L5 globally
# (experimental/cell_ancestor_verify.py).

_ANC_NUDGE = 0.10   # interior fraction, centroid -> cell origin (mode-0 side)


def _mode0_interior_pts(uuids: list[uuid_mod.UUID], b_oct) -> 'Points':
    """Decode bins to a point strictly inside each cell's mode-0 d_cell."""
    import hhg9.h9.region as rg
    from hhg9 import Points
    from hhg9.h9 import H9K
    from hhg9.h9.tail import tail_unpack_reversible
    nibs = _batch_int_to_nibbles([u.int for u in uuids], n=32)
    key_tail = nibs[:, -1]
    c2, _r_mo, c_mo = tail_unpack_reversible(key_tail)
    cent = _HEX_CENTROID[c_mo.astype(np.intp), c2.astype(np.intp)] * H9K.Ü
    oc, cells = hex_digits_reg(b_oct, nibs[:, :-1], tail=key_tail, place_terminal=False)
    xy_c = rg.regions_xy(cells, seed=cent)      # cell centroid (as h9_dec)
    xy_o = rg.regions_xy(cells)                 # cell origin (mode-0 side)
    xy = xy_c + _ANC_NUDGE * (xy_o - xy_c)
    return Points(xy[:, :2], domain=b_oct, oid=oc)


def h9_cell_parent(uuids: list[uuid_mod.UUID], reg=None) -> list[uuid_mod.UUID]:
    """Canonical parent bin of each *cell* — one layer up, mode-0 convention.

    Every layer-K cell is the canonical parent of exactly 9 layer-(K+1)
    cells: its 6 interior children plus its own 3 split children. Inputs may
    be bins at mixed layers; L0 cells raise (no parent).

    ``reg`` is unused (kept for API compatibility): the roll-up is pure
    address arithmetic — see :func:`h9_cell_ancestor`.
    """
    layers = np.atleast_1d(h9_layer(uuids))
    if np.any(layers < 1):
        raise ValueError('L0 cells have no parent')
    out: list = [None] * len(layers)
    for L in np.unique(layers):
        idx = np.flatnonzero(layers == L)
        for j, u in zip(idx, _cell_ancestor_batch([uuids[i] for i in idx],
                                                  int(L), int(L) - 1)):
            out[j] = u
    return out


def _cell_ancestor_batch(uuids, at_layer: int, target: int) -> list[uuid_mod.UUID]:
    """Address-space canonical ancestor for a uniform-layer batch."""
    from hhg9.h9.addressing import x_adr_cell_ancestor
    nibs = _batch_int_to_nibbles([u.int for u in uuids], n=32)
    hx = np.column_stack([nibs[:, :at_layer + 1], nibs[:, -1]])
    _, anc = x_adr_cell_ancestor(hx, target)
    full = np.full((len(uuids), 32), 0x0F, dtype=np.uint8)
    full[:, :target + 1] = anc[:, :-1]
    full[:, -1] = anc[:, -1]
    return [uuid_mod.UUID(int=v) for v in batch_nibbles_to_int(full)]


def h9_cell_ancestor(uuids, layer: int, reg=None) -> list[uuid_mod.UUID]:
    """Canonical layer-``layer`` ancestor of each cell (mode-0 convention).

    NOT the iterated one-level parent. The subdivision tree is the 9-ary
    d_cell tree, which nests exactly; a cell's mode-0 d_cell lies wholly
    inside one cell at every coarser layer, so its ancestor at any depth
    is the leaf-reified d_cell relation (the mode-0 reification of d_cells
    into x_cells happens at the leaf only — see the doctrine comment
    above). Composing :func:`h9_cell_parent` level-by-level would
    re-adjudicate splits at every layer and decohere at nested splits; the
    two relations coincide only for a single generation.

    Implemented in pure address space (``addressing.x_adr_cell_ancestor``):
    recover the region thread, truncate at ``layer``, and fold mode-1
    presentations to their canonical mode-0 registration
    (region_neighbours' upward cascade; the symbolic seam mirror when the
    fold crosses octants). Exact at any depth — no geometry, no nudge —
    and verified byte-identical both to the geometric mode-0-interior
    re-bin and to libhex9 for all cells L1–L4 at every target. ``reg`` is
    unused (kept for API compatibility). Cells already at ``layer`` pass
    through unchanged; input coarser than ``layer`` raises.
    """
    out = list(uuids)
    lay = np.atleast_1d(h9_layer(out))
    if np.any(lay < layer):
        raise ValueError('input coarser than target layer')
    deep = np.flatnonzero(lay > layer)
    if deep.size == 0:
        return out
    for L in np.unique(lay[deep]):
        idx = deep[lay[deep] == L]
        for i, u in zip(idx, _cell_ancestor_batch([out[i] for i in idx],
                                                  int(L), layer)):
            out[i] = u
    return out


def _anchor_hex_latlon(anchor: uuid_mod.UUID, at_layer: int, reg, inflate: float = 1.03):
    """The anchor hexagon's lat/lon ring, built from its UUID via H9P.hx.

    Slightly inflated (``inflate``) so the clip region is a strict superset of the
    true hexagon — completeness is then guaranteed by the downstream bin filter,
    which removes any extra hexes the larger polygon admitted.

    Ring vertices of a seam-straddling hexagon overhang the anchor's octant;
    they are folded into their true octant (``fold_to_octant``) before
    projection — a strictly in-octant projection would wrap them to nonsense
    (grid_face_vertex_oid_bug family).
    """
    from hhg9.h9 import H9P
    from hhg9.h9.polygon import fold_to_octant
    from hhg9 import Points
    b_oct = reg.domain('b_oct')
    g_gcd = reg.domain('g_gcd')
    dpts = h9_dec([anchor], b_oct)
    centroid = dpts.coords[0]
    oid = int(dpts.oid[0])
    c2, _r_mo, p_mo = tail_unpack_reversible(np.array([anchor.int & 0xF], dtype=np.uint8))
    hx = H9P.hx[int(p_mo[0]), int(c2[0])]                       # (6, 2) parent-origin-relative
    ring_rel = (hx - hx.mean(axis=0)) * (3.0 ** -at_layer) * inflate
    verts = centroid[None, :] + ring_rel                        # (6, 2) b_oct
    fold_xy, fold_oid = fold_to_octant(verts, oid)
    ll = reg.project(Points(fold_xy, b_oct, oid=fold_oid), [b_oct, g_gcd]).coords
    return ll                                                   # (6, 2) [lat, lon]


def h9_descendants(
        b_pts: 'Points',
        at_layer: int,
        generations_down: int,
        reg=None,
) -> list[list[uuid_mod.UUID]]:
    """Canonical descendants of each point's anchor hexagon.

    The descendant set is the H9 per-layer contract: every layer-`target`
    hexagon whose canonical ancestor (``h9_cell_ancestor``, mode-0
    convention: leaf mode-0 d_cell inside the anchor) is the anchor —
    exactly 9^generations_down cells, tiling the anchor hexagon exactly up
    to a one-leaf-cell fringe. Splits on the anchor's rim protrude by
    their far (mode-1) half but belong to exactly one canonical ancestor;
    rim splits owned by a neighbour belong there, not here.

    Reuses the proven pruned descent in ``HexMesh.create_clipped`` over the anchor
    hexagon, re-bins each descendant centroid to a canonical UUID, then filters on
    the bin test. Returns one list of UUIDs per input point.

    Parameters
    ----------
    b_pts : Points in b_oct domain
    at_layer : int — layer of the anchor hexagon
    generations_down : int — how many layers finer (>= 0)
    reg : Registrar (created if None)
    """
    target = at_layer + generations_down
    if not (0 <= at_layer <= target <= UUID_DEPTH):
        raise ValueError(
            f"need 0 <= at_layer <= at_layer + generations_down <= {UUID_DEPTH}; "
            f"got at_layer={at_layer}, generations_down={generations_down}")
    if reg is None:
        from hhg9 import Registrar
        reg = Registrar()
    from hhg9.h9 import H9O
    from hhg9.h9.grid import HexMesh
    from hhg9 import Points

    b_oct = reg.domain('b_oct')
    anchors = h9_bin_pts(b_pts, at_layer)
    if generations_down == 0:
        return [[a] for a in anchors]

    out: list[list[uuid_mod.UUID]] = []
    # Cache per distinct anchor: points sharing an anchor share descendants.
    cache: dict[int, list[uuid_mod.UUID]] = {}
    for anchor in anchors:
        key = anchor.int
        if key in cache:
            out.append(cache[key])
            continue
        # Canonical descendants stay within half a target-layer cell of the
        # anchor's boundary (only rim splits protrude, by their far half —
        # the d_cell tree nests exactly, so nothing telescopes deeper).
        # Inflate the enumeration clip generously; non-descendants are
        # filtered below.
        poly = _anchor_hex_latlon(anchor, at_layer, reg, inflate=1.6)
        mesh = HexMesh.create_clipped([target], poly, reg)
        faces = mesh[target]
        if len(faces) == 0:
            cache[key] = []
            out.append([])
            continue
        # Mode-safe centroid (seam hexes mix mode-0/mode-1 octant coords).
        oid_v = mesh.pts.oid[faces]
        mo_v = H9O.oid_mo[oid_v]
        match = (mo_v == mo_v[:, :1])
        cv = mesh.pts.coords[faces]
        cent = (cv * match[:, :, None]).sum(axis=1) / match.sum(axis=1, keepdims=True)
        cpts = Points(cent, b_oct, oid=oid_v[:, 0].astype(np.int32))
        canon = h9_bin_pts(cpts, target)
        # Keep only hexes whose CANONICAL ancestor is the anchor (mode-0
        # convention; exactly 9^g per anchor). Re-binning centroids here is
        # the centroid-on-seam disease (doctrine): split descendants sit ON
        # the anchor boundary and tie-break away.
        back = h9_cell_ancestor(canon, at_layer, reg=reg)
        seen, kept = set(), []
        for d, a in zip(canon, back):
            if a.int == anchor.int and d.int not in seen:
                seen.add(d.int)
                kept.append(d)
        # Completeness fallback: near octahedral vertices the clipped-mesh
        # enumeration can drop cross-seam descendants (strict in-octant
        # projection; see grid_face_vertex_oid_bug.md). Rescue by sampling
        # a disc in lat/lon — encode is frame-agnostic — and keeping cells
        # whose canonical ancestor is the anchor.
        want = 9 ** generations_down
        if len(kept) < want:
            ctr_lat, ctr_lon = h9_decode([anchor], reg=reg)
            cla, clo = float(np.ravel(ctr_lat)[0]), float(np.ravel(ctr_lon)[0])
            # Size the disc from the cell geometry (3^-L) directly, NOT from
            # the ring: at vertex cells the hx ring template is wrong
            # (grid_face_vertex_oid_bug family) and wraps past the
            # antimeridian, exploding the radius until the fixed grid is
            # coarser than a child cell and whole children fall between
            # samples.
            R = 135.0 * (3.0 ** -at_layer)
            gl = np.clip(np.linspace(cla - R, cla + R, 220), -89.999, 89.999)
            # Longitude span: converge at the disc's most polar latitude,
            # not the centre; a disc containing a pole spans all longitudes.
            lat_hi = abs(cla) + R
            if lat_hi >= 90.0:
                gn = clo + np.linspace(-180.0, 180.0, 220)
            else:
                cosl = max(np.cos(np.radians(lat_hi)), 0.05)
                gn = clo + np.linspace(-R, R, 220) / cosl
            GLa, GLo = np.meshgrid(gl, gn)
            glo = (GLo.ravel() + 180.0) % 360.0 - 180.0
            pu = h9_encode(GLa.ravel(), glo, reg=reg)
            fine = list({u.int: u for u in h9_bin(pu, target, reg=reg)}.values())
            fb = h9_cell_ancestor(fine, at_layer, reg=reg)
            have = set(k.int for k in kept)
            extra = sorted((f for f, a in zip(fine, fb)
                            if a.int == anchor.int and f.int not in have),
                           key=lambda u: u.int)
            kept = kept + extra
        cache[key] = kept
        out.append(kept)
    return out


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


def h9_label(u: uuid_mod.UUID, with_tail=True) -> str:
    """Convert an H9 UUID to a human-readable label string.
    Format: '<body>.<key>'  e.g. '32343.2'
    The body is the hex-digit address up to (not including) the first 0x0F
    sentinel nibble.  The key nibble (nibble 31) follows the dot.
    Without tail, the canonical label is unique, but cannot be inverted or re-binned!
    """
    nibs = _batch_int_to_nibbles([u.int], n=32)[0]          # (32,) uint8
    body_nibs = nibs[:31]
    sentinels = np.where(body_nibs == 0x0F)[0]
    stop = int(sentinels[0]) if len(sentinels) else 31       # full-depth → all 31
    body_str = ''.join(format(int(v), 'x') for v in body_nibs[:stop])
    if not with_tail:
        return body_str
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
