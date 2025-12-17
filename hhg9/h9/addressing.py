# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
H9 Addressing and Encoding.

This module handles the translation between **Barycentric Coordinates** (math) and
**Addressable Indices** (storage/display).

It provides mechanisms to:
1.  **Map** geometric cells to logical "Regions".
2.  **Pack** these regions into efficient integer formats (UInt64).
3.  **Convert** regions into hierarchical Hex-Digit strings (e.g., "N0A12...").
4.  **Coalesce** triangles (half-hexes) into full hexagons for binning.


**Key Concepts:**

* **Regions:** A simplified view of the grid where specific geometric cells are mapped
    to IDs 0-11, allowing for recursive fractal traversal.
* **Octants:** The 8 faces of the underlying octahedron, serving as the root of the address.
* **Hex Digits:** Base-9 digits (0-8) representing the recursive subdivision of the grid.

** Hexagon Address **
The primary goal here is to offer a means of being able to generate a unique, reversible
hexagonal grid address.  It needs to be a reliable key for all data.
The core structure is as follows
1*[0...B]  Octahedral hexagon identity (0..11).  The unit octahedron is made of 8 equilateral faces
           each of which is composed of three half-hexagons.  This gives 24 half-hexagons, and
           therefore 12 'bent' hexagons. That cover the entire octahedron.
           This is Layer 0.
L*[0...7]  Within each hexagon, there are a group of six full hexagons of the subsequent layer,
           and six half-hexagons of the subsequent layer.  They are all numbered between 0..7
           The specific pattern is documented elsewhere.
1*[mm|reg] Metadata; Without recognising the region-mode of the terminal hexagon, there is some ambiguity
           Therefore, we need a digit to indicate the region-mode.  It is also useful to record
           the root mode (which of the octahedron faces this address belongs to).
           Likewise, we want to record the terminating region (0..11) in order to recover an address in full.

           Of these values, only the region-mode of the terminal hexagon is considered to be essential for bin-hex key.

***Example***
Consider the address [5, 7, 6, 21]. What is its latitude and longitude?
1: Extract Metadata from 31
   21 => [0010, 0001]
   1 = term_mode of terminal 6
   0 = root_mode of octant
   0001 = terminal region
2: Extract Octant and C2 from initial hex
   root_hex = 5
   octant, c2 = b_oct.l0hex_back[root_hex, root_mode]

"""

from __future__ import annotations
from dataclasses import dataclass
from enum import unique, Enum
from functools import lru_cache
import numpy as np
from numpy.typing import NDArray
from hhg9.h9 import H9R, H9C
from hhg9.h9.protocols import RegionAddressLike, AddressPackerLike, H9CellLike, HexLUTLike, H9RegionLike


@unique
class Style(Enum):
    """
    Enumeration of various Encoding styles.

    * **HEX:** Standard hierarchical hex string.
    * **NUMERIC:** Integer-based representation.
    * **U64/UH64/UR64:** Packed 64-bit integer formats.
    """
    HEX = 0
    NUMERIC = 4
    U64 = 6
    UH64 = 7
    UR64 = 8


# --- Hex Address TailStyle enum and helper ---
@unique
class TailStyle(Enum):
    """How the final tail byte is encoded."""
    reversible = 0  # full metadata tail (invertible)
    key = 1         # short tail (binning/grouping)
    none = 2        # omit tail entirely

# ---------- Hex Address Tail packing helpers ----------------------------------------

# Reversible tail byte layout (one byte):
# bit7: parent-mode of terminating region (p_mo)
# bits6..5: terminating c2
# bit4: root mode (r_mo)
# bits3..0: terminating region id (h)


def tail_pack_reversible(
    p_mo: NDArray[np.uint8] | np.uint8,
    p_c2: NDArray[np.uint8] | np.uint8,
    r_mo: NDArray[np.uint8] | np.uint8,
    h: NDArray[np.uint8] | np.uint8,
) -> NDArray[np.uint8]:

    """Pack reversible tail metadata into one uint8 byte."""
    p_mo = np.asarray(p_mo, dtype=np.uint8)  # terminating hex mode of parent region
    p_c2 = np.asarray(p_c2, dtype=np.uint8)  # terminating hex c2 of parent region
    r_mo = np.asarray(r_mo, dtype=np.uint8)  # root region mode
    h = np.asarray(h, dtype=np.uint8)        # terminating region (under hex)
    return (((p_mo << 7) & 0x80) | ((p_c2 << 5) & 0x60) | ((r_mo << 4) & 0x10) | (h & 0x0F)).astype(np.uint8)


def tail_unpack_reversible(tail_ids: NDArray[np.uint8] | np.uint8):
    """Unpack reversible tail metadata (p_mo, p_c2, r_mo, h) from one uint8 byte."""
    tail_ids = np.asarray(tail_ids, dtype=np.uint8)
    p_mo = ((tail_ids & 0x80) >> 7).astype(np.uint8)  # terminating mode of parent region
    p_c2 = ((tail_ids & 0x60) >> 5).astype(np.uint8)    # terminating hex c2 of parent region
    r_mo = ((tail_ids & 0x10) >> 4).astype(np.uint8)  # root region mode
    h = (tail_ids & 0x0F).astype(np.uint8)            # terminating region
    return p_mo, p_c2, r_mo, h


def tail_pack_key(
    p_c2: NDArray[np.uint8] | np.uint8,    # terminating hex c2 of parent region
    r_mo: NDArray[np.uint8] | np.uint8,  # root region mode
) -> NDArray[np.uint8]:
    """Pack key tail (binning-safe) into one uint8 byte."""
    p_c2 = np.asarray(p_c2, dtype=np.uint8)
    r_mo = np.asarray(r_mo, dtype=np.uint8)
    return (((p_c2 & 0x03) << 5) | (r_mo & 0x01) << 4 | 0xF).astype(np.uint8)


def tail_unpack_key(short_tail: NDArray[np.uint8] | np.uint8):
    """Unpack key tail into (p_c2, r_mo)."""
    short_tail = np.asarray(short_tail, dtype=np.uint8)
    p_c2 = ((short_tail >> 1) & 0x03).astype(np.uint8)
    r_mo = (short_tail & 0x01).astype(np.uint8)
    return p_c2, r_mo


def tail_key_from_reversible(tail_ids: NDArray[np.uint8] | np.uint8) -> NDArray[np.uint8]:
    """Derive key tail from reversible tail without recomputing geometry."""
    tail_ids = np.asarray(tail_ids, dtype=np.uint8)
    # p_c2 = (tail_ids & 0x60) >> 5
    # r_mo = (tail_ids & 0x10) >> 4
    return (tail_ids & 0x70 | 0xF).astype(np.uint8)
    # return tail_pack_key(p_c2.astype(np.uint8), r_mo.astype(np.uint8))


# ---------- Region-ID scheme (even → mode 0, odd → mode 1) ----------


@dataclass(frozen=True, slots=True)
class RegionIdScheme(RegionAddressLike):
    """
    Immutable container for the Region-ID mapping scheme.

    Maps the 42 geometric cell IDs to a compact 12-region system.
    Parity rules are strictly enforced: Even IDs = Mode 0 (Down), Odd IDs = Mode 1 (Up).
    """
    rid2cell: NDArray[np.uint8]
    cell2rid: NDArray[np.uint8]
    modes: NDArray[np.uint8]
    props: NDArray[np.uint8]
    proto: NDArray[np.uint8]
    r_size: int


# @lru_cache(maxsize=1)
def _region_scheme(h9c: H9CellLike, h9r: H9RegionLike) -> RegionIdScheme:
    """
    Builds the Region ID Scheme once and freezes it.

    Enforces that parity equals mode (even→0, odd→1).

    Args:
        h9c: The cell lattice definition.
        h9r: The region definition.

    Returns:
        RegionIdScheme: The configured scheme.
    """
    rid2cell = np.array([
        0x49, 0x16,  # 0,1  protos (m0, m1) - outer
        0x2B, 0x34,  # 2,3  unshared -outer
        0x21, 0x3E,  # 4,5  unshared -outer
        0x26, 0x39,  # 6,7  shared - inner
        0x35, 0x2A,  # 8,9  shared - inner
        0x3A, 0x25,  # 10,11 shared - inner
        0x5F, 0x5F,  # 12, 13 OOB
        0x5F, 0x5F,  # 14, 15 OOB
    ], dtype=np.uint8)

    mo_c2 = np.array([
        [  # mode 0
           # s, s, u    s,  s, u    s, s,  u   Shared/unshared
            [6, 9, 2], [10, 7, 0], [8, 11, 4]  # c2=0,1,2
        ], [  # mode 1
          #  s, s, u    s,  s, u    s, s,  u   Shared/unshared
            [7, 10, 5], [11, 8, 3], [9, 6, 1]  # c2=0,1,2
        ]
    ], dtype=np.uint8)

    r_size = rid2cell.size
    cell2rid = np.full(256, -1, dtype=np.int16)
    cell2rid[rid2cell] = np.arange(r_size, dtype=np.int16)

    # Sanity: anchors + parity rule
    # These two values should come from / be sanitised against region protos.
    assert rid2cell[0] == 0x49 and rid2cell[1] == 0x16
    parity = (np.arange(r_size, dtype=np.uint8) & 1)

    # Enforce parity==mode only for in-bounds cells.
    # rid2cell[12..15] are OOB placeholders (0x5F) and should not participate in the parity check.
    oob_cell = np.uint8(0x5F)
    valid = rid2cell != oob_cell
    assert np.all(parity[valid] == h9c.mode[rid2cell[valid]]), "rid parity must match cell mode (excluding OOB)"
    proto = cell2rid[h9r.proto]
    return RegionIdScheme(rid2cell=rid2cell, cell2rid=cell2rid, modes=parity, props=mo_c2, proto=proto, r_size=r_size)


# ---------- Packer (Pack Regions) -----------------
@dataclass(frozen=True, slots=True)
class RegionPacker(AddressPackerLike):
    """
    Packs (N, L+1) region-ids into a backend representation.

    This class enforces the H9 region addressing **root nibble** protocol:

    * **0..7:** Octant ID (global, face-anchored).
    * **8, 9:** Unanchored prototypes (8=Down/Mode0, 9=Up/Mode1).
    * **A..E:** Reserved.
    * **F:** Error.

    Attributes:
        pack_fn: Callable taking nibbles -> packed words.
        unpack_fn: Callable taking packed words -> nibbles.
        octant_mode_fn: Optional callable mapping octant ID -> mode (0/1).
    """
    pack_fn: callable | None = None
    unpack_fn: callable | None = None
    octant_mode_fn: callable | None = None

    def _octant_to_proto(self, octants: NDArray[np.uint8]) -> NDArray[np.uint8]:
        if self.octant_mode_fn is not None:
            proto = self.octant_mode_fn(octants)
            return np.asarray(proto, dtype=np.uint8)
        # default: even octants→proto 0, odd→1
        return (octants & 1).astype(np.uint8)

    def encode(self, reg_ids: NDArray[np.uint8], octants: NDArray[np.uint8] = None, **kwargs):
        """
        Builds the root nibble and delegates to the backend pack function.

        Args:
            reg_ids: (N, L+1) array of region IDs. Column 0 must be prototype {0, 1}.
            octants: Optional (N,) array of octant IDs.
                * If provided, root nibble becomes octant (0..7).
                * If None, root nibble becomes 8 or 9 (unanchored proto tag).
        """
        reg_ids = np.asarray(reg_ids, dtype=np.uint8)
        assert reg_ids.ndim == 2, "reg_ids must be (N, L+1)"
        N, L1 = reg_ids.shape
        if L1 < 1:
            raise ValueError("reg_ids must have at least the root nibble")
        if not np.all((reg_ids[:, 0] == 0) | (reg_ids[:, 0] == 1)):
            raise ValueError("Layer 0 must be prototype ∈ {0,1}")

        # Construct nibbles with protocol root
        nibbles = reg_ids.copy()
        if octants is not None:
            octants = np.asarray(octants, dtype=np.uint8)
            if octants.shape[0] != N:
                raise ValueError("octants shape mismatch with reg_ids")
            nibbles[:, 0] = octants  # 0..7 tagged as face/octant
        else:
            nibbles[:, 0] = (nibbles[:, 0] & 1) + 8  # 8 or 9 tag

        if self.pack_fn is None:
            raise RuntimeError("RegionPacker.pack_fn is not configured")
        return self.pack_fn(nibbles.astype(np.uint8))

    def decode(self, words, **kwargs) -> NDArray[np.uint8]:
        """
        Decodes packed words into regions and octants.

        Returns:
            tuple: (octants, regions)
        """
        if self.unpack_fn is None:
            raise RuntimeError("RegionPacker.unpack_fn is not configured")
        nibbles = self.unpack_fn(words, **kwargs)
        nibbles = np.asarray(nibbles, dtype=np.uint8)
        octants = None
        root = nibbles[:, 0]
        out = nibbles.copy()
        is_oct = root < 8
        if np.any(is_oct):
            octants = root
            proto = self._octant_to_proto(root[is_oct])  # 0/1 per face
            out[is_oct, 0] = proto
        # For the rest, expect 8 or 9 tags
        non_oct = ~is_oct
        if np.any(non_oct):
            tag = root[non_oct]
            if not np.all((tag == 8) | (tag == 9)):
                raise ValueError("Decoded root nibble not octant/8/9")
            out[non_oct, 0] = (tag & 1).astype(np.uint8)

        # Final sanity: column 0 must be proto {0,1}
        if not np.all((out[:, 0] == 0) | (out[:, 0] == 1)):
            raise ValueError("Decoded address has non-proto at layer 0")
        return octants, out


@lru_cache(maxsize=1)
def region_packer(pack_fn=None, unpack_fn=None, octant_mode=None) -> AddressPackerLike:
    """Factory returning a RegionPacker using a given backend."""
    if pack_fn is None and hasattr(unpack_fn, 'pack') and hasattr(unpack_fn, 'unpack'):
        backend = unpack_fn
        pack_fn = getattr(backend, 'pack')
        unpack_fn = getattr(backend, 'unpack')
    return RegionPacker(pack_fn=pack_fn, unpack_fn=unpack_fn, octant_mode_fn=octant_mode)


# ---------- Packer (Pack Hex Addresses) -----------------
@dataclass(frozen=True, slots=True)
class HexPacker(AddressPackerLike):
    """
    Packs hex addresses into a backend representation.

    Structure: **[Octant + Supercell C2 + Hex Body + Tail Region]**

    **Nibble Stream Layout:**

    1.  **Octant (0..7):** The face ID.
    2.  **C2 (0..2):** The supercell C2 cluster of the address root.
    3.  **Hex Digits (0..8):** The body of the address, one nibble per layer.
    4.  **Tail Region (0..11):** The terminating region ID (offset by +2 for packing).

    Note:
        Unlike `RegionPacker`, this keeps Octant and C2 explicit and does not use the 8/9 protocol tags.
    """
    pack_fn: callable | None = None
    unpack_fn: callable | None = None

    def encode(self,
               hex_body: NDArray[np.uint8],
               octants: NDArray[np.uint8],
               c2s: NDArray[np.uint8],
               tail_regions: NDArray[np.uint8],
               **kwargs):
        """
        Encodes hex components into packed nibbles.

        Args:
            hex_body: (N, L) array of hex digits in 0..8.
            octants: (N,) array of octant IDs 0..7.
            c2s: (N,) array of supercell c2 values 0..2.
            tail_regions: (N,) array of terminating region IDs 0..11.

        Returns:
            Packed words via the backend `pack_fn`.
        """
        if self.pack_fn is None:
            raise RuntimeError("HexPacker.pack_fn is not configured")
        hex_body = np.asarray(hex_body, dtype=np.uint8)
        octants = np.asarray(octants, dtype=np.uint8)
        c2s = np.asarray(c2s, dtype=np.uint8)
        tail_regions = np.asarray(tail_regions, dtype=np.uint8)

        if hex_body.ndim != 2:
            raise ValueError("hex_body must be (N, L)")
        N, L = hex_body.shape
        if octants.shape != (N,):
            raise ValueError("octants must be shape (N,)")
        if c2s.shape != (N,):
            raise ValueError("c2s must be shape (N,)")
        if tail_regions.shape != (N,):
            raise ValueError("tail_regions must be shape (N,)")
        if not np.all(octants < 8):
            raise ValueError("octant must be in 0..7")
        if not np.all(c2s < 3):
            raise ValueError("c2 must be in 0..2")
        if not np.all(hex_body <= 8):
            raise ValueError("hex digits must be in 0..8")
        if not np.all(tail_regions < 12):
            raise ValueError("tail_regions must be in 0..11")

        nibbles = np.empty((N, 2 + L + 1), dtype=np.uint8)
        nibbles[:, 0] = octants  # 0..7
        nibbles[:, 1] = c2s  # 0..2
        if L:
            nibbles[:, 2:2 + L] = hex_body  # 0..8
        nibbles[:, 2 + L] = (tail_regions + 2)  # 0..11 (fits in a nibble)
        return self.pack_fn(nibbles)

    def decode(self, words, layers: int | None = None, **kwargs):
        """
        Decodes packed words into hex components.

        Args:
            words: Backend-packed payload.
            layers: Body length (hex digits). If None, inferred from non-zero columns.

        Returns:
            tuple: (octants, c2s, hex_body, tail_regions)
        """
        if self.unpack_fn is None:
            raise RuntimeError("HexPacker.unpack_fn is not configured")
        nibbles = self.unpack_fn(words, **kwargs)
        nibbles = np.asarray(nibbles, dtype=np.uint8)
        if nibbles.ndim != 2 or nibbles.shape[1] < 3:
            raise ValueError("decoded nibbles shape invalid for hex address")
        if layers is None:
            # infer L = total_cols - 3 (octant, c2, tail_region)
            cols_used = np.any(nibbles != 0, axis=0)
            used_idx = np.flatnonzero(cols_used)
            if used_idx.size == 0 or used_idx[-1] < 2:
                raise ValueError("cannot infer length: no non-zero columns beyond header")
            tail_col = int(used_idx[-1])
            layers = tail_col - 3  # subtract octant(0) and c2(1)
        octants = nibbles[:, 0]
        c2s = nibbles[:, 1]
        hex_body = nibbles[:, 2:3 + layers]
        tail_regions = nibbles[:, 3 + layers] - 2
        return octants, c2s, hex_body, tail_regions


@lru_cache(maxsize=1)
def hex_packer(pack_fn=None, unpack_fn=None) -> AddressPackerLike:
    """Factory returning a HexPacker using a given backend."""
    if pack_fn is None and hasattr(unpack_fn, 'pack') and hasattr(unpack_fn, 'unpack'):
        backend = unpack_fn
        pack_fn = getattr(backend, 'pack')
        unpack_fn = getattr(backend, 'unpack')
    return HexPacker(pack_fn=pack_fn, unpack_fn=unpack_fn)


# ---------- Neighbour calculation --------------------------------------

def neighbours(pts, layer=32, coalesce=True):
    """
    Calculates neighbors and optionally coalesces half-hexagons into hexagons.


    **Coalescing Logic:**
    At a specific layer, 3 "half-hex" triangles meet at a vertex. To form a valid
    Hexagon Grid for binning, these three must be merged (coalesced) into one logical hexagon.
    This involves checking the parent layer's mode and adjusting the C2 cluster accordingly.

    Args:
        pts (Points): The input barycentric points.
        layer (int): The depth at which to calculate neighbors/hexagons.
        coalesce (bool): If True, merges triangles into hexagons.

    Returns:
        Points: New points representing the neighbor/coalesced center.
    """
    from hhg9.h9.region import region_neighbours, regions_xy, xy_regions
    from hhg9 import Points
    dom = pts.domain
    oc, mode = pts.cm()
    coords = pts.coords.copy()
    x = coords[:, 0]
    y = coords[:, 1]
    c = oc[:]
    active = np.full(len(pts), 1, dtype=bool)
    regions = xy_regions(coords, mode, layer)  # no depth?!
    if coalesce:
        active = H9C.mode[regions[:, -2]].astype(bool)
    xa = x[active]
    ya = y[active]
    ca = c[active]
    nbr, c2 = region_neighbours(regions[active])
    hopped = regions[active, 0] != nbr[:, 0]
    xym = regions_xy(nbr[~hopped])
    xa[~hopped] = xym[:, 0]
    ya[~hopped] = xym[:, 1]
    if np.any(hopped):  # the octant_spanning neighbour is merely the inverted y-axis!
        ca[hopped] = dom.oid_nb[ca[hopped], c2[hopped]]  # Adjust the octant accordingly
        ya[hopped] = -ya[hopped]
    oc[active] = ca
    coords[active, 0] = xa
    coords[active, 1] = ya
    cmp = pts.invert_octant_ids(oc)
    return Points(coords, domain=dom, components=cmp)


# ---------- Emergent hex-digit per step (optional LUT) ----------------
@dataclass(frozen=True, slots=True)
class HexLUT(HexLUTLike):
    """Container for the massive Region-to-Hex lookup tables."""
    hex_oob: int
    hex_reg: NDArray[np.uint8]
    reg_hex: NDArray[np.uint8]


_m_c2_hx_v2024 = [
    # This is the early-version (2024/2025):
    # - mode 0 has a cluster of 3 '4' hexes around its origin.
    # - mode 1 has a cluster of 3 '5' hexes around its origin.
    # This dict is the ground-truth for all hexagon digits.
    # given sc.mode, sc.c2, region, region.c2 => hex-digit.
    # ROOT super-regions mark hex digits as 0,1,2 for each c2 (the hex ID is face-dependant)
    [  # super-region mode down (V)
        [  # cells of c2=0 of V super-region
            # (Each region has 3 hex digits - one in each c2)
            # regions 692 of each down are in c2=0. Their order does not represent c2
            [6, [0, 4, 7]],  # shared,   same mode as super-region 0x26 (pL=:3)
            [9, [7, 4, 2]],  # shared,   diff mode to super-region 0x2a
            [2, [3, 6, 2]],  # unshared, same mode as super-region 0x2B
        ],
        [  # cells of c2=1 of V supercell
            [10, [7, 0, 4]],  # shared,  same mode as super-region 0x3a
            [7, [2, 7, 4]],  # shared,   diff mode to super-region 0x39
            [0, [2, 3, 6]],  # unshared, same mode as super-region 0x49
        ],
        [  # cells of c2=2 of V supercell
            [8, [4, 7, 0]],  # shared,   same mode as super-region 0x35
            [11, [4, 2, 7]],  # shared,  diff mode to super-region 0x25
            [4, [6, 2, 3]],  # unshared, same mode as super-region 0x21
        ],
    ],
    [  # super-region mode up (Λ)
        [  # regions of c2=0 of Λ super-region
            [7, [0, 8, 5]],  # shared,   same mode as super-region 0x39
            [10, [8, 1, 5]],  # shared,  diff mode to super-region 0x3a
            [5, [3, 1, 6]],  # unshared, same mode as super-region 0x3E
        ],
        [  # regions of c2=1 of Λ supercell
            [11, [5, 0, 8]],  # shared,  same mode as super-region 0x25
            [8, [5, 8, 1]],  # shared,   diff mode to super-region 0x35
            [3, [6, 3, 1]],  # unshared, same mode as super-region 0x34
        ],
        [  # regions of c2=2 of Λ supercell
            [9, [8, 5, 0]],  # shared,   same mode as super-region 0x2a
            [6, [1, 5, 8]],  # shared,   diff mode to super-region 0x26
            [1, [1, 6, 3]],  # unshared, same mode as super-region 0x16
        ]
    ]
]

_m_c2_hx_v2025 = [
    # This is the late-version (2025/2026):
    # - mode 0 has a cluster of 3 '0' hexes around its origin.
    # - mode 1 has a cluster of 3 '1' hexes around its origin.
    # Layer i+1 hexes will have a cluster of 3 '2' hexes at the centres
    #     of the layer i+0 0/1/2 (and 3/4/5, 6/7/8) clusters
    # This dict is the ground-truth for all hexagon digits.
    # It considers the digits from the (triangular) region/super-region context.
    # Consider an equilateral triangle at Layer i.  In hhg9, this is divided into 3 half-hexes (aka c2) at Layer i.
    # - because each triangle in hhg9 is divided into 9 triangles (regions), each c2 contains 3 layer i+1 regions,
    # each having (according to its mode) 3 half-hexes.
    # regions are 'shared' or 'unshared'; six regions are shared across both modes. six regions are 1-mode only.
    # Given a Li; mode j, it's Li+1 hexagons are shared with every other Li; mode j triangle.
    # The hexagon=>sub-hexagon relationships look different, but are emergent from the definition as above.
    # Within every hexagon there will be child hexagons 0,1,2,3,4,5 and 3 'split' pairs of half-hexagons 6,7,8.
    # The splits are such that they do not share a c2.  For example, the two '6' half-hexagons might be in
    # modes [0, 2].  '6' half-hexes are 'wings' of the '0' hexagon, '7' half-hexes are the 'wings' of '1' hexagon,
    # and '8' half-hexes are the 'wings' of the '2' hexagon
    [  # Layer 'i+0'; super-region mode 0 (V), by c2 orientation, referenced by region-id (0..11) (centred with 0-hex)
        # Note: hex digits ['1', '5', '7'] are not found in i+1 of mode 0.
        [  # cells of c2=0 of V super-region
            # V: regions [6,9,2] are c2=0. Ordered from centre edge to vertex
            [6, [3, 0, 6]],  # shared,   V; cell:0x26; i+1: c2.0=hex-'3', c2.1=hex-'0', c2.2=hex-'6'
            [9, [6, 0, 4]],  # shared,   Λ; cell:0x2a; i+1: c2.0=hex-'6', c2.1=hex-'0', c2.2=hex-'4'
            [2, [2, 8, 4]],  # unshared, V; cell:0x2B; i+1: c2.0=hex-'2', c2.1=hex-'8', c2.2=hex-'4'
        ],
        [  # V: regions [a,7,0] are c2=1. Ordered from centre edge to vertex
            [10, [6, 3, 0]],  # shared,   V; cell:0x3a; i+1: c2.0=hex-'6', c2.1=hex-'3', c2.2=hex-'0'
            [7, [4, 6, 0]],  # shared,   Λ; cell:0x39; i+1: c2.0=hex-'4', c2.1=hex-'6', c2.2=hex-'0'
            [0, [4, 2, 8]],  # unshared, V; cell:0x49; i+1: c2.0=hex-'4', c2.1=hex-'2', c2.2=hex-'8'
        ],
        [  # V: regions [8,b,4] are c2=2. Ordered from centre edge to vertex
            [8, [0, 6, 3]],  # shared,   V; cell:0x35; i+1: c2.0=hex-'0', c2.1=hex-'6', c2.2=hex-'3'
            [11, [0, 4, 6]],  # shared,   Λ; cell:0x25; i+1: c2.0=hex-'0', c2.1=hex-'4', c2.2=hex-'6'
            [4, [8, 4, 2]],  # unshared, V; cell:0x21; i+1: c2.0=hex-'8', c2.1=hex-'4', c2.2=hex-'2'
        ],
    ],
    [  # Layer 'i+0'; super-region mode 0 (Λ), by c2 orientation, referenced by region-id (0..11) (centred with 1-hex)
        # Note: hex digits ['0', '4', '6'] are not found in i+1 of  mode 1:
        [  # cells of c2=0 of Λ super-region
            # Λ: regions [7,a,5] are c2=0. Ordered from centre edge to vertex
            [7, [3, 7, 1]],  # shared,   Λ; cell:0x39; i+1: c2.0=hex-'3', c2.1=hex-'7', c2.2=hex-'1'
            [10, [7, 5, 1]],  # shared,   V; cell:0x3a; i+1: c2.0=hex-'7', c2.1=hex-'5', c2.2=hex-'1'
            [5, [2, 5, 8]],  # unshared, Λ; cell:0x3e; i+1: c2.0=hex-'2', c2.1=hex-'5', c2.2=hex-'8'
        ],
        [  # Λ: regions [b,8,3] are c2=1. Ordered from centre edge to vertex
            [11, [1, 3, 7]],  # shared,   Λ; cell:0x25; i+1: c2.0=hex-'1', c2.1=hex-'3', c2.2=hex-'7'
            [8, [1, 7, 5]],  # shared,   V; cell:0x35; i+1: c2.0=hex-'1', c2.1=hex-'7', c2.2=hex-'5'
            [3, [8, 2, 5]],  # unshared, Λ; cell:0x34; i+1: c2.0=hex-'8', c2.1=hex-'2', c2.2=hex-'5'
        ],
        [  # Λ: regions [9,6,1] are c2=2. Ordered from centre edge to vertex
            [9, [7, 1, 3]],  # shared,   Λ; cell:0x2a; i+1: c2.0=hex-'7', c2.1=hex-'1', c2.2=hex-'3'
            [6, [5, 1, 7]],  # shared,   V; cell:0x26; i+1: c2.0=hex-'5', c2.1=hex-'1', c2.2=hex-'7'
            [1, [5, 8, 2]],  # unshared, Λ; cell:0x16; i+1: c2.0=hex-'5', c2.1=hex-'8', c2.2=hex-'2'
        ],
    ]
]


def _reg_hex_lut(oob, h9r, scheme: RegionAddressLike) -> HexLUT:
    """
    Builds the Region-to-[c2/Hex] lookup table.
    Given a region address.
    """
    rg_sz = scheme.r_size
    reg_idx = np.arange(rg_sz, dtype=np.uint8)
    reg_cls = scheme.rid2cell[reg_idx]
    mc2 = h9r.mcc2[:, reg_cls]
    mc2[mc2 == h9r.invalid_region] = oob
    # 2x12x12: This is 3 region layers i=[0,1,2] at a time.  Why?
    # We should probably limit this to 2x12x12x2.
    # At i=0, this determines the mode context. We need this because regions 6..11 are shared across modes.
    # At i=1, determined by c2 context: given region x, we can identify the c2 of i=0; (+hex-group).
    # At i=2, this determines the c2 context of i=1, for which we have a hex-digit.
    # What happens if we have less than 3 regions in the list? Root has 'virtual' ancestry identical to self.
    # Root hex ids are split.  The octahedral 'actual' id is between 0..11.
    # However, the internal representation of Layer 0
    # uses the C2 value (0,1,2) of the half-hex of each face as the hex-identity for mode=0 faces,
    # and (3-C2) % 3 for mode=1 faces.
    lut = np.full((2, rg_sz, rg_sz, 2), 0x0F, dtype=np.uint8)
    base = _m_c2_hx_v2025
    for p_mo in range(2):  # This is same as the parity of p_reg
        for p_c2 in base[p_mo]:  # i=1 region, i=1 c2 hexes.
            for c_reg, c2_hex in p_c2:
                c_mo = scheme.modes[c_reg]
                for g_reg, c2 in enumerate(mc2[c_mo]):
                    lut[p_mo, c_reg, g_reg] = [c2, c2] if c2 == 0x0f else [c2_hex[c2], c2]
    return lut


def _verify_hex_reg(lut):
    ref = [[0, 0, 0, 8, 0, 2],
           [0, 1, 0, 11, 0, 2],
           [0, 0, 1, 6, 0, 0],
           [0, 1, 1, 9, 0, 0],
           [0, 0, 2, 10, 0, 1],
           [0, 1, 2, 7, 0, 1],
           [1, 0, 0, 8, 1, 1],
           [1, 1, 0, 11, 1, 1],
           [1, 0, 1, 6, 1, 2],
           [1, 1, 1, 9, 1, 2],
           [1, 0, 2, 10, 1, 0],
           [1, 1, 2, 7, 1, 0],
           [2, 0, 0, 2, 0, 0],
           [2, 0, 1, 0, 0, 1],
           [2, 0, 2, 4, 0, 2],
           [2, 1, 0, 5, 1, 0],
           [2, 1, 1, 3, 1, 1],
           [2, 1, 2, 1, 1, 2],
           [3, 0, 0, 6, 0, 0],
           [3, 0, 1, 10, 0, 1],
           [3, 0, 2, 8, 0, 2],
           [3, 1, 0, 7, 1, 0],
           [3, 1, 1, 11, 1, 1],
           [3, 1, 2, 9, 1, 2],
           [4, 0, 0, 0, 0, 1],
           [4, 1, 0, 7, 0, 1],
           [4, 0, 1, 4, 0, 2],
           [4, 1, 1, 11, 0, 2],
           [4, 0, 2, 2, 0, 0],
           [4, 1, 2, 9, 0, 0],
           [5, 0, 0, 6, 1, 2],
           [5, 1, 0, 1, 1, 2],
           [5, 0, 1, 10, 1, 0],
           [5, 1, 1, 5, 1, 0],
           [5, 0, 2, 8, 1, 1],
           [5, 1, 2, 3, 1, 1],
           [6, 1, 0, 9, 0, 0],
           [6, 0, 2, 6, 0, 0],
           [6, 0, 0, 10, 0, 1],
           [6, 1, 1, 7, 0, 1],
           [6, 0, 1, 8, 0, 2],
           [6, 1, 2, 11, 0, 2],
           ]

    for ri in ref:
        a, b, c, d, e, f = ri
        ld = lut[a, b, c]
        if np.any(ld != [d, e, f]):
            print(ri, ' ld != def', ld, [d, e, f])
        else:
            print(ri, ' is good')


def _hex_reg_lut(oob, scheme: RegionAddressLike):
    """
    Builds the Hex-to-Region lookup table.
    """
    hx_sz = 9  # 0..8 are valid; oob = 0x0F.
    # Given a hex digit, it's mode and c2, we will return the region id, it's parent-mode, and c2.
    lut = np.full((hx_sz, 2, 3, 3), oob, dtype=np.uint8)
    base = _m_c2_hx_v2025
    for p_mo in range(2):
        mo_base = base[p_mo]
        for pc2, trx in enumerate(mo_base):
            for (c_reg, c2_hx) in trx:
                c_mo = scheme.modes[c_reg]
                for c2, hx in enumerate(c2_hx):
                    lut[hx, c_mo, c2] = [c_reg, p_mo, pc2]
    return lut


def _luts(scheme: RegionAddressLike):
    """Internal helper to construct intermediate mappings."""
    hx_c2_pmo = {}
    hx_cmo_c2 = {}
    #     [  # super-region mode down (V)
    #         [  # cells of c2=0 of V super-region
    #             [6, [0, 4, 7]],  # shared,   same mode as super-region 0x26 (pL=:3)
    #             [9, [7, 4, 2]],  # shared,   diff mode to super-region 0x2a
    #             [2, [3, 6, 2]],  # unshared, same mode as super-region 0x2B
    #     ],]
    for s_mo, s_c2s in enumerate(_m_c2_hx_v2025):  # for each sc.mode (0, 1) walk through the sc.c2s.
        for sc2, rg_hx in enumerate(s_c2s):  # for each sc.c2 (0,1,2) get the regions and their hexes.
            for plc, triple in enumerate(rg_hx):  # *do* need the parent pos here.
                t_mo = (plc & 1) ^ s_mo
                (rgn, hxs) = triple
                c_mo = int(scheme.modes[rgn])  # this is the mode of this region.
                for c_c2, hx in enumerate(hxs):  # for each c2 of this region, there is a hex.
                    k1 = (t_mo, hx, rgn)
                    if k1 in hx_c2_pmo:
                        print(f'duplicate key {k1} in hx_c2_pmo')
                    hx_c2_pmo[k1] = (s_mo, sc2, c_mo, c_c2)
                    # For parent lookup
                    k2 = (t_mo, hx, s_mo, sc2)
                    if k2 in hx_cmo_c2:
                        print(f'duplicate key {k2} in hx_cmo_c2')
                    hx_cmo_c2[k2] = rgn
    for k1, v in hx_c2_pmo.items():
        if k1[1] == 0 and k1[2] == 10:  # hex=0, rgn=10
            print("hx_c2_pmo entry:", k1, "->", v)
    return hx_c2_pmo, hx_cmo_c2


def _hex_luts(h9r, scheme: RegionAddressLike) -> HexLUT:
    hex_oob = 0x0F
    rh = _reg_hex_lut(hex_oob, h9r, scheme)
    hr = _hex_reg_lut(hex_oob, scheme)
    return HexLUT(hex_oob=hex_oob, hex_reg=hr, reg_hex=rh)


H9_RA = _region_scheme(H9C, H9R)
HEX_LUTS = _hex_luts(H9R, H9_RA)


def reg_hex_digits(cx, oc, dom, tail_style: TailStyle = TailStyle.reversible, scheme: RegionAddressLike = H9_RA):
    """
    Given a region chain and an octant ID, returns the Hex Hierarchy.
    :param cx: Region chain (proto + layers).
    :param oc: Octant ID.
    :param dom: b_oct
    :param tail_style: Choose the tail_style to use.
    :param scheme: RegionAddressLike (normally H9_RA)

    Returns:
        NDArray: The canonical hex-digit hierarchy (N, L+1)
        Final byte is meta-data (full tail is reversible; partial tail is hex-binning safe).

    """
    sz, cols = np.shape(cx)
    depth = cols - 1

    # Mode per point (from octant); we could have used the region root, but we need oc.
    mo = dom.oid_mo[oc]

    # Consider region child under root.
    c2 = H9R.mcc2[mo, cx[:, 1]]

    # Hex body: one hex digit per region step away from the proto.
    bdy = np.full((sz, depth), 0x0F, dtype=np.uint8)
    if depth > 0:
        # Layer-0 hex digit anchored by (octant, c2).
        bdy[:, 0] = dom.l0hex_by_id[oc, c2]  # given the octant, and the c2 we can identify the root hexagon.
        # Remaining hex digits via region-to-hex LUT.
        reg_hex = HEX_LUTS.reg_hex
        rx = scheme.cell2rid[cx]  # This gives us region_address ids [0...11]
        p, c = rx[:, 0], rx[:, 1]  # first region will be either 0, or 1 (protos).
        h = rx[:, -1]
        p_mo = scheme.modes[p]
        for ri in range(2, rx.shape[1]):  # we will go down the p, c line of each region.
            # (2, rg_sz, rg_sz, 2)
            h = rx[:, ri]  # [p, c, h]
            hx_c2 = reg_hex[p_mo, c, h]  # This gives us the c_mode hex
            hx = hx_c2[:, 0]
            c2 = hx_c2[:, 1]
            bdy[:, ri - 1] = hx
            p, c = c, h
            p_mo = scheme.modes[p]

        # Tail metadata uses one byte:
        # bit7: parent-mode of terminating region (p_mo)
        # bits6..5: terminating c2
        # bit4: root mode (mo)
        # bits3..0: terminating region id (h)
        if tail_style is TailStyle.reversible:
            tail_ids = tail_pack_reversible(p_mo, c2, mo, h)
            return np.column_stack([bdy, tail_ids])
        if tail_style is TailStyle.key:
            tail_ids = tail_pack_key(c2, mo)
            return np.column_stack([bdy, tail_ids])
        if tail_style is TailStyle.none:
            return bdy
        raise ValueError(f"unknown tail_style: {tail_style}")
    return bdy


def hex_digits_reg(hx, dom, tail=None, scheme: RegionAddressLike = H9_RA):
    """
    Inverts `reg_hex_digits` (Hex -> Regions).

    Args:
        hx: (N, L) hex-digit addresses.
        dom: Domain object.
        tail: Optional (N,) meta-tail nibble. If None, expects it in the last column of `hx`.

    Returns:
        tuple: (octants, region_chain)
    """
    hx = np.asarray(hx, dtype=np.uint8)
    if hx.ndim != 2:
        raise ValueError("hx must be (N, L[+1]):")

    sz, cols = hx.shape
    if cols < 2 and tail is None:
        raise ValueError("hx must contain at least one hex digit and one tail nibble")

    if tail is None:
        body = hx[:, :-1]  # (N, L): root + layer hex digits
        tail = hx[:, -1]  # (N,): meta-tail
    else:
        body = hx

    # unpack meta-tail: mode + tail region-id (RegionIdScheme id)
    c_mo, c2, r_mo, tail_h = tail_unpack_reversible(tail)

    layer = body.shape[1]

    # Recover canonical octant from root hex + mode
    root_hex = body[:, 0]
    hex_reg = HEX_LUTS.hex_reg
    oob = HEX_LUTS.hex_oob

    oct_c2 = dom.l0hex_back[root_hex, r_mo]  # (N, 2): [face_id, c2_root]
    r_oct = oct_c2[:, 0]

    # ROOT super-regions mark hex digits as 0,1,2 in line with nominal-c2.
    # However, Hex-L0 c2 is a bit odd and this might need looking at.
    # I think that c2=0 is stable, but c2=1/c2=2 might be swapped under 1 mode.
    # body[:, 0] = oct_c2[:, 1]  # This should be correct.

    # Bottom-up reconstruction: [tail, ..., proto]
    depth = layer + 1
    regs = np.full((sz, depth), oob, dtype=np.uint8)
    regs[:, layer] = tail_h

    i = layer - 1
    while True:
        if i < 1:
            break
        hx_d = body[:, i]
        rmc = hex_reg[hx_d, c_mo, c2]
        regs[:, i] = rmc[:, 0]
        c_mo = rmc[:, 1]
        c2 = rmc[:, 2]
        i -= 1

    regs[:, 0] = r_mo
    cells = scheme.rid2cell[regs]  # But now it is.
    return r_oct, cells


def hex_digits(pts, depth: int = 36, tail_style: TailStyle = TailStyle.reversible, scheme: RegionAddressLike = H9_RA):
    """
    Convert Points (barycentric) to canonical hex-digit hierarchy.

    Args:
        pts (Points): Barycentric points.
        depth (int): Layer level.
        tail_style (TailStyle): whether we want a key or reversible.

    Returns:
        NDArray: Array of hex digits (and tail if requested).
    """
    import hhg9.h9.region as rg
    dom = pts.domain
    oc, mo = pts.cm()
    cx = rg.xy_regions(pts.coords, mo, depth)  # regions are length 2+'depth'
    return reg_hex_digits(cx, oc, dom, tail_style, scheme=scheme)


def hex_layer(vals, layer: int = 18, tail_style: TailStyle = TailStyle.key):
    """
    Convert Points to unique hexagon address for the layer.
    This is **lossy** because it coalesces neighbors into the central hex.

    Args:
        vals (Points): Input points.
        layer (int): Hexagon layer.
        tail_style (TailStyle): Whether to include the terminating region tail.
            Because this is most used for hex-binning, the terminating tail is normally excluded.

    Returns:
        NDArray: Hex addresses for the specific layer.
    """
    pts = neighbours(vals, layer=layer, coalesce=True)  # We now have collapsed for this layer.
    return hex_digits(pts, layer, tail_style)


def hex_str_encode(pts, depth: int = 36, tail_style: TailStyle = TailStyle.reversible, scheme: RegionAddressLike = H9_RA):
    """Convert Points (barycentric) to canonical hex string representation.

    Format: <body hex digits><tail byte>
    - body: one hex char per digit (root + layers)
    - tail: two hex chars (one byte). For `TailStyle.none`, no tail is appended.

    This is intentionally derived from `hex_digits(...)` so tail layout is centralized.
    """
    hx = hex_digits(pts, depth=depth, tail_style=tail_style, scheme=scheme)
    hx = np.asarray(hx, dtype=np.uint8)
    if hx.ndim != 2:
        raise ValueError("hex_digits must return (N, L) or (N, L+1)")

    if tail_style is TailStyle.none:
        body = hx
        return np.array([''.join(f'{int(d):01x}' for d in row) for row in body], dtype=str)

    if hx.shape[1] < 2:
        raise ValueError("expected hex_digits output to include at least one body digit and a tail")

    body = hx[:, :-1]
    tail_ids = hx[:, -1]
    body_str = np.array([''.join(f'{int(d):01x}' for d in row) for row in body], dtype=str)
    if tail_style is TailStyle.reversible:
        tail_str = np.array([f'{int(t):02x}' for t in tail_ids], dtype=str)
    else:
        tail_str = np.array([f'{int(t >> 4):01x}' for t in tail_ids], dtype=str)
    return np.char.add(body_str, tail_str)


def hex_str_decode(adr, reg=None, scheme: RegionAddressLike = H9_RA):
    """
    Convert hex strings back to Points (barycentric).

    Args:
        adr (List[str]): Input hex strings.

    Returns:
        Points: Reconstructed coordinates.
    """
    from hhg9 import Points, Registrar
    import hhg9.h9.region as rg
    if reg is None:
        reg = Registrar()
    dom = reg.domain('b_oct')

    if len(adr) == 0:
        return Points(np.zeros((0, 2), dtype=float), domain=dom, components=np.zeros((0,), dtype=np.uint8))

    # Parse: last two hex chars are the reversible tail byte.
    tail = np.array([int(s[-2:], 16) for s in adr], dtype=np.uint8)
    body_strs = [s[:-2] for s in adr]
    body_len = len(body_strs[0])
    if any(len(s) != body_len for s in body_strs):
        raise ValueError("all addresses must have the same body length")

    body = np.array([[int(ch, 16) for ch in s] for s in body_strs], dtype=np.uint8)
    hx = np.column_stack([body, tail])

    oc, cells = hex_digits_reg(hx, dom, scheme=scheme)
    xy_m = rg.regions_xy(cells)
    return Points(xy_m[:, :2], domain=dom, components=oc)


def hex_key(hx: NDArray[np.uint8], *, copy: bool = True) -> NDArray[np.uint8]:
    """Rewrite a reversible hex address into a key address by rewriting the tail byte."""
    hx = np.asarray(hx, dtype=np.uint8)
    if hx.ndim != 2 or hx.shape[1] < 2:
        raise ValueError("hx must be (N, L+1) with a tail byte")
    out = hx.copy() if copy else hx
    out[:, -1] = tail_key_from_reversible(out[:, -1])
    return out


def hex_pack(pts, depth: int = 36, scheme: RegionAddressLike = H9_RA):
    """
    Convert Points to packed UInt64 (Hex Address Format).

    Uses `HexPacker` backend.

    Args:
        pts (Points): Input points.
        depth (int): Depth of address.

    Returns:
        NDArray[uint64]: Packed integers.
    """
    from hhg9.algorithms.packing import u64_pack

    hx = hex_digits(pts, depth=depth, tail_style=TailStyle.reversible, scheme=scheme)
    hx = np.asarray(hx, dtype=np.uint8)
    if hx.ndim != 2 or hx.shape[1] < 2:
        raise ValueError("expected reversible hex digits (N, L+1)")

    body = hx[:, :-1]
    tail_ids = hx[:, -1]
    tail_hi = ((tail_ids >> 4) & 0x0F).astype(np.uint8)
    tail_lo = (tail_ids & 0x0F).astype(np.uint8)

    nibbles = np.column_stack([body, tail_hi, tail_lo]).astype(np.uint8)
    return u64_pack(nibbles)


def hex_unpack(pts, reg=None, scheme: RegionAddressLike = H9_RA):
    """
    Convert packed UInt64 (Hex Address Format) back to Points.

    Args:
        pts (NDArray[uint64]): Packed integers.

    Returns:
        Points: Reconstructed coordinates.
    """
    from hhg9.algorithms.packing import u64_layers
    from hhg9 import Points, Registrar
    import hhg9.h9.region as rg

    if reg is None:
        reg = Registrar()
    dom = reg.domain('b_oct')

    words = np.asarray(pts)
    if words.ndim != 1:
        words = words.reshape(-1)

    nibbles = np.asarray(u64_layers(words), dtype=np.uint8)
    if nibbles.ndim != 2 or nibbles.shape[1] < 3:
        raise ValueError("decoded nibble array invalid for hex_unpack")

    # Infer the used width by stripping trailing 0x0F padding columns.
    used = np.any(nibbles != 0x0F, axis=0)
    used_idx = np.flatnonzero(used)
    if used_idx.size == 0:
        raise ValueError("no non-padding nibbles found")
    last = int(used_idx[-1])
    if last < 2:
        raise ValueError("not enough nibbles to recover tail")

    tail_lo = nibbles[:, last]
    tail_hi = nibbles[:, last - 1]
    body = nibbles[:, :last - 1]

    tail_ids = ((tail_hi << 4) | tail_lo).astype(np.uint8)
    hx = np.column_stack([body, tail_ids])

    oc, cells = hex_digits_reg(hx, dom, scheme=scheme)
    xy_m = rg.regions_xy(cells)
    return Points(xy_m[:, :2], domain=dom, components=oc)


def reg_pack(pts, depth: int = 14, reg=None, scheme: RegionAddressLike = H9_RA):
    """
    Convert Points to packed UInt64 (Region Address Format).

    Uses `RegionPacker` backend.
    """
    from hhg9.algorithms.packing import u64_pack, u64_layers
    import hhg9.h9.region as rg
    if reg is None:
        from hhg9 import Registrar
        reg = Registrar()
    b_oct = reg.domain('b_oct')
    oc, mo = pts.cm()
    cx = rg.xy_regions(pts.coords, mo, depth)
    packer = region_packer(pack_fn=u64_pack, unpack_fn=u64_layers, octant_mode=lambda o: np.take(b_oct.oid_mo, o))
    rx = scheme.cell2rid[cx]
    adx = packer.encode(rx, octants=oc)
    return adx


def reg_unpack(nibs, reg=None, scheme: RegionAddressLike = H9_RA):
    """Convert packed UInt64 (Region Address Format) back to Points."""
    from hhg9.algorithms.packing import u64_pack, u64_layers
    from hhg9 import Points, Registrar
    import hhg9.h9.region as rg
    if reg is None:
        from hhg9 import Registrar
        reg = Registrar()
    b_oct = reg.domain('b_oct')
    packer = region_packer(pack_fn=u64_pack, unpack_fn=u64_layers, octant_mode=lambda o: np.take(b_oct.oid_mo, o))
    ocr, dec = packer.decode(nibs)
    cells = scheme.rid2cell[dec]
    reg_mo_rt = rg.regions_xy(cells)
    reg_rt = reg_mo_rt[:, :2]  # just want the x,y.
    pts = Points(reg_rt, b_oct, components=ocr)
    return pts
