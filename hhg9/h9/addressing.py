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
L*[0...8]  Within each hexagon, there are a group of six full hexagons of the subsequent hex_layer,
           and six half-hexagons of the subsequent hex_layer.  They are all numbered between 0..8
           The specific pattern is documented elsewhere.
1*[mm|reg] Metadata; Without recognising the region-net_mode of the terminal hexagon, there is some ambiguity
           Therefore, we need a digit to indicate the region-net_mode.  It is also useful to record
           the root net_mode (which of the octahedron faces this address belongs to).
           Likewise, we want to record the terminating region (0..11) in order to recover an address in full.

           For the bin-hex key tail, both p_c2 and r_mo are essential: p_c2 disambiguates the terminal region within its parent (digit 6 can arise from multiple regions at different parent c2 values), and r_mo disambiguates the root octant (two octants of opposite mode can produce the same root hex digit).

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
from hhg9.h9 import H9R, H9C, H9K
from hhg9.h9.protocols import RegionAddressLike, AddressPackerLike, H9CellLike, HexLUTLike, H9RegionLike, BaryLoc
from hhg9.h9.tail import TailStyle, tail_pack_reversible, tail_pack_key, tail_unpack_reversible, \
    tail_key_from_reversible, tail_unpack_key


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
    UH64K = 7  # DEPRECATED & unwired (redundant + buggy key tail); use UH64A canonical bin
    UH64A = 8  # reversible uint64 address
    UR64 = 9


# ---------- Region-ID scheme (even → net_mode 0, odd → net_mode 1) ----------


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
    c2: NDArray[np.uint8]
    props: NDArray[np.uint8]
    proto: NDArray[np.uint8]
    r_size: int


# @lru_cache(maxsize=1)
def _region_scheme(h9c: H9CellLike, h9r: H9RegionLike) -> RegionIdScheme:
    """
    Builds the Region ID Scheme once and freezes it.

    Enforces that parity equals net_mode (even→0, odd→1).

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
        [  # net_mode 0
            # s, s, u    s,  s, u    s, s,  u   Shared/unshared
            [6, 9, 2], [10, 7, 0], [8, 11, 4]  # c2=0,1,2
        ], [  # net_mode 1
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
    mc_c2 = np.full((2, 12), 0x0F, dtype=np.uint8)
    for mo in [0, 1]:
        for c2, rx in enumerate(H9C.c2[mo]):
            for cx in rx:
                rg = cell2rid[cx]
                mc_c2[mo, rg] = c2

    # mc_c2 = np.array([[mo, cell2rid[cx], c2] for mo in [0, 1] for c2, rx in enumerate(H9C.c2[mo]) for cx in rx], dtype=np.uint8)
    # Enforce parity==net_mode only for in-bounds cells.
    # rid2cell[12..15] are OOB placeholders (0x5F) and should not participate in the parity check.
    oob_cell = np.uint8(0x5F)
    valid = rid2cell != oob_cell
    assert np.all(parity[valid] == h9c.mode[rid2cell[valid]]), "rid parity must match cell net_mode (excluding OOB)"
    proto = cell2rid[h9r.proto]
    return RegionIdScheme(rid2cell=rid2cell, cell2rid=cell2rid, modes=parity, props=mo_c2, c2=mc_c2, proto=proto,
                          r_size=r_size)


# ---------- Packer (Pack Regions) -----------------
@dataclass(frozen=True, slots=True)
class RegionPacker(AddressPackerLike):
    """
    Packs (hex_layer, L+1) region-ids into a backend representation.

    This class enforces the H9 region addressing **root nibble** protocol:

    * **0..7:** Octant ID (global, face-anchored).
    * **8, 9:** Unanchored prototypes (8=Down/Mode0, 9=Up/Mode1).
    * **A..E:** Reserved.
    * **F:** Error.

    Attributes:
        pack_fn: Callable taking nibbles -> packed words.
        unpack_fn: Callable taking packed words -> nibbles.
        octant_mode_fn: Optional callable mapping octant ID -> net_mode (0/1).
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
            reg_ids: (hex_layer, L+1) array of region IDs. Column 0 must be prototype {0, 1}.
            octants: Optional (hex_layer,) array of octant IDs.
                * If provided, root nibble becomes octant (0..7).
                * If None, root nibble becomes 8 or 9 (unanchored proto tag).
        """
        reg_ids = np.asarray(reg_ids, dtype=np.uint8)
        assert reg_ids.ndim == 2, "reg_ids must be (hex_layer, L+1)"
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
            raise ValueError("Decoded address has non-proto at hex_layer 0")
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
    3.  **Hex Digits (0..8):** The body of the address, one nibble per hex_layer.
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
            hex_body: (hex_layer, L) array of hex digits in 0..8.
            octants: (hex_layer,) array of octant IDs 0..7.
            c2s: (hex_layer,) array of supercell c2 values 0..2.
            tail_regions: (hex_layer,) array of terminating region IDs 0..11.

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
            raise ValueError("hex_body must be (hex_layer, L)")
        N, L = hex_body.shape
        if octants.shape != (N,):
            raise ValueError("octants must be shape (hex_layer,)")
        if c2s.shape != (N,):
            raise ValueError("c2s must be shape (hex_layer,)")
        if tail_regions.shape != (N,):
            raise ValueError("tail_regions must be shape (hex_layer,)")
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
    At a specific hex_layer, 3 "half-hex" triangles meet at a vertex. To form a valid
    Hexagon Grid for binning, these three must be merged (coalesced) into one logical hexagon.
    This involves checking the parent hex_layer's net_mode and adjusting the C2 cluster accordingly.

    Args:
        pts (Points): The input barycentric points.
        layer (int): The depth at which to calculate neighbors/hexagons.
        coalesce (bool): If True, merges triangles into hexagons.

    Returns:
        Points: New points representing the neighbour/coalesced centre - in same order, same number.
    """
    from hhg9.h9.region import region_neighbours, regions_xy, xy_regions
    from hhg9.h9.classifier import location
    from hhg9.h9 import H9O
    from hhg9 import Points

    dom = pts.domain
    oc, mode = pts.cm()
    oc = oc.copy()  # prevent mutation of pts.oid through view aliasing
    coords = pts.coords.copy()
    x = coords[:, 0]
    y = coords[:, 1]
    c = oc[:]
    # active = np.full(len(pts), 1, dtype=bool)     # [-0.6896473544905836, 0.3780076763554287]
    u, v = H9K.R3 * x, y
    locs = location(u, v, mode)
    active = locs == BaryLoc.INT  # external,vertex,edge - not moving.
    regions = xy_regions(coords[active], mode[active], layer)  # no depth?!
    if coalesce:
        local_m0 = H9C.mode[regions[:, -2]].astype(bool)
        active[active] = local_m0  # only net_mode 1 (net_mode 0 -> false).
        regions = regions[local_m0]
    xa = x[active]
    ya = y[active]
    ca = c[active]
    nbr, c2 = region_neighbours(regions)
    hopped = regions[:, 0] != nbr[:, 0]
    xym = regions_xy(nbr[~hopped])
    xa[~hopped] = xym[:, 0]
    ya[~hopped] = xym[:, 1]
    if np.any(hopped):  # the octant_spanning neighbour is merely the inverted y-axis!
        ca[hopped] = H9O.oid_nb[ca[hopped], c2[hopped]]  # Adjust the octant accordingly
        ya[hopped] = -ya[hopped]
    oc[active] = ca
    coords[active, 0] = xa
    coords[active, 1] = ya
    return Points(coords, domain=dom, oid=oc)


# ---------- Emergent hex-digit per step (optional LUT) ----------------
@dataclass(frozen=True, slots=True)
class HexLUT(HexLUTLike):
    """Container for the massive Region-to-Hex lookup tables."""
    hex_oob: int
    hex_reg: NDArray[np.uint8]
    reg_hex: NDArray[np.uint8]


_m_c2_hx_v2025 = [
    # This is the late-version (2025/2026):
    # - net_mode 0 has a cluster of 3 '0' hexes around its origin.
    # - net_mode 1 has a cluster of 3 '1' hexes around its origin.
    # Layer i+1 hexes will have a cluster of 3 '2' hexes at the centres
    #     of the hex_layer i+0 0/1/2 (and 3/4/5, 6/7/8) clusters
    # This dict is the ground-truth for all hexagon digits.
    # It considers the digits from the (triangular) region/super-region context.
    # Consider an equilateral triangle at Layer i.  In hhg9, this is divided into 3 half-hexes (aka c2) at Layer i.
    # - because each triangle in hhg9 is divided into 9 triangles (regions), each c2 contains 3 hex_layer i+1 regions,
    # each having (according to its net_mode) 3 half-hexes.
    # regions are 'shared' or 'unshared'; six regions are shared across both modes. six regions are 1-net_mode only.
    # Given a Li; net_mode j, it's Li+1 hexagons are shared with every other Li; net_mode j triangle.
    # The hexagon=>sub-hexagon relationships look different, but are emergent from the definition as above.
    # Within every hexagon there will be child hexagons 0,1,2,3,4,5 and 3 'split' pairs of half-hexagons 6,7,8.
    # The splits are such that they do not share a c2.  For example, the two '6' half-hexagons might be in
    # modes [0, 2].  '6' half-hexes are 'wings' of the '0' hexagon, '7' half-hexes are the 'wings' of '1' hexagon,
    # and '8' half-hexes are the 'wings' of the '2' hexagon
    [  # Layer 'i+0'; super-region net_mode 0 (V), by c2 orientation, referenced by region-id (0..11) (centred with 0-hex)
        # Note: hex digits ['1', '5', '7'] are not found in i+1 of net_mode 0.
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
    [  # Layer 'i+0'; super-region net_mode 0 (Λ), by c2 orientation, referenced by region-id (0..11) (centred with 1-hex)
        # Note: hex digits ['0', '4', '6'] are not found in i+1 of  net_mode 1:
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
    # At i=0, this determines the net_mode context. We need this because regions 6..11 are shared across modes.
    # At i=1, determined by c2 context: given region x, we can identify the c2 of i=0; (+hex-group).
    # At i=2, this determines the c2 context of i=1, for which we have a hex-digit.
    # What happens if we have less than 3 regions in the list? Root has 'virtual' ancestry identical to self.
    # Root hex ids are split.  The octahedral 'actual' id is between 0..11.
    # However, the internal representation of Layer 0
    # uses the C2 value (0,1,2) of the half-hex of each face as the hex-identity for net_mode=0 faces,
    # and (3-C2) % 3 for net_mode=1 faces.
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
    # Given a hex digit, it's net_mode and c2, we will return the region id, it's parent-net_mode, and c2.
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
    #     [  # super-region net_mode down (V)
    #         [  # cells of c2=0 of V super-region
    #             [6, [0, 4, 7]],  # shared,   same net_mode as super-region 0x26 (pL=:3)
    #             [9, [7, 4, 2]],  # shared,   diff net_mode to super-region 0x2a
    #             [2, [3, 6, 2]],  # unshared, same net_mode as super-region 0x2B
    #     ],]
    for s_mo, s_c2s in enumerate(_m_c2_hx_v2025):  # for each sc.net_mode (0, 1) walk through the sc.c2s.
        for sc2, rg_hx in enumerate(s_c2s):  # for each sc.c2 (0,1,2) get the regions and their hexes.
            for plc, triple in enumerate(rg_hx):  # *do* need the parent pos here.
                t_mo = (plc & 1) ^ s_mo
                (rgn, hxs) = triple
                c_mo = int(scheme.modes[rgn])  # this is the net_mode of this region.
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


def _r_adr_forward(oc, rx, c2_root, tail_style: TailStyle, scheme: RegionAddressLike):
    """Forward (root->leaf) engine: region-id chain -> hex digits + tail.

    Shared by reg_hex_digits (which derives rx/c2_root from a c_dig chain)
    and r_adr_to_x_adr (which takes the rid chain directly).

    Args:
        oc:      (N,) octant ids.
        rx:      (N, L+2) region-id (rid, 0..11) chain: [proto, root, levels].
        c2_root: (N,) c2 of the root hexagon within the octant.
        tail_style, scheme: as reg_hex_digits.
    """
    from hhg9.h9 import H9O
    sz, cols = np.shape(rx)
    depth = cols - 1
    r_mo = H9O.oid_mo[oc]
    c2 = c2_root

    # Hex body: one hex digit per region step away from the proto.
    bdy = np.full((sz, depth), 0x0F, dtype=np.uint8)
    if depth > 0:
        # Layer-0 hex digit anchored by (octant, c2).
        bdy[:, 0] = H9O.l0hex_by_id[oc, c2]  # given the octant, and the c2 we can identify the root hexagon.
        # Remaining hex digits via region-to-hex LUT.
        reg_hex = HEX_LUTS.reg_hex
        p, c = rx[:, 0], rx[:, 1]  # first region will be either 0, or 1 (protos).
        h = rx[:, -1]
        p_mo = scheme.modes[p]
        for ri in range(2, rx.shape[1]):  # we will go down the tri_points, c line of each region.
            # (2, rg_sz, rg_sz, 2)
            h = rx[:, ri]  # [tri_points, c, h]
            hx_c2 = reg_hex[p_mo, c, h]  # This gives us the c_mode hex
            hx = hx_c2[:, 0]
            c2 = hx_c2[:, 1]
            bdy[:, ri - 1] = hx
            p, c = c, h
            p_mo = scheme.modes[p]
        # now looped- normalise the tail.
        t_h = np.array([[6, 10, 8], [7, 11, 9]])[p_mo, c2]  # This is region of hex 3.
        hx_c2 = reg_hex[p_mo, p, t_h]
        c2 = hx_c2[:, 1]
        # Tail metadata uses one byte:
        # bit7: parent-net_mode of terminating region (par_mode)
        # bits6..5: terminating c2
        # bit4: root net_mode (mo)
        # bits3..0: terminating region id (h)
        if tail_style is TailStyle.reversible:
            # Single-nibble tail: p_mo is pinned 0 (canonical home); decode
            # supplies a fixed terminal region from c2, so only (c2, r_mo) ship.
            tail_ids = tail_pack_reversible(r_mo, p_mo, c2)
            return np.column_stack([bdy, tail_ids])
        if tail_style is TailStyle.key:
            tail_ids = tail_pack_key(r_mo, p_mo, c2)
            return np.column_stack([bdy, tail_ids])
        if tail_style is TailStyle.none:
            return bdy
        raise ValueError(f"unknown tail_style: {tail_style}")
    return bdy


def reg_hex_digits(cx, oc, dom, tail_style: TailStyle = TailStyle.reversible, scheme: RegionAddressLike = H9_RA):
    """
    Given a region chain and an octant ID, returns the Hex Hierarchy.
    :param cx: Region chain (proto + layers), as c_dig (classifier cell) values.
    :param oc: Octant ID.
    :param dom: b_oct
    :param tail_style: Choose the tail_style to use.
    :param scheme: RegionAddressLike (normally H9_RA)

    Returns:
        NDArray: The canonical hex-digit hierarchy (hex_layer, L+1)
        Final byte is meta-data (full tail is reversible; partial tail is hex-binning safe).

    """
    from hhg9.h9 import H9O
    if dom.name[:5] != 'b_oct':
        raise ValueError(f"reg_hex_digits: domain must be a b_oct, not {dom}")
    r_mo = H9O.oid_mo[oc]
    c2 = H9R.mcc2[r_mo, cx[:, 1]]

    # Points whose root cell is the invalid-region sentinel (0x5F) cannot be addressed.
    # These arise when neighbours() coalesces a point that lands exactly on a simplex
    # boundary.  Clamp c2 to 0 so downstream indexing doesn't crash; the resulting hex
    # addresses will be deduplicated away or overwritten by valid neighbours.
    c2 = np.where(c2 == 0x5F, np.uint8(0), c2)
    rx = scheme.cell2rid[cx]  # This gives us region_address ids [0...11]
    return _r_adr_forward(oc, rx, c2, tail_style, scheme)


def canonicalise(coords, oc, mo, dom, layer, scheme: RegionAddressLike = H9_RA):
    """Fold mode-1 leaf half-hexes to their canonical mode-0 parent ("bin" form).

    Packer-agnostic core of the canonical layer-L key. The three half-hexes
    meeting at a vertex (a mode-1 parent "split") share one binning hexagon, so
    after the fold every cell has a mode-0 terminal parent (p_mo == 0) and the
    (c2, r_mo) tail alone identifies it — which is why address == bin and bins are
    invertible. Folding is a property of the *cell*, so EDGE/VERTEX points fold
    too; only out-of-scope (EXT/UDF) points are left untouched. A fold that
    crosses an octant seam switches the octant via H9O.oid_nb (+ y-flip).

    Args:
        coords: (N, 2) b_oct coordinates.
        oc, mo: octant id and net_mode (e.g. from Points.cm()).
        dom:    b_oct domain.
        layer:  hierarchy depth.
        scheme: RegionAddressLike (normally H9_RA).

    Returns:
        (regions, oc): the canonical region chain (N, layer+2) and the
        possibly-octant-switched octant id. Feed to reg_hex_digits / reg_pack /
        the UUID packer — the single source of truth for canonical addresses.
    """
    from hhg9.h9 import H9O
    from hhg9.h9.region import region_neighbours, xy_regions
    from hhg9.h9.classifier import location
    oc = np.asarray(oc).copy()
    x, y = coords[:, 0], coords[:, 1]
    regions = xy_regions(coords, mo, layer)                     # (N, layer+2)
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
    return regions, oc


def hex_digits_reg(dom, hx, tail=None, scheme: RegionAddressLike = H9_RA, place_terminal=True):
    """
    Inverts `reg_hex_digits` (Hex -> Regions).

    Args:
        hx: (hex_layer, L) hex-digit addresses.
        dom: Domain object.
        tail: Optional (hex_layer,) meta-tail nibble. If None, expects it in the last column of `hex_points`.
        place_terminal: if True (default) append the canonical "3" terminal regions
            (a shared-vertex geometry proxy) below the body. Set False to omit them
            so the caller can instead seed regions_xy at the cell centroid; the
            backward walk (and hence the recovered cell chain) is identical either
            way, only the within-cell geometry placement differs.

    Returns:
        tuple: (octants, region_chain)
    """
    from hhg9.h9 import H9O
    hx = np.asarray(hx, dtype=np.uint8)
    if hx.ndim != 2:
        raise ValueError("hex_points must be (hex_layer, L[+1]):")
    if hx.shape[1] < 2 and tail is None:
        raise ValueError("hex_points must contain at least one hex digit and one tail nibble")

    if tail is None:
        body = hx[:, :-1]  # (hex_layer, L): root + hex_layer hex digits
        tail = hx[:, -1]   # (hex_layer,): meta-tail
    else:
        body = hx
    sz, ncols = body.shape

    # Unpack the tail. p_mo is read (not forced to 0) so RAW chains still decode;
    # canonical addresses simply carry p_mo == 0. c2 seeds the backward walk.
    c2, r_mo, c_mo = tail_unpack_reversible(tail)
    c2 = np.asarray(c2).astype(np.intp)
    c_mo = np.asarray(c_mo).astype(np.intp)

    hex_reg = HEX_LUTS.hex_reg
    oob = HEX_LUTS.hex_oob

    # Recover canonical octant from root hex + net_mode
    root_hex = body[:, 0]
    oct_c2 = H9O.l0hex_back[root_hex, r_mo]  # (hex_layer, 2): [face_id, c2_root]
    r_oct = oct_c2[:, 0]

    rids, real_layer, _root_ctx = _x_adr_backwalk(body, c_mo, c2)

    tail_h = np.array([[6, 10, 8], [7, 11, 9]], dtype=np.uint8)[c_mo, c2]   # region "3"
    seed_reg = hex_reg[3, c_mo, c2][:, 0]

    rows = np.arange(sz)
    regs = np.full((sz, ncols + 2), oob, dtype=np.uint8)
    regs[:, :ncols] = rids
    if place_terminal:
        regs[rows, real_layer + 1] = seed_reg
        regs[rows, real_layer + 2] = tail_h

    regs[:, 0] = r_mo
    cells = scheme.rid2cell[regs]
    return r_oct, cells


def _x_adr_backwalk(body, c_mo, c2):
    """Backward (leaf->root, right-to-left) engine over x_adr body digits.

    The tail-derived leaf context (c_mo, c2) seeds the walk; each body digit
    resolves through hex_reg to its region id (rid) and lifts the context one
    level. This bottom-up unzip is what makes x_list interpretable (glossary:
    split x_cells force right-to-left reading).

    Args:
        body: (N, ncols) body nibbles [root hex, digit_1..], 0x0F padded.
        c_mo, c2: (N,) leaf context from the unpacked tail.
    Returns:
        (rids, real_layer, (cm, cc)): per-level rids (N, ncols; col 0 unused,
        0x0F beyond each row's layer), each row's layer, and the recovered
        ROOT context (the L0 hexagon's (mode, c2) within the octant proto).
    """
    hex_reg = HEX_LUTS.hex_reg
    oob = HEX_LUTS.hex_oob
    sz, ncols = body.shape

    # Per-row layer = the deepest non-sentinel body nibble. Full addresses have no
    # sentinel (-> ncols-1); a bin carries 0xF beyond its layer, so this is how
    # the same decoder handles both (the "skip 0xF" the bin case needs).
    is_real = body != oob
    real_layer = (ncols - 1) - np.argmax(is_real[:, ::-1], axis=1)

    # Canonical "3" terminal: undo the 3-step (hex_reg[3, c_mo, c2]) to recover the
    # (c_mo, c2) context AT the deepest real body level.
    e3 = hex_reg[3, c_mo, c2]
    cm = e3[:, 1].astype(np.intp)
    cc = e3[:, 2].astype(np.intp)

    # Backward walk: each row starts at its own real_layer (rows in their sentinel
    # zone stay inactive and keep their seed context untouched).
    rids = np.full((sz, ncols), oob, dtype=np.uint8)
    for i in range(int(real_layer.max()), 0, -1):
        active = i <= real_layer
        d = np.where(active, body[:, i], 0)          # avoid indexing hex_reg with 0xF
        e = hex_reg[d, cm, cc]
        rids[active, i] = e[active, 0]
        cm = np.where(active, e[:, 1], cm)
        cc = np.where(active, e[:, 2], cc)
    return rids, real_layer, (cm, cc)


# ---------- Address-form conversions: x_adr <-> r_adr <-> d_adr --------------
# Three views of one hierarchy path (glossary taxonomy):
#   x_adr: root hex + x_dig chain + tail. What UUIDs store. Digits alone are
#          slot names — split x_cells force the bottom-up unzip to interpret.
#   r_adr: (oc, [proto, e_0 .. e_L]): the thread's region (t_cell) id per
#          scale — the same chain the encoder produces (canonicalise ->
#          cell2rid), with e_L normalised to the canonical "3" terminal.
#          Consecutive pairs (e_{i-1}, e_i) carry the digits (reg_hex);
#          octant-relative, so it travels with oc.
#   d_adr: (N, L+1, 2) [digit, mode] pairs, row 0 = (root hex, r_mo).
#          Ben's "combined x_cell/cell_mode = d_cell address": the mode of
#          digit i is parity(e_{i-1}) = scheme.modes[e_{i-1}] — the d_cell
#          side the thread runs through at that level. Self-contained: the
#          octant follows from row 0, and the region thread reconstructs
#          top-down because (digit, parent-ctx, child-mode) -> child-c2 is
#          unique (the two hex_reg preimages of a parent context always
#          differ in child mode).


@lru_cache(maxsize=1)
def _hex_reg_inv():
    """Inverse context map: [digit, p_mo, p_c2, c_mo] -> (c_c2, rid).

    hex_reg maps (digit, child ctx) -> (rid, parent ctx) two-to-one per
    digit; the child MODE disambiguates, so with it the downward walk is
    exact. 0xF entries are unreachable (invalid digit/mode pairings).
    """
    hex_reg = HEX_LUTS.hex_reg
    inv = np.full((9, 2, 3, 2, 2), 0x0F, dtype=np.uint8)
    for d in range(9):
        for cm in range(2):
            for cc in range(3):
                rid, pm, pc = hex_reg[d, cm, cc]
                inv[d, pm, pc, cm] = (cc, rid)
    return inv


def x_adr_to_r_adr(hx, tail=None, scheme: RegionAddressLike = H9_RA):
    """x_adr -> r_adr: recover the thread's region-id chain of an x_adr.

    The bottom-up recovery: the tail seeds the leaf context and each digit
    resolves to the region (t_cell) the thread occupies one scale up.

    Args:
        hx: (N, L+1[+1]) body nibbles (root hex + digits[, tail]), 0x0F padded.
        tail: (N,) tail nibbles if not in hx's last column.
    Returns:
        (oc, r_adr): octant ids (N,) and region chain (N, L+2):
        [proto (= r_mo), e_0 .. e_L], 0x0F padded beyond each row's layer.
    """
    from hhg9.h9 import H9O
    hx = np.asarray(hx, dtype=np.uint8)
    if tail is None:
        body, tail = hx[:, :-1], hx[:, -1]
    else:
        body = hx
    c2, r_mo, c_mo = tail_unpack_reversible(tail)
    c2 = np.asarray(c2).astype(np.intp)
    c_mo = np.asarray(c_mo).astype(np.intp)
    rids, real_layer, _ = _x_adr_backwalk(body, c_mo, c2)
    oc = H9O.l0hex_back[body[:, 0], r_mo][:, 0]
    sz, ncols = body.shape
    r_adr = np.full((sz, ncols + 1), HEX_LUTS.hex_oob, dtype=np.uint8)
    r_adr[:, 0] = r_mo
    r_adr[:, 1:ncols] = rids[:, 1:]                       # e_0 .. e_{L-1}
    seed_reg = HEX_LUTS.hex_reg[3, c_mo, c2][:, 0]        # e_L ("3" terminal)
    r_adr[np.arange(sz), real_layer + 1] = seed_reg
    return oc, r_adr


def r_adr_to_x_adr(oc, r_adr, tail_style: TailStyle = TailStyle.reversible,
                   scheme: RegionAddressLike = H9_RA):
    """r_adr -> x_adr: forward walk emitting the hex-digit body + tail.

    Rows must share one layer (group mixed-layer batches by layer first).

    Args:
        oc:    (N,) octant ids.
        r_adr: (N, L+2) region chain as returned by x_adr_to_r_adr, trimmed
               or 0x0F-padded uniformly.
    Returns:
        (N, L+2) body nibbles + tail (per tail_style).
    """
    from hhg9.h9 import H9O
    rx = np.asarray(r_adr)
    oob = HEX_LUTS.hex_oob
    real = rx[0] != oob
    if not np.array_equal(rx != oob, np.broadcast_to(real, rx.shape)):
        raise ValueError('r_adr_to_x_adr: rows must share one layer')
    rx = rx[:, real]
    r_mo = H9O.oid_mo[oc]
    c2_root = H9R.mcc2[r_mo, np.asarray(scheme.rid2cell)[rx[:, 1]]]
    return _r_adr_forward(oc, rx, c2_root, tail_style, scheme)


def x_adr_to_d_adr(hx, tail=None, scheme: RegionAddressLike = H9_RA):
    """x_adr -> d_adr: the digit chain zipped with its recovered modes.

    Returns:
        d_adr (N, L+1, 2): [digit, mode] per level; row 0 = (root hex, r_mo);
        (0x0F, 0x0F) beyond each row's layer. The mode of digit i is
        parity(e_{i-1}) — which d_cell side the thread runs through at that
        level: mode 1 at a split digit (6/7/8) means the containing parent
        is the lineage parent's neighbour.
    """
    hx = np.asarray(hx, dtype=np.uint8)
    if tail is None:
        body, tl = hx[:, :-1], hx[:, -1]
    else:
        body, tl = hx, np.asarray(tail)
    c2, r_mo, c_mo = tail_unpack_reversible(tl)
    c2 = np.asarray(c2).astype(np.intp)
    c_mo = np.asarray(c_mo).astype(np.intp)
    rids, _, _ = _x_adr_backwalk(body, c_mo, c2)
    oob = HEX_LUTS.hex_oob
    pad = body == oob
    modes = np.where(pad, oob, np.asarray(scheme.modes, dtype=np.uint8)[rids & 0xF])
    modes[:, 0] = r_mo                                    # root row: (hex, r_mo)
    digits = np.where(pad, oob, body)
    return np.stack([digits, modes], axis=-1)


def d_adr_to_x_adr(d_adr, tail_style: TailStyle = TailStyle.reversible,
                   scheme: RegionAddressLike = H9_RA):
    """d_adr -> x_adr: rebuild the body + tail from (digit, mode) pairs.

    The c2 thread is reconstructed top-down via the inverse context map;
    the leaf tail is the digit-3 step's unique inverse. Rows must share one
    layer.
    """
    oc, r_adr, tl = _d_adr_walk(d_adr)
    if tail_style is TailStyle.reversible:
        dm = np.asarray(d_adr)
        body = dm[:, dm[0, :, 0] != HEX_LUTS.hex_oob, 0]
        return np.column_stack([body, tl])
    return r_adr_to_x_adr(oc, r_adr, tail_style, scheme)


def d_adr_to_r_adr(d_adr, scheme: RegionAddressLike = H9_RA):
    """d_adr -> r_adr: thread the context top-down, collecting rids.

    Returns (oc, r_adr) in x_adr_to_r_adr's layout. Rows must share one layer.
    """
    oc, r_adr, _ = _d_adr_walk(d_adr)
    return oc, r_adr


def r_adr_to_d_adr(oc, r_adr, scheme: RegionAddressLike = H9_RA):
    """r_adr -> d_adr: digits via the forward walk, modes from rid parity."""
    hx = r_adr_to_x_adr(oc, r_adr, TailStyle.reversible, scheme)
    return x_adr_to_d_adr(hx, scheme=scheme)


@lru_cache(maxsize=1)
def _y_mirror():
    """Classifier-cell id map under the octant-seam frame flip (y -> -y).

    A descent chain mirrors entry-wise: every level's frame transform
    (offset subtract, x3 rescale) commutes with the flip, so the
    neighbour-octant presentation of a chain is the mirrored chain — the
    symbolic form of canonicalise's geometric xy_regions(flipped)
    re-descend. Out-of-scope slots mirror to sentinels and never appear in
    in-scope chains.
    """
    from hhg9.h9 import H9C, H9K
    from hhg9.h9.classifier import classify_cell
    off = H9C.off_xy
    return np.asarray(classify_cell(H9K.R3 * off[:, 0], -off[:, 1]), dtype=np.uint8)


def x_adr_cell_ancestor(hx, target: int, tail=None, scheme: RegionAddressLike = H9_RA):
    """Canonical layer-``target`` ancestor of each cell, in address space.

    The pure-address form of the mode-0 d_cell doctrine (no geometry, no
    nudge — exact at any depth): truncate the recovered region thread at
    the target layer; where the presentation there is mode-1 (the thread
    came up the far side of a split), fold to the canonical mode-0
    registration — region_neighbours' upward cascade within the octant,
    the symbolic seam mirror when the fold crosses octants — then re-emit
    digits + canonical tail through the forward walk.

    Args:
        hx / tail: as x_adr_to_r_adr. Rows must share one layer L > target.
        target: the ancestor layer.
    Returns:
        (oc, hx_out): octant ids and the canonical (N, target+2) body+tail.
    """
    from hhg9.h9 import H9C, H9O
    from hhg9.h9.region import region_neighbours
    oc, r_adr = x_adr_to_r_adr(hx, tail, scheme)
    oob = HEX_LUTS.hex_oob
    real = r_adr[0] != oob
    if not np.array_equal(r_adr != oob, np.broadcast_to(real, r_adr.shape)):
        raise ValueError('x_adr_cell_ancestor: rows must share one layer')
    layer = int(real.sum()) - 2
    if not 0 <= target < layer:
        raise ValueError('x_adr_cell_ancestor: need 0 <= target < layer')
    cx = np.asarray(scheme.rid2cell)[r_adr[:, :target + 2]]
    oc = np.asarray(oc).copy()

    # Fold mode-1 presentations to the canonical side (cf. canonicalise).
    active = np.asarray(H9C.mode)[cx[:, -2]] == 1
    if target > 0 and np.any(active):
        idx = np.flatnonzero(active)
        nbr, c2 = region_neighbours(cx[idx])
        hopped = cx[idx, 0] != nbr[:, 0]                # octant-spanning fold
        cx[idx[~hopped]] = nbr[~hopped]
        if np.any(hopped):
            hidx = idx[hopped]
            oc[hidx] = H9O.oid_nb[oc[hidx], c2[hopped]]
            cx[hidx] = _y_mirror()[cx[hidx]]            # seam = y-flip, symbolically

    r_mo = H9O.oid_mo[oc]
    c2r = H9R.mcc2[r_mo, cx[:, 1]]
    c2r = np.where(c2r == 0x5F, np.uint8(0), c2r)
    rx = np.asarray(scheme.cell2rid)[cx]
    out = _r_adr_forward(oc, rx, c2r, TailStyle.reversible, scheme)
    if target == 0:
        # L0 bins canonicalise to the mode-0 octant rep (h9_bin_pts' L0
        # branch): the root hexagon is shared by two octants; the canonical
        # tail carries its c2 in the mode-0 one, r_mo = 0.
        m0 = H9O.l0hex_back[out[:, 0], 0]               # [oid, c2] mode-0 side
        out[:, 1] = (m0[:, 1] & 3) << 1
        oc = m0[:, 0]
    return oc, out


def _d_adr_walk(d_adr):
    """Top-down context thread over (digit, mode) pairs.

    Each step's inverse lookup yields the region the thread occupies one
    scale up (e_{i-1}); the leaf region e_L and the reversible tail come
    from the unique inverse of the terminal digit-3 step.

    Returns (oc, r_adr, tail): octants, the region chain [proto, e_0..e_L],
    and the tail nibble.
    """
    from hhg9.h9 import H9O
    dm = np.asarray(d_adr)
    oob = HEX_LUTS.hex_oob
    real = dm[0, :, 0] != oob
    if not np.array_equal(dm[..., 0] != oob, np.broadcast_to(real, dm.shape[:2])):
        raise ValueError('d_adr walk: rows must share one layer')
    dm = dm[:, real]
    sz, cols = dm.shape[:2]
    root_hex, r_mo = dm[:, 0, 0], dm[:, 0, 1] & 1
    oct_c2 = H9O.l0hex_back[root_hex, r_mo]
    oc = oct_c2[:, 0]
    cc = oct_c2[:, 1].astype(np.intp)                 # root c2
    cm = r_mo.astype(np.intp)                         # root mode
    inv = _hex_reg_inv()
    r_adr = np.full((sz, cols + 1), oob, dtype=np.uint8)
    r_adr[:, 0] = r_mo
    for i in range(1, cols):
        d, m = dm[:, i, 0], dm[:, i, 1] & 1
        step = inv[d, cm, cc, m]                      # (N, 2): child c2, e_{i-1}
        if np.any(step[:, 0] == oob):
            raise ValueError('d_adr walk: invalid digit/mode for context')
        r_adr[:, i] = step[:, 1]
        cc = step[:, 0].astype(np.intp)
        cm = m.astype(np.intp)
    # Leaf: unique inverse of the terminal digit-3 step from the leaf ctx.
    hex_reg = HEX_LUTS.hex_reg
    e3 = hex_reg[3]                                   # (2, 3, 3): (rid, pm, pc)
    t_inv = np.full((2, 3, 2), oob, dtype=np.uint8)   # [pm, pc] -> (c_mo, c2)
    for tm in range(2):
        for tc in range(3):
            _, pm, pc = e3[tm, tc]
            t_inv[pm, pc] = (tm, tc)
    tmc = t_inv[cm, cc]
    r_adr[:, cols] = hex_reg[3, tmc[:, 0], tmc[:, 1]][:, 0]   # e_L
    tail = tail_pack_reversible(r_mo, tmc[:, 0], tmc[:, 1])
    return oc, r_adr, tail


def hex_digits(pts, layer: int = 36, tail_style: TailStyle = TailStyle.reversible, scheme: RegionAddressLike = H9_RA):
    """
    Convert Points (barycentric) to canonical hex-digit hierarchy.

    Args:
        pts (Points): Barycentric points.
        layer (int): Layer layer.
        tail_style (TailStyle): whether we want a key or reversible.

    Returns:
        NDArray: Array of hex digits (and tail if requested).
    """
    import hhg9.h9.region as rg
    dom = pts.domain
    oc, mo = pts.cm()
    cx = rg.xy_regions(pts.coords, mo, layer)  # regions are length 2+'depth'
    return reg_hex_digits(cx, oc, dom, tail_style, scheme=scheme)


def hex_layer(vals, layer: int = 18, tail_style: TailStyle = TailStyle.key):
    """
    Convert Points to unique hexagon address for the hex_layer.
    This is **lossy** because it coalesces neighbours into the central hex.

    Args:
        vals (Points): Input points.
        layer (int): Hexagon hex_layer.
        tail_style (TailStyle): Whether to include the terminating region tail.
            Because this is most used for hex-binning, the terminating tail is normally excluded.

    Returns:
        NDArray: Hex addresses for the specific hex_layer.
    """
    pts = neighbours(vals, layer=layer, coalesce=True)  # We now have collapsed for this hex_layer.
    return hex_digits(pts, layer, tail_style)


def hex_str_encode(pts, layer: int = 36, tail_style: TailStyle = TailStyle.reversible,
                   scheme: RegionAddressLike = H9_RA):
    """Convert Points (barycentric) to canonical hex string representation.

    Format: <body hex digits><tail byte>
    - body: one hex char per digit (root + layers)
    - tail: two hex chars (one byte). For `TailStyle.none`, no tail is appended.

    This is intentionally derived from `hex_digits(...)` so tail layout is centralized.
    """
    hx = hex_digits(pts, layer=layer, tail_style=tail_style, scheme=scheme)
    hx = np.asarray(hx, dtype=np.uint8)
    if hx.ndim != 2:
        raise ValueError("hex_digits must return (hex_layer, L) or (hex_layer, L+1)")

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


def hex_str_decode(adr, dom=None):
    """
    Convert hex strings back to Points (barycentric).
    Args:
        adr (List[str]): Input hex strings.
    Returns:
        Points: Reconstructed coordinates.
    """
    tail = np.array([int(s[-2:], 16) for s in adr], dtype=np.uint8)
    body_strs = [s[:-2] for s in adr]
    body_len = len(body_strs[0])
    if any(len(s) != body_len for s in body_strs):
        raise ValueError("all addresses must have the same body length")
    body = np.array([[int(ch, 16) for ch in s] for s in body_strs], dtype=np.uint8)
    hx = np.column_stack([body, tail])
    return hex_decode(hx, dom)


def hex_decode(adr, dom, scheme: RegionAddressLike = H9_RA):
    """
    Convert hex key array back to Points (barycentric).
    Args:
        adr ndarray of digits.
    Returns:
        Points: Reconstructed coordinates.
    """
    from hhg9 import Points
    import hhg9.h9.region as rg
    oc, cells = hex_digits_reg(dom, adr, scheme=scheme)
    xy_m = rg.regions_xy(cells)
    return Points(xy_m[:, :2], domain=dom, oid=oc)


def hex_key(hx: NDArray[np.uint8], *, copy: bool = True) -> NDArray[np.uint8]:
    """Rewrite a reversible hex address into a key address by rewriting the tail byte."""
    hx = np.asarray(hx, dtype=np.uint8)
    if hx.ndim != 2 or hx.shape[1] < 2:
        raise ValueError("hex_points must be (hex_layer, L+1) with a tail byte")
    out = hx.copy() if copy else hx
    out[:, -1] = tail_key_from_reversible(out[:, -1])
    return out


def hex_pack(pts, depth: int = 36, tail_style: TailStyle = TailStyle.reversible,
             scheme: RegionAddressLike = H9_RA, canonical: bool = True):
    """
    Convert Points to packed UInt64 (Hex Address Format).

    Uses `HexPacker` backend.

    Args:
        pts (Points): Input points.
        depth (int): Depth of address.
        tail_style (TailStyle): Controls how the tail metadata is packed.
            - reversible: tail is packed as one nibble (full reversible metadata;
              p_mo is canonically 0 so (p_c2, r_mo) fit a single nibble).
            - key: tail is packed as one nibble (high nibble only, binning key).
            - none: no tail is packed.
        canonical (bool): When True (default), fold mode-1 leaves to their mode-0
            parent first, so the address is a reliable unique BIN (address == bin):
            two points in the same hexagon pack to the same u64. Set False for the
            raw representative that preserves the exact terminal half-hex.
    Returns:
        NDArray[uint64]: Packed integers.
    """
    from hhg9.algorithms.packing import u64_pack

    if canonical:
        oc, mo = pts.cm()
        regions, oc = canonicalise(pts.coords, oc, mo, pts.domain, depth, scheme=scheme)
        hx = reg_hex_digits(regions, oc, pts.domain, tail_style, scheme=scheme)
    else:
        hx = hex_digits(pts, layer=depth, tail_style=tail_style, scheme=scheme)
    hx = np.asarray(hx, dtype=np.uint8)
    if hx.ndim != 2:
        raise ValueError("expected hex digits as (hex_layer, L) or (hex_layer, L+1)")

    if tail_style is TailStyle.none:
        nibbles = hx.astype(np.uint8)
        return u64_pack(nibbles)

    if hx.shape[1] < 2:
        raise ValueError("expected hex digits output to include at least one body digit and a tail")

    body = hx[:, :-1]
    tail_ids = hx[:, -1]

    if tail_style is TailStyle.reversible:
        # The reversible tail is a SINGLE nibble (see reg_hex_digits /
        # tail_pack_reversible: (p_mo<<3)|(p_c2<<1)|r_mo, values 0..13). It was
        # historically split into hi/lo nibbles, but the high nibble is always 0,
        # so packing it wasted one nibble — and at uint64 width that costs a whole
        # layer at the worst point. Pack the single tail nibble directly so a
        # single u64 reaches one layer deeper (sub-metre vs metre representatives).
        tail_lo = (tail_ids & 0x0F).astype(np.uint8)
        nibbles = np.column_stack([body, tail_lo]).astype(np.uint8)
        return u64_pack(nibbles)

    if tail_style is TailStyle.key:
        # Key tail is stored as a single nibble (high nibble). Low nibble is sentinel 0xF and is not packed.
        tail_hi = ((tail_ids >> 4) & 0x0F).astype(np.uint8)
        nibbles = np.column_stack([body, tail_hi]).astype(np.uint8)
        return u64_pack(nibbles)

    raise ValueError(f"unknown tail_style: {tail_style}")


def hex_unpack(pts, tail_style: TailStyle = TailStyle.reversible, reg=None, scheme: RegionAddressLike = H9_RA):
    """
    Convert packed UInt64 (Hex Address Format) back to Points.

    Args:
        pts (NDArray[uint64]): Packed integers.
        tail_style (TailStyle): Controls how the tail metadata is unpacked.
            - reversible: expects tail as two nibbles.
            - key: expects tail as one nibble (high nibble); not invertible for TailStyle.none.
            - none: not invertible (raises error).
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
    # if words.ndim != 1:
    #     words = words.reshape(-1)

    nibbles = np.asarray(u64_layers(words), dtype=np.uint8)
    if nibbles.ndim != 2 or nibbles.shape[1] < 2:
        raise ValueError("decoded nibble array invalid for hex_unpack")

    # Infer the used width by stripping trailing 0x0F padding columns.
    used = np.any(nibbles != 0x0F, axis=0)
    used_idx = np.flatnonzero(used)
    if used_idx.size == 0:
        raise ValueError("no non-padding nibbles found")
    last = int(used_idx[-1])

    if tail_style is TailStyle.none:
        raise ValueError("hex_unpack cannot invert TailStyle.none: no tail metadata")

    if tail_style is TailStyle.reversible:
        if last < 1:
            raise ValueError("not enough nibbles to recover reversible tail")
        # Single-nibble reversible tail (mirror of hex_pack): the final used nibble
        # is the tail; everything before it is the body.
        tail_ids = nibbles[:, last].astype(np.uint8)
        body = nibbles[:, :last]
        hx = np.column_stack([body, tail_ids])
        oc, cells = hex_digits_reg(dom, hx, scheme=scheme)

    elif tail_style is TailStyle.key:
        # Key tail is stored as a single nibble (high nibble); reconstruct sentinel low nibble (0xF).
        tail_hi = nibbles[:, last]
        body = nibbles[:, :last]
        tail_ids = ((tail_hi.astype(np.uint8) << 4) | np.uint8(0x0F)).astype(np.uint8)
        hx = np.column_stack([body, tail_ids])
        oc, cells = hex_digits_reg(dom, hx, scheme=scheme)

    else:
        raise ValueError(f"unknown tail_style: {tail_style}")

    xy_m = rg.regions_xy(cells)
    return Points(xy_m[:, :2], domain=dom, oid=oc)


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
    from hhg9 import Points
    from hhg9.h9 import H9O
    import hhg9.h9.region as rg
    if reg is None:
        from hhg9 import Registrar
        reg = Registrar()
    b_oct = reg.domain('b_oct')
    packer = region_packer(pack_fn=u64_pack, unpack_fn=u64_layers, octant_mode=lambda o: np.take(H9O.oid_mo, o))
    ocr, dec = packer.decode(nibs)
    cells = scheme.rid2cell[dec]
    reg_mo_rt = rg.regions_xy(cells)
    reg_rt = reg_mo_rt[:, :2]  # just want the x,y.
    pts = Points(reg_rt, b_oct, oid=ocr)
    return pts

