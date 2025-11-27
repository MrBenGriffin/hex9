"""
Part of the H9 project
Author: Ben
base/h9/addressing.py
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import unique, Enum
from functools import lru_cache, cache
import numpy as np
from numpy.typing import NDArray
from hhg9.h9 import H9R, H9C
from hhg9.h9.protocols import RegionAddressLike, AddressPackerLike, H9CellLike, HexLUTLike, H9RegionLike


@unique
class Style(Enum):
    """
    Various Encoding styles.
    Most work has been done with HEX/FULL.
    However, for hexgrid binning, it may be useful to consider others.
    """
    HEX = 0
    # EXTENDED = 2
    # HALFHEX = 3
    NUMERIC = 4
    U64 = 6
    UH64 = 7
    UR64 = 8

# ---------- Region-ID scheme (even → mode 0, odd → mode 1) ----------


@dataclass(frozen=True, slots=True)
class RegionIdScheme(RegionAddressLike):
    rid2cell: NDArray[np.uint8]
    cell2rid: NDArray[np.int16]
    modes: NDArray[np.uint8]
    proto: NDArray[np.uint8]
    r_size: int


# @lru_cache(maxsize=1)
def _region_scheme(h9c: H9CellLike, h9r: H9RegionLike) -> RegionIdScheme:
    """
    Build once, frozen. Parity encodes mode (even→0, odd→1).
    keep parity==H9C.mode[rid2cell].
    """
    rid2cell = np.array([
        0x49, 0x16,  # 0,1  protos (m0, m1)
        0x2B, 0x34,  # 2,3  unshared
        0x21, 0x3E,  # 4,5  unshared
        0x26, 0x39,  # 6,7  shared
        0x35, 0x2A,  # 8,9  shared
        0x3A, 0x25,  # 10,11 shared
    ], dtype=np.uint8)

    r_size = rid2cell.size
    cell2rid = np.full(256, -1, dtype=np.int16)
    cell2rid[rid2cell] = np.arange(r_size, dtype=np.int16)

    # Sanity: anchors + parity rule
    # These two values should come from / be sanitised against region protos.
    assert rid2cell[0] == 0x49 and rid2cell[1] == 0x16
    parity = (np.arange(r_size, dtype=np.uint8) & 1)
    assert np.all(parity == h9c.mode[rid2cell]), "rid parity must match cell mode"
    proto = cell2rid[h9r.proto]
    return RegionIdScheme(rid2cell=rid2cell, cell2rid=cell2rid, modes=parity, proto=proto, r_size=r_size)


# ---------- Packer (Pack Regions) -----------------
@dataclass(frozen=True, slots=True)
class RegionPacker(AddressPackerLike):
    """
    Packs (N, L+1) region-ids into a backend representation via injected pack/unpack.
    This class only enforces the H9 region addressing *root nibble* protocol:
      - 0..7  : octant id (global, face-anchored)
      - 8, 9  : unanchored prototypes (8=Down/mode0, 9=Up/mode1)
      - A..E  : reserved
      - F     : error (unused here)

    The backend provides two callables:
      pack_fn(nibbles:(N,L+1) uint8) -> words:any
      unpack_fn(words:any, L_plus_1:int) -> (N,L+1) uint8

    Optionally, provide octant_to_proto_fn(octant:uint8)->{0,1} to map faces to proto.
    If omitted, we default to even→0, odd→1.
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

    def encode(self, reg_ids: NDArray[np.uint8], octants: NDArray[np.uint8]=None, **kwargs):
        """Build root nibble then delegate to backend pack_fn.
        :param reg_ids: (N, L+1) region ids, with column 0 holding the prototype (0 or 1).
        :param octants: If octants is provided, root nibble becomes octant (0..7) and the prototype is
        restored on decode via octant_to_proto.
        If octants is None, root nibble becomes 8 or 9 (unanchored proto tag).
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
        """Decode regions
        :returns tuple: octants, regions
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
    """Allow passing a backend object with .pack/.unpack"""
    if pack_fn is None and hasattr(unpack_fn, 'pack') and hasattr(unpack_fn, 'unpack'):
        backend = unpack_fn
        pack_fn = getattr(backend, 'pack')
        unpack_fn = getattr(backend, 'unpack')
    return RegionPacker(pack_fn=pack_fn, unpack_fn=unpack_fn, octant_mode_fn=octant_mode)


# ---------- Packer (Pack Hex Addresses) -----------------
@dataclass(frozen=True, slots=True)
class HexPacker(AddressPackerLike):
    """
    Packs hex addresses (octant + supercell c2 + hex-body + tail region) into a backend via injected pack/unpack.

    Encoding layout per address (as a *nibble* stream):
      [ octant (0..7),  c2 (0..2),  hex_0, hex_1, ..., hex_{L-1},  tail_region (0..11) ]

    - octant is the face id 0..7
    - c2 is the *supercell* c2 of the address root (0..2)
    - hex_i are hex digits 0..8 per step
    - tail_region is the terminating region id (0..11)

    Notes:
    - This keeps octant and c2 explicit (unlike the string form which folds them into a 3-char prefix).
    - We deliberately *do not* reuse the RegionPacker's 8/9 tags; this is a separate protocol.
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
        :param hex_body: (N, L) array of hex digits in 0..8
        :param octants:  (N,) array of octant ids 0..7
        :param c2s:      (N,) array of supercell c2 values 0..2
        :param tail_regions: (N,) array of terminating region ids 0..11
        :return: backend-packed words via pack_fn
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
        nibbles[:, 0] = octants             # 0..7
        nibbles[:, 1] = c2s                 # 0..2
        if L:
            nibbles[:, 2:2+L] = hex_body    # 0..8
        nibbles[:, 2+L] = (tail_regions + 2)  # 0..11 (fits in a nibble)
        return self.pack_fn(nibbles)


    def decode(self, words, layers: int | None = None, **kwargs):
        """
        Decode packed hex addresses.
        :param words: backend-packed payload
        :param layers: body length (number of hex digits). If omitted, must be provided via kwargs to unpack_fn or derivable.
        :returns: tuple (octants, c2s, hex_body, tail_regions)
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
        hex_body = nibbles[:, 2:3+layers]
        tail_regions = nibbles[:, 3+layers] - 2
        return octants, c2s, hex_body, tail_regions


@lru_cache(maxsize=1)
def hex_packer(pack_fn=None, unpack_fn=None) -> AddressPackerLike:
    """Factory returning a HexPacker using a given backend with .pack/.unpack."""
    if pack_fn is None and hasattr(unpack_fn, 'pack') and hasattr(unpack_fn, 'unpack'):
        backend = unpack_fn
        pack_fn = getattr(backend, 'pack')
        unpack_fn = getattr(backend, 'unpack')
    return HexPacker(pack_fn=pack_fn, unpack_fn=unpack_fn)


# ---------- Neighbour calculation --------------------------------------
"""
Neighbours are primarily calculated in order to coalesce half-hexagons into hexagons
at a specific layer - This opens up the ability to hex-bin at that layer.
Coalescing means that we only consider one mode (the mode we choose makes no difference),
which then gives us a 'true' hexagon address for a specific layer.
"""


def neighbours(pts, layer=32, coalesce=True):
    """
    Given a set of points, return their C2 neighbour of a given depth as Points.
    Applies a plane seam transform when an octant seam is crossed.
    Depth/layers are sometimes different, and definitely differ when considering
    regions vs. hexagons.  Hexagon Layers - what we are interested in here - are
    set such that eg: NE is a hexagon at layer 0, and NAΛ0/NAV0 are at layer 1.
    Coalesce according to the parent value of the layer to be modified.
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
    if np.any(hopped):    # the octant_spanning neighbour is merely the inverted y-axis!
        ca[hopped] = dom.oid_nb[ca[hopped], c2[hopped]]  # Adjust the octant accordingly
        ya[hopped] = -ya[hopped]
    oc[active] = ca
    coords[active, 0] = xa
    coords[active, 1] = ya
    cmp = pts.invert_octant_ids(oc)
    return Points(coords, domain=dom, components=cmp)


def neighbour_regions(pts, layer=32, coalesce=True):
    """
    Given a set of points, return their C2 (half-hex neighbour region) at a given layer as regions and octant.
    Coalesce: ensure that every *parent* region of the affected layer has the same mode.
    This allows the parent region to act as a hexbin of 3 half-hexes:
    its neighbouring regions will now be empty (after migration only).
    See ex0110_poly_neighbours.py for the mechanics.
    """
    from hhg9.h9.region import H9R, region_neighbours, xy_regions
    b_oct = pts.domain
    oc, mode = pts.cm()
    work = np.full(len(pts), 1, dtype=bool)
    regions = xy_regions(pts.coords, mode, layer)  # This will be layer+2 long
    if coalesce:
        work = H9C.mode[regions[:, -3]].astype(bool)  # Grab the mode 1. These will be turned to neighbours!
    movers = regions[work]
    xa = pts.coords[work, 0]
    ya = pts.coords[work, 1]
    ca = oc[work]
    nbr, c2 = region_neighbours(movers)
    hopped = regions[work, 0] != nbr[:, 0]  # There is a change of root mode.
    if np.any(hopped):    # the octant_spanning neighbour is merely the inverted y-axis!
        c2h = c2[hopped]
        hopped_oc = b_oct.oid_nb[ca[hopped], c2h]  # Adjust the octant accordingly
        ca[hopped] = hopped_oc
        hopped_mode = b_oct.oid_mo[hopped_oc]
        hopped_xy = np.column_stack([xa[hopped], -ya[hopped]])
        # hopped_xy *= 0.9999  # prevent jitter.
        hopped_rg = xy_regions(hopped_xy * 0.8, hopped_mode, layer)
        nbr[hopped] = hopped_rg
    regions[work] = nbr  # all un-hopped are now back in regions.
    oc[work] = ca
    return regions, oc


# ---------- Emergent hex-digit per step (optional LUT) ----------------
@dataclass(frozen=True, slots=True)
class HexLUT(HexLUTLike):
    hex_oob: int                  # 0x0F (hex out-of-bounds)
    hex_reg: NDArray[np.uint8]    # (R, R); 0..8 or 0xF for invalid
    reg_hex: NDArray[np.uint8]    # (R, R); 0..8 or 0xF for invalid


_m_c2_hx = [
    # given sc.mode, sc.c2, region, region.c2 => hex-digit.
    # ROOT super-regions mark hex digits as 0,1,2 for each c2 (the hex ID is face-dependant)
    # Note that mode V has a cluster of 3 '4' hexes around its origin.
    # whereas mode Λ has a cluster of 3 '5' hexes around its origin.
    [  # super-region mode down (V)
        [  # region 'c' of c2=0 of V super-region
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


def _reg_hex_lut(h9r, h9c, scheme: RegionAddressLike) -> HexLUT:
    """
    Purpose of lut: given p_reg, c_reg, gc_reg  => hex.
    The hex value returned is the c2 of the c_reg that the gc_reg is in.
    """
    in_cells = h9c.in_scope
    mc2 = h9r.mcc2[:, in_cells]
    in_reg = scheme.cell2rid[in_cells]
    rg_sz = scheme.r_size
    # bc in_reg is unordered, we need to use the index for luts.
    reg_to_idx = np.full(rg_sz, 0x0F, dtype=np.uint8)
    reg_to_idx[in_reg] = np.arange(in_reg.size, dtype=np.uint8)
    lut = np.full((rg_sz, rg_sz, rg_sz), 0x0F, dtype=np.uint8)
    for p_reg in in_reg:
        p_mode = scheme.modes[p_reg]
        c2_hxs = _m_c2_hx[p_mode]
        for triple in c2_hxs:  # don't need the parent c2 here.
            for (c_reg, c2_hx) in triple:  # each half-hex is 3 regions.
                c_mode = scheme.modes[c_reg]
                c2r = mc2[c_mode]     # This will have 3 * [0,1,2,95] respectively; showing the c2 of each region (in_reg indexed)
                for g_reg in in_reg:  # grand-child.
                    g_idx = reg_to_idx[g_reg]
                    c2 = c2r[g_idx]
                    if c2 < 3:  # otherwise it's out of bounds.
                        lut[p_reg, c_reg, g_reg] = c2_hx[c2]
    return lut


def _luts(scheme: RegionAddressLike):
    # A hex digit has two halves: keyed by (cell.c2, cell, supercell.mode)
    # such that each cell.c2 is paired with it's long-edge neighbour.
    # hx_c2_pmo: given hx, and the cell it is a c2 of
    # return the supercell mode, supercell c2, child-mode, hex-c2 (within child region).
    # eg, (0,8) => 0,2,0,2
    #     (4,7) => 0,1,1,2
    # Generate hx_cmo_c2: given a hex digit, the cell.mode, and it's c2, find the region.
    # given a hex digit, it's c2 region_mode, and it's c2, find the region.
    hx_c2_pmo = {}
    hx_cmo_c2 = {}
    for s_mo, s_c2s in enumerate(_m_c2_hx):         # for each sc.mode (0, 1) walk through the sc.c2s.
        for sc2, rg_hx in enumerate(s_c2s):         # for each sc.c2 (0,1,2) get the regions and their hexes.
            for (rgn, hxs) in rg_hx:                # there will be 3 regions and each will have 3 hexes.
                c_mo = int(scheme.modes[rgn])       # this is the mode of this region.
                for c_c2, hx in enumerate(hxs):     # for each c2 of this region, there is a hex.
                    k1 = (hx, rgn)
                    if k1 in hx_c2_pmo:
                        print(f'duplicate key {k1} in hx_c2_pmo')
                    hx_c2_pmo[k1] = (s_mo, sc2, c_mo, c_c2)
                    k2 = (hx, c_mo, c_c2)
                    if k2 in hx_cmo_c2:
                        print(f'duplicate key {k2} in hx_cmo_c2')
                    hx_cmo_c2[(hx, c_mo, c_c2)] = rgn
    # for hx in range(9):
    #     for rgn in range(12):
    #         k1 = (hx, rgn)
    #         if k1 in hx_c2_pmo:
    #             print(f'{k1} => {hx_c2_pmo[k1]}')
    # for hx in range(9):
    #     for c_mo in range(2):
    #         for c2 in range(3):
    #             k2 = (hx, c_mo, c2)
    #             if k2 in hx_cmo_c2:
    #                 print(f'{k2} => {hx_cmo_c2[k2]}')
    return hx_c2_pmo, hx_cmo_c2


def _hex_reg_lut(scheme: RegionAddressLike):
    # Purpose of lut: given p_hex, c_hex, c_region => p_region.
    # The c_region is a region where one of it's c2 is the c_hex.
    hx_c2_pmo, hx_cmo_c2 = _luts(scheme)
    in_cells = H9C.in_scope
    in_reg = scheme.cell2rid[in_cells]
    hx_sz = 9  # 9 hexagon regions 0...8
    rg_sz = in_reg.shape[0]
    lut = np.full((hx_sz, hx_sz, rg_sz), 0x0F, dtype=np.uint8)
    # Here ch/pa are relative values.
    for ch_hx in range(hx_sz):               # given a child_hex...
        for c_rg in in_reg:                  # and a regions...
            if (ch_hx, c_rg) in hx_c2_pmo:   # does this hex appear as a c2 of the region?
                s_mo, s_c2, c_mo, c_c2 = hx_c2_pmo[(ch_hx, c_rg)]
                for pa_hx in range(hx_sz):
                    pa_rg = hx_cmo_c2[pa_hx, s_mo, s_c2]
                    if lut[pa_hx, ch_hx, c_rg] != 0x0F:
                        print(f'duplicate key {pa_hx, ch_hx, c_rg} in hex_reg. {lut[pa_hx, ch_hx, c_rg]} found; {pa_rg} overwrites.')
                    lut[pa_hx, ch_hx, c_rg] = pa_rg
    # for pa_hx in range(hx_sz):
    #     for ch_hx in range(hx_sz):
    #         for rg in range(rg_sz):
    #             if lut[pa_hx, ch_hx, rg] != 0x0F:
    #                 print(f'p:{pa_hx}, c:{ch_hx}, r:{rg:01x} => {int(lut[pa_hx, ch_hx, rg]):01x}')
    return lut


def _hex_luts(h9r, h9c, scheme: RegionAddressLike) -> HexLUT:
    rh = _reg_hex_lut(h9r, h9c, scheme)
    hr = _hex_reg_lut(scheme)
    return HexLUT(hex_oob=0x0F, hex_reg=hr, reg_hex=rh)


H9_RA = _region_scheme(H9C, H9R)
HEX_LUTS = _hex_luts(H9R, H9C, H9_RA)


def reg_hex_digits(cx, oc, dom):
    """
    Convert Points (barycentric) to hex string.
    """
    sz, depth = np.shape(cx)
    depth -= 1
    mo = dom.oid_mo[oc]          # we could have used the region root - but we need oc.
    c2 = H9R.mcc2[mo, cx[:, 1]]  # consider region child under root.
    bdy = np.full([sz, depth], 0xf, dtype=np.uint8)
    bdy[:, 0] = dom.l0hex_by_id[oc, c2]
    if depth == 0:
        return bdy
    reg_hex = HEX_LUTS.reg_hex
    rx = H9_RA.cell2rid[cx]
    p, c = rx[:, 0], rx[:, 1]
    for ri in range(2, rx.shape[1]):
        h = rx[:, ri]
        hx = reg_hex[p, c, h]
        bdy[:, ri-1] = hx
        p, c = c, h
    return bdy


def hex_digits(pts, depth: int = 36, scheme: RegionAddressLike = H9_RA):
    """
    Convert Points (barycentric) to hex string.
    """
    import hhg9.h9.region as rg
    sz = len(pts)

    dom = pts.domain
    oc, mo = pts.cm()
    cx = rg.xy_regions(pts.coords, mo, depth)
    c2 = H9R.mcc2[mo, cx[:, 1]]  # ignore the terminating region
    bdy = np.full([sz, depth+1], 0xf, dtype=np.uint8)
    bdy[:, 0] = dom.l0hex_by_id[oc, c2]
    if depth == 0:
        return bdy
    reg_hex = HEX_LUTS.reg_hex
    rx = scheme.cell2rid[cx]
    p, c = rx[:, 0], rx[:, 1]
    for ri in range(2, rx.shape[1]):
        h = rx[:, ri]
        hx = reg_hex[p, c, h]
        bdy[:, ri-1] = hx
        p, c = c, h
    return bdy


def hex_layer(vals, layer: int = 18):
    """
    Convert Points (barycentric) to unique hexagon address for the layer
    This is not invertible - it's lossy because neighbours will be coalesced.
    Coalescing depends upon the parent layer mode being identified.
    For Layer 0, this will be the octant's mo
    Then the c2 for each of those will be moved to its neighbour.
    """
    import hhg9.h9.region as rg
    from hhg9 import Points
    if not isinstance(vals, Points):
        raise TypeError('pts must be a Points object')
    coalesce = True
    b_oct = vals.domain
    result = np.full((len(vals), layer+1), 0x0F, dtype=np.uint8)
    ok = b_oct.valid(vals)  # don't even try for invalid points.
    ref = Points(vals.coords[ok].copy(), b_oct, vals.components[ok].copy())
    pts = neighbours(ref, layer=layer, coalesce=coalesce)  # We now have collapsed for this layer.
    result[ok] = hex_digits(pts, layer)
    return result


def hex_str_encode(pts, depth: int = 36, reg=None, scheme: RegionAddressLike = H9_RA):
    """Convert Points (barycentric) to hex string"""
    import hhg9.h9.region as rg
    sz = len(pts)
    reg_hex = HEX_LUTS.reg_hex
    oc, mo = pts.cm()
    dom = pts.domain  # b_oct
    o_c2_str = dom.oc_c2()
    cx = rg.xy_regions(pts.coords, mo, depth)
    rx = scheme.cell2rid[cx]
    r1 = cx[:, 1]
    mc2 = H9R.mcc2[mo, r1]   # The mode comes from points, the r1 gives us the c2.
    pfx = o_c2_str[oc, mc2]  # prefix for each point.
    bdy = np.full([sz, depth+2], 0xf, dtype=np.uint8)
    p, c = rx[:, 0], rx[:, 1]
    for ri in range(2, rx.shape[1]):
        h = rx[:, ri]
        hx = reg_hex[p, c, h]
        if np.any(hx == 0x0f):
            fx = np.where(hx == 0x0f)
            pm = scheme.modes[p]
            for ax in fx[0]:
                print(f'at index {ri} p=>c=>h {int(p[ax])}(m:{pm[ax]})=>{int(c[ax])}=>{int(h[ax])} is illegal.')
                print(cx[ax])
                print(rx[ax])
        bdy[:, ri-2] = hx
        p, c = c, h
    c_mo = scheme.modes[c]
    h = scheme.proto[c_mo]
    bdy[:, -2] = reg_hex[p, c, h]
    bdy[:, -1] = rx[:, -1]  # append terminating region
    body = np.array([''.join([f'{int(i):01x}' for i in row.tolist()]) for row in bdy])
    pb = np.strings.add(pfx, body)
    return pb


def hex_str_decode(adr, reg=None, scheme: RegionAddressLike = H9_RA):
    """Convert hex string array to Points (barycentric)"""
    from hhg9 import Points, Registrar
    import hhg9.h9.region as rg
    if reg is None:
        reg = Registrar()
    dom = reg.domain('b_oct')
    sz = len(adr)
    hex_reg = HEX_LUTS.hex_reg
    pfx = np.vectorize(lambda s: s[:3])(adr)
    rgn = np.vectorize(lambda s: int(s[-1], 16))(adr).astype(np.uint8)
    body = np.array([[int(ch, 16) for ch in s[3:-1][::-1]] for s in adr], dtype=np.uint8)
    hexes = body.shape[1]
    depth = hexes + 1  # for region terminator
    oc = np.vectorize(lambda s: dom.h9map[s]['id'])(pfx).astype(np.uint8)
    mo = np.vectorize(lambda s: dom.h9map[s]['mode'])(pfx).astype(np.uint8)
    regs = np.zeros([sz, depth], dtype=np.uint8)  # add the region (tail)
    regs[:, 0] = rgn
    hx_c = body[:, 0]
    for hx_i in range(1, hexes):
        hx_p = body[:, hx_i]
        rgn = hex_reg[hx_p, hx_c, rgn]  # hex_reg: p_hex, c_hex, c_region => p_region
        hx_c = hx_p
        regs[:, hx_i] = rgn
    regs[:, -1] = scheme.proto[mo]
    cells = np.flip(scheme.rid2cell[regs], axis=1)
    xy_m = rg.regions_xy(cells)
    result = Points(xy_m[:, :2], domain=dom, components=oc)
    return result


def hex_pack(pts, depth: int = 36, reg=None, scheme: RegionAddressLike = H9_RA):
    """Convert Points (barycentric) to nibble packed uint64"""
    from hhg9.algorithms.packing import u64_pack, u64_layers
    import hhg9.h9.region as rg
    sz = len(pts)
    reg_hex = HEX_LUTS.reg_hex
    hp = hex_packer(pack_fn=u64_pack, unpack_fn=u64_layers)
    oc, mo = pts.cm()
    cx = rg.xy_regions(pts.coords, mo, depth)
    rx = scheme.cell2rid[cx]
    c2 = H9R.mcc2[mo, cx[:, 1]]   # The mode comes from points, the r1 gives us the c2.
    body = np.full([sz, depth+1], 0xf, dtype=np.uint8)
    p, c = rx[:, 0], rx[:, 1]
    for ri in range(2, rx.shape[1]):
        h = rx[:, ri]
        hx = reg_hex[p, c, h]
        body[:, ri-2] = hx
        p, c = c, h
    c_mo = scheme.modes[c]
    h = scheme.proto[c_mo]
    body[:, -1] = reg_hex[p, c, h]
    t_reg = rx[:, -1]  # append terminating region
    words = hp.encode(body, oc, c2, t_reg)
    return words


def hex_unpack(pts, reg=None, scheme: RegionAddressLike = H9_RA):
    """Convert Points (barycentric) to nibble packed uint64"""
    from hhg9.algorithms.packing import u64_pack, u64_layers
    from hhg9 import Points, Registrar
    import hhg9.h9.region as rg
    if reg is None:
        reg = Registrar()
    dom = reg.domain('b_oct')
    sz = len(pts)
    hp = hex_packer(pack_fn=u64_pack, unpack_fn=u64_layers)
    oc, c2, body_f, rgn = hp.decode(pts)
    mo = dom.oid_mo[oc]
    hex_reg = HEX_LUTS.hex_reg
    body = body_f[:, ::-1]
    regs = np.zeros([sz, body.shape[1] + 1], dtype=np.uint8)  # add the region (tail)
    regs[:, 0] = rgn
    hx_c = body[:, 0]
    for hx_i in range(1, body.shape[1]):
        hx_p = body[:, hx_i]
        rgn = hex_reg[hx_p, hx_c, rgn]  # hex_reg: p_hex, c_hex, c_region => p_region
        hx_c = hx_p
        regs[:, hx_i] = rgn
    regs[:, -1] = scheme.proto[mo]
    cells = np.flip(scheme.rid2cell[regs], axis=1)
    xy_m = rg.regions_xy(cells)
    result = Points(xy_m[:, :2], domain=dom, components=oc)
    return result


def reg_pack(pts, depth: int = 14, reg=None, scheme: RegionAddressLike = H9_RA):
    """Convert Points (barycentric) to nibble packed uint64"""
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
    """nibble packed uint64 to Convert Points (barycentric)"""
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


# -------------------- POC: proofs of concept examples ---------------------

def poc_region_encode():
    """PoC region encode"""
    from hhg9 import Points, Registrar
    from hhg9.algorithms import gcd_rnd, wgs84
    from hhg9.algorithms.packing import u64_pack, u64_layers
    import hhg9.h9.region as rg
    reg = Registrar()
    b_oct = reg.domain('b_oct')
    samples = 1000
    scheme = H9_RA
    locr = gcd_rnd(samples)
    refs = Points(locr, 'g_gcd')
    b_refs = reg.project(refs, ['g_gcd', 'b_oct'])   # components filled
    adx = reg_pack(b_refs, 14, reg)
    # oc, mo = b_refs.cm()
    # cx = rg.xy_regions(b_refs.coords, mo, 14)
    # packer = region_packer(pack_fn=u64_pack, unpack_fn=u64_layers, octant_mode=lambda o: np.take(b_oct.oid_mo, o))
    # r_scheme = H9_RA
    # rx = r_scheme.cell2rid[cx]
    # adx = packer.encode(rx, octants=oc)
    # xc = u64_pack(rx)
    cv = u64_layers(adx)
    print(f'xc: {adx[0][0]:0x}')
    print("nibbles:", cv[0, :8])
    prt = reg_unpack(adx, reg, scheme)
    # ocr, dec = packer.decode(adx)
    # sanity: decoded proto (col 0) must equal mapping(octant)
    # assert np.array_equal(dec[:, 0], np.take(b_oct.oid_mo, oc)), "proto restored from octant mismatch"
    # assert np.array_equal(ocr, oc), "octant restore failed"
    # cells = scheme.rid2cell[dec]
    # reg_mo_rt = rg.regions_xy(cells)
    # reg_rt = reg_mo_rt[:, :2]  # just want the x,y.
    # prt = Points(reg_rt, b_oct, components=ocr)
    r_tp = reg.project(prt, ['b_oct', 'g_gcd'])   # components filled
    b_deltas = wgs84(refs.coords, r_tp.coords) * 1e+0
    print(f"mean(delta): {np.mean(b_deltas)}m; max(b_deltas): {np.max(b_deltas)}m")


def poc_stonehenge_h9():
    """hex9 address of stonehenge."""
    from hhg9 import Points, Registrar
    from hhg9.algorithms import wgs84
    reg = Registrar()
    ref = Points(np.array([[51.1787980000210725, -1.8261898473293335]]), 'g_gcd')
    b_ref = reg.project(ref, ['g_gcd', 'b_oct'])   # components filled
    ad = hex_str_encode(b_ref)
    print(ad[0])
    ptb = hex_str_decode(ad, reg)
    #  reference value (from above)                      51.1787980000210725, -1.8261898473293335
    b_bak = reg.project(ptb, ['b_oct', 'g_gcd'])  # 51.17879800002107,   -1.8261898473293454
    rx = wgs84(ref.coords, b_bak.coords) * 1e+9
    print(f'{rx[0]}nm')  #


def poc_hex_str_rnd():
    """hex9 address of rnd."""
    from hhg9 import Points, Registrar
    from hhg9.algorithms import wgs84
    from hhg9.algorithms import gcd_rnd
    # from hhg9.h9.region import neighbours
    reg = Registrar()
    locr = gcd_rnd(1)
    ref = Points(locr, 'g_gcd')
    b_ref = reg.project(ref, ['g_gcd', 'b_oct'])   # components filled
    depth = 8
    h9x = hex_str_encode(b_ref, depth)
    # print(f'{h9x}')
    n_ref = neighbours(b_ref, depth)
    h9n = hex_str_encode(n_ref, depth)
    k_ref = neighbours(n_ref, depth)
    h9k = hex_str_encode(k_ref, depth)
    for (o, n, k, p, c) in zip(h9x, h9n, h9k, b_ref.coords, b_ref.components):
        print(f'{o}={k} via {n}, ({p}):{c}')
    # pbk = hex_str_decode(h9x, reg)
    # b_bak = reg.project(pbk, ['b_oct', 'g_gcd'])  # 51.17879800002107,   -1.8261898473293454
    # rx = wgs84(ref.coords, b_bak.coords) * 1e+9
    # print(f'{rx}')  #


def poc_hex_pak():
    """hex9 address of rnd."""
    from hhg9 import Points, Registrar
    from hhg9.algorithms import wgs84
    from hhg9.algorithms import gcd_rnd
    reg = Registrar()
    locr = gcd_rnd(100)
    ref = Points(locr, 'g_gcd')
    b_ref = reg.project(ref, ['g_gcd', 'b_oct'])   # components filled
    packed = hex_pack(b_ref, 12)                  # 3 metres - not great!
    rtp = hex_unpack(packed, reg)
    b_bak = reg.project(rtp, ['b_oct', 'g_gcd'])  # 51.17879800002107,   -1.8261898473293454
    rx = wgs84(ref.coords, b_bak.coords)
    print(f'{rx}')  #


if __name__ == "__main__":
    # poc_hex_str_rnd()
    # poc_hex_pak()
    poc_stonehenge_h9()
    # poc_region_encode()
