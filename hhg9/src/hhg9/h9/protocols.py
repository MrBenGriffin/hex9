"""
Part of the H9 project
Author: Ben
hhg9/h9/protocols.py
This defines the contracts of the core objects being defined in h9 files.
"""
from __future__ import annotations  # PEP 563 (here to support python 3.8–3.10)

from enum import unique, IntEnum
from typing import Protocol, runtime_checkable  # (runtime_checkable allows isinstance)
import numpy as np
from typing import Tuple
from numpy.typing import NDArray


@unique
class BaryLoc(IntEnum):
    """
    Marks various locations of barycentric coordinates.
    """
    UDF = 0  # undefined
    EXT = 1
    INT = 2
    EDG = 3
    VTX = 4


@runtime_checkable
class FundamentalsLike(Protocol):
    """
    Canonical is found in constants.py
    Two base radicals: √3, and √2
    """
    R3: float  # √3
    W: float   # √2 (W: 'width' of a unit octahedron face)


@runtime_checkable
class DerivedLike(Protocol):
    """
    Canonical is found in constants.py
    Values derived once from Fundamentals; used across inequalities and transforms."""
    W: float  # Edge length of the *full* barycentric triangle: √2 (mirrors Fundamentals.W for convenience)
    H: float  # Height of the supercell triangle: H = W * √3 / 2
    Ḣ: float  # 1 * Height of child cells: Ḣ = H / 3; used in inequality tests
    Ẇ: float  # 2 * Height of child cells: Ẇ = 2 * Ḣ; used in inequality tests; alias of ΛC
    RH: float  # Convenience: √3 / 2 (useful for symmetric transforms)


@runtime_checkable
class LimitsLike(Protocol):
    """
    Canonical is found in constants.py
    Horizontal (TR, TL) and Vertical bounds for Λ and V triangles
    """
    TR: float  # +ve right-hand x-limit in classifier plane:  TR = H / √3 = √2/2 = W/2
    TL: float  # -ve left-hand  x-limit in classifier plane:  TL = -TR = -√2/2  = -W/2
    # the following are in barycentric-y order
    ΛC: float  # mode 1 supercell barycentric ceiling (not apex)! y:  +2·Ḣ
    VC: float  # mode 0 supercell barycentric ceiling (not apex)! y:  +1·Ḣ
    # The origin, y=0, is here.                                   y:   0·Ḣ
    ΛF: float  # mode 1 supercell barycentric floor (not base)!   y:  -1·Ḣ
    VF: float  # mode 0 supercell barycentric floor (not base)    y:  -2·Ḣ


@runtime_checkable
class LatticeLike(Protocol):
    """
    Canonical is found in constants.py
    Rectilinear lattice constants, use multiples of (U,V)
    as unit multipliers to calculate cell offsets.
    The barycentre of every equilateral triangle (cell) tiling
    the plane may then be identified using integer co-ordinates(x,y)
    such that it's cartesian position is at (x*U,y*V).
    Not every u,v will be a barycentre, but every barycentre is integer.
    """
    U: float  # horizontal unit: W / 6
    V: float  # vertical   unit: H / 9


@runtime_checkable
class H9ConstLike(Protocol):
    """
    Canonical is found in constants.py
    H9 Constants
    """
    radical: FundamentalsLike
    derived: DerivedLike
    limits: LimitsLike
    lattice: LatticeLike


@runtime_checkable
class H9ClassifierLike(Protocol):
    """
    Canonical is found in classifier.py
    Packed thresholds used by cell classification; precomputed to avoid drift/rebuilds.
    """
    h_levels: NDArray[np.float64]  # h_levels: horizontal tiers (y vs constants): [ΛC, VC, 0, ΛF, VF]
    p_levels: NDArray[np.float64]  # p_levels: positive slope (y - ẋ) vs [+Ẇ, 0, -Ẇ]
    n_levels: NDArray[np.float64]  # n_levels: negative slope (y + ẋ) vs [-Ẇ, 0, +Ẇ]
    mode_0_lim: Tuple[float, float]  # (VF barycentric floor, VC barycentric ceiling)
    mode_1_lim: Tuple[float, float]  # (ΛF barycentric floor, ΛC barycentric ceiling)
    encode: NDArray[np.uint8]        # shape (6,4,4)
    decode: NDArray[np.uint8]        # shape (96,3)
    eps: np.floating


@runtime_checkable
class H9CellLike(Protocol):
    """
    Cell properties
    concerned with the structure of the lattice for barycentric addressing
    """
    def __init__(self):
        self.off_ẋy = None

    """
    Canonical is found in lattice.py
    Cell Properties
    """
    count: int                    # count of cells in classifier.
    mode: NDArray[np.uint8]       # cell mode (0=down, 1=up)
    off_uv: NDArray[np.int8]      # uv co-ordinate. uv ∈ [-9..9]
    off_xy: NDArray[np.float64]   # metric barycentric co-ordinate (x, y)
    off_ẋy: NDArray[np.float64]   # metric √3x co-ordinate (ẋ, y)
    in_scope: NDArray[np.uint8]   # array of in-scope cell ids
    in_mode: NDArray[bool]        # (count) bools indicating if cell is in the mode.
    in_dn: NDArray[bool]          # (count) bools indicating if cell is in the down supercell
    in_up: NDArray[bool]          # (count) bools indicating if cell is in the up supercell
    downs: NDArray[np.uint8]      # array of cells in down supercell
    ups: NDArray[np.uint8]        # array of cells in up supercell
    c2: NDArray[np.uint8]         # [2, 3, 3] cells for each mode/c2 cluster


@runtime_checkable
class H9RegionLike(Protocol):
    """
    Canonical is found in region.py
    Region Constants
    """
    invalid_region: int         # 0x5f might eventually become 0x0f
    proto: NDArray[np.uint8]    # up/dn as a mode-ordered array. 0/1
    proto_up: int               # virtual up (0x16 to new proto)
    proto_dn: int               # virtual dn (0x49 to new proto)
    ids: NDArray[np.uint8]      # region_ids
    is_in: NDArray[bool]        # ids length indicating in_scope
    downs: NDArray[np.uint8]    # array of cells in down supercell
    ups: NDArray[np.uint8]      # array of cells in up supercell
    child: NDArray[np.uint8]    # (12,3)  uint8  (child transitions)
    mcc2: NDArray[np.uint8]     # [super_cell_mode,cell]->c2
    cmc2n: NDArray[np.uint8]    # [cell,super_cell_mode,c2]->neighbour_cell
    loc_offs: NDArray[np.uint8]


@runtime_checkable
class RegionAddressLike(Protocol):
    """Convert region-cells to region-ids"""
    rid2cell: NDArray[np.uint8]    # (R,)
    cell2rid: NDArray[np.int16]    # (<=256,), -1 means unmapped
    modes: NDArray[np.uint8]
    proto: NDArray[np.uint8]
    r_size: int                     # number of region ids


@runtime_checkable
class AddressPackerLike(Protocol):
    """Convert region lists to packed addresses"""
    def encode(self, reg_ids: NDArray[np.uint8], **kwargs) -> NDArray[np.uint64]: ...
    def decode(self, words: NDArray[np.uint64], **kwargs) -> NDArray[np.uint8]: ...


@runtime_checkable
class HexLUTLike(Protocol):
    """hex/region lookup tables"""
    hex_oob: int                  # 0x0F (hex out-of-bounds)
    hex_reg: NDArray[np.uint8]    # (R, R); 0..8 or 0xF for invalid
    reg_hex: NDArray[np.uint8]    # (R, R); 0..8 or 0xF for invalid


@runtime_checkable
class H9PolygonLike(Protocol):
    """Polygons under Lattice"""
    hh: NDArray[np.float64]  # half-hex (mode, c2) 4 pts (x,y)
    hx: NDArray[np.float64]  # hex (mode, c2) 6 pts (x,y)
    tx: NDArray[np.float64]  # cell (mode, c2, ord) 3 pts (x,y)
    se: NDArray[np.float64]  # supercell edges (mode) 9 pts (x,y)
    sv: NDArray[np.float64]  # supercell vertices (mode) 3 pts (x,y)
    gd: NDArray[np.float64]  # unshared points of a cell excluding (0,0).
