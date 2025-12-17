# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
H9 Protocol Definitions.

This module defines the structural contracts (Interfaces) for core objects in the H9 system using Python's `Protocol` mechanism.
These protocols allow for static type checking and runtime verification via `isinstance`, ensuring that
different parts of the system (constants, classifiers, addressers) adhere to the expected structure
without requiring strict inheritance.

**Key Protocols:**
    * **FundamentalsLike:** Basic math constants.
    * **H9ConstLike:** The aggregate constants container.
    * **H9ClassifierLike:** The look-up tables for cell classification.
    * **H9CellLike & H9RegionLike:** Contracts for grid cell and region properties.
"""

from __future__ import annotations  # PEP 563 (here to support python 3.8–3.10)

from enum import unique, IntEnum
from typing import Protocol, runtime_checkable
import numpy as np
from typing import Tuple, Any
from numpy.typing import NDArray


@unique
class BaryLoc(IntEnum):
    """
    Enumeration for locations of barycentric coordinates relative to a supercell.
    """
    UDF = 0  #: Undefined location.
    EXT = 1  #: External (outside the supercell).
    INT = 2  #: Internal (strictly inside the supercell).
    EDG = 3  #: Edge (on the boundary, but not a vertex).
    VTX = 4  #: Vertex (at a corner of the supercell).


@runtime_checkable
class FundamentalsLike(Protocol):
    """
    Protocol for fundamental radical constants.

    Canonical implementation found in ``constants.py``.
    """
    R3: float  #: The square root of 3 (:math:`\sqrt{3}`).
    W: float  #: The square root of 2 (:math:`\sqrt{2}`), width of a unit octahedron face.


@runtime_checkable
class DerivedLike(Protocol):
    """
    Protocol for derived geometric constants.

    Canonical implementation found in ``constants.py``.
    """
    W: float  #: Edge length of the full barycentric triangle (:math:`\sqrt{2}`).
    H: float  #: Height of the supercell triangle (:math:`H = W \frac{\sqrt{3}}{2}`).
    Ḣ: float  #: Height of a single child cell (:math:`\dot{H} = H / 3`).
    Ẇ: float  #: Width parameter for child cells (:math:`\dot{W} = 2\dot{H}`).
    RH: float  #: Ratio of height to width (:math:`\frac{\sqrt{3}}{2}`).


@runtime_checkable
class LimitsLike(Protocol):
    """
    Protocol for supercell boundary limits.

    Canonical implementation found in ``constants.py``.
    """
    TR: float  #: Positive right-hand x-limit (:math:`+W/2`).
    TL: float  #: Negative left-hand x-limit (:math:`-W/2`).
    ΛC: float  #: Mode 1 (Up) ceiling (:math:`y = +2\dot{H}`).
    VC: float  #: Mode 0 (Down) ceiling (:math:`y = +1\dot{H}`).
    ΛF: float  #: Mode 1 (Up) floor (:math:`y = -1\dot{H}`).
    VF: float  #: Mode 0 (Down) floor (:math:`y = -2\dot{H}`).


@runtime_checkable
class LatticeLike(Protocol):
    """
    Protocol for rectilinear lattice steps.

    Canonical implementation found in ``constants.py``.
    """
    U: float  #: Horizontal unit step (:math:`W / 6`).
    V: float  #: Vertical unit step (:math:`H / 9`).


@runtime_checkable
class H9ConstLike(Protocol):
    """
    Aggregate protocol for all H9 constants.

    Canonical implementation found in ``constants.py``.
    """
    radical: FundamentalsLike
    derived: DerivedLike
    limits: LimitsLike
    lattice: LatticeLike


@runtime_checkable
class H9ClassifierLike(Protocol):
    """
    Protocol for the Barycentric Classifier LUTs.

    Canonical implementation found in ``classifier.py``.
    """
    h_levels: NDArray[np.float64]  #: Horizontal tiers [ΛC, VC, 0, ΛF, VF].
    p_levels: NDArray[np.float64]  #: Positive slope tiers [+Ẇ, 0, -Ẇ].
    n_levels: NDArray[np.float64]  #: Negative slope tiers [-Ẇ, 0, +Ẇ].
    mode_0_lim: Tuple[float, float]  #: (Floor, Ceiling) for Mode 0.
    mode_1_lim: Tuple[float, float]  #: (Floor, Ceiling) for Mode 1.
    encode: NDArray[np.uint8]  #: Encoding LUT shape (6, 4, 4).
    decode: NDArray[np.uint8]  #: Decoding LUT shape (96, 3).
    eps: np.floating  #: Machine epsilon for float64.


@runtime_checkable
class H9CellLike(Protocol):
    """
    Protocol for Cell Properties within the Lattice.

    Canonical implementation found in ``lattice.py``.
    """
    off_ẋy: NDArray[np.float64]  #: Metric scaled coordinates (:math:`\dot{x}, y`).
    count: int  #: Count of cells in classifier.
    mode: NDArray[np.uint8]  #: Cell mode (0=down, 1=up).
    off_uv: NDArray[np.int8]  #: Lattice (u, v) coordinates.
    off_xy: NDArray[np.float64]  #: Metric barycentric coordinates (x, y).
    in_scope: NDArray[np.uint8]  #: Array of in-scope cell IDs.
    in_mode: NDArray[bool]  #: Boolean mask: is cell in the specified mode?
    in_dn: NDArray[bool]  #: Boolean mask: is cell in the Down supercell?
    in_up: NDArray[bool]  #: Boolean mask: is cell in the Up supercell?
    downs: NDArray[np.uint8]  #: Array of cells belonging to Down supercell.
    ups: NDArray[np.uint8]  #: Array of cells belonging to Up supercell.
    c2: NDArray[np.uint8]  #: C2 cluster ID [2, 3, 3] for each mode.


@runtime_checkable
class H9RegionLike(Protocol):
    """
    Protocol for Region Constants and transitions.

    Canonical implementation found in ``region.py``.
    """
    invalid_region: int  #: Marker for invalid regions (e.g., 0x5F).
    proto: NDArray[np.uint8]  #: Up/Down as a mode-ordered array (0/1).
    proto_up: int  #: Virtual Up protocol ID.
    proto_dn: int  #: Virtual Down protocol ID.
    ids: NDArray[np.uint8]  #: Array of region IDs.
    is_in: NDArray[bool]  #: Boolean array indicating in-scope status.
    downs: NDArray[np.uint8]  #: Array of cells in Down supercell.
    ups: NDArray[np.uint8]  #: Array of cells in Up supercell.
    child: NDArray[np.uint8]  #: Child transitions (12, 3).
    mcc2: NDArray[np.uint8]  #: [super_cell_mode, cell] -> c2 mapping.
    cmc2n: NDArray[np.uint8]  #: [cell, super_cell_mode, c2] -> neighbour_cell mapping.
    loc_offs: NDArray[np.uint8]  #: Location offsets.


@runtime_checkable
class RegionAddressLike(Protocol):
    """
    Protocol for converting between region-cells and region-IDs.
    """
    rid2cell: NDArray[np.uint8]  #: Map Region ID -> Cell ID.
    cell2rid: NDArray[np.uint8]  #: Map Cell ID -> Region ID (-1 if unmapped).
    modes: NDArray[np.uint8]  #: Mode of the regions.
    props: NDArray[np.uint8]
    proto: NDArray[np.uint8]  #: Protocol array.
    r_size: int  #: Total number of region IDs.


@runtime_checkable
class AddressPackerLike(Protocol):
    """
    Protocol for packing/unpacking addresses.
    """

    def encode(self, reg_ids: NDArray[np.uint8], **kwargs: Any) -> NDArray[np.uint64]:
        """Encode region IDs into packed 64-bit integers."""
        ...

    def decode(self, words: NDArray[np.uint64], **kwargs: Any) -> NDArray[np.uint8]:
        """Decode packed 64-bit integers back into region IDs."""
        ...


@runtime_checkable
class HexLUTLike(Protocol):
    """
    Protocol for Hex/Region lookup tables.
    """
    hex_oob: int  #: Out-of-bounds marker (e.g., 0x0F).
    hex_reg: NDArray[np.uint8]  #: Hex -> Region LUT (size R, R).
    reg_hex: NDArray[np.uint8]  #: Region -> Hex LUT (size R, R).


@runtime_checkable
class H9PolygonLike(Protocol):
    """
    Protocol for Polygon shapes defined under the Lattice.
    """
    hh: NDArray[np.float64]  #: Half-hex (mode, c2) 4 pts (x, y).
    hx: NDArray[np.float64]  #: Hexagon (mode, c2) 6 pts (x, y).
    tx: NDArray[np.float64]  #: Cell triangle (mode, c2, ord) 3 pts (x, y).
    se: NDArray[np.float64]  #: Supercell edges (mode) 9 pts (x, y).
    sv: NDArray[np.float64]  #: Supercell vertices (mode) 3 pts (x, y).
    gd: NDArray[np.float64]  #: Unshared points of a cell excluding (0, 0).