# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
H9 Constants (Single Source of Truth).

This module declares H9 constants **once** in float64 and freezes them.
This acts as the single source of truth for radicals like :math:`\\sqrt{3}`, :math:`\\sqrt{2}`,
and derived geometric bounds.

**Why this exists:**
    Calculated values (like :math:`\\sqrt{3} / 2`) can vary slightly depending on the order of
    operations due to float64 ULP (Unit in Last Place) variations. By defining them strictly
    here, we ensure stability across downstream edge cases and inequality tests.

**Notation:**
    * **Λ (Lambda):** Denotes Supercell Mode 1 ("Up" or "Peak").
    * **V:** Denotes Supercell Mode 0 ("Down" or "Valley").
    * **C/F:** Suffixes for Ceiling/Floor in barycentric orientation (not Apex/Base!).
    * **Supercell Barycentre:** Fixed at 0; each supercell spans zero.

See ``constants.md`` for further documentation.
"""

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Fundamentals:
    """
    Base radical constants.

    Because of variations found in float64 ULP (units of least precision),
    we calculate the constants just once.

    Attributes:
        R3 (float): The square root of 3 (:math:`\\sqrt{3}`).
        W (float): The square root of 2 (:math:`\\sqrt{2}`), representing the
            edge length of the supercell triangle.
    """
    R3: float
    W: float


@dataclass(frozen=True, slots=True)
class Derived:
    """
    Values derived once from Fundamentals.

    These are used across inequalities and coordinate transforms.

    Attributes:
        W (float): Edge length of the *full* barycentric triangle (:math:`\\sqrt{2}`).
            Mirrors ``Fundamentals.W`` for convenience.
        H (float): Height of the supercell triangle. Calculated as :math:`H = W \\frac{\\sqrt{3}}{2}`.
        Ḣ (float): Height of a single child cell (1/3 of total Height). :math:`\\dot{H} = H / 3`.
        Ẇ (float): Width parameter for child cells, used in inequality tests. :math:`\\dot{W} = 2 \\dot{H}`.
            Also acts as an alias for ``ΛC``.
        RH (float): Ratio of Height to Width (:math:`\\frac{\\sqrt{3}}{2}`). Useful for symmetric transforms.
    """
    W: float
    H: float
    Ḣ: float
    Ẇ: float
    RH: float


@dataclass(frozen=True, slots=True)
class Limits:
    """
    Horizontal (TR, TL) and Vertical bounds for Λ (Mode 1) and V (Mode 0) triangles.

    These values define the bounding box of the triangles in the classifier plane.
    Vertical values are defined relative to the barycenter (y=0).

    Attributes:
        TR (float): Positive right-hand x-limit in classifier plane (:math:`+W/2`).
        TL (float): Negative left-hand x-limit in classifier plane (:math:`-W/2`).
        ΛC (float): Mode 1 (Up) supercell barycentric ceiling. :math:`y = +2\\dot{H}`.
        VC (float): Mode 0 (Down) supercell barycentric ceiling. :math:`y = +1\\dot{H}`.
        ΛF (float): Mode 1 (Up) supercell barycentric floor. :math:`y = -1\\dot{H}`.
        VF (float): Mode 0 (Down) supercell barycentric floor. :math:`y = -2\\dot{H}`.
    """
    TR: float
    TL: float
    ΛC: float
    VC: float
    ΛF: float
    VF: float


@dataclass(frozen=True, slots=True)
class Lattice:
    """
    Rectilinear lattice steps.

    Uses multiples of (U, V) as unit multipliers to calculate cell offsets.
    The barycentre of every equilateral triangle (cell) tiling the plane may be
    identified using integer co-ordinates (x, y) such that its Cartesian position
    is at :math:`(xU, yV)`.

    Note:
        Not every (u, v) combination will be a valid barycentre, but every
        valid barycentre is an integer coordinate.

    Attributes:
        U (float): Horizontal unit step (:math:`W / 6`).
        Ü (float): Scaled horizontal unit (:math:`\\sqrt{3} U`). Note: Effectively equal to ``Ḣ``.
        V (float): Vertical unit step (:math:`H / 9`).
    """
    U: float
    Ü: float
    V: float


@dataclass(frozen=True, slots=True)
class H9Const:
    """
    The master container for H9 Grid Constants.

    This class aggregates Fundamentals, Derived values, Limits, and Lattice steps.
    It provides property accessors for the most commonly used values to simplify
    import statements in other modules.
    """
    radical: Fundamentals
    derived: Derived
    limits: Limits
    lattice: Lattice

    @property
    def H(self) -> float:
        """Height of the supercell triangle."""
        return self.derived.H

    @property
    def R3(self) -> float:
        """Square root of 3."""
        return self.radical.R3

    @property
    def ΛF(self) -> float:
        """Mode 1 (Up) floor."""
        return self.limits.ΛF

    @property
    def ΛC(self) -> float:
        """Mode 1 (Up) ceiling."""
        return self.limits.ΛC

    @property
    def VF(self) -> float:
        """Mode 0 (Down) floor."""
        return self.limits.VF

    @property
    def VC(self) -> float:
        """Mode 0 (Down) ceiling."""
        return self.limits.VC

    @property
    def U(self) -> float:
        """Lattice horizontal unit."""
        return self.lattice.U

    @property
    def V(self) -> float:
        """Lattice vertical unit."""
        return self.lattice.V

    @property
    def Ü(self) -> float:
        """Lattice scaled horizontal unit (Unicode U with diaeresis)."""
        return self.lattice.Ü

    @property
    def Ẇ(self) -> float:
        """Child cell width parameter (Unicode W with dot above)."""
        return self.derived.Ẇ

    @property
    def Ḣ(self) -> float:
        """Child cell height parameter (Unicode H with dot above)."""
        return self.derived.Ḣ


def h9_constants() -> H9Const:
    """
    Factory method to initialize the H9Const singleton.

    Calculates all derived values from the two fundamental roots to ensure
    floating-point consistency.

    Returns:
        H9Const: The frozen, initialized constants object.
    """
    # Define the Fundamental constants
    # In this grid, the fundamental edge-length of the supercell triangle is √2
    # due to its relation with the unit-octahedral shape of the underlying system.
    # We choose W (√2) as a fundamental alongside R3 (√3).
    r3 = np.sqrt(3.0, dtype=np.float64)  # √3 because equilateral triangles.
    w = np.sqrt(2.0, dtype=np.float64)   # √2 from unit-octahedron edge length
    k_fundamentals = Fundamentals(r3, w)

    # Define the Derived constants
    h = w * r3 / 2.0  # supercell triangle height via Pythagoras
    ḣ = h / 3.        # Vertical spacing between stacked child cells: Ḣ = H / 3
    ẇ = 2 * ḣ         # slope-window used in y±ẋ tests; alias of ΛC, ẇ is 2/3 of h
    rh = r3 / 2.      # ratio of height to width. 0.8660
    k_derived = Derived(w, h, ḣ, ẇ, rh)

    # Define the Limits constants
    tr = h / r3
    tl = -tr
    # Limits declares ceilings / floors in barycentric order descending.
    λc = +ẇ   # mode 1 (up) triangle ceiling   y: +2·Ḣ
    vc = +ḣ   # mode 0 (down) triangle ceiling y: +1·Ḣ
    λf = -ḣ   # mode 1 (up) triangle floor     y: -1·Ḣ
    vf = -ẇ   # mode 0 (down) triangle floor   y: -2·Ḣ
    k_limits = Limits(tr, tl, λc, vc, λf, vf)

    # Define the Lattice constants
    # Lattice (small grid steps; parent step = (3U, 3V))
    u, v = w / 6., h / 9.  # horizontal/vertical unit multipliers.
    # Note: The second argument passed to Lattice here is ḣ (which equals √3 * u)
    k_lattice = Lattice(u, ḣ, v)

    return H9Const(
        radical=k_fundamentals,
        derived=k_derived,
        limits=k_limits,
        lattice=k_lattice,
    )


H9K: H9Const = h9_constants()  # singleton for typical use

# Tolerant checks (float64-safe)
assert np.isclose(H9K.ΛF, -H9K.VC, rtol=1e-12, atol=1e-15)
assert np.isclose(H9K.ΛC, -H9K.VF, rtol=1e-12, atol=1e-15)

# Ordering sanity
assert H9K.ΛC > 0.0 > H9K.ΛF
assert H9K.VC > 0.0 > H9K.VF

# Span consistency (optional)
# total span in Λ equals total span in V by construction:
assert np.isclose((H9K.ΛC - H9K.ΛF), (H9K.VC - H9K.VF), rtol=1e-12, atol=1e-15)

# Apex farther than base; combined band ordering
assert H9K.ΛC > H9K.VC > 0.0 > H9K.ΛF > H9K.VF


@dataclass(frozen=True, slots=True)
class OctConst:
    """These are octahedral map properties."""
    oct_vrt: NDArray[np.float64]
    oct_str: NDArray
    oid_tht: NDArray[np.uint8]
    oid_cmp: NDArray[np.uint8]
    cmp_oid: NDArray[np.uint8]
    oid_str: NDArray[np.uint8]
    oid_mo: NDArray[np.uint8]
    oid_nb: NDArray[np.uint8]
    nb_c2p: NDArray[np.bool]
    nb_c2map: NDArray[np.uint8]
    edges_by_id: NDArray
    l0hex_by_id: NDArray[np.uint8]
    l0hex_back: NDArray[np.uint8]


def oct_constants() -> OctConst:
    """
    Factory method to initialize the OctConst singleton.
    """
    # Define axes of symmetry for octahedron
    # These are the underlying processes
    axes = np.array([['A', 'P'], ['E', 'W'], ['N', 'S']])
    v_pos = np.zeros((6, 3), dtype=np.float64)
    v_str = np.zeros((6,), dtype='<U1')
    vertices = {}
    for i, axis in enumerate(axes):
        for j, v in enumerate(axis):
            k = j * 2 - 1
            vertex = [0, 0, 0]
            vertex[i] = k
            vertices[v] = tuple(vertex)
            v_str[i*2+j] = v
            v_pos[i*2+j] = vertex

    # # Generate all face permutations using np.meshgrid
    # face_permutations = np.stack(np.meshgrid(*axes), axis=-1).reshape((-1, 3))
    # # Generate a canonical list of edges also. Why?
    # # It could generate the ID and name of the 12 root0 hexagons.
    # edges = np.unique(face_permutations[:, [[0, 1], [1, 2], [0, 2]]].reshape((-1, 2)), axis=0)[:, ::-1]
    # for face_arr in face_permutations:
    #     face = ''.join(face_arr[::-1])  # Reverse order to match face naming
    #     triple = np.asarray([vertices[s] for s in face_arr])
    #     sign = tuple(np.sum(triple, axis=1).tolist())

    props = {
        # Octahedral Triangle Identities differ. 0x16, 0x49
        # Note that OID do not share the id % 2 == mode logic(!)
        # Also, 'c2' is always equator=0; with nb.c2=1 == self.c2 = 2
        # AP EW  NS    θ  c2 hexagon region  mo o_id, c2* ngh  hex_id
        (+1, +1, +1): (2, ('EA', 'NA', 'NE'), 0, 0, (4, 2, 1), (0, 4, 5),   'NEA'),  # 0 'NEA' N:5, E:8, A: 0
        (-1, +1, +1): (5, ('EP', 'NE', 'NP'), 1, 1, (5, 0, 3), (1, 5, 7),   'NEP'),  # 1 'NEP' N:4, E:7, P: 0
        (+1, -1, +1): (5, ('WA', 'NW', 'NA'), 1, 2, (6, 3, 0), (2, 6, 4),   'NWA'),  # 2 'NWA'
        (-1, -1, +1): (2, ('WP', 'NP', 'NW'), 0, 3, (7, 1, 2), (3, 7, 6),   'NWP'),  # 3 'NWP' N:5, W:8, P: 0
        (+1, +1, -1): (5, ('EA', 'SE', 'SA'), 1, 4, (0, 5, 6), (0, 8, 10),  'SEA'),  # 4 'SEA' S:4, E:7, A: 0
        (-1, +1, -1): (2, ('EP', 'SP', 'SE'), 0, 5, (1, 7, 4), (1, 9, 8),   'SEP'),  # 5 'SEP' S:5, E:8, P: 0
        (+1, -1, -1): (2, ('WA', 'SA', 'SW'), 0, 6, (2, 4, 7), (2, 10, 11), 'SWA'),  # 6 'SWA' S:5, W:8, A: 0
        (-1, -1, -1): (5, ('WP', 'SW', 'SP'), 1, 7, (3, 6, 5), (3, 11, 9),  'SWP')   # 7 'SWP' S:4, W:7, P: 0
    }
    # 0   1        2      3       4      5,  6
    i_th, i_edges, i_mo, i_oid, i_c2n, i_hx, i_str = 0, 1, 2, 3, 4, 5, 6
    oid_th = np.zeros((8,), dtype=np.int8)
    oid_cmp = np.zeros((8, 3), dtype=np.int8)
    cmp_oid = np.empty((3, 3, 3), dtype=np.int8)
    oid_str = np.empty((8, ), dtype='<U3')
    oid_mo = np.zeros((8,), dtype=np.uint8)
    oid_nb = np.zeros((8, 3), dtype=np.int8)
    nb_c2p = np.zeros((8, 3), dtype=bool)
    nb_c2map = np.zeros((8, 3), dtype=np.uint8)
    edges_by_id = np.empty((8, 3), dtype='<U2')
    l0hex_by_id = np.empty((8, 3), dtype=np.uint8)
    l0hex_back = np.full((12, 2, 2), 0x0F, dtype=np.uint8)
    for sign, row in props.items():
        o_id = row[i_oid]
        o_mode = row[i_mo]
        oid_mo[o_id] = o_mode
        oid_th[o_id] = row[i_th]
        edges_by_id[o_id] = np.array(row[i_edges], dtype='<U2')
        cmp_oid[sign] = o_id
        oid_cmp[o_id] = list(sign)
        oid_str[o_id] = row[i_str]
        l0_hex = np.array(row[i_hx], dtype=np.uint8)
        l0hex_by_id[o_id] = l0_hex
        for c2, hx in enumerate(l0_hex):
            l0hex_back[hx, o_mode] = [o_id, c2]
        l0_hex = np.array(row[i_hx], dtype=np.uint8)
        for c2, hx in enumerate(l0_hex):
            l0hex_back[hx, o_mode] = [o_id, c2]
        oid_nb[o_id] = np.array(list(row[i_c2n]))

    preserve_edges = {'EA', 'EP', 'NA', 'WA', 'WP', 'SA'}  # axial edges preserve; diagonals swap
    swap_code = {
        'NP': 2, 'SE': 2, 'SP': 2,  # reflect across C2=1 axis (y = ẋ)
        'NE': 3, 'NW': 3, 'SW': 3,  # reflect across C2=2 axis (y = −ẋ)
    }
    for f in range(8): # Boolean preserve matrix: True where the edge label is in preserve_edges
        for i, e in enumerate(edges_by_id[f]):
            nb_c2p[f, i] = (e in preserve_edges)
        for ci, e in enumerate(edges_by_id[f]):
            if nb_c2p[f, ci]:
                nb_c2map[f, ci] = 0
            else:
                try:
                    nb_c2map[f, ci] = swap_code[e]
                except KeyError as ke:
                    raise KeyError(
                        f"Unknown swap edge label '{e}'. Ensure it is in swap_code or preserve_edges.") from ke

    return OctConst(
        v_pos,
        v_str,
        oid_th,
        oid_cmp,
        cmp_oid,
        oid_str,
        oid_mo,
        oid_nb,
        nb_c2p,
        nb_c2map,
        edges_by_id,
        l0hex_by_id,
        l0hex_back,
    )


H9O: OctConst = oct_constants()  # singleton for typical use

