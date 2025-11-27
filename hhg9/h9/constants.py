"""
Part of the H9 project
Author: Ben
base/h9/geometry.py
H9 constants (single source of truth)
This module declares H9 constants **once** in float64 and freezes them.
This is the single source of truth for, eg, √3, √2 and so on.
The reason to do this is that we want to avoid small variations in the float64 ULP
(Unit at Last Place) which often happens with composite calculations.
This helps to stabilise many downstream edge cases.
Λ/V denote supercell mode (1/0).
CF denote Ceiling/Floor in barycentric orientation, not Apex/Base!
The supercell barycentre is fixed at 0 so each supercell must span zero.
See constants.md for further documentation.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, slots=True)
class Fundamentals:
    """
    Because of variations found in float64 ULP (units of least precision),
    we calculate the constants just once, based upon two base radicals: √3, and √2
    """
    R3: float  # √3
    W: float  # √2 (edge length of the supercell triangle)


@dataclass(frozen=True, slots=True)
class Derived:
    """Values derived once from Fundamentals; used across inequalities and transforms."""
    W: float  # Edge length of the *full* barycentric triangle: √2 (mirrors Fundamentals.W for convenience)
    H: float  # Height of the supercell triangle: H = W * √3 / 2
    Ḣ: float  # 1 * Height of child cells: Ḣ = H / 3; used in inequality tests
    Ẇ: float  # 2 * Height of child cells: Ẇ = 2 * Ḣ; used in inequality tests; alias of ΛC
    RH: float  # Convenience: √3 / 2 (useful for symmetric transforms)


@dataclass(frozen=True, slots=True)
class Limits:
    """Horizontal (TR, TL) and Vertical bounds for Λ and V triangles """
    TR: float  # +ve right-hand x-limit in classifier plane:  TR = H / √3 = √2/2 = W/2
    TL: float  # -ve left-hand  x-limit in classifier plane:  TL = -TR = -√2/2  = -W/2
    # the following are in barycentric-y order
    ΛC: float  # mode 1 supercell barycentric ceiling (not apex)! y:  +2·Ḣ
    VC: float  # mode 0 supercell barycentric ceiling (not apex)! y:  +1·Ḣ
    # The origin, y=0, is here.                                   y:   0·Ḣ
    ΛF: float  # mode 1 supercell barycentric floor (not base)!   y:  -1·Ḣ
    VF: float  # mode 0 supercell barycentric floor (not base)    y:  -2·Ḣ


@dataclass(frozen=True, slots=True)
class Lattice:
    """
    Rectilinear lattice steps, use multiples of (U,V)
    as unit multipliers to calculate cell offsets.
    The barycentre of every equilateral triangle (cell) tiling
    the plane may then be identified using integer co-ordinates(x,y)
    such that it's cartesian position is at (x*U,y*V).
    Not every u,v will be a barycentre, but every barycentre is integer.
    """
    U: float  # horizontal unit: W / 6
    Ü: float  # √3*U
    V: float  # vertical   unit: H / 9


@dataclass(frozen=True, slots=True)
class H9Const:
    """H9 Constants"""
    radical: Fundamentals
    derived: Derived
    limits: Limits
    lattice: Lattice

    @property
    def H(self): return self.derived.H

    @property
    def R3(self): return self.radical.R3

    @property
    def ΛF(self): return self.limits.ΛF

    @property
    def ΛC(self): return self.limits.ΛC

    @property
    def VF(self): return self.limits.VF

    @property
    def VC(self): return self.limits.VC

    @property
    def U(self): return self.lattice.U

    @property
    def V(self): return self.lattice.V

    @property
    # There is no capital U with dot above in unicode!
    def Ü(self): return self.lattice.Ü

    @property
    def Ẇ(self): return self.derived.Ẇ

    @property
    def Ḣ(self): return self.derived.Ḣ


def h9_constants() -> H9Const:
    """
    H9Const factory method.
    :return: H9Const
    """

    # Define the Fundamental constants
    # In this grid, the fundamental edge-length of the supercell triangle is √2
    # due to its relation with the unit-octahedral shape of the underlying system.
    # We choose W (√2) as a fundamental alongside R3 (√3);
    r3 = np.sqrt(3.0, dtype=np.float64)  # √3 because equilateral triangles.
    w = np.sqrt(2.0, dtype=np.float64)   # √2 from unit-octahedron edge length
    k_fundamentals = Fundamentals(r3, w)

    # Define the Derived constants
    # All constants are now derived from r3, w as above.
    # This helps to eliminate floating point variation.
    h = w * r3 / 2.0  # supercell triangle height via Pythagoras
    ḣ = h / 3.  # Vertical spacing between stacked child cells: Ḣ = H / 3
    ẇ = 2 * ḣ  # slope-window used in y±ẋ tests; alias of ΛC,  ẇ 2/3 of h
    rh = r3 / 2.  # ratio of height to width. 0.8660
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

