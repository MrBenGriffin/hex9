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
