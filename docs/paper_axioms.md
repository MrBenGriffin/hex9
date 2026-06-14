# Hex9 Canonical Axiom Set

---

## Axiom 1 — Domain (Ellipsoidal Manifold)

The Earth is modelled as a smooth reference ellipsoid (e.g. WGS84), supporting a
well-defined normal field and latitude/longitude parameterisation.

- This is the continuous substrate being discretised.
- No projection is privileged at this level.

---

## Axiom 2 — Discrete Carrier (Simplicial Primacy)

The ellipsoid is discretised by a finite, recursively refinable 2D simplicial complex.

- The primitive cell is a triangle (2-simplex).
- All higher structure is derived from simplicial adjacency.
- No non-simplicial base cells exist in the canonical representation.

*Consequence: hex-like structures are derived, not primitive.*

---

## Axiom 3 — Global Topological Regularity

The simplicial complex forms a valid triangulation of S² with:

- no boundary,
- no exceptional cell types,
- curvature expressed metrically, not topologically.

This enforces uniform representation class across the entire domain and excludes
complexes with topological defects (such as forced pentagonal faces).

*Consequence: combined with Axiom 2 and Axiom 4 — simplicial carrier, even valence
from vertex closure, and this axiom's uniformity requirement — the admissible complex
is the octahedral triangulation.*

---

## Axiom 4 — Mode Transport

Mode transport is consistent when traversing any closed loop through the triangulation
returns the mode to its starting value (trivial holonomy, in the language of
differential geometry; equivalently, a flat ℤ₂ connection, in gauge-theoretic terms).
This is equivalent to the existence of a globally coherent mode field: fix the mode
of any one face and every other face's mode follows directly.

Any refinement operator must preserve this consistency. Failure to do so introduces
a transport defect.

*Consequence (proved in §6): refinements satisfying this condition have odd linear
subdivision factor k; the admissible aperture class is {k² : k odd, k > 1}, and
aperture 9 is minimal.*

---

## Axiom 5 — Refinement Invariance

Any refinement operator:

- preserves simplicial type,
- preserves adjacency relations,
- commutes with global indexing,
- preserves parity transport (Axiom 4).

Refinement never introduces new structural categories.

---

## Axiom 6 — Canonical Orientation (Chirality Fixing)

The global orientation of simplices is fixed by a single consistent chirality choice
induced by embedding into the reference ellipsoid frame.

- This resolves the residual Z₂ symmetry left after Axioms 2–5.
- The choice is globally coherent, not locally independent.

*Consequence: this is where the half-hexagon orientation freedom collapses to
one surviving solution per chirality.*

---

## Axiom 7 — Geodetic Anchoring (Minimal Frame Fixing)

A minimal geodetic frame is fixed:

- poles define the primary axis,
- a single meridian defines longitudinal zero,
- together inducing a canonical partition of the domain.

This is not arbitrary once fixed — it is the reference gauge of the system.

---

## Axiom 8 — Unique Cell Addressing

Every cell in the hierarchy has a unique address, and every valid address identifies
exactly one cell. This bijection is compatible with hierarchical structure: a cell's
address encodes its position in the refinement hierarchy, and the parent–child
relationship is recoverable directly from address structure.

Geographic identity is read directly from combinatorial position. No coordinate
transformation is required to infer location from address or address from location.

---

## Axiom 9 — Dual Consistency (Hex Emergence)

The dual graph of the simplicial complex induces a structured hexagonal lattice:

- hexagons are derived dual cells,
- the total topological defect of 12 required by the Euler characteristic of S²
  (Σ over vertices of (6 − valence) = 12) is concentrated at the 6 seed vertices
  — a deficit of 2 at each — rather than appearing as face anomalies,
- no independent hex tiling is assumed at the primitive level.

*Hex9 structure is emergent from simplices, not fundamental.*

