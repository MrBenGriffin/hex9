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

*Consequence: combined with Axiom 2 and Axiom 5, the minimal admissible complex
is the octahedral triangulation.*

---

## Axiom 4 — Mode Transport

Mode transport defines a ℤ₂-valued 1-cocycle on the dual graph of the triangulation
(established by Axioms 2–3). Consistency of this transport — the existence of a
globally coherent mode field — requires the cocycle to be flat: trivial holonomy
over all closed loops.

Any refinement operator must preserve the flatness of this ℤ₂ connection. Failure
to do so introduces a transport defect.

- Preservation under refinement requires the linear subdivision factor k to be odd.
- The minimal non-trivial odd k is 3, giving aperture k² = 9.

*Consequence: the admissible aperture class is {k² : k odd, k > 1}; aperture 9 is minimal.*

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

*Consequence: this is where the half-hexagon orientation freedom collapses to a
unique solution per handedness.*

---

## Axiom 7 — Geodetic Anchoring (Minimal Frame Fixing)

A minimal geodetic frame is fixed:

- poles define the primary axis,
- a single meridian defines longitudinal zero,
- together inducing a canonical partition of the domain.

This is not arbitrary once fixed — it is the reference gauge of the system.

---

## Axiom 8 — Structure–Label Isomorphism

There exists a structure-preserving bijection between the combinatorial structure of
the simplicial complex (simplices, adjacency, duals) and the system of geographic
identifiers (octants, faces, cell addresses), such that geographic identity is read
directly from combinatorial position.

No coordinate transformation is required to infer location from label or label from
location.

---

## Axiom 9 — Dual Consistency (Hex Emergence)

The dual graph of the simplicial complex induces a structured hexagonal lattice:

- hexagons are derived dual cells,
- 12 global topological defects (required by the Euler characteristic of S²) are
  absorbed into the simplicial structure as vertices rather than face anomalies,
- no independent hex tiling is assumed at the primitive level.

*Hex9 structure is emergent from simplices, not fundamental.*

