---
title: "Hex9: A Quasi-Authalic, Quasi-Continuous Hexagonal DGGS on the Reference Ellipsoid"
author: Ben Griffin
date: June 2026
abstract: |
  Discrete global grid systems are conventionally designed over a prior
  coordinate reference system and inherit its compromises. Hex9 inverts the
  direction of design: we ask what requirements a hierarchical grid must satisfy
  to be geometrically coherent — intrinsic orientability, flat mode transport,
  vertex closure, refinement commutativity — and show that these requirements,
  taken together, essentially determine the grid. The admissible cell primitive
  is the triangle; the admissible seed is the octahedral triangulation of S²;
  admissible refinement has odd linear factor, minimally aperture 9; and the
  hexagonal dual lattice admits exactly one orientation per chirality — a result
  established by machine-verified exhaustive enumeration. The structure that
  survives is a shifted-aperture-9 hexagonal hierarchy in which every cell
  carries a unique address derived from the construction alone: truncated at
  level L, the address is a DGGS cell identifier in the sense of OGC Topic 21;
  carried to the limit, a function recovers from it a point on the reference
  ellipsoid to arbitrary precision. The addressing is quasi-continuous — position
  is recoverable everywhere except on a measure-zero set of seams — rather than
  continuous in the strict ISO 19111 sense; the same mathematical object serves
  as both cell identifier and position-recovery coordinate, with no prior
  coordinate reference system as input. A separable geometric realisation — an
  analytical octahedral base projection composed with an
  optimal-transport-derived area-correcting warp — places the grid on WGS84 with
  quasi-uniform cell areas: at level 5, 99% of the 708,588 cells lie within
  0.005% of ideal area, with residual deviation confined to the six octahedral
  vertices required by the topology. The combinatorial grid is projection- and
  ellipsoid-independent; only the warp is specific to the reference body, and it
  is recomputable for any ellipsoid, terrestrial or planetary.

  **Keywords:** discrete global grid system (DGGS) · coordinate reference system
  (CRS) · hexagonal grid · octahedron · aperture 9 · optimal transport ·
  equal-area projection · spatial indexing · WGS84
nocite: |
  @*
---

*Draft for review. This file is the canonical source (it supersedes the earlier
per-section `paper_arc_*.md` fragments). Code artefacts use the prefix `h9`
(e.g. `h9_encode`); prose uses Hex9 throughout.*

## Introduction

Discrete global grid systems face a set of familiar tradeoffs. Cell shape,
aperture, orientation, and global embedding each involve compromises between
equal area, hierarchical consistency, polar behaviour, and computational
tractability — compromises often governed by a prior choice of coordinate
reference system, which in turn constrains what tradeoffs remain.

Hex9 is an exploration of a different resolution strategy: rather than choosing
among tradeoffs, we ask what requirements a grid must satisfy to be
geometrically coherent — consistent at every edge, every vertex, and every
level of the hierarchy — and examine how far those requirements determine the
structure on their own.

These requirements significantly reduce the design freedom. A minimal set —
orientability, parity transport, vertex closure, and refinement commutativity —
strongly constrains admissible cell shape, embedding class, and aperture
structure. What appears to be a large design space resolves to a small family
of solutions, selecting a restricted class up to global symmetry.

This derivation has a direct consequence: because the structure is derived
rather than chosen, every cell carries a distinct identity tied to a specific
surface location. Cell identity serves as a self-contained locator — no prior
coordinate reference system is needed to establish where a cell is.

The derivation was discovered rather than designed. The project began as a
planar hexagonal tiling study and first targeted the icosahedron — the natural
choice, given its near-spherical geometry and the precedent of existing
icosahedral grids. The tiling could not be made globally consistent, and the
failure resisted diagnosis: the obstruction is not geometric but topological.
The odd valence of icosahedral vertices makes a consistent two-colouring of
faces impossible, and that two-colouring turns out to be load-bearing for
everything a coherent hexagonal hierarchy needs. Once the parity constraint was
understood, the octahedron emerged by elimination as the unique Platonic solid
with equilateral faces and even-valent vertices. The system presented here is
the consequence of following that constraint to its conclusions.

The paper develops this argument constructively. The first part traces the
coherence arc, showing step by step how each requirement eliminates
alternatives. The second part describes the AK+Warp projection — Anders
Kaseorg's octahedral base projection composed with an area-correcting warp —
the engineering path from the abstract octahedral structure to a quasi-authalic
realisation on the reference ellipsoid (typically WGS84).

---

## Axiom Set

### Axiom 1 — Domain (Ellipsoidal Manifold)

The Earth is modelled as a smooth reference ellipsoid (e.g. WGS84), supporting a
well-defined normal field and latitude/longitude parameterisation.

- This is the continuous substrate being discretised.
- No projection is privileged at this level.

### Axiom 2 — Discrete Carrier (Simplicial Primacy)

The ellipsoid is discretised by a finite, recursively refinable 2D simplicial
complex.

- The primitive cell is a triangle (2-simplex).
- All higher structure is derived from simplicial adjacency.
- No non-simplicial base cells exist in the canonical representation.

*Consequence: hex-like structures are derived, not primitive.*

### Axiom 3 — Global Topological Regularity

The simplicial complex forms a valid triangulation of S² with:

- no boundary,
- no exceptional cell types,
- curvature expressed metrically, not topologically.

This enforces uniform representation class across the entire domain and excludes
complexes with topological defects (such as forced pentagonal faces).

*Consequence: combined with Axiom 2 and Axiom 4 — simplicial carrier, even
valence from vertex closure, and this axiom's uniformity requirement — the
admissible complex is the octahedral triangulation.*

### Axiom 4 — Mode Transport

Mode transport is consistent when traversing any closed loop through the
triangulation returns the mode to its starting value (trivial holonomy, in the
language of differential geometry; equivalently, a flat ℤ₂ connection, in
gauge-theoretic terms). This is equivalent to the existence of a globally
coherent mode field: fix the mode of any one face and every other face's mode
follows directly.

Any refinement operator must preserve this consistency. Failure to do so
introduces a transport defect.

*Consequence (proved in §6): refinements satisfying this condition have odd
linear subdivision factor k; the admissible aperture class is {k² : k odd,
k > 1}, and aperture 9 is minimal.*

### Axiom 5 — Refinement Invariance

Any refinement operator:

- preserves simplicial type,
- preserves adjacency relations,
- commutes with global indexing,
- preserves parity transport (Axiom 4).

Refinement never introduces new structural categories.

### Axiom 6 — Canonical Orientation (Chirality Fixing)

The global orientation of simplices is fixed by a single consistent chirality
choice induced by embedding into the reference ellipsoid frame.

- This resolves the residual Z₂ symmetry left after Axioms 2–5.
- The choice is globally coherent, not locally independent.

*Consequence: this is where the half-hexagon orientation freedom collapses to
one surviving solution per chirality.*

### Axiom 7 — Geodetic Anchoring (Minimal Frame Fixing)

A minimal geodetic frame is fixed:

- poles define the primary axis,
- a single meridian defines longitudinal zero,
- together inducing a canonical partition of the domain.

This is not arbitrary once fixed — it is the reference gauge of the system.

### Axiom 8 — Unique Cell Addressing

Every cell in the hierarchy has a unique address, and every valid address
identifies exactly one cell. This bijection is compatible with hierarchical
structure: a cell's address encodes its position in the refinement hierarchy,
and the parent–child relationship is recoverable directly from address
structure.

Geographic identity is read directly from combinatorial position. No coordinate
transformation is required to infer location from address or address from
location.

### Axiom 9 — Dual Consistency (Hex Emergence)

The dual graph of the simplicial complex induces a structured hexagonal lattice:

- hexagons are derived dual cells,
- the total topological defect of 12 required by the Euler characteristic of S²
  (Σ over vertices of (6 − valence) = 12) is concentrated at the 6 seed vertices
  — a deficit of 2 at each — rather than appearing as face anomalies,
- no independent hex tiling is assumed at the primitive level.

*Hex9 structure is emergent from simplices, not fundamental.*

### Main Theorem

Axioms 1–9 admit a discrete global grid that is **unique up to two free
choices** — a global chirality (Axiom 6) and a geodetic gauge (Axiom 7). That
grid is **Hex9**: the shifted-aperture-9 hexagonal grid on the octahedral
triangulation of the reference ellipsoid, with the cell hierarchy and addressing
developed in §§6–10.

Concretely, the axioms force each link of the following chain, and the arc
sections that follow constitute its proof:

1. **Carrier.** The discrete carrier is a simplicial complex, and the only
   triangulation of S² with even, uniform valence and no exceptional cells is
   the octahedral triangulation (Axioms 2–4; §§1–5).
2. **Aperture.** Refinements preserving mode transport have odd linear factor
   *k*; the admissible aperture class is {*k*² : *k* odd, *k* > 1}, of which
   **9 is minimal** (Axioms 4–5; §6).
3. **Hex emergence.** The dual carries the 12 units of topological defect
   required by the Euler characteristic of S² as 4-valent cells at the 6
   octahedral vertices, not as face anomalies (Axiom 9; §7).
4. **Orientation.** The residual orientation freedom collapses to a single
   chiral pair (Axiom 6; §8).
5. **Identity.** Every cell carries a unique hierarchical address from which its
   geographic location is read directly, with no prior coordinate reference
   system (Axiom 8; §§9–10).

The geometric realisation on the ellipsoid (§11) is a separable engineering
step: it places this combinatorial object onto WGS84 but adds no degrees of
freedom to the structure the axioms determine.

---

## 1. The Simplicial Carrier

A discrete global grid partitions the globe into a finite collection of cells.
The choice of cell shape is the first apparent design decision: squares,
triangles, hexagons, and other polygons all tile the plane, and several tile the
globe with appropriate modifications.

The triangle stands apart from this class. It is the minimal polygon — three
edges, three vertices, no simpler closed shape exists — and the only one whose
orientation is intrinsic: given any triangle, a consistent notion of clockwise
and counterclockwise is determined by its vertex ordering alone, without
additional structure. Every other polygon either decomposes into triangles or
requires external reference to define orientation consistently across a tiling.

This intrinsic orientability makes the triangle the natural substrate for a
globally coherent grid. We take the triangular field — a tiling of the globe by
triangles — as the simplicial carrier on which the remaining coherence
requirements will act.

A coherent grid carrier must satisfy several requirements simultaneously:
operators such as interpolation and refinement must be definable without
auxiliary choices; each cell must support a unique, intrinsic coordinate system;
refinement must be closed and structure-preserving; and no cell-type exceptions
or singular vertices may appear. The triangle is the only 2-cell that satisfies
all of these at once. Its barycentric coordinates are intrinsic — defined by the
vertices alone, with no ambient metric required. It is closed under subdivision.
It forms a simplicial complex without diagonal ambiguity.

On curved surfaces this distinction becomes decisive. A triangulation absorbs
curvature at its vertices without changing cell type; tilings by higher polygons
must place their topological obligations somewhere visible — as exceptional
cells (the twelve pentagons of icosahedral hexagonal grids) or exceptional
corners (the eight 3-valent corners of cube-based quadrilateral grids). The
triangular field is therefore not one choice among many — every other option
either reduces to it or requires additional structure not established by the
coherence requirements alone.

---

## 2. Mode

In any consistent triangulation of an orientable surface, every edge is shared
by exactly two triangles. The intrinsic orientation established in §1 ensures
those two triangles carry opposite orientations: no two triangles of the same
orientation can share an edge.

The result is a consistent parity assignment over faces induced by orientation.
In the familiar planar picture these are the *up* and *down* triangles; on the
globe they generalise to two classes that partition every face in the
triangulation. We call this two-colouring the **mode** of the triangular field —
mode 0 for the down (negative) class, mode 1 for the up (positive) class. This
labelling is not imposed from outside — it is a structural property of the
triangulation itself. Fix the mode of any one face, and the mode of every other
face follows directly; the only freedom is which face to start from, which
amounts to a global reflection.

Every face carries a mode value. Every interior edge is incident to exactly two
faces of opposite mode. The field is bipartite on its faces.

As the ℤ₂ orientation cocycle of the simplicial surface, realised as a global
face bipartition, **mode** is the first emergent global invariant of simplicial
orientation — a constraint every grid built over the field must either respect
or violate. Everything downstream either respects that cocycle, preserving mode
consistently across refinement and adjacency, or breaks it, introducing
inconsistency or defects. Every edge crossing is a mode-flipping step; what
follows from global consistency of those steps is the subject of §3.

---

## 3. Mode Transport

Section 2 established that every edge crossing carries a mode flip. The transport
operator on each edge is that flip — the orientation difference between the two
faces sharing the edge, inherited directly from the intrinsic orientability of
§1. It is combinatorial in origin: no metric or embedding is required to define
it.

We now ask what it means for this transport to be globally consistent.

Define a **mode transport** as a rule that assigns, to any path through the
triangular field (a sequence of edge crossings from face to face), a net mode
change in ℤ₂. The transport is **consistent** if the net change depends only on
the endpoints of the path — not on the route taken; equivalently, traversing any
closed loop returns the mode to its starting value. This is the condition
differential geometry calls trivial holonomy and gauge theory calls a flat ℤ₂
connection; we will simply say the transport is **flat**. On any orientable
surface flatness is equivalent to the existence of a globally consistent mode
field: fix the mode of any one face and every other face's mode follows. Any
refinement must preserve flatness; otherwise it introduces a transport defect.

The requirement becomes non-trivial when we demand that consistency extends to
all refinements of the triangulation. A refinement introduces new faces, edges,
and vertices — and the transport must remain consistent at every scale. This is
the content of Axiom 4, and its consequences for the admissible aperture class
are developed in §6.

What the transport implies immediately is simpler: any structure built over the
triangular field — any labelling, orientation, or subdivision — must account for
the mode flip at every edge crossing, or introduce a defect. The mode is not
merely a local colouring; it is a global invariant that any grid must either
propagate correctly or break.

At individual edges the transport is binary and clean. At vertices — where
multiple edges meet — the accumulated transports must also close consistently.
What this implies for the global topology is the subject of §4.

---

## 4. Vertex Closure

Section 3 established that mode transport must be flat — any closed loop must
return the mode to its starting value. Vertices are where this condition is most
constraining: at every vertex, a ring of triangular faces meets, and traversing
that ring forms a closed loop.

Consider a vertex where k triangular faces meet (valence k). Moving from face to
face around the vertex traverses exactly k edges. Each edge crossing carries a
mode flip. For the transport around this closed loop to return to identity, the
total number of flips must be even — requiring k to be even.

This is the **vertex closure condition**: it selects triangulations in which
every vertex has even valence. It ensures that the face adjacency graph is
bipartite — no odd cycles in the dual.

Face bipartiteness alone does not require uniform valence. Many irregular
triangulations of S² admit a consistent mode assignment while mixing different
even valences. The vertex closure condition, taken alone, is compatible with
non-uniform even valence, and we make no claim that such triangulations fail
under refinement.

The selection of uniform valence comes instead from the regularity requirement
of Axiom 3: the seed complex admits no exceptional cell types and no
distinguished vertex classes. A triangulation mixing different even valences
contains several distinct vertex-star types, and every structure built over it —
refinement rules, addressing, adjacency — would have to distinguish those
classes explicitly. Axiom 3 excludes this by requirement, not by theorem: every
vertex star is of the same type, so every vertex behaves identically under
refinement and addressing. Uniform even valence is therefore imposed as a
regularity condition, not derived as a topological necessity.

Given uniform even valence, the Euler characteristic of S² constrains what is
possible. For a triangulation with V vertices, E edges, F faces:

    V − E + F = 2,   with   E = 3F/2

For uniform valence v the relation 2E = vV gives:

    V = 12 / (6 − v)

For v = 4 (the minimum even valence greater than 2): V = 6, F = 8, E = 12.
For v = 6: the denominator vanishes — uniform valence 6 cannot close on the
globe. For v ≥ 8: the formula yields a negative vertex count — impossible.

The octahedral triangulation — V=6, E=12, F=8, uniform valence 4 — is the one
abstract triangulation of S² satisfying all conditions. The division of labour
is explicit: vertex closure, a derived condition, forces even valence;
regularity (Axiom 3), a stated requirement, forces uniform valence; the Euler
characteristic then leaves v = 4 as the only possibility.

*Consequence: a ℤ₂ face-mode can be defined on any triangulation of S² with a
bipartite dual graph — equivalently, on any triangulation in which every vertex
has even valence. Vertex closure is the derived part of this argument;
uniformity of valence is required by Axiom 3 rather than derived. Together they
admit exactly one triangulation of S²: the octahedral one.*

---

## 5. The Octahedral Embedding

Section 4 established that the minimal regular seed complex compatible with mode
transport on the globe has exactly 6 vertices, 12 edges, and 8 triangular faces,
with uniform vertex valence 4. There is exactly one abstract triangulation of S²
with this combinatorial structure — and it is the octahedral triangulation. Its
geometric realisation as a convex polyhedron is the regular octahedron.

On the sphere, every embedding of the octahedral seed is equivalent: the
sphere's symmetry group O(3) contains the full octahedral group O_h in any
orientation, and no embedding is preferred. The reference ellipsoid breaks this.
An ellipsoid of revolution retains only the symmetries fixing its polar axis —
continuous rotation about the axis, the equatorial mirror, and the vertical
mirror planes (the group D∞h). The poles are the fixed points of this residual
symmetry.

An embedded octahedron shares with the ellipsoid exactly those symmetries common
to both, and the shared group depends on which octahedral axis is aligned with
the polar axis. Aligning a vertex pair (a 4-fold axis) retains D4h, of order 16;
a face pair (a 3-fold axis) retains D3d, of order 12; an edge pair retains D2h,
of order 8; a generic orientation retains almost nothing. Pole-on-vertex
anchoring is therefore not one choice among equals: it is the unique orientation
class preserving the maximal common symmetry of seed and surface.

This maximal residual symmetry is what the construction uses. Under D4h the 8
octant faces form a single orbit — every octant face is equivalent to every
other — so a single projection function serves all 8 octants, with mode-1 faces
obtained from mode-0 by one reflection (y → −y). The anchoring also reduces the
continuous rotational gauge freedom to the discrete 4-fold rotation, which the
meridian anchoring of Axiom 7 resolves. Pole anchoring does not impose a
coordinate system; it selects the embedding in which the seed inherits the most
structure from the surface.

The result is a canonical partition of the globe into 8 triangular octants — not
a projection choice, but the direct consequence of mode transport closure
applied to a closed orientable surface. The 8 octants are the top-level cells of
a grid hierarchy following these constraints. Further structure — refinement,
orientation, dual cells — is built over this seed. The octahedral embedding is
not one possible global frame among many; it is the frame the coherence
requirements construct.

---

## 6. Refinement Commutativity

The octahedral seed established in §5 defines a base simplicial complex on which
a hierarchical refinement operator acts. Each triangular face is subdivided into
k² child triangles by linear refinement at scale factor k. The refinement
operator must preserve simplicial type, adjacency relations, and the flatness of
mode transport (§3), and must commute with the global indexing of the mesh.

Refinement is chosen to commute with mode transport because mode is the global
coherence field of the system. Without commutativity, refinement would not
preserve identity under scale, and hierarchical addressing would cease to be
stable under composition. This choice is not geometrically mandatory, but it is
required for refinement to function as a consistent coordinate extension rather
than a sequence of unrelated discretisations.

These requirements constrain admissible values of k. First, k must be a positive
integer to ensure that refinement is a well-defined subdivision of the
simplicial structure into congruent refinement classes.

Second, refinements that preserve mode transport consistency are those in which
the mode of every child triangle agrees with the mode inherited from its parent
at every induced edge (a homomorphism of the ℤ₂ transport structure, in
algebraic terms).

For even k, the refinement fails this test. Consider k=2: a mode-0 parent
produces 4 children — 3 corner children of mode 0 and 1 inverted central child
of mode 1. At the boundary between two adjacent parents (which have opposite
modes), this places mode-0 children of the mode-0 parent adjacent to mode-0
children of the mode-1 parent — directly contradicting the required mode flip at
that boundary. The transport operator disagrees with itself across the
inter-parent edge. This breaks the flatness of mode transport across the
refinement, violating commutativity with global indexing.

For odd k, the refinement preserves parity alignment between parent and child
simplices, ensuring that mode transport is consistently inherited at all scales.
Admissible values of k are therefore restricted to odd integers within this
refinement class: **k ∈ {1, 3, 5, 7, …}**

When restricted to refinements that preserve mode transport consistency and
commute with global indexing, composing two valid refinements always yields a
valid refinement (the admissible class forms a semigroup, in algebraic terms).
Within this class, odd integers define valid scales, and the minimal non-trivial
scale is k = 3.

We therefore select k = 3 as the base refinement operator. Higher-scale
refinements are obtained by composing this operator, yielding a hierarchy
indexed by powers of 3.

In the k=3 case, refinement introduces a systematic lateral displacement of
child simplex centroids relative to the parent geometry. This displacement is
not an artefact of embedding, but the geometric manifestation of
parity-preserving refinement under ℤ₂ transport. In the dual structure, this
offset becomes visible as the characteristic shift in the induced hexagonal
lattice, giving rise to the **shifted-aperture-9 hierarchy**.

---

## 7. Dual Projection — Hexagonal Structure

The dual of a triangulation exchanges faces and vertices: each triangular face
becomes a dual node, and each vertex becomes a dual face whose valence equals
the vertex degree.

In the refined octahedral triangulation, interior vertices attain valence 6,
producing hexagonal dual cells. The six vertices of the octahedral seed
constitute the only non-uniform elements of the triangulation; their deviation
from valence 6 induces the only non-hexagonal dual cells in the system — six
four-sided dual cells, one per seed vertex, each carrying a valence deficit of
2, together accounting for the total defect of 12 required by the Euler
characteristic (§4). Refinement spreads regular valence-6 structure throughout
the interior while localising irregularity to the seed.

It follows that the hexagonal lattice is not imposed but emerges as the dual of
the locally regular simplicial field. The octahedral embedding provides the
minimal finite source of irregularity required by the Euler characteristic of
S², while preserving maximal hexagonal regularity elsewhere.

Within each octant, the dual lattice forms a half-hexagonal region (the
**half-hexagon**) bounded by seed edges. The k=3 refinement introduces a
systematic lateral displacement of these regions in the dual lattice. This shift
is the spatial manifestation of enforcing refinement as a homomorphism of the
mode transport structure.

The planar structure has a crystallographic signature. The hexagonal tiling cut
along the half-hexagon long edges in the Hex9 arrangement has wallpaper group
p31m — a strict reduction from the p6m of the uncut tiling. Colouring by mode
reduces it further: every mirror of p31m exchanges the two modes (a
mode-preserving mirror would make the tiling achiral, contradicting the chiral
pair of §8), so the mode-preserving subgroup is exactly the rotational part, p3.
The ℤ₂ of §2 is, in crystallographic terms, the quotient p31m / p3.

![Crystallographic structure of the d_cell tiling: the translational unit (left) and its IUC symmetry — three-fold centres, mirror and glide lines, fundamental domain (right). The symmetry-reduction chain is p6m → p31m → p3 (§7).^[Source figure `paper_figures/f20.png`.]](paper_figures/f20.png){width=85%}

The resulting system consists of a hexagonal dual lattice with a minimal,
seed-localised defect structure and a refinement-induced shift. The remaining
degree of freedom is the global and inter-octant orientation of this lattice.

---

## 8. Orientation Selection

The half-hexagon established in §7 is the natural octant carrier — the region of
the dual hexagonal lattice bounded by the three seed edges of a single octant
face. Its internal structure admits a further decomposition: the half-hexagon
divides into three equilateral sub-regions, each corresponding to one third of
the octant face.

Each sub-region can be independently oriented in two ways: the hexagonal cells
within it may be arranged in one of two configurations related by reflection.
With three sub-regions and two choices each, there are 2³ = 8 candidate
orientations for the half-hexagon as a whole.

The mode transport constraint — specifically, the vertex closure condition
established in §4 — acts on the internal boundaries between sub-regions. At every
vertex where two or more sub-regions meet, the mode transport around that vertex
must return to identity: trivial holonomy. This is the same condition that
selects for even valence globally; applied to the internal half-hexagon
boundaries, it filters the 8 candidates.

Exactly 2 of the 8 combinations satisfy the internal closure condition at every
boundary vertex. The two survivors are related by a global reflection — a chiral
pair, geometrically distinct but structurally equivalent up to handedness.

![The surviving chiral pair: the two orientations of the 9-cell equilateral that satisfy internal closure (three half-hexagons each, coloured by orientation class). The members are mirror images — the residual chirality Axiom 6 fixes.^[Generated by `experimental/halfhex_further.py`; enumeration counts verified by `experimental/halfhex_verify.py`.]](paper_figures/f2.pdf){width=55%}

This count is not an assertion. The full enumeration is machine-verified by
`experimental/halfhex_verify.py` (checks V0–V6): of the 49 distinct hextile
solutions (24 chiral pairs + 1 self-mirror), the long-edge constraint (A) admits
18, the three-equilateral structural constraint (B) admits 8, and their
intersection A ∩ B is exactly the recorded Hex9 chiral pair. The constructive
2³ → 2 argument above and this enumeration agree.

![The 49 distinct hextile solutions, chiral pairs grouped and the self-mirror marked, with the Hex9 pair highlighted. Of 49 = 24 pairs + 1 self-mirror, constraint A admits 18, constraint B admits 8, and A ∩ B is the highlighted pair.^[Generated by `experimental/halfhex_further.py`; counts machine-verified by `experimental/halfhex_verify.py`.]](paper_figures/f1.pdf){width=85%}

Axiom 6 resolves the remaining freedom: a single consistent chirality choice,
induced by the embedding into the reference ellipsoid frame, selects one of the
two. The orientation of the half-hexagon — and with it the entire hexagonal
lattice across all 8 octants — is determined: no further choice remains.

This closes the enumeration. Each cell in the hierarchy is fixed by its octant,
its refinement level, and its index within that octant at that level. No two
cells share the same address; no address refers to more than one cell. The
hexagonal lattice is globally indexed with no residual ambiguity. What follows
from this is the subject of §9.

---

## 9. Hex9 Cell Identity — Structure as Locator

The system defined by Axioms 1–9 and constructed through the preceding steps is
**Hex9**: a shifted-aperture-9 hexagonal grid on an octahedral embedding of the
reference ellipsoid, with a cell hierarchy in which every cell has a distinct
address, derived from simplicial coherence requirements alone.

![The seed solid and the 12 root cells. Left: the octahedron with each octant face creased into its three d_cell facets (24 faces, the diploid form that names the d_cell). Right: coloured per root x_cell, hue by octahedral axis, light/dark for the mode-0/mode-1 halves — at L0 every root cell is one of the 12 topological pentagons. The faceting is illustrative; the cells live on the smooth ellipsoid.^[Source figure `paper_figures/f21.png`.]](paper_figures/f21.png){width=92%}

Each Hex9 cell is identified by its octant (one of 8) and its path through the
refinement hierarchy. This pair is not a coordinate computed from a prior
reference system — it is the cell's identity. The bijection of Axiom 8 (Unique
Cell Addressing) guarantees that this identity encodes a specific, unambiguous
geographic region. At each finite refinement level the correspondence between
addresses and cells is exact: one address, one cell, no exceptions. Cell
location is recovered directly from combinatorial structure; no external
coordinate transformation is required to establish it.

Hex9 is a discrete spatial reference system in the sense of OGC Abstract
Specification Topic 21 [@ogc_topic21], which characterises a DGGS as a spatial
reference system that partitions and addresses the globe through a hierarchical
tessellation of cells. What distinguishes Hex9 within this class is the
direction of derivation. The conventional workflow — select a coordinate
reference system, then design a grid over it — is inverted. Here, the geometric
coherence requirements of §§1–8 define the structure; the structure constitutes
the locating system. No prior CRS is required as input.

This correspondence is discrete, not continuous. Hex9 cell identity uniquely
identifies a geographic region at each resolution. The hierarchy of nested
regions converges to a point as refinement increases, but Hex9 does not provide
continuous coordinates in the sense of a projected CRS. For metric operations —
distances, areas, interpolation — a geometric realisation onto the reference
ellipsoid remains necessary, and is the subject of §11.

The enumeration completed in §8 is what makes self-contained location possible.
A grid with residual design freedom — orientations, apertures, or embeddings
left as choices — cannot serve as its own locating system, because different
choices produce different grids that disagree on cell identity. Hex9 has no
residual freedom. Every cell is where it is because every constraint converges
on that location.

---

## 10. Addressing and Continuity

### 10.0 Notation — the c/t/d/x grid taxonomy

The constructions of §§6–8 generate four overlapping grid spaces, designated by
single letters: **c**, **t**, **d**, **x**. The sections that follow use this
vocabulary constantly; this section fixes it. (The authoritative glossary,
maintained alongside the implementation, extends these definitions.)

**t_cell** — a triangular cell of the refined field: the working unit of the
simplicial carrier. Each t_cell carries an intrinsic mode (0 = ∇, 1 = Λ; §2) and
subdivides into 9 child t_cells at the next level (§6). Within a parent context,
a child t_cell occupies one of 12 positional classes, written as a **region** id
(0–11): 6 classes are shared across both parent modes, 3 occur only under a
mode-0 parent, and 3 only under a mode-1 parent.

**c2** — the label (0, 1, 2) of a t_cell edge, assigned by edge gradient: 0 =
flat (horizontal), 1 = forward (positive slope), 2 = back (negative slope).
Adjacent triangles agree on the c2 value of their common edge, and the c2
progression around any triangle is clockwise regardless of mode. *Etymology:*
"c2" is shorthand for **"colouring 2"** — one of the two distinct
three-colourings of the wallpaper group **p31m** (Grünbaum & Shephard 1987
[@grunbaum1987tilings], *Tilings and Patterns*, §8.3 "Color Pattern Types", p.
433, type IH38; PP25[3]₁/p31m[3]₁ and PP25[3]₂/p31m[3]₂ — see the
`grunbaum_shephard_*` figures, the second with the Hex9 d_cell overlay aligned).
Note a **very early naming mismatch**: despite the "2", the Hex9 c2 labelling
actually corresponds to Grünbaum & Shephard's *first* colouring, **p31m[3]₁**,
not the second.

**c_cell** — a slot in the 96-position classifier grid: the raw output of the
three-family inequality classification of §10a. Twelve of the 96 slots are the
in-scope t_cell classes; the remainder are out of scope by construction.

**d_cell** (half-hexagon) — three t_cells grouped by their shared long edge. The
c2 value of that long edge is the d_cell's digit (d_dig ∈ {0, 1, 2}), and the
d_cell inherits the mode of its t_cells. The d_cell is the fundamental domain of
the tiling argument in §8.

**x_cell** (hexagon) — one mode-0 d_cell joined with one mode-1 d_cell on their
matching c2 edge. This is the public cell of the Hex9 grid: 12 root x_cells cover
the globe, and each x_cell has 9 children.

**x_dig** — the digit (0–8) naming a child x_cell within its parent. Read it in
ternary: the high trit encodes mode ownership (0 = the child's mode-0 half is
interior to the parent context; 1 = the mode-1 half is interior; 2 = the child
is **split**, straddling two parents), and the low trit records the c2
orientation of the child's long edge. The three split children per parent
(x_dig ∈ {6, 7, 8}) are the only cells with two valid parents (§10b).

**Lists and addresses** — c_list, t_list, and d_list are digit sequences over
strict single-parent trees: they compose left-to-right and are prefix-sortable.
An **x_list** is the sequence of x_digs; because of the split cells it is
resolved right-to-left. An **x_adr** is an x_list plus a **tail** — a single
metadata byte that resolves split-cell parentage and terminal state (§10b). The
tail is metadata only: it never participates in geometric computation.

![Anatomy of one parent triangle: t_cells, c2 edges, d_cells, and the assembled x_cells with x_dig labels — the c/t/d/x taxonomy in one picture.^[Generated by `examples/ex0400_anatomy.py`.]](paper_figures/ex0400_anatomy_f3.png){width=90%}

### 10a. Identity as Locator

A Hex9 address is a pair: an octant index (one of 8) and a refinement path — a
sequence of x_dig values recording which child x_cell was entered at each level
of the hierarchy. An x_adr of depth L identifies exactly one x_cell at level L on
the reference ellipsoid.

The reversibility of this mapping rests on the construction. At each level, a
parent x_cell contains exactly 9 child x_cells, whose arrangement is determined
by the t_cell → d_cell → x_cell sequence established in §§6–8. Each x_dig (0–8)
selects one of those 9 children unambiguously. The refinement tree is a strict
single-parent structure for t_cells and d_cells; x_cells inherit this property
except at the 3 split x_cells per level, whose parentage is resolved by the tail
of the x_adr.

![The x-layer: a parent x_cell and its 9 child x_cells labelled by x_dig (0–8), coloured by high-trit class (0–2 / 3–5 / 6–8). The three split children (6,7,8) straddle the parent boundary; there is no central child.^[Generated by `examples/ex0400_anatomy.py`.]](paper_figures/ex0400_anatomy_f4.png){width=85%}

Once the projection from the reference ellipsoid to the octant 2D plane is
complete, the encode direction (point → x_adr) operates via a sequence of linear
inequality evaluations. Three families of parallel lines partition the octant
plane into the triangular grid:

- horizontal bands: $y$ compared against fixed thresholds;
- positive-slope bands: $y - \sqrt{3}\,x$ compared against fixed thresholds;
- negative-slope bands: $y + \sqrt{3}\,x$ compared against fixed thresholds.

A point's t_cell at a given level is determined by which band it occupies on each
family — a small fixed number of comparisons, with no geometric distance
computation and no iterative search. At each successive refinement level the
thresholds scale by 1/3, maintaining identical structure. The c_cell (the
96-slot classifier combining horizontal and slope bands) maps directly to a
t_cell, and from there to the d_cell and x_cell via the construction of §§6–8.

![The classifier (c-layer): three band families partition the octant plane; their indices compose the 96-slot c_grid digit, which maps to a t_cell and thence to d_cell and x_cell. Locating a point is band membership, not search.^[Generated from the `addressing.py` classifier (`paper_figures/f5.png`).]](paper_figures/f5.png){width=85%}

The mapping operates in both directions. Given an x_adr, the corresponding
geographic region is recovered by tracing the digit sequence from the octant
root: each x_dig selects a child x_cell whose boundary is determined by the
refinement geometry. Given any point on the ellipsoid, its x_adr at level L is
recovered by projecting to the octant plane and evaluating the inequality
sequence down to depth L.

Both directions are exact at each finite level. Encode produces no
approximation: every point belongs to exactly one x_cell at every level. Decode
recovers a cell whose geographic extent is precisely determined by the
octahedral embedding and the refinement geometry.

This reversibility qualifies Hex9 as a locating system rather than merely an
indexing scheme. Location is not inferred from the address — it is recovered from
it by retracing the construction.

### 10b. Identity as Key

A Hex9 address is also a spatial key — a string over a small digit alphabet that
can be stored, compared, and sorted without reference to any geometric
structure. This makes Hex9 addresses directly usable as database keys, hash
keys, or binning primitives.

The key property is that prefix order corresponds to containment. All cells whose
address begins with a given prefix σ are contained within the cell identified by
σ. This means spatial containment queries reduce to prefix comparisons: to find
all level-L cells within a given level-K region (L > K), select all addresses
sharing that region's prefix. No geometric computation is needed.

Spatial joins between two datasets reduce similarly: two observations share a
cell at level K if and only if their addresses share a common prefix of length K.
Aggregation across levels is achieved by truncating addresses to the desired
depth. These operations are efficient and require no coordinate arithmetic.

Because the address space is defined by the refinement structure rather than by a
numerical coordinate grid, there are no edge effects, no wrap-around anomalies,
and no cells that straddle index boundaries. Every cell has exactly one address;
every address identifies exactly one cell. The key space is clean.

This last claim is grounded in the address tail. The x_list alone is not always
sufficient: at each level, 3 of the 9 child x_cells (those with x_dig in {6,7,8})
straddle the c2 boundary between two adjacent d_cells and have two valid parents.
Beyond this, two cells with distinct terminal regions can produce identical digit
sequences when those regions generate the same hex digit from different parent c2
contexts.

The key tail carries two fields that resolve these ambiguities. The first is
p_c2 — the parent c2 of the terminal region. A concrete instance: hex digit 6 at
the terminal level arises from both region 9 (mode 1, parent c2 = 0) and region 6
(mode 0, parent c2 = 2) within the same parent context. Both paths produce an
identical digit body; without p_c2 they collide as keys. With p_c2 = 0 or
p_c2 = 2 respectively, the two addresses are distinct. The second field is r_mo —
the root octant's net_mode. Two octants of opposite mode can produce the same
root hex digit; without r_mo the decoder cannot recover which octant the address
originates from, and cells in distinct geographic regions would share the same
key.

The full reversible tail additionally carries p_mo and h. The p_mo field records
the actual parent mode of the terminal region; for split x_cells, this may differ
from the key tail's canonical mode-0 assumption. Without p_mo, the decoder
recovers the mode-0 parent's representative, not the exact terminal cell. The h
field identifies the terminal d_cell (one of 12, bits 3–0); without it,
reconstruction returns the x_cell centre. With h, the precise d_cell centroid is
recovered. The reversible tail is fully invertible: the address body together
with the reversible tail uniquely and exactly determines both the geographic
region and a representative point within it.

The tail fields in summary:

| Field | Bits | Carries | Without it |
|---|---|---|---|
| p_c2 | 2 | parent c2 of the terminal region | distinct cells collide as keys (same digit body from different parent c2 contexts) |
| r_mo | 1 | root octant net mode | octant unrecoverable from the root digit; cross-octant key collisions |
| p_mo | 1 | actual parent mode of the terminal region (reversible tail only) | decoder returns the canonical mode-0 representative, not the exact cell |
| h | 4 | terminal d_cell id, 0–11 (reversible tail only) | reconstruction returns the x_cell centre, not the exact d_cell centroid |

The key tail (p_c2, r_mo) suffices for unique binning; the reversible tail (all
four fields) gives an exact round trip to a representative point.

One caution follows directly. Truncating an address to length K always
identifies a valid ancestor at the corresponding level, but not necessarily the
*canonical* one: a split cell encoded under its mode-1 parent truncates into the
mode-1 lineage. Binning by naive prefix-cutting therefore silently produces two
bins for the same cell whenever non-canonical addresses are present. The correct
operation derives the canonical ancestor via the tail before truncating:
prefix-cutting is exact for resolution identification; canonical ancestry
requires the tail.

A worked example: central London. The Prime Meridian is a c2 boundary in this
region. At level 4, the cells around Greenwich sit in three adjacent hexagons
with non-adjacent prefixes:

| L4 address | Location |
|---|---|
| 43483 | corner of north London |
| 43486 | east of Greenwich |
| 43527 | west of Greenwich |

43527 appears to have jumped lineage: it belongs to 4352 (south-west England)
despite being geographically adjacent to 43486 in 4348 (east England). The jump
is not an anomaly — it is the visible signature of the split digits. Digits 6–8
carry high ternary trit 2: thesecells straddle the d_cell boundary, and the
canonical mode-0 parent convention places geographically adjacent cells on
opposite sides of that boundary into different canonical lineages, exactly as the
construction requires.

![The split-cell lineage jump over central London and the Thames estuary (Equal Earth, EPSG:8857). Hexagons are labelled by Hex9 address; the bold **4348** and **4352** mark the two level-3 parent cells, and the heavy line is their boundary. Cell **43527** lies west of Greenwich yet belongs to the **4352** lineage, while its geographic neighbours **43486** and **43483** belong to **4348** — the canonical mode-0 convention placing adjacent cells across the c2 (Prime-Meridian) boundary into different lineages, exactly as §10b describes.^[Rendered in QGIS over an OpenStreetMap backdrop; hex boundaries and addresses from `hhg9`.]](paper_figures/f6_epsg8857_web.jpg){width=90%}

### 10c. Identity as Label

The full Hex9 address space is bipartite: at every level, cells carry a mode (0
or 1) inherited from the refinement structure established in §2. For internal
computation — transport, refinement, adjacency — mode is a meaningful structural
property. For labelling purposes, exposing it is unnecessary.

The mode-0 cells at each level form a complete cover of the globe: every point on
the reference ellipsoid is contained in exactly one mode-0 cell at every level.
This makes the mode-0 hierarchy a natural canonical labelling system. Any cell —
regardless of its own mode — can be identified by the address of its enclosing
mode-0 cell at the same level.

Labels constructed this way are self-contained: a label always names a mode-0
cell, so no mode flag, lookup table, or supplementary field is required to
interpret it. The label is the identity. This is the complement of the address
tail of §10b: the tail is needed when the exact terminal cell — possibly mode-1,
or a specific d_cell within it — must be recovered. A label makes the opposite
trade: by always naming the enclosing mode-0 cell, it needs no tail at all.

This collapse is not a loss of information. The bipartite structure remains
present in the refinement geometry; the label simply presents a face of it that
is uniform and compact. A label identifies a specific geographic region at a
specific resolution. Nothing more is needed; nothing is omitted.

### 10d. Adjacency from Refinement Paths

Two x_cells are adjacent if they share a boundary edge on the reference
ellipsoid. In Hex9, adjacency is recoverable from the refinement structure
because the x_cell geometry is fully determined by the t_cell → d_cell → x_cell
construction.

Within a parent x_cell, the 9 child x_cells tile the parent's region. Their
shared edges are the c2 edges of the underlying d_cells — the same long-edge
alignment that the orientation selection of §8 fixed globally. Adjacency is
realised as a fixed, finite lookup: every cell has exactly three neighbours, one
per c2 value, given by a constant table keyed by (cell, parent mode, c2). The
digit structure mirrors this: each split digit k+6 (k ∈ {0, 1, 2}) names the pair
of d_cells flanking interior child k, and the low ternary trit of every x_dig
records the c2 orientation of the cell's long edge.

![Sibling adjacency within a parent: each child t_cell edge is internal (blue, shared with a sibling) or external (red, cross-parent); a child's adjacency class is its red-edge count (interior 0, mid-edge 1, vertex 2).^[Generated by `examples/ex0400_anatomy.py`; classes verified against `region.py`'s neighbour builder.]](paper_figures/ex0400_anatomy_f7.png){width=85%}

The same table classifies every cell's relationship to its parent boundary.
Within each parent, cells fall into three classes: interior cells, whose
neighbours all share the parent; mid-edge cells, with exactly one neighbour in an
adjacent parent; and vertex cells, with two. For the latter classes the lookup
flags the crossing and identifies the relevant child in the neighbouring parent —
a computable operation on the address structure, stepping up one level in the
refinement tree and applying the same c2 edge relationships that govern interior
adjacency. Where the neighbouring parent lies across an octant seam, the octant
congruence of §5 reduces the hop to an octant-index lookup composed with the
y-reflection.

Adjacency across octant seams follows from the d_cell c2 alignment at octant
boundaries. The orientation selection of §8 ensures that d_cells at an octant
edge meet long-edge to long-edge with the d_cells of the neighbouring octant. The
x_cells that form across this boundary are assembled by the same d_cell join rule
as everywhere else. Seam-crossing adjacency is therefore not a special case — it
is the same c2-edge query applied at the octant boundary.

In each case adjacency is a finite computation on the combinatorial structure of
the refinement tree and the c2 edge table. No geometric distance query is needed,
and no location in the grid requires a different procedure from any other.

### 10e. Continuity

At any finite level L, a Hex9 address identifies a cell of finite geographic
extent. As L increases, cell diameter decreases by a factor of 3 at each level,
converging toward zero. An infinite address sequence — a path carried to all
levels of the hierarchy — therefore defines a nested sequence of cells whose
diameters tend to zero.

The reference ellipsoid is a compact metric space. By Cantor's intersection
theorem, a nested sequence of closed regions with diameters converging to zero
has exactly one point in its intersection. An infinite Hex9 address sequence
identifies exactly that point: not a region, but a location on the ellipsoid.

This convergence lets a function recover position from an address to arbitrary
precision: decoding an address of growing length yields a sequence of points
converging to a unique location on the ellipsoid (Appendix A). A finite address
identifies a region; a sufficiently long address identifies a location to any
required tolerance.

This is not the same as being a coordinate reference system in the strict sense
of ISO 19111, and we do not claim it is. Two limitations are intrinsic. First,
the address alphabet is discrete: addresses form a totally disconnected sequence
space, not a continuum, so they are not coordinates in the real-valued sense.
Second, the point → address map is discontinuous on the measure-zero set of
d_cell seams (the split-cell boundaries of §10b): arbitrarily close points on
opposite sides of a seam receive addresses that differ in their leading digits.
ISO 19111 presupposes continuous coordinates, and Hex9 does not meet that
requirement.

What Hex9 offers is better described, by analogy with its quasi-authalic
geometry, as **quasi-continuous**: position is recoverable from the address by a
function, to arbitrary precision, everywhere except on a measure-zero seam set,
and the cell hierarchy converges to points rather than terminating at a finite
floor. We are not aware of another DGGS whose cell identifier doubles as a
position-recovery coordinate in this way, though we do not claim the property is
unique. At any fixed finite resolution Hex9 remains a discrete system — cells are
regions, the bijection is between addresses and regions — and the
quasi-continuous behaviour emerges only in the limit. This is consistent with
§9: Hex9 is a discrete spatial reference system that approaches, but does not
attain, a continuous one.

### 10f. Seams and Valence Defects

The Euler characteristic of S² requires that any triangulation of the globe carry
topological defects. In Hex9 these are absorbed at the six octahedral vertices,
each of which is surrounded by 4 hexagonal cells rather than 6. Understanding why
neither these defects nor octant seams require special handling requires tracing
the constructive sequence: t_cells → d_cells → x_cells.

At each refinement level, each triangular face (t_cell) is subdivided into 9
child t_cells. Three adjacent t_cells — grouped by their shared long edge (c2
edge) — form a half-hexagon (d_cell). A d_cell carries an intrinsic mode (0 or
1). One mode-0 d_cell and one mode-1 d_cell, joined on their matching c2 edge,
form a hexagonal cell (x_cell). The hexagonal grid emerges entirely from this
sequence; no independent hex tiling is assumed.

*(The t_cell → d_cell → x_cell constructive sequence is the anatomy figure of
§10.0.)*

At an octant seam, the d_cell c2 alignment ensures that d_cells on either side of
the shared octant edge meet long-edge to long-edge. This alignment is not
enforced separately — it is the consequence of the orientation selection of §8,
which chose precisely the arrangement in which c2 edges align at every boundary.
The x_cells that straddle the seam are formed by the same d_cell joining rule
that applies everywhere else. The seam is a boundary in the refinement tree, not
a discontinuity in the construction.

At an octahedral vertex, four octant faces meet rather than six. The surrounding
d_cells at this vertex have c2=0 — the flat (horizontal) edge — as their shared
long edge. This is the c2=0 convention: not a patch, but the consequence of the
same vertex-loop transport closure established in §4. Even valence at the
octahedral vertices (valence 4) satisfies the mode transport condition; the
surrounding x_cells are formed by the same d_cell join rule. The resulting
x_cells are fewer (4 rather than 6) but structurally identical in construction.

In both cases the construction proceeds without branching. The t_cell → d_cell →
x_cell sequence applies uniformly across the entire globe — at seams, at defect
vertices, and in the interior. The absence of special cases follows from the
coherence of the construction, not from exception handling added afterward.

---

## 11. Geometric Realisation (AK + Warp)

Sections 1–10 establish the Hex9 grid as a combinatorial object: mode transport,
t_cells, d_cells, x_cells, the refinement hierarchy, and the address structure
are all defined without reference to any specific map projection or reference
body. The geometric realisation is a separable step that places this abstract
structure onto the WGS84 reference ellipsoid.

Three concerns are independent: (1) the combinatorial structure and hierarchy
(§§1–10); (2) the base projection from octahedron to ellipsoid; (3) the area
correction. Each can be understood, improved, or substituted without disturbing
the others. The warp is ellipsoid-specific; the grid is not.

```{=latex}
\begin{figure}[h]
\centering
\resizebox{\textwidth}{!}{%
\begin{tikzpicture}[node distance=8mm and 16mm, >={Latex[length=2mm]},
  box/.style={draw, rounded corners, align=center, font=\small,
    minimum height=10mm, inner sep=4pt},
  op/.style={font=\footnotesize\itshape, align=center}]
  \node[box] (g)   {Geodetic\\(WGS84 lat/lon)};
  \node[box, right=of g]   (raw) {b\_raw\\(octant)};
  \node[box, right=of raw] (oct) {b\_oct\\(warped)};
  \node[box, right=34mm of oct] (adr) {Hex9 address\\(x\_adr / UUID)};
  \draw[->] (g)   -- node[op, above]{AK projection\\(\S11a)} (raw);
  \draw[->] (raw) -- node[op, above]{authalic warp\\(\S11b)} (oct);
  \draw[->] (oct) -- node[op, above]{partition cycle\\(\S10a, \S13a)} (adr);
  \draw[->, dashed] (adr.south) to[out=-150,in=-30]
    node[op, below]{Newton--Raphson inverse warp $+$ decode} (g.south);
\end{tikzpicture}}
\caption{The geometric-realisation pipeline (\S11). A WGS84 geodetic coordinate
reaches a combinatorial Hex9 address through the AK base projection, the authalic
warp, and the partition cycle; the inverse (dashed) retraces the chain with a
Newton--Raphson warp inversion. The three concerns --- combinatorial hierarchy,
base projection, area correction --- are independent and separately substitutable.}
\end{figure}
```

### 11a. The Base Projection

The AK octahedral projection maps each of the 8 octant faces to the corresponding
region of the reference ellipsoid. Its forward formula, designed analytically by
Anders Kaseorg [@kaseorg_octahedral] from a force-directed dataset, applies a
tangent substitution to each octant coordinate and couples the three axes with a
fourth-root term modulated by a parameter α ≈ 3.2278. This coupling partially
compensates for the area asymmetry between mode-0 and mode-1 triangular cells
that arises from the non-uniform Jacobian of the octant face.

The projection is smooth and has an analytical Jacobian — properties that make it
suitable as the inner layer beneath the Sinkhorn warp. The 8 octahedral vertices
coincide exactly with ellipsoidal surface points; no projection computation is
needed at those locations. The 8 octant faces are mutually equivalent under a
y-coordinate reflection, so one projection function serves all octants.

No closed-form inverse exists. The backward pass — ellipsoid to octant — is
realised by numerical root-finding; the fast path is a guarded Gauss–Newton
iteration on the analytic Jacobian (§13f), with an exact beam search as the
reference and as the fallback near seams and vertices. The forward map is smooth and injective
on each octant, so an inverse is guaranteed to exist; the numerical method is an
implementation choice, not a structural requirement.

Without further correction, the AK projection introduces roughly ±20% area
deviation across the octant face — larger apparent cells near octant corners,
smaller near the centre.

A second, subtler artefact is an inter-mode area bias. The octant face is a right
isosceles triangle, so its three corners are not geometrically equivalent and the
AK Jacobian is not constant over the face; mode-0 and mode-1 triangles sample
that non-uniform field at systematically different centroid positions. Measured
by geodesic area on WGS84, the mode means differ by 0.394% at L4 and 0.131% at
L5, shrinking roughly threefold per refinement level (projected ≈ 0.04% at L6).
The coupling parameter α is not the cause — removing it (α = 0) worsens both the
global deviation (σ ≈ 7.6% → ≈ 17%) and the asymmetry — and the Sinkhorn warp,
operating at the hexagon level, cannot rebalance areas between the two halves of
a hexagon, so the residual carries through to the warped result at slightly
reduced magnitude. At L5 and finer it is negligible for practical use.

### 11b. The Authalic Warp

The warp corrects the area deviation left by the base projection. It is derived
by Sinkhorn optimal transport [@cuturi2013sinkhorn]: treating the L4 (or L5)
triangle vertices as a discrete mass distribution on the octant, the Sinkhorn
iteration finds the minimal-displacement redistribution that equalises projected
cell areas against the geodesic areas they subtend on WGS84. The result is a
displacement field — corrections in b_oct coordinates — rather than absolute
positions, which improves numerical conditioning.

The displacement field is precomputed once per ellipsoid and stored. At runtime,
it is applied via a Clough-Tocher interpolant [@clough1965finite] (C1
continuous). The inverse warp uses the forward interpolant to obtain an initial
estimate, then refines by Newton-Raphson to a tolerance of 10⁻¹⁴ in b_oct
(barycentric) units. The geodetic round-trip g → b_oct → g, measured on WGS84 at
validation points that include a near-pole location (89.99°N) and the Greenwich
seam, returns to within 1.8 nm of the original position — many orders of
magnitude below any geodetic relevance, and comfortably under the 7 nm design
threshold.

The achieved area uniformity (L5, all 708,588 hexagons, WGS84, production warp
file, geodesic areas): mean deviation exactly 0.000% (confirming closure: the
cell areas sum to the ellipsoid surface area); area deviation min −3.57%, max
+4.80%; mean absolute deviation 0.001%; log-ratio standard deviation 1.99×10⁻⁴.
Half of all cells are within 0.0002% of ideal area; 99% within 0.0044%; 99.99%
within 0.43%. The extreme values are a balanced ±4% pair affecting a very small
number of cells immediately adjacent to the six octahedral vertices; the bulk
distribution is highly uniform. The warp is quasi-authalic rather than strictly
authalic. A strictly authalic projection constrains only the Jacobian
determinant, permitting severe shear and cell elongation. The optimal-transport
derivation implicitly regularises against shear by minimising displacement: the
result trades a small area residual for a smooth, low-distortion displacement
field.

![Per-hex area deviation, colour scale capped at ±1% (magnitude view). The cap reveals the structure: the interior is near-white — the equal-area result — while the small cells at the six octahedral vertices, whose residuals reach the ±4–5% extremes, saturate the scale. Cells beyond ±1% are clipped.^[Generated by `examples/ex0081w_warped_authalics.py` (`snow_globe`, `lim_pct=1.0`).]](paper_figures/ex0081wau_5_web.jpg){width=58%}

![Per-hex area deviation, Mollweide, on a scale clipped to roughly the p1–p99 band (pattern view). The clip makes the spatial structure legible: a quiet near-white interior with the octant seam skeleton and six vertex blooms picked out. The colourbar is clipped and does **not** reach the true extremes (min −3.57%, max +4.80%).^[Generated by `examples/ex0118_mollweide.py` (L5).]](paper_figures/ex0118_mollweide_L5_web.jpg){width=92%}

![The north-pole vertex at L5, where the "quasi" in quasi-authalic lives. Every hexagon is one L5 cell — about 720 km² (≈ 33 km across) on WGS84 — and almost all of them are white (equal-area to within the warp residual); only the small cluster abutting the pole vertex carries a visible ±area deviation. The residual is real, bounded, and confined to a handful of cells at the irreducible vertex singularity.^[Crop of the L5 grid at the north pole; generated by `examples/ex0081w_warped_authalics.py`. Each cell is an L5 Hex9 cell (510,065,622 km² / (12·9⁵) ≈ 720 km²; 708,588 cells globally).]](paper_figures/ex0081wau_NP_crop_l5.png){width=95%}

Six irreducible vertex singularities remain at the octahedral poles — the
geographic N and S poles and the four equatorial points at 0°, 90°, 180°, 270°E.
This residual is topological in origin: the Gauss-Bonnet theorem requires
concentrated curvature at the 4-valent octahedral vertices, and no smooth warp
can remove it entirely. The singularities are fixed and geometrically determined.
On WGS84 they are not symmetric: the geographic poles produce more pronounced
residuals than the equatorial vertices, a consequence of ellipsoidal oblateness
concentrating curvature at the minor-axis tips.

### 11c. The Native Space

After the warp, each x_cell occupies a well-defined region in b_oct — the octant
barycentric space. In b_oct, the hexagonal cells are congruent and regular: no
cell is larger, smaller, or differently shaped than any other (modulo the
irreducible vertex residuals). b_oct is the natural coordinate space for Hex9
computation; it is where the inequality evaluations of §10a operate and where
address digits correspond to regular geometric subdivisions.

Any shape variation that appears when Hex9 cells are rendered in a conventional
map projection — elongated cells near the poles in Mercator, apparent distortion
in geographic lat/lon — is a consequence of that reprojection, not a property of
the cells. The distortion belongs to the map. The cells, in their native space,
are what they are by construction.

This is the practical meaning of the claim in §9 that Hex9 is a spatial reference
system rather than merely an indexing scheme: b_oct is a legitimate projection in
its own right. Conventional geographic projections are derived from it, not the
other way around.

![Tissot indicatrices on the AK+Warp b_oct butterfly net: near-constant-radius circles across the whole net demonstrate equal area, and their near-circularity the low shear the optimal-transport derivation buys (§11b). Mercator and geographic comparison panels are to follow.^[Generated by `examples/ex0121_tissot_svg.py`.]](paper_figures/ex0121_tissot_50_warp_file_butterfly_0500.pdf){width=95%}

### 11d. Projection Independence

The AK+Warp combination is the default realisation of the Hex9 grid on WGS84,
chosen because quasi-equal-area cells make point counts and density estimates
geographically comparable. It is not the only valid realisation.

The combinatorial Hex9 structure is projection-independent. Any projection from
the octahedron to the ellipsoid can be composed with the addressing scheme to
produce a valid Hex9 coordinate system. The Lee conformal projection
[@lee1965conformal] produces angle-preserving cells at the cost of non-uniform
area; the Snyder octahedral equal-area projection [@snyder1992equal] achieves
strict det(J) = 1 analytically. Both compose cleanly with the Hex9 hierarchy; the
cell identity, digit assignment, and adjacency structure are unchanged across
projection choices.

The warp is also ellipsoid-specific but not ellipsoid-exclusive. Recomputing the
Sinkhorn displacement field against a different reference body (GRS80, Bessel, a
planetary ellipsoid) yields a Hex9 realisation for that body. The addressing
hierarchy and all combinatorial structure carry over unchanged; only the
area-correction data changes. WGS84 and GRS80 are sufficiently close that a
single warp file serves both in practice — the ellipsoidal difference is
negligible relative to the warp residual.

Projection interchangeability is a consequence of the design philosophy: the warp
corrects for a specific ellipsoid, but the grid it corrects is defined
independently of any ellipsoid. This separation is not an accident of
implementation — it is the consequence of insisting that topology, hierarchy, and
area correction be treated as distinct concerns from the outset.

---

## 12. Comparison with Prior Art

The differences between the established systems — H3, S2, HEALPix — are well
documented in the literature; this section makes only the comparisons that bear
on Hex9's claims.

| Property | H3 | S2 | HEALPix | A5 | Hex9 |
|---|---|---|---|---|---|
| Base polyhedron | Icosahedron | Cube | (sphere-native) | Dodecahedron | Octahedron |
| Cell shape | Hex (+ 12 pentagons) | Quadrilateral | Mixed quad | Pentagon | Hex (+ 12 pentagons) |
| Aperture | 7 | 4 | 4 | 5 then 4 | 9 (shifted) |
| Equal area | No | No | Yes (strict) | Yes | Quasi (p99 < 0.005%) |
| Strict ancestry at all levels | No | Yes | Yes | — | Half-hex: yes; hex: by convention |
| Distance isotropy | Yes | No ($\sqrt{2}$) | Yes | No (elongated) | Yes |
| Dual DGGS/CRS | No | No | No | No | Yes (quasi-CRS) |
| Reference body | Sphere | Sphere | Sphere | Ellipsoid | Any ellipsoid (per-body warp; WGS84 + sphere trained) |

**Exception cells.** Euler's theorem requires exactly 12 topological pentagons —
cells with five neighbours — in any spherical tiling by hexagons and pentagons,
at every refinement level, independent of resolution. No hexagonal DGGS escapes
this; the systems differ in where the obligation lands. H3 places its 12
pentagons at the icosahedral vertices, where they are first-class exception
cells: five children instead of seven, a distinct geometry, and an
`is_pentagon()` guard that every correct H3 implementation must carry. The
pentagons are the topological price of odd vertex valence — a 5-valent vertex
cannot sustain the face two-colouring on which a consistent hexagonal subdivision
depends (§4). Hex9's 12 topological pentagons sit at the six octahedral vertices
(two per vertex), where the 4-valent geometry absorbs them: they carry ordinary
addresses, are constructed by the same d_cell join rule as every other cell
(§10f), and require no API guard. In Hex9's native planar domain the six vertices
lie on the boundary of the coordinate space, so the defect cells straddle the
edge of the map rather than appearing as interior anomalies.

**Ancestry.** The strictly nested unit in Hex9 is the half-hexagon (d_cell), not
the hexagon. The d_cell hierarchy is a single-parent tree: a d_cell address
composes left-to-right, prefix-truncation always yields its unique ancestor, and
no edge of the finest-level half-hexagon tiling crosses a coarser half-hexagon
boundary. Hexagons are assembled from two half-hexagons, and 3 of the 9 child
hexagons per level (digits 6–8) straddle a d_cell boundary and have two valid
parents; the canonical mode-0 convention selects one, and the exact ancestor is
recovered from the address tail (§10b). Under that convention the canonical
parent function is well-defined at every level and multi-resolution roll-up is
exact — but, unlike the unconditional quad hierarchies of S2 and HEALPix,
hexagon roll-up requires deriving the canonical ancestor (via the tail) before
truncating, and the split-cell ambiguity can nest: a run of split digits stays
ambiguous until the tail resolves it. H3's aperture-7 subdivision is weaker still
— it does not nest children inside parents at all: a child hexagon may overlap
two coarser cells, parent assignment is approximate, and a shared parent does not
imply a shared grandparent. In short, Hex9's half-hexagon hierarchy is exactly
nested; its hexagon hierarchy is nested by convention.

**Area.** HEALPix is strictly equal-area but pays in cell shape: its cells are
quadrilaterals of visibly varying geometry. H3 and S2 are not equal-area; cell
areas vary by tens of percent across the globe, which biases any analysis that
compares counts or densities between regions. Hex9 is quasi-authalic by
construction of the warp (§11b): the residual is bounded, characterised, and
confined to the neighbourhoods of the six vertex singularities.

**Shape and area, independently surveyed.** An independent survey of cell
geometry — enclosing-cone aspect ratio and WGS84 geodesic cell area over N = 5000
uniformly-sampled cells per system — places the four systems into two area tiers
separated by roughly 100×. A5 and Hex9 form an equal-area tier (area coefficient
of variation 0.01% and 0.10% respectively); H3 and S2 do not (12.5% and 14.5%),
and the gap is a step, not a gradient. The survey reproduces Hex9's warp
characterisation independently — worst cell ≈ +4.8%, mean absolute deviation
0.001%, all residual pooled at the twelve octahedral-vertex defects — on a
third-party tool and against the WGS84 datum, corroborating §11b. On cell shape
the order is H3 (aspect ratio 1.06, roundest) < S2 (1.24) < Hex9 (1.37) < A5
(2.14): Hex9 reaches the equal-area tier while staying markedly rounder than the
only other equal-area system in the comparison. Both metrics are
resolution-invariant (reproduced to four decimals five levels coarser), and the
dispersion ranking A5 < Hex9 < H3 < S2 is the same under every statistic tested
(CV, p90, p95, max/min). Hex9 is therefore the better combined shape-and-area
cell among the equal-area systems; A5 distributes a hair of residual more evenly
(max/min 1.009 vs Hex9's 1.059), but does so with substantially less round cells.

**Aperture.** Hex9's aperture is 9, but the subdivision is not the centred
aperture-9 scheme of the DGGS literature. In a centred scheme one child sits
concentrically at the parent's centroid; in Hex9 the parent centroid falls on the
shared long edge of its two half-hexagons, so there is no central child and the 9
children form the shifted half-hexagon arrangement of §8. The shift is not a
cost. A centred aperture-9 grid must descend one triangulation level below its
own cells to describe their boundaries; Hex9's half-hexagon is its own primitive
at every level, and point membership reduces to a fixed set of linear
inequalities (§10a). Comparisons of aperture across systems should note this
distinction.

We coin "shifted aperture 9" to be explicit about the offset. Each parent has
exactly nine children — aperture 9 in the strict sense — but they carry a
translational shift rather than sitting centred on the parent. This parallels
H3, whose aperture-7 children carry a rotational offset between successive
resolutions yet are still called aperture 7; we prefer to surface the offset in
the name rather than leave it implicit.

**Reference body.** H3, S2, and HEALPix are defined on the sphere; ellipsoidal
use requires an auxiliary-latitude transformation with its own distortion budget.
Hex9 is defined against any two-axis reference ellipsoid directly: the warp is
computed from geodesic areas on that body and recomputed for any other (§11d).
Trained warps currently exist for WGS84 and the sphere; the C implementation
(libhex9) is tuned for WGS84. WGS84 is the default, not a constraint.

**Distance isotropy.** A hexagonal tiling gives every cell six neighbours at a
single edge-to-edge distance, so the grid privileges no direction and
nearest-neighbour and traversal distances are uniform. A quadrilateral grid does
not: S2's edge neighbours and corner neighbours differ by a factor of $\sqrt{2}$, which
biases distance and adjacency computations along the diagonal. Hex9 and H3 are
isotropic in this sense; A5's pentagonal cells are markedly elongated
(aspect ≈ 2.14), so their neighbour distances are not uniform.

**Dual DGGS/CRS role.** This is the row no other system marks "yes", and it is
the paper's central claim (§9, §10e, Appendix A): a single Hex9 address is a
DGGS zone identifier when truncated at a level (OGC Topic 21), and in the limit a
function recovers from it a point on the ellipsoid to arbitrary precision. We
state this as the weaker, defensible claim — the addressing is *quasi-continuous*
(§10e), not a strict ISO 19111 CRS: the recovery map is discrete-valued and jumps
across a measure-zero set of seams, where continuity in the ISO sense fails. The
distinction from the others is nonetheless real: H3, S2, and HEALPix are indexing
schemes layered over a pre-existing CRS; they identify cells, but their
identifiers do not double as position-recovery coordinates.

---

## 13. Implementation

Hex9 is implemented in **libhex9**, a C/C++ core library with several front-ends:
a PostGIS extension, a Python accelerator binding, and a set of Python
command-line tools. The earlier Python package (`hhg9`) remains as a readable
reference implementation and produced the warp characterisation of §11b.

### 13a. The partition cycle

The algorithmic heart of encoding is a repeated three-step cycle on octant
coordinates: classify the point into one of the 9 child regions (the linear
inequality evaluation of §10a); subtract the child's origin; scale by 3. Each
iteration emits one address digit and returns the coordinates to the original
numerical scale, so there is no accumulating floating-point drift; the cycle is
O(L) per address, numerically stable, and exact in integer arithmetic. It is also
the CRS limit of §10e made operational: the offset removal composed with the ×3
scaling is the inverse of a contraction mapping, and iterating it indefinitely
converges to the encoded point.

Decoding runs the cycle in reverse, with one structural caveat: the split cells
of §10b make the x_list right-to-left, so the decoder first reads the tail to fix
the terminal mode and then resolves canonical parentage upward in a single pass.

### 13b. Address encoding

A Hex9 address packs into a single standard 128-bit UUID as 32 nibbles: the first
30 carry the hierarchy path (layers 0–29, one base-9 digit per level), and the
final two carry the tail (§10b). Those two tail nibbles take one of two forms. The
**reversible** tail packs the terminal region (the d_cell id h), p_mo, p_c2 and
r_mo — everything needed for an exact round trip to a representative point. The
**bin** (key) tail packs only p_c2 and r_mo, with 0xF in the low nibble as a
sentinel — enough to bin and join uniquely, but not to recover the exact terminal
cell. A single 128-bit value therefore serves either as an exact reversible
coordinate or as a pure spatial key, with no companion field.

The format drops directly into any database, PostGIS column, or API that accepts a
standard UUID; hierarchy traversal is nibble truncation and spatial binning is
prefix comparison, with no decoding step. The hierarchy reaches layer 29 — far
below any geodetic resolution (a layer-29 cell is of nanometre scale) — so 128
bits is effectively unlimited precision in practice.

### 13c. The core library and tooling

The `libhex9` C/C++ core implements the full pipeline: the AK base projection
with its analytic Jacobian, the authalic warp (Clough–Tocher interpolant,
Newton–Raphson inverse), encoding and decoding, k-ring neighbour computation, and
cell-polygon generation, with the trained WGS84 warp embedded in the library. A
Python accelerator (`hex9_ext`) exposes the core to Python at native speed with
OpenMP batch encoding, and two Python command-line tools wrap the common
workflows: `h9_csv` appends a reversible `h9_uuid` column — and, optionally,
canonical bin or label columns — to a latitude/longitude CSV, streaming so it
handles large files; `h9_choropleth` turns a point CSV into an adaptive Hex9
choropleth as a GeoJSON feature collection, with no PostGIS or QGIS in the loop.
The `hhg9` Python package remains the readable reference: it produced the warp
characterisation of §11b and the figures in this paper, with all grid operations
vectorised over NumPy arrays.

### 13d. PostGIS extension

`libhex9` ships a PostGIS extension (`postgis_hex9`) that exposes the full surface
in SQL: `h9_encode`/`h9_decode`, `h9_bin` and `h9_label`, the cell- and
grid-polygon builders (`h9_cell`, `h9_grid`), neighbour queries (`h9_kring`,
`h9_kdisk`, `h9_neighbors`), `h9_common_ancestor` for exact roll-up, and
`h9_adaptive` for population-driven mixed-resolution layers — so a Hex9 address is
a first-class spatial key inside the database. It is built on the C/C++ core
rather than Python, since most managed PostGIS environments treat Python as an
untrusted language. (Distribution is handled directly through `libhex9`: PostGIS
is winding down its third-party extension distribution programme.)

### 13e. On not registering a CRS

A PROJ/GDAL CRS plugin was prototyped (`h9_boct`, a C++ port of the hierarchy
walk and warp inversion), and the round trip matches the reference to within tens
of nanometres. We do not, however, pursue registering Hex9 as a PROJ/GDAL
coordinate reference system. The reason is the one developed in §10e: Hex9 is
quasi-continuous, not a strict ISO 19111 CRS, so presenting it through the CRS
machinery would overstate what it is. Applications instead use the core library
and its bindings directly, treating Hex9 as the discrete addressing system it is.

### 13f. Inverse projection by guarded Gauss–Newton

The base-projection inverse — ellipsoid point to octant face coordinate — is the
one numerically non-trivial step. `libhex9` provides a fast analytic-Jacobian
solver for it, root-identical to the exact beam search in the smooth interior, at
one analytic forward-and-Jacobian evaluation per iteration instead of several
finite-difference evaluations each running an ECEF→geodetic (Bowring) step. The
unknowns are the two face coordinates $(f_x, f_y)$; the residual
$r = P(f_x, f_y) - E$ is the 3-vector, in ECEF metres, between the AK forward map
$P$ and the target point $E$ on the WGS84 ellipsoid. Because the forward map lands
exactly on the ellipsoid, the least-squares root coincides with the true geodetic
match — there is no auxiliary-latitude (Bowring) step inside the loop. Each
iteration solves the $2\times2$ normal equations $(J^{\mathsf{T}}J)\,\delta =
-J^{\mathsf{T}} r$ from the $3\times2$ Jacobian $J = \partial P/\partial(f_x,f_y)$,
which is fully analytic: the chain of $\partial\mathrm{AK}/\partial(u,v,w)$
(tangent-power derivatives and the ellipsoid-normalising quotient) composed with
the affine $\partial(u,v,w)/\partial(f_x,f_y)$, and checked against central
differences. A backtracking line search (halving the step until $\lVert r
\rVert^2$ decreases) makes it a strictly damped, guarded Gauss–Newton iteration; a
seam/vertex guard and a failure veto route the thin skin around seams and vertices
to the exact beam search, which remains the reference. The implementation is
`libhex9/tools/newton_invert_aj.h`.

---

## 14. Applications

### 14a. Binning and density estimation

The primary use case is the one that motivated the quasi-authalic default:
aggregating point observations into cells whose areas are comparable anywhere on
Earth. Encoding is the O(L) partition cycle; binning is prefix truncation of the
UUID — with the canonical-ancestor derivation of §10b applied first wherever
non-canonical addresses may be present. Because cell areas are uniform to within
the warp residual, raw counts are densities up to a single global constant: no
per-cell area correction, no latitude adjustment. Grids that are not equal-area
(H3, S2) require explicit area normalisation for the same task, and the
normalisation factor varies by location.

*[Figure 15: population density heatmap binned to Hex9 cells, one country or
global. (F15)]*

Cells need not share a level. Because parentage is exact (§12), a single layer
can mix resolutions — coarse cells where data is sparse, fine cells where it is
dense — without T-junctions, seams, or interpolation between levels, and the
mixed layer still rolls up losslessly. Refinement can therefore be driven by the
data itself: subdivide a cell only while the population it contains stays within a
target band, stopping when the count falls low enough or a depth limit is
reached. Figure 23 shows this for Thimphu, Bhutan, where a population layer drives
adaptive refinement across L5 to L12 — coarse L5 cells over empty terrain,
refining to L12 along the inhabited valley floors. Each cell is shaded by
population density (count per authalic cell area), so the fill is directly
comparable across levels without area correction — a single coherent grid whose
cells span seven levels at once.

![Thimphu, Bhutan — population-driven adaptive refinement spanning L5–L12 in one mixed-resolution Hex9 layer; fill is population density per authalic cell area, directly comparable across levels without area correction.^[Rendered in QGIS via `h9_adaptive()` over the Bhutan population layer [@hdx_bhutan_pop].]](paper_figures/thimpu_chloropleth_web.jpg){width=80%}

### 14b. Spatial joins and multi-resolution analysis

Two datasets encoded to Hex9 addresses join on shared prefixes: equality of
level-K prefixes is co-location at level K, computed without geometry. Roll-up
from L to any coarser K is exact by the strict ancestry property (§12):
aggregates computed at fine resolution recombine losslessly at coarse resolution,
which is not guaranteed in systems where parentage is approximate. The same
property makes Hex9 addresses suitable as the spatial component of composite
database keys, with standard B-tree indexes serving range and containment queries
that would otherwise need a spatial index.

### 14c. Display and rendering

The canonical cell polygon is the regular hexagon in b_oct; everything else is
reprojection (§11c). For display in a conventional CRS, cell edges are densified
and reprojected — the curvature that appears belongs to the target projection.
Two practical notes. First, hexes straddling the antimeridian render incorrectly
in naive geographic plots (drawn across the full map width); the defect is the
renderer's, not the data's, and a bounds filter suffices. Second, at the six
octahedral vertices the 4-valent cells are correct as constructed (§10f) and need
no special-case rendering.

```{=latex}
\begin{figure}[ht]
\centering
\includegraphics[width=0.9\linewidth]{ex0097_butterfly0500_2_web.jpg}\\[2pt]
{\footnotesize (a) b\_oct (native): every cell a congruent regular hexagon.}\\[8pt]
\begin{minipage}[t]{0.49\linewidth}\centering
\includegraphics[width=\linewidth]{bm_mollweide_web.jpg}\\[2pt]
{\footnotesize (b) Mollweide (equal-area): areas preserved, shapes shear.}
\end{minipage}\hfill
\begin{minipage}[t]{0.49\linewidth}\centering
\includegraphics[width=\linewidth]{bm_mercator_web.jpg}\\[2pt]
{\footnotesize (c) Mercator (conformal): cells balloon toward the poles.}
\end{minipage}
\caption{The same Hex9 layer in three projections --- the distortion belongs to
the map, not the cells (\S14c). In the native b\_oct space (a) every cell is a
congruent regular hexagon; an equal-area projection (b) keeps cell areas
comparable while shearing their shapes; a conformal projection (c) preserves
local angles but inflates the polar cells without bound (the largest are the
octahedral-vertex cells). Sources: b\_oct butterfly from
\texttt{examples/ex0097\_smp\_grid.py}; Mollweide and Mercator rendered in QGIS.
Backdrop: NASA Blue Marble.}
\end{figure}
```

### 14d. Graticule alignment

The octant face spans exactly 90° in latitude and longitude, and 360 carries two
factors of 3, so trisection of the degree system is at least as natural as
bisection — and more useful: bisecting 90° gives 45°, 22.5°, 11.25° (non-standard
graticule values), while trisecting gives 30° and 10°, both standard. At 10°
spacing the graticule fits the octant face in 9 clean intervals; L1 hexes span
30°, L2 hexes span 10°. In arc-minutes the octant (5,400′) trisects cleanly four
times (1800′ → 600′ → 200′ → 66.67′) before breaking — deeper than bisection
achieves. A 10° graticule is therefore the natural companion grid for Hex9
visualisations: the alignment is a consequence of the shared factor of 9 between
base-360 and the ternary hierarchy, not a design choice in the projection.

### 14e. Other reference bodies

The addressing scheme, hierarchy, and interpolation machinery are
ellipsoid-agnostic; only the Sinkhorn warp is computed against a specific body,
at a one-time cost of roughly a day of unattended compute (§11d). WGS84 and GRS80
share a warp file in practice — their difference is negligible against the warp
residual. Legacy ellipsoids (Bessel 1841, Clarke 1866) and planetary bodies with
IAU reference ellipsoids (the Moon, Mars) require only their own warp files and
their own prime-meridian conventions; the mathematical object is unchanged.
Highly irregular bodies (Phobos, small asteroids) are out of scope, as for every
current DGGS.

---

## 15. Conclusion and Future Work

The argument of this paper runs in one direction: from coherence requirements to
structure, and from structure to location. Orientability selects the triangle;
flat mode transport and vertex closure select even valence; regularity selects
the octahedron; refinement commutativity selects odd apertures and the minimal
aperture 9; the half-hexagon tiling enumeration — machine-verified end to end —
leaves exactly one orientation per chirality. Setting the geometric realisation
aside, precisely two genuine free choices remain in the entire construction:
which chirality, and which bijection assigns the nine digits. Both are
conventions in the same sense as the prime meridian: the grid does not depend on
them, and a variant making the other choice is the same mathematical object.

Everything the system offers follows from that determinacy. Because no residual
design freedom exists, cell identity is well-defined without reference to any
prior coordinate system, and the address can serve as both a DGGS cell
identifier and, in the limit, a position-recovery coordinate — quasi-continuous
rather than a strict ISO 19111 CRS (§10e) — the dual role developed in §9 and
§10e and formalised in Appendix A. The geometric realisation is a separable concern: the AK base
projection and the optimal-transport warp place the abstract structure on WGS84
with quasi-uniform areas, and either layer can be substituted without disturbing
the grid.

Future work falls into four groups:

**Warp refinement.** A strictly authalic variant is achievable in principle
(det J = 1 everywhere) at the cost of unconstrained shear; the present
quasi-authalic compromise is deliberate, but the trade-off curve — area residual
against shape regularity, via the shape-dampening parameter — deserves systematic
characterisation. The angular radius of the elevated-deviation zone around the
six vertex singularities is a one-time, layer-independent measurement not yet
made. Deeper-layer warps (L6+ derived rather than interpolated) would reduce the
inter-mode residual further.

**Implementation and maintenance.** The core engineering work — the `libhex9`
C/C++ library, its PostGIS extension, the Python accelerator, and the CLI tools
(§13) — is in place; the forward effort is supporting, hardening, and improving
it rather than building new infrastructure. Warp files for legacy ellipsoids
(Bessel 1841, Clarke 1866) and at least one planetary body would demonstrate the
generality claim concretely.

**Verification.** The R-rotation form of constraint B (§8) remains available as
an independent cross-check of the tiling enumeration. The C1 continuity of the
warp at octant boundaries is asserted from the construction and should be
characterised explicitly.

**Standardisation and community.** A formal channels considered earlier is no
longer being pursued: registration as a PROJ/GDAL CRS; because Hex9 is 
better identified as quasi-continuous rather than a strict ISO 19111 CRS 
(§10e, §13e). Discussions with PostGIS likewise suggested that it is better to 
persue PostGIS integration as a separate 3rd party rather than look to integrate
into the core; and this thought has been readily accepted.
The path remains open engagement with the discrete-global-grid community around
the existing `libhex9` implementation, rather than a formal standards submission.
The OGC/ISO terminology mapping of Appendix B is offered in that spirit — as a
bridge for readers coming from those standards, not a claim of conformance.

---

## Appendix A — The CRS Limit

**Claim.** Every infinite Hex9 address sequence σ = (a₀, a₁, a₂, …) uniquely
determines a point on the WGS84 ellipsoid.

**Setup.** Let Ω ⊂ b_oct be the closed octant face: a compact convex triangle of
diameter d₀. One step of the Hex9 decode maps Ω into one of its 9 child cells by

$$p \;\mapsto\; \tfrac{1}{3}\,p + \mathrm{offset}(C_i),$$

a Lipschitz contraction with ratio 1/3. For an address prefix (a₀, …, a_L), let
S_L ⊂ Ω be the corresponding cell — compact and convex. Then

$$S_0 \supset S_1 \supset S_2 \supset \cdots, \qquad
  \mathrm{diam}(S_L) \;\leq\; d_0 \left(\tfrac{1}{3}\right)^{L} \;\to\; 0.$$

**Completeness.** Each octant face is a closed bounded subset of ℝ² and hence a
complete metric space; the S_L are closed subsets of it.

**Cantor's intersection theorem** (nested compact sets with diameters tending to
zero in a complete metric space) gives

$$\bigcap_{L=0}^{\infty} S_L \;=\; \{p^{*}\},$$

a single point of b_oct. Equivalently, via Cauchy sequences: choose any p_L ∈
S_L; for L, M > N both p_L and p_M lie in S_N, so d(p_L, p_M) ≤ d₀ (1/3)^N → 0;
the sequence is Cauchy, its limit p* lies in every closed S_N, and uniqueness
follows from diam(S_L) → 0.

**Lifting to the ellipsoid.** The composition

$$\text{b\_oct} \xrightarrow{\;\text{AuthalicWarp}^{-1}\;} \text{b\_raw}
  \xrightarrow{\;\text{AK}\;} \text{c\_ell}$$

is continuous: the warp is C1 by the Clough-Tocher construction, and the AK
projection is smooth on the octant interior and at its vertices (below).
Continuous maps preserve limits, so p* maps to a unique point on the ellipsoid.

**Continuity at the octahedral vertices.** The octahedral vertices are the points
satisfying |x| + |y| + |z| = 1 and x² + y² + z² = 1 simultaneously — exactly the
points with two zero coordinates. There the octahedron and the sphere coincide,
the AK map reduces to the identity followed by axis scaling to the ellipsoid, and
its apparent indeterminacy is a removable singularity; the Jacobian at these
points is the well-defined limit of the linearisation. No actual singularity
exists in the map.

**Almost-everywhere bijectivity.** The map σ ↦ p* is injective except on the
d_cell seam boundaries — the split-cell case, where 3 of the 9 children per level
admit two valid parent sequences (§10b). The seam boundaries form a set of measure
zero on the ellipsoid; away from them, infinite addresses and ellipsoidal points
are in bijection. The canonical mode-0 convention (§10b) selects one address for
seam points, exactly as decimal notation selects 0.5 over 0.4999….

**The dual claim** follows: σ truncated to L digits names a compact cell of area
510,065,622 km² / (12 · 9^L) on WGS84 (≈ 42.5 million km² at L0, 719.8 km² at L5)
— a DGGS cell in the sense of OGC Topic 21; σ in the limit names a point, a
position recoverable by a function to arbitrary precision. This is the
quasi-continuous coordinate role of §10e, not a continuous ISO 19111 coordinate
system: the recovery map is well-defined and continuous in the decode direction
(infinite address → point), but the address alphabet is discrete and the encode
direction jumps across the measure-zero seam set just established. The
contraction ratio 1/3 is geometrically determined by the ternary subdivision, so
the convergence rate is explicit: one address digit buys a factor-3 reduction in
positional uncertainty.

---

## Appendix B — OGC / ISO Terminology Mapping

This appendix maps Hex9 vocabulary to the two standards the dual claim of §9
engages: OGC Abstract Specification Topic 21 (20-040r3, 2021) [@ogc_topic21] for
the DGGS role, and ISO 19111:2019 [@iso19111] for the CRS role. The full taxonomy
is maintained in the implementation glossary; the tables here cover the terms a
standards reviewer needs.

Status codes: **C** — confident (the standard defines the term with a direct
correspondence); **P** — provisional (a reasonable correspondence whose exact
wording is to be confirmed against the spec text); **—** — Hex9-specific, with no
standard equivalent.

A vocabulary note on Topic 21: the 2021 edition (Part 1) renamed the 2017
edition's *cell* to *zone*, and a cell identifier to a *zone identifier* (ZID).
This paper keeps "cell" in prose for readability; the OGC-facing term is "zone".

### Hex9 → OGC Topic 21 (DGGS)

| Hex9 term | OGC Topic 21 term | Status |
|---|---|---|
| Hex9 (the system) | Discrete Global Grid System (DGGS) | C |
| `x_grid` at level L | Discrete Global Grid (DGG) — one tessellation / level | C |
| `x_cell` | zone (2021); cell (2017) | C |
| `x_adr` / `x_list` | zone identifier (ZID) | C |
| octant (8 seed faces) | base polyhedron face / level-0 partition | P |
| 12 root `x_cells` | level-0 zones | C |
| refinement level L | refinement level / resolution | C |
| aperture 9 (shifted) | aperture (refinement ratio) | C |
| `x_dig` | sub-zone index within a parent zone | P |
| parent/child `x_cell` | zone hierarchy / containment | C |
| AK+Warp realisation | DGGS Reference Frame (cell geometry on the globe) | P |
| mode (0/1 parity) | — (internal face orientation) | — |
| `c2` edge label | sub-face identifier | P |
| `d_cell` (half-hexagon) | — (sub-cell primitive) | — |

### Hex9 → ISO 19111:2019 (referencing by coordinates)

| Hex9 term | ISO 19111 term | Status |
|---|---|---|
| WGS84 reference ellipsoid | ellipsoid / geodetic reference frame (datum) | C |
| Prime Meridian anchoring (Axiom 7) | prime meridian | C |
| `x_adr` carried to the limit (§10e, App. A) | position recoverable by a function (quasi-continuous; not a strict ISO coordinate) | P |
| Hex9 as a locating system | coordinate reference system (CRS) — *quasi*, see §10e | P |
| b_oct | a coordinate system realised by AK+Warp | P |
| encode / decode (point ↔ address) | coordinate operation / conversion | P |
| octant 2D plane coordinates | coordinate system (CS) axes | P |

The dual role of §9, §10e, and Appendix A is the bridge between these two tables:
one Hex9 address is a Topic 21 zone identifier when truncated at a level and, in
the limit, a position recoverable by a function to arbitrary precision. We make
the weaker, defensible claim — *quasi-continuous*, by analogy with
quasi-authalic — rather than asserting a strict ISO 19111 CRS: the address space
is discrete and the point→address map is discontinuous on a measure-zero seam set
(§10e), so continuity in the ISO sense does not hold. Final verification of the
exact spec wording against the current editions of both standards is a
pre-submission task (§15).

---

## Figures

Full figure specifications, production notes, and final captions are maintained
in `paper_figures.md`; images live in `paper_figures/`. The index below lists the
figures and their section anchors.

| # | Section | Working title |
|---|---|---|
| F1 | §8 / §2 | The 49 tilings, Hex9 pair highlighted |
| F2 | §8 | The two equilateral tilings (L/R chiral pair) |
| F3 | §10.0 | Anatomy of one parent triangle: t / c2 / d / x |
| F4 | §10a | x-layer: x_dig 0–8 children, high-trit colouring + shared c-regions |
| F5 | §10a | Classifier (c-layer): 96-slot c_grid, three band families → c_dig |
| F6 | §10b | London / Prime Meridian split-cell example |
| F7 | §10d / §10f | Sibling adjacency: internal vs cross-parent edges |
| F11 | §11 | Pipeline diagram b_raw → warp → b_oct → address |
| F12 | §11b | Area-deviation globe, ±1% capped (magnitude view) |
| F12b | §11b | Area-deviation Mollweide, clipped scale (pattern view: seam skeleton) |
| F12c | §11b | North-pole L5 crop — where "quasi" lives |
| F13 | §11c | Tissot indicatrices on the b_oct butterfly net (warp applied) |
| F15 | §14a | Population heatmap binned to Hex9 |
| F16 | §14c | Same layer in b_oct / Mollweide / Mercator |
| F20 | §2 / §7 / §8 | Wallpaper group of the d_cell tiling (p31m) |
| F21 | §9 / §10 (graphical abstract) | Seed solid (24 d_cell facets) + 12 root x_cells |
| F23 | §14a | Thimphu adaptive refinement (L5–L12), population density choropleth |

---

## Acknowledgements {.unnumbered}

The octahedral base projection (§11a) is due to Anders Kaseorg
[@kaseorg_octahedral]. Anthropic's Claude was used as a drafting and editing
assistant; the author alone is responsible for all claims, derivations, and
decisions in this work.

## References {.unnumbered}

::: {#refs}
:::
