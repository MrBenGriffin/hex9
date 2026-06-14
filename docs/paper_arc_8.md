## 8. Orientation Selection

  The half-hexagon established in §7 is the natural octant carrier — the region of
  the dual hexagonal lattice bounded by the three seed edges of a single octant face.
  Its internal structure admits a further decomposition: the half-hexagon divides into
  three equilateral sub-regions, each corresponding to one third of the octant face.

  Each sub-region can be independently oriented in two ways: the hexagonal cells
  within it may be arranged in one of two configurations related by reflection. With
  three sub-regions and two choices each, there are 2³ = 8 candidate orientations for
  the half-hexagon as a whole.

  The mode transport constraint — specifically, the vertex closure condition
  established in §4 — acts on the internal boundaries between sub-regions. At every
  vertex where two or more sub-regions meet, the mode transport around that vertex
  must return to identity: trivial holonomy. This is the same condition that selects for
  even valence globally; applied to the internal half-hexagon boundaries, it filters
  the 8 candidates.

  Exactly 2 of the 8 combinations satisfy the internal closure condition at every
  boundary vertex. The two survivors are related by a global reflection — a chiral
  pair, geometrically distinct but structurally equivalent up to handedness.

  [Figure 2: the surviving chiral pair — the two orientations of the 9-cell
  equilateral that satisfy internal closure, three half-hexagons each, coloured
  by orientation class; the members are mirror images]

  This count is not an assertion. The full enumeration is machine-verified by
  `experimental/halfhex_verify.py` (checks V0–V6): of the 49 distinct hextile
  solutions (24 chiral pairs + 1 self-mirror), the long-edge constraint (A)
  admits 18, the three-equilateral structural constraint (B) admits 8, and their
  intersection A ∩ B is exactly the recorded Hex9 chiral pair. The constructive
  2³ → 2 argument above and this enumeration agree.

  [Figure 1: the 49 hextile solutions, chiral pairs grouped and the self-mirror
  marked, with the Hex9 pair highlighted; caption carries the counts
  49 = 24 pairs + 1 self-mirror, A → 18, B → 8, A ∩ B → 2]

  Axiom 6 resolves the remaining freedom: a single consistent chirality choice,
  induced by the embedding into the reference ellipsoid frame, selects one of the two.
  The orientation of the half-hexagon — and with it the entire hexagonal lattice
  across all 8 octants — is determined: no further choice remains.

  This closes the enumeration. Each cell in the hierarchy is fixed by its octant, its
  refinement level, and its index within that octant at that level. No two cells share
  the same address; no address refers to more than one cell. The hexagonal lattice is
  globally indexed with no residual ambiguity. What follows from this is the subject of §9.
