## 6. Refinement Commutativity

  The octahedral seed established in §5 defines a base simplicial complex on 
  which a hierarchical refinement operator acts. Each triangular face is
  subdivided into k² child triangles by linear refinement at scale factor k.
  The refinement operator must preserve simplicial type, adjacency relations,
  and the flatness of mode transport (§3), and must commute with the global
  indexing of the mesh.

  Refinement is chosen to commute with mode transport because mode is the 
  global coherence field of the system. Without commutativity, refinement 
  would not preserve identity under scale, and hierarchical addressing would 
  cease to be stable under composition. This choice is not geometrically 
  mandatory, but it is required for refinement to function as a 
  consistent coordinate extension rather than a sequence of 
  unrelated discretisations.

  These requirements constrain admissible values of k.
  First, k must be a positive integer to ensure that refinement is a well-defined 
  subdivision of the simplicial structure into congruent refinement classes.

  Second, refinements that preserve mode transport consistency are those in which
  the mode of every child triangle agrees with the mode inherited from its parent
  at every induced edge (a homomorphism of the ℤ₂ transport structure, in algebraic terms).

  For even k, the refinement fails this test. Consider k=2: a mode-0 parent
  produces 4 children — 3 corner children of mode 0 and 1 inverted central child
  of mode 1. At the boundary between two adjacent parents (which have opposite modes),
  this places mode-0 children of the mode-0 parent adjacent to mode-0 children of
  the mode-1 parent — directly contradicting the required mode flip at that boundary.
  The transport operator disagrees with itself across the inter-parent edge.
  This breaks the flatness of mode transport across the refinement,
  violating commutativity with global indexing.

  For odd k, the refinement preserves parity alignment between parent and child 
  simplices, ensuring that mode transport is consistently inherited at all scales. 
  Admissible values of k are therefore restricted to odd integers within 
  this refinement class: **k ∈ {1, 3, 5, 7, …}**

  When restricted to refinements that preserve mode transport consistency and
  commute with global indexing, composing two valid refinements always yields
  a valid refinement (the admissible class forms a semigroup, in algebraic terms).
  Within this class, odd integers define valid scales, and the minimal non-trivial
  scale is k = 3.

  We therefore select k = 3 as the base refinement operator. Higher-scale
  refinements are obtained by composing this operator, yielding a hierarchy indexed
  by powers of 3.

  In the k=3 case, refinement introduces a systematic lateral displacement of 
  child simplex centroids relative to the parent geometry. 
  This displacement is not an artefact of embedding, 
  but the geometric manifestation of parity-preserving refinement under ℤ₂ 
  transport. In the dual structure, this offset becomes visible as the 
  characteristic shift in the induced hexagonal lattice, giving rise to 
  the **shifted-aperture-9 hierarchy**.
