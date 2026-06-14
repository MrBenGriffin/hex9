## 4. Vertex Closure

  Section 3 established that mode transport must be flat — any closed loop must
  return the mode to its starting value. Vertices are where this condition is most
  constraining: at every vertex, a ring of triangular faces meets, and traversing
  that ring forms a closed loop.

  Consider a vertex where k triangular faces meet (valence k). Moving from face to
  face around the vertex traverses exactly k edges. Each edge crossing carries a mode
  flip. For the transport around this closed loop to return to identity, the total
  number of flips must be even — requiring k to be even.

  This is the **vertex closure condition**: it selects triangulations in which every
  vertex has even valence. It ensures that the face adjacency graph is bipartite — no odd
  cycles in the dual.

  Face bipartiteness alone does not require uniform valence. Many irregular
  triangulations of S² admit a consistent mode assignment while mixing different even
  valences. The vertex closure condition, taken alone, is compatible with non-uniform
  even valence, and we make no claim that such triangulations fail under refinement.

  The selection of uniform valence comes instead from the regularity requirement of
  Axiom 3: the seed complex admits no exceptional cell types and no distinguished
  vertex classes. A triangulation mixing different even valences contains several
  distinct vertex-star types, and every structure built over it — refinement rules,
  addressing, adjacency — would have to distinguish those classes explicitly. Axiom 3
  excludes this by requirement, not by theorem: every vertex star is of the same type,
  so every vertex behaves identically under refinement and addressing. Uniform even
  valence is therefore imposed as a regularity condition, not derived as a topological
  necessity.

  Given uniform even valence, the Euler characteristic of S² constrains what is
  possible. For a triangulation with V vertices, E edges, F faces:

    V − E + F = 2,   with   E = 3F/2

  For uniform valence v the relation 2E = vV gives:

    V = 12 / (6 − v)

  For v = 4 (the minimum even valence greater than 2): V = 6, F = 8, E = 12.
  For v = 6: the denominator vanishes — uniform valence 6 cannot close on the globe.
  For v ≥ 8: the formula yields a negative vertex count — impossible.

  The octahedral triangulation — V=6, E=12, F=8, uniform valence 4 — is the one
  abstract triangulation of S² satisfying all conditions. The division of labour is
  explicit: vertex closure, a derived condition, forces even valence; regularity
  (Axiom 3), a stated requirement, forces uniform valence; the Euler characteristic
  then leaves v = 4 as the only possibility.

  *Consequence: a ℤ₂ face-mode can be defined on any triangulation of S² with a
  bipartite dual graph — equivalently, on any triangulation in which every vertex
  has even valence. Vertex closure is the derived part of this argument; uniformity
  of valence is required by Axiom 3 rather than derived. Together they admit exactly
  one triangulation of S²: the octahedral one.*
