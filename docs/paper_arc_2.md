2. Mode

  In any consistent triangulation of an orientable surface, every edge is shared by
  exactly two triangles. The intrinsic orientation established in §1 ensures those
  two triangles carry opposite orientations: no two triangles of the same orientation
  can share an edge.

  The result is a consistent parity assignment over faces induced by 
  orientation; unique up to global inversion. In the familiar
  planar picture these are the *up* and *down* triangles; on the globe they generalise
  to two classes that partition every face in the triangulation. We call this
  two-colouring the **mode** of the triangular field — mode 0 for the down (negative)
  class, mode 1 for the up (positive) class. This labelling is not imposed from
  outside — it is a structural property of the triangulation itself. The only freedom
  is a global swap of the two classes, which amounts to a reflection.

  Every face carries a mode value. 
  Every interior edge is incident to exactly two faces of opposite mode. 
  The field is bipartite on its faces.

  As the ℤ₂ orientation cocycle of the simplicial surface, realised as a global
  face bipartition, **mode** is the first emergent global invariant of simplicial
  orientation — a constraint every grid built over the field must either respect
  or violate. Everything downstream either respects that cocycle, preserving mode
  consistently across refinement and adjacency, or breaks it, introducing
  inconsistency or defects. Every edge crossing is a mode-flipping step; what
  global consistency of those steps demands is the subject of §3.
