## 10f. Seams and Valence Defects

  The Euler characteristic of S² requires that any triangulation of the globe
  carry topological defects. In Hex9 these are absorbed at the six octahedral
  vertices, each of which is surrounded by 4 hexagonal cells rather than 6.
  Understanding why neither these defects nor octant seams require special handling
  requires tracing the constructive sequence: t_cells → d_cells → x_cells.

  At each refinement level, each triangular face (t_cell) is subdivided into 9
  child t_cells. Three adjacent t_cells — grouped by their shared long edge
  (c2 edge) — form a half-hexagon (d_cell). A d_cell carries an intrinsic mode
  (0 or 1). One mode-0 d_cell and one mode-1 d_cell, joined on their matching c2
  edge, form a hexagonal cell (x_cell). The hexagonal grid emerges entirely from
  this sequence; no independent hex tiling is assumed.

  [Figure: t_cells → d_cells → x_cells constructive sequence]

  At an octant seam, the d_cell c2 alignment ensures that d_cells on either side
  of the shared octant edge meet long-edge to long-edge. This alignment is not
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

  [Figure: octahedral vertex neighbourhood showing c2=0 d_cell arrangement]

  In both cases the construction proceeds without branching. The t_cell → d_cell
  → x_cell sequence applies uniformly across the entire globe — at seams, at
  defect vertices, and in the interior. The absence of special cases follows from
  the coherence of the construction, not from exception handling added afterward.
