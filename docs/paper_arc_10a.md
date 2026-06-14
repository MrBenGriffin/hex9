## 10a. Identity as Locator

  A Hex9 address is a pair: an octant index (one of 8) and a refinement path — a
  sequence of x_dig values recording which child x_cell was entered at each level
  of the hierarchy. An x_adr of depth L identifies exactly one x_cell at level L
  on the reference ellipsoid.

  The reversibility of this mapping rests on the construction. At each level, a
  parent x_cell contains exactly 9 child x_cells, whose arrangement is determined
  by the t_cell → d_cell → x_cell sequence established in §§6–8. Each x_dig (0–8)
  selects one of those 9 children unambiguously. The refinement tree is a
  strict single-parent structure for t_cells and d_cells; x_cells inherit this
  property except at the 3 split x_cells per level, whose parentage is resolved
  by the tail of the x_adr.

  [Figure: 9-child x_cell subdivision with x_dig labels]

  Once the projection from the reference ellipsoid to the octant 2D plane is
  complete, the encode direction (point → x_adr) operates via a sequence of
  linear inequality evaluations. Three families of parallel lines partition the
  octant plane into the triangular grid:

    — horizontal bands:       y  compared against fixed thresholds
    — positive-slope bands:   y − √3·x  compared against fixed thresholds
    — negative-slope bands:   y + √3·x  compared against fixed thresholds

  A point's t_cell at a given level is determined by which band it occupies on
  each family — a small fixed number of comparisons, with no geometric distance
  computation and no iterative search. At each successive refinement level the
  thresholds scale by 1/3, maintaining identical structure. The c_cell (the
  96-slot classifier combining horizontal and slope bands) maps directly to a
  t_cell, and from there to the d_cell and x_cell via the construction of §§6–8.

  [Figure: three-family line partition of the octant plane showing t_cell grid]

  The mapping operates in both directions. Given an x_adr, the corresponding
  geographic region is recovered by tracing the digit sequence from the octant
  root: each x_dig selects a child x_cell whose boundary is determined by the
  refinement geometry. Given any point on the ellipsoid, its x_adr at level L
  is recovered by projecting to the octant plane and evaluating the inequality
  sequence down to depth L.

  Both directions are exact at each finite level. Encode produces no
  approximation: every point belongs to exactly one x_cell at every level.
  Decode recovers a cell whose geographic extent is precisely determined by the
  octahedral embedding and the refinement geometry.

  This reversibility qualifies Hex9 as a locating system rather than merely an
  indexing scheme. Location is not inferred from the address — it is recovered
  from it by retracing the construction.
