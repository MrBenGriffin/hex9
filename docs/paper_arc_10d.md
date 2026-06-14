## 10d. Adjacency from Refinement Paths

  Two x_cells are adjacent if they share a boundary edge on the reference
  ellipsoid. In Hex9, adjacency is recoverable from the refinement structure
  because the x_cell geometry is fully determined by the t_cell → d_cell → x_cell
  construction.

  Within a parent x_cell, the 9 child x_cells tile the parent's region. Their
  shared edges are the c2 edges of the underlying d_cells — the same long-edge
  alignment that the orientation selection of §8 fixed globally. Adjacency is
  realised as a fixed, finite lookup: every cell has exactly three neighbours,
  one per c2 value, given by a constant table keyed by (cell, parent mode, c2).
  The digit structure mirrors this: each split digit k+6 (k ∈ {0, 1, 2}) names
  the pair of d_cells flanking interior child k, and the low ternary trit of
  every x_dig records the c2 orientation of the cell's long edge.

  [Figure: 9-child x_cell tiling showing internal adjacency edges]

  The same table classifies every cell's relationship to its parent boundary.
  Within each parent, cells fall into three classes: interior cells, whose
  neighbours all share the parent; mid-edge cells, with exactly one neighbour in
  an adjacent parent; and vertex cells, with two. For the latter classes the
  lookup flags the crossing and identifies the relevant child in the neighbouring
  parent — a computable operation on the address structure, stepping up one level
  in the refinement tree and applying the same c2 edge relationships that govern
  interior adjacency. Where the neighbouring parent lies across an octant seam,
  the octant congruence of §5 reduces the hop to an octant-index lookup composed
  with the y-reflection.

  Adjacency across octant seams follows from the d_cell c2 alignment at octant
  boundaries. The orientation selection of §8 ensures that d_cells at an octant
  edge meet long-edge to long-edge with the d_cells of the neighbouring octant.
  The x_cells that form across this boundary are assembled by the same d_cell join
  rule as everywhere else. Seam-crossing adjacency is therefore not a special case
  — it is the same c2-edge query applied at the octant boundary.

  [Figure: octant boundary with matching c2 edges across seam]

  In each case adjacency is a finite computation on the combinatorial structure of
  the refinement tree and the c2 edge table. No geometric distance query is needed,
  and no location in the grid requires a different procedure from any other.
