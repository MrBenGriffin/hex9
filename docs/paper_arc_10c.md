## 10c. Identity as Label

  The full Hex9 address space is bipartite: at every level, cells carry a mode (0 or 1)
  inherited from the refinement structure established in §2. For internal computation
  — transport, refinement, adjacency — mode is a meaningful structural property.
  For labelling purposes, exposing it is unnecessary.

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

  This collapse is not a loss of information. The bipartite structure remains present
  in the refinement geometry; the label simply presents a face of it that is uniform
  and compact. A label identifies a specific geographic region at a specific resolution.
  Nothing more is needed; nothing is omitted.
