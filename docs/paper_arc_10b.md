## 10b. Identity as Key

  A Hex9 address is also a spatial key — a string over a small digit alphabet that
  can be stored, compared, and sorted without reference to any geometric structure.
  This makes Hex9 addresses directly usable as database keys, hash keys, or binning
  primitives.

  The key property is that prefix order corresponds to containment. All cells whose
  address begins with a given prefix σ are contained within the cell identified by σ.
  This means spatial containment queries reduce to prefix comparisons: to find all
  level-L cells within a given level-K region (L > K), select all addresses sharing
  that region's prefix. No geometric computation is needed.

  Spatial joins between two datasets reduce similarly: two observations share a cell
  at level K if and only if their addresses share a common prefix of length K.
  Aggregation across levels is achieved by truncating addresses to the desired depth.
  These operations are efficient and require no coordinate arithmetic.

  Because the address space is defined by the refinement structure rather than by
  a numerical coordinate grid, there are no edge effects, no wrap-around anomalies,
  and no cells that straddle index boundaries. Every cell has exactly one address;
  every address identifies exactly one cell. The key space is clean.

  This last claim is grounded in the address tail. The x_list alone is not always
  sufficient: at each level, 3 of the 9 child x_cells (those with x_dig in {6,7,8})
  straddle the c2 boundary between two adjacent d_cells and have two valid parents.
  Beyond this, two cells with distinct terminal regions can produce identical digit
  sequences when those regions generate the same hex digit from different parent c2
  contexts.

  The key tail carries two fields that resolve these ambiguities. The first is p_c2
  — the parent c2 of the terminal region. A concrete instance: hex digit 6 at the
  terminal level arises from both region 9 (mode 1, parent c2 = 0) and region 6
  (mode 0, parent c2 = 2) within the same parent context. Both paths produce an
  identical digit body; without p_c2 they collide as keys. With p_c2 = 0 or p_c2 = 2
  respectively, the two addresses are distinct. The second field is r_mo — the root
  octant's net_mode. Two octants of opposite mode can produce the same root hex
  digit; without r_mo the decoder cannot recover which octant the address originates
  from, and cells in distinct geographic regions would share the same key.

  The full reversible tail additionally carries p_mo and h. The p_mo field records
  the actual parent mode of the terminal region; for split x_cells, this may differ
  from the key tail's canonical mode-0 assumption. Without p_mo, the decoder
  recovers the mode-0 parent's representative, not the exact terminal cell. The h
  field identifies the terminal d_cell (one of 12, bits 3–0); without it,
  reconstruction returns the x_cell centre. With h, the precise d_cell centroid is
  recovered. The reversible tail is fully invertible: the address body together with
  the reversible tail uniquely and exactly determines both the geographic region and
  a representative point within it.

  The tail fields in summary:

  | Field | Bits | Carries | Without it |
  |---|---|---|---|
  | p_c2 | 2 | parent c2 of the terminal region | distinct cells collide as keys (same digit body from different parent c2 contexts) |
  | r_mo | 1 | root octant net mode | octant unrecoverable from the root digit; cross-octant key collisions |
  | p_mo | 1 | actual parent mode of the terminal region (reversible tail only) | decoder returns the canonical mode-0 representative, not the exact cell |
  | h | 4 | terminal d_cell id, 0–11 (reversible tail only) | reconstruction returns the x_cell centre, not the exact d_cell centroid |

  The key tail (p_c2, r_mo) suffices for unique binning; the reversible tail (all
  four fields) gives an exact round trip to a representative point.

  One caution follows directly. Truncating an address to length K always identifies
  a valid ancestor at the corresponding level, but not necessarily the *canonical*
  one: a split cell encoded under its mode-1 parent truncates into the mode-1
  lineage. Binning by naive prefix-cutting therefore silently produces two bins for
  the same cell whenever non-canonical addresses are present. The correct operation
  derives the canonical ancestor via the tail before truncating: prefix-cutting is
  exact for resolution identification; canonical ancestry requires the tail.

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
  carry high ternary trit 2: these cells straddle the d_cell boundary, and the
  canonical mode-0 parent convention places geographically adjacent cells on
  opposite sides of that boundary into different canonical lineages, exactly as
  the construction requires.
