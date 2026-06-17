## 12. Comparison with Prior Art

  The differences between the established systems — H3, S2, HEALPix — are well
  documented in the literature; this section makes only the comparisons that
  bear on Hex9's claims.

  | Property | H3 | S2 | HEALPix | A5 | Hex9 |
  |---|---|---|---|---|---|
  | Base polyhedron | Icosahedron | Cube | (sphere-native) | Dodecahedron | Octahedron |
  | Cell shape | Hex (+ 12 pentagons) | Quadrilateral | Mixed quad | Pentagon | Hex (+ 12 pentagons) |
  | Aperture | 7 | 4 | 4 | 5 then 4 | 9 (shifted) |
  | Equal area | No | No | Yes (strict) | Yes | Quasi (p99 < 0.005%) |
  | Strict ancestry at all levels | No | Yes | Yes | — | Half-hex: yes; hex: by convention |
  | Distance isotropy | Yes | No (√2) | Yes | No (elongated) | Yes |
  | Dual DGGS/CRS | No | No | No | No | Yes (quasi-CRS) |
  | Reference body | Sphere | Sphere | Sphere | Ellipsoid | Any ellipsoid (per-body warp; WGS84 + sphere trained) |

  **Exception cells.** Euler's theorem requires exactly 12 topological
  pentagons — cells with five neighbours — in any spherical tiling by hexagons
  and pentagons, at every refinement level, independent of resolution. No
  hexagonal DGGS escapes this; the systems differ in where the obligation
  lands. H3 places its 12 pentagons at the icosahedral vertices, where they
  are first-class exception cells: five children instead of seven, a distinct
  geometry, and an `is_pentagon()` guard that every correct H3 implementation
  must carry. The pentagons are the topological price of odd vertex valence —
  a 5-valent vertex cannot sustain the face two-colouring on which a
  consistent hexagonal subdivision depends (§4). Hex9's 12 topological
  pentagons sit at the six octahedral vertices (two per vertex), where the
  4-valent geometry absorbs them: they carry ordinary addresses, are
  constructed by the same d_cell join rule as every other cell (§10f), and
  require no API guard. In Hex9's native planar domain the six vertices lie on
  the boundary of the coordinate space, so the defect cells straddle the edge
  of the map rather than appearing as interior anomalies.

  **Ancestry.** The strictly nested unit in Hex9 is the half-hexagon (d_cell),
  not the hexagon. The d_cell hierarchy is a single-parent tree: a d_cell
  address composes left-to-right, prefix-truncation always yields its unique
  ancestor, and no edge of the finest-level half-hexagon tiling crosses a
  coarser half-hexagon boundary. Hexagons are assembled from two half-hexagons,
  and 3 of the 9 child hexagons per level (digits 6–8) straddle a d_cell
  boundary and have two valid parents; the canonical mode-0 convention selects
  one, and the exact ancestor is recovered from the address tail (§10b). Under
  that convention the canonical parent function is well-defined at every level
  and multi-resolution roll-up is exact — but, unlike the unconditional quad
  hierarchies of S2 and HEALPix, hexagon roll-up requires deriving the canonical
  ancestor (via the tail) before truncating, and the split-cell ambiguity can
  nest: a run of split digits stays ambiguous until the tail resolves it. H3's
  aperture-7 subdivision is weaker still — it does not nest children inside
  parents at all: a child hexagon may overlap two coarser cells, parent
  assignment is approximate, and a shared parent does not imply a shared
  grandparent. In short, Hex9's half-hexagon hierarchy is exactly nested; its
  hexagon hierarchy is nested by convention.

  **Area.** HEALPix is strictly equal-area but pays in cell shape: its cells
  are quadrilaterals of visibly varying geometry. H3 and S2 are not
  equal-area; cell areas vary by tens of percent across the globe, which
  biases any analysis that compares counts or densities between regions. Hex9
  is quasi-authalic by construction of the warp (§11b): the residual is
  bounded, characterised, and confined to the neighbourhoods of the six
  vertex singularities.

  **Shape and area, independently surveyed.** An independent survey of cell
  geometry — enclosing-cone aspect ratio and WGS84 geodesic cell area over
  N = 5000 uniformly-sampled cells per system — places the four systems into
  two area tiers separated by roughly 100×. A5 and Hex9 form an equal-area
  tier (area coefficient of variation 0.01% and 0.10% respectively); H3 and S2
  do not (12.5% and 14.5%), and the gap is a step, not a gradient. The survey
  reproduces Hex9's warp characterisation independently — worst cell ≈ +4.8%,
  mean absolute deviation 0.001%, all residual pooled at the twelve
  octahedral-vertex defects — on a third-party tool and against the WGS84
  datum, corroborating §11b. On cell shape the order is H3 (aspect ratio 1.06,
  roundest) < S2 (1.24) < Hex9 (1.37) < A5 (2.14): Hex9 reaches the equal-area
  tier while staying markedly rounder than the only other equal-area system in
  the comparison. Both metrics are resolution-invariant (reproduced to four
  decimals five levels coarser), and the dispersion ranking
  A5 < Hex9 < H3 < S2 is the same under every statistic tested (CV, p90, p95,
  max/min). Hex9 is therefore the better combined shape-and-area cell among the
  equal-area systems; A5 distributes a hair of residual more evenly
  (max/min 1.009 vs Hex9's 1.059), but does so with substantially less round
  cells.

  **Aperture.** Hex9's aperture is 9, but the subdivision is not the centred
  aperture-9 scheme of the DGGS literature. In a centred scheme one child sits
  concentrically at the parent's centroid; in Hex9 the parent centroid falls
  on the shared long edge of its two half-hexagons, so there is no central
  child and the 9 children form the shifted half-hexagon arrangement of §8.
  The shift is not a cost. A centred aperture-9 grid must descend one
  triangulation level below its own cells to describe their boundaries; Hex9's
  half-hexagon is its own primitive at every level, and point membership
  reduces to a fixed set of linear inequalities (§10a). Comparisons of
  aperture across systems should note this distinction.

  We coin "shifted aperture 9" to be explicit about the offset. Each parent has
  exactly nine children — aperture 9 in the strict sense — but they carry a
  translational shift rather than sitting centred on the parent. This parallels
  H3, whose aperture-7 children carry a rotational offset between successive
  resolutions yet are still called aperture 7; we prefer to surface the offset
  in the name rather than leave it implicit.

  **Reference body.** H3, S2, and HEALPix are defined on the sphere;
  ellipsoidal use requires an auxiliary-latitude transformation with its own
  distortion budget. Hex9 is defined against any two-axis reference ellipsoid
  directly: the warp is computed from geodesic areas on that body and recomputed
  for any other (§11d). Trained warps currently exist for WGS84 and the sphere;
  the C implementation (libhex9) is tuned for WGS84. WGS84 is the default, not a
  constraint.

  **Distance isotropy.** A hexagonal tiling gives every cell six neighbours at a
  single edge-to-edge distance, so the grid privileges no direction and
  nearest-neighbour and traversal distances are uniform. A quadrilateral grid
  does not: S2's edge neighbours and corner neighbours differ by a factor of √2,
  which biases distance and adjacency computations along the diagonal. Hex9 and
  H3 are isotropic in this sense; A5's pentagonal cells are markedly elongated
  (aspect ≈ 2.14), so their neighbour distances are not uniform.

  **Dual DGGS/CRS role.** This is the row no other system marks "yes", and it is
  the paper's central claim (§9, §10e, Appendix A): a single Hex9 address is a
  DGGS zone identifier when truncated at a level (OGC Topic 21), and in the limit
  a function recovers from it a point on the ellipsoid to arbitrary precision. We
  state this as the weaker, defensible claim — the addressing is
  *quasi-continuous* (§10e), not a strict ISO 19111 CRS: the recovery map is
  discrete-valued and jumps across a measure-zero set of seams, where continuity
  in the ISO sense fails. The distinction from the others is nonetheless real:
  H3, S2, and HEALPix are indexing schemes layered over a pre-existing CRS; they
  identify cells, but their identifiers do not double as position-recovery
  coordinates.

  [Figure: side-by-side cell renderings of H3 / S2 / HEALPix / Hex9 over the
  same region, same nominal resolution — optional, if licensing permits]
