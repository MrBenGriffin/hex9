## 12. Comparison with Prior Art

  The differences between the established systems — H3, S2, HEALPix — are well
  documented in the literature; this section makes only the comparisons that
  bear on Hex9's claims.

  | Property | H3 | S2 | HEALPix | A5 | Hex9 |
  |---|---|---|---|---|---|
  | Base polyhedron | Icosahedron | Cube | (sphere-native) | Dodecahedron | Octahedron |
  | Cell shape | Hex (+ 12 pentagons) | Quadrilateral | Mixed quad | Pentagon | Hex |
  | Aperture | 7 | 4 | 4 | 5 then 4 | 9 (shifted) |
  | Equal area | No | No | Yes (strict) | Yes | Quasi (p99 < 0.005%) |
  | Strict ancestry at all levels | No | Yes | Yes | — | Yes |
  | Distance isotropy | Yes | No (√2) | Yes | No (elongated) | Yes |
  | Dual CRS/DGGS | No | No | No | No | Yes |
  | Reference body | Sphere | Sphere | Sphere | Ellipsoid | Ellipsoid (WGS84) |

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

  **Ancestry.** In Hex9 the canonical parent function is a well-defined map at
  every level, and the canonical prefix at level L−1 is a function of the
  canonical prefix at level L alone; by induction, two cells sharing a
  canonical ancestor at any level share canonical ancestors at all coarser
  levels. Multi-resolution roll-up is therefore exact. H3's aperture-7
  subdivision does not nest children inside parents: a child hexagon may
  overlap two coarser cells, parent assignment is approximate, and a shared
  parent does not imply a shared grandparent. S2 and HEALPix have strict
  hierarchies; Hex9 matches them while retaining hexagonal cells.

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

  **Reference body.** H3, S2, and HEALPix are defined on the sphere;
  ellipsoidal use requires an auxiliary-latitude transformation with its own
  distortion budget. Hex9 is defined against the reference ellipsoid directly:
  the warp is computed from geodesic areas on WGS84, and recomputed for any
  other reference body (§11d).

  [Figure: side-by-side cell renderings of H3 / S2 / HEALPix / Hex9 over the
  same region, same nominal resolution — optional, if licensing permits]
