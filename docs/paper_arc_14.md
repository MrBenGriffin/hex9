## 14. Applications

### 14a. Binning and density estimation

  The primary use case is the one that motivated the quasi-authalic default:
  aggregating point observations into cells whose areas are comparable
  anywhere on Earth. Encoding is the O(L) partition cycle; binning is prefix
  truncation of the UUID — with the canonical-ancestor derivation of §10b
  applied first wherever non-canonical addresses may be present. Because cell
  areas are uniform to within the warp residual, raw counts are densities up
  to a single global constant: no per-cell area correction, no latitude
  adjustment. Grids that are not equal-area (H3, S2) require explicit area
  normalisation for the same task, and the normalisation factor varies by
  location.

  [Figure 15: population density heatmap binned to Hex9 cells, one country or
  global]

  Cells need not share a level. Because parentage is exact (§12), a single
  layer can mix resolutions — coarse cells where data is sparse, fine cells
  where it is dense — without T-junctions, seams, or interpolation between
  levels, and the mixed layer still rolls up losslessly. Refinement can
  therefore be driven by the data itself: subdivide a cell only while the
  population it contains stays within a target band, stopping when the count
  falls low enough or a depth limit is reached. Figure 23 shows this for
  Thimphu, Bhutan, where a population layer drives adaptive refinement across
  L5 to L12 — coarse L5 cells over empty terrain, refining to L12 along the
  inhabited valley floors. Each cell is shaded by population density (count per
  authalic cell area), so the fill is directly comparable across levels without
  area correction — a single coherent grid whose cells span seven levels at
  once.

  [Figure 23: Thimphu, Bhutan — population-driven adaptive refinement spanning
  L5–L12 in one mixed-resolution Hex9 layer; fill is population density per
  authalic cell area]

### 14b. Spatial joins and multi-resolution analysis

  Two datasets encoded to Hex9 addresses join on shared prefixes: equality of
  level-K prefixes is co-location at level K, computed without geometry.
  Roll-up from L to any coarser K is exact by the strict ancestry property
  (§12): aggregates computed at fine resolution recombine losslessly at
  coarse resolution, which is not guaranteed in systems where parentage is
  approximate. The same property makes Hex9 addresses suitable as the spatial
  component of composite database keys, with standard B-tree indexes serving
  range and containment queries that would otherwise need a spatial index.

### 14c. Display and rendering

  The canonical cell polygon is the regular hexagon in b_oct; everything else
  is reprojection (§11c). For display in a conventional CRS, cell edges are
  densified and reprojected — the curvature that appears belongs to the
  target projection. Two practical notes. First, hexes straddling the
  antimeridian render incorrectly in naive geographic plots (drawn across the
  full map width); the defect is the renderer's, not the data's, and a bounds
  filter suffices. Second, at the six octahedral vertices the 4-valent cells
  are correct as constructed (§10f) and need no special-case rendering.

  [Figure: the same Hex9 layer rendered in b_oct (regular), Mollweide, and
  Mercator — distortion belongs to the map]

### 14d. Graticule alignment

  The octant face spans exactly 90° in latitude and longitude, and 360 carries
  two factors of 3, so trisection of the degree system is at least as natural
  as bisection — and more useful: bisecting 90° gives 45°, 22.5°, 11.25°
  (non-standard graticule values), while trisecting gives 30° and 10°, both
  standard. At 10° spacing the graticule fits the octant face in 9 clean
  intervals; L1 hexes span 30°, L2 hexes span 10°. In arc-minutes the octant
  (5,400′) trisects cleanly four times (1800′ → 600′ → 200′ → 66.67′) before
  breaking — deeper than bisection achieves. A 10° graticule is therefore the
  natural companion grid for Hex9 visualisations: the alignment is a
  consequence of the shared factor of 9 between base-360 and the ternary
  hierarchy, not a design choice in the projection.

### 14e. Other reference bodies

  The addressing scheme, hierarchy, and interpolation machinery are
  ellipsoid-agnostic; only the Sinkhorn warp is computed against a specific
  body, at a one-time cost of roughly a day of unattended compute (§11d).
  WGS84 and GRS80 share a warp file in practice — their difference is
  negligible against the warp residual. Legacy ellipsoids (Bessel 1841,
  Clarke 1866) and planetary bodies with IAU reference ellipsoids (the Moon,
  Mars) require only their own warp files and their own prime-meridian
  conventions; the mathematical object is unchanged. Highly irregular bodies
  (Phobos, small asteroids) are out of scope, as for every current DGGS.
