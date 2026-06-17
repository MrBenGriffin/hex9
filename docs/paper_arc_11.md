## 11. Geometric Realisation (AK + Warp)

  Sections 1–10 establish the Hex9 grid as a combinatorial object: mode transport,
  t_cells, d_cells, x_cells, the refinement hierarchy, and the address structure
  are all defined without reference to any specific map projection or reference
  body. The geometric realisation is a separable step that places this abstract
  structure onto the WGS84 reference ellipsoid.

  Three concerns are independent: (1) the combinatorial structure and hierarchy
  (§§1–10); (2) the base projection from octahedron to ellipsoid; (3) the area
  correction. Each can be understood, improved, or substituted without disturbing
  the others. The warp is ellipsoid-specific; the grid is not.

  [Figure: pipeline b_raw → AuthalicWarp → b_oct → Hex9 address]

---

### 11a. The Base Projection

  The AK octahedral projection maps each of the 8 octant faces to the
  corresponding region of the reference ellipsoid. Its forward formula, designed
  analytically by Anders Kaseorg from a force-directed dataset, applies a tangent
  substitution to each octant coordinate and couples the three axes with a
  fourth-root term modulated by a parameter α ≈ 3.2278. This coupling partially
  compensates for the area asymmetry between mode-0 and mode-1 triangular cells
  that arises from the non-uniform Jacobian of the octant face.

  The projection is smooth and has an analytical Jacobian — properties that make
  it suitable as the inner layer beneath the Sinkhorn warp. The 8 octahedral
  vertices coincide exactly with ellipsoidal surface points; no projection
  computation is needed at those locations. The 8 octant faces are mutually
  equivalent under a y-coordinate reflection, so one projection function serves
  all octants.

  No closed-form inverse exists. The backward pass — ellipsoid to octant — is
  realised by numerical root-finding (beam search or Newton-Raphson). The forward
  map is smooth and injective on each octant, so an inverse is guaranteed to exist;
  the numerical method is an implementation choice, not a structural requirement.

  Without further correction, the AK projection introduces roughly ±20% area
  deviation across the octant face — larger apparent cells near octant corners,
  smaller near the centre.

  A second, subtler artefact is an inter-mode area bias. The octant face is a
  right isosceles triangle, so its three corners are not geometrically
  equivalent and the AK Jacobian is not constant over the face; mode-0 and
  mode-1 triangles sample that non-uniform field at systematically different
  centroid positions. Measured by geodesic area on WGS84, the mode means
  differ by 0.394% at L4 and 0.131% at L5, shrinking roughly threefold per
  refinement level (projected ≈ 0.04% at L6). The coupling parameter α is not
  the cause — removing it (α = 0) worsens both the global deviation
  (σ ≈ 7.6% → ≈ 17%) and the asymmetry — and the Sinkhorn warp, operating at
  the hexagon level, cannot rebalance areas between the two halves of a
  hexagon, so the residual carries through to the warped result at slightly
  reduced magnitude. At L5 and finer it is negligible for practical use.

---

### 11b. The Authalic Warp

  The warp corrects the area deviation left by the base projection. It is derived
  by Sinkhorn optimal transport: treating the L4 (or L5) triangle vertices as
  a discrete mass distribution on the octant, the Sinkhorn iteration finds the
  minimal-displacement redistribution that equalises projected cell areas against
  the geodesic areas they subtend on WGS84. The result is a displacement field
  — corrections in b_oct coordinates — rather than absolute positions, which
  improves numerical conditioning.

  The displacement field is precomputed once per ellipsoid and stored. At runtime,
  it is applied via a Clough-Tocher interpolant (C1 continuous). The inverse warp
  uses the forward interpolant to obtain an initial estimate, then refines by
  Newton-Raphson to a tolerance of 10⁻¹⁴ in b_oct (barycentric) units. The
  geodetic round-trip g → b_oct → g, measured on WGS84 at validation points that
  include a near-pole location (89.99°N) and the Greenwich seam, returns to within
  1.8 nm of the original position — many orders of magnitude below any geodetic
  relevance, and comfortably under the 7 nm design threshold.

  The achieved area uniformity (L5, all 708,588 hexagons, WGS84, production warp
  file, geodesic areas): mean deviation exactly 0.000% (confirming closure: the
  cell areas sum to the ellipsoid surface area); area deviation min −3.57%, max
  +4.80%; mean absolute deviation 0.001%; log-ratio standard deviation
  1.99×10⁻⁴. Half of all cells are within 0.0002% of ideal area; 99% within
  0.0044%; 99.99% within 0.43%. The extreme values are a balanced ±4% pair
  affecting a very small number of cells immediately adjacent to the six
  octahedral vertices; the bulk distribution is highly uniform. The warp is
  quasi-authalic rather than strictly authalic. A strictly authalic projection constrains only
  the Jacobian determinant, permitting severe shear and cell elongation. The
  optimal-transport derivation implicitly regularises against shear by minimising
  displacement: the result trades a small area residual for a smooth, low-distortion
  displacement field.

  Six irreducible vertex singularities remain at the octahedral poles — the
  geographic N and S poles and the four equatorial points at 0°, 90°, 180°, 270°E.
  This residual is topological in origin: the Gauss-Bonnet theorem requires
  concentrated curvature at the 4-valent octahedral vertices, and no smooth warp
  can remove it entirely. The singularities are fixed and geometrically determined.
  On WGS84 they are not symmetric: the geographic poles produce more pronounced
  residuals than the equatorial vertices, a consequence of ellipsoidal oblateness
  concentrating curvature at the minor-axis tips.

---

### 11c. The Native Space

  After the warp, each x_cell occupies a well-defined region in b_oct — the
  octant barycentric space. In b_oct, the hexagonal cells are congruent and
  regular: no cell is larger, smaller, or differently shaped than any other
  (modulo the irreducible vertex residuals). b_oct is the natural coordinate
  space for Hex9 computation; it is where the inequality evaluations of §10a
  operate and where address digits correspond to regular geometric subdivisions.

  Any shape variation that appears when Hex9 cells are rendered in a conventional
  map projection — elongated cells near the poles in Mercator, apparent distortion
  in geographic lat/lon — is a consequence of that reprojection, not a property
  of the cells. The distortion belongs to the map. The cells, in their native
  space, are what they are by construction.

  This is the practical meaning of the claim in §9 that Hex9 is a spatial reference
  system rather than merely an indexing scheme: b_oct is a legitimate projection
  in its own right. Conventional geographic projections are derived from it, not
  the other way around.

  [Figure 13: Tissot indicatrices on the b_oct butterfly net — circles of
  near-constant radius across the whole map demonstrate equal area and low shear;
  Mercator and geographic comparison panels to follow]

---

### 11d. Projection Independence

  The AK+Warp combination is the default realisation of the Hex9 grid on WGS84,
  chosen because quasi-equal-area cells make point counts and density estimates
  geographically comparable. It is not the only valid realisation.

  The combinatorial Hex9 structure is projection-independent. Any projection from
  the octahedron to the ellipsoid can be composed with the addressing scheme to
  produce a valid Hex9 coordinate system. The Lee conformal projection produces
  angle-preserving cells at the cost of non-uniform area; the Snyder octahedral
  equal-area projection achieves strict det(J) = 1 analytically. Both compose
  cleanly with the Hex9 hierarchy; the cell identity, digit assignment, and
  adjacency structure are unchanged across projection choices.

  The warp is also ellipsoid-specific but not ellipsoid-exclusive. Recomputing
  the Sinkhorn displacement field against a different reference body (GRS80,
  Bessel, a planetary ellipsoid) yields a Hex9 realisation for that body. The
  addressing hierarchy and all combinatorial structure carry over unchanged;
  only the area-correction data changes. WGS84 and GRS80 are sufficiently close
  that a single warp file serves both in practice — the ellipsoidal difference
  is negligible relative to the warp residual.

  Projection interchangeability is a consequence of the design philosophy: the
  warp corrects for a specific ellipsoid, but the grid it corrects is defined
  independently of any ellipsoid. This separation is not an accident of
  implementation — it is the consequence of insisting that topology, hierarchy,
  and area correction be treated as distinct concerns from the outset.
