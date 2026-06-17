## 15. Conclusion and Future Work

  The argument of this paper runs in one direction: from coherence
  requirements to structure, and from structure to location. Orientability
  selects the triangle; flat mode transport and vertex closure select even
  valence; regularity selects the octahedron; refinement commutativity
  selects odd apertures and the minimal aperture 9; the half-hexagon tiling
  enumeration — machine-verified end to end — leaves exactly one orientation
  per chirality. Setting the geometric realisation aside, precisely two
  genuine free choices remain in the entire construction: which chirality,
  and which bijection assigns the nine digits. Both are conventions in the
  same sense as the prime meridian: the grid does not depend on them, and a
  variant making the other choice is the same mathematical object.

  Everything the system offers follows from that determinacy. Because no
  residual design freedom exists, cell identity is well-defined without
  reference to any prior coordinate system, and the address can serve as both
  DGGS cell identifier and, in the limit, CRS coordinate — the dual claim of
  §9 and §10e, formalised in Appendix A. The geometric realisation is a
  separable concern: the AK base projection and the optimal-transport warp
  place the abstract structure on WGS84 with quasi-uniform areas, and either
  layer can be substituted without disturbing the grid.

  Future work falls into four groups:

  **Warp refinement.** A strictly authalic variant is achievable in principle
  (det J = 1 everywhere) at the cost of unconstrained shear; the present
  quasi-authalic compromise is deliberate, but the trade-off curve — area
  residual against shape regularity, via the shape-dampening parameter —
  deserves systematic characterisation. The angular radius of the
  elevated-deviation zone around the six vertex singularities is a one-time,
  layer-independent measurement not yet made. Deeper-layer warps (L6+ derived
  rather than interpolated) would reduce the inter-mode residual further.

  **Implementation.** The PostGIS C extension is the main outstanding
  engineering item, and the strongest practical evidence for the
  standardisation path. Warp files for legacy ellipsoids (Bessel 1841,
  Clarke 1866) and at least one planetary body would demonstrate the
  generality claim concretely.

  **Verification.** The R-rotation form of constraint B (§8) remains
  available as an independent cross-check of the tiling enumeration. The
  C1 continuity of the warp at octant boundaries is asserted from the
  construction and should be characterised explicitly.

  **Standardisation.** The intended path is publication, then engagement with
  the OGC DGGS Standards Working Group toward a Community Standard, with the
  PostGIS extension as the working-implementation evidence. The OGC/ISO
  terminology mapping is provided in Appendix B (§10.0, glossary); a final
  pass confirming the exact wording against the current editions of Topic 21
  and ISO 19111 remains a pre-submission task.
