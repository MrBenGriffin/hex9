## Appendix A — The CRS Limit

  **Claim.** Every infinite Hex9 address sequence σ = (a₀, a₁, a₂, …) uniquely
  determines a point on the WGS84 ellipsoid.

  **Setup.** Let Ω ⊂ b_oct be the closed octant face: a compact convex
  triangle of diameter d₀. One step of the Hex9 decode maps Ω into one of its
  9 child cells by

  $$p \;\mapsto\; \tfrac{1}{3}\,p + \mathrm{offset}(C_i),$$

  a Lipschitz contraction with ratio 1/3. For an address prefix (a₀, …, a_L),
  let S_L ⊂ Ω be the corresponding cell — compact and convex. Then

  $$S_0 \supset S_1 \supset S_2 \supset \cdots, \qquad
    \mathrm{diam}(S_L) \;\leq\; d_0 \left(\tfrac{1}{3}\right)^{L} \;\to\; 0.$$

  **Completeness.** Each octant face is a closed bounded subset of ℝ² and
  hence a complete metric space; the S_L are closed subsets of it.

  **Cantor's intersection theorem** (nested compact sets with diameters
  tending to zero in a complete metric space) gives

  $$\bigcap_{L=0}^{\infty} S_L \;=\; \{p^{*}\},$$

  a single point of b_oct. Equivalently, via Cauchy sequences: choose any
  p_L ∈ S_L; for L, M > N both p_L and p_M lie in S_N, so
  d(p_L, p_M) ≤ d₀ (1/3)^N → 0; the sequence is Cauchy, its limit p* lies in
  every closed S_N, and uniqueness follows from diam(S_L) → 0.

  **Lifting to the ellipsoid.** The composition

  $$\text{b\_oct} \xrightarrow{\;\text{AuthalicWarp}^{-1}\;} \text{b\_raw}
    \xrightarrow{\;\text{AK}\;} \text{c\_ell}$$

  is continuous: the warp is C1 by the Clough-Tocher construction, and the AK
  projection is smooth on the octant interior and at its vertices (below).
  Continuous maps preserve limits, so p* maps to a unique point on the
  ellipsoid.

  **Continuity at the octahedral vertices.** The octahedral vertices are the
  points satisfying |x| + |y| + |z| = 1 and x² + y² + z² = 1 simultaneously —
  exactly the points with two zero coordinates. There the octahedron and the
  sphere coincide, the AK map reduces to the identity followed by axis
  scaling to the ellipsoid, and its apparent indeterminacy is a removable
  singularity; the Jacobian at these points is the well-defined limit of the
  linearisation. No actual singularity exists in the map.

  **Almost-everywhere bijectivity.** The map σ ↦ p* is injective except on
  the d_cell seam boundaries — the split-cell case, where 3 of the 9 children
  per level admit two valid parent sequences (§10b). The seam boundaries form
  a set of measure zero on the ellipsoid; away from them, infinite addresses
  and ellipsoidal points are in bijection. The canonical mode-0 convention
  (§10b) selects one address for seam points, exactly as decimal notation
  selects 0.5 over 0.4999….

  **The dual claim** follows: σ truncated to L digits names a compact cell of
  area 510,065,622 km² / (12 · 9^L) on WGS84 (≈ 42.5 million km² at L0,
  719.8 km² at L5) — a DGGS cell in the sense of
  OGC Topic 21; σ in the limit names a point — a coordinate in the sense of
  ISO 19111. The contraction ratio 1/3 is geometrically determined by the
  ternary subdivision, so the convergence rate is explicit: one address digit
  buys a factor-3 reduction in positional uncertainty.
