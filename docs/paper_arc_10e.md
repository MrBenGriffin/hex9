## 10e. Continuity

  At any finite level L, a Hex9 address identifies a cell of finite geographic
  extent. As L increases, cell diameter decreases by a factor of 3 at each level,
  converging toward zero. An infinite address sequence — a path carried to all
  levels of the hierarchy — therefore defines a nested sequence of cells whose
  diameters tend to zero.

  The reference ellipsoid is a compact metric space. By Cantor's intersection
  theorem, a nested sequence of closed regions with diameters converging to zero
  has exactly one point in its intersection. An infinite Hex9 address sequence
  identifies exactly that point: not a region, but a location on the ellipsoid.

  This means Hex9 addresses behave like real-valued coordinates in the limit. A finite
  address identifies a region; an infinite address identifies a point. The discrete
  address space and the continuous ellipsoid surface are related by this convergence:
  increasing address length corresponds to increasing positional precision, with no
  discontinuity in the relationship between the two.

  At any fixed finite resolution, Hex9 remains a discrete system — cells are regions,
  not points, and the bijection is between addresses and regions. The continuous
  behaviour emerges in the limit, not at any single level. This is consistent with
  the honest characterisation of §9: Hex9 is a discrete spatial reference system
  whose structure converges to a continuous one as resolution increases.
