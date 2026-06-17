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

  This convergence lets a function recover position from an address to arbitrary
  precision: decoding an address of growing length yields a sequence of points
  converging to a unique location on the ellipsoid (Appendix A). A finite address
  identifies a region; a sufficiently long address identifies a location to any
  required tolerance.

  This is not the same as being a coordinate reference system in the strict sense
  of ISO 19111, and we do not claim it is. Two limitations are intrinsic. First,
  the address alphabet is discrete: addresses form a totally disconnected sequence
  space, not a continuum, so they are not coordinates in the real-valued sense.
  Second, the point → address map is discontinuous on the measure-zero set of
  d_cell seams (the split-cell boundaries of §10b): arbitrarily close points on
  opposite sides of a seam receive addresses that differ in their leading digits.
  ISO 19111 presupposes continuous coordinates, and Hex9 does not meet that
  requirement.

  What Hex9 offers is better described, by analogy with its quasi-authalic
  geometry, as **quasi-continuous**: position is recoverable from the address by a
  function, to arbitrary precision, everywhere except on a measure-zero seam set,
  and the cell hierarchy converges to points rather than terminating at a finite
  floor. We are not aware of another DGGS whose cell identifier doubles as a
  position-recovery coordinate in this way, though we do not claim the property is
  unique. At any fixed finite resolution Hex9 remains a discrete system — cells are
  regions, the bijection is between addresses and regions — and the
  quasi-continuous behaviour emerges only in the limit. This is consistent with
  §9: Hex9 is a discrete spatial reference system that approaches, but does not
  attain, a continuous one.
