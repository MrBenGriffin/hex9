# Abstract (draft)

  Discrete global grid systems are conventionally designed over a prior
  coordinate reference system and inherit its compromises. Hex9 inverts the
  direction of design: we ask what requirements a hierarchical grid must
  satisfy to be geometrically coherent — intrinsic orientability, flat mode
  transport, vertex closure, refinement commutativity — and show that these
  requirements, taken together, essentially determine the grid. The admissible
  cell primitive is the triangle; the admissible seed is the octahedral
  triangulation of S²; admissible refinement has odd linear factor, minimally
  aperture 9; and the hexagonal dual lattice admits exactly one orientation
  per chirality — a result established by machine-verified exhaustive
  enumeration. The structure that survives is a shifted-aperture-9 hexagonal
  hierarchy in which every cell carries a unique address derived from the
  construction alone: truncated at level L, the address is a DGGS cell
  identifier in the sense of OGC Topic 21; carried to the limit, a function
  recovers from it a point on the reference ellipsoid to arbitrary precision.
  The addressing is quasi-continuous — position is recoverable everywhere except
  on a measure-zero set of seams — rather than continuous in the strict ISO
  19111 sense; the same mathematical object serves as both cell identifier and
  position-recovery coordinate, with no prior coordinate reference system as
  input. A separable geometric realisation — an analytical
  octahedral base projection composed with an optimal-transport-derived
  area-correcting warp — places the grid on WGS84 with quasi-uniform cell
  areas: at level 5, 99% of the 708,588 cells lie within 0.005% of ideal
  area, with residual deviation confined to the six octahedral vertices
  required by the topology. The combinatorial grid is projection- and
  ellipsoid-independent; only the warp is specific to the reference body, and
  it is recomputable for any ellipsoid, terrestrial or planetary.

**Keywords:** discrete global grid system (DGGS) · coordinate reference
system (CRS) · hexagonal grid · octahedron · aperture 9 · optimal transport ·
equal-area projection · spatial indexing · WGS84

*Notes: the closing sentence of §9 ("Every cell is where it is because every
constraint converges on that location") is a candidate final line for the
abstract if a less technical register is wanted.*
