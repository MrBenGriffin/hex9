## 9. Hex9 Cell Identity — Structure as Locator

  The system defined by Axioms 1–9 and constructed through the preceding steps
  is **Hex9**: a shifted-aperture-9 hexagonal grid on an octahedral embedding of
  the reference ellipsoid, with a cell hierarchy in which every cell has a distinct
  address, derived from simplicial coherence requirements alone.

  Each Hex9 cell is identified by its octant (one of 8) and its path through the
  refinement hierarchy. This pair is not a coordinate computed from a prior reference
  system — it is the cell's identity. The bijection of Axiom 8 (Unique Cell Addressing) guarantees that this identity
  encodes a specific, unambiguous geographic region. At each finite refinement level
  the correspondence between addresses and cells is exact: one address, one cell, no
  exceptions. Cell location is recovered directly from combinatorial structure; no
  external coordinate transformation is required to establish it.

  Hex9 is a discrete spatial reference system in the sense of OGC Abstract Specification
  Topic 21 [@ogc_topic21], which characterises a DGGS as a spatial reference system that
  partitions and addresses the globe through a hierarchical tessellation of cells. What
  distinguishes Hex9 within this class is the direction of derivation. The conventional
  workflow — select a coordinate reference system, then design a grid over it — is
  inverted. Here, the geometric coherence requirements of §§1–8 define the structure;
  the structure constitutes the locating system. No prior CRS is required as input.

  This correspondence is discrete, not continuous. Hex9 cell identity uniquely identifies
  a geographic region at each resolution. The hierarchy of nested regions converges to
  a point as refinement increases, but Hex9 does not provide continuous coordinates in
  the sense of a projected CRS. For metric operations — distances, areas, interpolation
  — a geometric realisation onto the reference ellipsoid remains necessary, and is the
  subject of §11.

  The enumeration completed in §8 is what makes self-contained location possible. A
  grid with residual design freedom — orientations, apertures, or embeddings left as
  choices — cannot serve as its own locating system, because different choices produce
  different grids that disagree on cell identity. Hex9 has no residual freedom. Every
  cell is where it is because every constraint converges on that location.
