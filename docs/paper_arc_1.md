## 1. The Simplicial Carrier

  A discrete global grid partitions the globe into a finite collection of cells. The choice of cell shape is the first apparent design decision: squares, triangles, hexagons, and other polygons all tile the plane, and several tile the globe with appropriate modifications.

  The triangle stands apart from this class. It is the minimal polygon — three edges, three vertices, no simpler closed shape exists — and the only one whose orientation is intrinsic: given any triangle, a consistent notion of clockwise and counterclockwise is determined by its vertex ordering alone, without additional structure. Every other polygon either decomposes into triangles or requires external reference to define orientation consistently across a tiling.


  This intrinsic orientability makes the triangle the natural substrate for a globally coherent grid. We take the triangular field — a tiling of the globe by triangles — as the simplicial carrier on which the remaining coherence requirements will act.


  A coherent grid carrier must satisfy several requirements simultaneously: operators such as interpolation and refinement must be definable without auxiliary choices; each cell must support a unique, intrinsic coordinate system; refinement must be closed and structure-preserving; and no cell-type exceptions or singular vertices may appear. The triangle is the only 2-cell that satisfies all of these at once. Its barycentric coordinates are intrinsic — defined by the vertices alone, with no ambient metric required. It is closed under subdivision. It forms a simplicial complex without diagonal ambiguity.

  On curved surfaces this distinction becomes decisive. A triangulation absorbs curvature at its vertices without changing cell type; tilings by higher polygons must place their topological obligations somewhere visible — as exceptional cells (the twelve pentagons of icosahedral hexagonal grids) or exceptional corners (the eight 3-valent corners of cube-based quadrilateral grids). The triangular field is therefore not one choice among many — every other option either reduces to it or requires additional structure not established by the coherence requirements alone.
