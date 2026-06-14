# Hex9 Paper — Introduction (draft)

Discrete global grid systems face a set of familiar tradeoffs. Cell shape, aperture, orientation, and global embedding each involve compromises between equal area, hierarchical consistency, polar behaviour, and computational tractability — compromises often governed by a prior choice of coordinate reference system, which in turn constrains what tradeoffs remain.

Hex9 is an exploration of a different resolution strategy: rather than choosing among tradeoffs, we ask what requirements a grid must satisfy to be geometrically coherent — consistent at every edge, every vertex, and every level of the hierarchy — and examine how far those requirements determine the structure on their own.

These requirements significantly reduce the design freedom. A minimal set — orientability, parity transport, vertex closure, and refinement commutativity — strongly constrains admissible cell shape, embedding class, and aperture structure. What appears to be a large design space resolves to a small family of solutions, selecting a restricted class up to global symmetry.

This derivation has a direct consequence: because the structure is derived rather than chosen, every cell carries a distinct identity tied to a specific surface location. Cell identity serves as a self-contained locator — no prior coordinate reference system is needed to establish where a cell is.

The derivation was discovered rather than designed. The project began as a planar hexagonal tiling study and first targeted the icosahedron — the natural choice, given its near-spherical geometry and the precedent of existing icosahedral grids. The tiling could not be made globally consistent, and the failure resisted diagnosis: the obstruction is not geometric but topological. The odd valence of icosahedral vertices makes a consistent two-colouring of faces impossible, and that two-colouring turns out to be load-bearing for everything a coherent hexagonal hierarchy needs. Once the parity constraint was understood, the octahedron emerged by elimination as the unique Platonic solid with equilateral faces and even-valent vertices. The system presented here is the consequence of following that constraint to its conclusions.

The paper develops this argument constructively. The first part traces the coherence arc, showing step by step how each requirement eliminates alternatives. The second part describes the AK+Warp projection — Anders Kaseorg's octahedral base projection composed with an area-correcting warp — the engineering path from the abstract octahedral structure to a quasi-authalic realisation on the reference ellipsoid (typically WGS84).

