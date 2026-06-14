## 13. Implementation

  Three implementations exist at different stages of maturity: a Python
  reference implementation (the `hhg9` package), a C++ PROJ plugin
  (`h9_boct`), and a planned PostGIS extension wrapping the C++ core.

### 13a. The partition cycle

  The algorithmic heart of encoding is a repeated three-step cycle on octant
  coordinates: classify the point into one of the 9 child regions (the linear
  inequality evaluation of §10a); subtract the child's origin; scale by 3.
  Each iteration emits one address digit and returns the coordinates to the
  original numerical scale, so there is no accumulating floating-point drift;
  the cycle is O(L) per address, numerically stable, and exact in integer
  arithmetic. It is also the CRS limit of §10e made operational: the offset
  removal composed with the ×3 scaling is the inverse of a contraction
  mapping, and iterating it indefinitely converges to the encoded point.

  Decoding runs the cycle in reverse, with one structural caveat: the split
  cells of §10b make the x_list right-to-left, so the decoder first reads the
  tail to fix the terminal mode and then resolves canonical parentage upward
  in a single pass.

### 13b. Address encodings

  Two packed forms cover the practical range:

  **UUID (128 bits).** Thirty-one nibbles carry the L0–L30 hierarchy path —
  one base-9 digit per level — and the final nibble carries the key tail
  (p_c2 and r_mo; §10b). At L30 the cell scale is sub-atomic, so 128 bits is
  effectively unlimited precision for any practical application. The format
  drops directly into any database, PostGIS column, or API that accepts a
  standard UUID; hierarchy traversal is nibble truncation, and spatial
  binning is prefix comparison, with no decoding step.

  **(UUID, adr_byte) pair.** One companion byte (5 bits used: p_mo and the
  terminal d_cell id h) restores what the key tail discards, enabling the
  exact round trip to a representative point — full CRS behaviour. The
  separation is deliberate: spatial indexing needs the UUID alone; coordinate
  recovery needs the pair; most applications never touch the companion byte.

### 13c. Python reference implementation

  The `hhg9` package implements the full pipeline: domain and projection
  management, the AK base projection with its analytical Jacobian, the
  authalic warp (Clough-Tocher forward, Newton-Raphson inverse), encoding and
  decoding, neighbour computation, polygon generation, and GeoPackage/GeoTIFF
  export. All grid operations are vectorised over NumPy arrays. The warp
  characterisation of §11b is produced by this implementation.

### 13d. PROJ plugin

  `h9_boct` is a C++ port of the hierarchy walk and warp inversion as a PROJ
  projection plugin, making Hex9 coordinates available to the standard
  geospatial toolchain (proj, GDAL, QGIS) as a CRS. Round-trip accuracy is
  within tens of nanometres of the Python reference; the residual difference
  traces to Newton-Raphson iteration depth in the inverse warp [see the
  precision note in §11b].

### 13e. PostGIS extension (planned)

  A C extension wrapping the `h9_boct` core will expose encode, decode,
  binning, and neighbour functions inside PostGIS. The implementation
  deliberately excludes Python: most managed PostGIS environments treat it as
  an untrusted language. A working PostGIS extension is also evidence of
  implementability for the OGC submission path.
