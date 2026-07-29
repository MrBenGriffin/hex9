# DGGS Audit Metrics

A metric set for auditing discrete global grid systems more comprehensively than the
criteria of Sahr, White & Kimerling (2003). Sections A–G apply to any DGGS; the appendix
records Hex9-specific conventions and answers.

## A. Geometric distributions

| Measure | What it tests | Preferred result |
|---|---|---|
| Area distribution | Variation in WGS84 ellipsoidal cell area, reported as one family: CV (100·σ_A/Ā); percentiles of A_i/Ā at p50, p90, p95, p99, p99.9; range max(A)/min(A); threshold exceedance (count and share outside e.g. ±0.1% authalicity, and whether exceedances are spatially localised) | CV 0%, ratios 1.0, exceedances small and localised |
| Cell-centre quality | Whether the published "centre" is the centroid; centre-to-centroid displacement | ~0, or documented convention |
| Positional quantization error | Mean/max distance from an arbitrary point to its owning cell centre, per level — the error an end user actually feels | Tight, predictable per-level bound |
| Aspect ratio / circularity | Enclosing-cone score; catches elongation | 1.0 |
| Compactness | Isoperimetric quotient 4πA/P²; catches perimeter roughness (kept alongside aspect ratio because both → 1.0 for a circle but fail on different pathologies) | 1.0 |
| Convexity / self-intersection | Geodesic convexity of cells; no self-intersecting boundaries (cf. rHEALPix polar caps) | Convex, simple polygons |
| Edge-length range | Longest / shortest geodesic boundary edge | 1.0 |
| Interior-angle range | Variation in rendered cell corner angles | Small, no pathological tails |
| Cell-orientation anisotropy | How cell orientation varies over the globe; matters for flow/directional data | Smooth, documented |
| Cone-point profile | Area error as a function of distance from each cone point | Smooth, bounded, reproducible |
| Tissot field | Local Jacobian: area scale, angular distortion, ellipse axis ratio | Smooth spatial variation, bounded maxima |
| Projection continuity | Jump in position, scale, bearing, or Jacobian across seams | Zero or explicitly bounded |

## B. Structural and topological measures

| Measure | What it establishes |
|---|---|
| Cell-type uniformity | Number and list of first-class cell types |
| Neighbour-degree distribution | Counts of degree 6, degree 5, etc.; making units of curvature explicit |
| Defect localisation | Where differing cell-type instances occur and whether their treatment is uniform |
| Face/seam complexity | Base faces, seam edges, seam-crossing transforms, and maximum transitions for a query/path |
| Grid symmetry group | Which symmetries of the sphere the grid preserves |
| Whole-globe coverage | Explicit test that the partition covers the entire sphere/ellipsoid, poles included |
| Partition validity | No gaps, no overlaps, and a deterministic boundary-ownership rule: every point (including on seams) is owned by exactly one cell per level. *This is the single home for the boundary rule; other sections reference it.* |
| Ellipsoid handling | Whether the system is defined on the sphere, via authalic latitude, or on the true ellipsoid; the error introduced by that conversion, audited rather than assumed |
| Orientation stability | Inter-level axis rotation; 0° for stable-frame systems, or explicit parity transport otherwise |
| Symbolic transport | Whether direction, neighbour, and local-frame symbols retain a level-independent meaning |
| Neighbour semantics | Edge- vs vertex-neighbour distinction; k-ring shape consistency across the globe |
| Internal consistency invariants | System-defined structural invariants checkable by machine (for Hex9: two-colour validity — see appendix) |

## C. Hierarchy

| Measure | What it establishes |
|---|---|
| Exact lineage | Parent/child relation is exact under address truncation or refinement; state precisely what truncation of an address means |
| Exact ownership | A geometric-containment ancestor relation: which cell at a coarser level geometrically contains a given point/cell |
| Lineage–ownership agreement | Rate at which symbolic descendants are geometrically contained in parents; report disagreements explicitly |
| Cross-level regularity | Whether the same partition/address rule applies at every level |

## D. Addressing and round trips

| Test | Required result |
|---|---|
| Canonical uniqueness | No aliases: two distinct valid addresses never name the same cell (encode∘decode does not test this direction) |
| address → canonical geometry → address | Exact identity for every valid address |
| lat/lon → address → ownership test | The original point is owned by the decoded cell, including seams and cone points |
| lat/lon → polyhedral coordinates → lat/lon | Continuous-map reconstruction residual; report maximum and percentiles in metres/nanometres. Distinguish this true inverse-map error from point → cell-centre quantization error (section A) |
| address → centre/vertices → inverse partition | Centre returns the same address; vertices follow the boundary rule of section B |
| Coordinate determinism | Decoded canonical lat/lon bytes are identical across supported platforms |
| Numerical stability | Worst residual near cone points and seams |
| Error-handling contract | Defined behaviour on invalid addresses and out-of-domain points |

## E. Computational costs

| Operation | What to measure |
|---|---|
| Point → owner cell | Throughput and p50/p95/p99 latency; allocations |
| Address → centre/boundary | Latency and numerical residual; six-corner generation cost |
| Inverse partition | Cost of geometric ownership lookup |
| Parent / ancestor | Lineage vs ownership availability and cost |
| Child enumeration | Time to produce children; output-sensitive but bounded |
| Neighbour lookup / k-ring | Including special cases and worst case |
| Cell geometry / area | Karney-area cost, geodesic boundary cost, and cacheability |
| Region covering | Number of output cells or ranges; false-positive **and false-negative** area; runtime |
| Grid distance / line tracing | Cell-step distance availability and its correlation with geodesic distance; cost of tracing a line/arc between two cells |
| Curve traversal | Space-filling-curve behaviour in one place: forward generation, prefix truncation, locality (neighbour-code distance), interval fragmentation for a region query, and inverse cost |
| Batch aggregation | Cells/sec for roll-up and drill-down across resolutions |
| Memory | Bytes per cell ID, per temporary lookup, and any seam/face tables (address-width law lives in section F) |
| Branch burden | Exceptional branches per operation; lookup tables, face-specific transforms, and predicates such as is_pentagon() |
| Parallel scaling | Throughput vs threads, determinism, and shared-state requirements |
| Asymptotic claims | Stated O(·) per operation with fixed-width assumptions made explicit, verified against the measurements above |

## F. Scale and range

| Measure | What to report |
|---|---|
| Resolution range | Minimum and maximum supported levels, e.g. L0–L30 |
| Refinement law | e.g. 9 children per cell; linear scale changes by 3, area by 9 per level |
| Cell-count law | e.g. N = 12·9^L |
| Physical scale | Mean/min/max cell area, diameter, and edge length at every level |
| Usable precision range | Highest level at which ownership, geometry, and round trips meet stated numerical tolerances |
| Cross-level distortion stability | Area and shape statistics by level; ensure cone-point error (section A profile) does not grow unexpectedly |
| Address-width range | Bits required per level; maximum level supported by fixed-width IDs; whether operations require arbitrary precision. *Single home for address-size facts.* |

## G. Ecosystem and standards

| Measure | What to report |
|---|---|
| Standards conformance | OGC API-DGGS / OGC Topic 21 (ISO 19170) conformance classes supported |
| Reference implementation | Existence, language coverage, licence |
| Cross-implementation fidelity | Published test vectors; independent implementations agree byte-for-byte on canonical outputs |

## Appendix: Hex9-specific conventions

### Structural invariants

- **Two-colour validity** — every triangular adjacency flips colour; no parity contradiction around a closed loop.

### Data formats

| Purpose | Recommended representation |
|---|---|
| Canonical ID | Versioned binary 128-bit ID; textual Base32/Base64url/hex form for APIs |
| Human-readable address | `hex9:<version>:<level>:<address>` with an explicitly defined canonical alphabet |
| Analytical tables | Parquet/GeoParquet with hex9_id, level, value columns, and optional geometry |
| Database storage | Native binary ID or fixed-length byte column; index by canonical ID and optional curve key |
| API response | Canonical ID plus level, centre, area, optional boundary, and provenance metadata |
| Geometry exchange | GeoJSON, WKB/GeoPackage, or GeoParquet — but always include the authoritative Hex9 ID |
| Map rendering | Vector tiles or simplified GeoJSON; treat these as display artifacts, not authoritative geometry |
| Range scans | Curve-order interval ranges, clearly labelled as an optional ordering index |
| Raster-like arrays | Cell-ID keyed arrays / Parquet / Zarr-style chunks, rather than forcing a rectangular raster |
