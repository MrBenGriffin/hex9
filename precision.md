
# Domains, Projections, and Precision

The Hex9 framework uses a hierarchical triangular grid built on an octahedral
base.  Internally, coordinates flow through a chain of *domains*; each
*projection* is a reversible transform between two adjacent domains.

---

## Core Domains

| Domain  | Symbol                  | Coordinates                        | Purpose                                                       |
|---------|-------------------------|------------------------------------|---------------------------------------------------------------|
| `g_gcd` | Geodetic (degrees)      | (lon, lat) on WGS84 ellipsoid      | External interface — standard geographic coordinates          |
| `r_gcd` | Geodetic (radians)      | (lon, lat) in radians              | Internal radian form; used by root-finder and AK formula      |
| `c_ell` | Cartesian ellipsoidal   | (x, y, z) ECEF, WGS84             | 3D Cartesian on the WGS84 ellipsoid                           |
| `c_oct` | Cartesian octahedral    | (x, y, z) on unit octahedron      | Intermediate between globe and equilateral triangle           |
| `b_raw` | Geometric barycentric   | (b₁, b₂) + face index             | Per-face barycentric coordinates, **before** authalic warp    |
| `b_oct` | Authalic barycentric    | (b₁, b₂) + face index             | Equal-area–warped barycentric — the grid's home domain        |
| `n_oct` | Flat net octahedral     | (x, y) 2D net                     | Flat 2D unfolding for display and diagramming                 |

All domains except `g_gcd` and `r_gcd` carry an **octant identifier** (face
index 0–7) that is passed between projections unchanged.  `b_raw` and `b_oct`
coordinates live inside a √2-side equilateral triangle on each face and require
the face index for full interpretation.

---

## Core Projections

| Projection       | Class / key          | Forward                          | Reverse                           |
|------------------|----------------------|----------------------------------|-----------------------------------|
| `g_gcd ↔ r_gcd`  | `RGCD_GCD`           | degrees → radians                | radians → degrees                 |
| `r_gcd ↔ c_ell`  | `EllipsoidGCD`       | radians lat/lon → ECEF           | ECEF → radians lat/lon            |
| `c_ell ↔ c_oct`  | `AKOctahedralEllipsoid` | ECEF → octahedral (AK formula) | octahedral → ECEF (root-finder) |
| `c_oct ↔ b_raw`  | `OctahedralOctants`  | octahedral → geometric bary.     | geometric bary. → octahedral      |
| `b_raw ↔ b_oct`  | `BrawBoct`           | geometric → authalic (warp.undo) | authalic → geometric (warp.do)    |
| `b_oct ↔ n_oct`  | `BaryNet`            | authalic bary. → 2D net          | 2D net → authalic bary.           |

---

## The Non-Trivial Step: `c_ell ↔ c_oct`

The transform from ellipsoidal Cartesian (`c_ell`) to local octahedral
Cartesian (`c_oct`) is the step that underpins the entire pipeline.

This mapping must be:
* Continuous across octant boundaries.
* Symmetric across mirrored faces.
* Stable at seams and vertices.

### Forward (`c_ell → c_oct`): AK formula

The forward direction uses a tangent-based normalisation of the ellipsoidal
Cartesian coordinates — the AK formula.  It is a closed-form, well-conditioned
map from ECEF to the unit octahedron.

### Reverse (`c_oct → c_ell`): numeric root-finder

The inverse has no reliable closed form.  The AK normalisation includes a
`tan(φ/2)` term that grows without bound near seams and vertices; inverting
it analytically produces large floating-point instabilities, especially near
octant boundaries.

Hex9 uses a **domain-specific hierarchical root-finder**:

* It iteratively searches for the ellipsoidal coordinate whose forward
  projection matches the target `c_oct` position.
* A set of 12 precomputed region-offset maps exploits the fractal structure
  of the grid, reusing the same lookup table at every subdivision layer.
* Each step uses haversine distances over very short arcs, avoiding
  expensive global geodesic calls.
* Near-seam and near-vertex ambiguity is resolved using the octant geometry,
  not heuristics.

#### Python implementation
Depth ≈ 34 levels, beam width ≈ 5.  Achieves sub-nanometre round-trip
precision in practice using standard float64 arithmetic.

#### C++ PROJ plugin (`h9_boct`)
A PROJ-compatible plugin implements the same algorithm in C++ at
depth ≈ 40, beam width ≈ 6, exposing the full `g_gcd ↔ b_oct` pipeline via
the standard PROJ `cct` / `proj` interfaces.

### Why no closed-form inverse?

In simplified form, one component of the forward projection behaves like:

```
t = tan(φ / 2)
```

where φ is the angular displacement within a local octahedral wedge.
`tan()` grows extremely rapidly near ±π/2 (seams and vertex-adjacent regions).
Inverting `φ = 2 arctan(t)` in floating point — particularly near seams where
ellipsoidal eccentricity compounds the instability — is unreliable across the
full domain.  Root-finding avoids these issues by treating the forward
projection as a black box and solving numerically.

---

## The Authalic Warp: `b_raw ↔ b_oct`

The AK formula maps the ellipsoid conformally onto the octahedron, but the
resulting face triangles are *not* equal-area.  `b_raw` holds the raw geometric
barycentric coordinates.  An authalic warp (`BrawBoct`) is applied to produce
`b_oct`, ensuring that equal areas on the ellipsoid correspond to equal areas
in barycentric space.  The warp is stored as a precomputed interpolator and
applied once per batch (per direction), with per-octant y-sign flips handled
via a vectorised `np.where`.

---

## Precision limits (float64)

Throughout this document (and in the project source), **nm = nanometres**
(1 × 10⁻⁹ m), not nautical miles.

### What we measure

*Round-trip error* (`rt_nm`): the WGS84 geodesic distance (GeographicLib)
between the input geodetic point and the point recovered by the canonical
round-trip:

```
g_gcd → r_gcd → c_ell → c_oct → b_raw → b_oct
                                        → b_raw → c_oct → c_ell → r_gcd → g_gcd
```

### Observed global accuracy

Global random sample (10,000 points, GCD-uniform on the sphere):

| Statistic | Error |
|-----------|-------|
| Median    | < 1 nm |
| 99th pct. | < 7 nm |
| Maximum   | < 7 nm |

Named landmark examples (see `test_boct_roundtrip.py`):

| Location | Round-trip error |
|----------|-----------------|
| Great Pyramid | 0.98 nm |
| Stonehenge | 1.31 nm |
| Null Island | < 1 nm |
| Near N. Pole | < 1 nm |
| Antimeridian | < 1 nm |
