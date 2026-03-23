
> **This file has been superseded by [precision.md](precision.md)** — which
> contains the updated domain table (including `b_raw` and `r_gcd`), the
> corrected projection table, and clarified root-finder documentation.

# Domains, Projections, and Chains
The Hex9 framework uses a hierarchical triangular grid built on an octahedral 
base. Internally, we work with several coordinate `domains`, each with its own 
representation. A `projection` is simply a reversible transform between two 
domains.

## Core Domains

| Domain  | Symbol                 | Coordinates                   | Purpose                                                  |
|---------|------------------------|-------------------------------|----------------------------------------------------------|
| `g_gcd` | Geodetic               | (lat, lon) on WGS84 ellipsoid | External interface: standard geographic coordinates      |
| `c_ell` | Cartesian ellipsoidal  | (x, y, z) in ECEF (WGS84)     | Working space for 3D ellipsoidal geometry                |
| `c_oct` | Cartesian octahedral   | (x, y, z) of unit Octahedron  | Intermediary between globe and equilateral triangle      | 
| `b_oct` | Barycentric octahedral | (b₁, b₂) + face index         | Hierarchical, equilateral grid in normalized coordinates |
| `n_oct` | Flat Net Octahedral    | (x, y) + face index           | 2D Depiction for diagramming purposes                    |

All of the above (except `g_gcd`) carry an **octant identifier** (face index) that is passed between projections. `b_oct` coordinates live inside a √2‑side equilateral triangle on each face and therefore require the face index for context.

## Core Projections

| Projection      | Class                 | Engine                                     | Forward                    | Reverse                       |
|-----------------|-----------------------|--------------------------------------------|----------------------------|-------------------------------|
| `c_ell<->c_oct` | AKOctahedralEllipsoid | 'AK' with normalisation **+ root‑finding** | ECEF (WGS84) → Octahedral  | Octahedral → ECEF (WGS84)     || `c_ell<->c_oct` | AKOctahedralEllipsoid | 'AK' core with normalisation    | Octahedral  → ECEF (WGS84) | ECEF (WGS84) → Octahedral     |
| `c_oct<->b_oct` | OctahedralOctants     | Per Face Rigid 3D->2D transform            | Octahedral → Equilateral   | Equilateral → Octahedral      | 
| `b_oct<->n_oct` | BaryNet               | Per Face Rigid transform                   | Equilateral → Map Position | Map Position →    Equilateral |

The transform from ellipsoidal Cartesian (c_ell) to local octahedral Cartesian (c_oct) is the non-trivial step that underpins the entire projection pipeline.

This mapping must be:
	*	Continuous across octant boundaries
	*	Symmetric across mirrored faces
	*	Stable at seams and vertices

Unlike the forward transform (c_ell → c_oct), the inverse (c_oct → c_ell) cannot be expressed as a reliable closed-form solution.
The AK core projection applies a tangent-based normalisation when mapping the ellipsoid onto an octahedral surface. While this behaves well in the forward direction, the math becomes ill-conditioned for inversion — particularly near seams, poles, and vertices, where slopes become extreme.

To handle this, Hex9 uses a numeric root-finding step for the reverse projection. The solver:
	*	Iteratively searches for the ellipsoidal coordinate whose forward 
projection matches the target c_oct position
	*	Uses a precomputed region-offset map to guarantee the correct octant 
choice across seams
	*	Resolves near-vertex ambiguity while preserving overall symmetry

The search is narrow-beam (≈35 iterations maximum) and converges rapidly, achieving sub-nanometre round-trip precision in practice.

### Sidebar: Why No Closed-Form Inverse?

The AK projection’s core mapping includes a tangent-based normalisation of the ellipsoidal Cartesian coordinates. 
In simplified form, one component of the forward projection behaves like:

t = \tan\!\left(\frac{φ}{2}\right)

where φ is the angular displacement within a local octahedral wedge.

The problem is that tan() grows extremely rapidly near ±π/2, which correspond to seams and vertex-adjacent regions. 
When attempting to invert `phi = 2 arctan(t)` floating-point variations in t near vertical asymptotes produce huge angular instabilities, often amplified by ellipsoidal eccentricity.
Because of this, an analytic inverse would be unreliable across the domain, especially where two or more octants meet.

Root-finding avoids these issues entirely by treating the forward projection 
as a black box and directly solving for the matching ellipsoidal coordinate numerically.

**Why a Domain-Specific Root Finder?**
While general-purpose solvers such as Newton-Raphson or SciPy’s optimizers can
attempt to invert mathematical functions, they have no awareness of the 
geometric constraints or discontinuities inherent in the octahedral projection
domain. In Hex9, seams, mirrored faces, and vertex crossings make these 
boundaries critical: without handling them explicitly, generic solvers often 
converge to the wrong octant or fail entirely near singularities.

The Hex9 solver avoids these pitfalls by leveraging the structure of the 
projection space itself. It uses a small set of 12 precomputed region-offset 
maps that are reused recursively at each subdivision layer, exploiting the 
fractal nature of the hierarchical grid. By taking “small, safe steps” at 
each level, the solver calculates local offsets using haversine distances 
rather than expensive global geodesic computations, since each step operates 
over very short arcs.

This domain-specific strategy achieves rapid convergence in ~35 iterations 
and consistently delivers sub-nanometre round-trip precision using standard 6
4-bit floats — performance that would be unstable or prohibitively expensive 
with general-purpose methods.


# Precision limits (float64)

## Units: 
Throughout this document (and in the project files), nm = nanometres (1e-9 m), 
not nautical miles.

## Domains:
The project uses Domain classes to handle different numeric systems

## What we measure
- Round‑trip errors (rt_nm):
  - WGS84 geodesic distance (GeographicLib) between the input geodetic
  point and the point obtained by the canonical round‑trip (e.g., `g_gcd → c_ell → c_oct → b_oct → c_oct → c_ell → g_gcd`).

