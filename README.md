## Hex9
`Hex9` is an **ongoing project** developing a novel *Discrete Global Grid System* (`DGGS`)
built on a purpose-designed **octahedral coordinate reference system** (`CRS`).
These are two distinct, layered components:

* **The CRS pipeline** maps any point on the WGS84 ellipsoid through a chain of
  coordinate domains — ending in a pair of authalic barycentric coordinates
  (`b_oct`) on the surface of a unit octahedron.  This mapping is continuous,
  bijective, and achieves sub-nanometre round-trip fidelity.

* **The DGGS** lives entirely in `b_oct` space.  The octahedral face is
  subdivided hierarchically into half-hexagons, producing a hexagonal grid
  whose geometry, hierarchy, and addresses are all deterministically derivable
  from the coordinates — no look-up tables, no precomputed databases.

Hex9 supports population mapping, environmental modelling, heat-mapping,
hex-binning, and other geospatial analyses.

---

### The CRS Pipeline

The pipeline from geodetic coordinates to the grid's native space is:

```
g_gcd  →  r_gcd  →  c_ell  →  c_oct  →  b_raw  →  b_oct
```

| Step | Domain | Description |
|------|--------|-------------|
| `g_gcd` | Geodetic (degrees) | Standard lon/lat on WGS84 |
| `r_gcd` | Geodetic (radians) | Same, converted to radians for internal use |
| `c_ell` | ECEF Cartesian | 3D Cartesian on the WGS84 ellipsoid |
| `c_oct` | Octahedral Cartesian | Projected onto the unit octahedron via the AK formula |
| `b_raw` | Geometric barycentric | Per-face barycentric coordinates (unwarped) |
| `b_oct` | Authalic-ish barycentric | Near Equal-area–warped barycentric — the grid's home space |

The non-trivial step is `c_ell ↔ c_oct`: the forward direction uses the AK
tangent–normalisation formula; the inverse has no reliable closed form and is
solved numerically with a domain-specific hierarchical root-finder.
A PROJ-compatible plugin (`h9_boct`) exposes the full `g_gcd ↔ b_oct`
pipeline to any PROJ-aware application.

For display, `b_oct` coordinates are projected into a 2D net (`n_oct`) that is
fully independent of any cartographic map projection chosen for the final output.

---

### The DGGS

#### Why hexagonal tiling?
* Hexagonal tiling is unique among regular tilings: every neighbour shares a full edge.
* Hexagonal tiling corresponds to optimal circle packing.
* However:
  * Hexagons cannot be tiled with hexagons.
  * The sphere cannot be covered by a pure hexagonal tiling.
* So, although ideal HHGs have a long history in geospatial modelling, most global HHGs involve trade-offs:
  * Slight distortion to approximate the sphere.
  * A finite number of hierarchical layers.
  * Partial or approximate inter-layer transitions.
  * Additional polygon types (commonly pentagons) for closure.
  * Precomputed databases of fixed reference points.

**Hex9** takes a new approach: the grid is defined in the warped barycentric
space `b_oct`, where the tiling is exact and the hierarchy is unlimited in depth.
The display is decoupled from the grid — the net projection (`b_oct → n_oct`)
handles rendering, not addressing.

#### Addressing
A location encodes to a 128-bit **UUID** — the canonical Hex9 address.
The Great Pyramid at Giza (29°58′45.82″N, 31°8′3.46″E) encodes to:
```
00701523-8841-1584-7742-040444242844
```
The same address as a Hex9 *label* (`body.key`):
```
0070152388411584774204044424284 . 4
│╰──────────── 31 body digits (layers 0–30) ────╯   ╰─ single-nibble key
╰─ root hexagon (0–B, 12 global roots)
```
The first digit is one of 12 root hexagons that cover the Earth; each
subsequent digit subdivides the enclosing hexagon by 9, down to layer 30.
The single key nibble carries the minimum metadata — the `c2` identity and
octant face mode — needed to reverse the address back to geodetic
coordinates, so the UUID alone is self-inverting.

The same point truncated to a **bin** at the first 11 layers:
```
 0:  0.0
 1:  00.2
 2:  007.2
 3:  0070.4
 4:  00701.2
 5:  007015.4
 6:  0070152.4
 7:  00701523.4
 8:  007015238.0
 9:  0070152388.2
10:  00701523884.0
```
The key nibble (after the dot) is recomputed at each layer: truncating a
deeper address does **not** directly yield the parent-layer bin — the `c2`
identity and octant face mode must be carried explicitly.

#### Accuracy

Hex9 addresses are **binned** — a coordinate resolves to the cell that
contains it, and every point in that cell shares one address. The cell area
at layer *L* is *E* / (12 × 9^L) for total Earth surface area *E*, so:

* **UUID** (layer 30) resolves to ≈ 1,000 nm² (≈ 30 nm across).
* **uint64** (layer 14) resolves to ≈ 1.9 m² (≈ 1.4 m across).

For maximum positional fidelity, a **raw representative** string at layer 35
round-trips to within **7 nm** globally — projection-limited, since the
layer-35 cell itself is far smaller (≈ 0.02 nm²). ('nm' here is SI
nanometres, not nautical miles.)

**Great Pyramid** — layer-35 raw
```
reference : 29°58'45.817792004858"N, 31°8'3.457294813097"E
address   : 00701523884115847742040444242847137308
∂ 1.029 nm  geodesic round-trip error
```

**Stonehenge** — layer-35 raw
```
reference : 51°10'43.672800075871"N, 1°49'34.283450385600"W
address   : 4352166438362084635124244718646587130d
∂ 3.162 nm  geodesic round-trip error
```

#### Compact uint64 addresses

A Nazca Spiral at 14.680°S, 75.102°W has the uint64 address
`0xB404155374658884`.  Reading the hex nibbles from the top:

```
0xB404155374658884
  ↑↑↑↑↑
  ||||└─── Layer 4, hexagon 1
  |||└──── Layer 3, hexagon 4
  ||└───── Layer 2, hexagon 0
  |└────── Layer 1, hexagon 4
  └─────── Layer 0, hexagon B (root)
```

When written in hex, the leading digits spell out the root-to-leaf path
through the hierarchy — the address is directly eye-readable with a crib:
the first regions B → 4 → 0 → 4 → 1 each cover 1/9 the area of the
preceding layer.

The layer area formula: for total Earth surface area *E*, a hexagon at layer *L*
covers *E* / (12 × 9^L).

Hex9 natively supports:
* **UUID / uint128** (128-bit, 32 hex nibbles) — canonical address to layer 30.
* **uint64** (64-bit, 16 hex nibbles) — compact address to layer 14.

At layer 30, a single hexagon covers roughly 1,000 nm² (one thousand square nanometres).

---

### What's included

* A working Python implementation of the full CRS pipeline and DGGS, precise
  to sub-micrometre accuracy using geodesics.
* A PROJ-compatible plugin (`h9_boct`) for integration with PROJ-aware tools.
* Unit tests for the H9 grid engine.
* Examples and tutorials for geospatial tasks.

---

### Documentation

* [Introduction](introduction.md) — foundational insight and grid geometry
* [Precision](precision.md) — domain table, projection table, root-finder detail, accuracy benchmarks
* [Enumeration](enumeration.md) — mathematical foundations of the half-hexagon hierarchy
* [Examples](examples/examples.md) — step-by-step example guide (updated for 0.1.1a1)
* [Early thoughts](assets/docs/past.md) — historical notes

---

![Octahedral Projection as Hexgrid](images/rhombus.jpg)

---

### How can I help?

* Explore the grid and experiment with it.
* Suggest improvements to enhance understanding or usability.
* Contribute bug fixes or enhancements via pull requests.

### Why NOT H9?
Hex9 is a self-funded solo project written as a proof-of-concept and demonstrator,
to show that new approaches to the HHG question remain open.
It is not trying to compete with other HHG systems — it only aims to offer
another approach, one that works particularly well for grid-layer transitions.

![Octahedral Projection](images/butterfly.jpg)
