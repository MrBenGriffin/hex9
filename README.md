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
| `b_oct` | Authalic barycentric | Equal-area–warped barycentric — the grid's home space |

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
The Great Pyramid at Giza (29°58′45.82″N, 31°8′3.46″E) at layer 36:
```
0070143470686461861005464283175018506
```
Breaking it down:
```
0 - 0701434706864618610054642831750185 - 06
│   ╰──────────── 35 layer digits ───────────╯   ╰─ 2-digit metadata tail
╰─ root hexagon (0–B, 12 global roots)
```
The root hexagon is one of 12 that cover the Earth. Each subsequent digit
subdivides the enclosing hexagon by 9. The two-digit tail carries the
minimum metadata needed to reverse the address back to geodetic coordinates.

There is also a *key* form — used for hex-binning — where the tail encodes
only containment, not full reversal.  The Great Pyramid key at layer 36 is:
```
0070143470686461861005464283175018500
```
Truncating a key to a shorter length does **not** directly yield the parent-layer
key; the `c2` identity and octant face mode must be carried explicitly.

First 11 layers of the pyramid address key:
```
 0:  00
 1:  002
 2:  0072
 3:  00704
 4:  007012
 5:  0070140
 6:  00701430
 7:  007014344
 8:  0070143474
 9:  00701434700
10:  007014347064
```

#### Round-trip accuracy — globally < 7 nm

**Great Pyramid**
```
29°58'45.817792004858"N, 31°8'3.457294813097"E  (reference)
29°58'45.817792004871"N, 31°8'3.457294813071"E  (round-trip)
0070143470686461861005464283175018506  (L35 address)
∂ 0.984 nm  geodesic error
```

**Stonehenge**
```
51°10'43.672800075871"N, 1°49'34.283450385600"W  (reference)
51°10'43.672800075845"N, 1°49'34.283450385640"W  (round-trip)
4352164061084274326815104253457062812  (L35 address)
∂ 1.315 nm  geodesic error
```

#### Compact uint64 addresses

A Nazca Spiral at 14.680°S, 75.102°W has the uint64 address
`0x8515044362475050`.  Reading the hex nibbles from the top:

```
0x8515044362475050
  ↑↑↑↑↑
  ||||└─── Layer 4, hexagon 0
  |||└──── Layer 3, hexagon 5
  ||└───── Layer 2, hexagon 1
  |└────── Layer 1, hexagon 5
  └─────── Layer 0, hexagon 8
```

When written in hex, the first digits of the address spell out the root-to-leaf
path through the hierarchy — the address is directly eye-readable with a crib.
See the image below for the first five regions 8 → 5 → 1 → 5 → 0, each
covering 1/9 the area of the preceding layer.

The layer area formula: for total Earth surface area *E*, a hexagon at layer *L*
covers *E* / (12 × 9^L).

Hex9 natively supports:
* **uint128** string addresses (32 hex nibbles) — indexes to layer 28 and beyond.
* **uint64** integer addresses (16 hex nibbles) — layer 13 by default.

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
