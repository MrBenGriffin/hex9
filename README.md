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
g_gcd  →  r_gcd  →  c_sph  →  c_oct  →  b_raw  →  b_oct
```

| Step | Domain | Description |
|------|--------|-------------|
| `g_gcd` | Geodetic (degrees) | Standard lon/lat on WGS84 |
| `r_gcd` | Geodetic (radians) | Same, converted to radians for internal use |
| `c_sph` | Spherical Cartesian | Unit authalic sphere — exact, area-preserving datum reduction (authalic latitude) |
| `c_oct` | Octahedral Cartesian | Projected onto the unit octahedron via the AK formula |
| `b_raw` | Geometric barycentric | Per-face barycentric coordinates (unwarped) |
| `b_oct` | Near-equal-area barycentric | Equal-area-warped barycentric — the grid's home space |

The `g_gcd → c_sph` step is the only place the datum appears: geodetic
latitude is converted to authalic latitude (exactly area-preserving, by
Karney's 6th-order series — the same method as PROJ), after which the
octahedral machinery is purely spherical and serves any ellipsoid with a
single trained equal-area field. True WGS84 ECEF remains available as the
`c_ell` domain, and the earlier per-ellipsoid chain (`c_ell` in place of
`c_sph`, with an ellipsoid-trained warp) is selectable per registrar.

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
00701524-8532-2631-5617-370661508102
```
The same address as a Hex9 *label* (`body.key`):
```
0070152485322631561737066150810 . 2
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
 7:  00701524.2
 8:  007015238.0
 9:  0070152485.2
10:  00701524853.2
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
address   : 0070152485322631561737066150810703830
∂ 0.343 nm  geodesic round-trip error
```

**Stonehenge** — layer-35 raw
```
reference : 51°10'43.672800075871"N, 1°49'34.283450385600"W
address   : 435216643842881806281737034377063022d
∂ 0.807 nm  geodesic round-trip error
```

*(Addresses in these examples are computed under the default via-sphere
chain. Hex9 is pre-alpha: the earlier per-ellipsoid chain produces
different addresses at depth and remains selectable per registrar.)*

#### Compact uint64 addresses

A Nazca Spiral at 14.680°S, 75.102°W has the uint64 address
`0xB404155315732370`.  Reading the hex nibbles from the top:

```
0xB404155315732370
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

#### Two names for every cell: the h9 address and the curve address

Every Hex9 cell also has a second, fully interchangeable name: its **curve
address**. The two are bijective — either can be converted to the other, at
any layer — and each is better at a different job.

The **h9 address** (everything above) is the *geometric* spelling. Each digit
says **where**: which of the 12 roots, then which ninth of that hexagon, and
so on down. It is computed directly from coordinates by arithmetic, and the
key nibble makes it self-inverting — the address alone recovers the location.

The **curve address** is the *sequential* spelling. Hex9's cells at any layer
form a single **Hamiltonian circuit** — a closed, Moore-style space-filling
curve that visits every cell exactly once, stepping from each cell to an
edge-neighbour and returning to its start. A cell's curve address says
**when** that circuit reaches it: after the `c` marker, one digit picks the
root's position along the circuit, and each following base-9 digit ranks the
cell among its parent's nine children in circuit order.

A street address versus a page number: the h9 address tells you where the
cell is; the curve address tells you where it falls in the single global
ordering (eg, the order Santa might take to visit every house). The Great Pyramid,
in both spellings:

```
layer   h9 address      curve address
  0     0.0             c0
  1     00.2            c08
  2     007.2           c088
  3     0070.4          c0882
  4     00701.2         c08828
  5     007015.4        c088286
```

Compare the columns. In the h9 spelling the key nibble is recomputed at every
layer (albeit at O(1) cost), and a child's digits do not always extend its 
parent's —  going up a layer means a trivial, but necessary calculation, not 
just shortening. However, the curve spelling nests *exactly*: chop digits off 
the end and what remains **is** the parent's address, every time, with 
nothing recomputed.

The Pyramid's chain happens to be tidy in both spellings, so the difference
is easiest to see one layer down. Here are the nine lineage children of a
layer-5 cell over central Spain, in circuit order:

```
curve          h9
c6135880       4772076.1
c6135881       4772073.3
c6135882       4772070.5
c6135883       4772074.1
c6135884       4772078.5
c6135885       4772072.3
c6135886       4772675.5    ← does not extend 477207
c6135887       4772677.3    ← does not extend 477207
c6135888       4772671.1    ← does not extend 477207
```

Every curve child is the parent's address `c613588` plus one rank digit —
no exceptions, at any depth. In the h9 spelling, six children extend the
parent's digits `477207`, but three are spelled through the neighbouring
`477267`: cells that hang under this parent in the hierarchy, yet whose
geometric digit-path runs through a different coarse cell. This is why h9
addresses cannot be coarsened by truncation alone — and why the curve
spelling, which can be, is worth having alongside.

What each is good at:

| | h9 address | curve address |
|---|---|---|
| Digits mean | *where* — position within each parent | *when* — rank along the space-filling curve |
| Encode / decode a location | direct, arithmetic, self-inverting | via conversion to h9 |
| Parent / coarser bin | recompute (decode and re-bin) | drop trailing digits — exact |
| Sorting a column of addresses | groups by spelling, not proximity | curve order: consecutive addresses are edge-neighbours, so nearby cells cluster |
| Natural fit | encoding, display, geometry | database keys, range scans, spatial indexing |

The curve digits are produced by a small finite-state machine (36 states,
machine-verified) walked down the hierarchy — so a curve address alone does
not reveal geographic position; conversion back to geometry goes through the
h9 machinery. One caveat applies to both spellings equally: a prefix names a
cell's *lineage* descendants — the index tree — whose union has a fractal
boundary, not the exact geometric hexagon. The geometric relation is
*ownership*, covered next.

Neither name replaces the other. The h9 address remains canonical; the curve
address is a companion representation for workloads — sorting, batching,
range queries — where a one-dimensional order with spatial locality is worth
having.

#### Lineage and ownership

Both spellings name the same tree: every cell has nine *lineage* children —
the cells its h9 digits extend, and equally the cells its circuit ranks
subdivide. But hexagons cannot be tiled by hexagons, so lineage is not
geometric containment: some of a cell's lineage descendants lie partly —
and after a few layers, even *entirely* — outside it. This is true of every
hexagon-based DGGS, not just Hex9; for hexagonal zones the two relations
never coincide.

So there is a second, equally important relation: **ownership**. Every fine
cell is owned by exactly one coarse cell — the one that geometrically
contains it (boundary cases are assigned deterministically). Ownership
partitions the globe at every depth: no gaps, no double counting — which is
what aggregation ("roll these statistics up three layers") actually
requires. Lineage answers "what does the index tree hang under this cell?";
ownership answers "what is geographically inside it?". Different questions,
different cell sets.

In Hex9, ownership is exact and arithmetic. The grid is built on a
fully-nested half-hexagon subdivision (each half-hexagon is exactly nine
finer half-hexagons), so the owned set of any cell at relative depth *d*
contains exactly **9^d** cells — every cell, every depth — and is computed
by address arithmetic (`h9_descendants`), with no geometry tests. A cell
straddling its owner's boundary splits as its two half-hexagons, one in and
one out, so partial-overlap weights are exactly 1 and ½ — no per-cell
fractions to tabulate.

This lineage/ownership distinction is under active discussion for
standardisation in
[OGC API — DGGS issue #108](https://github.com/opengeospatial/ogcapi-discrete-global-grid-systems/issues/108)
("owned sub-zones"): the proposal that a DGGS API should offer both verbs,
since each answers a different question. Other hexagonal systems can serve
ownership too — for example by centre-point containment — with owned counts
that vary from cell to cell; Hex9's half-hexagon carrier is what makes the
count a constant 9^d with two-valued weights. A side-by-side rendering of
the two relations across several systems is in
[docs/dggs/dggs_ownership.png](docs/dggs/dggs_ownership.png).

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
