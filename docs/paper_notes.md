# Hex9 — Paper Notes
*(Code artefacts use the prefix `h9` — e.g. `h9_encode`, `h9_boct` — as an identifier-friendly abbreviation; prose uses Hex9 throughout.)*
*Working notes toward a formal publication and OGC submission.*

---

## Working Title (placeholder)
**Hex9: A Quasi-Authalic Hexagonal DGGS with Dual CRS Compliance on the WGS84 Ellipsoid**

---

## Core Novelty Claims

### 1. Dual CRS / DGGS by construction
The Hex9 address is simultaneously:
- A **cell identifier** when truncated at layer L (satisfies OGC Topic 21 — DGGS)
- A **point coordinate** in the limit as L → ∞ (satisfies ISO 19111 — CRS)

This is the central theoretical claim. The address length *is* the precision — a 6-digit address defines a specific area on the ellipsoid, not a point with an unstated number of significant figures. No other system makes both claims simultaneously via the same mathematical object.

**Distinguishing from prior art** (space-filling curves, p-adic numbers, quadtree addressing):
- Those systems also have the limit-to-point property, but on projected planes or abstract spaces
- Hex9's limit point is on the WGS84 ellipsoid, with quasi-uniform cell areas globally
- The quasi-authalic warp means address length corresponds to *consistent* physical area anywhere on Earth
- Hex9 explicitly satisfies two named ISO/OGC standards by construction — not by analogy

### 2. Strict hierarchy and complete ancestry
- Every child cell belongs to exactly one parent at every level
- Shared parent at layer L implies shared parent at ALL coarser layers
- Contrast with H3: a shared parent does not necessarily imply a shared grandparent
- Enables reliable spatial aggregation, roll-up queries, and multi-resolution analysis

**Proof sketch.**

Define the canonical parent function φ: (level-L x_cells) → (level-(L-1) x_cells) as follows. For x_cells with x_dig in 0–5 (non-split), the parent is uniquely determined by the digit and the canonical mode inherited from above — there is no ambiguity. For x_cells with x_dig in 6–8 (split, high trit = 2), two valid geometric parents exist; φ assigns the mode-0 parent by the canonical convention. In both cases φ is well-defined: every x_cell has exactly one canonical parent.

**Key lemma.** The canonical prefix at level L−1 is determined solely by the canonical prefix at level L. Formally: if two x_cells C₁ and C₂ have identical canonical addresses at level L (same x_dig sequence a₀a₁…a_L), then they have identical canonical addresses at level L−1 (same prefix a₀…a_{L-1}).

*Proof of lemma.* The canonical parent assignment at level L depends on: (a) the digit a_L, and (b) whether that cell is a split cell (a_L ∈ {6,7,8}) and if so, its mode. Mode is determined by the canonical mode-0 convention applied at level L, which is itself a function of the canonical prefix a₀…a_L — not of digits beyond L. Therefore φ(C₁) and φ(C₂) receive identical inputs and produce identical outputs. ∎

Applying the lemma inductively: shared canonical ancestry at level L ⟹ shared canonical ancestry at level L−1 ⟹ … ⟹ shared root. Shared ancestry at any layer implies shared ancestry at all coarser layers.

**Contrast with H3.** H3's icosahedral base has 5-valent vertices. The 12 pentagonal cells at each layer have 5 children instead of 6, breaking the uniform ternary structure. H3's parent assignment at level L can therefore depend on whether a cell is a pentagon descendant — a property that is not a simple function of the L-digit prefix. Grandparent sharing does not follow from parent sharing because the resolution-dependent pentagon structure can differ at higher levels.

### 3. Quasi-authalic hexagonal cells
- Cell shape: consistent hexagons (contrast HEALPix: mixed/distorted; S2: quadrilateral)
- Pre-warp baseline: ±20% area deviation
- **Warp progression:**

  | Stage | MAE | Notes |
  |---|---|---|
  | No warp | ~20% | Base octahedral projection |
  | L4 Sinkhorn (pre-gradient) | ~5% at vertices, <1% global | Warp derivation layer |
  | L4 + seam-band gradient | MAE ~0.45% | 300-iter gradient descent on seam ring |
  | L5 Sinkhorn (production, `WGS84_l5_warp_data`) | p99 0.0052%, σ=1.89×10⁻⁴ | re-measured 2026-06-10 via ex0080w; supersedes all earlier characterisations |

- **SUPERSEDED 2026-06-12** — the block below is the pre-F6-retrain field. The
  current canonical L5 field (deployed F6 retrain, measured via ex0081w; the
  numbers used in the paper) is: mean exactly 0.000% (closure), min −3.57%, max
  +4.80%, MAE 0.001%, log-ratio σ 1.99×10⁻⁴; |%dev| p50 0.0002% · p99 0.0044% ·
  p99.99 0.43%. The old one-sided −6.93%/+0.078% spike is now a balanced ±4%
  pair at the six octahedral vertices. Do not quote the figures below in prose.

- **Current warp characterisation** (`WGS84_l5_warp_data.npz`, 708,588 hexagons, L5, WGS84 ≡ GRS80, measured 2026-06-10 via `ex0080w_warped_authalics.py`):
  - Area min/max/ideal: 669,935,030 / 720,395,109 / 719,833,841 m²
  - % dev min/mean/max: −6.932% / 0.000% (exact — confirms closure: areas sum to ellipsoid total) / +0.078%
  - log-ratio std dev: 0.00018920
  - |% dev| percentiles: p50=0.00072% · p90=0.00223% · p95=0.00295% · p99=0.00524% · p99.9=0.03913% · p99.99=0.28009%
  - edge anisotropy (max/min − 1): p50=32.83% · p90=39.40% · p95=40.73% · p99=44.26% · p99.9=59.64% · p99.99=69.64%
  - edge CV (shape regularity): p50=12.00% · p90=13.76% · p95=13.98% · p99=14.70% · p99.9=18.46% · p99.99=20.74%
  - **Note**: min −6.93% is a very small number of cells right at seam/vertex locations (p99.99=0.28% covers all but ~71 cells)
  - **Superseded**: two earlier stat sets (p99 0.024%/σ 2.03×10⁻⁴ and p99 0.014%/max −0.330%/σ 5.24×10⁻⁵) circulated in these notes; both predate this run and should not be quoted
  - **WGS84 vs GRS80**: identical to the last significant figure — shared warp file confirmed empirically
- Singularities are analogous to poles in cylindrical projections: known, fixed, geometrically determined (geographic poles + 4 equatorial points at 0°/90°/180°/270°E); polar vertices more pronounced than equatorial due to WGS84 ellipsoidal oblateness

### 4. Distance isotropy
- Hexagonal cells have no √2 distance artefact
- Contrast S2: rectilinear subdivision causes diagonal vs cardinal distance asymmetry

---

## Comparison with Prior Art

| Property | H3 | S2 | HEALPix | Hex9 |
|---|---|---|---|---|
| Cell shape | Hex (+ pentagons) | Quad | Mixed | Hex (+ 12 pentagons) |
| Equal area | No | No | Yes (strict) | ~Yes (p99 < 0.005%; balanced ±4% tail at the 6 vertices; p99.99=0.43%) |
| Strict ancestry at all levels | No | Yes | Yes | Half-hex: yes; hex: by convention (tail) |
| Distance isotropy | Yes | No (√2) | Yes | Yes |
| Dual CRS/DGGS | No | No | No | Yes |
| Reference body | Sphere | Sphere | Sphere | Ellipsoid (WGS84) |

*Note: H3/S2/HEALPix differences are well-documented in existing literature; the comparison table needs one focused paragraph, not exhaustive treatment.*

---

## Mathematical Foundations

### Narrative framing

The construction may appear to involve many independent design decisions. In practice, each early choice closes off more options than expected — the system is largely self-determining once the initial constraints are accepted.

The paper should tell this as a story people can follow and relate to. The honest version includes the back-pedalling: Hex9 did not emerge from a clean sequence of deliberate choices. The icosahedron was tried first and failed — not for a geometric reason that was immediately obvious, but for a topological one (odd vertex valency) that took time to identify. Once the octahedron was chosen and the equilateral triangle accepted as the primitive, the consequences unfold almost without further decision:

- The octahedron's even-valent vertices require a 2-colouring of faces → the diploid structure is not a choice, it is forced
- The diploid structure means the fundamental domain is the half-hexagon → the tiling problem is defined
- The half-hexagon tiling problem has 49 solutions → of these, 9 pairs afford hexagonal tiling → of those, exactly 1 pair also tiles as 3 equilateral triangles → the tiling is uniquely selected
- The chirality of the mirror pair forces the mode convention → mode is not labelled, it is inherited
- The 4-valent octahedral vertex forces the c2=0 boundary treatment → not a workaround, the correct topological response
- The fractal cascade then determines the addressing, the CRS limit, and the warp requirement

The degrees of freedom are fewer than they appear. Setting the projection aside, the tiling of the octahedron involves exactly two genuine free choices:

1. **Chirality** — which of the two mirror solutions to use. The practical question was where to place the pole-touching hexagons (the cells at the octahedral vertices): on the Europe/Americas axis or the Asia/Pacific axis. The Asia/Pacific placement was chosen because it keeps Europe — and the cartographic conventions inherited from European geographic history — more intact. This is not a defence of eurocentric thinking; it is a consequence of it. Global coordinate systems (the prime meridian, standard map orientations) are already set by that history, and minimising disruption to existing conventions is the pragmatic justification.

   Hex9 also adopts the Greenwich prime meridian — but the grid has no mathematical requirement to do so. Any longitude could serve as the reference; a hypothetical `Hex9.22` variant holding 22°E as its prime meridian would be equally valid and would produce an identical mathematical object, just rotated. The Greenwich adoption is chosen for the same reason as the chirality: it is the least controversial option, and departing from it would require a stronger political argument than the grid itself can make. The grid doesn't mind.

   This separability has a practical consequence: Hex9 is not Earth-specific. An extraterrestrial application — lunar mapping, a Martian DGGS — would substitute the body's reference ellipsoid parameters and adopt its native prime meridian convention (Mars uses a small crater, Airy-0; the Moon uses the mean Earth-direction; all chosen by the same kind of arbitrary historical stake-in-the-ground as Greenwich). The mathematical object is unchanged; only the warp parameters and the longitude zero differ.
2. **Digit assignment** — which of the 9 child positions within a parent x_cell receives which x_dig (0–8). The post-2025 assignment is not arbitrary: digits 0–2 label the three x_cells whose mode-0 d_cell half is interior to the current t_par; digits 3–5 label those whose mode-1 d_cell half is interior; digits 6–8 are the split x_cells that straddle two t_pars. The low ternary digit (mod 3) within each group identifies edge orientation (flat/forward/back). This makes the diploid structure legible in the digit — the high ternary trit (0, 1, or 2) directly encodes mode ownership. Any bijection is mathematically valid; this one was chosen because it makes the structure visible.

Everything else either follows from the constraints or is a geographic/engineering decision (pole alignment, warp method, ellipsoid choice) that does not affect the mathematical object itself. Two choices for a system this complete is a remarkably small number.

*The goal for the introduction: lead the reader to feel that Hex9 is not so much invented as discovered — the system that was waiting to be found once the right questions were asked.*

---

These layers are independent and should be presented in order — each one builds on the previous.

### Layer 1 — Diploid (crystallographic) structure
The octahedron has O_h symmetry (order 48). Hex9's fundamental domain is not a triangle but a **half-hexagon** — each octant face appears in two orientations (∇/△, or mo=0/mo=1 in the implementation). This diploid structure is a crystallographic observation, not merely a geometric convenience. It is the reason the two-mode addressing works and why the seam between modes has the specific boundary behaviour it does.

### Layer 2 — The tiling theorem
Half-hexagons tile half-hexagons. At a linear scale of 1:3 (area ratio 1:9), there are **49 distinct arrangements** of 9 sub-tiles within the parent half-hexagon. Count confirmed by OR-Tools CP-SAT exhaustive enumeration (`experimental/halfhex.py`, space = 27-cell half-hexagon).

**On the count of 49 — paper paragraph:**

> The count of 49 is established by exhaustive enumeration, formulated as an 
> exact cover problem. The target space is a half-hexagon comprising 27 
> triangular cells (9 sub-tiles × 3 cells each, at 3× linear scale). Nine 
> congruent copies of the half-hexagon tile are placed; each copy admits 6 
> distinct orientations (rotations by integer multiples of 60°). A boolean 
> variable is introduced for each legal (orientation, position) pair — a 
> placement that fits entirely within the space. The exact cover constraint 
> requires every cell to be covered by exactly one active placement. The 
> OR-Tools CP-SAT solver [Perron & Furnon] is run in full enumeration mode, 
> exhausting all satisfying assignments. Solutions are normalised to an 
> orientation string — one digit 0–5 per cell in a fixed reading order — and 
> duplicates are removed by hashing. This yields exactly 49 distinct tilings;
> the solver certifies no further solutions exist. Code is provided in the 
> supplementary material. The number 49 arises as one tiling is its own 
> reflection.

The naive upper bound is 6⁹ ≈ 10M; edge-compatibility constraints reduce this to 49. A closed-form derivation is possible in principle but would be no simpler than the enumeration. The half-hexagon has one non-trivial symmetry (the apex mirror reflection), which is why the 49 includes a chiral pair (solutions 09 and 48) rather than being fully asymmetric.

*TODO: clean up `halfhex.py` for supplementary material — the `__main__` block already has the correct 27-cell half-hexagon space and the 49-solution result is confirmed. The Dollar-puzzle space is no longer present. Needs light documentation pass before publication.*

### Layer 3 — Uniqueness of the fractal-capable tiling

The Hex9 solution can be derived constructively — no exhaustive search required. The argument has three steps.

**Step 1 — Tiling an equilateral triangle with 3 half-hexes.**
There are exactly **2 ways** to tile an equilateral triangle with 3 half-hexes: a chiral pair (L and R, mirror images of each other). 
CSAT-verified (`halfhex_further.py`, 9-cell equilateral triangle space, 2 solutions confirmed).

**Step 2 — 2³ = 8 combinations.**
A half-hexagon is by definition 3 equilateral triangles. 
Each of the 3 triangles is independently filled by L or R from Step 1 → **2³ = 8 combinations**. These correspond exactly to the 4 chiral pairs from the full 49 CSAT-enumerated tilings that have the 3-triangle structure — no separate proof required.

**Step 3 — Hexagonal plane-tiling (constraint A).**
For a tiling to produce proper hexagonal cells when joined with its mirror partner, the long edge of every sub-tile must meet the long edge of another (long-edge-to-long-edge). 
Of the 8 combinations, exactly **2 survive** this constraint — and they are mirrors of each other. These are the Hex9 solutions.

This is a fully machine-verified constructive proof — all three steps confirmed by CSAT. The 49-tiling enumeration (`halfhex.py`) and the 18-tiling hexagon-affordance enumeration (`halfhex_further.py`) confirm it from the other direction: the 49 → 4 pairs (constraint B) → 1 pair (constraint A) path arrives at the same unique solution.

**Why constraint A selects a mirror pair, not a single solution.**
None of the valid tilings can tile the hexagonal plane alone — each requires its mirror partner. The plane-tiling unit is always the pair: a whole hexagon formed by joining the two mirror half-hexes on their long edge. This is not a limitation but a consequence of the diploid structure (Layer 1): the two modes are complementary, not interchangeable.

**The Hex9 solutions** (results 09 and 48 of the 49-solution run, mirrors of each other):
- `044044043040230225322512511`
- `442442425420250205300130113`

Both use orientations {0,2,4} twice each and {1,3,5} once each. This even/odd asymmetry is not accidental — it directly reflects the diploid crystallographic structure (Layer 1). The unique fractal-capable tiling carries the signature of the underlying symmetry group.

**The origin of mode.** The 3-symmetry of the Hex9 solution partitions the 9 sub-tiles into 3 equilateral triangles (3 half-hexes each). The two chiralities of the mirror pair correspond directly to the two modes (mode 0 = ∇, mode 1 = Λ). The assignment — which chirality is mode 0 — is an arbitrary convention, analogous to choosing a prime meridian: either triangle of the 3 can be designated mode 0, but once the choice is made it is fixed throughout the system. From that point on, the mode of any d_cell is determined unambiguously by which of the 3 equilateral triangles it belongs to — i.e., by its t_cell. This is why mode is a well-defined, stable property of a d_cell, not an ad hoc label: it is inherited from the 3-symmetry of the unique tiling.

**Proof status (machine-verified 2026-06-10):** The full chain is verified end-to-end by `experimental/halfhex_verify.py` (all checks pass):
- V1: the 9-cell equilateral admits exactly 2 tilings by 3 half-hexes — a chiral pair.
- V2: the 27-cell half-hexagon admits exactly 1 decomposition into 3 equilaterals — so the 3-triangle structure is unique.
- V3: exactly 49 tilings by 9 half-hexes — 24 chiral pairs + 1 self-mirror. Also verified: no two distinct placement-sets share an orientation string, so the hash-dedup in `halfhex.py` is sound (raw solution count = distinct strings = 49).
- V4 (constraint A, long-edge-to-long-edge): exactly 18 survivors — 9 chiral pairs, all within the 49.
- V5 (constraint B, 3-triangle structure, implemented as a geometric post-filter: every placed piece lies wholly within one of V2's 3 equilateral regions): exactly 8 — 4 chiral pairs.
- V6 (A ∩ B): exactly 2 solutions — the recorded Hex9 mirror pair (strings match `halfhex.py` results 09/48).

Note: V5 implements constraint B in its *3-triangle-structure* form (the form used by Steps 1–3 above). The equivalent *R-rotation* form (pair closed under 120° rotation, formal definition below) is not separately implemented; if desired as an independent cross-check, it remains a small addition.

**Constraint B — formal definition.**

The 27-cell half-hexagon space has a 3-fold rotational symmetry: the three equilateral triangle sub-regions (each comprising 9 cells, one per constituent triangle of the parent half-hexagon) are permuted by rotation R by 120° about the long axis. R acts on both cell positions and tile orientations: a half-hex at position p with orientation θ maps to position R(p) with orientation (θ + 2) mod 6 (each 120° rotation adds one step in the 6-orientation cycle).

A tiling T of the 27-cell space is **3-symmetric** if R(T) = T (invariant under R). A chiral pair (T, T') is **3-symmetric as a pair** if R(T) = T' (one partner maps to the other under R; R(T') then equals T by applying R again). Both are meaningful; the Hex9 case is the latter — the two mirror Hex9 solutions are exchanged by R, so the pair is closed under the 3-fold rotation.

**Automation.** Given the 49 orientation strings (one digit 0–5 per cell in fixed reading order), R is a computable permutation on strings: apply the cell-position permutation and add 2 mod 6 to each digit. For each of the 49 solutions T: compute R(T), check whether R(T) is in the 49, and whether R(T) is the mirror partner of T (from the 9 hexagon-affordance pairs). The 4 pairs satisfying constraint B and the 1 pair satisfying both A and B are then identified without manual inspection.

*DONE (2026-06-10): automated as the 3-triangle-structure post-filter in `experimental/halfhex_verify.py` (V5/V6) — 4 pairs satisfy B, 1 pair satisfies A ∩ B, confirmed automatically. The R-permutation form described above remains unimplemented and is now optional (an independent cross-check, not needed for the proof chain).*

**Key asset:** a diagram illustrating the 49 candidates and the Hex9 solution — candidate for Figure 1.

*TODO: locate/prepare diagram in publishable format. CSAT code/formulation to go in supplementary material or appendix.*

**CSAT implementation:** `experimental/halfhex.py` using Google OR-Tools CP-SAT. The solver machinery is validated — the same code was used to solve a published Stack Exchange tiling puzzle independently. The `__main__` block contains the Hex9 proof space (27-cell half-hexagon); the 49-solution result and both Hex9 solutions are confirmed and recorded in the source comments.

### Layer 4 — The fractal cascade
The unique 1:9 subdivision applied recursively gives the ternary hierarchy. Each level is geometrically self-similar. This self-similarity is what gives the Hex9 address its dual nature: finite prefix = DGGS cell; infinite sequence = CRS point (Cauchy sequence converging on the ellipsoid).

### Layer 4b — Octahedral vertex defects and the 3-orientation solution
Discovered May 2025 when stitching octahedral faces together and observing hex-digit discontinuities at octahedral vertices.

**The problem:** On the plane (and at all ordinary face-interior vertices), 6 triangles meet at every vertex — the standard hexagonal tiling is consistent throughout. But at the 6 octahedral vertices, only **4 triangles meet** — an angle deficit of 60°. This is topologically *necessary*: the concentrated Gaussian curvature at octahedral vertices is precisely what allows the flat net to close into a polyhedron (Gauss-Bonnet). A naive enumeration that works everywhere else breaks at these 4-valent vertices.

**The solution:** 3 super-hex orientations, with c2=0 as the **unique orientation** that meets at octahedral vertices. On sub-layers c2 corresponds normally (c2=c2), but at the octahedral boundary layer, only c2=0 faces meet. This is not an engineering workaround — it is the correct topological treatment of the curvature singularity.

**Consequence:** Added complexity to half-hex neighbour finding (finding the long-edge neighbour across an octahedral boundary). Resolved by the pole-alignment insight below.

*Note: this was a non-obvious result worked out independently; LLMs at the time found the problem too complex.*

### Layer 4c — Pole alignment and octant symmetry
Insight concurrent with Layer 4b (May 2025).

**The insight:** Aligning the octahedral poles with the geographic poles (N/S) makes all 8 octant faces **mathematically identical** with respect to the ellipsoid — same shape, same area, same projection function. The rotation is applied in the c_oct → b_raw step (formerly c_oct → b_oct before the rename).

**Downstream simplifications:**
- Seam crossing: `y → -y` (mode-1 face = mode-0 face with inverted y) — resolves neighbour heavy lifting
- Single warp function serves all 8 octants (mode-1 handled by sign flip in `AuthalicWarp.do()`)
- u/v calculations uniform across all octants
- `_m_c2_hx_v2025` seam consistency becomes tractable

**Geographic consequence:** The ~5% deviation singularities fall at the geographic poles plus 4 equatorial points at 0°/90°/180°/270°E — inevitable given the topology, but fixed at known, symmetric, geometrically determined locations.

**For the paper:** *"Aligning octahedral poles with geographic poles renders all 8 octant faces equivalent under the ellipsoidal projection, reducing the projection problem from 8 distinct cases to one, with mode-1 faces handled by a y-coordinate reflection."*

### Layer 4c.i — Chirality choice
The two mirror Hex9 solutions (results 09 and 48 of the 49-tiling enumeration) are mathematically equivalent. The choice between them is a geographic convention — the selected chirality avoids splitting major political and continental boundaries across octant seams (Europe intact, Texas intact, etc.). Global polities and continents fall more naturally within the chosen orientation.

*"The chirality is selected on geographic grounds — the chosen orientation avoids splitting major political and continental boundaries across octant seams."*

**Suggested figure:** octant boundary lines overlaid on a political world map — documents the convention and allows readers to verify the placement. Useful for implementors.

This is analogous to the Prime Meridian at Greenwich: not mathematically special, but a deliberate practical convention that minimises disruption.

### Layer 4d — Why the octahedron is unique among Platonic solids

Hex9's two-mode (diploid) structure requires a consistent 2-colouring of faces (△/▽ — mode-0/mode-1). A consistent 2-colouring exists if and only if every vertex has **even valency**.

| Platonic solid | Face type | Vertex valency | Hex9-compatible? |
|---|---|---|---|
| Tetrahedron | Equilateral △ | 3 (odd) | No |
| **Octahedron** | **Equilateral △** | **4 (even)** | **Yes** |
| Icosahedron | Equilateral △ | 5 (odd) | No |
| Cube | Square | 3 (odd) | No (also wrong face type) |
| Dodecahedron | Pentagon | 3 (odd) | No (also wrong face type) |

The octahedron is the **unique** Platonic solid satisfying both conditions: equilateral triangular faces AND even-valent vertices. Hex9 is octahedral by necessity, not merely by preference.

**This directly explains H3's pentagons.** H3 uses an icosahedral base with hexagonal cells but requires 12 pentagonal exception cells — one at each 5-valent icosahedral vertex. Those pentagons are the topological price of odd valency: the parity of the face 2-colouring cannot be maintained at a 5-valent vertex, forcing an exception. Partial/disconnected icosahedral solutions are possible but require an explicit exception set at all 12 vertices.

*For the paper: "The octahedron is the unique Platonic solid admitting a hexagonal DGGS without exception cells in the addressing scheme. Icosahedral constructions (e.g. H3) require pentagonal exception cells at their 12 odd-valent vertices — a direct consequence of the same topological obstruction."*

### Layer 4d.i — The 12 required pentagons (Euler's theorem)

Euler's theorem for the sphere is inescapable. **Any** tiling of S² by hexagons and pentagons requires exactly 12 **topological** pentagons. This is not a design choice — it is a topological theorem, and it is layer-transitive.

**Terminology.** A *topological pentagon* is a cell with exactly 5 neighbours in the adjacency graph (5-valent cell). This is distinct from a *geometric pentagon* — a cell that looks pentagonal when drawn in some projection but has 6 topological neighbours. Only topological pentagons are counted by Euler's theorem. The two should not be confused.

**Proof.** Let H = number of hexagonal cells, P = number of pentagonal cells. The tiling graph is trivalent (3 cells meet at each edge-junction). Standard Euler F − E + V = 2:

```
F = H + P
E = (6H + 5P) / 2          (each edge shared by 2 faces)
V = (6H + 5P) / 3          (trivalent: 3 faces per vertex)

F − E + V = (H + P) − (6H + 5P)/2 + (6H + 5P)/3 = 2
          = H + P − (6H + 5P)/6 = 2
          = H + P − H − 5P/6 = 2
          = P/6 = 2
          → P = 12
```

This holds for **any** H ≥ 0. The count of 12 is entirely independent of grid resolution.

**Layer-transitivity.** The proof makes no assumption about H. At layer L, Hex9 has 12 × 9ᴸ total cells; exactly 12 of them are topological pentagons, at every layer L — including L = 34 with 12 × 9³⁴ ≈ 2.8 × 10³² cells. The 12 locations are fixed: the 6 octahedral vertices (2 topological pentagons per vertex), aligned with the geographic N/S poles and the 4 equatorial points at 0°/90°/180°/270°E.

**Application to Hex9.** The 12 topological pentagons correspond to the 12 root x_cells (L=0); at L≥1, they are the x_cells covering the octahedral vertices. The angular deficit at each 4-valent vertex (Layer 4b) is what makes them topologically 5-valent — 4 triangular faces meeting instead of 6.

**The geometric near-pentagons are different.** The L5 warp area statistics show ~1000 cells with notably distorted shapes near the seams. These are *geometrically* near-pentagonal (they look like pentagons in lat/lon projection) but are topological hexagons — they have 6 neighbours. They are not counted by Euler's theorem and their number varies with layer and warp quality. Only the 12 at the octahedral vertices are topological pentagons.

**How Hex9 handles the 12.** Hex9's addressing scheme assigns valid x_adr values to all 12 pentagon-cells. There is no `is_pentagon()` flag, no exception branch, no special API path. The angular deficit is absorbed by the 4-valent vertex geometry (Layer 4b) and by the authalic warp.

**The n_oct planar projection externalises the defect.** In the n_oct octahedral unfolding (Hex9's native planar coordinate system), all 6 octahedral vertices lie on the *boundary* of the coordinate domain — on the edges of the unfolded net, not in the interior. The 12 topological pentagons straddle these boundary positions. From any interior point, the tiling looks like a regular hexagonal grid. The topological defect is not hidden — it is placed where it belongs geometrically: at the boundary of the map.

This is not a coincidence of the layout choice. The octahedral net must have all vertices on or at the edge by construction: unfolding places the faces flat, and the face-vertices (octahedral vertices) end up at the corners and edges of the resulting polygon. The n_oct domain inherits this property directly.

This is a general topological necessity, not a property specific to n_oct. S² is compact; R² is not — no continuous bijection from S² into R² exists. Any global projection must externalise the topological obstruction: it either sends points to infinity (Mercator pushes the poles to ±∞ at the top and bottom of the map; stereographic and gnomonic send one point to ∞) or it cuts along a seam (the cut becomes the boundary of the domain). The choice of *where* to place the obstruction is the degree of freedom each projection exercises; keeping it off the boundary sends it into the interior as a visible anomaly.

**Contrast with H3.** H3's 12 topological pentagons are also fixed across layers — the same Euler count — but they are first-class exception cells in the API: every H3 implementation must guard against `is_pentagon(cell)`. Hex9's 12 require no such handling.

*For the paper: "Euler's theorem requires exactly 12 topological pentagonal cells (5-neighbour cells) in any spherical hexagonal tiling, at every refinement level. Hex9 has exactly these 12, located at the 6 octahedral vertices (2 per vertex) — a fixed, geometrically determined set independent of layer. In Hex9's native planar projection (n_oct), all 6 octahedral vertices lie on the boundary of the coordinate domain, so the 12 topological pentagons straddle boundary edges rather than appearing as interior exceptions. Unlike H3, where pentagonal exception cells require explicit API guards, Hex9's 12 pentagons carry valid x_adr addresses and require no special handling."*

### Layer 4e — The equilateral triangle as primitive

Hex9 is grounded in the equilateral triangle as its mathematical primitive. This is a deliberate design choice that gives Hex9 a deep mathematical pedigree:

- **Tiling**: plane-tiling properties are well-understood; Hex9 inherits them directly
- **Rotation**: 60°/120° rotations are native; the half-hex and its 6 orientations follow naturally
- **Ternary subdivision**: 1→9 scaling is natural in equilateral geometry — a triangle subdivides into 9 at 3× linear scale, and uniquely preserves the equilateral shape of sub-triangles
- **Platonic solids**: the octahedron is 8 equilateral triangles; the same primitive underlies the tetrahedron and icosahedron — but as shown above, only the octahedron has even-valent vertices
- **Established literature**: crystallography, discrete geometry, and geometric analysis all use the equilateral as a starting point — the foundations are familiar

The one symmetry of the equilateral triangle **not** exploited in the main construction is the apex mirror (reflection through the axis from polar vertex to base midpoint). This symmetry is used for tests and validation. The Hex9 tiling is chiral — the two mirror Hex9 solutions from the 49-tiling enumeration are related by exactly this reflection — which is why it is excluded from the construction but remains available.

### Layer 4f — The partition cycle and ternary arithmetic

The addressing scheme is computed by a repeated cycle that is both the algorithmic heart of Hex9 and the operational realisation of the CRS limit:

1. **Classify** — determine which of the 9 child regions contains the point
2. **Remove offset** — subtract the child region's origin (translate to child-local coordinates)
3. **×3** — scale up by 3 (zoom into the child's coordinate space)
4. **Repeat** — each iteration yields one address digit

**Why this works cleanly:**
- Each iteration operates at the same numerical scale — no accumulating floating-point drift
- The offset removal + ×3 is a contraction mapping; the infinite cycle converges to a point — this is the CRS limit made computational
- The cycle is O(L) per address lookup, numerically stable, and exact in integer arithmetic

**Why ×3 specifically:**
The ternary base is a consequence, not an arbitrary choice. It is the unique integer scaling that maps the equilateral triangle back to itself with consistent sub-triangle orientation:
- ×2 gives 4 sub-triangles of mixed orientation (2 up, 2 down) — parity breaks
- ×3 gives 9 sub-triangles of consistent, predictable orientation — parity preserved
- No other integer scaling preserves the diploid structure at every level

**The beam search** (`h9_boct.cpp`) is the practical implementation of the classify step when exact arithmetic is expensive — a beam of candidates is searched rather than computing exact triangle membership. Implementation detail; belongs in the implementation section, not the mathematical foundations.

### Layer 4g — The CRS limit: formal sketch

**Claim:** Every infinite Hex9 address sequence σ = (a₀, a₁, a₂, ...) uniquely determines a point on the WGS84 ellipsoid.

**Setup.** Let Ω ⊂ b_oct be the closed octant face (a compact convex triangle, diameter d₀). The partition cycle inverse — one step of the Hex9 decode — maps Ω into one of 9 child cells via:

$$p \;\mapsto\; \tfrac{1}{3}\,p + \text{offset}(C_i)$$

a Lipschitz contraction with ratio 1/3. For an address prefix (a₀, ..., a_L), let S_L ⊂ Ω be the corresponding cell (compact, convex). Then:

$$S_0 \supset S_1 \supset S_2 \supset \cdots \qquad \text{and} \qquad \mathrm{diam}(S_L) \;\leq\; d_0 \cdot \left(\tfrac{1}{3}\right)^L \;\to\; 0$$

**Completeness of b_oct.** Each octant face in b_oct is a closed bounded triangle in ℝ². As a closed bounded subset of the complete metric space ℝ², it is itself complete. The S_L are closed (compact) subsets of this complete space.

**Cantor's intersection theorem** (nested compact sets in a complete metric space, diameters → 0) gives:

$$\bigcap_{L=0}^{\infty} S_L \;=\; \{p^*\}$$

a single point in b_oct. Every infinite Hex9 address uniquely determines a point in b_oct.

*Equivalently via Cauchy sequences:* pick any sequence of points p_L ∈ S_L. For any L, M > N, both p_L and p_M lie in S_N, so d(p_L, p_M) ≤ diam(S_N) ≤ d₀ · (1/3)^N → 0. The sequence is Cauchy; by completeness of b_oct it converges to a limit p*. Since p_L ∈ S_N for all L ≥ N and S_N is closed, p* ∈ S_N for every N, so p* ∈ ∩S_L. Uniqueness follows from diam → 0.

**Lifting to the ellipsoid.** The composition:

$$\text{b\_oct} \xrightarrow{\text{AuthalicWarp}^{-1}} \text{b\_raw} \xrightarrow{\text{AKOctahedral}} \text{c\_ell}$$

is continuous: the warp is C1 (CloughTocher construction); the AK projection is smooth on the octahedron interior and at its vertices (see below). Continuous maps preserve limits, so p* maps to a unique point on the WGS84 ellipsoid.

**Continuity at the octahedral vertices.** The octahedral vertices are the unique points satisfying both |x|+|y|+|z|=1 (octahedron) and x²+y²+z²=1 (sphere) simultaneously — points where two coordinates are zero and one is ±1. At these points the sphere and octahedron coincide exactly; no projection is required. The AK map reduces to the identity (then axis-scaling to the ellipsoid), which is trivially continuous. The Jacobian at these points is the linearisation of that identity map, computed in the implementation as a well-defined limit via infinitesimal perturbation. No singularity exists; the core formula's indeterminate form at those points is removable.

**Almost-everywhere bijectivity.** The map σ ↦ p* is injective everywhere except the d_cell seam boundaries (the split x_cell case: 3 of 9 children per level have two valid parent sequences). The seam boundaries form a set of measure zero on the ellipsoid. Away from seams, encoding is a bijection between infinite Hex9 addresses and ellipsoidal points.

**The dual claim** follows immediately:
- σ truncated to L digits → compact cell of area 510.07M km² / (12 · 9^L) — 719.8 km² at L5 (DGGS, OGC Topic 21)
- σ in the limit → unique point on WGS84 (CRS, ISO 19111)

The same mathematical object serves both roles. The contraction ratio 1/3 is geometrically determined (ternary subdivision); the convergence rate diam ∝ (1/3)^L is explicit and computable.

### Layer 5 — The warp
Applied on top of the abstract fractal structure to reconcile it with a specific reference ellipsoid. Does not affect the crystallographic, tiling, or topological properties — those are properties of the mathematical object itself. The warp is ellipsoid-specific; the layers above are not.

---

## Project History / Motivation

A narrative arc for the introduction — showing the work was discovered, not reverse-engineered.

**Origin:** Started as a planar hexagonal tiling project. The goal was to extend it to a global grid on a polyhedron.

**Icosahedral attempt:** Initial target was the icosahedron (the natural choice given H3's existence and the icosahedron's near-spherical shape). The tiling could not be made to work consistently. The failure was not immediately diagnosable — it took significant time to identify that the root cause was the 5-valency of icosahedral vertices preventing consistent face 2-colouring (parity). The obstruction is topological, not geometric, and is non-obvious.

**The octahedron:** Once the parity constraint was understood, the octahedron emerged as the unique viable candidate — the only Platonic solid with equilateral faces and even-valent vertices. Not chosen arbitrarily; arrived at by elimination.

**May 2025 — hex-digit discontinuity:** When stitching octahedral faces together, a discontinuity in hex-digit assignment was observed at octahedral vertices (4-valent, not 6). Root cause: the concentrated Gaussian curvature at octahedral vertices (Gauss-Bonnet) requires a distinct topological treatment. Solution required 3 super-hex orientations and c2=0 as the unique meeting orientation at octahedral boundaries. Worked out independently over several deep sessions; the problem was beyond LLM capability at the time.

**May 2025 — pole alignment:** Concurrent insight that aligning octahedral poles with geographic poles renders all 8 octant faces mathematically identical, reducing 8 projection cases to 1 and resolving the neighbour-finding complexity as a simple y-sign flip.

**2025 — digit convention change:** Addressing scheme digit assignment changed from middles={3,4,5} to middles={0,1,2} (v2025 scheme), giving geometric meaning to the centre labels. All addresses changed; all code and documentation uses v2025 values.

*This history should inform the introduction and acknowledgements. The icosahedral dead end in particular is worth a paragraph — it shows the octahedral choice is necessary, not arbitrary, and that the topological obstruction is a genuine non-obvious result.*

---

## Design Philosophy

Hex9 inverts the usual direction of projection design. Most map projections and DGGS work **ellipsoid → surface**: starting with real-world geometry and seeking a convenient representation. Hex9 works **properties → structure → ellipsoid**: the desired mathematical properties are specified first (hierarchical, hexagonal, addressable, quasi-authalic), the structure that achieves them is chosen (octahedron + ternary subdivision), and the ellipsoidal correspondence is derived afterwards as an optimisation step.

Other polyhedron-based systems (H3, HEALPix, Fuller/Dymaxion) also start from a polyhedron, so that alone is not novel. What is specific to Hex9 is that the warp is computed *after* the structure is fixed — explicitly separating three independent concerns:

1. **Topology** — the octahedral structure and its rotational symmetry
2. **Hierarchy** — the ternary subdivision and addressing scheme
3. **Area equivalence** — the Sinkhorn-derived warp reconciling the ideal structure with ellipsoidal geometry

Each component can be understood, analysed, and improved independently. The warp can be recomputed for a different ellipsoid without touching the hierarchy; the hierarchy can be extended to finer layers without affecting the warp.

*This separation of concerns is worth a paragraph in the introduction.*

---

## Mathematical Construction

### Pipeline
```
WGS84 ellipsoid → b_raw (octahedral barycentric) → AuthalicWarp → b_oct → Hex9 address
```

### Base construction
- Base polyhedron: octahedron (12 root hexagonal cells)
- Subdivision rule: ternary (each cell → 9 children at each layer)
- Hierarchy: 12 × 9^L cells at layer L
- Addressing: base-9 ternary digits encoding the path through the hierarchy

### AK octahedral projection (b_raw → c_ell)

The base projection from the unit octahedron to the WGS84 ellipsoid uses an analytical formula designed by Anders Kaseorg. It is smooth, vectorisable, and has a closed-form Jacobian — properties that make it suitable as the inner projection layer beneath the Sinkhorn warp.

**Forward direction (octahedron → ellipsoid):**

Let `(u, v, w)` be the unit-octahedron coordinates of a point (one octant, so all three values share the same sign pattern). Define the substitution:

$$x_i = \tan\!\left(\frac{\pi u_i}{2}\right)$$

applied to each coordinate. Then the (unnormalised) projection is:

$$y_u = x_u \cdot (x_v^2 + x_w^2 + \alpha\, x_v^2 x_w^2)^{1/4}$$
$$y_v = x_v \cdot (x_u^2 + x_w^2 + \alpha\, x_u^2 x_w^2)^{1/4}$$
$$y_w = x_w \cdot (x_u^2 + x_v^2 + \alpha\, x_u^2 x_v^2)^{1/4}$$

where α ≈ 3.2278 (optimised empirically; see `ak_octahedral.py`). Finally, normalise to the ellipsoid:

$$p = \frac{(y_u,\, y_v,\, y_w)}{\sqrt{(y_u^2 + y_v^2)/a^2 + y_w^2/b^2}}$$

where `a`, `b` are the WGS84 semi-major and semi-minor axes.

**Invariants.** The 8 octahedral vertices (unit axis vectors `±(1,0,0)`, `±(0,1,0)`, `±(0,0,1)`) map exactly to the corresponding ellipsoidal surface points without projection — the octahedron and ellipsoid share these 8 points, so no computation is needed there. Two faces of the octahedron have exactly zero coordinates on one axis, handled separately.

**Inverse direction (ellipsoid → octahedron):**

No closed-form inverse exists. The backward pass uses a beam search (`find_coords`, beam_width=6): candidate octant points are projected forward and compared to the target by great-circle distance, with the beam iteratively refined. The beam search is the practical implementation of the inversion; it is an implementation detail rather than a mathematical characterisation. For the paper, it suffices to note that the forward map is smooth and injective on each octant, so a root-finding inversion exists; the beam search is one efficient numerical realisation.

**Jacobian.** A full analytic Jacobian `∂(y_u,y_v,y_w)/∂(u,v,w)` is implemented (see `_core_jacobian`, `_norm_jacobian` in `ak_octahedral.py`). This Jacobian is used by the Newton-Raphson polish in the authalic warp inverse.

**Intrinsic mode asymmetry.** The triangular subdivision of each octant face produces two triangle classes — mode-0 (∇, apex toward edge) and mode-1 (△, apex toward interior) — that tile the face in equal numbers. Because the octant face is a right isoceles triangle (one 90° vertex at the octahedron pole, two 45° vertices at equatorial midpoints), the three corners are not geometrically equivalent. The AK Jacobian is therefore not constant over the face, and the mode-0 and mode-1 triangle centroids sample that non-uniform field at systematically different positions: mode-1 centroids sit slightly more toward the interior of the face, where the projection stretches area slightly more.

This produces a small but consistent inter-mode area bias that is independent of the Sinkhorn warp. Empirically (measured by geodesic area on WGS84 at L4 and L5):

| Layer | Mode-0 mean dev | Mode-1 mean dev | Asymmetry |
|-------|----------------|----------------|-----------|
| L4    | −0.196%        | +0.198%        | −0.394%   |
| L5    | −0.066%        | +0.066%        | −0.131%   |

The asymmetry reduces by roughly 3× per refinement level (consistent with the Jacobian variation becoming negligible over smaller triangles), projecting to ~0.04% at L6 and negligible thereafter.

The coupling parameter α does *not* cause this asymmetry. Removing α (setting α = 0) worsens both global deviation (std ~7.6% → ~17%) and mode asymmetry (~0.424% → ~0.457%), while the canonical α ≈ 3.2278 partially compensates. The Sinkhorn warp operates at the hexagon level (pairing one mode-0 and one mode-1 triangle per cell), so it cannot rebalance within-hexagon mode areas; the asymmetry carries through to the warped result at slightly reduced magnitude. For practical DGGS use the residual is negligible at L5 and finer.

### Authalic warp
The warp corrects area deviation from the base octahedral projection.

**Derivation:**
1. Problem statement: find displacement field *w* in b_oct space such that projected cell areas approximate uniform ellipsoidal area
2. Theoretical framing: optimal transport problem; continuous formulation is the Monge-Ampère equation
3. Practical solution: Sinkhorn iteration on L4 triangle vertices (~1 day compute, runs unattended)
4. Source: regular L4 grid in b_raw; Target: Sinkhorn-optimised positions
5. Pre-computed; stored as `l4_boct_warp_data.npz`

**Forward warp:**
- CloughTocher2DInterpolator on displacement field (not absolute positions — better numerical conditioning)
- C1 continuous
- Ghost-row padding mirrors equatorial points across the mode-0/mode-1 seam for boundary stability
- NearestND fallback for points outside convex hull (seam/boundary edge cases)

**Inverse warp:**
- Linear interpolator for initial guess
- Newton-Raphson polish (25 iterations, tolerance 1e-14 ≈ 0.25mm)
- Bidirectional precision: forward and inverse both achieve ~7nm

**Why quasi-authalic rather than strictly authalic:**

A strictly authalic projection has Jacobian determinant = 1 everywhere — area exactly preserved. But the Jacobian *matrix* can still be far from identity: principal scale factors can be very unequal (e.g. 0.5 × 2.0 = 1), producing severe shear and cell elongation even at exact equal-area. Near the limits of some projections this becomes extreme (Tissot indicatrix remains equal-area but highly eccentric).

The Sinkhorn/OT derivation avoids this. Optimal transport minimises *displacement* while achieving area equalisation — the minimum-displacement objective is an implicit regulariser against shear. It will not introduce large shape distortion just to hit det(J) = 1 exactly. Snyder's octahedral equal-area projection achieves strict det(J) = 1 analytically but with no constraint on the Jacobian's eccentricity.

The result: Hex9's quasi-authalic warp trades a small area residual (p99 < 0.005%; balanced ±4% tail at the six vertex singularities) for a smooth, low-shear displacement field whose distortion is bounded and characterised. For DGGS use cases:
- **Point binning / density**: the ~0.066% area residual is negligible; strict authalic confers no practical advantage
- **Distance / shape operations within a cell**: low-shear cells are preferable to elongated equal-area cells

OGC Topic 21 permits quasi-equal-area for exactly this reason. The quasi-authalic choice is principled, not a compromise forced by implementation difficulty.

**Shape regularity as a tunable parameter.** The gradient warp exposes a `SHAPE_DAMPEN` parameter that trades area equality against hexagon shape regularity. `SHAPE_DAMPEN = 0` gives the quasi-authalic optimum; higher values penalise triangles deviating from equilateral (= hexagon angles deviating from 120°), at a small area cost. The goal is not perfect regular hexagons — it is avoiding degenerate elongation ("pencil" cells) near seams and vertex regions, where strict authalicity alone permits severe shape distortion. The practical target is convex hexagons with interior angles bounded well away from 0° and 180°. Empirically, this constraint appears to cost little in area deviation while producing cells usable for distance and shape operations across the full grid.

**Experimental history** (prior approaches, documented under `experimental/`):
- Monge-Ampère PDE (continuous OT — correct formulation but intractable boundary conditions)
- Bernstein polynomials
- Authalic march (geodesic-based iterative approach)
- LinearNDInterpolator (insufficient smoothness)
- Sinkhorn + CloughTocher selected for accuracy, stability, and invertibility

---

## Addressing

Two complementary forms, both efficiently composed.

### UUID (128-bit)
32 nibbles = 128 bits, structured as:
- **Nibbles 0–30**: the L0..L30 hierarchy path — 30 address digits, one per layer
- **Nibble 31**: key tail = `(p_c2 << 1) | r_mo` — 6-value terminal state encoding

Self-contained for all spatial operations:
- **Spatial indexing and binning**: no decoding required — direct nibble comparison
- **Hierarchy traversal**: truncate to N nibbles → L(N-1) ancestor; prefix comparison gives parent/child/sibling
- **Database compatibility**: standard 128-bit UUID format — drops into any database, PostGIS, or API naturally

At L30 the cell size is sub-atomic — 128 bits is effectively infinite precision for any practical application.

### (UUID, adr_byte) pair — reversible
UUID + 1 companion byte (5 bits used):
- **bit 4**: p_mo — parent net_mode of terminal region
- **bits 3–0**: h — terminal region id (0..11)

The companion byte carries the sub-region state the UUID alone cannot recover. The pair enables exact round-trip to lat/lon — full CRS precision. The UUID alone is sufficient for DGGS operations; the pair is required for CRS operations.

**The separation is deliberate:** it cleanly delineates spatial indexing needs (UUID only) from exact coordinate recovery needs (UUID + adr_byte). Most applications never need the companion byte.

### Split x_cells and x_list composability

**Why c_list / t_list / d_list are left-to-right composable.**
Each of these address spaces forms a strict single-parent tree: every cell has exactly one parent, and the mapping `(parent, digit) → child` is a well-defined function. Reading left-to-right — apply digit d₁ to the root, yielding a level-1 cell; apply d₂ to that cell, yielding level-2; and so on — produces a unique cell at each step. The sequences are prefix-sortable: spatial proximity is reflected in shared prefixes.

**Why x_list is not.**
3 of the 9 child `x_cells` at each layer straddle the d_cell boundary and have **two valid parent x_cells** — one mode-0, one mode-1. These are exactly the cells whose high ternary trit is 2 (x_digs 6, 7, 8) — their high trit signals the split directly: trit 0 = mode-0 interior, trit 1 = mode-1 interior, trit 2 = straddles the boundary. Exactly 3 split cells arise at every level because the d_cell boundary is a single edge of the parent and the tiling places exactly 3 child half-hexes along any such edge.

A split x_cell C has x_dig = k in parent P (mode-0) and x_dig = k' ≠ k in parent P' (mode-1). The mode of C — which determines canonical parentage — also determines how C's own children are enumerated at the next level down. Mode propagates recursively: the mode of a cell depends on the mode of its canonical parent, which depends on its grandparent's mode, and so on. This chain only terminates when the canonical parent assignment at every level is known.

**The high-trit / split connection.** The fact that high trit = 2 marks the split cells is not coincidental — it is the same diploid structure in the digit. Trit 0 and trit 1 correspond to the two d_cell halves (mode-0 and mode-1) of each x_cell; trit 2 is the boundary between them. High trit 2 in the x_dig therefore carries the same information as "this cell requires a second parent to be disambiguated" — it is the digit-level signature of the seam.

**Consequence**: decoding x_list left-to-right requires knowing the canonical parent mode at each level before interpreting the next digit. But canonical parent mode is only recoverable starting from the terminal cell, working upward. Left-to-right decoding therefore requires resolving the entire right end of the sequence first — it cannot proceed purely sequentially from the left.

**The tail is the minimal fix.** `x_adr` = `x_list` + one appended `t_dig` (the "would-be next child" — a virtual one-step extension). From the tail, c2 and parent mode of the terminal cell are directly recoverable. With the terminal mode known, the canonical parent at level L is determined; from that, the canonical parent at L−1; and so on upward. The x_list is decoded right-to-left in one pass.

**Convention**: the mode-0 parent is canonical for split cells. This is a deliberate convention, not a mathematical uniqueness failure — analogous to a point on a timezone boundary. Either parent is geometrically valid; mode-0 is chosen for consistency.

**The tail is metadata only.** It must not be used for offset calculation, coordinate reconstruction, polygon generation, or any geometric operation. Only the x_list digits participate in geometry. Using the tail as an address step computes coordinates at layer L+1 and silently corrupts results.

In the UUID: nibbles 0–30 = `x_list`; nibble 31 = tail (the "key tail"). The `adr_byte = (p_c2 << 1) | r_mo` is a packed re-encoding of what the tail carries.

*For reviewers who probe ambiguity: the `t_adr` and `d_adr` trees are the unambiguous underlying structures; the x_adr tail is the minimal addition that makes the hexagonal address equally unambiguous. The tail adds one digit (4 bits in the UUID); no other addition suffices.*

**Concrete example — London and the Prime Meridian.**

The PM is a c2 boundary in this region, making London a natural illustration of split-cell logic.

- **L1**: hex `43` — mainland UK (excluding Shetland), Western Europe to northern Spain and northern Italy.
- **L2–L3**: the PM splits southern England into `435` and `434`, with Scotland and parts of NI in `432`.
- **L4 prefixes**: East England is `4348` (canonical, mode-0); the same cells are reachable as `4358` via the mode-1 parent. SW England (excl. Cornwall) is `4352`. Kent, Dover, NW France, W Belgium are `4342`.

Central London straddles the c2 boundary (the PM). At L4, Greenwich-area cells sit in three adjacent hexagons with non-adjacent prefixes:

| L4 address | Location |
|---|---|
| `43483` | Corner of north London |
| `43486` | East of Greenwich |
| `43527` | West of Greenwich |

`43527` has jumped lineage — it is in `4352` (SW England) despite being geographically adjacent to `43486` (`4348`, East England). The PM boundary runs between them.

At L5, this resolves further:

| L5 address | Location |
|---|---|
| `434868` | Central London / Greenwich |
| `434836` | Northeast |
| `435875` | Heathrow (west) |
| `435877` | Northwest |
| `435272` | Southwest |

At first glance the prefix jumps (`4348...` ↔ `4352...`) across adjacent cells appear arbitrary. They are not: x_digs 6, 7, 8 are the split x_digs — recognisable by their high ternary trit (6=2×3+0, 7=2×3+1, 8=2×3+2; trit 2 = straddles the d_cell boundary). The mode-0 parent is canonical for these cells by convention. Following the 678 mode-0 canonical rule consistently produces these addresses. The "jump" is the visible signature of the split cell — geographically adjacent cells on opposite sides of the d_cell boundary are in different canonical lineages, as expected.

**Prefix cutting: layer vs canonical ancestry.**
The layer ID is precisely that — `len(x_list) - 1` gives the resolution level (L0 is the root, so a single-digit address is L0, two digits is L1, etc.), and nothing more. Prefix cutting correctly identifies which layer you are at; it makes no guarantee about canonical lineage. These are orthogonal properties and conflating them is the source of subtle bugs.

The failure mode is most visible in **binning**: aggregating a dataset to a coarser layer by prefix-cutting to length L will silently produce two separate bins for the same cell when non-canonical addresses are present — `4348` and `4358` bin separately despite being the same L4 cell. No error is raised; the data is just split. The correct approach is to derive the canonical layer-L ancestor via the tail before binning — which is precisely what the tail is for. Tail-based canonical derivation is the right tool; prefix cutting is only correct for pure layer identification.

Cutting a prefix from an x_adr always identifies a valid ancestor at the correct layer — `4358` is genuinely the layer-4 ancestor of `435875`. But the prefix cut does not guarantee a *canonical* ancestor: the canonical form of that ancestor is `4348`, not `4358`. Prefix cutting lands in whichever lineage the original address was encoded in; canonical ancestry requires recanonicalisation.

A `recanonicalise()` operation converts a non-canonical address to its canonical (mode-0) form. It is non-trivial because the effect cascades: if an ancestor at level L is in a mode-1 lineage, all its descendants inherit mode-1 digit assignments (their 6/7/8 splits were enumerated under mode-1). Recanonicalising a single level does not fix the descendants — the operation must rewrite digits from the topmost non-canonical level downward, level by level. The tail identifies whether the address is in a non-canonical state and where the cascade begins.

### Prefix sortability
`t_list` and `d_list` are prefix-sortable — spatial locality is encoded directly in the address prefix. Enables efficient spatial range queries and database indexing without secondary spatial indices. `x_list` is not prefix-sortable in the same way due to the right-to-left dependency, but `x_adr` (with tail) enables equivalent operations once the tail is used to resolve split cells.

### Hex ↔ half-hex address conversion
Outstanding work. The conversion involves retaining the mode/parity value of ancestral hexagons through the hierarchy path. Related to the `_m_c2_hx_v2025` table and the tail encoding in the UUID.

*TODO: complete and formalise the hex ↔ half-hex address conversion.*

### OGC/ISO terminology mapping
To be resolved before submission — mapping Hex9 terminology to OGC/ISO standard terms:

| Hex9 term | OGC/ISO term |
|---|---|
| layer | refinement level |
| aperture | 9 (each hex → 9 children; consequence of ×3 scaling) |
| mode | face orientation / parity |
| region | fundamental region |
| hex | cell |
| half-hex | rhombic sub-cell |
| b_oct / b_raw | internal CRS / intermediate coordinate reference system |
| c2 | sub-face identifier |

Hex9 aperture = 9 (ternary²). Compare: H3 ≈ 7, S2 = 4, HEALPix = 4.

**Important distinction — not centred-aperture-9.** In a standard centred aperture-9 hexagonal grid, one child hexagon sits at the parent's centroid — the subdivision is concentric. Hex9 is *not* this. The parent hexagon comprises two half-hexes (mode-0 and mode-1); the parent centroid falls at their shared long-edge boundary, not inside any child. The 9 children are arranged in the half-hex staircase pattern, half-shifted so the parent centre lies at a child boundary. There is no central child. This distinction should be stated explicitly wherever aperture is compared across DGGS systems.

The half-shifted arrangement looks unwieldy compared to centred-aperture-9, but this appearance is misleading. A centred-aperture-9 hexagonal grid requires a fundamental triangle of area 1/3 of Hex9's half-hex in order to describe its own cell boundaries — it must descend one triangulation level below the cell to reach its geometric primitive. Hex9 operates directly at the half-hex level: the half-hex *is* the primitive, and no finer subdivision is needed. The half-shift is not a cost; it is what allows the grid to work at the coarsest sufficient granularity.

The half-shifted arrangement carries two further concrete payoffs:

1. **Scale invariance of the fundamental cell.** The half-hex is the primitive cell at *every* refinement layer without exception. The grid is self-similar in the strongest sense: no new cell types, no alternating patterns, no additional edges introduced at finer resolutions. A layer-1 half-hex and a layer-10 half-hex are geometrically identical objects, differing only in scale.

2. **Cell membership by linear inequality.** In triangular (barycentric) coordinates, the half-hex boundary is a staircase of axis-aligned edges. Testing whether a point belongs to a given half-hex reduces to a small, fixed set of integer linear inequalities — no trigonometry, no floating-point polygon intersection, no edge-case handling. The beam search in `h9_boct.cpp` exploits this directly.

*TODO: review OGC Topic 21 and ISO 19111 terminology lists systematically and complete the mapping table.*

### Relationship to the dual CRS/DGGS claim
- UUID truncated to L digits → DGGS cell at layer L (OGC Topic 21)
- (UUID, adr_byte) pair at full depth → exact point on WGS84 ellipsoid (ISO 19111)
- The same 128-bit value serves both roles — the companion byte is the bridge

*Implementation: `hhg9/h9/uuid_address.py` — `h9_encode`, `h9_decode`, `h9_bin`*

---

## The Native Projection

**b_oct is the defining coordinate space.** In b_oct, every Hex9 hexagonal cell is a congruent regular polygon with exact coordinates. This is not incidental — it is the mathematical foundation of the system.

Any shape variation visible when Hex9 cells are rendered in Mercator, geographic (lat/lon), or any other conventional CRS is a property of the reprojection, not of the cells. The distortion belongs to the map, not to the hexagons. Densification of cell edges for display is reprojection sampling; the canonical cell polygon is always the regular hexagon in b_oct.

**Contrast with Sahr (DGGRID/ISEA):** DGGRID uses the Icosahedral Snyder Equal Area projection, but hexagons are defined by their geographic boundaries on the sphere — they are intrinsically curved. There is no planar coordinate space in which a DGGRID hexagon is a regular polygon with exact coordinates; the projection is implicit in the construction. Hex9 inverts this: the regular polygon in b_oct is the *definition*; the geographic boundary is the *derived* quantity. The projection is explicit, named, and invertible.

This distinction is what makes Hex9 a proper CRS rather than only a DGGS. b_oct is a legitimate projection in its own right — not an intermediate representation. Conventional geographic projections are the derived ones when working with Hex9 data.

*For the paper:* *"Hex9 defines a native coordinate space (b_oct) in which all hexagonal cells are congruent regular polygons. Apparent shape variation in external projections (Mercator, geographic) reflects the reprojection, not the cells — the distortion is on the map. This contrasts with prior hexagonal DGGS work (e.g. DGGRID/ISEA) in which cells are defined by geographic boundaries on the sphere and have no planar coordinate representation in which they are regular."*

---

## Projection Independence

The Hex9 grid system (topology, ternary hierarchy, addressing, diploid structure) is **projection-independent**. It will work on any octahedral projection — the warp is a separable, interchangeable component.

**Projections known to work:**
- Hex9 Sinkhorn authalic warp (default — quasi-EA, ~0.2% MAE, chosen for binning/spatial indexing utility)
- Lee Conformal — implemented and working well; demonstrates clean separability
- Snyder Octahedral Equal-Area — the analytical equal-area octahedral projection; attempted but insufficient implementation detail available in the literature to reproduce fully. Hex9's Sinkhorn warp achieves the same goal numerically.

**Why quasi-EA as default:**
Equal-area (or quasi-EA) is critical for the primary use cases of binning and spatial indexing:
- Cells represent consistent areas → point counts are geographically comparable
- Density analysis is meaningful across the globe
- H3 and S2 are not equal-area, which introduces systematic bias in density estimation

**For the paper:** treat the projection as a separable component in a short dedicated section — not the main contribution, but important context. Note that the grid is usable with other projections for applications where conformality or other properties are preferred.

## Why Hexagonal Tiling

The advantages of hexagonal grids over square or triangular grids for spatial indexing are well-established in the literature. A short paragraph with citations suffices — this is not a contribution of Hex9 but context for it.

Key properties to cite:
- **Isotropy**: equal distance to all 6 neighbours (squares have √2 diagonal artefact)
- **Consistent neighbourhood**: all neighbours share an edge (not just a corner)
- **Area efficiency**: hexagons are the most efficient polygon for tiling (closest to circle)
- **No directional bias** in spread/growth/transit operations

*TODO: compile citation list — Sahr, White & Kimerling (2003); Tsai (2000); others on hexagonal grids for spatial analysis.*

---

## Ellipsoid Generality

- The addressing scheme, hierarchy, and interpolation machinery are ellipsoid-agnostic
- Only the Sinkhorn warp is ellipsoid-specific (computed against geodesic areas of the target body)
- Recomputation: ~1 day unattended compute per ellipsoid
- **WGS84 and GRS80** share a warp file in practice — ellipsoidal difference (0.1mm semi-major axis, 10th s.f. flattening) is negligible relative to the <1% warp residual. *To verify empirically before submission.*

**Practical warp files needed:**
| Ellipsoid | Use |
|---|---|
| WGS84 / GRS80 | Global GIS, GPS |
| Bessel 1841 | European legacy |
| Clarke 1866 | North American legacy |

**Planetary extension:**
The framework generalises to any body for which a reference ellipsoid is defined. Only the Sinkhorn warp requires recomputation against the target body's geodesic areas. Relevant for lunar, Martian, and other planetary mapping programmes (NASA/ESA IAU reference ellipsoids). Highly irregular bodies (Phobos, small asteroids) are out of scope — no DGGS handles non-ellipsoidal bodies well currently.

---

## Implementation

- Python reference implementation: `hhg9` package
- PROJ plugin: `h9_boct.cpp` — C++ port of hierarchy walk, ~30nm accuracy (vs ~7nm Python; difference is N-R iteration depth in inverse warp)
- PostGIS extension: planned C extension wrapping `h9_boct.cpp`; Python excluded (untrusted language in most PostGIS environments)

---

## Suggested Paper Structure

1. Introduction — motivation, DGGS landscape, gap being filled, design philosophy
2. Mathematical foundations
   - Diploid / crystallographic structure
   - Half-hexagon tiling theorem
   - Uniqueness of the fractal-capable 1:9 arrangement (CSAT proof, Figure 1)
   - The fractal cascade and dual CRS/DGGS interpretation
3. Construction — octahedral base, ternary subdivision, addressing scheme
4. Authalic warp — OT formulation, Sinkhorn derivation, CT interpolation, N-R inverse
5. Properties — area equivalence, hierarchy, isotropy, dual CRS/DGGS claim
6. Comparison — table + one paragraph vs H3/S2/HEALPix
7. Implementation — Python, PROJ plugin, PostGIS
8. Applications — binning, display, PostGIS queries
9. Generalisation — ellipsoid extension, planetary bodies
10. Conclusion + future work (fully authalic variant, L8+ bounded construction)

---

## Suggested Venues

- *International Journal of Geographical Information Science (IJGIS)*
- *Computers & Geosciences*
- ACM SIGSPATIAL (conference — useful for early feedback before journal submission)

---

## OGC Submission Path

1. Publish paper (establishes priority and peer-reviewed reference)
2. Approach OGC DGGS Standards Working Group
3. Target: OGC Community Standard (lighter-weight, faster than full standard)
4. PostGIS extension as evidence of working implementation

---

## Open Questions / To Do Before Submission

- [ ] Verify GRS80 warp residual empirically (expected: negligible vs <1% bound)
- [x] Finalise warp file — production file is `WGS84_l5_warp_data.npz`, characterisation complete
  - re-measured 2026-06-10: p99 0.0052%, min −6.93%/max +0.078%, log-ratio σ 1.89×10⁻⁴; edge anisotropy p50 32.8%, CV p50 12.0% (earlier figures superseded)
- [ ] Measure precise angular radius of elevated-deviation zone (one-time geometric measurement, layer-independent)
  - PostGIS: `h9_vertex_zone(geom)` — angular distance check against 6 fixed octahedral vertex points
- [ ] Formalise the CRS/DGGS limit claim mathematically
- [ ] Characterise C1 continuity of warp at octant boundaries explicitly
- [ ] Refactor ellipsoid-specific code to parameterise reference ellipsoid
  - `projections/ak_octahedral.py` — AKOctahedralEllipsoid has hardcoded WGS84 values; refactor into ellipsoid parameter table
  - `h9/grid.py` — WGS84 constants present, likely test-only; verify and isolate
- [ ] Complete PostGIS C extension (strengthens OGC submission)
- [ ] Decide on journal vs conference first
- [ ] Define correct 27-cell space for 9-piece half-hexagon tiling in `halfhex.py` and run to get solution count + identify fractal-capable arrangement

---

## Graticule Alignment

The octant face subtends exactly 90° in both latitude and longitude extent.
360 = 2³ × 3² × 5 carries two factors of 3, so trisection of the degree
system is at least as natural as bisection — and in practice more useful
cartographically.  Bisection of 90° yields 45°, 22.5°, 11.25° (non-standard
graticule values); trisection yields 30°, 10° — both standard.

At a 10° graticule spacing this gives 9 intervals per octant face — a clean
integer fit producing visible alignment between the graticule lines and the
octant boundaries.  The Hex9 ternary hierarchy reinforces this: L1 hexes span
3 graticule intervals (30°), L2 hexes span 1 (10°).  L3 breaks at ≈ 3.33°
in decimal degrees, but the alignment extends further in arc-minutes: the
octant is 5,400′, which trisects cleanly four times (1800′ → 600′ → 200′ →
66.67′) before breaking — deeper than bisection achieves.

The 30° Tissot indicatrix grid aligns with every third 10° graticule line and
with the L1 hex scale, making a 10° graticule the natural companion grid for
Hex9 visualisations.  This is a consequence of the shared factor of 9 between
base-360 and the Hex9 ternary subdivision — not a design choice in the
projection.

---

## Known Display Limitations

**Antimeridian-crossing polygons in QGIS**: hexes that straddle 180° longitude render incorrectly in geographic/Mercator projections (drawn across the full map width). The data is correct — this is a QGIS rendering issue. Workaround: apply feature filter expression `(x_max(@geometry) - x_min(@geometry)) < 180`. Affected hexes are a small fixed set at each layer (descendants of the ~2-4 L0 hexes straddling 180°). Not an issue in PostGIS or spherical geometry contexts.

---

*Notes compiled 2026-04-14. Raw — needs structure and refinement.*
*Terminology: see `glossary.md` for the canonical c/t/d/x grid taxonomy, cell-vs-digit distinction, and OGC/ISO mapping.*


---

## Notes

**Main theorem** (not an axiom): The system defined by Axioms 1–9 is the primary
discrete reference representation of the ellipsoid; all other DGGS representations
are derivable as maps from this substrate. This is the conclusion the paper
establishes, not a starting assumption.

**Motivating principle** (not an axiom): All admissible constructions minimise
auxiliary coordinate choices, hidden orientation conventions, and transformation
depth between geometry and interpretation. This principle motivates the axiom set
but does not constrain it independently.

