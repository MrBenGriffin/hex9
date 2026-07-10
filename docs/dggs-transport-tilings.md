# Transport tilings — when does a DGGH admit a fully-nested substructure?

Status: working note, 2026-07-10 (Ben + assistant). Companion to
`dggs-two-hierarchies-note.md` and the OGC API-DGGS issue
[#108](https://github.com/opengeospatial/ogcapi-discrete-global-grid-systems/issues/108)
thread; the question was prompted by Sahr (2011, fig. 4) and Jérôme
St-Louis's engagement on #108. Everything below is planar unless marked
spherical; claims are tagged VERIFIED / TO VERIFY / CONJECTURE.

## 0. Scope

Everything in this note quantifies over systems whose refinement is
defined on a **planar lattice per polyhedral face** (Eisenstein ℤ[ω]
for tri/hex, Gaussian ℤ[i] for quads), transported to the sphere by a
fixed per-face projection — which covers every system named here, and
is zone-shape-agnostic (the lemmas are about the lattice map, so
rhombus-zone systems like DGGAL's RI7H are included). Two rings sit
outside: planar-but-non-lattice substitution systems (Class S, §3 —
definitions apply, the arithmetic doesn't), and sphere-native
constructions (Voronoi/SCVT hierarchies, statistical apertures) — to
which none of these claims reach. Available lattice apertures are the
values of the lattice's norm form: **Loeschian numbers** a²+ab+b²
(1,3,4,7,9,12,13,…) on the triangular lattice, sums of two squares
(1,2,4,5,8,9,10,…) on the square lattice. Corollaries: 7 is Loeschian
but not a sum of two squares, so **no aperture-7 quad lattice system
exists** — but aperture-7 *triangulations* do (same lattice as the
hexagons, zone-shape-agnostic): no per-level similar subdivision
(equilateral triangles are rep-n² only), but with the alternating
chirality word the triangulation nests in ITSELF at stride 2 (m = 1 —
triangles tile triangles, 49 per coarse). That object is exactly the
centre-triangle carrier of §3b: there is an A7 triangulation hiding
under H3, and unlike H3, it nests.

## 1. Definition

A **transport** for a DGGH (zones Z_L per level, aperture a) is a family
of tilings T_L, one per level, with a finite prototile set, satisfying:

- **T1 — nesting.** Every transport cell t ∈ T_L is *exactly* the union
  of its children in T_{L+1}. The transport hierarchy is fully-nested:
  ancestry is transitive, exact, computable by identifier truncation.
- **T2 — expression.** Every zone z ∈ Z_L is *exactly* the union of a
  constant number m of transport cells of T_L.
- **T3 — compatibility.** The transport aperture equals a (this follows
  from T1+T2 with m constant: |t| = |z|/m at every level).

Notes. T need **not** be a rep-tile: T1 asks for exact nesting of
tilings, not self-similarity — congruent-but-rotated children, or a
small set of prototiles alternating by level, are admissible. "Finite
prototile set" (finite type) is the load-bearing restriction. Two
classes must be distinguished (Ben, 2026-07-10):

- **Class L — lattice-aligned**: every transport cell sits in the
  common lattice frame of its level (Hex9 half-hexes, H3 stride-2
  centre-triangles). The multiplier-ring criterion of §3 applies.
- **Class S — similarity-composite (substitution)**: tiles may be
  individually rotated within a level, pinwheel-style. The §3 criterion
  is SILENT here — see the pinwheel caveat in §3.

Consequences (these are what #108 calls the *documented ownership
convention* made concrete):

- Ownership = resolve zone → transport cell once (the only conventional
  step, choosing one of its m cells), truncate within the transport,
  express back. All non-transitivity lives in the resolution step.
- Owned-descendant counts are exactly a^d; owned sets partition; the
  owned ancestor's interior intersects the zone (a genuine #106
  sub-zone).
- Straddler weights are exact rationals k/m (Hex9: m = 2 ⇒ weights 1
  and ½, machine-verified 2026-07-10).

## 2. Instances

| System (aperture) | Transport | m | Status |
|---|---|---|---|
| S2, HEALPix, rHEALPix, ISEA4T/9R (quad/9-rhombus) | the zones themselves | 1 | trivially VERIFIED (fully-nested) |
| Hex9 (9) | rep-9 half-hexagon (d_cell) | 2 | **VERIFIED** — exact integer lattice, `h9_cell_ancestor`/`h9_descendants` |
| ISEA4H-class (4) | half-hexagon (triamond trapezoid) | 2 | **VERIFIED planar 2026-07-10** (machine, transport_check.py): the trapezoid is rep-4 — 4 direct-similar half-scale copies, rotations {0,120,180,240}, exact to 1e-15 at depth 3. Split assignment propagates by role (centre child inherits the parent split; edge child splits along the shared coarse edge — the parent diameter rides child-grid edges via the radial-third-edge property, so no sibling is sliced). Spherical caveat in §4 |
| ISEA3H-class (3) | pentagonal tiling at 1/3 hexagon area | 3 | TO VERIFY (Sahr 2011 fig. 4, planar); see §3 note on level parity |
| H3, ISEA7H (7) | none lattice-aligned at any stride (THEOREM, §3) — but H3 has an exact stride-2 centre-triangle transport (§3b, m = 6), and a Class-S (pinwheel-type) per-level carrier is OPEN | — | Class L settled; Class S open; ownership always exact via lattice arithmetic |
| A5 (pentagon, 4) | unknown | — | OPEN — worth asking Felix Palmer |

## 3. The rotation-commensurability criterion

Hexagonal refinements are multiplications by an Eisenstein integer κ
with norm a; each level rotates the lattice by α = arg(κ).

- a = 3: κ = 1−ω, α = 30° — **commensurable** with the 60° lattice
  symmetry (directions cycle with period 2 levels).
- a = 4: κ = 2, α = 0° — commensurable.
- a = 9: κ = (1−ω)², α = 60° ≡ 0° — commensurable (Hex9).
- a = 7: κ = 3+ω, α = arctan(√3/5) ≈ 19.107° — **incommensurable**
  (an irrational multiple of π; standard for Eisenstein integers of
  prime norm ≠ 3 — TO FORMALIZE with a proper citation).

**Classification lemma (aperture 7).** Norm-7 Eisenstein integers are
exactly the units times 3+ω or 3+ω̄, so there are precisely two
aperture-7 refinements of a hex lattice — the two chiralities. Every
lattice aperture-7 system is therefore a *word* in {κ, κ̄} (H3
alternates; a forward-only system repeats one letter) plus labelling
and assembly conventions. Consequences: index-space ownership methods
(GBT-style correction sets — cf. W. Renoud's Signal analysis,
2026-07-10: green/red difference sets, 6-fold equivariant, per-depth
tables, pentagon = deleted digit direction, base-zone carry-rotation =
cone-point holonomy) are word-agnostic and hence apply to any lattice
aperture-7 system; but the *tables* are word-dependent — only the
alternating word gets the even-depth collapse (the stride-2 carrier),
a forward-only word has genuinely distinct tables at every depth and
provably never a shortcut.

**Theorem (aperture-7, Class L).** No lattice-aligned transport exists
for forward-only a = 7 at any stride: κⁿ = real×unit would make (3+ω)ⁿ
and (3+ω̄)ⁿ associates — impossible for distinct primes in a UFD. This
part is settled.

**Class S is genuinely open — the pinwheel caveat.** The earlier
direction-accumulation sketch (finitely many new edge directions per
level ⇒ contradiction) is REFUTED as a general argument by the
Conway–Radin pinwheel: one 1:2:√5 right triangle subdividing exactly
into 5 similar copies, one rotated by arctan(½) = arg(2+i) — an
irrational multiple of π, with tile orientations becoming dense under
refinement. Exact nesting, finite type, irrational rotation: the
pinwheel is the forward-only aperture-5 rotating quad made to work by
dropping lattice alignment. It satisfies T1-style nesting but is not
known to satisfy T2 for any actual DGGS's zones — so whether a
"hexagonal pinwheel" for √7 exists (cf. Sadun's generalized pinwheels)
is an open question, and the aperture-7 impossibility CONJECTURE now
lives entirely in Class S. Consistent evidence against, but not proof:
H3's descendant limit shape is the fractal Gosper island, cf.
St-Louis's draft "Area under the fractal curve" (#108 attachment).
Note also: a Class-S carrier, if one exists, still supports hierarchical
addressing (substitution tilings have canonical addresses), so it would
be *useful*, not just a curiosity — it could rescue odd-depth exact
ownership for H3.

Conversely, commensurability is what lets a = 3, 4, 9 have polygonal
carriers at all.

**Generalization (exactly as far as the crystallographic restriction
allows).** The same argument runs on any plane lattice whose multiplier
ring is a UFD with a finite unit group — and there are precisely two
interesting ones: triangular/hexagonal → Eisenstein ℤ[ω] (units = 60°
rotations), and square/quad → Gaussian ℤ[i] (units = 90° rotations).
For quads: aligned apertures are exactly a = 2^s·m² (2 ramifies:
2 = −i(1+i)²); S2/HEALPix are a = 4 = 2²; an aperture-2 diamond
quadtree (κ = 1+i, 45°/level) aligns at stride 2 automatically —
the Gaussian twin of ISEA3H; and a forward-only rotating-quad system
(e.g. a = 5, κ = 2+i, arctan(½)/level) never aligns at any stride, by
the same distinct-conjugate-primes proof. Pentagons (A5) are the
principled exception: no plane lattice has 5-fold symmetry, so there is
no multiplier ring and the criterion is silent — A5's carrier question
is genuinely different, not merely unchecked. Note the criterion's
silence cuts both ways: A5's unequal-pentagon subdivision is already
composite in flavour, so if A5 has a carrier it will be Class S — the
question is open, not closed.

## 3b. The parity escape hatch — stride transports (Ben, 2026-07-10)

The criterion above is per-level; Ben's observation: apply it **per
stride**. A DGGH with per-level multipliers κ₁, κ₂, … admits an aligned
frame at stride n iff the composite κ₁···κₙ is an integer times a unit
(net rotation ≡ 0 mod 60°). At an aligned stride the transport machinery
comes back to life, restricted to matching parities (relative depth
d ≡ 0 mod n — for n = 2, Ben's `a%2 == d%2`).

Minimal aligned strides of the systems at hand:

| System | multipliers | minimal stride | stride scale m | why |
|---|---|---|---|---|
| ISEA4H | 2, 2, … | 1 | 2 | aligned every level |
| Hex9 | 3, 3, … | 1 | 3 | aligned every level |
| ISEA3H | 1−ω, 1−ω, … | 2 | 3 | **automatic**: (1−ω)² = −3ω = 3·unit — the ramified prime squares to alignment, no alternation needed |
| H3 | 3+ω, 3+ω̄, alternating | 2 | 7 | conjugate alternation: κκ̄ = 7. VERIFIED empirically 2026-07-10: H3 grid orientations alternate ~22.0°/~41.9°, same-parity resolutions aligned |
| forward-only aperture 7 (Ben's unpublished H3 analogue, dropped for exactly this failure) | 3+ω, 3+ω, … | **none, at any n** | — | κⁿ = real·unit would make (3+ω)ⁿ, (3+ω̄)ⁿ associates — impossible in a UFD (distinct primes) |
| ISEA7H (DGGAL) | 3+ω, 3+ω̄, alternating | 2 | 7 | VERIFIED 2026-07-10: orientations alternate ~38.8°/58.5° (measured), and the stride-2 census holds (49 fine centre-triangles per coarse, dggs_transport.py) — jerstlouis implemented it H3-style |
| IGEO7 / ISEA7H (DGGRID) | ? | ? | ? | TO CHECK: if DGGRID compounds the rotation instead of alternating, it has no hatch at any stride (same proof as forward-only) — it would be the first published forward-only system |

**The centre-triangle transport.** At any aligned stride (scale m,
centre-children preserved) there is a universal candidate: cut every
hexagon into its 6 centre-triangles. Then:

- T1 holds exactly: coarse and fine triangle tilings are the standard
  triangular tiling at scales s and s/m, same orientation, sharing the
  anchor centre; the coarse vertex lattice is m × the fine one, so each
  coarse triangle is exactly m² fine triangles. (Corner offsets satisfy
  3δ ∈ centre-lattice, so mδ stays in the vertex set for every m —
  corner classes may swap when m ≢ 1 mod 3, which is harmless.)
- Ownership is canonical at the coarse end: the triangle-tiling vertices
  3-colour into {centres, corner-A, corner-B} and every triangle has
  exactly one vertex of each colour — hence exactly one owning hexagon.
  Only the leaf-side resolution (which of a zone's 6 triangles
  represents it) is conventional, as always.
- Icosahedral vertex zones are 5 triangles — integral (contrast the
  m = 3 pentagonal claim's 5m/6 = 2.5 in §4).

For H3 this gives: **no per-level transport (§3 conjecture stands), but
an exact stride-2 triangle transport** — in #108 terms, H3 could serve
`owned-only` at even relative depths with exact counts. For ISEA3H,
stride 2 gives an aligned aperture-9 frame — Hex9's setting — so the
Sahr pentagonal tiling, the half-hex story and this section may be three
views of one structure. First check (2026-07-10, censal + visual):
dggs_transport.py's ISEA3H panel measures exactly 9 fine centre-triangles
per coarse centre-triangle at stride 2 over DGGAL geometry — the
stride-2 nesting holds at first contact (float-geometry; full
transport_check.py treatment still queued).

The load-bearing unknown was count exactness: owned counts of m^(2d) per
hexagon can't hold per coarse *triangle* (census m^(2d)/6 non-integral),
so it must emerge from the 6-fold assembly. **VERIFIED for H3
(empirically, 2026-07-10)**: with the leaf convention "representative
triangle = centre + first boundary edge" and owner =
latlng_to_cell(representative-triangle centroid, coarse res) — seven
lines of public H3 API — owned counts are exactly 49 at relative depth 2
(40/40 random non-pentagon anchors) and exactly 2401 at relative depth 4
(8/8 anchors). Float-geometry verification; the exact-arithmetic (Eisenstein)
proof of the equidistribution is still open, as is the pentagon-anchor
case and behaviour within a face-edge/vertex neighbourhood. Note the
owner rule is the H3 analogue of Hex9's h9_cell_ancestor doctrine: a
single deep re-bin of a representative interior point — the triangle
transport is what makes it exact at even relative depths.

**Odd depths by composition (Ben's practical reading, 2026-07-10).**
owner_odd(z, La) = owner_even(cell_to_parent(z), La): one native
lineage step Ld→Ld−1, then the exact even stride. Counts stay exactly
7^d (each Ld−1 cell has exactly 7 lineage children, disjoint across
parents) and the partition property is inherited; the single fuzzy step
cannot compound because the deep part is exact. Measured: owned = 343 =
7³ (6/6 anchors) with 12 cells (3.5%) entirely outside at d=3; owned =
16807 = 7⁵ with 96 (0.57%) outside at d=5 — a DECAYING boundary band
(Hex9's signature behaviour) versus lineage's constant ~3.5%/6.5%. The
pinwheel question (§3) is precisely whether this last step can be made
exact: a √7 substitution carrier would take the odd-depth band to zero.

## 3c. Periodicity of the split field (Ben's question, 2026-07-10)

The half-hex prototile is not an aperiodic tile (it tiles periodically),
but the two verified half-hex transports differ sharply in how their
split-orientation fields behave:

- **Aperture 4: limit-periodic.** The role cascade (centre child
  inherits the parent split; edge child takes the shared-edge direction)
  makes level L's split pattern periodic with minimal period 2^L — the
  infinite hierarchy is the classical *half-hex substitution tiling*
  (Tilings Encyclopedia,
  <https://tilings.math.uni-bielefeld.de/substitution/half-hex/>:
  "easily seen to be limitperiodic", a p-adic cut-and-project tiling;
  scheme in Frettlöh 2002; the substitution appears in Grünbaum &
  Shephard as Exercise 10.1.3). Limit-periodic = 2-adic layering of
  periodic structures, the chair-tiling family. Non-periodic, but with
  no irrational rotations — still firmly Class L. Operationally: a cell's split is
  exactly computable from its ancestry (the 2-adic digits of its
  address), but there is NO level-uniform periodic lookup table.
- **Aperture 9 (Hex9): periodic fixed point.** MEASURED at L6: the
  split field is lattice-periodic with a small fixed period (21/21
  splits preserved under a 3-cell lattice translation; three orientation
  classes) — the p31m[3]₁ / c2 colouring is a genuine fixed point of the
  rep-9 substitution, the same table serving every level. Root cause:
  ×3 is odd and centre-preserving, so cell roles do not alternate;
  ×2 alternates centre/edge roles forever, forcing the 2-adic cascade.

This is a further design-choice point for §7: Hex9's LUT-per-level
implementation is possible *because* its colouring is a periodic fixed
point; an aperture-4 implementation would need ancestry-dependent split
resolution at every level (exact, but not table-uniform).

**Operational note (Ben + W. Renoud exchange, 2026-07-10).**
"Ancestry-dependent" is implementable as a digit-fold automaton:
o = fn(o, digit) over the address, then a terminal LUT keyed on
(state, layer parity) selects the representative carrier cell — after
which ownership is one exact lattice division. For planar-interior
GBT/H3 cells the fold degenerates to parity counting (digits are fixed
lattice vectors; frames don't rotate per digit); the automaton earns
its keep only at pentagon-adjacent paths and base-cell crossings, where
H3's own base-rotation and K-axis tables already ARE the fn. Hex9's
3-bit tail is precisely this fold's final state *memoized in the datum*
(p_c2 is the mcc2-LUT fold replayed from the root; the tail caches it),
which is what makes inversion and roll-up O(1) rather than O(layers).
Portable consequence: any GBT-family index with spare bits could cache
the fold state — "an H3 with a tail" — and buy the same O(1)
ownership. The cache is NOT strictly necessary (Ben, 2026-07-10): the
fold state space is tiny (~parity × exceptional rotation × pentagon
flag; a ~200-entry table) and replayable on every read, so uint64
suffices computationally. What the fold cannot recover is the
*convention* — which representative carrier cell was chosen is a
tie-break outside the datum, so fold-based systems must carry it
out-of-band, per-dataset. That is exactly what #108 proposal item 3
standardizes: the DGGRS metadata declaration is the out-of-band tail —
same bits, coarser granularity. Three tiers: convention in the datum
(Hex9), in the dataset (fold + declaration), or nowhere (status quo —
identical indexes, incompatible owned sets).

## 4. Spherical conditions

Sahr's planar figures "drop to some form of icosahedral" on the sphere
(Ben's observation). A planar transport survives the polyhedral net iff:

- **S1 — face closure.** T_L restricted to each polyhedron face tiles it
  exactly (face edges lie in ∂T_L at every level). Hex9 satisfies this
  by construction (half-hexes native to the octant in b_oct).
- **S2 — cone-point closure.** At polyhedron vertices the tiling must
  close under the vertex rotation group (cf. the local-net holonomy
  guard). The naive integrality test: an icosahedral vertex zone is 5/6
  of a hexagon, so a same-level expression needs 5m/6 ∈ ℕ. For the
  claimed ISEA3H pentagonal transport (m = 3): 2.5 — not integral. For
  the now-verified ISEA4H half-hex (m = 2): 5/3 — not integral either,
  so the planar m = 2 transport cannot express icosahedral vertex
  pentagons as-is; the m = 6 centre-triangle fallback can (5 triangles).
  Note the naive test is not the last word: Hex9's octahedral vertex
  zones also fail it naively (2/3 hexagon, m = 2 ⇒ 4/3), yet Hex9 is
  exact globally — the fold/d_cell construction resolves vertices in a
  way the planar count doesn't capture. Whether an ISEA4H-class system
  could do the analogous thing at its 12 vertices is OPEN; what is clear
  is that the planar transport does not survive the sphere for free,
  which is exactly why DGGRS metadata should declare what its
  implementation actually provides.

## 5. Machine verification plan

CSAT-style, per candidate system:

1. Implement T_L in exact arithmetic (Eisenstein integers / exact
   rationals on the triangular lattice; Hex9 already has this in b_oct).
2. Check to depth d: T1 (child unions = parent, no gaps/overlaps), T2
   (zone = exact union of m cells), and the induced ownership properties
   (partition, a^d counts, interior-intersection, k/m weights).
3. Spherical: S1/S2 at one face edge and one vertex of the polyhedron.

Artifacts so far: `docs/dggs/transport_check.py` (H3 stride-2 counts +
½-weights + the aperture-4 half-hex rep-4 check, all PASS) and
`docs/dggs/dggs_transport.py` →
`dggs_transport.png` (the visual, six panels spanning the taxonomy,
with pale reference-zone fills at both scales — top row
aligned-every-level: S2 m=1 rep-4, Hex9 half-hex m=2 rep-9,
aperture-4 hex half-hex m=2 rep-4 (exact planar construction); bottom row
stride-2 with the rotated intermediate dashed: ISEA3H m=6 rep-9, H3 m=6
rep-49, ISEA7H (DGGAL) m=6 rep-49 — each with one gold-outlined coarse
transport cell tiled exactly by its blue owned fine cells; prints
16/9/4 // 9/49/49 as a soft check). Hex9's row is already covered by
the existing `h9_cell_ancestor`/`h9_descendants` machinery and
dggs_ownership.py.
Queue, by load-bearing-ness: (1) H3 stride-2 owned-count exactness
(49^d per hexagon — §3b's open unknown); (2) ISEA3H stride-2 ditto
(m² = 9); (3) ISEA4H per-level (triangles m = 6, and Sahr's half-hex
m = 2 as the finer alternative); (4) DGGRID ISEA7H rotation policy
(compound vs alternate — decides whether it gets a hatch at all).

## 5b. Lineage = ownership: when? (Ben's "(ever?)", 2026-07-10)

For hexagon-shaped zones, **never**: if transitive lineage coincided
with ownership at every depth, descendant sets would nest and the limit
shape of a cell's descendants would be the cell itself — but that limit
is the subdivision attractor (Gosper island for a=7), never a hexagon,
because hexagons don't tile hexagons. The two relations part at depth 1
and stay parted. Exactly two escape routes: (a) zones = the carrier
cells themselves — Hex9's d_cell layer has lineage = ownership exactly,
polygonally, at every depth (0.00% commute at all k); (b) zones = the
fractal attractors (Gosper cells) — exact but infinite-perimeter.
One-sentence form for community education: *you can have hexagonal
zones, or lineage = ownership, but not both — unless you accept carrier
cells or fractal cells.* The API consequence: ownership is a second
verb (aggregation, dedup, joins) beside lineage (navigation,
truncation), not a replacement — the historical failure mode was
calling the lineage verb for ownership jobs.

## 6. Why this matters (the #108 hook)

It turns "documented ownership convention" from prose into a checkable
declaration: a DGGRS SHOULD declare its transport (prototiles, m,
aperture) when one exists, or else its exact lattice ownership rule
(aperture-7 systems). Clients then know ownership and weights are exact,
and *which* of the four properties (transitivity, count regularity,
containment, partition) each served relation carries. Feeds the paper
(§12, carrier/mode-transport) and the ISO 19170 alignment note.

## 7. The design-choice reading (Ben, 2026-07-10)

The six-panel figure doubles as an argument for Hex9's design choice
among hexagonal DGGS. Every other hexagonal panel carries an asterisk:
ISEA3H/H3/ISEA7H get their transport only at stride 2 (odd levels by
convention or composition), and the aperture-4 class — which does match
Hex9's m = 2 half-hex at every level in the plane (verified §2) —
loses that transport at the icosahedral vertices (5m/6 = 5/3, §4),
where Hex9's octahedral fold construction is machine-verified exact.
Hex9 is the hexagonal system with no asterisk — every level, m = 2,
no rotated intermediate to skip, and whole-sphere exactness shipped —
because aperture 9 = 3² is aligned at every step and the d_cell fold
handles the cone points. And the causality runs the
other way from the rest: elsewhere the transport was discovered
underneath zones designed first; in Hex9 the d_cell (transport) is
primary and the hexagon is a reified pair, so addressing, ancestry and
ownership are literally the same arithmetic (no convention beyond
mode-0, no parity condition). Compactly: **Hex9 takes the stride-2
escape hatch ISEA3H needs and makes it the ground floor — every Hex9
level is an "even" level.** The honestly-stated price is step
granularity (×3 linear per level vs ×√3), which is what the
single-nibble tail buys back. Candidate prose for paper §12.

## References

- Sahr, K. (2011), *Hexagonal discrete global grid systems for
  geospatial computing* (fig. 4: aperture-3 pentagonal and aperture-4
  half-hex underlying tilings).
- OGC API-DGGS issue #108 thread (St-Louis: draft "Area under the
  fractal curve" + szPolynomials.py, sub-zone counts for arbitrary
  apertures; local copies in session scratchpad 2026-07-10).
- uber/h3#1095 (correct sub-zones in native H3 terms).
- Hex9 paper: Axioms 2/4, §1 Simplicial Carrier, §3 Mode Transport, §6
  Refinement Commutativity, §12 comparison.
- Grünbaum & Shephard, *Tilings and Patterns*, W.H. Freeman 1987
  (rep-tiles; the general tiling vocabulary; the half-hex substitution
  is Exercise 10.1.3).
- Frettlöh, D. (2002), *Nichtperiodische Pflasterungen mit
  ganzzahligem Inflationsfaktor*, Univ. Dortmund (the half-hex
  cut-and-project scheme / limit-periodicity).
- Tilings Encyclopedia, "Half-Hex":
  <https://tilings.math.uni-bielefeld.de/substitution/half-hex/>.
