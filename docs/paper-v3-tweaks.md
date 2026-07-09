# Paper v3 tweak tracker

v2 replacement submitted to arXiv 2026-07 (see paper-v2-tweaks.md, now closed).
This file collects everything queued for the v3 replacement. Add items freely;
mark ☑ when applied to `paper-draft.md`.

## Done (2026-07-08 batch — axiom-review findings, Ben's read-through)

- ☑ **Axiom 4 was self-referential**: it used "mode" as its subject, but mode
  is only defined in §2. Reworded to be self-contained: the transport is
  defined inside the axiom (assign the nontrivial element of ℤ₂ to every
  face-to-face edge crossing; consistency = closed loops compose to identity =
  bipartite face-adjacency graph), and "mode" is introduced as the *name* of
  the resulting coherent binary face field, with a forward pointer to §§2–3.

- ☑ **Axiom 8 overclaimed for x_cells**: "every cell has a unique address" is
  true of t_cells and d_cells (single-parent) but split x_cells (x_dig ∈
  {6,7,8}, three per level) have two valid parents and hence two lineage
  bodies (glossary.md:108, :163 — the §10b/c locator/key/label trichotomy).
  Axiom 8 now states the bijection as cell ↔ **canonical** address, with a
  parenthetical on split hexagons and the lineage-path/address distinction
  (forward ref to §10). Main Theorem item 5 and abstract updated to "unique
  canonical address". This aligns the axiom with §10 rather than weakening it.

- ☑ **§1/§2 mode-from-orientability derivation was false** (Ben's 3-simplex
  counterargument, confirmed): the boundary of the 3-simplex is orientable and
  coherently orientable, yet its face-adjacency graph is K₄ — no mode
  two-colouring exists. Old §2 opening claimed adjacent triangles "carry
  opposite orientations" on *any* orientable surface; it conflated coherent
  orientation (exists iff orientable; adjacent faces get the SAME handedness —
  planar up and down triangles are both ccw) with the mode bipartition
  (exists iff the dual graph is bipartite iff every vertex valence is even).
  Fixes applied:
  - §1: triangle's orientation claim softened to what is defensible — the
    orientation *datum* is purely combinatorial (vertex ordering mod even
    permutations, no metric); cw/ccw labels additionally need the surface's
    orientation ("viewing side"). Other polygons need a cyclic boundary
    order / metric / embedding.
  - §2: mode now *defined* as a proper two-colouring of faces, with two
    explicit cautions (mode ≠ orientation; existence not automatic —
    tetrahedron and icosahedron counterexamples named). Existence ⟺ bipartite
    dual, deferred to §§3–4. "ℤ₂ orientation cocycle" framing removed.
  - §3: transport no longer "inherited from the intrinsic orientability of
    §1" — it is the demanded flip, defined uniformly whether or not a mode
    field exists. "On any orientable surface flatness ⟺ mode field" corrected
    to "on any connected triangulation" (orientability is irrelevant to that
    equivalence).
  - Net effect strengthens the paper: if mode followed free from
    orientability, Axiom 4 would be vacuous. The tetrahedron/icosahedron show
    it is a real selective constraint — the one that eliminates the
    icosahedral seed.

- ☑ **§4 honesty-guard paragraphs clarified** (Ben: "not sure what it's
  getting at"; "irregular"?; vertex-star *type* jump). The two paragraphs
  after the vertex closure condition are the disclaimer that flatness alone
  does not force the octahedron: even valence is derived, uniformity is not.
  Rewritten to (a) open with the point ("Vertex closure alone does not finish
  the argument"), (b) replace the loose word "irregular" with explicit
  mixed-even-valence phrasing + a concrete example (midpoint-subdivided
  octahedron: valences {4,6}, dual bipartite, mode fine), and (c) define
  vertex-star type inline — combinatorial isomorphism class of the star,
  fixed by valence alone (same type ⟺ same k; NOT even-vs-odd) — and tie it
  to Axiom 3's "no distinguished vertex classes" as its formalisation.
  Footnote added: the example is combinatorially the aperture-4 refinement
  §6 rejects — no tension, since aperture 4 preserves flat mode transport
  (valences stay even) and fails only hexagonalisation (octant won't tile by
  half-hexagons).

- ☑ **§5 ellipsoid scoping + triaxial footnote + two-roles sentence** (Ben:
  "reference ellipsoid" argument too broad — triaxial bodies lose D∞h).
  Main text now scopes explicitly: geodetic reference ellipsoids (WGS84,
  GRS80) are ellipsoids of revolution, with Ben's salience progression ("the
  more the body's symmetry breaks, the more of the embedding it pins":
  sphere → nothing; revolution → polar axis, meridian gauge remains;
  triaxial → all three axes, finite residue). Footnote on triaxial, verified
  group theory: symmetry drops to D2h (order 8); maximal-common-symmetry
  principle survives but TIES — vertex-aligned and 45°-rotated (equatorial
  edge axes onto equatorial principal axes) both share the full D2h (face
  alignment dies: no C2 about a C3 axis in O_h) — so a convention is needed;
  oblate-limit continuity selects vertex alignment. Octant orbit survives
  (D2h order 8 acts freely on 8 faces → single orbit), so one-projection,
  one-warp economy holds; §11 warp is numeric/body-agnostic, keeping the
  abstract's "any ellipsoid" honest. Also split the maximal-residual-symmetry
  paragraph into its two orthogonal roles (Ben's point): combinatorial
  anchoring of the seed vs the realisation warp — symmetry makes the warp
  correction exactly uniform across octants (uniformity stated as fact;
  minimality as what uniformity buys, per honest-claims policy).

- ☑ **§9/§10a address-pair contradiction fixed + §10.0 figure-anchored**
  (Ben). §9: "identified by its octant and its path" → "identified by its
  path" / "this path is not a coordinate". §10a opening rewritten: the pair
  is (path, metadata), NOT (octant, path) — tail's root-mode + level-0 digit
  *derive* the octant; parent-c2 resolves the 6–8 split ambiguity (§10b
  forward ref). §10.0 "Lists and addresses": tail corrected byte → **nibble**,
  resolving "root octant and split-cell parentage" (was "split-cell parentage
  and terminal state"). Figures redistributed into 10.0 per taxon: fig 6
  (ex0400_anatomy_f3) moved above t_cell; NEW G&S composite
  (paper_figures/grunbaum_shephard_c2.png, built from grunbaum_shephard_1+2,
  overlay panel right) into c2 with attribution caption; fig 8 (f5.png
  classifier) into c_cell; fig 7 (ex0400_anatomy_f4 trit colouring) into
  x_dig. §10a keeps cross-refs ("pictured in §10.0") in place of the moved
  figures. Composite v2: white matte (v1 flattened alpha to black — Ben
  caught it), stacked vertically (each source is itself a two-panel image:
  plain colourings / d_cell overlay; the horizontal strip made 4 tiny
  panels), width 80%. Panels are Ben's redrawn vectors (Tol palette,
  embedded attribution), NOT scans — caption says "redrawn after
  [@grunbaum1987tilings]", so no reproduction-rights concern. Sources fixed
  by Ben 2026-07-08 (date → 1987; second image's credit replaced with a
  d_cell-overlay explainer note); composite rebuilt from them.
- ☑ **§10a geocode + §10b prefix-containment qualification** (Ben). §10a
  close: "qualifies the Hex9 address as a geocode, and Hex9 itself as a
  locating system" — geocode adopted for the address; "locating system" KEPT
  for the system because Appendix B maps that exact phrase to ISO 19111 CRS
  (3.1.9). §10b para 2 rewritten with the two-fold qualification: (i) bare
  bodies are lineage paths — a split-digit body (466666 as illustration, per
  Ben; not a literal from the text) resolves to different cells under
  different tails, so prefix ops are over canonical keys; (ii) split children
  geometrically straddle their two parents (only the mode-0 half is inside
  the canonical parent) — containment is exact as canonical ancestry, exact
  geometrically at the d_cell layer (§10e), and explicitly NOT whole-hex
  nesting, "a claim no hexagonal hierarchy can make, since hexagons do not
  tile hexagons" (turns the caveat into the §12 contrast). Operational paras
  now say "canonical addresses/prefix"; truncation forward-refs the
  canonical-ancestry caution. Cleanup in para 4: dropped "no cells that
  straddle index boundaries" (split cells DO straddle parent boundaries;
  clause kept only edge-effects/wrap-around sense) and "every cell has
  exactly one address" → "exactly one canonical address" (matches Axiom 8
  fix).

- ☑ **§10b stale reversible-tail paragraph replaced** (Ben: predates the
  last breaking change; current methodology = hhg9/h9/tail.py +
  uuid_address.py canonical fold). Old text described a byte-sized "full
  reversible tail" with a 4-bit h field (terminal d_cell 0–11, d_cell
  centroid recovery) — that is the legacy scheme (the nibble-30 h_term slot,
  now the 0xF/0xE bin/key marker). Rewritten to the single-nibble tail:
  p_mo (bit 3) / p_c2 (bits 2–1) / r_mo (bit 0); canonicalisation (half-hex →
  mode-0 fold before the tail is written) pins p_mo = 0, so reversible and
  key styles emit identical nibbles; round-trip recovers the canonical
  cell's region + representative point, never the source point. Table
  updated (h row removed, bit positions added); "both tails" paragraph
  replaced; §10.0 pointer trimmed ("details its fields"). Follow-up (Ben):
  "reversible" is resolution-relative — added sentence: reversibility bounded
  by resolution not encoding; ~40 nm at full UUID depth (matches §13b) so
  exact for any geodetic purpose; up to a cell radius at coarse bins, by
  design. §13b checked —
  already current (self-inverting UUID, marker nibble, supersession note),
  no change needed. NB out of paper scope: full-hex9 addressing elsewhere in
  hhg9 has not yet migrated to this tail method (Ben; it will); glossary
  tail subsection (h field, 0xF sentinel rows) may need a sync pass when it
  does.

- ☑ **Body-uniqueness census + Fig 9 mechanism (43585)** (Ben's question:
  post-canonicalisation, do tail-less bodies still clash?). ANSWER: NO —
  NEW experimental/body_census.py enumerates every cell L0–L5 via HexMesh
  geometry + canonical h9_bin_pts addressing: 12/108/972/8,748/78,732/708,588
  bodies, zero collisions, zero duplicate labels, all layers complete. §10b
  rewritten accordingly: (i) canonical bodies are unique names even without
  tails (footnoted to the census script); (ii) but lineage ≠ ancestry —
  43585 cuts to 4358 while its canonical L3 ancestor is 4348 (4358 = same
  hexagon's non-canonical mode-1-parent reading; canonical 4358 names a
  DIFFERENT hexagon elsewhere — Ben's synonymy point); (iii) tail rationale
  reframed: not uniqueness but READING — pins parent context so decode is
  local/O(1) (p_c2: else global re-descent; r_mo: root hexagon spans two
  octants, selects frame). Table "Without it" column updated to match; the
  466666 dual-reading illustration replaced by the verified 43585 case.
  Operational paras: joins/containment via canonical (tail-aware) truncation
  h9_bin/h9_ancestors; raw prefix cut = lineage grouping. London example
  extended with the 43585 paragraph (the deeper jump: interior child of a
  split parent — mode-0-half children extend 4348, mode-1-half children
  extend 4358); "thesecells" typo fixed. Fig 9 caption rewritten (two jump
  forms; 43581/43585 wholly interior, 43587 on the rim). Structural note:
  uniqueness is inductive — each cell's canonical name claims exactly one
  (parent, digit) slot; census is the machine check through L5. Second
  independent check (Ben, 2026-07-08): examples/ex_10092_uuid.py compares
  tail-stripped h9_label strings over mesh.addrs — 78,732 at L4, 0
  duplicates. h9_label docstring now carries the caveat: without the tail
  the canonical label is unique but not (locally) invertible or re-binnable.
- ☑ **grid.py mesh.addr tail bug FIXED** (found during census, NOT paper):
  the hand-rolled UUID packers in HexMesh.create and create_clipped took the
  HIGH nibble of the hex_layer tail byte → every mesh.addr label got tail .0
  (7,236/8,748 L3 labels disagreed with the canonical pipeline; bodies all
  agreed). Per Ben ("grid.py should use the canonical tail.py logic —
  rolling its own was a mistake"): both packers replaced with h9_bin_pts on
  the mode-safe centroids (canonicalise + tail.py packing + L0 special case
  for free); unused hex_layer/TailStyle/batch_nibbles_to_int/uuid imports
  dropped. Verified: 0 mismatches vs canonical pipeline at L0–L4, correct
  tail histograms, London 4348.2 now in mesh set, ex_10092_uuid.py runs,
  full pytest suite 464 passed / 18 skipped.

- ☑ **10.0 slimmed** (Ben approved 2026-07-08): c2 etymology → footnote
  (kept the G&S citation + naming-mismatch note; figure stays in place);
  t_cell's 12-region-class detail moved to §10b's p_c2 paragraph, where the
  region-9/region-6 example first exercises it. 10.0 stays one section,
  definition + figure per letter.
- ☑ **Desync sweep after the census reframe** (Ben's request). Fixed:
  §10c "fails as a name in both directions / p_c2 and r_mo collisions" —
  now: canonical bodies never collide (census), but the body falls short as
  a name because (i) lineage ≠ ancestry (43585→4358 vs canonical 4348) and
  (ii) not locally readable (global re-descent without the tail). §13b
  "binning is prefix comparison, no decoding step" → bin *comparison* is
  plain equality/prefix; bin *derivation* is truncation WITH tail
  canonicalisation, not a raw digit cut. §14a "wherever non-canonical
  addresses may be present" (stale — stored addresses are all canonical) →
  raw cut walks lineage, differs from ancestry in the thinning §12 band.
  §14b "equality of level-K prefixes is co-location" → equality of
  canonical level-K bins. Checked and already in sync: §12 (Ancestry
  paragraph, split-cell band + (1/6)·3^(1−k) decay, nesting figure),
  Appendix A (the contraction argument runs on the t_cell lineage reading —
  triangles nest strictly; hexagon assembly plays no role in convergence).
- ☐ **FLAG (implementation vs prose)**: §12 claims coarser bins derive
  "from the address alone … in constant time with no re-encoding from
  geometry (§13b)". hhg9's h9_bin currently decodes to a point and re-bins
  geometrically (h9_dec → h9_bin_pts). Address-only derivation (digit cut +
  tail canonicalisation via right-to-left unzip) is presumably implementable
  — and may exist in libhex9 — but the Python route today does re-encode
  from geometry. Either implement the address-arithmetic bin or soften the
  §12 sentence before v3.

- ☑ **Ben's 10d-and-beyond batch (2026-07-08)**:
  - §10d ¶2–3 decompressed + grid-qualified: parent x_cell tiling stated as
    Ben's decomposition (6 interior child x_cells = 12 connected d_cells + 6
    rim d_cells: 3 mode-0 halves of own canonical splits, 3 mode-1 halves
    owned by neighbours); adjacency explicitly computed at the t_cell layer
    (3 neighbours per c2 edge, table keyed by region id/parent mode/c2);
    x_cell adjacency composes through t → d → x; boundary classes stated as
    child-t_cell classes matching the figure.
  - §10f opens "As stated in §4, …".
  - Newton–Raphson vs Gauss–Newton DISAMBIGUATED (they are NOT the same;
    both genuinely used): warp inverse = plain N–R on square 2×2
    (octahedral_barycentric.py:621, Cramer) — §11b/§13c usages correct;
    AK projection inverse = guarded G–N on 3×2 residual (§13f, libhex9).
    Fig 11 tikz label de-methodised ("decode + inverse warp + inverse
    projection"); caption names both methods with § refs; footnote at §11a
    explains the distinction (coincide only when square+invertible).
  - Fig 13 (Mollweide) caption: clip *reason* added (unclipped, vertex
    blooms own the scale; field washes out).
  - §11c footnote: b_oct "barycentric" is historical — planar Cartesian on
    the unfolded net, NOT simplex coordinates.
  - §11d Snyder: honesty scoped — asserted on published properties, not
    exercised (no working implementation found); composition claim is
    structural.
  - §13b: 32 nibbles = 31 path (root digit + base-9 per level) + 1 tail
    (was "30 + 2"). ("no decoding step" sentence was already fixed in the
    desync sweep.)
  - §13c: GeoPlegma paragraph added (Hex9 backend contributed; OGC API —
    DGGS-conformant tooling; DGGRS interface over DGGRID/H3/S2; footnote
    URL github.com/GeoPlegma/GeoPlegma). UPDATE WHEN PR ACCEPTED — currently
    "under review as a pull request".
  - §14: hard-coded "Figure 23" → "The figure below" (Thimphu); hard figure
    numbers are fragile now that §10.0 gained a figure. Only remaining hard
    number is the Figure 15 placeholder text itself.
- ☑ **Figure 15 RESOLVED** (Ben's choice, 2026-07-08): placeholder replaced
  with the existing ex0252 composite (paper_figures/ex0252_sn_B1kL12_CaSN.jpg)
  — NLCD 2024 land cover filling L12, Sierra Nevada DEM hillshade at L14,
  L9–L11 address-labelled overlay grid, shared prefix 6045431. Bridging
  paragraph added to §14a (raster sources sampled once per cell at the
  centroid; label elision as map graticule; legend states the equal-area
  constants). Figure-index rows updated in paper-draft.md and
  paper_figures.md; F23 companion note patched (F23 now carries the density
  story alone). No population heatmap needed.
- ☑ **Canonical cell roll-up LANDED in hhg9** (2026-07-08, follows the
  6–12-scatter finding): new h9_cell_parent / h9_cell_ancestor in
  uuid_address.py — cell question (canonical K-ancestor, mode-0 convention)
  now distinct from the point question (h9_bin family). Implementation:
  decode to centroid nudged 10% toward cell origin (strictly interior to
  the mode-0 half; same cure as libhex9 full_id_from_cell), bin one layer
  up; multi-level = composition (direct deep re-bin differs on exactly 1/9
  of cells — nested splits). Verified exactly 9 children/parent for every
  cell, L1..L5 (experimental/cell_ancestor_verify.py, ALL PASS) + London
  KAT (cell_parent(43585.1) = 4348.2). Suite: 470 passed. C-side verified
  meanwhile: h9_bin_uuid (address-arithmetic, point semantics) is CLEAN —
  2M-point test, split cells → exactly their 2 geometric parents, no
  scatter.
- ☑ **h9_cell_parent named in the paper** (Ben approved 2026-07-08): §10b
  operational clause now distinguishes `h9_bin` (full-depth addresses) from
  `h9_cell_ancestor` (cell keys); §10b caution names
  `h9_cell_parent`/`h9_cell_ancestor`; §13c states the operations are
  implemented in hhg9 and verified (nine-per-parent enumeration through L5,
  composition identity, London worked example) with a provenance footnote to
  experimental/cell_ancestor_verify.py. When the C port lands, optionally
  add it to the libhex9 feature list in §13c's first sentence.
- ☑ **§12 flag RESOLVED — C port landed** (2026-07-08, subagent;
  independently re-verified): h9kring::h9_cell_parent_uuid /
  h9_cell_ancestor_uuid (core/h9_kring.h:402-593), C ABI + ext bindings
  (cell_parent/cell_ancestor). Backward pass seeded c_mo=0
  (canonicalise-always), origin+centroid from the digit chain, 0.10 nudge,
  containment descent, identity cascade — no warp/ellipsoid round-trip, so
  §12's "derived from the address alone … no re-encoding from geometry"
  stands as written. Verified: byte-identical to hhg9 for ALL cells L1–L4,
  9-per-parent C-only through L5 (708,588 cells), London KAT (43585.1 →
  4348.2), composition over all L5 cells, ctest 11/11, hhg9 pytest 470,
  fossil-probe vertex point clean at every layer. Fossil F3 comments now
  point at the new functions. Notable: the agent independently rediscovered
  the mode-1-half disease inside uuid_from_cxcy (180/972 wrong at L2
  without the fold) — third sighting, same cure. §13c may now list the
  operation under libhex9 as well as hhg9 (optional one-word edit).
- ☑ **docs/addressing-doctrine.md WRITTEN** (libhex9/docs/, 2026-07-08):
  canonicalise-always contract, the three roll-up operations, the
  recurring centroid-on-seam disease (three sightings, one cure), fossil
  register — F1 under review (census evidence suggests retirable; needs a
  dedicated parse_label/common_ancestor probe), F2 standing (C-side
  decode(bin); hhg9 parity open), F3 RETIRED (h9_cell_parent/ancestor),
  F4 standing by design — plus the verification-artefact index. §13c also
  updated: libhex9 feature list gains "canonical cell ancestry (§10b)",
  and the hhg9 sentence upgraded to "implemented in both libraries,
  byte-identical across the pair for every cell through level 4". PDF
  rebuilt (48 pp).
- ☐ **Still open (library, not paper)**: F1 retirement probe
  (parse_label/common_ancestor over all split bodies); F2 C-side
  decode(bin) parity; consider folding canonicalise into uuid_from_cxcy /
  identity_from_uuid's bin path.
- ☑ **h9_descendants → canonical semantics (2026-07-08, DONE — suite green)**:
  filter switched from centroid re-bin to h9_cell_ancestor == anchor;
  docstrings/doctrine comment updated; enumeration clip inflated 1.03→1.6
  (canonical splits protrude up to R/2); tests strengthened to len ==
  9^g exactly + brute-force oracle now canonical (box 0.55→0.80).
  RESOLVED (2026-07-08, two independent fixes, all 470 tests pass):
  (1) Per Ben's directive the oid seam convention now has exactly one
  implementation: factored out of CompositeDomain.binning into
  CompositeDomain.seam_oid(neg, free) (hhg9/base/composite.py);
  OctahedralNet.binning resolves its own criteria — pt_face for the
  containing face, BaryNet.backward (c2-aware) to the bary triangle,
  on-edge ⇒ the opposite sv-corner's ECEF axis-bit is free (corner→axis
  map SV_CORNER_AXIS verified by projection: mode0 (0,2,1), mode1
  (0,1,2)) — then hands over to seam_oid. 3D-faithful across net cuts:
  every net copy of a seam point derives the same canonical oid. Audit:
  the only other binning override is Points.binning (a delegate) —
  b_oct/b_raw/s_oct are axes-2 CompositeDomains whose inherited binning
  raises, so no other skip exists.
  (2) The actual 8/9 dropper at vertex anchors: the fallback disc was
  sized from _anchor_hex_latlon's ring, and at vertex cells the hx ring
  template WRAPS PAST THE ANTIMERIDIAN (verts at lon ±178 for 022.0) —
  R exploded to 535° so the fixed 220² grid (spacing ~4.9°) was coarser
  than an L3 child (~3.3°) and split child 0227.1 fell between samples.
  Fix = the tracker's pre-approved option: size the disc from cell
  geometry directly, R = 135°·3⁻ᴸ (uuid_address.py, h9_descendants).
  (3) ROOT FIX for the 5th sighting (same session): the "vertex ring
  template" was a misnomer — the hx template is fine; the bug was
  STRICTLY-IN-OCTANT PROJECTION of overhanging ring verts (the actual
  grid_face_vertex_oid_bug mechanism). New canonical utility
  hhg9.h9.polygon.fold_to_octant(verts, oid): exact unfold across the
  violated sv edge — neighbour = oid ^ (1 << SV_CORNER_AXIS[mode][k]),
  coords by axis-matched corner isometry, iterated ≤3 for corner
  overhangs (cone-angle fold-order ambiguity at the 6 vertices is
  inherent; either representative is faithful). _anchor_hex_latlon now
  folds before projecting → 022.0 ring is a sane hexagon (was lon ±178).
  SV_CORNER_AXIS promoted to hhg9.h9.polygon (canonical, next to the sv
  LUT); OctahedralNet imports it. Also fixed fallback disc longitude
  span: converge at the disc's most-polar latitude and go full-lon when
  the disc contains a pole (poles were 79/81). All 6 octahedral vertices
  now 9/9 (L2,g1) and 81/81 (L3,g2); suite 470 passed. Candidate reuse:
  grid.py's strict-in-octant sites (create_clipped descent) could adopt
  fold_to_octant to kill the remaining sightings at source.
  STILL OPEN (follow-ups): (a) libhex9 PARITY: the C++ GcdBoctLib
  resolver (when the lib is loaded — NOT in this dev env, where the
  pure-Python g_gcd→b_raw→b_oct bridge runs) does its own octant
  assignment; the seam_oid mode-0 convention must be checked/ported
  there, same bucket as the pending cell_parent/ancestor C port —
  n_oct binning and h9_descendants/fold_to_octant are Python-only (no C
  counterpart exists); (b) tracker prose said vertices default to the
  LOWEST-numbered mode-0 octant — the implemented rule flips the HIGHEST
  free axis-bit (e.g. P(-1,0,0) → 5/SEP, not 3/NWP; both mode-0) — prose
  corrected here, keep seam_oid as the authority; (c) encode-path binning
  runs at ak_octahedral.backward:150 on c_ell coords (signs scale-invariant
  so valid) but exact-0 detection depends on trig: lon ±90/180 and poles
  yield ~4e-10 m positive noise, not 0.0 — currently benign (positive noise
  = free-bit default; probe: all 9 seam/vertex cases matched the exact
  convention) but a latent tolerance question.
- ☐ **(superseded by above) original 4th-sighting record**:
  membership filter re-bins descendant centroids at the anchor layer;
  probe on X=4348 returned 7/9 canonical children (lost 43488 + 43587,
  kept 43486 by tie-break). Fix candidate: filter by h9_cell_ancestor ==
  anchor — but that is a semantics DECISION (descendants = canonical
  children vs point-bin membership), Ben's call; note
  test_descendants_complete_vs_bruteforce is self-consistent with the old
  semantics and the dggs_nesting figure caption already CLAIMS mode-0
  binding ("each bound to its owner by the mode-0 convention") — the
  implementation currently doesn't honour that claim at tie-break cells,
  so either code moves to the caption or the caption softens. Related
  to-audit: C k_disk/adaptive/common_ancestor on cell inputs; PostGIS
  surface (expose cell_parent in SQL; audit any bin-of-bin flows).

## Pending

- ☐ **Address-arithmetic h9_cell_ancestor (Ben's rid/mode idea, confirmed
  2026-07-09)**: current impl (Py+C) is GEOMETRIC (decode → mode-0 nudge →
  re-bin); correct + byte-verified, but nudge margins scale 3⁻ᴸ so it
  degrades near L25+ in doubles. CONFIRMED RULE (Ben's formulation, checked
  empirically over all cells L2–L4, every cut): recover the mode of each
  digit leaf→root (the machinery survives as hex_digits_reg's backward
  context thread / the C backward pass; per-level RID 0..11 is
  side-bearing, parity = recovered mode, so rid chain = d_cell address);
  then at a split digit (6/7/8) with recovered mode 1, the containing
  parent is the lineage parent's NEIGHBOUR across that split's edge — and
  the hop cascades upward (the neighbour re-registers in its own frame;
  root hex/octant can change, e.g. 725.1→72.2 oct 1→3). All cut-level
  presentation folds are odd-rid→even-rid (mode-1 thread → canonical
  mode-0 registration); flip rates 1/3, 4/9, 39/81 → ~1/2 (matches raw
  p_mo=50%). Pure-address ancestor = truncate + symbolic
  neighbour-hop cascade (odometer-style carry; C has the machinery in
  k_ring/resolve_frames, Python would need it built). Prototypes:
  scratchpad rid_chain_proto.py / rid_fold_analysis.py. Options: (a)
  expose rid/mode chain only, (b) full pure-address ancestor Py+C, (c)
  keep geometric + document depth caveat.
- ☑ **RESOLVED (2026-07-09, Ben's d_cell doctrine) — h9_cell_ancestor is
  the DIRECT leaf-reified d_cell relation, not parent∘parent**: the
  2026-07-08 composition implementation was wrong doctrine, exposed by
  the dggs_nesting figure (729 = 9³ but 108 ENTIRELY OUTSIDE — deep
  tongues/voids, exactly 1/6 of area displaced; the composition
  re-adjudicates splits at every layer, i.e. "naively consider the
  x_cell at each level and the hexagon decoheres"). Correct doctrine
  (Ben): the subdivision tree is on d_cells — an x_cell's territory is
  18 next-layer d_cells (12 complete = 6 interior children + its 6
  split halves), the d_cell tree is rep-9 and nests EXACTLY, and mode-0
  reification of d_cells into x_cells happens ONCE, at the leaf. So
  ancestor(cell, L) = single deep re-bin of the cell's mode-0-interior
  point at L (well-defined: the mode-0 d_cell lies wholly inside one
  cell at every coarser layer). h9_cell_parent unchanged (coincides at
  one generation). Verified: 81 grandchildren per cell for ALL 108 L1
  cells; direct vs composed membership differs at exactly 1/9 of cells
  (nested splits; e.g. 5267.4 → 52.4 direct vs 58.0 composed); figure
  now 729 = 702 in / 27 rim-split straddle / 0 outside — the paper.tex
  §12 caption ("no Hex9 descendant lies outside the anchor") is TRUE
  as written and now machine-verified under canonical semantics (old
  Felix-era panel had wrong count 739 from point-bin tie-breaks).
  docs/dggs/dggs_nesting.png regenerated; arxiv/paper_figures copies
  refresh at the v3 rebuild step. Suite 470 green incl. rewritten
  test_cell_ancestor_direct_not_composed (pins direct ≠ composed at
  1/9). C MIGRATED same day: h9kring::h9_cell_bin_at_uuid engine
  (descent to arbitrary target layer); parent/ancestor now wrappers;
  byte-identical to hhg9 all cells L1–L4 × all coarser targets, 9^g
  count-set exact, C-only L5 sweep (708,588), nested-split + London
  KATs, ctest 11/11.
- ☐ Continue Ben's full offline read-through pass (carried over from v2
  tracker); further findings land here.

## When v3 goes up

1. Rebuild: `make -f Makefile.paper` (PDF) and regenerate `arxiv/main.tex` +
   re-strip `\linenumbers` + rezip (see `arxiv/SUBMISSION_NOTES.md`).
2. Inspect §§1–4 (rewritten prose) and Axioms 4/8.
3. Commit + tag (`paper_v3`).
4. Replace on arXiv (same ID, becomes v3); update JOSIS materials if sent.
