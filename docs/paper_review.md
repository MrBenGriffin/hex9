# Paper Review — readability + gaps (2026-06-10)

Scope: paper_intro, paper_outline, paper_axioms, paper_arc_1–9, 10a–10f, 11, paper_notes.

Overall: the arc structure is strong and the derivation narrative ("requirements, not choices")
is compelling and distinctive. The main problems are (a) a terminology cliff at §10, (b) one
mathematically shaky link in the derivation chain (§4), (c) statistics that disagree between
arc 11 and the notes, and (d) whole sections that exist only in notes form.

---

## A. Highest-priority issues

### A1. The §10 terminology cliff (biggest readability blocker)
Arcs 1–9 use only general vocabulary (faces, mode, octants). §10a then uses, without
definition: `x_dig`, `x_cell`, `t_cell`, `d_cell`, `c_cell` ("96-slot classifier"), `c2`,
`x_adr`, "tail", "split x_cells". The constructive sequence t_cell → d_cell → x_cell is
first *explained* in §10f — after being *used* in 10a, 10b, and 10d.

**Fix:** add a short notation section between §9 and §10 (or move 10f's second paragraph
and its figure to the front of §10) introducing the c/t/d/x taxonomy. An inline condensation
of `glossary.md`. Define "region" (10b uses "region 9", "region 6" with no definition
anywhere in the arcs). `c_cell` is mentioned exactly once — either define it or cut it.

### A2. §4 (Vertex Closure) — ✅ RESOLVED (2026-06-10)
Rewritten with the honest weaker argument: vertex closure (derived) forces even valence;
Axiom 3's regularity requirement (imposed, not derived) selects uniform valence; Euler
leaves v = 4 as the only possibility. The "refinement instability" necessity claim is
removed; §4 now explicitly states no claim is made that mixed even valences fail.
Axiom 3's consequence line updated to credit Axiom 4 (was Axiom 5).

### A3. Conflicting warp statistics — single source of truth needed
Three different stat sets are in circulation:

| Source | p99 | min/max dev | log-ratio σ |
|---|---|---|---|
| arc 11b + notes "Current warp characterisation" (`l5_warp_data.npz`) | 0.024% | −6.948% / +0.140% | 2.03×10⁻⁴ |
| notes "Why quasi-authalic" + Open Questions (`WGS84_l5_warp_data.npz`, marked production) | 0.014–0.015% | −0.330% / +0.097% | 5.24×10⁻⁵ |
| notes warp-progression table | "p99 0.024%" row labelled "current" | — | — |

Arc 11b cites the *older* file. Decide which warp file is canonical, rerun the
characterisation once, and update every site. Same for precision: arc 11b says
"sub-millimetre … ~0.25 mm"; notes say "~7nm bidirectional" and "1e-14 ≈ 0.25mm".
(1e-14 b_oct units × ~6.4×10⁶ m ≈ 64 nm — neither 0.25 mm nor 7 nm; recompute and state
one number with its derivation.)

### A4. §5 pole-anchoring — ✅ RESOLVED (2026-06-10)
Rewritten as a maximal-symmetry argument: sphere offers O(3) (no preferred embedding);
ellipsoid of revolution retains D∞h; the shared symmetry of an embedded octahedron is
D4h (order 16) for vertex-pair alignment, D3d (12) for face-pair, D2h (8) for edge-pair,
~nothing generic. Pole-on-vertex anchoring uniquely maximises shared symmetry, and under
D4h the 8 octant faces form a single orbit — giving the Layer 4c payoffs (single
projection function, mode-1 via y → −y) directly in the arc.

### A5. §8 "exactly 2 of 8 survive" — ⚙ HALF RESOLVED (2026-06-10)
**Done:** the full enumeration chain is now machine-verified by
`experimental/halfhex_verify.py` (V0–V6, all pass): 2 equilateral tilings (chiral pair);
unique 3-equilateral decomposition; 49 tilings (24 pairs + 1 self-mirror, no hash
collisions in the dedup); constraint A → 18 (9 pairs); constraint B (3-triangle
structure, now automated) → 8 (4 pairs); A ∩ B → exactly the recorded Hex9 pair.
The "manually evaluated" caveat in notes Layer 3 is cleared.

**Still open:** arc 8 itself must cite the enumeration (verify script + Figure 1) so the
"exactly 2 of the 8" claim reads as a verified result, not an assertion.

---

## B. Readability — cross-cutting

1. **Naming**: ✅ RESOLVED (2026-06-10) — "Hex9" is canonical in all prose; `h9` survives
   only as the code/API prefix (`h9_encode`, `h9_boct`). Convention noted at the top of
   paper_notes.md; add the one-line convention to the implementation section when written.
2. **Holonomy/flat-connection boilerplate is repeated 4×** (Axiom 4, arc 3 ¶3, arc 3 ¶4 —
   which restate *each other* almost verbatim — and arc 6). Define once in §3, in one crisp
   sentence with both vocabularies, then just say "flatness" thereafter. Arc 3 ¶3–¶4 should
   be merged into one paragraph.
3. **Arc 6 verbal tic**: "(… , in algebraic terms)", "(… , in geometric terms)" appears six
   times in one page. Keep at most one or two.
4. **Arc 1 repetition**: "Every other polygon either decomposes into triangles or requires
   additional structure" appears twice nearly verbatim (¶2 and ¶4). Also the claim
   "higher-order tilings must [introduce topological defects]" is too loose — S2's cube-based
   quad grid is the obvious counterexample a reviewer will raise; the defect there is the
   8 corner vertices, so rephrase in terms of where defects land, not whether they exist.
5. **Typos / grammar**:
   - arc 2 last paragraph: "what / what follows" — duplicated word across the line break.
   - Axiom 6: "collapses to a one surviving solution" → "to one surviving solution".
   - arc 5 mixes curly and straight apostrophes; ragged line breaks mid-sentence.
   - intro: "The paper hopes to develop this argument" → "develops". Don't hope in print.
   - intro uses "AK+Warp" before "AK" is ever expanded — name Kaseorg or say "the base
     octahedral projection" at first use.
6. **Heading style**: axioms use `#`/`##` markdown; arcs use plain "1. Title" with 2-space
   indented body text. Standardise to markdown headers + normal paragraphs before any
   conversion to LaTeX/Sphinx (the indentation will bite you in some renderers).
7. **Outline broken reference**: outline line 11 points to `paper_arc.md`; the file is
   `paper_arc_1.md`.
8. **Axiom 9 defect count is ambiguous**: "12 global topological defects … absorbed … as
   vertices". There are 6 vertices carrying total defect 12 (2 per 4-valent vertex), and 12
   topological pentagons (2 per vertex, per notes Layer 4d.i). Say which count you mean —
   the current wording invites "but the octahedron has 6 vertices" objections.
9. **Axiom 4 contains a theorem**: "Refinements satisfying this condition have odd k" is a
   *result* (proved in arc 6), not an axiom. Move it fully into the *Consequence* footer so
   axioms state only requirements.
10. **Arc 10b is the densest page in the paper** (p_c2, r_mo, p_mo, h, net_mode, canonical
    mode-0 convention, all in two paragraphs). It needs: a small table of the tail fields
    (name / bits / what it disambiguates / failure without it), and ideally the London/PM
    worked example from notes — that example is excellent and currently unused.
11. **Arc 10c vs 10b apparent contradiction**: 10c says "the mode of any cell is recoverable
    from its position in the refinement tree without external metadata"; 10b spends a page
    explaining why the tail metadata is needed to recover exact mode/parentage. The
    resolution (labels use the mode-0 cover; keys need the tail for exact cells) is real but
    unstated — one bridging sentence in 10c fixes it.
12. **Arc 10e**: "By completeness, any sequence of nested closed regions with diameters
    converging to zero has a unique point of intersection" — this is Cantor's intersection
    theorem and needs compactness/closedness stated; notes Layer 4g does it correctly.
    Align 10e's language with 4g (or just cite forward to the formal version).

---

## C. What's missing (gaps to fill)

Sections that exist only as notes or not at all:

1. **Abstract + keywords** — nothing drafted.
2. **Notation/terminology section** (see A1) — the single highest-leverage addition.
3. **Comparison with prior art** — the H3/S2/HEALPix table and the H3-pentagon analysis
   (notes Layers 4d, 4d.i — some of the strongest material in the project) have no arc.
   Outline sections 6–8 of the notes' "Suggested Paper Structure" (Comparison,
   Implementation, Applications) have no files.
4. **Conclusion / future work** — nothing drafted.
5. **References** — zero citations anywhere in the arcs. Needed: Kaseorg, Sahr/White/
   Kimerling, Snyder, Lee, Cuturi (Sinkhorn), Clough–Tocher, OR-Tools, OGC Topic 21,
   ISO 19111, H3/S2/HEALPix docs. (Notes has the TODO; start the .bib now.)
6. **Figures** — 8 inline placeholders (10a ×2, 10d ×2, 10f ×2, 11 ×2) plus Figure 1
   (49 tilings) and the suggested octant-boundaries-over-political-map figure. None prepared
   in publishable form yet.
7. **Formal CRS-limit proof** — notes Layer 4g is genuinely good and essentially
   journal-ready; promote it to an appendix or into §10e. The open-questions checkbox
   "Formalise the CRS/DGGS limit claim" is closer to done than the list admits.
8. **Mode asymmetry analysis** (notes, AK section: the L4/L5 ±0.066% table, α ablation) —
   strong honest-limitations material, absent from arc 11a. Reviewers reward this; add a
   paragraph.
9. **Project history / icosahedral dead-end narrative** — notes say it should inform the
   intro; the intro currently has none of it. One paragraph ("the icosahedron was tried
   first and failed for a topological reason") would both humanise and strengthen the
   necessity argument.
10. **Graticule alignment note** — orphaned in notes; a natural short subsection for §11c
    or Applications.
11. **OGC/ISO terminology mapping table** — flagged TODO, needed before submission.
12. **Verify arc 9's OGC quote** — "Topic 21 … defines a DGGS as a kind of spatial
    reference system": check exact wording against the current spec edition before print.

## D. Per-file one-liners (where not covered above)

- **intro** — good shape; add the history paragraph and expand AK at first use.
- **arc 1** — tighten repetition; soften the "higher-order tilings must" claim.
- **arc 2** — fix "what what"; otherwise the best-written early arc.
- **arc 3** — merge ¶3/¶4; end-of-section forward references are good, keep them.
- **arc 4** — see A2; this is the section to rework before anything else.
- **arc 5** — see A4; import Layer 4c congruence argument; fix typography.
- **arc 6** — trim the "in X terms" tic; the k=2 counterexample is clear and good.
- **arc 7** — clean; consider stating "12 non-hexagonal dual cells / 6 vertices" explicitly
  to pre-empt the Axiom 9 count confusion.
- **arc 8** — see A5; add citation to enumeration + Figure 1.
- **arc 9** — strong; the "no residual freedom" closing argument is the paper's thesis in
  miniature — consider echoing it in the abstract.
- **10a** — needs A1's notation section first; the three-band inequality description is
  good and concrete.
- **10b** — tail-field table + London example (B10).
- **10c** — bridging sentence (B11).
- **10d** — "readable from the x_dig values alone" — show the actual rule or table;
  currently hand-wavy relative to the precision of neighbouring sections.
- **10e** — align with Layer 4g (B12).
- **10f** — its ¶2 belongs at the head of §10 (A1); otherwise the strongest §10 piece.
- **11 / 11a–d** — update stats (A3); add mode-asymmetry paragraph (C8); 11c's "the
  distortion belongs to the map" is a quotable line — keep it.
- **axioms** — B8, B9; also consider stating the Main Theorem (notes §Notes) right after
  the axiom list so the reader knows where the axioms are headed.
- **outline** — fix `paper_arc.md` ref; add the missing sections (C3, C4) so the outline
  matches the notes' suggested structure.
- **notes** — fine as a quarry; the Open Questions list should absorb the items above.

---

## Status update — 2026-06-10 (second pass)

**Resolved this pass:**
- **A1** ✅ — notation section created: `paper_arc_10_0.md` (c/t/d/x taxonomy, c2, region,
  split digits, lists/addresses/tail); added to outline as 10.0.
- **A3** ✅ — definitive L5 statistics measured from `WGS84_l5_warp_data.npz` via
  `ex0080w_warped_authalics.py`: mean 0.000% (closure), min −6.93%, max +0.078%,
  σ(log) 1.89×10⁻⁴, p99 0.0052%, p99.99 0.28%. Arc 11b, notes characterisation block,
  comparison table, and open-questions entry all updated; both earlier stat sets marked
  superseded. Remaining: the NR-precision figure (0.25 mm vs 7 nm conflict) is flagged
  as an inline TODO in arc 11b — needs one verified measurement.
- **B1–B9, B11, B12** ✅ — naming; arc 3 ¶3/¶4 merged (defines "flat" once); arc 6 tic
  trimmed to two glosses; arc 1 repetition cut and tiling-defect claim made precise
  (defects land in cells or corners, with H3/S2 examples); typos fixed (arc 2, Axiom 6,
  intro "develops", AK expanded at first use); headers converted to markdown across all
  arcs (## titles, ### for 11a–d); outline ref fixed; Axiom 9 defect count made precise
  (Σ(6−valence)=12 at 6 vertices, deficit 2 each); Axiom 4's odd-k theorem moved to its
  Consequence footer; 10c bridging sentences added (label vs tail trade); 10e now cites
  Cantor's intersection theorem.
- **B10** ✅ — 10b: tail-field summary table, prefix-cutting vs canonical-ancestry
  caution, and the London/Prime Meridian worked example (ported from notes).
- **10d** ✅ — concrete adjacency content from the implementation: constant LUT keyed by
  (cell, parent mode, c2); interior/mid-edge/vertex three-class structure; split digit
  k+6 ↔ interior child k wing relationship; low-trit = c2; octant hop = index lookup +
  y-reflection.
- **arc 7** ✅ — explicit defect counts added (six 4-sided dual cells, deficit 2 each).

**Still open (the C-list, unchanged):** abstract; comparison/implementation/applications/
conclusion sections; references (.bib); figure production (9 placeholders + Figure 1);
promote Layer 4g CRS-limit proof to an appendix; mode-asymmetry paragraph for 11a;
icosahedral-dead-end paragraph for the intro; graticule subsection; OGC/ISO mapping
completion; verify the Topic 21 quotation; arc 8 citation of the (now machine-verified)
enumeration.

---

## Status update — 2026-06-10 (third pass: missing sections drafted)

**Drafted this pass:**
- `paper_abstract.md` — abstract + keywords (≈250 words; echoes §9's thesis).
- `paper_arc_12.md` — Comparison (table; exception cells/Euler; strict ancestry;
  area; the not-centred-aperture-9 distinction; sphere vs ellipsoid).
- `paper_arc_13.md` — Implementation (partition cycle; UUID/adr_byte encodings;
  hhg9; PROJ plugin; PostGIS plan).
- `paper_arc_14.md` — Applications (binning/density; joins & exact roll-up;
  rendering; graticule alignment — ported from notes; other reference bodies).
- `paper_arc_15.md` — Conclusion & future work (two-free-choices framing; warp,
  implementation, verification, standardisation work-streams).
- `paper_appendix_a.md` — the CRS limit, promoted from notes Layer 4g (fixed an
  area-formula slip: cell area = total/(12·9^L), not "719 km²/9^L"; notes fixed too).
- Intro: icosahedral dead-end paragraph added. Arc 11a: mode-asymmetry paragraph added.
- `paper_figures.md` — full figure specification: 19 figures, each with section,
  content requirements, and the closest existing example as source.
- Outline updated to cover all of the above.

**Now the only structural gaps are:** references/.bib; figure *production* (specs
exist); OGC/ISO terminology verification; the NR-precision measurement (arc 11b TODO);
arc 8's citation of the verify script; and the C-list polish items (Topic 21 quote check).