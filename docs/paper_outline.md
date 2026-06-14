# Paper Outline — Hex9

## Abstract
→ docs/paper_abstract.md

## Introduction
→ docs/paper_intro.md

## Axiom Set
→ docs/paper_axioms.md

## Arc Sections (Foundational)

1. The Simplicial Carrier          → paper_arc_1.md
2. Mode                            → paper_arc_2.md
3. Mode Transport                  → paper_arc_3.md
4. Vertex Closure                  → paper_arc_4.md
5. The Octahedral Embedding        → paper_arc_5.md
6. Refinement Commutativity        → paper_arc_6.md
7. Dual Projection                 → paper_arc_7.md
8. Orientation Selection           → paper_arc_8.md
9. Hex9 Cell Identity — DGGS = CRS   → paper_arc_9.md

## Section 10 — Addressing & Continuity

10.0 Notation — c/t/d/x taxonomy  → paper_arc_10_0.md
10a. Identity as Locator          → paper_arc_10a.md
10b. Identity as Key              → paper_arc_10b.md
10c. Identity as Label            → paper_arc_10c.md
10d. Adjacency from Refinement    → paper_arc_10d.md
10e. Continuity                   → paper_arc_10e.md
10f. Seams and Valence Defects    → paper_arc_10f.md

## Section 11 — Geometric Realisation (AK+Warp)

→ paper_arc_11.md

11a. The Base Projection (AK)    — octant-to-ellipsoid, Anders Kaseorg formula, α ≈ 3.2278, mode asymmetry
11b. The Authalic Warp           — Sinkhorn OT, CT interpolant, NR inverse, L5 stats
11c. The Native Space            — b_oct as definitional; distortion is representational
11d. Projection Independence     — interchangeability, ellipsoid extension, AK+Warp as default

## Section 12 — Comparison with Prior Art
→ paper_arc_12.md   (H3/S2/HEALPix table; exception cells; ancestry; area; aperture; reference body)

## Section 13 — Implementation
→ paper_arc_13.md   (partition cycle; UUID + adr_byte; hhg9; PROJ plugin; PostGIS plan)

## Section 14 — Applications
→ paper_arc_14.md   (binning/density; joins & roll-up; rendering; graticule alignment; other bodies)

## Section 15 — Conclusion and Future Work
→ paper_arc_15.md

## Appendix A — The CRS Limit
→ paper_appendix_a.md   (Cantor argument, lifting, a.e. bijectivity — promoted from notes Layer 4g)

## Figures
→ paper_figures.md   (specification of all figures, sources, production notes)

## References
→ docs/references.bib   (core entries started 2026-06-14; Kaseorg key needs a citable URL)

## Still to draft
- OGC/ISO terminology pass (glossary.md table → verified against current spec editions;
  Topic 21 edition 20-040r3 confirmed, DGGS-as-SRS paraphrase verified in arc 9)
- Figure *production* of bespoke figures (F3/F4/F5 anatomy; F12/F13 warp; F6/F15/F17 maps)
- F23 (Thimphu adaptive, L5–L12) — image exists; caption + §14a placement done
