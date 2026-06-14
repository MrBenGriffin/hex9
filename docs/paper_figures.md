# Figure Specifications

Every `[Figure: …]` placeholder in the arcs, plus recommended additions.
"Source" names the existing example believed closest; ✳ marks figures that
likely need bespoke (but small) plotting code.

| # | Section | Working title | Source                                                                   |
|---|---|---|--------------------------------------------------------------------------|
| F1 | §8 / §2 (Foundations) | The 49 tilings, Hex9 pair highlighted | `experimental/halfhex_further.py` (emits `hh.hh.svg`)                    |
| F2 | §8 | The two equilateral tilings (L/R chiral pair) | same script (`eq.hh.svg`)                                                |
| F3 | §10.0 | Anatomy of one parent triangle: t/c2/d/x | ✳ from `hhg9.h9.region` / `polygon`                                      |
| F4 | §10a | 9-child x_cell subdivision, x_dig labels | ✳ (same geometry as F3, hexagon framing)                                 |
| F5 | §10a | Three-family band partition of the octant | ✳ simple matplotlib                                                      |
| F6 | §10b | London / Prime Meridian split-cell example | `ex0262_greenwich_seam.py` / `ex0263_…_zoom.py`                          |
| F7 | §10d | Sibling adjacency classes within a parent | ✳ (F4 with class colouring)                                              |
| F8 | §10d / §10f | Octant seam: c2 edges matching across boundary | `ex0262_greenwich_seam.py`                                               |
| F9 | §10f | merged into F3 (four-panel nested progression) | — see F3                                                                 |
| F10 | §10f | Octahedral vertex neighbourhood (4 faces, c2=0) | `ex0063b_octant.py` or polar crop of `ex0118_mollweide.py`               |
| F11 | §11 | Pipeline diagram b_raw → warp → b_oct → address | ✳ schematic (draw.io/TikZ, not generated)                                |
| F12 | §11b | Area-deviation globe, RdBu diverging | `ex0080w_warped_authalics.py` — `snow_globe()` (currently commented out) |
| F13 | §11c | Tissot indicatrices: b_oct vs Mercator vs geographic | `ex0120_tissot.py` / `ex0121_tissot_svg.py`                              |
| F14 | §12 | (optional) H3/S2/HEALPix/Hex9 same-region cells | external libs; licensing check                                           |
| F15 | §14a | Population heatmap binned to Hex9 | `ex0301_heatmap.py` + `examples/hh_heatmaps` (GBR/BTN/BHS data)          |
| F16 | §14c | Same layer in b_oct / Mollweide / Mercator | `ex0118_mollweide.py` + `ex0300_backdrop.py`                             |
| F17 | Intro/§5 | Octant boundaries over political world map (chirality convention) | `ex0118_mollweide.py` with country backdrop                              |
| F18 | (optional) | Butterfly/net unfolding of the octahedral grid | `butterfly.png` (exists)                                                 |
| F19 | (optional, §10e) | Zoom sequence strip: one point through L1→L7 | `ex0098_zoom_sequence.py`                                                |
| F20 | §2/§7/§8 (Foundations) | Wallpaper group of the d_cell tiling (p31m, confirmed) | `p31m.png` (exists)                                                      |
| F21 | §9/§10 or graphical abstract | Seed solid (24 d_cell facets) + 12 root x_cells coloured | `symmetry.png` (exists)                                                  |
| F22 | §10a/§10d | Unit tile of the enumeration: nested x_dig labels over the periodic lattice | `hex_enum.png` (exists)                                                  |

## Content requirements per figure

**F1 — the 49 tilings (candidate Figure 1 of the paper).**
Grid of all 49 solutions, chiral pairs adjacent (the existing SVG already
groups them), self-mirror solution marked, and the Hex9 pair (solutions
09/48, strings `044044…511` / `442442…113`) outlined or starred. Caption
carries the counts: 49 = 24 pairs + 1 self-mirror; constraint A → 18;
constraint B → 8; A∩B → 2. The verify script's counts are the caption's
authority.

**F3 — anatomy (the §10.0 companion; absorbs F9).** Redesign decisions
(2026-06-10): flat orthographic 2D throughout (no perspective — the figure's
job is metric: equilateral angles and congruence must read true; F21 is the
one 3D figure); no separate plane-tiling panels (F20/F22 cover "in the
large"); and a **2×3 grid** whose axes are the two distinct operations:
columns = the taxonomy (t / d / x), rows = hierarchy levels (i above, i+1
below). Down-arrows = refinement, annotated "×9" (§6); across-arrows =
assembly, annotated "group 3 by long edge" (t→d) and "join mode pair on c2"
(d→x) (§7). Panels:
  - t column: parent triangle; below, the 9-grid with mode two-tone (∇/Λ)
    introduced here.
  - d column: the parent's 3 d_cells with one c2 label set (0/1/2); below,
    the child-level d_cells (drop internal t-edges — densest panel).
  - x column: one hexagon straddling two ghosted parents (a hexagon never
    fits one triangle — emergence made visible); below, its 9 children —
    the shifted-aperture-9 picture, no central child, geometry only.
Ghost at most one layer per panel. The bottom-right panel is deliberately
unannotated: F4 is its call-out enlargement (digits, high-trit colouring,
split cells), giving visual continuity between the two figures. This figure
replaces F9, carries the whole notation section, and sets the mode tones
used by F4/F7 and matching F21's light/dark convention.

*Iteration (2026-06-10, after field-canvas draft):* the continuous-field
render (t→d/x blending across, level i→i+1 down, mode two-tone) shows
emergence and hierarchy well but lacks specimens. Hybrid design: keep the
field canvas, add three **unpacked call-outs** in the margins, each with a
leader line to a highlighted instance in the field — one t_cell (anchors the
mode colour key), one d_cell exploded as 3 t_cells with small gaps and the
long edge marked c2, one x_cell exploded as its 2 d_cells parted at the join
edge. Outline one level-i parent and its level-i+1 footprint on the canvas
for the ×9 story. Production: margin labels must not be black-on-black;
articulate the quadrant boundaries (thin dividers or margin ticks) so the
blend reads as structure, not decoration.

**F4/F7 — children and adjacency (can be one figure with two panels).**
F4 is the *hexagon-level* figure (not the t-cell 9-grid, which may appear
faintly beneath). Panel a: a parent x_cell with its 9 children labelled 0–8,
coloured by high trit (mode-0 interior / mode-1 interior / split), showing
there is no central child and the parent centroid lies on the d_cell edge,
with the split children straddling the parent boundary. Panel b (F7): same
geometry, cells coloured by adjacency class (interior / mid-edge / vertex),
with the cross-parent edges drawn heavier.

**F5 — band partition (the classifier / encode figure).** The three line
families are the three c2 edge directions extended into parallel families;
labelling each family with its c2 number reinforces the notation. Subject is
the encode operation of §10a: the octant triangle overlaid with the three
families (horizontal, ±√3 slope) at level k, one marked point with its three
band values annotated — "locating a point is band membership, not search" —
and an inset at level k+1 showing identical structure with thresholds scaled
by 1/3. (F5 = how a point finds its t_cell; F4 = how 9 children sit in a
parent hexagon; F3 = how the four vocabularies name one piece of geometry.)

**F6 — London.** Map of Greenwich/central London with L4 (and optionally L5)
hex boundaries and address labels from the worked example in §10b
(43483 / 43486 / 43527); the Prime Meridian drawn. The figure shows the
"lineage jump" across the c2 boundary visually.

**F10 — vertex neighbourhood.** The immediate neighbourhood of one octahedral
vertex (ideally a geographic pole): 4 octant faces, the c2=0 d_cells meeting
at the vertex, and the 4 surrounding x_cells, annotated with the valence-4
count. A polar azimuthal view reads best.

**F12 — deviation globe.** The `snow_globe` rendering with the fresh L5 run:
RdBu, centre 0, symmetric limits; caption quotes mean 0.000%, p99 0.0052%,
min −6.93% and points at the six vertex zones. Consider a second panel
zoomed at one pole.

**F13 — Tissot.** Same indicatrix grid (30° spacing per §14d) in three
projections; b_oct panel shows near-circular indicatrices away from
vertices; caption ties to "the distortion belongs to the map" (§11c).

**F17 — chirality convention.** World political map with octant edges
overlaid, showing the seams avoiding Europe/Texas etc. — documents the
chirality choice and lets implementors verify placement (notes, Layer 4c.i).

**F20 — crystallographic structure (p31m).** Left: the translational unit of
the d_cell tiling. Right: IUC symmetry overlay — 3-fold centres, mirrors,
glides, fundamental domain. Caption should present the symmetry-reduction
chain: plain hex tiling p6m → long-diameter cuts in the Hex9 arrangement
p31m → mode colouring p3, with the observation that the mirrors lost in the
last step exchange the two modes — the ℤ₂ of §2 realised as p31m/p3 — and
that the 3-fold centres are the rotation R of the constraint-B argument (§8).
**Verify before captioning:** p31m confirmed for the uncoloured cut pattern;
remaining check is that the mode-coloured group is exactly p3 (the
colour-symmetry analysis).

**F21 — the seed solid and the 12 root x_cells.** Left: the octahedron with
each octant face creased into its 3 d_cell facets — 24 faces, the diploid
(dyakis dodecahedron) form that names the d_cell. Right: coloured per root
x_cell, hue by octahedral axis — green at the N/S polar axis, red at the
0°–180° equatorial axis (Atlantic/Pacific), blue at the 90°E–90°W axis —
with light/dark shades as the mode-0/mode-1 halves. Four cells per axis, two
per vertex: the colouring makes visible that at L0 *every* root cell is one
of the 12 topological pentagons (2 per vertex; §12 / notes Layer 4d.i).
The figure carries five claims at once: 12 roots = 24 d_cells ÷ 2; the
diploid etymology (companion to F20); root cells straddling octant seams by
the ordinary join rule (§10f at L0); the pole-touching cells of the
chirality convention; and the pentagon-cell identification above. Candidate
graphical-abstract / opening image. Caption caution: state that the faceting
is illustrative — the cells live on the smooth ellipsoid surface, not on a
polyhedral approximation.

**F22 — unit tile of the enumeration.** The rhombic translational unit of the
digit assignment, with x_dig labels nested at three successive levels (large
= level k, medium = k+1, small = k+2). Carries three claims: the whole-plane
enumeration is one translational unit repeated (the same unit cell as F20 —
symmetry of geometry and symmetry of labelling, paired); self-similarity
across levels (§10a's "identical structure, thresholds scaled by 1/3"); and
the split digits 6/7/8 visible as a pattern along parent boundaries (§10b's
seam signature). Legibility: full-width placement, caption anchoring the
three label scales; consider outlining one parent hexagon and its 9 children
as a reading entry point. Companion to F4 (one parent, clean) — F22 is the
"and so on everywhere" continuation.

## Production notes

- Vector (SVG/PDF) for all diagram figures; raster acceptable for F12/F15.
- One colour scheme across F3/F4/F7/F9 (the t/d/x anatomy family) — same
  colours for mode-0/mode-1/split throughout the paper.
- Each figure file name: `fig_<id>_<slug>.<ext>` in `docs/paper_figures/`.
