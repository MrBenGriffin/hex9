# Figure Specifications

Every `[Figure: …]` placeholder in the arcs, plus recommended additions.
"Source" names the existing example believed closest; ✳ marks figures that
likely need bespoke (but small) plotting code.

| # | Section | Working title | Source                                                                   |
|---|---|---|--------------------------------------------------------------------------|
| F1 | §8 / §2 (Foundations) | The 49 tilings, Hex9 pair highlighted | `experimental/halfhex_further.py` (emits `hh.hh.svg`)                    |
| F2 | §8 | The two equilateral tilings (L/R chiral pair) | same script (`eq.hh.svg`)                                                |
| F3 | §10.0 | Anatomy of one parent triangle: t/c2/d/x | ✳ `examples/ex0400_anatomy.py` (primitives validated; layout WIP)         |
| F4 | §10a | x-layer: x_dig 0–8 children, high-trit colouring + shared c-regions overlay | `examples/ex0400_anatomy.py` `figure_f4` → `ex0400_anatomy_f4.png` (supersedes `f4_hex_outline.png`/`f4_better.png`) |
| F7 | §10d/§10f | Sibling adjacency: internal vs cross-parent edges (blue/red) | `examples/ex0400_anatomy.py` `figure_f7` → `ex0400_anatomy_f7.png` |
| F5 | §10a | Classifier (c-layer): 96-slot c_grid, three band families → c_dig | `f5.png` (exists, amended) — the c-layer counterpart to F4's x-layer |
| F6 | §10b | London / Prime Meridian split-cell example | `ex0262_greenwich_seam.py` / `ex0263_…_zoom.py`                          |
| F7 | §10d | Sibling adjacency classes within a parent | ✳ (F4 with class colouring)                                              |
| F8 | §10d / §10f | Octant seam: c2 edges matching across boundary | `ex0262_greenwich_seam.py`                                               |
| F9 | §10f | merged into F3 (four-panel nested progression) | — see F3                                                                 |
| F10 | §10f | Octahedral vertex neighbourhood (4 faces, c2=0) | `ex0063b_octant.py` or polar crop of `ex0118_mollweide.py`               |
| F11 | §11 | Pipeline diagram b_raw → warp → b_oct → address | ✳ schematic (draw.io/TikZ, not generated)                                |
| F12 | §11b | Area-deviation globe, RdBu diverging (magnitude view) | `ex0081w_warped_authalics.py` — `snow_globe()` → `ex0081wau_5.png` (exists) |
| F12b | §11b | Area-deviation Mollweide, clipped scale (pattern view: seam skeleton) | `ex0118_mollweide.py` → `ex0118_mollweide_L5.png` (exists)               |
| F13 | §11c | Tissot indicatrices on the b_oct butterfly net (warp applied) | `ex0121_tissot_svg.py` → `ex0121_tissot_50_warp_file_butterfly_0500.svg` (exists) |
| F14 | §12 | (optional) H3/S2/HEALPix/Hex9 same-region cells | external libs; licensing check                                           |
| F15 | §14a | Population heatmap binned to Hex9 | `ex0301_heatmap.py` + `examples/hh_heatmaps` (GBR/BTN/BHS data)          |
| F16 | §14c | Same layer in b_oct / Mollweide / Mercator | `ex0118_mollweide.py` + `ex0300_backdrop.py`                             |
| F17 | Intro/§5 | Octant boundaries over political world map (chirality convention) | `ex0118_mollweide.py` with country backdrop                              |
| F18 | (optional) | Butterfly/net unfolding of the octahedral grid | `butterfly.png` (exists)                                                 |
| F19 | (optional, §10e) | Zoom sequence strip: one point through L1→L7 | `ex0098_zoom_sequence.py`                                                |
| F20 | §2/§7/§8 (Foundations) | Wallpaper group of the d_cell tiling (p31m, confirmed) | `p31m.png` (exists)                                                      |
| F21 | §9/§10 or graphical abstract | Seed solid (24 d_cell facets) + 12 root x_cells coloured | `symmetry.png` (exists)                                                  |
| F23 | §14a | Thimphu adaptive refinement (L5–L12), population density choropleth | `thimpu_chloropleth.png` (exists, from QGIS via `h9_adaptive()`)                                                  |

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
one 3D figure); no separate plane-tiling panels (F20 covers "in the
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

**F4 — the x-layer (generated: `examples/ex0400_anatomy.py` `figure_f4`, output
`ex0400_anatomy_f4.png`).** Two panels (parent mode 0 / mode 1 in the
foreground), each an **overlay of both concentric parents**: the foreground mode
drawn solid, the other mode **ghosted** behind it (faded, no digits). Each hexagon
carries its **`x_dig` (0–8)** as the large white digit; the region/`c_dig` id sits
**once per region**, at the t_cell centre (the shared vertex of that region's three
hexes), cross-referencing straight to F5. **Colour = high-trit class** (CVD-safe,
shared with F3's t/d/x): blue = `x_dig` 0–2, amber = 3–5, red = 6–8 (the split
cells); split children are drawn as their half-hex **wings**, the others as full
hexagons. The digits are generated from the ground-truth LUT `_m_c2_hx_v2025` in
`addressing.py` (clusters of 0/1/2, 3/4/5, 6/7/8).

The figure makes four points. (1) **No central child** — the parent centre lands
on a d_cell edge, so the origin is ringed by three hexes, not one. (2) The **split
children (6/7/8) straddle the parent boundary** — the two-parent case that makes
`x_list` right-to-left. (3) The two opposite parents form a **hexagram** whose
central hexagon is the **6 c-regions shared by both modes** — geometrically
*identical* cells (same origin, same child mode), while each mode contributes only
**3 unique outer c-regions** (mode-0: 0x21/0x2b/0x49; mode-1: 0x16/0x34/0x3e),
whose ghost ids pop out of the faded background. (4) Those shared core hexes carry
**different `x_dig`s under each parent** (e.g. 0/6/3 vs 1/7/5 around region 0x35) —
so the ghost digits are deliberately suppressed, and the c-region id is what is
shown as invariant. This is the visual link tying the whole c/t/d/x grid together:
one set of c-regions (the classifier substrate of §10a / F5) underlies both modes,
with mode and parent context supplying the differing x-layer addresses.

**F7 — adjacency (delivered: `examples/ex0400_anatomy.py` `figure_f7`, in the
ex0110 idiom).** Two panels (parent mode 0 / mode 1). Each child t_cell's three
**edges are coloured blue (internal — shared with a sibling in the same parent)
or red (external — cross-parent)**, the same internal/external convention as
`ex0110_poly_neighbours.py`. A child's adjacency class is then simply its
**red-edge count**: interior (0) = 25/2a/39, mid-edge (1) = 26/35/3a, vertex
(2) = 21/2b/49 — a clean 3-3-3 per mode, reinforced by a faint class fill. The
red edges trace the parent perimeter (the cross-parent edge set). The test is
geometric (a child edge is external iff it lies on a parent side) and is
**asserted equal to the ground-truth `region.py` `_neighbour_builder` class** for
all 18 cells, so the figure is verified, not asserted. The central hex cluster
(3 hexes around the origin) is called out as the x_dig 0 (mode 0) / 1 (mode 1)
cluster — the link to F4. Green-free palette (steel/amber/brick + blue/red edges)
avoids colliding with F3 (mode) and F4 (high-trit). Output `ex0400_anatomy_f7.png`.

**F5 — the classifier (c-layer).** Delivered as `f5.png` (amended from the
earlier C2 draft). Shows the encode operation of §10a: a point's coordinates
are resolved by three band families whose indices compose the classifier digit
`c_dig = h_id<<4 | p_id<<2 | n_id` (0x00–0x5F). The 96-slot `c_grid` is labelled
in hex (`00`…`5F`, including out-of-scope cells); the horizontal family gives
`h_id` (thresholds Λc=2h/3, Vc=h/3, 0, Λf=−h/3, Vf=−2h/3), and the two diagonal
families give `p_id` and `n_id` with their slope inequalities. The in-scope
octant is the blue triangle with its t/d substructure; cell membership is the
condition `f < y ≤ c−|x|`; constants pinned (h=√6/2, w=√2, ẋ=x/3). The message:
locating a point is band membership, not search. F5 is the c-layer counterpart
to F4's x-layer (`x_dig`); together with F3 (t/d/x geometry) they cover the full
c→t→d→x taxonomy. Caption caveat: this is dense — consider a stripped companion
(bands + one marked point + its three band values) if a gentler version is
wanted for the body, keeping the full grid as a reference/appendix figure.

**F6 — London.** Map of Greenwich/central London with L4 (and optionally L5)
hex boundaries and address labels from the worked example in §10b
(43483 / 43486 / 43527); the Prime Meridian drawn. The figure shows the
"lineage jump" across the c2 boundary visually.

**F10 — vertex neighbourhood.** The immediate neighbourhood of one octahedral
vertex (ideally a geographic pole): 4 octant faces, the c2=0 d_cells meeting
at the vertex, and the 4 surrounding x_cells, annotated with the valence-4
count. A polar azimuthal view reads best.

**F12 — deviation globe (magnitude view).** The `snow_globe` rendering with
the F6-retrain L5 run: RdBu, centre 0, **symmetric** limits (±5%, `lim_pct`),
with a colourbar. Caption quotes the canonical hex-level stats — mean exactly
0.000% (closure), min −3.57%, max +4.80%, MAE 0.001%, σ(log) 1.99×10⁻⁴, p99
0.0044%, p99.99 0.43% — and points at the six octahedral vertex zones, where the
worst hex (+4.80%) sits just inside ±5%. Global render (`ex0081wau_5.png`) = near-white
globe with only faint vertex specks — almost featureless by design (that *is*
the equal-area result). Because it reads as blank at page size, a **vertex-zoom
inset is recommended** (same ±5% scale) to show the residual is real, bounded,
and localised to the vertices; F12b carries the spatial pattern at full globe
scale. NB the old min −6.93% / max +0.078% set is SUPERSEDED — do not requote it.

**F12b — deviation Mollweide (pattern view).** `ex0118_mollweide_L5.png`. The
complement to F12: an asymmetric scale **clipped to roughly the p1–p99 band
(≈ −0.3% to +0.2%)**, so the near-white interior recedes and the octant **seam
skeleton + six vertex blooms saturate** — making the *spatial pattern* of the
residual legible where the symmetric magnitude view shows only specks. **Caption
caveat (required):** the colourbar range is clipped and does NOT reach the true
extremes — actual per-hex deviation spans min −3.57% to max +4.80% (MAE 0.001%),
concentrated at the six octahedral vertices and the octant seams, which is why
those features saturate. Without this note the figure reads as harsh/extreme
everywhere rather than "interior quiet, defects on the seams." Both F12 and F12b
colour per-hex area deviation (not the pointwise RDE field).

**F13 — Tissot (delivered: b_oct butterfly).**
`ex0121_tissot_50_warp_file_butterfly_0500.svg`. Tissot indicatrices over the
AK+Warp b_oct projection laid out as the octahedral butterfly net, with a
continent backdrop and the faint hex mesh. The indicatrices are **circles of
near-constant radius across the entire net — interior, continents, and out to
the wing-tip (octahedral vertex) regions**: constant size demonstrates the
quasi-authalic (equal-area) property, and their near-circularity demonstrates
the low shear that the optimal-transport derivation buys "for free" (§11b — a
strictly authalic projection would permit visible ellipse elongation; these do
not). Slight ellipticity is visible only at the central pinch and immediately
around the vertices. This is the b_oct panel of the comparison the §11c
placeholder asks for; the Mercator and geographic panels (same 30° indicatrix
spacing, §14d) are still to add to complete the "distortion belongs to the map"
triptych — but the b_oct net alone already carries the equal-area + low-shear
claim. Vector SVG; keep as SVG/PDF.

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

**F23 — Thimphu adaptive refinement (the versatility showcase).** Produced
via `h9_adaptive()` over the Bhutan population layer, rendered in QGIS over an
OSM-style terrain/road basemap (`thimpu_chloropleth.png`). Method for the
caption: cells refine on population, target band min 50 / max 200 per cell,
across **L5 (coarse terrain) to L12 (dense valley corridors)** — seven levels
in one coherent layer with no inter-level seams or T-junctions. Fill is a
ranked population *density* (population ÷ authalic cell area), binned for colour
by log base 9 — ln(value)/ln(9) — so each refinement step is one colour bin,
matching the aperture. **Data provenance:** Bhutan high-resolution population
density (`btn_general_2020.csv`, from the HDX dataset [@hdx_bhutan_pop]); the
source extent overruns the national boundary, so it was first clipped to
Bhutan's borders before binning. The figure makes three claims at once: deep
hierarchy
(L12) with no degeneracy; mixed-resolution cells coexisting with exact roll-up
(§12/§14a); and authalic cells making density directly comparable across levels
without per-cell area correction (§11b/§14a). Caption keeps the depth claim
*relative* (cells span L5–L12) rather than committing to an absolute L12 metric
cell size. Companion to F15 (uniform-level heatmap) — F23 is the adaptive
counterpart.

## Final captions — drop-in figures (images on disk, verified 2026-06-14)

These six images exist and are publication-ready; the text below is the final
caption for each (the f3/f5 PNGs on disk are superseded hand-drawn drafts — they
are replaced by the F3/F4/F5 plotter, not captioned here).

**F1 (`f1.svg`) — The enumeration.** *The 49 distinct hextile solutions. Chiral
pairs are grouped; the single self-mirror solution is set apart; the Hex9 pair
(highlighted) is the unique solution surviving both constraints. Of 49 = 24
chiral pairs + 1 self-mirror, the long-edge constraint admits 18 and the
three-equilateral structural constraint admits 8; their intersection is the
highlighted pair. Counts machine-verified by `halfhex_verify.py`.* (§8 / §2)

**F2 (`f2.svg`) — The chiral pair.** *The two equilateral tilings that survive
internal closure: a 9-cell equilateral tiled by three half-hexagons, in the only
two admissible orientations. Colour denotes orientation class (the PP23[3]1
three-colouring); the digit strings (044040222 / 442420200) are the canonical
encodings. The two members are mirror images — the residual chirality that
Axiom 6 fixes.* (§8)

**F20 (`f20.png`) — Crystallographic structure.** *Left: the translational unit
of the d_cell tiling. Right: its IUC symmetry — three-fold centres (triangles),
mirror and glide lines (dashed), and the fundamental domain (shaded). The
symmetry-reduction chain is plain hexagonal tiling p6m → the Hex9 long-diameter
cut p31m → the mode-coloured group p3; the mirrors lost at the final step are
exactly those that exchange the two modes (the ℤ₂ of §2), and the three-fold
centres are the rotation R of the constraint-B argument (§8).* (§2 / §7 / §8)
[Verify before final: that the mode-coloured group is exactly p3.]

**F21 (`f21.png`) — The seed solid and the 12 root cells.** *Left: the
octahedron with each octant face creased into its three d_cell facets — 24 faces,
the diploid (dyakis-dodecahedral) form that names the d_cell. Right: coloured per
root x_cell, hue by octahedral axis (green N/S polar, red 0°/180°, blue 90°E/W),
light/dark shades the mode-0/mode-1 halves. Four cells per axis, two per vertex:
at L0 every root cell is one of the 12 topological pentagons. The faceting is
illustrative — the cells live on the smooth ellipsoid, not a polyhedron.*
(§9 / §10; graphical-abstract candidate)

**F18 (`f18.png`) — flagged, likely drop.** A Tissot-on-butterfly render
(continents + indicatrices, no hex mesh) that overlaps F13. Use *only* if a
no-grid or pre-warp contrast panel is wanted; otherwise omit — F13 carries the
butterfly-Tissot story with the mesh and the warp applied. [Decision needed.]

## Production notes

- Vector (SVG/PDF) for all diagram figures; raster acceptable for F12/F15.
- One colour scheme across F3/F4/F7/F9 (the t/d/x anatomy family) — same
  colours for mode-0/mode-1/split throughout the paper.
- Each figure file name: `fig_<id>_<slug>.<ext>` in `docs/paper_figures/`.
