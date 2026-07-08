# d-cell space-filling curve (traversal grammar) search

Investigates whether the canonical Hex9 d-cell 9-dissection admits a
hierarchically consistent space-filling curve — the strong (Hilbert/S2-style)
sense of "SFC", not merely a prefix code or a per-level Hamiltonian path.

The dissection itself is fixed: these scripts *decode* the machine-verified
tiling strings from `experimental/halfhex.py` / `halfhex_verify.py` (the
chiral mirror pair, 27 triangle cells with orientation digits) back into the
nine child trapezoids, then search purely over traversals.

## Results (2026-07-06, all machine-checked)

- **No single-type grammar exists.** One base curve with endpoints among the
  four corners {P0, P1, P2, P3} and the long-edge midpoint M, under any
  variant regime — forward-only, with reversal, or with Hilbert-style mirror
  variants — on either chirality. Exhaustive. Yet a plain Hamiltonian path on
  the sibling edge-adjacency graph exists: Hamiltonicity is not the binding
  constraint; cross-level grammar consistency is. (`sfc_grammar.py`)
- **Multi-type curve systems exist abundantly**, even forward-only and
  chirality-uniform, i.e. as a pure overlay on the canonical dissection with
  no mirror state. Greatest-fixed-point over all 25 ordered arc types leaves
  23 viable. (`sfc_multitype.py`)
- **The minimum closed system has exactly 6 arc types** (sizes ≤ 5 ruled out
  exhaustively over the 23 GFP survivors), e.g.
  {P0→P1, P0→P2, P0→P3, P1→P3, P2→P1, P3→P0}.
  Expansion to depth 3 (729 cells) is verified in exact rational arithmetic:
  continuity, coverage, endpoint correctness. Compare S2's 4-state Hilbert
  machine. (`sfc_minimal.py`, figure `h9_sfc.svg`)

Structural note: the two 9-dissections are a chiral mirror pair and all child
placements are pure rotations, so mirrored curve variants would require the
mirror *dissection* inside σ-children — different geometry at depth, not just
a different ordering. Unlike S2's symmetric 2×2 split, mirror state is not an
overlay here; the result above shows it is also not needed.

## Fixed-point-anchor results (2026-07-06, second session)

Prompted by other.png / other_2.png (freehand shape-family curves with
interior starts): endpoints need not be named lattice points — for a
self-similar system they are *forced*, as fixed points of the first/last
child maps, computable exactly over Q. With derived anchors (interior
allowed, reversal allowed), exhaustively:

- **1 shape**: impossible (288 configs). No Gosper-analogue single pattern
  exists on this dissection. (`sfc_fixedpoint.py`)
- **2 shapes, no mirror**: impossible (1.33M configs). (`sfc_fixedpoint2.py`)
- **1 shape + mirror**: impossible (1,152 configs). (`sfc_mirror_fixed.py`)
- **2 shapes + mirror {A, Ꙭ, B, ᗺ}: EXISTS** — solutions are abundant
  (5 found in the first 4% of ~21M configs; anchors land on corners, e.g.
  A: P0→P2, B: P0→P3). Independently verified by exact expansion to depth 3
  (729 cells; continuity, coverage, endpoints) including the σ-image type.
  (`sfc_mirror_fixed.py big`, `sfc_mirror_verify.py`)

So mirror symmetry is provably load-bearing: two essential shapes suffice
with it and are impossible without it. **Caveat**: mirrored children imply
mirrored sub-dissections at depth (a *twin* hierarchy) — this curve system
is not an overlay on the canonical single-chirality hex9 tree. For the
canonical tree as-is, the 6-type forward-only named-point system above
remains the honest minimum.

### Self-mirror shapes are impossible (the constrained case)

Ben's other_2.png cast was {A, B, ᗺ} with A self-mirror (3 effective hands
instead of 4). This is provably unachievable, by a three-step argument
(`tilings_sigma.py` verifies the machine-checkable steps):

1. A self-mirror curve forces σ(cell_k) = cell_{10−k} on its level-1 cells,
   so its tile's dissection must be σ-symmetric as a set of 9 pieces.
   T1 and T2 are each other's mirrors, not self-mirrors.
2. Of all **49** half-hex 9-tilings (count matches `halfhex.py`), exactly
   **one** is σ-symmetric.
3. A palindromic visiting order has exactly one slot (the middle) for
   σ-fixed pieces — but the unique symmetric tiling has **three**
   self-symmetric pieces. Two of them can never be placed.

Hence no self-mirror shape exists in this dissection family; freehand
self-mirror motifs are necessarily approximate, and 2 chiral shapes × 2
hands (4 effective types) is the true floor. See `h9_sigma_tiling.svg`
(the gold pieces are the three self-symmetric ones; dashed line = axis).

### Face-continuity: edge-to-edge travel is impossible (2026-07-07)

Question (Ben): the found curves jump past intermediaries — consecutive
cells sometimes meet only at a corner. Can a system travel edge-to-edge
only, at every depth (face-continuity, as Hilbert/Peano/Gosper do)?

**No — proved on both available routes:**

- **Edge-interior anchors** (entry-edge/exit-edge classes; such systems are
  born face-continuous): exhaustively impossible with <= 2 essential shapes,
  with or without mirror — 21.2M fixed-point configurations, zero
  (`sfc_edgefix.py`). >= 3 shapes remains open (CP-SAT-scale).
- **Vertex anchors** (corners + M): the level-1 edge-handoff constraint
  raises the minimum from 6 to 7 types (`sfc_edge.py`) but every production
  choice hops at depth 2. The per-side ray-descent automaton is fully alive
  (92/92 states, `sfc_ray.py`), so there is no per-type obstruction — the
  failure is coupling. The exact decision (`sfc_cpsat.py`): CP-SAT over all
  17,728 edge-constrained productions of the 23 GFP types, free type-subset
  choice, coinductive ray-support booleans (the per-side ray condition is
  necessary AND sufficient — convexity pins each handoff to its parent
  edge). Verdict: **INFEASIBLE**. Reversal is covered (a reversed arc is an
  order-swapped type, whose menus were fully enumerated). Pipeline
  cross-check: `sfc_prune.py` (existential relaxation nonempty, so the
  CP-SAT step was genuinely needed).

Consequence: corner contact is intrinsic to d-cell SFCs at these anchors —
the 6-type system's Sierpinski-style point handoffs are optimal, not a
search artifact. Open gaps: >= 3 essential shapes with edge-interior
anchors; the twin-hierarchy (mirror) analogue of the CP-SAT decision.

### Figures

- `h9_sfc.svg` — 6-type named-point curve on the canonical tree, levels 1–3.
- `h9_sfc_mirror.svg` — the verified 2-shape mirror system, shape A levels
  1–3; cell fill shows dissection chirality (pale = mirrored), level-1 cells
  carry hand glyphs (mirrored glyphs drawn mirrored, ′ = reversed).
  Rebuild both curve/tiling figures with `python3 sfc_mirror_render.py`.
- `h9_sigma_tiling.svg` — T1, T2, and the unique σ-symmetric tiling.

The tile mirror in these scripts is the affine involution
σ(p) = [[-1,-1],[0,1]]·p + (6,0) (P0↔P1, P2↔P3, M fixed). Note the trap:
the lattice swap (a,b)→(b,a) is an isometry of the triangular lattice but
does NOT stabilise this trapezoid; using it yields perfectly
self-consistent ghost solutions to the wrong problem. `sanity()` asserts
the involution and containment before any search.

Also checked (hexagon-centre / port lens, from other.png): the dissection
has 2 interior hexagon pairs and 5 boundary spines (ports (0,1), (2,0),
(5,0), (4,2), (2,3)); grammar systems anchored on ports alone, or ports+M,
are empty — the hexagon-centre walk works at one level but is not
hierarchically consistent.

## Running

Each script is standalone (`sfc_multitype.py` and `sfc_minimal.py` import
from `sfc_grammar.py`; run from this directory):

    python3 sfc_grammar.py     # single-type refutation + adjacency context
    python3 sfc_multitype.py   # GFP over arc-type systems, per regime
    python3 sfc_minimal.py     # minimal system, depth-3 verification, SVG

Possible next steps: OR-Tools `AddCircuit` formulation to maximise
edge-adjacent transitions jointly across the six productions; chaining across
the sibling d-cell / hexagon and octant seams (top type P0→P1 runs long-edge
end to end — the hexagon diameter — which is the natural interface); mapping
curve order onto the deployed address stream (d_cell = hex_digit + parent
region mode) as a stateful relabelling.
