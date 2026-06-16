## 10.0 Notation — the c/t/d/x grid taxonomy

  The constructions of §§6–8 generate four overlapping grid spaces, designated
  by single letters: **c**, **t**, **d**, **x**. The sections that follow use
  this vocabulary constantly; this section fixes it. (The authoritative
  glossary, maintained alongside the implementation, extends these definitions.)

  **t_cell** — a triangular cell of the refined field: the working unit of the
  simplicial carrier. Each t_cell carries an intrinsic mode (0 = ∇, 1 = Λ; §2)
  and subdivides into 9 child t_cells at the next level (§6). Within a parent
  context, a child t_cell occupies one of 12 positional classes, written as a
  **region** id (0–11): 6 classes are shared across both parent modes, 3 occur
  only under a mode-0 parent, and 3 only under a mode-1 parent.

  **c2** — the label (0, 1, 2) of a t_cell edge, assigned by edge gradient:
  0 = flat (horizontal), 1 = forward (positive slope), 2 = back (negative
  slope). Adjacent triangles agree on the c2 value of their common edge, and
  the c2 progression around any triangle is clockwise regardless of mode.
  *Etymology:* "c2" is shorthand for **"colouring 2"** — one of the two distinct
  three-colourings of the wallpaper group **p31m** (Grünbaum & Shephard 1986,
  *Tilings and Patterns*, §8.3 "Color Pattern Types", p. 433, type IH38;
  PP25[3]₁/p31m[3]₁ and PP25[3]₂/p31m[3]₂ — see the `grunbaum_shephard_*`
  figures, the second with the Hex9 d_cell overlay aligned). Note a **very 
  early naming mismatch**: despite the "2", the Hex9 c2 labelling actually 
  corresponds to Grünbaum & Shephard's *first* colouring, **p31m[3]₁**, 
  not the second.

  **c_cell** — a slot in the 96-position classifier grid: the raw output of
  the three-family inequality classification of §10a. Twelve of the 96 slots
  are the in-scope t_cell classes; the remainder are out of scope by
  construction.

  **d_cell** (half-hexagon) — three t_cells grouped by their shared long edge.
  The c2 value of that long edge is the d_cell's digit (d_dig ∈ {0, 1, 2}),
  and the d_cell inherits the mode of its t_cells. The d_cell is the
  fundamental domain of the tiling argument in §8.

  **x_cell** (hexagon) — one mode-0 d_cell joined with one mode-1 d_cell on
  their matching c2 edge. This is the public cell of the Hex9 grid: 12 root
  x_cells cover the globe, and each x_cell has 9 children.

  **x_dig** — the digit (0–8) naming a child x_cell within its parent. Read it
  in ternary: the high trit encodes mode ownership (0 = the child's mode-0 half
  is interior to the parent context; 1 = the mode-1 half is interior; 2 = the
  child is **split**, straddling two parents), and the low trit records the c2
  orientation of the child's long edge. The three split children per parent
  (x_dig ∈ {6, 7, 8}) are the only cells with two valid parents (§10b).

  **Lists and addresses** — c_list, t_list, and d_list are digit sequences over
  strict single-parent trees: they compose left-to-right and are
  prefix-sortable. An **x_list** is the sequence of x_digs; because of the
  split cells it is resolved right-to-left. An **x_adr** is an x_list plus a
  **tail** — a single metadata byte that resolves split-cell parentage and
  terminal state (§10b). The tail is metadata only: it never participates in
  geometric computation.

  [Figure: one parent triangle annotated with t_cells, c2 edges, d_cells, and
  the assembled x_cells with x_dig labels]
