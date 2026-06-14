# Hex9 Terminology Glossary
*Working glossary — authoritative for code, paper, and external docs.*

---

## The Four Grid Levels

Hex9 has four overlapping grid spaces, each designated by a single letter: **c, t, d, x**.
Each level has a consistent set of named concepts: cell, digit, grid, list, address, polygon, parent, child.

---

### c — Classifier

| Term | Definition |
|---|---|
| `c_cell` | A cell in the 96-slot classifier grid. Identified by `c_dig`. Includes out-of-scope cells. |
| `c_dig` | Raw classifier cell ID: uint8, encoded as `h_id<<4 \| p_id<<2 \| n_id`. Range 0x00–0x5F. |
| `c_grid` | The full 96-position classification space. Contains all `c_cells`, including out-of-scope. |
| `c_list` | Ordered sequence of `c_dig` values, most-significant first. Left-to-right composable. |
| `c_adr` | `c_list` with root metadata. Self-contained. |

---

### t — Triangle

| Term | Definition |
|---|---|
| `t_cell` | An in-scope triangle cell; one of the 12 valid `c_cells` with a defined barycentre. Has an intrinsic mode (0=∇, 1=Λ) and a unique `t_dig`. |
| `t_dig` | Compact alias for a `t_cell`, remapped from `c_dig` to the range 0..11. The remapping keeps LUTs small and addresses readable. e.g. `t_dig 0` = `c_dig 0x49`, `t_dig 3` = `c_dig 0x34`. |
| `t_grid` | The 12-cell in-scope triangle grid. |
| `t_par` | A `t_cell` acting as parent of 9 child `t_cells` at the next layer. (Legacy name: `super_cell`.) |
| `t_chd` | A `t_cell` in its role as child of a `t_par`. |
| `t_list` | Ordered sequence of `t_dig` values. Left-to-right composable (strict single-parent tree). |
| `t_adr` | `t_list` with root metadata. Self-contained. |
| `t_poly` | The triangular polygon of a `t_cell`. |

**The 12 t_cells** comprise: 6 shared across both modes + 3 mode-0 only + 3 mode-1 only.

---

### d — Diploid (Half-hexagon)

Named for the *diploid* (dyakis dodecahedron) — the crystallographic form whose face structure underlies the two-mode H9 tiling.

| Term | Definition |
|---|---|
| `d_cell` | A half-hexagon cell. Composed of 3 `t_cells` (ordered: vertex → centre). Has an intrinsic mode and a `d_dig`. |
| `d_dig` | The c2 value (0, 1, or 2) identifying a `d_cell` within its `t_par`. Labels the c2 of the `d_cell`'s long edge. |
| `d_grid` | The half-hexagon grid. 3 `d_cells` per `t_par`. |
| `d_list` | Ordered sequence of `d_dig` values. Left-to-right composable (strict single-parent tree). |
| `d_adr` | `d_list` with root metadata. Self-contained. |
| `d_poly` | The half-hexagon polygon of a `d_cell`. |

**d_cell internal structure:** 3 `t_digs` (0..2), ordered from the triangle at the vertex of the `t_par` toward the centre.

**d_cell parent relationships:**
- Parent `t_dig` (= its c2): which of the 3 `d_cells` within its `t_par` it is
- Parent `x_dig` (mode): which half (mode 0 or mode 1) of its `x_cell` it provides

---

### x — Hexagon

| Term | Definition |
|---|---|
| `x_cell` | A hexagonal cell. Formed by joining one mode-0 `d_cell` with one mode-1 `d_cell` on their matching c2 (long) edge. The primary output cell of the Hex9 grid. |
| `x_dig` | Hexagon address digit (0..8) identifying an `x_cell` within its parent `x_cell`. The public-facing H9 address digit. Requires metadata for full interpretation (see `x_adr`). |
| `x_grid` | The hexagonal grid. 9 child `x_cells` per parent `x_cell`; 12 root `x_cells` (from 24 diploid faces / 2). |
| `x_list` | Ordered sequence of `x_dig` values, most-significant first. **Right-to-left** — must be unzipped bottom-up due to split `x_cells`. |
| `x_adr` | `x_list` + **tail**. Fully self-contained (see below). |
| `x_poly` | The hexagonal polygon of an `x_cell`. |

**Root count:** 12 root `x_cells` = 24 diploid faces ÷ 2. Both 12s (root cells and t_grid size) share this origin.

---

## Modes

| Mode | Symbol | Orientation | Description |
|---|---|---|---|
| 0 | ∇, V | Flat up, apex down | The "down" triangle. c2=0 `d_cell` touches top-right vertex of `t_par`. |
| 1 | Λ | Apex up, flat down | The "up" triangle. |

Mode is an intrinsic property of a `t_cell`. The c2=0 `d_cell` placement of mode 0 determines all other `d_cell` placements by symmetry. Mode-0 and mode-1 `d_cells` join on matching c2 edges to form `x_cells`.

---

## c2 Edges

The three edges of a `t_cell`, labelled by the gradient of the edge line:

| c2 | Edge | Gradient |
|---|---|---|
| 0 | flat | horizontal (zero gradient) |
| 1 | forward | positive slope (x = y style) |
| 2 | back | negative slope (x = −y style) |

Adjacent triangles in the plane share the same c2 value on their common edge. The c2 progression is always **clockwise**, regardless of mode.

---

## Cell vs Digit

- A **cell** is the mathematical space: a geometric region of a grid (`t_cell` = triangular region, `x_cell` = hexagonal region).
- A **digit** is the symbol used to name a cell within a parent's context.

These are distinct. The same `x_cell` may carry different `x_dig` values depending on which parent is canonical (the split `x_cell` case). `t_cells`, `d_cells` have unique digits (single-parent); `x_cells` may not (two-parent for split cells).

---

## Address Forms and Composability

### Composability

| Form | Direction | Single-parent? | Self-contained? |
|---|---|---|---|
| `c_list` | left → right | Yes | Yes |
| `t_list` | left → right | Yes | Yes |
| `d_list` | left → right | Yes | Yes |
| `x_list` | right → left | No (3 split cells/level) | No |
| `x_adr` | right → left | No (resolved by tail) | **Yes** |

`c_list`, `t_list`, `d_list` are prefix-sortable and support efficient spatial range queries without secondary indices.

`x_list` is right-to-left because 3 of the 9 `x_digs` per level (the split `x_cells`) have two valid parents. Parentage cannot be determined from the digit alone; the list must be unzipped bottom-up.

### x_adr = x_list + tail

The **tail** is a single metadata byte appended to the `x_list`. It is never an address step — using it as a digit would compute coordinates at layer L+1 and silently corrupt results. Two tail styles serve distinct purposes:

**Reversible tail** (8 bits): `p_mo` (bit 7) | `p_c2` (bits 6–5) | `r_mo` (bit 4) | `h` (bits 3–0)
- `p_mo`: parent mode of the terminal region. Required for exact centroid recovery of split x_cells (x_dig ∈ {6,7,8}), where the canonical mode-0 parent convention may not hold.
- `p_c2`: parent c2 of the terminal region. Needed to invert the terminal hex digit: digit 6 alone maps to multiple terminal regions depending on the parent c2 context.
- `r_mo`: root octant net_mode. Needed to recover the octant from the root hex digit; two octants of opposite mode can produce the same root hex digit.
- `h`: terminal region id (0..11). Identifies the specific d_cell (half-hexagon) within the terminal x_cell, enabling exact centroid recovery rather than x_cell-centre approximation.

Packed by `tail_pack_reversible(p_mo, p_c2, r_mo, h)`; invertible via `tail_unpack_reversible`.

**Key tail** (8 bits, high nibble carries data; low nibble is sentinel `0xF`): `p_c2` (bits 6–5) | `r_mo` (bit 4) | `0xF` (bits 3–0)
- Carries `p_c2` and `r_mo` only. For split x_cells, the mode-0 parent is canonical (p_mo implicit = 0). `h` is replaced by the sentinel `0xF`.
- Sufficient for unique binning: no two distinct x_cells produce the same (body, key_tail) pair.
- Not fully invertible: reconstructed centroid is the mode-0 canonical representative, not necessarily the exact d_cell centroid.

Packed by `tail_pack_key(p_c2, r_mo)`; derived from reversible by `tail_key_from_reversible` (strips p_mo and h, keeps p_c2 and r_mo).

**The tail is metadata only.** Only the `x_list` digits participate in geometric computations (polygon generation, coordinate reconstruction, offset calculation).

In packed format: reversible tail occupies two nibbles; key tail occupies one nibble (high nibble only).

### Split x_cells

3 of the 9 child `x_cells` per parent straddle the `d_cell` c2 boundary and have two valid parent `x_cells`.

**Convention:** the mode-0 parent is canonical.

The tail encodes which parent is intended without altering the geometric computation. This is analogous to a point on a timezone boundary: it belongs to one zone by convention, not by geometry.

---

## Legacy / Alternative Names

| Current | Legacy / Alias | Notes |
|---|---|---|
| `t_par` | `super_cell` | t_cell in parent role |
| `t_chd` | — | t_cell in child role |
| `d_cell` | `half_hex`, `h_cell` | 'h_cell' avoided: reads as 'hex cell' |
| `x_cell` | `hex`, `hexagon`, `cell` | OGC uses 'cell' for this |
| `x_dig` | `hex_digit` | 'hex_digit' avoided: reads as hexadecimal |
| `x_adr` | `h9_address` | |
| `c2` | `sub-face identifier` | OGC provisional |
| `t_grid` | `classifier plane` | |

---

## OGC / ISO Mapping (provisional)

| H9 term | OGC / ISO term |
|---|---|
| `x_cell` | cell |
| `x_dig` at layer L | refinement level L identifier |
| `x_grid` at layer L | DGGS at refinement level L |
| mode | face orientation / parity |
| `d_cell` | rhombic sub-cell (provisional) |
| aperture | 9 (ternary²) |
| `x_adr` | cell identifier (ISO 19111 / OGC Topic 21) |

*TODO: systematic review against OGC Topic 21 and ISO 19111 terminology lists.*
