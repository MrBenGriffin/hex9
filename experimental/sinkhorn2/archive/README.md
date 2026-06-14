# Archived training scripts

These scripts were side-quests or superseded stages of the authalic-warp
training pipeline. They are kept for provenance only. The live pipeline is the
`stage1..4` / `run_pipeline.py` set in the parent directory.

## Sinkhorn lineage

- **sinkhorn_01_l4t.py** — a near-identical copy of `sinkhorn_01_l4.py`. The
  only behavioural change is `t_pts_gcd = t_grid[gpts.coords]`, a broken index
  expression (the working form is the per-triangle gather). Abandoned tweak.
- **sinkhorn_01_polish.py** — seam-strip polish. Re-runs Sinkhorn with weight
  corrections restricted to the boundary strip, to attack the post-Sinkhorn
  distortion band 1–2 rings inside each octant edge. Superseded: the boundary
  artefact is now removed at source by pinning the lateral edges.
- **sinkhorn_02_gradient.py** — a faster alternative to `sinkhorn_01_polish.py`:
  analytic gradient descent on seam-strip vertices only. Superseded for the
  same reason; its gradient is the ancestor of the stage2/stage3 gradient.
- **sinkhorn_l3_tgrid.py** — an L3 dry-run of the L4 Adam t-grid solver, for
  fast logic testing. Production uses L4 (now `stage2_gradient_l4.py`).

## Gradient lineage (all L5 Adam, in evolution order)

- **gradient_l5.py** — base L5 Adam solver. Quadratic area loss; has a
  `SEAM_RINGS` / `build_seam_strip` mechanism that is a no-op stub (returns
  all-True). No mirror symmetry.
- **gradient_l5log.py** — same, with the log-ratio area loss.
- **gradient_l5log_froml4.py** — adds full interior mirror symmetry
  (dx antisymmetric, dy symmetric) and the shape dampener.
- *(gradient_mb_l5log_froml4.py — adds the mode-balance correction; this one is
  LIVE, renamed to `stage3_gradient_l5.py`.)*

## Why they were retired

The root cause of the catastrophic seam/apex round-trip error was that the
lateral octant edges were not preserved by the warp: `snap_boundary_analytic`
projected edge vertices onto the boundary line but let them slide tangentially,
and the inference Clough–Tocher interpolant then bulged off the edge. The seam
polish/strip scripts treated the symptom. The live pipeline removes the cause:
lateral-edge vertices are pinned to zero displacement during training, and the
inference warp multiplies the displacement by an edge-vanishing blend `φ`.
