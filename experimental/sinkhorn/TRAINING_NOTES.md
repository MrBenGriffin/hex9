# Warp-training playbook (post-F6, 2026-06-12)

For the next agent who trains a warp. Read this before touching the
pipeline. The F6 retrain (WGS84, 2026-06-12) is the reference run; its
provenance is `retrain_f6.sh` + `output/retrain_f6.log`. The **Sphere**
run (also 2026-06-12, provenance `retrain_sphere.sh` +
`output/retrain_sphere.log`) validated the warm-start fast path — see
"Sphere reference" at the end.

## How to run a full retrain

Copy `retrain_f6.sh` and change the ellipsoid. Drive the stages
explicitly, as it does — **do not use `run_pipeline.py` as-is**: its
resume/handoff globs predate the `output/stage2|3` layout (stage 2 writes
`output/stage2/l4_tgrid_<ELL>_best.npz`, stage 3
`output/stage3/l05_<ELL>_best.npz`; the orchestrator looks in the script
root and for `l05_<ELL>_k*.npz`). Also: stage 1 auto-resumes from any
`output/q_<ELL>_l4_iter*.npz` — archive those first or your "fresh" run
silently continues an old one.

`Sphere` is already registered in `config.ELLIPSOIDS` (a=6371008.8, f=0).
WGS84 took ~9.7 h end-to-end on the M-series box (stage 1 Sinkhorn L4
~100 outer iters, stage 2 Adam L4 5000 iters, stage 3 Adam L5 10000
iters, stage 4 pack).

**Warm-start fast path (PROVEN by the Sphere run):** a new field close
to an existing one (flattening contributes only ~0.3%) does not need
stage 1. Stage 2's `--warm-start` CT-interpolates a displacement from
ANY (source_pts, target_pts) file, so warm-start stage 2 from the
nearest deployed `hhg9/data/<ELL>_l5_warp_data.npz`, then run stages
3–4. Pass `--lr 5e-5` to stage 2 (its own guidance for near-converged
warm-starts; avoids the iter-0 transient). Make sure every stage gets
the target `--ellipsoid` — that is what the optimiser actually
corrects. Sphere from WGS84 took **57 min** end-to-end vs ~9.7 h full,
and reached the identical stage-2 L4 floor (0.00200) and final L5 MAE
(0.00067).

## Design invariants — do not regress these

1. **Lateral-edge vertices must SLIDE, never pin.** Stage 1's
   `snap_boundary_analytic` + ghost halo keeps them on the analytic edge
   but free tangentially; stages 2/3 freeze them (move_mask) while
   preserving the slide, and x-mirror-symmetrise everything. The old
   shipped field pinned them to zero next to km-scale interior deltas —
   that is what caused the F6 needle/sliver bug (CT bulged a normal
   component between the pinned vertices).
2. **Tangent vertex data is necessary but NOT sufficient.** The exact
   seam guarantee lives in the production loader
   (`hhg9/domains/octahedral_barycentric.py`, AuthalicWarp.__init__):
   lateral ghost bands + corner dihedral-orbit padding + edge-tangent
   gradient projection. It is field-agnostic — a new Sphere npz gets it
   automatically. Don't re-implement any of it in training.
3. **`raw` is the production edge mode.** Never ship a field expecting
   `feather` or `bypass`; with slide-bearing fields the feather is
   actively destructive (shears km of slide into 350 m).
4. Keep the trainer's exact x-mirror symmetry and equator dy=0 — the
   loader's guarantees assume both.

## Validation gate (all must pass before deploying)

```
python experimental/sinkhorn/validate_f6.py --warp <packed>.npz --ellipsoid <ELL>
```
(forces raw mode itself; checks vertex tangency, CT tangency ~1e-16,
cross-frame seam agreement, equator symmetry, gradient-spike scan,
round-trip, seam transects; `--ellipsoid` sets the registrar for the
transect check [7]). Then `H9_PROBE_ELLIPSOID=<ELL> python
probe_seam_band.py` (post-deploy) and `python area_stats.py
--ellipsoid <ELL> --depth 5` for canonical hex-level stats
(ex0081w-equivalent, dedupes the 3× hex emission, no plotting).

**Do not use as oracles:** `test_warp.py`, `warp.py`, and
`config.edge_blend` — they encode the abandoned pinned-edge design
(identity-on-edge assertions that a slide field correctly violates).

## Deploying

`stage4_pack.py <ckpt> --ellipsoid Sphere` →
`Sphere_l5_warp_data.npz`; copy to `hhg9/data/`. The b_oct domain loads
`{ellipsoid}_l5_warp_data.npz` by registrar ellipsoid name (WGS84
fallback) — and prefers `{ell}_pkdo_warp.npz` if one exists; make sure
none shadows your file. Archive the superseded npz under
`archive/` first.

Expect **global** address churn at fine levels after any retrain
(WGS84 F6: O(50–170 m) everywhere, ~5 km within a few km of seams), not
just near seams. Re-baseline anchor fixtures, and tell Ben before
merging.

## C-side parity

libhex9 must ship the same field. Easiest: export the final vertex
gradients (`AuthalicWarp` exposes `self.ct_dx` / `self.ct_dy`; first
`len(src)` rows of `.grad` are the real vertices) and let the C ACT
consume them directly — bit-exact parity without porting the ghost/
projection construction. Full brief:
`/Users/ben/Documents/Projects/libhex9/docs/warp-port-brief-f6-cside.md`.

## Reference numbers (WGS84 F6 canonical)

L5 training MAE 0.00066; hex-level (ex0081w, depth 5, 708,588 bins):
mean 0 (closure), min −3.56853%, max +4.80312%, log-ratio σ 1.99e-4,
|%dev| p50 0.00021% / p99 0.00444% / p99.99 0.43%. The ±4% tail is the
six octahedral-vertex cells (structural; the L4/L5 adaptive-grid idea
targets it). Edge-anisotropy p50 ≈33% and elongation p50 ≈12% are the
projection's inherent shape distortion — identical on/off seam, not a
warp defect. On-edge tangential slide peaks ~5.5 km a few cells off the
corners — steep corner boundary layers are EXPECTED, not ringing.

## RDE analysis tools (2026-06-12)

Per-hex deviation at ANY layer is the cell average of one fixed
pointwise field (the warp is frozen L5 CT), so finer-layer figures come
from field evaluation, not polygon enumeration:

- `rde_field.py --ellipsoid <ELL> --res 900 --spot-check 2000` —
  pointwise RDE ℓ, Tissot anisotropy (Jacobian SVD), warp strain, via
  central FD through the production b_oct→g_gcd chain onto the authalic
  sphere (area-exact for any ellipsoid). Spot-check anchors it to
  geographiclib hex areas (median agreement ~1 ppm). FD is h-robust
  3e-7…3e-5.
- `rde_subdivide.py --ellipsoid <ELL>` — exact triangle 9-tree (parent
  avg = mean of 9 equal-b_oct-area children): uniform per-depth
  deviation table + adaptive refinement that splits only UNRESOLVED
  cells (child spread > max(tol, rel·|dev|)), so finite-width hot bands
  are measured, not tiled. Depth-5 triangle row cross-checks the
  canonical hex stats.

Three regimes (pointwise). (1) Quiet interior — |dev| < 5e-4 over
~95.5–96% of area, p50 |ℓ| ~2e-5. (2) Seam boundary layers — REAL
sub-L5-lattice texture, wavelength ≈ the L5 cell (~2.7e-3 units ≈
19 km); lateral band ~59–70 km wide at 5e-4 falling to ~0.7–2.4 km at
1–4%. **Seam pointwise amplitude is ±3% (corner-adjacent seam rises to
~±5%) — NOT ±15%.** (3) Six corner cores: the ONLY ±15%+ feature, on
vanishing area — finite angle-dependent pointwise limit blowing up as
you refine into the vertex (res-900 grid sees ±15%; depth-10 subdivide
sees +72%; pole +55% on-axis, peak +74% at ~5 km, crater −91% within
~100 m). Hex σ climbs toward the pointwise σ ≈ 2.0e-3 as cells resolve
the texture. These corner cores are the irreducible target; the seam is
a modest ±3% swell.

**CORRECTION (2026-06-13):** an earlier draft of this section
attributed "±15%" to the seam. That was wrong — the ±15% lives in the
corner halos, not the seam. Provenance below; re-run to verify.

PROVENANCE OF THE NUMBERS (all WGS84 unless noted, 2026-06-13):
- Quiet interior 96.0% (= 1 − 0.0400 area-frac above 5e-4) and per-class
  band widths: `python experimental/sinkhorn/rde_subdivide.py
  --ellipsoid WGS84 --table-depth 6 --max-depth 10 --tol 5e-4 --rel
  0.25`. Writes `output/rde_tree_WGS84.npz`. Lateral-seam 2.72e-2 of
  octant at 5e-4 (~59 km), → 2.4 km at 1%; corner cells stay UNRESOLVED
  at depth 10 (spread 0.37); seam cells RESOLVED (spread ≤0.007) at
  ±2.7%/±3.1% (lateral/equator). Worst leaf +72.4% at lat 89.93°.
- Pointwise field + region split: `python
  experimental/sinkhorn/rde_field.py --ellipsoid WGS84 --res 900
  --spot-check 2000` → `output/rde_field_WGS84.npz` (keys: xy, ell,
  sig1, sig2, div, shear, emax). Global ℓ min/max −0.1447/+0.1513 at
  lat 89.7° (a POLAR CORNER). Loading that npz and classifying by
  feature (corner halo r=0.02, seam band 0.02): interior ±0.17%
  (aniso 1.41 median); seam-excl-corner ±3% (aniso ≤2.0); corner halo
  −13.5%…+16.3% (238/349260 samples, aniso ≤2.66). Spot-check vs
  geographiclib geodesic hex areas: median 1.3e-6.
- Sphere equivalents: same scripts with `--ellipsoid Sphere`; Sphere
  quiet interior 95.5%, same shape (Sphere ~0.5% noisier on the seam).

## Sphere reference (2026-06-12, deployed)

Warm-start fast path from the WGS84 F6 field; provenance
`retrain_sphere.sh` + `output/retrain_sphere.log`. 57 min end-to-end.
Stage-2 baseline MAE 0.00243 (better than WGS84's own 0.00454 stage-2
baseline), L4 floor 0.00200; final L5 training MAE 0.00066644; packed
max displacement 9.682e-3 b_oct units. validate_f6 full PASS
(tangency/frame agreement ~2e-16); seam probes continuous at
±89.1/51.48. Hex-level (`area_stats.py`, depth 5, 708,588 bins):
mean 0 (closure), min −3.53141%, max +4.91981%, log-ratio σ 2.019e-4,
|%dev| p50 0.00018% / p99 0.00470% / p99.99 0.387%. Same structural
±4–5% octahedral-vertex tail as WGS84. Gotcha fixed during the run:
`Registrar.ellipsoid_area` returned the WGS84 area for f=0 (formula
singular at e=0 + silent fallback) — now returns 4πa²
(`hhg9/base/registrar.py`).
