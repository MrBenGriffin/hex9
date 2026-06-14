# Sphere warp retrain brief (astronomical)

*Prepared 2026-06-12, immediately after the F6 WGS84 retrain. Self-contained;
read `TRAINING_NOTES.md` (same directory) first — it holds the invariants,
traps, and validation gate that apply to every retrain. This file is just the
Sphere-specific execution plan.*

## Goal

Train and deploy `Sphere_l5_warp_data.npz` — the authalic warp for a perfect
sphere (`config.ELLIPSOIDS['Sphere']`: a = 6371008.8 m, f = 0), intended for
astronomical / celestial equal-area binning. Even at f = 0 the warp is needed:
it corrects the octahedral projection's own areal distortion, which dominates
(the WGS84 field differs from spherical only by the ~0.3% flattening term).
The radius only scales absolute m²; the warp itself is scale-invariant.

## Pre-flight

1. `git status` — the F6 work (loader changes in
   `hhg9/domains/octahedral_barycentric.py`, deployed WGS84 npz, validation
   scripts) may still be uncommitted from 2026-06-12. Don't lose it.
2. Archive anything matching `output/q_Sphere_l4_iter*.npz` (stage 1
   auto-resumes from those) and confirm no `Sphere_pkdo_warp.npz` exists in
   `hhg9/data/` (it would shadow the npz at load).
3. Extend `validate_f6.py`: it builds a default (WGS84) `Registrar` for its
   transect check [7]; add an `--ellipsoid` arg that calls
   `config.apply_ellipsoid(reg, name)` before projecting. Checks [1]–[6] are
   ellipsoid-agnostic already. Same applies to `probe_seam_band.py` and any
   area-stat run.

## Fast path (try first — untested but principled)

Skip stage 1: the spherical optimum is ~0.3% away from the WGS84 one, and
stage 2's `--warm-start` CT-interpolates a displacement from any
(source_pts, target_pts) npz.

```sh
cd experimental/sinkhorn
python stage2_gradient_l4.py --ellipsoid Sphere \
    --warm-start ../../hhg9/data/WGS84_l5_warp_data.npz
python stage3_gradient_l5.py --ellipsoid Sphere \
    --warm-start output/stage2/l4_tgrid_Sphere_best.npz
python stage4_pack.py output/stage3/l05_Sphere_best.npz --ellipsoid Sphere
```

`--ellipsoid Sphere` on EVERY stage — that's what switches the area objective
to f = 0. If stage 2 stalls or the warm-started MAE is poor (> ~2× the WGS84
baseline at the same stage), fall back to the full pipeline: copy
`retrain_f6.sh`, change the ellipsoid, run all four stages (~10 h for WGS84
on this machine; sphere from warm start should be far less).

## Gate, deploy, record

1. `python validate_f6.py --warp Sphere_l5_warp_data.npz --ellipsoid Sphere`
   (after the pre-flight extension) — must fully PASS. The loader's seam
   machinery (ghost padding + gradient projection) is field-agnostic and
   applies automatically; expect tangency/frame agreement at ~1e-16.
2. Deploy: `cp Sphere_l5_warp_data.npz ../../hhg9/data/`. Selected at runtime
   by `registrar.set_ellipsoid(name='Sphere', a=6371008.8, f=0.0)` (or
   `config.apply_ellipsoid(reg, 'Sphere')`).
3. Record canonical stats (ex0081w-style run under the Sphere registrar) in
   the paper notes / memory. Expect the same shape as WGS84 F6: σ ≈ 2e-4,
   balanced ±4% tail at the six octahedral vertices, p50 ≈ 0.0002%.
4. If libhex9 needs the spherical field too, export gradients per
   `libhex9/docs/warp-port-brief-f6-cside.md` (Option A) — the loader exposes
   `self.ct_dx` / `self.ct_dy` for exactly this.

## Astronomical wrinkle to confirm with Ben

Celestial coordinates (RA/Dec) are conventionally viewed from INSIDE the
sphere — if the application needs east-west mirroring relative to the
geographic convention, that belongs in the coordinate/domain layer, not in
the warp (the warp is x-mirror symmetric; train nothing differently).
Confirm the intended RA convention before wiring `g_gcd`-equivalents.
