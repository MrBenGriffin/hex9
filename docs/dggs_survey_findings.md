# DGGS survey — hex9 added, with an authalicity table

Ran your `dggs_survey` (skar enclosing-cone aspect ratio) with **hex9** added as a
fourth system alongside H3, S2 and A5, and extended it with a second table for
**authalicity** (cell-area uniformity). N = 5000 uniform-on-sphere cells per
system, seed 0xC0FFEE, gap_tol 1e-3. Both metrics are resolution-invariant for
all four systems (verified), so the survey level is purely a float-precision
choice: each system is read at its finest *numerically clean* resolution — H3 r15
and S2 L30 at their maxima, hex9 at L15 (its L29 leaf is ~60 nm) and A5 at r20
(its r30 boundary coords quantize and corrupt the area; see notes). **hex9's
authalic warp must be enabled** (`warp_init()` + `set_use_warp(True)`); it is off
by default, and the raw un-warped lattice is *not* equal-area (~7% area CV) — the
warp is the entire equal-area mechanism.

## Aspect ratio — enclosing-cone circularity (1.0 = circle, lower = rounder)

| system   | levels | cells | mean   | median | p99    | max    |
|----------|:------:|------:|-------:|-------:|-------:|-------:|
| H3 r15   |  0–15  |  5000 | 1.0624 | 1.0529 | 1.2057 | 1.2554 |
| S2 L30   |  0–30  |  5000 | 1.2428 | 1.2248 | 1.6081 | 1.7262 |
| hex9 L15 |  0–29  |  5000 | 1.3708 | 1.4117 | 1.5114 | 1.9060 |
| A5 r20   |  0–30  |  5000 | 2.1372 | 2.1333 | 2.2952 | 2.3181 |

## Authalicity — cell-area uniformity (CV% → 0 and ratios → 1.0 = ideal)

**Reference surface matters here.** skar and the lat/lon → vec3 conversion are
purely *spherical* (radius 6 371 008.8 m); the WGS84 ellipsoid is not used by the
solver. But the grids' I/O is WGS84 geodetic lon/lon, and they are not all
equal-area on the same surface — so we measure cell area two ways: the true
**WGS84 ellipsoidal** (geographiclib geodesic) area, *and* the spherical area, and
report both. The WGS84 figure is the physically meaningful "equal-area on the real
Earth" test and the common datum across all four systems, so it is primary.

`p50/p90/p95` are area percentiles divided by the mean (dimensionless, comparable
across cell sizes). Measured at a mid resolution (cells ~m²–km²): both the
geodesic and chord area computations floor for sub-millimetre cells, and
authalicity is resolution-invariant (next section), so a mid level is the clean,
representative choice.

| system | levels | CV% WGS84 | p50/μ  | p90/μ  | p95/μ  | max/min | CV% sphere |
|--------|:------:|----------:|-------:|-------:|-------:|--------:|-----------:|
| A5     |  0–30  |      0.01 | 1.0000 | 1.0000 | 1.0000 | 1.009   |       0.40 |
| hex9   |  0–29  |      0.10 | 1.0000 | 1.0001 | 1.0001 | 1.059   |       0.41 |
| H3     |  0–15  |     12.47 | 1.0153 | 1.1513 | 1.1669 | 2.002   |      12.46 |
| S2     |  0–30  |     14.48 | 1.0190 | 1.1828 | 1.2067 | 2.080   |      14.47 |

**A5 and hex9 form a genuine equal-area tier; H3 and S2 are not equal-area** — the
gap is ~100×, not a smooth gradient. A5 CV 0.01 %, hex9 CV 0.10 % (MAE 0.001 % over
the full L5 grid of 708,588 cells); H3 and S2 sit at 12–15 %.

Both equal-area grids are authalic on the **WGS84 ellipsoid**, not the sphere:
measuring on the sphere understates each (A5 0.40→0.01, hex9 0.41→0.10), and H3/S2
are datum-insensitive because they are not equal-area on either surface. (The
datum question is what surfaced this: skar and the vec3 step are purely spherical,
but the grids' I/O is WGS84 lon/lat.)

A5 vs hex9 is a fine distinction of *where the residual lives*. A5 spreads a hair
of deviation evenly — tightest worst-case (max/min 1.009). hex9 is **bulk-exact**:
p95/μ = 1.0001 (95 % of cells within 0.01 % of the mean), with all error pooled at
the **12 octahedral-vertex defects** (max/min 1.059, worst ≈ +4.8 % on the full
grid). Two shapes of "equal-area," both excellent — and hex9 reaches it while
staying far rounder than A5 (aspect 1.37 vs 2.14). So **hex9 is A5-tier on area
with markedly better shape**: among the equal-area pair it is the better combined
shape-and-area cell. (Aspect ratio is spherical throughout — skar is a spherical
solver and circularity is essentially datum-insensitive; the warp reshapes the
hex9 distribution slightly — median 1.37→1.41, p99 tightens, worst-case defects
elongate — but leaves the mean at 1.37.)

**Transitivity / robustness of the ranking.** All four dispersion measures
(CV, p90/μ, p95/μ, max/min) induce the *same* strict order
A5 < hex9 < H3 < S2 — the authalicity ranking is independent of which statistic
is used, and holds on both the sphere and the WGS84 ellipsoid. It is also
resolution-invariant: re-running 5 levels coarser reproduces every CV% and ratio
to 4 decimal places; only the mean cell area scales. Note the percentiles show
the *bulk* is
far tighter than `max/min`: e.g. hex9's p90/μ is 1.0001 against a max/min of 1.059,
so the extremes are driven by a handful of structural outliers (hex9's 12
octahedral-vertex defects, H3's 12 pentagons, S2's 8 cube corners), not the typical
cell. `max/min` is therefore the tail metric; CV and the percentiles describe
the population.

### Does the ellipsoid affect shape too?

Aspect ratio is measured on the sphere (skar is a spherical solver), so the
ellipsoid perturbs it as well — but only by the local N–S/E–W scale anisotropy
M/N = (1−e²)/(1−e²sin²φ), at most ~0.67 % at the equator and tapering to 0 at the
poles. Re-running with the geodetic→geocentric latitude correction (the dominant
ellipsoidal repositioning) confirms it is a 3rd–4th-decimal effect for every
system:

| system | AR sphere | AR geocentric | Δ       |
|--------|----------:|--------------:|--------:|
| H3     |    1.0621 |        1.0624 | +0.0003 |
| S2     |    1.2448 |        1.2450 | +0.0002 |
| A5     |    2.1362 |        2.1362 | +0.0001 |
| hex9   |    1.3734 |        1.3732 | −0.0002 |

(sensitivity-probe values; the Δ is the point, not the absolute AR). The shifts
are far below the 0.1–0.8 gaps between systems, so the shape ranking is
datum-independent. Unlike area — where A5 swung 30× because it is engineered to be
*exactly* equal-area on the ellipsoid — no grid targets an exact aspect ratio, so
there is nothing for the ~0.67 % distortion to amplify.

### Resolution span — mean cell area per layer (= Earth area / cell count)

| system | levels | base cells | coarsest (L0) | finest             | cells @ finest |
|--------|:------:|:----------:|--------------:|-------------------:|---------------:|
| H3     |  0–15  |    122     | 4.18e12 m²    | 0.895 m²   (r15)   |       5.70e14  |
| S2     |  0–30  |      6     | 8.50e13 m²    | 7.37e-5 m² (L30)   |       6.92e18  |
| A5     |  0–30  |     12     | 4.25e13 m²    | 2.95e-5 m² (r30)   |       1.73e19  |
| hex9   |  0–29  |     12     | 4.25e13 m²    | 9.02e-15 m² (L29)  |       5.65e28  |

Cell counts: H3 `2+120·7^r`, S2 `6·4^L`, A5 `60·4^(r−1)` for r≥1 (aperture 5
then 4), hex9 `12·9^L` (aperture 9). A5 and hex9 share 12 base cells, hence the
identical L0 area. Nominal means agree with the measured per-cell means above
(e.g. hex9 L15: 0.2064 nominal vs 0.2076 measured), an independent check.

## Notes on method

- **Enable hex9's authalic warp.** It is off by default in the binding; without
  `warp_init()` + `set_use_warp(True)`, `cell()` returns the raw un-warped lattice,
  whose area CV is ~7.4% (max/min ~1.4) — meaningless, since the warp *is* the
  equal-area mechanism. With it on, hex9 is equal-area to MAE 0.001% (CV 0.10%).
  Aspect ratio is nearly warp-independent (mean 1.37 either way; the warp only
  reshapes the distribution slightly). This footgun produced an earlier, retracted
  hex9 authalicity figure of 7.45%.
- **Aspect ratio** is straight from skar (`skar.solve(..., geo='vec3')`,
  `r.aspect_ratio`). hex9 plugs into the iterator protocol unchanged:
  `encode(lon,lat) → uuid`, `cell(uuid, layer, 0) → closed hex ring`.
- **Authalicity** is geometry-only (no skar) and computed two ways. Primary:
  **WGS84 ellipsoidal** area via geographiclib's geodesic polygon area (lon/lat
  recovered exactly from the vec3, since to_vec3 read them as spherical). This is
  the real-Earth equal-area test and the datum all four grids use for I/O.
  Secondary (sphere): planar-chord area `0.5·‖Σ wᵢ×wᵢ₊₁‖·R²`, preferred over
  spherical excess because at these cell sizes Girard's `Σangles − (n−2)π` cancels
  to noise.
- For the chord area, use **centroid-relative** vertices `wᵢ = vᵢ − v̄`. With
  absolute O(1) unit vectors, a sub-mm² cell's cross products cancel in the ~15th
  digit and the area floors — S2 L30 lost ~9% of cells and reported a bogus
  max/min ~96. Centred, S2 recovers all 5000 cells and matches its textbook
  cube-to-sphere ratio. The geodesic (WGS84) area has a *coarser* floor (it
  cannot resolve sub-mm² cells), so authalicity is read at a mid resolution where
  both methods are clean — legitimate since it is resolution-invariant.
- **Two systems are read below their max level**, both to dodge that same floor,
  and legitimately because both metrics are resolution-invariant:
  hex9 at L15 (L29 cells ~60 nm) and A5 at r20. A5 at r30 floors the *area* to a
  bogus 72% CV / max-min 4.05 (its boundary coords quantize at max res); its
  true authalicity is stable at 0.40% across r12/r20. Aspect ratio is unaffected
  at r30 (a ratio cancels the quantization), staying ~2.14 across r5–r30.

Reproducer: `tools/dggs_survey_hex9.py` (single file; surveys hex9 always, plus
H3/S2/A5 when importable).
