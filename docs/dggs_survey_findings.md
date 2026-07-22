# DGGS survey — six systems, with an authalicity table

Ran the `dggs_survey` (skar enclosing-cone aspect ratio) across **six** systems —
H3, S2, A5, **ISEA3H** (via dggal), **HEALPix** (via healpy) and **hex9** — with a
second table for **authalicity** (cell-area uniformity). N = 5000 uniform-on-sphere
cells per system, seed 0xC0FFEE, gap_tol 1e-3. Both metrics are
resolution-invariant for all six systems (verified: a 5-level-coarser run
reproduces every statistic to 4 decimal places — including across ISEA3H's
odd/even level parity, whose grids are rotated 30° relative to each other), so
the survey level is purely a float-precision choice: each system is read at its
finest *numerically clean* resolution — H3 r15 and S2 L30 at their maxima, hex9
at L15 (its L29 leaf is ~60 nm), A5 at r20 (its r30 boundary coords quantize and
corrupt the area), ISEA3H at L20 (dggal max 33) and HEALPix at nside 2^15 (max
2^29, where boundary step-points collapse like hex9's L29).

Re-run 2026-07-22 against current hex9 (the binding now enables the authalic
warp at import — the old opt-in `set_use_warp` footgun is closed upstream).
hex9's numbers moved slightly with the code: worst-case aspect improved
1.906 → 1.691 (p99 1.51 → 1.55, mean unchanged at 1.37); area CV 0.10 → 0.12 %.

## Aspect ratio — enclosing-cone circularity (1.0 = circle, lower = rounder)

`min` is the roundest sampled cell — each system's best case — so min…max
brackets the observed roundness range.

| system      | levels | cells | min    | mean   | median | p99    | max    |
|-------------|:------:|------:|-------:|-------:|-------:|-------:|-------:|
| H3 r15      |  0–15  |  5000 | 1.0000 | 1.0624 | 1.0529 | 1.2057 | 1.2554 |
| ISEA3H L20  |  0–33  |  5000 | 1.0844 | 1.1839 | 1.1675 | 1.3334 | 1.3539 |
| S2 L30      |  0–30  |  5000 | 1.0020 | 1.2428 | 1.2248 | 1.6081 | 1.7262 |
| HPX n2^15   |  0–29  |  5000 | 1.0004 | 1.3603 | 1.1770 | 2.3563 | 2.4273 |
| hex9 L15    |  0–29  |  5000 | 1.0032 | 1.3712 | 1.4115 | 1.5464 | 1.6913 |
| A5 r20      |  0–30  |  5000 | 1.9871 | 2.1372 | 2.1333 | 2.2952 | 2.3181 |

The `min` column splits the systems into two families. H3, S2, HEALPix and hex9
all *touch the circle* — their best cells are essentially round (1.000–1.003) and
their distributions spread upward from there. ISEA3H and A5 are *bounded away*
from it: no ISEA3H cell is rounder than 1.084, no A5 cell rounder than 1.987.
Bounded-away cuts both ways — A5's band is narrowly bad (2.0–2.3), while
ISEA3H's is narrowly *good*: min 1.084 to max 1.354 is the tightest total range
of any system, the most shape-consistent grid surveyed.

Two means are close but the distributions are not: HEALPix (1.360) and hex9
(1.371). HEALPix is bimodal — its equatorial diamonds are round (median 1.177)
but its polar cells elongate badly (p99 2.36 against a median of 1.18 — a
median-to-p99 spread ~9× hex9's), a *population* feature of the polar caps, not
a handful of outliers.
hex9 is the reverse: a higher median (1.41) but a short tail (max 1.69) —
every hex9 cell is rounder than HEALPix's worst percentile.

## Authalicity — cell-area uniformity (CV% → 0 and ratios → 1.0 = ideal)

**Reference surface matters here.** skar and the lat/lon → vec3 conversion are
purely *spherical* (radius 6 371 008.8 m); the WGS84 ellipsoid is not used by the
solver. But the grids' I/O is WGS84 geodetic lon/lat, and they are not all
equal-area on the same surface — so we measure cell area two ways: the true
**WGS84 ellipsoidal** (geographiclib geodesic) area, *and* the spherical area, and
report both. The WGS84 figure is the physically meaningful "equal-area on the real
Earth" test and the common datum across all six systems, so it is primary.

`p50/p90/p95` are area percentiles divided by the mean (dimensionless, comparable
across cell sizes). Measured at a mid resolution (cells ~m²–km²): both the
geodesic and chord area computations floor for sub-millimetre cells, and
authalicity is resolution-invariant (below), so a mid level is the clean,
representative choice. (S2's area row is read at L25 for exactly this reason:
at L30 its ~mm² cells sit below the geodesic-area floor and report a bogus
CV 17 % / max/min 8.5. At L25 the floor clears and the numbers are stable.)

Three sphere-side columns need care. `geoCV%` re-reads each grid's geodetic
lon/lat as if it were spherical — a *datum mismatch* for grids that handle the
ellipsoid internally — and is kept only for comparability. `natCV%` measures
each grid on the sphere it *natively addresses*: for A5, ISEA3H and hex9 (†)
that is the **WGS84 authalic sphere** — all three convert geodetic latitude to
authalic latitude internally and do their equal-area work there (hex9 ≥2.0.0:
analytic series + Sphere-trained warp as its single regime; dggal's ISEA3H:
Snyder on the authalic sphere, its `latGeodeticToAuthalic` matching our
closed-form q-function to <1e-11°; A5: same design) — while for H3, S2 and
HEALPix the native sphere is the coordinate sphere itself (natCV = geoCV).

| system  | levels | CV% WGS84 | p50/μ  | p90/μ  | p95/μ  | max/min | geoCV% | natCV% |
|---------|:------:|----------:|-------:|-------:|-------:|--------:|-------:|-------:|
| A5      |  0–30  |      0.00 | 1.0000 | 1.0000 | 1.0000 | 1.000   |   0.40 |  0.00† |
| ISEA3H  |  0–33  |      0.00 | 1.0000 | 1.0000 | 1.0000 | 1.000   |   0.40 |  0.00† |
| hex9    |  0–29  |      0.12 | 1.0000 | 1.0001 | 1.0001 | 1.087   |   0.42 |  0.12† |
| HEALPix |  0–29  |      0.40 | 0.9989 | 1.0064 | 1.0077 | 1.014   |   0.00 |  0.00  |
| H3      |  0–15  |     12.47 | 1.0153 | 1.1512 | 1.1669 | 2.002   |  12.46 |  12.46 |
| S2      |  0–30  |     14.48 | 1.0189 | 1.1828 | 1.2067 | 2.080   |  14.47 |  14.47 |

† natCV% measured on the WGS84 **authalic sphere** — the surface these three
grids address internally (geodetic→authalic latitude conversion built in), so
for them geoCV% is a datum mismatch, not grid distortion. Unmarked systems
address the coordinate sphere directly, so their natCV% = geoCV%.

**The equal-area tier now has four members — A5, ISEA3H, hex9, HEALPix — and H3
and S2 remain ~30–100× away**; the gap is a cliff, not a gradient. An earlier
draft read the geoCV column as a symmetric ±0.4 % "cost of the other surface";
the native-sphere column shows that was an artifact. A5, ISEA3H and hex9 are
equal-area on **both** their surfaces at once — exactly on the ellipsoid *and*
on their authalic sphere (natCV reproduces each grid's WGS84 CV to the last
digit, as it must: the authalic map preserves relative areas by construction,
and hex9's 0.12 % vertex-defect residual is surface-independent). The ~0.4 %
in geoCV is purely the geodetic-vs-spherical latitude reparameterization — the
area-element spread (≤ ~0.67 % point-wise, smooth in latitude), not grid
distortion. The real asymmetry in the tier is *who ships the datum handling*:
the trio build the authalic conversion in; HEALPix (astronomy heritage) leaves
the datum to the caller, so fed geodetic coordinates naïvely it genuinely
carries 0.40 % CV on the real Earth (its WGS84 max/min of 1.014 is that smooth
latitude factor, with no structural outliers) — though a user who pre-applies
the authalic map themselves recovers exactness there too. H3 and S2 are
datum-insensitive because they are not equal-area on any surface.

Within the ellipsoid-native trio the distinction is *how* each achieves it. A5
and dggal's ISEA3H are exactly equal-area by construction (measured CV 0.00 %,
the residual at the geodesic-area measurement floor — Snyder's ISEA projection
is analytically equal-area, applied by dggal on the authalic sphere). hex9 is
**bulk-exact** numerically through its corrective warp: p95/μ = 1.0001 (95 % of
cells within 0.01 % of the mean), with all error pooled at the **12
octahedral-vertex defects** (max/min 1.087). Caveat for the exact pair: the 12
icosahedral-vertex pentagons (5/6 of a hexagon's area in ISEA3H) were not
sampled — 5000 draws from ~3.5 × 10¹⁰ cells — so their max/min 1.000 describes
the hexagonal population, just as H3's tail metric excludes its 12 pentagons.

**Shape and area together.** ISEA3H is the strongest combined cell in this
survey — exact equal-area *and* the tightest shape band (1.08–1.35) — with the
caveat that cell distortion is only one axis: its aperture-3 hierarchy is the
weakest surveyed on hierarchical coherence (lineage/commutation drift of
44–57 %, growing with depth — see `docs/dggs/dggs_commute.py` and the transport
note). hex9 remains A5-tier on area with markedly better shape than A5 (1.37 vs
2.14), and rounder in the worst case than HEALPix's 99th percentile. Among
equal-area systems the shape order is ISEA3H < hex9 ≲ HEALPix (mean) < A5, with
HEALPix's polar tail the largest shape excursion in the tier.

**Transitivity / robustness of the ranking.** All four dispersion measures
(CV, p90/μ, p95/μ, max/min) induce the *same* strict order
A5 ≈ ISEA3H < hex9 < HEALPix < H3 < S2 on the ellipsoid — the authalicity
ranking is independent of which statistic is used (on native spheres the tier
compresses to A5 ≈ ISEA3H ≈ HEALPix at 0.00 < hex9 at 0.12, and H3 < S2 is
unchanged). It is also
resolution-invariant: re-running 5 levels coarser reproduces every CV% and ratio
to 4 decimal places; only the mean cell area scales. Note the percentiles show
the *bulk* is far tighter than `max/min`: e.g. hex9's p90/μ is 1.0001 against a
max/min of 1.087, so the extremes are driven by a handful of structural outliers
(hex9's 12 octahedral-vertex defects, H3's 12 pentagons, S2's 8 cube corners),
not the typical cell. `max/min` is therefore the tail metric; CV and the
percentiles describe the population. HEALPix inverts the pattern: its max/min
(1.014) is *smooth* latitude dependence with no outlier class at all — but its
shape tail is populational (the polar caps), the mirror image of the
hexagonal systems' area tails.

### Does the ellipsoid affect shape too?

Aspect ratio is measured on the sphere (skar is a spherical solver), so the
ellipsoid perturbs it as well — but only by the local N–S/E–W scale anisotropy
M/N = (1−e²)/(1−e²sin²φ), at most ~0.67 % at the equator and tapering to 0 at the
poles. Re-running with the geodetic→geocentric latitude correction (the dominant
ellipsoidal repositioning) confirms it is a 3rd–4th-decimal effect for every
system probed:

| system | AR sphere | AR geocentric | Δ       |
|--------|----------:|--------------:|--------:|
| H3     |    1.0621 |        1.0624 | +0.0003 |
| S2     |    1.2448 |        1.2450 | +0.0002 |
| A5     |    2.1362 |        2.1362 | +0.0001 |
| hex9   |    1.3734 |        1.3732 | −0.0002 |

(sensitivity-probe values from the original four-system run; the Δ is the point,
not the absolute AR. ISEA3H and HEALPix were not re-probed but sit in the same
regime — the perturbation is a property of the datum, not the grid.) The shifts
are far below the 0.1–0.8 gaps between systems, so the shape ranking is
datum-independent. Unlike area — where A5 swung 30× because it is engineered to
be *exactly* equal-area on the ellipsoid — no grid targets an exact aspect
ratio, so there is nothing for the ~0.67 % distortion to amplify.

### Resolution span — mean cell area per layer (= Earth area / cell count)

| system  | levels | base cells | coarsest (L0) | finest              | cells @ finest |
|---------|:------:|:----------:|--------------:|--------------------:|---------------:|
| H3      |  0–15  |    122     | 4.18e12 m²    | 0.895 m²    (r15)   |       5.70e14  |
| S2      |  0–30  |      6     | 8.50e13 m²    | 7.37e-5 m²  (L30)   |       6.92e18  |
| A5      |  0–30  |     12     | 4.25e13 m²    | 2.95e-5 m²  (r30)   |       1.73e19  |
| ISEA3H  |  0–33  |     12     | 4.25e13 m²    | 9.17e-3 m²  (L33)   |       5.56e16  |
| HEALPix |  0–29  |     12     | 4.25e13 m²    | 1.47e-4 m²  (n2^29) |       3.46e18  |
| hex9    |  0–29  |     12     | 4.25e13 m²    | 9.02e-15 m² (L29)   |       5.65e28  |

Cell counts: H3 `2+120·7^r`, S2 `6·4^L`, A5 `60·4^(r−1)` for r≥1 (aperture 5
then 4), ISEA3H `2+10·3^L` (aperture 3; L0 = the 12 pentagons, verified against
dggal `countZones`), HEALPix `12·4^k` (nside = 2^k), hex9 `12·9^L` (aperture 9).
Four systems now share the 12-base-cell coarsest level — A5 and hex9 with 12
full cells, ISEA3H with its 12 pentagons, HEALPix with its 12 base quads — hence
the identical L0 area. Nominal means agree with the measured per-cell means
above (e.g. hex9 L15: 0.2064 nominal vs 0.2065 measured; ISEA3H L20: 1.4629e4
nominal vs 1.463e4; HEALPix n2^15: 3.9586e4 vs 3.959e4), an independent check.

## Notes on method

- **hex9's authalic warp is now on by default.** Since the 2026-07 binding,
  `hex9_ext` initialises the warp at import (and raises if the warp blob is
  missing) — `cell()` always returns warped, equal-area geometry. The old
  opt-in API (`set_use_warp`) is gone, closing the footgun that once produced a
  retracted raw-lattice authalicity figure of 7.45%. Aspect ratio was always
  nearly warp-independent (mean 1.37 either way).
- **Aspect ratio** is straight from skar (`skar.solve(..., geo='vec3')`,
  `r.aspect_ratio`). hex9 plugs into the iterator protocol unchanged:
  `encode(lon,lat) → uuid`, `cell(uuid, layer, 0) → closed hex ring`. The `min`
  column is the roundest converged cell per system (the script also prints the
  best/worst cell ids).
- **ISEA3H** uses dggal's *refined* boundary (`getZoneRefinedWGS84Vertices`,
  ~30–60 points per zone), not the 6 corners: ISEA edges are straight in the
  Snyder projection plane, hence curved on the sphere — corners alone understate
  both the area and the enclosing cone. dggal applies ISEA on the authalic
  sphere, which is why its zones are exactly equal-area *on the ellipsoid*.
- **HEALPix** uses healpy nested pixels with `boundaries(step=12)` (48 unit-vec3
  points per cell, no lat/lng round-trip): HEALPix edges are not geodesics
  (constant-z and spiral arcs), so corners alone misstate area and cone here
  too. Its angles are read as geodetic lon/lat like every other system — the
  common-datum convention of this survey; HEALPix is sphere-native equal-area.
- **Authalicity** is geometry-only (no skar) and computed two ways. Primary:
  **WGS84 ellipsoidal** area via geographiclib's geodesic polygon area (lon/lat
  recovered exactly from the vec3, since to_vec3 read them as spherical). This is
  the real-Earth equal-area test and the datum all six grids use for I/O.
  Secondary (sphere): planar-chord area `0.5·‖Σ wᵢ×wᵢ₊₁‖·R²`, preferred over
  spherical excess because at these cell sizes Girard's `Σangles − (n−2)π` cancels
  to noise.
- **Native-sphere areas** (natCV%) map each boundary's geodetic latitudes to
  authalic latitudes (closed-form q-function at WGS84 e², verified against
  dggal's `latGeodeticToAuthalic` series to <1e-11°) before the same chord-area
  computation. Note neither grid exposes a runtime ellipsoid *option* any more:
  hex9 1.x had a selectable WGS84-trained warp field, removed in 2.0.0 because
  one point could carry two addresses (one regime now — authalic series +
  Sphere warp; see libhex9 docs/warp-regimes.md), and dggal's DGGRS objects
  have no datum toggle — only the public latitude-conversion functions, which
  is exactly what this survey uses.
- For the chord area, use **centroid-relative** vertices `wᵢ = vᵢ − v̄`. With
  absolute O(1) unit vectors, a sub-mm² cell's cross products cancel in the ~15th
  digit and the area floors — S2 L30 lost ~9% of cells and reported a bogus
  max/min ~96. Centred, S2 recovers all 5000 cells and matches its textbook
  cube-to-sphere ratio. The geodesic (WGS84) area has a *coarser* floor (it
  cannot resolve sub-mm² cells), so authalicity is read at a mid resolution where
  both methods are clean — legitimate since it is resolution-invariant.
- **Three systems are read below their max level**, all to dodge float floors,
  and legitimately because both metrics are resolution-invariant: hex9 at L15
  (L29 cells ~60 nm), A5 at r20 (r30 floors the *area* to a bogus 72% CV — its
  boundary coords quantize; aspect ratio survives, staying ~2.14 across
  r5–r30), and HEALPix at nside 2^15. S2 is surveyed at L30 for shape but its
  area row is read at L25 (see above).

Reproducer: `tools/dggs_survey_hex9.py` in libhex9 (single file; surveys hex9
always, plus H3 / S2 / A5 / ISEA3H (dggal) / HEALPix (healpy) when importable;
`argv[1]` coarsens every system by N levels for the invariance check).
