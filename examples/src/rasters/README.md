# Demo rasters for ex0251 / ex0252

Small public-domain crops, committed so the GeoTIFF examples run from a clean
checkout. About 1.2 MB in total, against roughly 13 GB for the full sources.

| file | example | source | resolution |
|---|---|---|---|
| `ex0251_nlcd_ak_2016.tif` | ex0251 | NLCD 2016 Land Cover, Alaska (USGS/MRLC) | 30 m (native) |
| `ex0251_hillshade.tif` | ex0251 | hillshade over a USGS DEM | 5 m (native) |
| `ex0251_geology.tif` | ex0251 | Chiginagak surface-geology survey (USGS GEMS) | 5 m (native) |
| `ex0251_geology.json` | ex0251 | rock-unit label table for the above | — |
| `ex0252_nlcd_2024.tif` | ex0252 | Annual NLCD 2024 Land Cover, CONUS (USGS/MRLC) | 30 m (native) |
| `ex0252_hillshade.tif` | ex0252 | hillshade over a USGS DEM | 1 m (native) |

Every raster is at its **source resolution**. The saving is entirely in the
window: each is cropped to the area its example actually samples, plus 12%.

All are works of the US Government, or derived from them, and are in the
public domain (17 U.S.C. §105). See `NOTICE.md` at the repository root.

## Sizing a crop: match the LayerSpec level, not the base layer

Each layer is sampled **once per hexagon centroid**, so a raster wants to be at
least as fine as the hexagons it fills. The trap is that the layers do not all
render at the loop's `layer`: in both examples the hillshade is drawn at
`layer + 2`, two levels finer, and Hex9 has aperture 9 — so those hexagons are
**nine times smaller in area, three times smaller across** than the base layer's.

| example | base layer | hillshade at | hex across-flats | hillshade source |
|---|---|---|---|---|
| ex0251 | L10 (118.6 m) | **L12** | 13.18 m | 5 m → ~2.6 samples/hex |
| ex0252 | L12 (13.18 m) | **L14** | 1.47 m | 1 m → ~1.5 samples/hex |

Sizing against the base layer instead gives rasters far too coarse: 20 m and
5 m hillshades put one raster cell across 2.3 and 11.6 hexagons respectively,
which shows up as blocked, flat-looking relief — subtle for ex0251, obvious for
ex0252. Neither hillshade can usefully be coarsened; ex0252's is already at the
source's 1 m limit.

The land-cover layers are the reverse case. NLCD is 30 m natively, which is
coarser than ex0252's L12 hexagons (13.18 m); that is a limit of NLCD, not
something a re-cut can fix.

Cropping the *window* is where the saving comes from — the examples sample only
about 12.6 × 9.7 km (ex0251) and 1.45 × 1.13 km (ex0252). At source resolution
those windows total ~5.5 MB, against ~13 GB for the full datasets.

Two consequences worth knowing:

- **The crops are window-specific.** They cover each example's `centre` and
  `size` plus a margin (40% for ex0251, 60% for ex0252). Move the window, or
  raise the level much beyond the ones above, and you must re-cut.
- **`ex0251_geology.tif` covers less than the plotted area, by design.** The
  survey it derives from covered the mountain itself — about 5.9 km around the
  summit, against ex0251's 7.5 km half-width — so hexagons near the frame edge
  legitimately have no rock label. It is shipped whole rather than padded to
  the window. Do not "fix" this.

## Re-cutting from the full sources

The originals are not in the repository. NLCD comes from MRLC
(<https://www.mrlc.gov/data>); the DEMs behind the hillshades are USGS 3DEP
(<https://www.usgs.gov/3d-elevation-program>), hill-shaded locally; the geology
is from the USGS GEMS package for Chiginagak.

Given a full source, a crop is one `gdal_translate`. Compute the window in the
source CRS about the example's centre, then:

```sh
gdal_translate -projwin <ulx> <uly> <lrx> <lry> \
               -co COMPRESS=DEFLATE -co PREDICTOR=2 -co ZLEVEL=9 -co TILED=YES \
               full_source.tif ex0251_hillshade.tif
```

Use `-projwin_srs EPSG:4326` to give the window in lat/lon rather than in the
source projection. Add `-tr <res> <res> -r average` only if you have checked the
resolution against the table above — every shipped raster is native.

To find the window a changed `centre`/`size` needs, instrument the sampler
rather than deriving it: wrap `hhg9.geo.gdal.sample_gdal`, record the min/max of
the coordinates passed to it, and run the example against the full sources. The
extents above were measured that way, and each shipped crop was then verified to
contain every sampled point with 88–776 m to spare.
