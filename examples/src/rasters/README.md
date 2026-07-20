# Demo rasters for ex0251 / ex0252

Small public-domain crops, committed so the GeoTIFF examples run from a clean
checkout. About 1.2 MB in total, against roughly 13 GB for the full sources.

| file | example | source | native → shipped |
|---|---|---|---|
| `ex0251_nlcd_ak_2016.tif` | ex0251 | NLCD 2016 Land Cover, Alaska (USGS/MRLC) | 30 m → 30 m |
| `ex0251_hillshade.tif` | ex0251 | hillshade over a USGS DEM | 5 m → 20 m |
| `ex0251_geology.tif` | ex0251 | Chiginagak surface-geology survey (USGS GEMS) | 5 m → 20 m |
| `ex0251_geology.json` | ex0251 | rock-unit label table for the above | — |
| `ex0252_nlcd_2024.tif` | ex0252 | Annual NLCD 2024 Land Cover, CONUS (USGS/MRLC) | 30 m → 30 m |
| `ex0252_hillshade.tif` | ex0252 | hillshade over a USGS DEM | 1 m → 5 m |

All are works of the US Government, or derived from them, and are in the
public domain (17 U.S.C. §105). See `NOTICE.md` at the repository root.

## Why these are cropped and downsampled

Both examples sample each layer **once per hexagon centroid**, so raster detail
finer than the hex spacing is thrown away. ex0251 renders L10 (~110 m across a
hexagon) and ex0252 renders L12 (~26 m), which makes the 5 m and 1 m hillshades
roughly 20× oversampled in each direction. Downsampling to 20 m and 5 m is
therefore invisible in the output while cutting ~17 MB to ~1.2 MB.

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
               -tr 20 20 -r average \
               -co COMPRESS=DEFLATE -co PREDICTOR=2 -co ZLEVEL=9 -co TILED=YES \
               full_source.tif ex0251_hillshade.tif
```

Omit `-tr`/`-r` to keep native resolution (the NLCD layers are shipped native).
Use `-projwin_srs EPSG:4326` if you would rather give the window in lat/lon
than in the source projection.
