# Hex9 examples — index

Ninety-nine scripts accumulated as a proof-of-concept progression rather than a
course. They are grouped by number, and the numbering is meaningful: each band
below is a coherent stage. Nothing here is renumbered, because the numbers are
cited from the paper, `docs/paper_figures.md` and the notes.

Run them from this directory (`cd examples && python ex0251_geotiff.py`), though
the GeoTIFF examples now resolve their own paths and work from anywhere.

## Start here

A reading order for someone meeting Hex9 for the first time. Six scripts, in
this order, cover the whole construction:

| | example | what it shows |
|---|---|---|
| 1 | `ex0042_plate_oct.py` | the bare octahedral projection — the thing everything else stands on |
| 2 | `ex0045_plate_net.py` | how the octahedron unfolds to a flat net (needs a cache; see below) |
| 3 | `ex0062_one_addr.py` | a single address, walked through digit by digit |
| 4 | `ex0110_poly_neighbours.py` | half-hexagons and how neighbours work |
| 5 | `ex0118_mollweide.py` | equal-area quality over the whole globe |
| 6 | `ex0252_geotiff.py` | the payoff — real raster data composed onto hexagons |

## Bands

**ex0010–ex0046 — foundations and projection.** Plate-carrée pixel handling
(`ex0010`, `ex0011`), globe and sphere (`ex0020`, `ex0030`), the projection
cache (`ex0041`), the octahedral projection (`ex0042`), the net (`ex0045`), the
warp matrix (`ex0046`). `ex0010`–`ex0030` are marked "supporting system test —
no octahedral projection": they exercise the raster plumbing only.

**ex0051–ex0065 — addressing, seams, and octant analysis.** Addresses and
round-trips (`ex0060`–`ex0062`), seam stitching (`ex0059`), vertices
(`ex0064`), and the single-octant area-deviation family (`ex0063*`, `ex0065*`).
`ex0065*` is really a Chebyshev/PKDO parameterisation study — closer in spirit
to `experimental/` than to a demo, kept here in case warp-as-parameterisation
is revisited.

**ex0075–ex0099 — sampling, authalics, zooms.** Barycentric and grid sampling
(`ex0094`–`ex0097`), the authalic deviation runs (`ex0080`, `ex0080w`,
`ex0081w` — the canonical warp statistics come from these), and zoom sequences
(`ex0098`, `ex0099`).

**ex0101–ex0124 — grids, meshes, and global metrics.** Hexagon identity as
colour (`ex0101`, `ex0102`), neighbours (`ex0110`–`ex0113`), direct hex meshes
(`ex0114`–`ex0117`), and the global quality figures: Mollweide (`ex0118`),
GeoPackage export (`ex0119`), Tissot (`ex0120`, `ex0121`), printable tiles
(`ex0122_hex_tiles`), warp continuity check (`ex0122_warp_check`), and the
vertex census (`ex0124`).

**ex0200–ex0263 — applied composition.** Heatmaps (`ex0200`), GeoTIFF
composition (`ex0250`–`ex0253`), polygon-clipped grids (`ex0260`, `ex0261`),
seam handling (`ex0262`, `ex0263_*_seam_*`), and the Bhutan population
choropleth (`ex0263_bhutan_big`).

**ex0300–ex0310 — export and backdrops.** OSM backdrops (`ex0300`), direct-RGB
fills (`ex0301`), GeoPackage/GeoTIFF export (`ex0302`), Stonehenge
(`ex0310_*`).

**ex0400 — paper figures.** `ex0400_anatomy.py` generates the F3/F4/F5/F7
figure family.

**ex4000–ex4003 — algorithmic proof-of-concept.** uint64 packing, AK
round-trips, addressing and plate-pixel fidelity. These are verification
scripts, not demonstrations.

**ex_1xxxx — scratch probes.** Small single-question checks accumulated
alongside the work: octant ids, region round-trips, c2 colouring, sizes, modes,
hex-step, densify, uuid, and notably `ex_10082_postgis.py` and
`ex_10093_moore.py`, which are the entry points for the PostGIS and Moore-curve
material otherwise living in libhex9.

## What each example needs

Most run against nothing but the repository. The exceptions:

| example | needs | status |
|---|---|---|
| `ex0251`, `ex0252` | `src/rasters/*` | committed, runs from a clean checkout |
| `ex0121`, `ex0122_hex_tiles` | Natural Earth layers | downloaded on demand, prompts first |
| `ex0041`, `ex0045` | `src/tissot_3600x1800.npz` | **not in the repo** — 179 MB projection cache, regenerate with `ex0041_cache.py` (~10 min). It fails with a message telling you so. |
| `ex0098` | `src/ex0098/` frames | **not in the repo** — the source imagery for the zoom video is not redistributed |
| `ex0250` | full Alaska NLCD | **not in the repo** — unlike `ex0251`, this one still points into `experimental/personal/` |
| `ex0263_bhutan_big` | `src/bhutan.geojson` | **not in the repo** — 161k population points |

## Two things worth knowing before reading the renders

**Net choice.** Most regional renders select a fixed `n_oct` net layout
(`ex0251`, `ex0252`, `ex0300`, `ex0302`, …), fitting the region into a
preselected net. `ex0260_composition` does the more general thing — it builds a
**dynamic local net** fitted to the region, which is what lets it carry a
non-convex four-octant band around the pole without tearing. The choice governs
whether a bounded region spanning a seam renders cleanly.

**Orientation.** `plot_hex` takes `rotate_north=True`, which turns the whole
scene by a multiple of 90° so north lands within ±45° of up. The quarter-turn
snap keeps it exact: coordinates become swaps and sign flips, and a raster
backdrop is reindexed by `np.rot90` with no resampling. The north arrow then
shows the residual tilt, and the plotted aspect swaps for odd turns.

Enabled in `ex0260_composition`, `ex0300_backdrop`, `ex0301_heatmap`. Not
needed by `ex0251`/`ex0252`, whose windows are already near north-up. It has no
effect on polar frames such as `ex0260`'s C-band case, where the pole sits
inside the picture and "north" has no single direction — that case passes
`rotate_north=False` deliberately.

Still unrotated: `ex0302_gpkg` renders on its side, and `ex0261_polygrid`
defines its own local `plot_hex` rather than using the library renderer, so it
gets neither rotation nor an arrow. A scale bar is drawn only by
`ex0263_89_seam_zoom`, which does it in about four lines against tangent-plane
metres — that generalises if it is wanted elsewhere.
