# Part of the Hex9 (H9) Project
# Copyright ©2026, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Imagery sources for hex fills and pixel backdrops.

Every factory here returns a **sampler**: a callable taking
:class:`~hhg9.Points` in ``b_oct`` and returning an ``(N, C)`` float32 array.
That is exactly the contract of :attr:`~hhg9.rendering.composition.LayerSpec.source`
and of the ``sampler`` argument to
:func:`~hhg9.rendering.composition.make_backdrop`, so one source can drive a
per-hex fill and a per-pixel backdrop interchangeably:

    from hhg9.rendering.imagery import make_wmts_source
    from hhg9.rendering.composition import make_backdrop, LayerSpec

    esri = make_wmts_source(CAPS_URL, 'World_Imagery', reg, b_oct,
                            attribution='Imagery © Esri')
    backdrop = make_backdrop(reg, b_oct, n_oct, bbox_n, px_w, px_h, esri)
    spec = LayerSpec(level=12, kind='fill', source=esri)

Samplers carry their imagery credit on the callable — ``.attribution`` for a
single line, and ``.credits`` for an ordered stack built by
:func:`make_composite_source`. Use :func:`credits_of` to collect them and pass
the result to :func:`~hhg9.rendering.render.plot_hex`, so that published renders
carry the attribution their sources require.

Optional dependencies are imported lazily: ``owslib`` for WMTS, GDAL for
raster sources. Importing this module needs neither.
"""

import math
import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import KDTree

from hhg9 import Points

TILE_PX = 256
MERC_R = 6378137.0
MERC_MAX = math.pi * MERC_R


def credits_of(*sources) -> list:
    """Ordered, de-duplicated imagery credits for one or more samplers.

    Reads ``.credits`` (a composite's ordered stack) in preference to
    ``.attribution`` (a single source's line), preserving first-seen order so
    the topmost imagery is credited first.
    """
    out = []
    for src in sources:
        if src is None:
            continue
        got = list(getattr(src, 'credits', None) or ())
        if not got:
            one = getattr(src, 'attribution', '')
            got = [one] if one else []
        for c in got:
            if c and c not in out:
                out.append(c)
    return out


def make_pc_source(img_file, reg, p_pix, b_oct, *, attribution=''):
    """Build a PlateCarrée KDTree background source.

    Returns a callable (b_pts: Points) -> (N, 4) float32 RGBA [0, 1].
    `attribution` is stored on the callable for on-frame imagery credit.
    """
    from matplotlib import image
    img = image.imread(img_file, 'png')
    pc_px = p_pix.adopt(img)
    pc_el = reg.project(pc_px, [p_pix, 'g_gcd'])
    tree = KDTree(pc_el.coords)

    def source(b_pts):
        ref = reg.project(b_pts, [b_oct, 'g_gcd'])
        _, idx = tree.query(ref.coords, workers=-1)
        s = pc_px.samples[idx].astype(np.float32)
        if s.shape[1] < 4:
            s = np.hstack((s, np.ones((s.shape[0], 1), dtype=s.dtype)))
        return s

    source.attribution = attribution
    return source


def _make_tiled_source(reg, b_oct, *, max_zoom, is_geographic, grid_of,
                       fetch_tile, gain=1.0, gamma=1.0, attribution='',
                       with_coverage=False):
    """Shared tile-stitch/sample core for slippy-tile backgrounds.

    Zoom selection, tile-grid stitching, native-CRS extent, and bilinear
    sampling are independent of *how* tiles are addressed.  Callers supply:

        grid_of(z)       -> (n_cols, n_rows) tile-grid dimensions at zoom z
        fetch_tile(tx,ty,z) -> (TILE_PX, TILE_PX, 3|4) float32 [0,1], or None
                              for a missing tile.  A 4th channel is treated as
                              per-pixel coverage (alpha); 3-channel tiles are
                              fully opaque.

    `is_geographic` selects WGS84 (lon/lat) vs Web-Mercator axis math.
    Returns a callable (b_pts: Points) -> (N, 3) float32 [0, 1], or (N, 4) with
    a coverage-alpha channel when `with_coverage` (0 = no imagery here, e.g.
    outside a partial-coverage mosaic — composite over another source).
    """
    c_oct = reg.domain('c_oct')
    c_ell = reg.domain('c_ell')
    g_gcd = reg.domain('g_gcd')

    def source(b_pts: Points) -> np.ndarray:
        ll = reg.project(b_pts, [b_oct, c_oct, c_ell, g_gcd])
        lats = ll.coords[:, 0]
        lons = ll.coords[:, 1]
        lat_min = float(np.clip(lats.min(), -85.0, 85.0))
        lat_max = float(np.clip(lats.max(), -85.0, 85.0))
        lon_min, lon_max = float(lons.min()), float(lons.max())

        # Zoom: target ~2× per-axis point density
        span_deg = max(lon_max - lon_min, lat_max - lat_min, 1e-6)
        z = max(0, min(max_zoom, round(
            math.log2(360.0 * int(math.sqrt(len(b_pts.coords))) * 2.0 / (TILE_PX * span_deg))
        )))

        # Tile grid dimensions at this zoom (Mercator 2^z × 2^z or
        # WGS84 2^(z+1) × 2^z, per the supplied grid_of)
        n_cols, n_rows = grid_of(z)

        def _ll_to_tile(lon, lat):
            tx = min(int((lon + 180.0) / 360.0 * n_cols), n_cols - 1)
            if is_geographic:
                ty = min(int((90.0 - lat) / 180.0 * n_rows), n_rows - 1)
            else:
                lat_r = math.radians(max(-85.0511, min(85.0511, lat)))
                ty = min(int((1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n_rows),
                         n_rows - 1)
            return tx, ty

        tx_min, ty_min = _ll_to_tile(lon_min, lat_max)
        tx_max, ty_max = _ll_to_tile(lon_max, lat_min)

        # Stitch tiles (rows top-to-bottom).  Canvas carries RGB + coverage alpha:
        # a missing tile (None) stays alpha 0; a 3-channel tile is fully opaque; a
        # 4-channel tile brings its own per-pixel alpha (mosaic footprint edge).
        n_tx = tx_max - tx_min + 1
        n_ty = ty_max - ty_min + 1
        canvas = np.zeros((n_ty * TILE_PX, n_tx * TILE_PX, 4), dtype=np.float32)
        for ty in range(ty_min, ty_max + 1):
            for tx in range(tx_min, tx_max + 1):
                arr = fetch_tile(tx, ty, z)
                if arr is None:
                    continue
                iy = (ty - ty_min) * TILE_PX
                ix = (tx - tx_min) * TILE_PX
                if arr.shape[2] == 4:
                    canvas[iy:iy + TILE_PX, ix:ix + TILE_PX] = arr
                else:
                    canvas[iy:iy + TILE_PX, ix:ix + TILE_PX, :3] = arr
                    canvas[iy:iy + TILE_PX, ix:ix + TILE_PX, 3] = 1.0

        # Extent of stitched canvas in native CRS coordinates
        if is_geographic:
            x_left = -180.0 + (tx_min / n_cols) * 360.0
            x_right = -180.0 + ((tx_max + 1) / n_cols) * 360.0
            y_top = 90.0 - (ty_min / n_rows) * 180.0
            y_bot = 90.0 - ((ty_max + 1) / n_rows) * 180.0
            # Query points are already in lon/lat — use directly
            pts_q = np.stack([lats, lons], axis=1)  # (lat=y, lon=x)
        else:
            x_left = (tx_min / n_cols) * 2 * MERC_MAX - MERC_MAX
            x_right = ((tx_max + 1) / n_cols) * 2 * MERC_MAX - MERC_MAX
            y_top = MERC_MAX - (ty_min / n_rows) * 2 * MERC_MAX
            y_bot = MERC_MAX - ((ty_max + 1) / n_rows) * 2 * MERC_MAX

            # Convert lat/lon to Mercator for query
            def _ll_to_merc(lon_deg, lat_deg):
                x = math.radians(lon_deg) * MERC_R
                lat_r = math.radians(max(-85.0511, min(85.0511, lat_deg)))
                y = math.log(math.tan(math.pi / 4 + lat_r / 2)) * MERC_R
                return x, y

            merc_xy = np.array([_ll_to_merc(lo, la) for lo, la in zip(lons.tolist(), lats.tolist())])
            pts_q = merc_xy[:, ::-1]  # (y, x) order

        # RegularGridInterpolator needs monotone-increasing axes.
        # Canvas rows are top→bottom (y_top > y_bot), so flip.
        h, w = canvas.shape[:2]
        xs = np.linspace(x_left, x_right, w)
        ys = np.linspace(y_bot, y_top, h)  # increasing after flip
        canvas = canvas[::-1]

        n_ch = 4 if with_coverage else 3  # sample the coverage channel only if asked
        channels = [
            RegularGridInterpolator(
                (ys, xs), canvas[:, :, c],
                method='linear', bounds_error=False, fill_value=0.0,
            )(pts_q)
            for c in range(n_ch)
        ]
        out = np.stack(channels, axis=1).astype(np.float32)
        if gain != 1.0 or gamma != 1.0:
            out[:, :3] = np.clip(out[:, :3] * gain, 0.0, 1.0) ** (1.0 / gamma)
        return out

    source.attribution = attribution
    return source


def make_xyz_source(url_template, reg, b_oct, *,
                    max_zoom=21, is_geographic=False, with_coverage=False,
                    cache_dir='xyz_cache', gain=1.0, gamma=1.0, attribution=''):
    """Build a background source from a slippy XYZ tile template.

    `url_template` is a `{z}/{x}/{y}` template (Web-Mercator, Google/OSM row
    convention), e.g. an OpenAerialMap mosaic URL.  Missing tiles (404) are
    reported as no-coverage, so a partial mosaic can composite over another
    source (see make_composite_source) instead of showing black.

    Args:
        url_template:  URL with `{z}` `{x}` `{y}` placeholders.
        max_zoom:      Deepest zoom the service provides.
        is_geographic: True for a WGS84 (EPSG:4326) tile scheme; default False
                       (Web Mercator).
        with_coverage: return (N, 4) with a coverage-alpha channel (for
                       compositing) instead of (N, 3).

    Returns:
        Callable  (b_pts: Points) -> (N, 3) float32 [0, 1], or (N, 4) if
        `with_coverage`.
    """
    import hashlib
    import io
    import urllib.request
    from PIL import Image as PILImage

    os.makedirs(cache_dir, exist_ok=True)
    key0 = hashlib.md5(url_template.encode()).hexdigest()[:8]

    def fetch_tile(tx, ty, z):
        path = os.path.join(cache_dir, f'{key0}_{z}_{tx}_{ty}.png')
        if os.path.exists(path):
            img = PILImage.open(path)
        else:
            url = url_template.format(z=z, x=tx, y=ty)
            try:
                with urllib.request.urlopen(url) as resp:
                    img = PILImage.open(io.BytesIO(resp.read()))
            except Exception:
                return None  # missing tile → no coverage
            img.save(path)
        has_alpha = img.mode in ('RGBA', 'LA', 'PA')
        arr = np.asarray(img.convert('RGBA'), dtype=np.float32) / 255.0
        if not has_alpha:
            # RGB tile (or legacy RGB cache): derive coverage from non-black, so
            # a mosaic's footprint edge (transparent → saved black) drops out.
            arr[:, :, 3] = (arr[:, :, :3].max(axis=2) > 0.02).astype(np.float32)
        return arr

    def grid_of(z):
        n = 1 << z
        return n, n

    return _make_tiled_source(
        reg, b_oct, max_zoom=max_zoom, is_geographic=is_geographic,
        with_coverage=with_coverage,
        grid_of=grid_of, fetch_tile=fetch_tile, gain=gain, gamma=gamma,
        attribution=attribution,
    )


def make_composite_source(over, under, *, attribution=None):
    """Alpha-composite `over` onto `under`, per query point.

    `over` must return (N, 4) with a coverage-alpha channel (e.g. a
    make_xyz_source(..., with_coverage=True) mosaic).  Where `over` has no
    coverage — outside a partial mosaic's footprint, or a missing tile — the
    `under` source shows through instead of black.  The soft coverage edge
    (bilinear) blends the two cleanly at the footprint boundary.

    `under` may return (N, 3) or (N, 4); its alpha, if any, is ignored (it is the
    always-present base layer).  Returns a callable -> (N, 4) opaque RGBA.
    """
    def source(b_pts):
        o = over(b_pts)                       # (N, 4): RGB + coverage
        u = under(b_pts)                      # (N, 3|4): base layer
        a = np.clip(o[:, 3:4], 0.0, 1.0)
        rgb = o[:, :3] * a + u[:, :3] * (1.0 - a)
        opaque = np.ones((rgb.shape[0], 1), dtype=np.float32)
        return np.hstack((rgb.astype(np.float32), opaque))

    over_attr = getattr(over, 'attribution', '')
    under_attr = getattr(under, 'attribution', '')
    # Ordered credit stack, top-of-stack (deepest subject) first → base last.  A
    # nested composite already has .credits; a raw source contributes its single
    # attribution.  The joined .attribution is kept for the legacy single-line path.
    over_credits = list(getattr(over, 'credits', None) or ([over_attr] if over_attr else []))
    under_credits = list(getattr(under, 'credits', None) or ([under_attr] if under_attr else []))
    source.credits = over_credits + under_credits
    if attribution is None:
        attribution = ' · over '.join(source.credits)
    source.attribution = attribution
    return source


def make_wmts_source(capabilities_url, layer_name, reg, b_oct, *,
                     tile_matrix_set=None,
                     cache_dir='wmts_cache',
                     gain=1.1, gamma=1.1, attribution=''):
    """Build a WMTS background source via owslib tile-level fetching.

    Fetches individual tiles for the bounding box of each query batch, stitches
    them, and samples with bilinear interpolation.  Works with both Web Mercator
    (GoogleMapsCompatible / EPSG:3857) and geographic (WGS84 / EPSG:4326) tile
    matrix sets — the CRS is auto-detected from the capabilities.

    Tiles are cached individually on disk; animation frames that share tiles at
    the same zoom level skip the network entirely on re-runs.

    Zoom level is auto-selected to match ~2× per-axis point density.
    `tile_matrix_set` can be passed explicitly if auto-detection picks the wrong
    one (the available names are printed on first run).

    Args:
        capabilities_url:  WMTS GetCapabilities URL.
        layer_name:        WMTS layer identifier (e.g. 's2cloudless-2020').
        reg:               Shared Registrar.
        b_oct:             b_oct domain.
        tile_matrix_set:   TileMatrixSet name, or None to auto-detect.
        cache_dir:         Directory for per-tile PNG cache files.

    Returns:
        Callable  (b_pts: Points) -> (N, C) float32 [0, 1].
    """
    import hashlib
    import io
    import urllib.parse
    import urllib.request
    from owslib.wmts import WebMapTileService
    from PIL import Image as PILImage

    os.makedirs(cache_dir, exist_ok=True)
    wmts = WebMapTileService(capabilities_url)
    layer = wmts.contents[layer_name]

    available = list(layer.tilematrixsetlinks.keys())
    if tile_matrix_set and tile_matrix_set in available:
        tms = tile_matrix_set
    else:
        # Prefer Mercator; fall back to any available TMS (e.g. WGS84-only services)
        merc_names = ['GoogleMapsCompatible', 'g', 'EPSG:3857', 'WebMercatorQuad',
                      'googlemapscompatible', 'EPSG_3857']
        tms = next((t for t in merc_names if t in available), available[0])
    print(f'  WMTS: layer={layer_name!r}  tms={tms!r}  available={available}')

    tile_fmt = layer.formats[0] if layer.formats else 'image/jpeg'
    tms_obj = wmts.tilematrixsets[tms]

    # Detect CRS: geographic (WGS84 / EPSG:4326) or projected (Mercator)
    crs_str = getattr(tms_obj, 'crs', '') or ''
    is_geographic = '4326' in crs_str or 'CRS84' in crs_str or 'WGS84' in crs_str.upper()

    zoom_ids = sorted(tms_obj.tilematrix, key=lambda x: int(x.split(':')[-1]))
    max_zoom = int(zoom_ids[-1].split(':')[-1])
    zoom_to_id = {int(tid.split(':')[-1]): tid for tid in zoom_ids}

    # Build a KVP GetTile URL directly — owslib's gettile() crashes on some
    # services due to a bug in its restonly property (iterates operations as a
    # dict rather than a list of named objects).
    kvp_base = capabilities_url.split('?')[0].replace('WMTSCapabilities.xml', '')

    def _tile_url(tilematrix_id, row, col):
        return kvp_base + '?' + urllib.parse.urlencode({
            'SERVICE': 'WMTS', 'REQUEST': 'GetTile', 'VERSION': '1.0.0',
            'LAYER': layer_name, 'STYLE': 'default', 'FORMAT': tile_fmt,
            'TILEMATRIXSET': tms, 'TILEMATRIX': tilematrix_id,
            'TILEROW': row, 'TILECOL': col,
        })

    def _fetch_tile(tx, ty, z):
        key = f"{layer_name}:{tms}:{z}:{tx}:{ty}"
        path = os.path.join(cache_dir, hashlib.md5(key.encode()).hexdigest()[:16] + '.png')
        if os.path.exists(path):
            return np.asarray(PILImage.open(path).convert('RGB'), dtype=np.float32) / 255.0
        with urllib.request.urlopen(_tile_url(zoom_to_id[z], ty, tx)) as resp:
            img = PILImage.open(io.BytesIO(resp.read())).convert('RGB')
        img.save(path)
        return np.asarray(img, dtype=np.float32) / 255.0

    def grid_of(z):
        # Read from capabilities — works for both Mercator 2^z × 2^z and
        # WGS84 2^(z+1) × 2^z layouts.
        tm = tms_obj.tilematrix[zoom_to_id[z]]
        return tm.matrixwidth, tm.matrixheight

    return _make_tiled_source(
        reg, b_oct, max_zoom=max_zoom, is_geographic=is_geographic,
        grid_of=grid_of, fetch_tile=_fetch_tile, gain=gain, gamma=gamma,
        attribution=attribution,
    )


def make_bm_source(vrt_path, reg, b_oct, gamma=0.9, *,
                   name='bm_wkt', with_coverage=False, attribution=''):
    """Build a GDAL raster background source (Blue Marble, a georeferenced
    photo, or any GeoTIFF/VRT).

    Opens `vrt_path`, registers its CRS via Wkt + Wkt_4978, and returns a
    callable (b_pts: Points) -> (N, 3) float32 [0, 1], or (N, 4) with a
    coverage-alpha channel when `with_coverage` (for make_composite_source).

    Coverage comes from the raster's alpha band (band 4, e.g. written by
    geotag_image.py) if present, else from the in-footprint / non-black region;
    out-of-raster samples are 0.  So a partial ortho composites over whatever is
    beneath instead of painting its bounding rectangle black.

    gamma < 1 lifts midtones (Blue Marble tends to be low-contrast).
    `name` is the registered CRS-domain key — pass a UNIQUE name per file when
    building more than one raster source, or they collide on the registrar.
    `attribution` is stored on the callable for on-frame imagery credit.

    Note: make_geotiff_sampler reads full bands into memory each call.
    Use the 10800×10800 tile set, not 21600×21600.
    """
    from hhg9.geo.gdal import wkt_geotiff_meta, make_geotiff_sampler
    c_oct = reg.domain('c_oct')
    c_ell = reg.domain('c_ell')
    ds, _, bm_wkt = wkt_geotiff_meta(vrt_path, reg=reg, name=name)
    chain = [b_oct, c_oct, c_ell, bm_wkt]
    sampler = make_geotiff_sampler(ds, reg, chain)

    def source(b_pts):
        vals = sampler(b_pts).astype(np.float32)
        rgb = vals[:, :3]
        if gamma != 1.0:
            rgb = np.clip(rgb, 0.0, 1.0) ** gamma
        if not with_coverage:
            return rgb
        if vals.shape[1] >= 4:
            cov = np.clip(vals[:, 3:4], 0.0, 1.0)  # alpha band = footprint
        else:
            cov = (rgb.max(axis=1, keepdims=True) > 0.0).astype(np.float32)
        return np.hstack((rgb, cov))

    source.attribution = attribution
    return source
