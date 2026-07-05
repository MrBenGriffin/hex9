# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Zoom sequence centred on a lat/lon point of interest.

Frame L shows:
  - Background pixel map zoomed to a 1/3^L sub-region of the full net.
  - Hex overlays:
      L=0 — layers 1, 2 (net shape is the implicit L0 cue, as per ex0097)
      L>0 — layers L, L+1, L+2
  - Middle-layer hex (L+1, for L>0) highlighted in gold.
  - Dashed yellow rectangle indicating the next frame's viewport (debug aid).

Background sources are swappable per-frame via bg_sources.  Two built-in
source builders are provided:
  make_pc_source  — PlateCarrée KDTree (the original approach, good for L=0)
  make_bm_source  — GDAL VRT sampler (Blue Marble or similar, for L>=1)

Note: make_geotiff_sampler reads full raster bands into memory.  Use the
10800×10800 Blue Marble variant (not 21600×21600) to keep RAM below ~400 MB.
Build a VRT first:
  gdalbuildvrt src/blue_marble.vrt src/world.200408.3x10800x10800.A*.tif

Point of Interest: Hall of Maps, Palazzo Vecchio, Florence, Italy.

Last Tested
16 Jun 2026 0.1.3a0 (passed) 895.7s
28 Mar 2026 (new)
"""
import os

import numpy as np
from matplotlib import image, pyplot as plt
from matplotlib.collections import PolyCollection
from osgeo import gdal
from scipy.ndimage import distance_transform_edt
from scipy.spatial import KDTree

from hhg9 import Registrar, Points
from hhg9.algorithms.geometry import inside_convex_polygon_cw
from hhg9.h9 import H9O, H9P
from hhg9.h9.addressing import hex_str_encode
from hhg9.h9.binning import hex_reduce, hex_parents
from hhg9.h9.grid import qa_grid, hex_verts_in_noct, poly_net_field
from hhg9.h9.tail import tail_unpack_reversible


# POI_LAT = 43.7694687631026    # Hall of Maps, Palazzo Vecchio, Florence
# POI_LON = 11.25624466034606

# POI_LAT, POI_LON = 48.858735900477676, 2.294496130260531  # Eiffel's Tower
# POI_LAT, POI_LON = 34.632049981869564, 77.62554188766468  # Samsten Ling
# POI_LAT, POI_LON =24.136360966675497, 23.245733016051407  # Al Kufrah.
# POI_LAT, POI_LON = 51.481588012311560, 0.00223723016213   # London
POI_LON, POI_LAT = 1.640250018, 49.098794825    # box on lawn.
# ── helpers ──────────────────────────────────────────────────────────────────

def _fit_viewport(pts, img_w, img_h, vp):
    """Map n_oct coords to integer pixel indices using a fixed viewport.

    Returns px (M,), py (M,), in_vp (N,) bool mask over pts.
    """
    x, y = pts.coords[:, 0], pts.coords[:, 1]
    sx = (img_w - 1) / (vp[1] - vp[0])
    sy = (img_h - 1) / (vp[3] - vp[2])
    pvx = np.rint((x - vp[0]) * sx).astype(int)
    pvy = (img_h - 1) - np.rint((y - vp[2]) * sy).astype(int)
    in_vp = (pvx >= 0) & (pvx < img_w) & (pvy >= 0) & (pvy < img_h)
    return pvx[in_vp], pvy[in_vp], in_vp


def _hex_at_layer(poi_b, layer, b_oct, n_oct, reg):
    """Return the centroid (n_oct) and polygon (H, 6, 2) for the hex at
    `layer` containing poi_b."""
    h_num, h_v, _, _ = hex_reduce(poi_b, layer)
    h_par, h_oid, h_scale = hex_parents(b_oct, h_v, h_num)
    xc2, _, xpm = tail_unpack_reversible(h_v[:, -1])   # (c2, r_mo, p_mo)
    verts_n = hex_verts_in_noct(h_par, h_oid, xpm, xc2, h_scale, n_oct)
    ctr_b = Points(h_par, b_oct, oid=h_oid)
    ctr_n = reg.project(ctr_b, [b_oct, n_oct])
    return ctr_n.coords[0], verts_n.coords.reshape(-1, 6, 2)


def _viewport(L, centre_n, net_xmin, net_xmax, net_ymin, net_ymax):
    """Viewport [xmin, xmax, ymin, ymax] at zoom level L.

    L=0 returns the full net extent.  L>0 shrinks by 3^L per axis,
    centred on centre_n, clamped to the net bounds.
    """
    if L == 0:
        return [net_xmin, net_xmax, net_ymin, net_ymax]
    span_x = (net_xmax - net_xmin) / (3.0 ** L)
    span_y = (net_ymax - net_ymin) / (3.0 ** L)
    cx, cy = centre_n
    return [
        max(net_xmin, cx - span_x / 2),
        min(net_xmax, cx + span_x / 2),
        max(net_ymin, cy - span_y / 2),
        min(net_ymax, cy + span_y / 2),
    ]


def _project_noct_pts(coords_n, n_oct, b_oct, reg, tol=0.0):
    """Assign face oids to arbitrary n_oct coordinates and project to b_oct.

    Iterates over n_oct.face_polys to determine which face each point belongs
    to (union of sub-polygons for c2-split faces), then projects face-by-face.

    Returns:
        b_pts  — Points in b_oct (valid projections only).
        idx    — int array mapping each output point back to its row in coords_n.
    """
    n = len(coords_n)
    b_pt_list, idx_list = [], []
    for sign, polys in n_oct.face_polys.items():
        oid = H9O.cmp_oid[sign]
        in_face = np.zeros(n, dtype=bool)
        for poly in polys:
            in_face |= inside_convex_polygon_cw(coords_n, poly, tol=tol)
        if not np.any(in_face):
            continue
        b_face = reg.project(Points(coords_n[in_face], n_oct, oid), [n_oct, b_oct])
        valid = b_oct.valid(b_face)
        if np.any(valid):
            b_pt_list.append(b_face.select(valid))
            idx_list.append(np.where(in_face)[0][valid])
    if not b_pt_list:
        return None, np.array([], dtype=int)
    return Points.concat(b_pt_list), np.concatenate(idx_list)


def _viewport_grid(vp, img_w, img_h, n_oct, b_oct, reg):
    """Generate a full img_w × img_h sample of b_oct points for the viewport.

    Returns px (M,), py (M,), b_pts (M Points in b_oct).
    M ≈ img_w * img_h minus any seam gaps.
    """
    xl = np.linspace(vp[0], vp[1], img_w)
    yl = np.linspace(vp[3], vp[2], img_h)   # row 0 = top = high y
    xx, yy = np.meshgrid(xl, yl)
    coords = np.stack([xx.ravel(), yy.ravel()], axis=1)
    px_all = np.tile(np.arange(img_w), img_h)
    py_all = np.repeat(np.arange(img_h), img_w)
    tol = 0.5 * (vp[1] - vp[0]) / img_w   # half-pixel seam tolerance
    b_pts, idx = _project_noct_pts(coords, n_oct, b_oct, reg, tol=tol)
    return px_all[idx], py_all[idx], b_pts


def _pick_source(bg_sources, L):
    """Select the source callable for frame L.

    bg_sources may be:
      - a single callable  → used for all frames
      - a dict {int: callable} → max key ≤ L is used; falls back to min key
    """
    if callable(bg_sources):
        return bg_sources
    candidates = [k for k in bg_sources if k <= L]
    key = max(candidates) if candidates else min(bg_sources)
    return bg_sources[key]


def _common_prefix(strs):
    """Longest common leading substring across strs (character-wise).

    Equal to os.path.commonprefix; inlined to avoid the import.  Returns '' for
    an empty list and the whole string for a single element.
    """
    if not strs:
        return ''
    lo, hi = min(strs), max(strs)
    for i, c in enumerate(lo):
        if c != hi[i]:
            return lo[:i]
    return lo


# ── source builders ───────────────────────────────────────────────────────────

def make_pc_source(img_file, reg, p_pix, b_oct, *, attribution=''):
    """Build a PlateCarrée KDTree background source.

    Returns a callable (b_pts: Points) -> (N, 4) float32 RGBA [0, 1].
    `attribution` is stored on the callable for on-frame imagery credit.
    """
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
                       fetch_tile, gain=1.0, gamma=1.0, attribution=''):
    """Shared tile-stitch/sample core for slippy-tile backgrounds.

    Zoom selection, tile-grid stitching, native-CRS extent, and bilinear
    sampling are independent of *how* tiles are addressed.  Callers supply:

        grid_of(z)       -> (n_cols, n_rows) tile-grid dimensions at zoom z
        fetch_tile(tx,ty,z) -> (TILE_PX, TILE_PX, 3) float32 [0,1] RGB

    `is_geographic` selects WGS84 (lon/lat) vs Web-Mercator axis math.
    Returns a callable (b_pts: Points) -> (N, 3) float32 [0, 1].
    """
    import math
    from scipy.interpolate import RegularGridInterpolator

    TILE_PX = 256
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
                ty = min(int((1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n_rows), n_rows - 1)
            return tx, ty

        tx_min, ty_min = _ll_to_tile(lon_min, lat_max)
        tx_max, ty_max = _ll_to_tile(lon_max, lat_min)

        # Stitch tiles (rows top-to-bottom)
        n_tx = tx_max - tx_min + 1
        n_ty = ty_max - ty_min + 1
        canvas = np.zeros((n_ty * TILE_PX, n_tx * TILE_PX, 3), dtype=np.float32)
        for ty in range(ty_min, ty_max + 1):
            for tx in range(tx_min, tx_max + 1):
                arr = fetch_tile(tx, ty, z)
                iy = (ty - ty_min) * TILE_PX
                ix = (tx - tx_min) * TILE_PX
                canvas[iy:iy + TILE_PX, ix:ix + TILE_PX] = arr

        # Extent of stitched canvas in native CRS coordinates
        if is_geographic:
            x_left  = -180.0 + (tx_min / n_cols) * 360.0
            x_right = -180.0 + ((tx_max + 1) / n_cols) * 360.0
            y_top   =  90.0 - (ty_min / n_rows) * 180.0
            y_bot   =  90.0 - ((ty_max + 1) / n_rows) * 180.0
            # Query points are already in lon/lat — use directly
            pts_q = np.stack([lats, lons], axis=1)   # (lat=y, lon=x)
        else:
            MERC_MAX = math.pi * 6378137.0
            x_left  = (tx_min / n_cols) * 2 * MERC_MAX - MERC_MAX
            x_right = ((tx_max + 1) / n_cols) * 2 * MERC_MAX - MERC_MAX
            y_top   = MERC_MAX - (ty_min / n_rows) * 2 * MERC_MAX
            y_bot   = MERC_MAX - ((ty_max + 1) / n_rows) * 2 * MERC_MAX
            # Convert lat/lon to Mercator for query
            def _ll_to_merc(lon_deg, lat_deg):
                x = math.radians(lon_deg) * 6378137.0
                lat_r = math.radians(max(-85.0511, min(85.0511, lat_deg)))
                y = math.log(math.tan(math.pi / 4 + lat_r / 2)) * 6378137.0
                return x, y
            merc_xy = np.array([_ll_to_merc(lo, la) for lo, la in zip(lons.tolist(), lats.tolist())])
            pts_q = merc_xy[:, ::-1]   # (y, x) order

        # RegularGridInterpolator needs monotone-increasing axes.
        # Canvas rows are top→bottom (y_top > y_bot), so flip.
        h, w = canvas.shape[:2]
        xs = np.linspace(x_left, x_right, w)
        ys = np.linspace(y_bot, y_top, h)      # increasing after flip
        canvas = canvas[::-1]

        channels = [
            RegularGridInterpolator(
                (ys, xs), canvas[:, :, c],
                method='linear', bounds_error=False, fill_value=0.0,
            )(pts_q)
            for c in range(3)
        ]
        out = np.stack(channels, axis=1).astype(np.float32)
        if gain != 1.0 or gamma != 1.0:
            out = np.clip(out * gain, 0.0, 1.0) ** (1.0 / gamma)
        return out

    source.attribution = attribution
    return source


def make_xyz_source(url_template, reg, b_oct, *,
                    max_zoom=21, is_geographic=False,
                    cache_dir='xyz_cache', gain=1.0, gamma=1.0, attribution=''):
    """Build a background source from a slippy XYZ tile template.

    `url_template` is a `{z}/{x}/{y}` template (Web-Mercator, Google/OSM row
    convention), e.g. an OpenAerialMap mosaic URL.  Missing tiles (404) read as
    black, so partial-coverage mosaics composite over whatever is underneath.

    Args:
        url_template:  URL with `{z}` `{x}` `{y}` placeholders.
        max_zoom:      Deepest zoom the service provides.
        is_geographic: True for a WGS84 (EPSG:4326) tile scheme; default False
                       (Web Mercator).

    Returns:
        Callable  (b_pts: Points) -> (N, 3) float32 [0, 1].
    """
    import hashlib, io, os, urllib.request
    from PIL import Image as PILImage

    TILE_PX = 256
    os.makedirs(cache_dir, exist_ok=True)
    key0 = hashlib.md5(url_template.encode()).hexdigest()[:8]

    def fetch_tile(tx, ty, z):
        path = os.path.join(cache_dir, f'{key0}_{z}_{tx}_{ty}.png')
        if os.path.exists(path):
            return np.asarray(PILImage.open(path).convert('RGB'), dtype=np.float32) / 255.0
        url = url_template.format(z=z, x=tx, y=ty)
        try:
            with urllib.request.urlopen(url) as resp:
                img = PILImage.open(io.BytesIO(resp.read())).convert('RGB')
        except Exception:
            return np.zeros((TILE_PX, TILE_PX, 3), dtype=np.float32)
        img.save(path)
        return np.asarray(img, dtype=np.float32) / 255.0

    def grid_of(z):
        n = 1 << z
        return n, n

    return _make_tiled_source(
        reg, b_oct, max_zoom=max_zoom, is_geographic=is_geographic,
        grid_of=grid_of, fetch_tile=fetch_tile, gain=gain, gamma=gamma,
        attribution=attribution,
    )


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
    import hashlib, io, os, urllib.parse, urllib.request
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
                   name='bm_wkt', attribution=''):
    """Build a GDAL raster background source (Blue Marble, a georeferenced
    photo, or any GeoTIFF/VRT).

    Opens `vrt_path`, registers its CRS via Wkt + Wkt_4978, and returns a
    callable (b_pts: Points) -> (N, C) float32 [0, 1].

    gamma < 1 lifts midtones (Blue Marble tends to be low-contrast).
    `name` is the registered CRS-domain key — pass a UNIQUE name per file when
    building more than one raster source, or they collide on the registrar.
    `attribution` is stored on the callable for on-frame imagery credit.

    Note: make_geotiff_sampler reads full bands into memory each call.
    Use the 10800×10800 tile set, not 21600×21600.
    """
    from osgeo import gdal
    from hhg9.geo.gdal import wkt_geotiff_meta, make_geotiff_sampler
    c_oct = reg.domain('c_oct')
    c_ell = reg.domain('c_ell')
    ds, _, bm_wkt = wkt_geotiff_meta(vrt_path, reg=reg, name=name)
    chain = [b_oct, c_oct, c_ell, bm_wkt]
    sampler = make_geotiff_sampler(ds, reg, chain)

    def source(b_pts):
        vals = sampler(b_pts).astype(np.float32)
        if gamma != 1.0:
            vals = np.clip(vals, 0.0, 1.0) ** gamma
        return vals

    source.attribution = attribution
    return source


# ── main ─────────────────────────────────────────────────────────────────────

def run(*, reg=None, flavour='rhombus', scale=1201, max_layer=4, bg_sources,
        frames_per_level: int = 0, centre: str = 'poi',
        start_frame: int = 0, end_frame=None, skip_existing: bool = False):
    """Generate zoom-sequence PNGs.

    Args:
        reg:              Registrar to use.  Pass the same instance used to build
                          bg_sources so domain/projection state is shared.  A new
                          Registrar is created if None.
        flavour:          Net layout name.
        scale:            Pixel density (triangle side in pixels) for the full grid.
        max_layer:        Highest zoom level to render (inclusive).
        bg_sources:       Background colour source(s).  A single callable or a
                          dict {int: callable} mapping minimum-L to source.
                          Callable signature: (b_pts: Points) -> (N, 3|4) float [0,1].
        frames_per_level: 0 → render one PNG per integer level (original behaviour).
                          N → render a smooth animation: N frames per level transition
                          up to max_layer.  Output files are named frame_NNNN.png.
        centre:           'hex' (default) — interpolate between hex centroids (Ken Burns
                          drift toward POI across each level transition).
                          'poi' — lock viewport centre to the POI's net coordinates from
                          L=1 onwards; L=0→1 still drifts in from the global centre.
        start_frame:      Resume: skip every frame with index < start_frame.  In animate
                          mode the index is the frame number (frame_NNNN); otherwise it is
                          the layer L.  Frame index for the start of level L is 1 + L*fpl.
        end_frame:        Stop after this frame index (inclusive); None renders to the end.
        skip_existing:    Skip any frame whose output PNG already exists on disk — the
                          simplest resume after an interrupted run.
    """
    if reg is None:
        reg = Registrar()
    n_oct = reg.domain(f'n_oct:{flavour}')
    b_oct = reg.domain('b_oct')
    g_gcd = reg.domain('g_gcd')

    img_w, img_h = n_oct.image_dims(pixels=scale)

    # POI in b_oct
    poi_g = Points(np.array([[POI_LAT, POI_LON]]), g_gcd)
    poi_b = reg.project(poi_g, [g_gcd, b_oct])

    # Full grid — generated once, shared across all frames
    pt_list = []
    for name, sdom in n_oct.sides.items():
        oid = sdom.oid
        mo = H9O.oid_mo[oid]
        prj = n_oct.projs[name]
        for c2 in [0, 1, 2]:
            hhp = H9P.hh[mo, c2]
            placed_mode = prj.c2trans[c2][0] if prj.c2trans is not None else -1
            grid = qa_grid(hhp, scale, affine=prj.c2_affine(c2), net_mode=placed_mode)
            pix, msk = grid[2], grid[3]
            if np.any(msk):
                pt_list.append(Points(pix[msk] + prj.offset, n_oct, oid))

    pts_n = Points.concat(pt_list)
    b_pxs = reg.project(pts_n, [n_oct, b_oct])
    ok = b_oct.valid(b_pxs)
    pts_n = pts_n.select(ok)
    b_pts = b_pxs.select(ok)

    # Net coordinate bounds
    xc, yc = pts_n.coords[:, 0], pts_n.coords[:, 1]
    net_xmin, net_xmax = float(xc.min()), float(xc.max())
    net_ymin, net_ymax = float(yc.min()), float(yc.max())
    net_span_x = net_xmax - net_xmin
    net_span_y = net_ymax - net_ymin

    # Pre-compute viewport centres for each integer level.
    # Level 0: net midpoint.  Level L>0: centroid of hex at layer L+1.
    centres = [np.array([(net_xmin + net_xmax) / 2.0, (net_ymin + net_ymax) / 2.0])]
    for _L in range(1, max_layer + 1):
        c_n, _ = _hex_at_layer(poi_b, _L + 1, b_oct, n_oct, reg)
        centres.append(np.array(c_n))

    # centre='poi': lock to the POI's exact n_oct position from L=1 onwards.
    # L=0→1 still drifts from the global midpoint to the POI (centres[0] unchanged).
    if centre == 'poi':
        poi_n = reg.project(poi_b, [b_oct, n_oct])
        poi_centre = poi_n.coords[0].copy()
        for _L in range(1, max_layer + 1):
            centres[_L] = poi_centre

    def _viewport_t(t: float) -> list:
        """Continuous viewport at fractional zoom level t."""
        L_lo = min(int(t), max_layer - 1)
        L_hi = L_lo + 1
        alpha = t - L_lo
        cx, cy = (1.0 - alpha) * centres[L_lo] + alpha * centres[L_hi]
        span_x = net_span_x / (3.0 ** t)
        span_y = net_span_y / (3.0 ** t)
        return [
            max(net_xmin, cx - span_x / 2),
            min(net_xmax, cx + span_x / 2),
            max(net_ymin, cy - span_y / 2),
            min(net_ymax, cy + span_y / 2),
        ]

    # Build the sequence of (frame_index, t, label) tuples.
    # frames_per_level=0 → one entry per integer level, labelled by L.
    # frames_per_level=N → lead-in hold + linear zoom, labelled by frame index.
    tx_rot = np.rad2deg(-n_oct.projs['NEA'].theta)
    animate = frames_per_level > 0
    if not animate:
        frame_seq = [(L, float(L), f'L{L:02d}') for L in range(max_layer + 1)]
    else:
        fpl = frames_per_level
        ts = [0.0] * 1  # lead-in: hold global view
        ts += [i / fpl for i in range(max_layer * fpl + 1)]
        frame_seq = [(f, t, f'frame_{f:04d}') for f, t in enumerate(ts)]

    # Per-frame rendering
    flv = flavour.replace(':', '_')
    os.makedirs('output/ex0098', exist_ok=True)
    for frame_idx, t, label in frame_seq:
        if frame_idx < start_frame or (end_frame is not None and frame_idx > end_frame):
            continue
        L = min(int(t), max_layer)
        f_name = f'output/ex0098/{flv}_{scale}_{label}.png'
        if skip_existing and os.path.exists(f_name):
            print(f'  {label}  skip (exists)')
            continue
        print(f'  {label}  (t={t:.3f}  L={L})')
        src_fn = _pick_source(bg_sources, L)

        if animate or L > 0:
            vp = _viewport_t(t) if animate else _viewport(L, centres[L], net_xmin, net_xmax, net_ymin, net_ymax)
            overlay_layers = [L, L + 1, L + 2] if L > 0 else [1, 2]
        else:
            vp = [net_xmin, net_xmax, net_ymin, net_ymax]
            overlay_layers = [1, 2]

        # Viewport pixel placement + colour sampling.
        # Discrete L=0: subsample the pre-projected global grid (resolution matches scale).
        # All other cases: generate a fresh img_w × img_h grid in the viewport.
        if not animate and L == 0:
            px, py, in_vp = _fit_viewport(pts_n, img_w, img_h, vp)
            b_pts_bg = b_pts.select(in_vp)
        else:
            px, py, b_pts_bg = _viewport_grid(vp, img_w, img_h, n_oct, b_oct, reg)
        rgba_vp = src_fn(b_pts_bg)
        if rgba_vp.shape[1] < 4:
            rgba_vp = np.hstack(
                (rgba_vp, np.ones((rgba_vp.shape[0], 1), dtype=rgba_vp.dtype))
            )
        out = np.zeros((img_h, img_w, 4), dtype=np.float32)
        out[py, px] = rgba_vp

        # EDT nearest-neighbour fill for intra-net gaps (zoomed frames only;
        # L=0 unsampled pixels are genuine OOB and should stay transparent)
        if L > 0:
            sampled = out[:, :, 3] > 0
            if not np.all(sampled):
                _, nn = distance_transform_edt(~sampled, return_indices=True)
                rows, cols = np.where(~sampled)
                out[rows, cols] = out[nn[0][rows, cols], nn[1][rows, cols]]

        fig = plt.figure(figsize=(img_w / 100, img_h / 100), dpi=100, frameon=False)
        fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        ax.set_xlim(vp[0], vp[1])
        ax.set_ylim(vp[2], vp[3])
        ax.imshow(out, extent=[vp[0], vp[1], vp[2], vp[3]],
                  alpha=1.0, origin='upper', aspect='auto')

        # Alpha ramp: new finest layer fades in over first 50% of the level.
        # Static frames (not animate) always use full alpha.
        frac = t - int(t)
        ramp_alpha = 0.5 if not animate else min(frac / 0.9, 0.5)

        # Hex outlines — coarse (thick) → fine (thin)
        # L=0: hex_reduce on the global b_pts (layers 1–2 are well-sampled at scale).
        # L>0: poly_net_field on the viewport rectangle guarantees every hex in view.
        vp_pts = None
        if L > 0:
            vp_pts = Points(np.array([
                [vp[0], vp[2]], [vp[1], vp[2]],
                [vp[1], vp[3]], [vp[0], vp[3]],
            ]), n_oct)
        lw = [2.0, 1.0, 0.5]
        coarse_hexes, coarse_v_k, coarse_layer = None, None, overlay_layers[0]
        mid_hexes, mid_v_k = None, None
        for li, layer in enumerate(overlay_layers):
            if L == 0:
                hex_num, hex_v_k, _, _ = hex_reduce(b_pts, layer)
            else:
                lattice_n = poly_net_field(vp_pts, layer)
                b_lattice, _ = _project_noct_pts(lattice_n.coords, n_oct, b_oct, reg)
                if b_lattice is None:
                    continue
                hex_num, hex_v_k, _, _ = hex_reduce(b_lattice, layer)
            if hex_v_k is None or len(hex_v_k) == 0:
                continue
            hex_par, hex_oid, hex_scale = hex_parents(b_oct, hex_v_k, hex_num)
            xc2, _, xpm = tail_unpack_reversible(hex_v_k[:, -1])   # (c2, r_mo, p_mo)
            verts_n = hex_verts_in_noct(hex_par, hex_oid, xpm, xc2, hex_scale, n_oct)
            hexes = verts_n.coords.reshape(-1, 6, 2)
            # Finest (newest) layer fades in; coarsest (oldest) layer fades out
            # over the tail of the level so it leaves frame gracefully; middle
            # holds at full opacity.
            if li == len(overlay_layers) - 1:
                ec_a = ramp_alpha
            elif li == 0:
                ec_a = 0.5 * (min((1.0 - frac) / 0.3, 1.0) if animate else 1.0)
            else:
                ec_a = 0.5
            ax.add_collection(PolyCollection(
                hexes, facecolors='none', ec=(1, 1, 1, ec_a),
                linewidth=lw[li], antialiaseds=True,
            ))
            if li == 0:
                coarse_hexes, coarse_v_k = hexes, hex_v_k
            elif li == 1:
                mid_hexes, mid_v_k = hexes, hex_v_k

        # Address labels (L>0, coarse layer ≤~40 hexes in view).
        # Cross-fade across the level boundary: the parent (coarse) layer holds
        # full then fades out over the tail, while the child (middle) layer fades
        # in starting slightly earlier — so the two overlap through the switch
        # instead of hard-cutting.  The child reaches full exactly at frac→1,
        # handing off continuously to the next level where it becomes the coarse
        # layer.  Children inherit the parent prefix, so it strips cleanly from
        # both label sets and the bottom-left caption stays stable.
        def _draw_layer_labels(hexes, v_k, alpha, plen):
            if hexes is None or alpha <= 0.0:
                return
            ctrs = np.mean(hexes, axis=1)  # (N, 2)
            # Bottom-left vertex = min(x + y) per hex; offset 3% toward centroid
            bl_idx = np.argmin(hexes[:, :, 0] + hexes[:, :, 1], axis=1)
            bl_verts = hexes[np.arange(len(hexes)), bl_idx]  # (N, 2)
            label_pos = bl_verts + 0.03 * (ctrs - bl_verts)
            strs = [''.join(f'{int(d):01x}' for d in row[:-1]) for row in v_k]
            for (lx, ly), lbl in zip(label_pos, strs):
                ax.text(lx, ly, lbl[plen:],
                        fontsize=12, ha='left', va='bottom',
                        color=(1.0, 1.0, 1.0, alpha),
                        zorder=100, clip_on=True)

        hex_text = f'Hex9 // Demo'
        if coarse_hexes is not None and coarse_hexes.shape[0] <= 40:
            lab_str = [''.join(f'{int(d):01x}' for d in row[:-1]) for row in coarse_v_k]
            prefix = _common_prefix(lab_str)
            plen = len(prefix)
            parent_a = min((1.0 - frac) / 0.3, 1.0) if animate else 1.0
            child_a = max(0.0, min((frac - 0.55) / 0.45, 1.0)) if animate else 0.0
            _draw_layer_labels(coarse_hexes, coarse_v_k, parent_a, plen)
            if mid_hexes is not None and mid_hexes.shape[0] <= 120:
                _draw_layer_labels(mid_hexes, mid_v_k, child_a, plen)
            # Bottom-left caption — the shared root never fades (it is the frame's
            # anchor, not part of the per-hex label crossfade).
            if prefix:
                hex_text += f' // H9 Prefix: {prefix}'

        ax.text(0.012, 0.012, hex_text,
                transform=ax.transAxes,
                fontsize=16, ha='left', va='bottom', family='monospace',
                color=(1.0, 0.85, 0.0, 1.0),
                zorder=150, clip_on=False)

        # Gold highlight: middle-layer hex containing the POI
        _, cur_poly = _hex_at_layer(poi_b, L, b_oct, n_oct, reg)
        ax.add_collection(PolyCollection(
            cur_poly, facecolors='none',
            edgecolors=[(1.0, 0.85, 0.0, 1.0)], linewidth=2.0,
        ))
        if L > -1:
            _, gold_poly = _hex_at_layer(poi_b, L + 1, b_oct, n_oct, reg)
            ax.add_collection(PolyCollection(
                gold_poly,
                facecolors='none',
                edgecolors=[(1.0, 0.85, 0.0, 1.0)],
                linewidth=1.5,
            ))
        if L > -2:
            _, gold_poly = _hex_at_layer(poi_b, L + 2, b_oct, n_oct, reg)
            ax.add_collection(PolyCollection(
                gold_poly,
                facecolors='none',
                edgecolors=[(1.0, 0.85, 0.0, ramp_alpha)],
                linewidth=1.0,
            ))
        # Imagery credit for the active source (lower-right)
        credit = getattr(src_fn, 'attribution', '')
        if credit:
            ax.text(0.988, 0.012, credit,
                    transform=ax.transAxes,
                    fontsize=11, ha='right', va='bottom',
                    color=(1.0, 1.0, 1.0, 0.85),
                    zorder=150, clip_on=False)

        # Dashed yellow rectangle: next frame's viewport
        # if L < max_layer:
        #     next_ctr, _ = _hex_at_layer(poi_b, L + 2, b_oct, n_oct, reg)
        #     nsp_x = net_span_x / (3.0 ** (L + 1))
        #     nsp_y = net_span_y / (3.0 ** (L + 1))
        #     ncx, ncy = next_ctr
        #     nx0 = max(net_xmin, ncx - nsp_x / 2)
        #     nx1 = min(net_xmax, ncx + nsp_x / 2)
        #     ny0 = max(net_ymin, ncy - nsp_y / 2)
        #     ny1 = min(net_ymax, ncy + nsp_y / 2)
        #     ax.plot(
        #         [nx0, nx1, nx1, nx0, nx0],
        #         [ny0, ny0, ny1, ny1, ny0],
        #         color='yellow', lw=1.5, linestyle='--', zorder=200,
        #     )
        fig.savefig(f_name, dpi=100)
        plt.close(fig)
        print(f'  saved {f_name}')


if __name__ == '__main__':
    gdal.UseExceptions()
    _reg = Registrar()
    _p_pix = _reg.domain('p_pix')
    _b_oct = _reg.domain('b_oct')

    EOX_WMTS='https://tiles.maps.eox.at/wmts/1.0.0/WMTSCapabilities.xml'
    ESRI_WMTS='https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/WMTS/1.0.0/WMTSCapabilities.xml'
    OAM_URL='https://tiles.openaerialmap.org/6107d91f343da30006976e12/0/6107d91f343da30006976e13/{z}/{x}/{y}'

    _pc = make_pc_source('src/bm_3600x1800.png', _reg, _p_pix, _b_oct,
                         attribution='NASA Visible Earth · Blue Marble')
    _s2 = make_wmts_source(EOX_WMTS, 's2cloudless-2020', _reg, _b_oct, gain=1.1, gamma=1.1,
                           attribution='Sentinel-2 cloudless 2020 © EOX IT Services GmbH')
    _s3 = make_wmts_source(ESRI_WMTS, 'World_Imagery', _reg, _b_oct, gain=1.0, gamma=1.0,
                           attribution='Esri World Imagery · Esri, Maxar, Earthstar Geographics')
    _s4 = make_xyz_source(OAM_URL, _reg, _b_oct, max_zoom=22, gain=1.0, gamma=1.0,
                          attribution='Pierre d\'HUY ©2021  · OpenAerialMap, CC BY-SA 4.0')
    _s5 = make_bm_source('lawn171.tif', _reg, _b_oct, gamma=1.0,
                         name='lawn171_wkt', attribution='')
    _s6 = make_bm_source('lawn_1mm.tif', _reg, _b_oct, gamma=1.0,
                         name='lawn_1mm', attribution='')
    _s7 = make_bm_source('fly_1mm.tif', _reg, _b_oct, gamma=1.0,
                         name='fly_wkt', attribution='')

    run(
        reg=_reg,
        flavour='butterfly:0500',
        scale=600,
        max_layer=22,
        bg_sources={
            # 0: _pc,
            0: _s2,
            6: _s3,
            10: _s4,
            # 13: _s5,   # lawn/table photo (0.8 m) — viewport fits it ~L16
            14: _s6,   # lawn — viewport fits it ~L20
            17: _s7,  # fly (1 cm) — viewport fits it ~L20
        },
        frames_per_level=24,   # set >0 for smooth animation, e.g. 24
        skip_existing=True,    # resume: skip frames whose PNG is already on disk
        # start_frame=300,     # or resume from an explicit frame index
    )