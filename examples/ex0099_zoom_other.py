# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Zoom sequence in Web Mercator (EPSG:3857): equal-area H9 hexes on a
conformal base map.

Counterpart to ex0098 (conformal land on equal-area net).  Here H9 hexes
are projected into Mercator space — their shapes distort toward the poles,
making the equal-area property visible by contrast.

The backdrop is a WMTS tile canvas.  For Mercator sources (ESRI World Imagery)
tiles are stitched directly — no resampling needed.  For WGS84 sources (EOX
S2 cloudless) the equirectangular canvas is bilinearly resampled to Mercator
so geography aligns correctly with the hex overlay at all zoom levels.
Tile disk-cache from ex0098 is shared.

Hex overlays are computed via n_oct (butterfly layout, fewer rigid transforms
than rhombus) then projected to lat/lon and geodesically densified for
coarse layers before conversion to Mercator metres.

Point of Interest: Rotch Library, MIT, Cambridge MA.

Last Tested
16 Jun 2026 0.1.3a0 (passed) 53.3s
28 Mar 2026 (new)
"""
import hashlib
import io
import math
import os
import urllib.parse
import urllib.request

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from owslib.wmts import WebMapTileService
from PIL import Image as PILImage
from scipy.interpolate import RegularGridInterpolator

from hhg9 import Registrar, Points
from hhg9.algorithms.geometry import inside_convex_polygon_cw
from hhg9.h9 import H9O
from hhg9.h9.binning import hex_reduce, hex_parents
from hhg9.h9.grid import hex_verts_in_noct, poly_net_field, hex_props
from hhg9.h9.polygon import region_grid
from hhg9.h9.tail import tail_unpack_reversible


POI_LAT, POI_LON = 42.359486311195376, -71.09321831450787   # Rotch Library, MIT


# ── Mercator helpers ──────────────────────────────────────────────────────────

MERC_R   = 6_378_137.0
MERC_MAX = math.pi * MERC_R            # ≈ 20 037 508 m  (half world width)


def ll_to_merc(lats, lons):
    """(lat_deg, lon_deg) arrays → (x, y) Web Mercator metres."""
    lats = np.clip(np.asarray(lats, dtype=float), -85.0511, 85.0511)
    lons = np.asarray(lons, dtype=float)
    x = np.radians(lons) * MERC_R
    y = np.log(np.tan(math.pi / 4 + np.radians(lats) / 2)) * MERC_R
    return x, y


def merc_to_ll(xs, ys):
    """(x, y) Web Mercator metres → (lat_deg, lon_deg) arrays."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    lons = np.degrees(xs / MERC_R)
    lats = np.degrees(2 * np.arctan(np.exp(ys / MERC_R)) - math.pi / 2)
    return lats, lons


# ── viewport ─────────────────────────────────────────────────────────────────

def _viewport_merc(t, poi_mx, poi_my):
    """Mercator viewport [x_min, x_max, y_min, y_max] at fractional zoom t.

    Span shrinks by 3^t centred on the POI.  Tile fetcher clamps to valid range.
    """
    span = 2.0 * MERC_MAX / (3.0 ** t)
    return [poi_mx - span / 2.0, poi_mx + span / 2.0,
            poi_my - span / 2.0, poi_my + span / 2.0]


# ── n_oct helpers ─────────────────────────────────────────────────────────────

def _project_noct_pts(coords_n, n_oct, b_oct, reg, tol=0.0):
    """Assign face oids to n_oct coordinates and project to b_oct.
    Shared pattern with ex0098.  Returns (b_pts, idx).
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


def _noct_vp_from_merc(vp_merc, n_oct, b_oct, g_gcd, reg, n_edge=20):
    """Convert a Mercator viewport bbox to a Points polygon in n_oct space.

    Densifies each edge of the bbox (to capture curvature after projection),
    then projects:  Merc → lat/lon → g_gcd → b_oct → n_oct.
    Returns Points in n_oct with no oid — suitable for poly_net_field.
    """
    x_min, x_max, y_min, y_max = vp_merc
    corners = np.array([[x_min, y_min], [x_max, y_min],
                         [x_max, y_max], [x_min, y_max]])
    pts_m = []
    for i in range(4):
        p0, p1 = corners[i], corners[(i + 1) % 4]
        for k in range(n_edge):
            pts_m.append(p0 + k / n_edge * (p1 - p0))
    pts_m = np.array(pts_m)
    # Clamp y to valid Mercator range before converting
    pts_m[:, 1] = np.clip(pts_m[:, 1], -MERC_MAX * 0.9999, MERC_MAX * 0.9999)
    lats, lons = merc_to_ll(pts_m[:, 0], pts_m[:, 1])
    b_pts = reg.project(Points(np.stack([lats, lons], axis=1), g_gcd), [g_gcd, b_oct])
    n_pts = reg.project(b_pts, [b_oct, n_oct])
    return Points(n_pts.coords, n_oct)


# ── tile fetcher ──────────────────────────────────────────────────────────────

TILE_PX = 256


def make_tile_fetcher(capabilities_url, layer_name, *,
                      tile_matrix_set=None, cache_dir='wmts_cache'):
    """Build a tile-canvas fetcher; handles both Mercator and WGS84 TMS.

    Returns callable  (vp_merc, z) → (canvas, x_left, x_right, y_bot, y_top)
    where canvas is (H, W, 3) float32 [0, 1] north-at-top, and the four
    extent values are always in Web Mercator metres.

    For WGS84 sources (e.g. EOX S2 cloudless) tile indices and extents are
    computed in lat/lon; the canvas extent is converted to Mercator for
    display.  The image will be slightly stretched at global scale (PlateCarrée
    pixels rendered in a Mercator frame) but is accurate at local/city scale.

    Tiles are disk-cached using the same directory and key format as
    make_wmts_source in ex0098, so caches are shared.
    """
    os.makedirs(cache_dir, exist_ok=True)
    wmts  = WebMapTileService(capabilities_url)
    layer = wmts.contents[layer_name]

    available = list(layer.tilematrixsetlinks.keys())
    if tile_matrix_set and tile_matrix_set in available:
        tms = tile_matrix_set
    else:
        merc_names = ['GoogleMapsCompatible', 'g', 'EPSG:3857', 'WebMercatorQuad',
                      'googlemapscompatible', 'EPSG_3857']
        tms = next((t for t in merc_names if t in available), available[0])
    print(f'  WMTS: layer={layer_name!r}  tms={tms!r}  available={available}')

    tile_fmt = layer.formats[0] if layer.formats else 'image/jpeg'
    tms_obj  = wmts.tilematrixsets[tms]
    crs_str  = getattr(tms_obj, 'crs', '') or ''
    is_geographic = ('4326' in crs_str or 'CRS84' in crs_str
                     or 'WGS84' in crs_str.upper())
    zoom_ids = sorted(tms_obj.tilematrix, key=lambda z: int(z.split(':')[-1]))
    max_zoom = int(zoom_ids[-1].split(':')[-1])
    zoom_to_id = {int(z.split(':')[-1]): z for z in zoom_ids}

    kvp_base = capabilities_url.split('?')[0].replace('WMTSCapabilities.xml', '')

    def _tile_url(tilematrix_id, row, col):
        return kvp_base + '?' + urllib.parse.urlencode({
            'SERVICE': 'WMTS', 'REQUEST': 'GetTile', 'VERSION': '1.0.0',
            'LAYER': layer_name, 'STYLE': 'default', 'FORMAT': tile_fmt,
            'TILEMATRIXSET': tms, 'TILEMATRIX': tilematrix_id,
            'TILEROW': row, 'TILECOL': col,
        })

    def _fetch_tile(tx, ty, z):
        key  = f"{layer_name}:{tms}:{z}:{tx}:{ty}"
        path = os.path.join(cache_dir,
                            hashlib.md5(key.encode()).hexdigest()[:16] + '.png')
        if os.path.exists(path):
            return np.asarray(PILImage.open(path).convert('RGB'),
                              dtype=np.float32) / 255.0
        with urllib.request.urlopen(_tile_url(zoom_to_id[z], ty, tx)) as resp:
            img = PILImage.open(io.BytesIO(resp.read())).convert('RGB')
        img.save(path)
        return np.asarray(img, dtype=np.float32) / 255.0

    def _stitch(tx_min, ty_min, tx_max, ty_max, z):
        n_tx = tx_max - tx_min + 1
        n_ty = ty_max - ty_min + 1
        canvas = np.zeros((n_ty * TILE_PX, n_tx * TILE_PX, 3), dtype=np.float32)
        for ty_i in range(ty_min, ty_max + 1):
            for tx_i in range(tx_min, tx_max + 1):
                arr = _fetch_tile(tx_i, ty_i, z)
                iy = (ty_i - ty_min) * TILE_PX
                ix = (tx_i - tx_min) * TILE_PX
                canvas[iy:iy + TILE_PX, ix:ix + TILE_PX] = arr
        return canvas

    def fetch(vp_merc, z):
        """Stitch tiles covering vp_merc at zoom level z.

        Returns (canvas, x_left, x_right, y_bot, y_top) in Mercator metres.
        canvas rows are north-to-south (row 0 = y_top).
        """
        z = max(0, min(max_zoom, int(round(z))))
        x_min, x_max, y_min, y_max = vp_merc

        if not is_geographic:
            # ── Mercator tiles (GoogleMapsCompatible) ──────────────────────
            n_tiles = 2 ** z

            def _merc_to_tile(mx, my):
                tx = int(np.clip((mx + MERC_MAX) / (2 * MERC_MAX) * n_tiles,
                                 0, n_tiles - 1))
                ty = int(np.clip((MERC_MAX - my) / (2 * MERC_MAX) * n_tiles,
                                 0, n_tiles - 1))
                return tx, ty

            tx_min, ty_min = _merc_to_tile(x_min, y_max)
            tx_max, ty_max = _merc_to_tile(x_max, y_min)
            canvas = _stitch(tx_min, ty_min, tx_max, ty_max, z)
            xl  = -MERC_MAX + (tx_min       / n_tiles) * 2 * MERC_MAX
            xr  = -MERC_MAX + ((tx_max + 1) / n_tiles) * 2 * MERC_MAX
            yt  =  MERC_MAX - (ty_min       / n_tiles) * 2 * MERC_MAX
            yb  =  MERC_MAX - ((ty_max + 1) / n_tiles) * 2 * MERC_MAX
            return canvas, xl, xr, yb, yt

        else:
            # ── WGS84 / geographic tiles ────────────────────────────────────
            # Read tile grid dimensions for this zoom from capabilities
            # (WGS84 TMS is typically 2^(z+1) cols × 2^z rows, not 2^z × 2^z)
            tm     = tms_obj.tilematrix[zoom_to_id[z]]
            n_cols = tm.matrixwidth
            n_rows = tm.matrixheight

            # Convert Mercator viewport corners to lat/lon for tile indexing
            vp_lats, vp_lons = merc_to_ll(
                np.array([x_min, x_max, x_min, x_max]),
                np.array([y_min, y_min, y_max, y_max]),
            )
            lon_l = float(vp_lons[0])          # x_min → westernmost lon
            lon_r = float(vp_lons[1])          # x_max → easternmost lon
            lat_t = float(vp_lats[2])          # y_max → northernmost lat
            lat_b = float(vp_lats[0])          # y_min → southernmost lat

            def _ll_to_tile(lon, lat):
                tx = int(np.clip((lon + 180) / 360 * n_cols, 0, n_cols - 1))
                ty = int(np.clip((90 - lat) / 180 * n_rows, 0, n_rows - 1))
                return tx, ty

            tx_min, ty_min = _ll_to_tile(lon_l, lat_t)
            tx_max, ty_max = _ll_to_tile(lon_r, lat_b)
            canvas = _stitch(tx_min, ty_min, tx_max, ty_max, z)

            # WGS84 lat/lon extent of stitched canvas
            lon_l_c = -180 + (tx_min       / n_cols) * 360
            lon_r_c = -180 + ((tx_max + 1) / n_cols) * 360
            lat_t_c = float(np.clip(90 - (ty_min       / n_rows) * 180,
                                    -85.0511, 85.0511))
            lat_b_c = float(np.clip(90 - ((ty_max + 1) / n_rows) * 180,
                                    -85.0511, 85.0511))

            # Resample equirectangular canvas → Mercator grid covering the
            # viewport.  Uses RegularGridInterpolator (bilinear) so each output
            # pixel correctly samples the WGS84 canvas at its true lat/lon.
            h_c, w_c = canvas.shape[:2]
            lat_axis = np.linspace(lat_t_c, lat_b_c, h_c)  # decreasing top→bot
            lon_axis = np.linspace(lon_l_c, lon_r_c, w_c)  # increasing left→right
            # RGI needs monotone-increasing axes — flip lat
            canvas_flip = canvas[::-1]
            lat_axis_inc = lat_axis[::-1]

            out_h = out_w = max(h_c, w_c)   # square output matching tile density
            merc_xs = np.linspace(x_min, x_max, out_w)
            merc_ys = np.linspace(y_max, y_min, out_h)   # top→bottom
            xx_m, yy_m = np.meshgrid(merc_xs, merc_ys)
            out_lats, out_lons = merc_to_ll(xx_m.ravel(), yy_m.ravel())
            pts_q = np.stack([out_lats, out_lons], axis=1)

            resampled = np.stack([
                RegularGridInterpolator(
                    (lat_axis_inc, lon_axis), canvas_flip[:, :, c],
                    method='linear', bounds_error=False, fill_value=0.0,
                )(pts_q)
                for c in range(3)
            ], axis=1).reshape(out_h, out_w, 3).astype(np.float32)

            # Return with exact viewport Mercator extent — no stretch
            return resampled, x_min, x_max, y_min, y_max

    return fetch


def _pick_source(bg_sources, L):
    if callable(bg_sources):
        return bg_sources
    candidates = [k for k in bg_sources if k <= L]
    key = max(candidates) if candidates else min(bg_sources)
    return bg_sources[key]


# ── hex geometry in Mercator ──────────────────────────────────────────────────

def _clip_poly_to_viewport(poly, x_min, x_max, y_min, y_max):
    """Sutherland-Hodgman clip of an (N, 2) polygon array to the viewport rectangle.

    Returns a clipped (M, 2) array, or None if the polygon is entirely outside.
    Works correctly for large hexes that straddle or dwarf the viewport.
    """
    def _sh_clip(pts, inside, intersect):
        out = []
        if not pts:
            return out
        prev, p_in = pts[-1], inside(pts[-1])
        for curr in pts:
            c_in = inside(curr)
            if c_in:
                if not p_in:
                    out.append(intersect(prev, curr))
                out.append(curr)
            elif p_in:
                out.append(intersect(prev, curr))
            prev, p_in = curr, c_in
        return out

    def _xi(a, b, x):           # intersection with vertical x = x
        t = (x - a[0]) / (b[0] - a[0])
        return [x, a[1] + t * (b[1] - a[1])]

    def _yi(a, b, y):           # intersection with horizontal y = y
        t = (y - a[1]) / (b[1] - a[1])
        return [a[0] + t * (b[0] - a[0]), y]

    pts = poly.tolist()
    pts = _sh_clip(pts, lambda p: p[0] >= x_min, lambda a, b: _xi(a, b, x_min))
    pts = _sh_clip(pts, lambda p: p[0] <= x_max, lambda a, b: _xi(a, b, x_max))
    pts = _sh_clip(pts, lambda p: p[1] >= y_min, lambda a, b: _yi(a, b, y_min))
    pts = _sh_clip(pts, lambda p: p[1] <= y_max, lambda a, b: _yi(a, b, y_max))
    return np.array(pts) if len(pts) >= 3 else None


def _densify_noct_verts(verts_flat, n_per_edge):
    """Linearly interpolate hex edges in n_oct space.

    verts_flat  (H, 6, 2) n_oct coords.
    n_per_edge  number of interpolated points per edge (including start, excluding end).

    Returns (H * 6 * n_per_edge, 2) flat array and n_per_edge (unchanged).
    Adjacent hexes share edges traversed in the same coordinate space, so
    interpolation produces identical intermediate points → no delamination.
    """
    if n_per_edge <= 1:
        return verts_flat.reshape(-1, 2), 6
    v      = verts_flat                                        # (H, 6, 2)
    v_next = np.roll(v, -1, axis=1)                           # (H, 6, 2) shifted
    t      = np.linspace(0.0, 1.0, n_per_edge, endpoint=False)
    pts    = v[:, :, None, :] + t[None, None, :, None] * (v_next - v)[:, :, None, :]
    return pts.reshape(-1, 2), 6 * n_per_edge


def _verts_to_merc(hex_par, hex_oid, flat_noct, n_pts_per_hex,
                   n_oct, b_oct, c_oct, c_ell, g_gcd, reg):
    """Project n_oct flat coords → b_oct → g_gcd → Mercator.

    hex_par        (H,) parent indices.
    hex_oid        (H,) octahedral face oids.
    flat_noct      (H * n_pts_per_hex, 2) already-densified n_oct coords.
    n_pts_per_hex  points per hex in flat_noct.

    Returns (H, n_pts_per_hex, 2) Mercator array; rows with any NaN indicate
    projection failure (caller should skip those hexes).
    """
    H      = len(hex_par)
    total  = H * n_pts_per_hex
    result = np.full((total, 2), np.nan)

    # Compute which oid each flat point belongs to (each hex repeated n_pts_per_hex)
    oid_per_pt = np.repeat(hex_oid, n_pts_per_hex)

    for side, proj in n_oct.projs.items():
        oid  = n_oct.sides[side].oid
        mask = oid_per_pt == oid
        if not np.any(mask):
            continue
        b_side = reg.project(Points(flat_noct[mask], n_oct, oid), [n_oct, b_oct])
        g_side = reg.project(b_side, [b_oct, c_oct, c_ell, g_gcd])
        mx, my = ll_to_merc(g_side.coords[:, 0], g_side.coords[:, 1])
        result[mask] = np.stack([mx, my], axis=1)

    return result.reshape(H, n_pts_per_hex, 2)


_ANALYTIC_LAYER_THRESHOLD = 4   # analytic for L < this; scanline for L >=


def _hexes_for_layer_analytic(vp_merc, layer, n_oct, b_oct, c_oct, c_ell, g_gcd, reg):
    """Analytic hex mesh for coarse layers (L < _ANALYTIC_LAYER_THRESHOLD).

    Enumerates all hexes via region_grid LUTs across all octant faces —
    no viewport-to-n_oct conversion needed, so large hexes that straddle the
    viewport boundary are never missed.  Feeds the same hex_verts_in_noct →
    densify → Mercator path as the scanline route, so boundary-hex handling
    is identical.  Returns no key addresses (labels are skipped at coarse layers).
    """
    par_list, oid_list, xpm_list, xc2_list = [], [], [], []

    for side, sdom in n_oct.sides.items():
        oid  = int(sdom.oid)
        mode = int(H9O.oid_mo[oid])

        if layer == 0:
            supercells = [(mode, np.zeros(2, dtype=float))]
        else:
            supercells = [(int(sc_mode), np.asarray(sc_origin, dtype=float))
                         for _, sc_mode, sc_origin, _ in region_grid(layer - 1, mode)]

        for sc_mode, sc_origin in supercells:
            for c2 in range(3):
                par_list.append(sc_origin)
                oid_list.append(oid)
                xpm_list.append(sc_mode)
                xc2_list.append(c2)

    if not par_list:
        return [], np.empty((0,), dtype=int)

    hex_par  = np.array(par_list,  dtype=float)        # (H, 2)
    hex_oid  = np.array(oid_list,  dtype=np.uint8)     # (H,)
    xpm      = np.array(xpm_list,  dtype=np.uint8)     # (H,)
    xc2      = np.array(xc2_list,  dtype=np.uint8)     # (H,)
    H        = len(hex_par)
    hex_scale = 3.0 ** -layer

    verts_n = hex_verts_in_noct(hex_par, hex_oid, xpm, xc2, hex_scale, n_oct)

    side_m     = float(hex_props(layer)[1])
    n_per_edge = max(1, int(math.ceil(side_m / 50_000.0)))
    flat_n, n_pts = _densify_noct_verts(verts_n.coords.reshape(H, 6, 2), n_per_edge)

    verts_m = _verts_to_merc(hex_par, hex_oid, flat_n, n_pts,
                             n_oct, b_oct, c_oct, c_ell, g_gcd, reg)

    x_min, x_max, y_min, y_max = vp_merc
    span = x_max - x_min
    polys_merc = []
    for vm in verts_m:
        if not np.all(np.isfinite(vm)):
            continue
        bb_xmin = float(vm[:, 0].min())
        bb_xmax = float(vm[:, 0].max())
        bb_ymin = float(vm[:, 1].min())
        bb_ymax = float(vm[:, 1].max())
        if bb_xmin <= x_max + span and bb_xmax >= x_min - span and \
                bb_ymin <= y_max + span and bb_ymax >= y_min - span:
            polys_merc.append(vm)

    return polys_merc, np.empty((0,), dtype=int)


def _hexes_for_layer_scanline(vp_merc, layer, n_oct, b_oct, c_oct, c_ell, g_gcd, reg):
    """Return (polys_merc, hex_v_k) for hexes visible in the viewport (scanline fill).

    polys_merc  list of (K, 2) Mercator (x, y) arrays (densified in n_oct space).
    hex_v_k     (H, depth) int array of hex addresses, parallel to polys_merc.
    """
    vp_n = _noct_vp_from_merc(vp_merc, n_oct, b_oct, g_gcd, reg)
    lattice_n = poly_net_field(vp_n, layer)
    b_lattice, _ = _project_noct_pts(lattice_n.coords, n_oct, b_oct, reg)
    if b_lattice is None:
        return [], np.empty((0,), dtype=int)

    hex_num, hex_v_k, _, _ = hex_reduce(b_lattice, layer)
    if hex_v_k is None or len(hex_v_k) == 0:
        return [], np.empty((0,), dtype=int)

    hex_par, hex_oid, hex_scale = hex_parents(b_oct, hex_v_k, hex_num)
    xpm, xc2, _, _ = tail_unpack_reversible(hex_v_k[:, -1])
    verts_n = hex_verts_in_noct(hex_par, hex_oid, xpm, xc2, hex_scale, n_oct)

    H = len(hex_par)
    side = hex_props(layer)[1]
    n_per_edge = max(1, int(math.ceil(side / 50_000.0)))
    flat_n, n_pts = _densify_noct_verts(verts_n.coords.reshape(H, 6, 2), n_per_edge)

    verts_m = _verts_to_merc(
        hex_par, hex_oid, flat_n, n_pts,
        n_oct, b_oct, c_oct, c_ell, g_gcd, reg)

    polys_merc, valid_keys = [], []
    for vm, vk in zip(verts_m, hex_v_k):
        if not np.all(np.isfinite(vm)):
            continue
        polys_merc.append(vm)
        valid_keys.append(vk)

    return polys_merc, (np.array(valid_keys) if valid_keys
                        else np.empty((0,), dtype=int))


def _hexes_for_layer(vp_merc, layer, n_oct, b_oct, c_oct, c_ell, g_gcd, reg):
    """Dispatch to analytic (coarse) or scanline (fine) hex enumeration."""
    if layer < _ANALYTIC_LAYER_THRESHOLD:
        return _hexes_for_layer_analytic(vp_merc, layer, n_oct, b_oct, c_oct, c_ell, g_gcd, reg)
    return _hexes_for_layer_scanline(vp_merc, layer, n_oct, b_oct, c_oct, c_ell, g_gcd, reg)


def _hex_at_poi_merc(poi_b, layer, n_oct, b_oct, c_oct, c_ell, g_gcd, reg):
    """Return a (K, 2) Mercator polygon for the hex at `layer` containing the POI."""
    h_num, h_v, _, _ = hex_reduce(poi_b, layer)
    h_par, h_oid, h_scale = hex_parents(b_oct, h_v, h_num)
    xpm, xc2, _, _ = tail_unpack_reversible(h_v[:, -1])
    verts_n = hex_verts_in_noct(h_par, h_oid, xpm, xc2, h_scale, n_oct)

    side = hex_props(layer)[1]
    n_per_edge = max(1, int(math.ceil(side / 50_000.0)))
    flat_n, n_pts = _densify_noct_verts(verts_n.coords.reshape(1, 6, 2), n_per_edge)
    vm = _verts_to_merc(h_par, h_oid, flat_n, n_pts,
                        n_oct, b_oct, c_oct, c_ell, g_gcd, reg)
    if not np.all(np.isfinite(vm[0])):
        return None
    return vm[0]


# ── main ──────────────────────────────────────────────────────────────────────

def run(*, reg=None, flavour='butterfly', scale=800, max_layer=9, bg_sources,
        frames_per_level: int = 0):
    """Generate zoom-sequence PNGs in Web Mercator (EPSG:3857).

    Args:
        reg:              Shared Registrar (or None to create a new one).
        flavour:          n_oct layout; 'butterfly' or 'mortar' recommended.
        scale:            Output image width/height in pixels (square output).
        max_layer:        Highest H9 zoom level to render.
        bg_sources:       Single callable or dict {min_L: callable}.
                          Callable signature: (vp_merc, z) → (canvas, xl, xr, yb, yt).
                          Build with make_tile_fetcher().
        frames_per_level: 0 → one PNG per integer level.
                          N → smooth animation; N frames per level, 3^t zoom.
    """
    if reg is None:
        reg = Registrar()
    n_oct = reg.domain(f'n_oct:{flavour}')
    b_oct = reg.domain('b_oct')
    c_oct = reg.domain('c_oct')
    c_ell = reg.domain('c_ell')
    g_gcd = reg.domain('g_gcd')

    img_w = img_h = scale   # square: Mercator viewport spans equal x and y

    poi_g   = Points(np.array([[POI_LAT, POI_LON]]), g_gcd)
    poi_b   = reg.project(poi_g, [g_gcd, b_oct])
    _pmx, _pmy = ll_to_merc(np.array([POI_LAT]), np.array([POI_LON]))
    poi_mx, poi_my = float(_pmx[0]), float(_pmy[0])

    animate = frames_per_level > 0
    if not animate:
        frame_seq = [(L, float(L), f'L{L:02d}') for L in range(max_layer + 1)]
    else:
        fpl = frames_per_level
        ts = [i / fpl for i in range(max_layer * fpl + 1)]
        frame_seq = [(f, t, f'frame_{f:04d}') for f, t in enumerate(ts)]

    for frame_idx, t, label in frame_seq:
        L    = min(int(t), max_layer)
        frac = t - int(t)
        print(f'  {label}  (t={t:.3f}  L={L})')

        vp     = _viewport_merc(t, poi_mx, poi_my)
        span_m = vp[1] - vp[0]
        # Tile zoom: target ~1px per tile pixel at output resolution
        z_tiles = max(0, round(math.log2(img_w / 256.0 * 2.0 * MERC_MAX / span_m)))

        src_fn = _pick_source(bg_sources, L)
        canvas, xl, xr, yb, yt = src_fn(vp, z_tiles)

        fig = plt.figure(figsize=(img_w / 100, img_h / 100), dpi=100, frameon=False)
        fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        ax.set_xlim(vp[0], vp[1])
        ax.set_ylim(vp[2], vp[3])
        # canvas rows are north-to-south → origin='upper' maps row-0 to y_top
        ax.imshow(canvas, extent=[xl, xr, yb, yt],
                  origin='upper', aspect='auto', zorder=0)

        ramp_alpha = 0.5 if not animate else min(frac / 0.9, 0.5)
        overlay_layers = [L, L + 1, L + 2] if L > 0 else [1, 2]
        lw = [2.0, 1.0, 0.5]

        coarse_polys, coarse_v_k = None, None
        for li, ov_layer in enumerate(overlay_layers):
            polys_m, v_k = _hexes_for_layer(
                vp, ov_layer, n_oct, b_oct, c_oct, c_ell, g_gcd, reg)
            if not polys_m:
                continue
            ec_a = ramp_alpha if li == len(overlay_layers) - 1 else 0.5
            clipped_m = [c for p in polys_m
                         if (c := _clip_poly_to_viewport(p, *vp)) is not None]
            if not clipped_m:
                continue
            coll = PolyCollection(
                clipped_m, facecolors='none', ec=(1, 1, 1, ec_a),
                linewidth=lw[li], antialiaseds=True,
            )
            ax.add_collection(coll)
            if li == 0:
                coarse_polys, coarse_v_k = polys_m, v_k

        # Labels on coarsest layer when ≤ 40 hexes are in view
        if coarse_polys is not None and L > 0 and len(coarse_polys) <= 40:
            label_alpha = min(frac / 0.3, 1.0) if animate else 1.0
            ctrs    = np.array([np.mean(p, axis=0) for p in coarse_polys])
            bl_verts = np.array([p[np.argmin(p[:, 0] + p[:, 1])]
                                  for p in coarse_polys])
            label_pos = bl_verts + 0.2 * (ctrs - bl_verts)
            lab_str = [''.join(f'{int(d):01x}' for d in row[:-1])
                       for row in coarse_v_k]
            for (lx, ly), lbl in zip(label_pos, lab_str):
                ax.text(lx, ly, lbl,
                        fontsize=10, ha='left', va='bottom',
                        color=(1.0, 1.0, 1.0, label_alpha),
                        zorder=100, clip_on=True)

        # Gold outlines: L, L+1, L+2 hexes containing the POI
        for hi, (gl, lw_g) in enumerate([(L, 2.0), (L + 1, 1.5), (L + 2, 1.0)]):
            poly = _hex_at_poi_merc(
                poi_b, gl, n_oct, b_oct, c_oct, c_ell, g_gcd, reg)
            if poly is None:
                continue
            poly_c = _clip_poly_to_viewport(poly, *vp)
            if poly_c is None:
                continue
            alpha = ramp_alpha if hi == 2 else 1.0
            gold = PolyCollection(
                [poly_c], facecolors='none',
                edgecolors=[(1.0, 0.85, 0.0, alpha)], linewidth=lw_g,
            )
            ax.add_collection(gold)

        f_name = f'output/ex0099_{flavour}_{scale}_{label}.png'
        fig.savefig(f_name, dpi=100)
        plt.close(fig)
        print(f'  saved {f_name}')


if __name__ == '__main__':
    _reg = Registrar()
    EOX_WMTS  = 'https://tiles.maps.eox.at/wmts/1.0.0/WMTSCapabilities.xml'
    ESRI_WMTS = ('https://services.arcgisonline.com/arcgis/rest/services/'
                 'World_Imagery/MapServer/WMTS/1.0.0/WMTSCapabilities.xml')

    _s2   = make_tile_fetcher(EOX_WMTS,  's2cloudless-2020', cache_dir='wmts_cache')
    _esri = make_tile_fetcher(ESRI_WMTS, 'World_Imagery',    cache_dir='wmts_cache')

    run(
        reg=_reg,
        flavour='butterfly',
        scale=1000,
        max_layer=9,
        bg_sources={0: _s2, 7: _esri},
        frames_per_level=2,
    )
