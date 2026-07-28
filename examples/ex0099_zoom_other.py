# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Sky zoom: a starfield dive toward the galactic centre, with H9 hex overlays.

Counterpart to ex0098 (Earth imagery zoom).  The background is not sampled
imagery but a single point catalogue — AT-HYG (~2.55M stars) — splatted onto
the frame canvas with a limiting apparent magnitude that deepens as the zoom
advances:

    m_lim(t) = MAG_BASE + MAG_PER_LEVEL * t

Each H9 level shrinks the viewport span ×3, i.e. on-screen solid angle ×9.
Star counts grow ≈×9 per ~2 magnitudes of depth (N(<m) ≈ 10^0.5m), so
MAG_PER_LEVEL ≈ 1.9 keeps the on-screen star density visually constant while
the field genuinely deepens.  One source, one continuous gate — no imagery
switchover seam.  AT-HYG bottoms out near mag 11–12, so the field stops
thickening beyond t ≈ 3; hex subdivision continues regardless.

Star flux is zero-pointed at m_lim (auto-exposure): a star on the limit is
just visible in every frame, and newly-admitted stars fade in over FADE_MAG
so nothing pops.  Colour comes from the B−V colour index via a small
blackbody-ish LUT (blue ↔ white ↔ amber — a luminance + blue/yellow axis).

Sky mapping: RA/dec become lon/lat on the H9 sphere datum,
lon = (ra·15 + 180) % 360 − 180 (the h9_sky convention; view is NOT mirrored,
i.e. this is the "outside the celestial sphere" orientation).  athyg.csv
stores `ra` in HOURS (0–24), like the HYG csv format — hence the ×15.

Data: tools/astro/data/athyg.csv in the libhex9 repo (not shipped here).
First run parses the ~465 MB csv and caches a compact npz per flavour in
output/ex0099/; subsequent runs start in seconds.

AT-HYG catalogue © David Nash · astronexus.com · CC BY-SA.

Last Tested
24 Jul 2026 (new, replaces the Mercator zoom variant)
"""
import csv
import os

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import PolyCollection
from scipy.ndimage import gaussian_filter

from hhg9 import Registrar, Points
from hhg9.algorithms.geometry import inside_convex_polygon_cw
from hhg9.h9 import H9O
from hhg9.h9.binning import hex_reduce, hex_parents
from hhg9.h9.grid import hex_verts_in_noct, poly_net_field
from hhg9.h9.tail import tail_unpack_reversible

# Baade's Window (NGC 6522 field): the classic low-extinction sightline into
# the galactic bulge (A_V ≈ 1.7) — the densest clean Gaia region on the sky,
# so the magnitude gate keeps thickening the field all the way to the G≈20.7
# floor.  (Sgr A* itself sits behind A_V ≈ 25–40 of nuclear dust: the deepest
# frames there would show the dark rift, not the bulge.)
POI_RA, POI_DEC = 270.892, -30.034
POI_LON = (POI_RA + 180.0) % 360.0 - 180.0
POI_LAT = POI_DEC

ATHYG_CSV = '/Users/ben/Documents/Projects/libhex9/tools/astro/data/athyg.csv'
# Optional Gaia DR3 deep cone around the POI (see libhex9 tools/astro/
# fetch_gaia_cone.py) — magnitude-laddered so each band's cone covers the
# viewport at the zoom where the band surfaces.  Merged in when present;
# extends the catalogue floor from t≈3 (AT-HYG) to t≈7.5 (G≈20.7).
GAIA_CSV = '/Users/ben/Documents/Projects/libhex9/tools/astro/data/gaia_baade.csv'
CACHE_DIR = 'output/ex0099'

# ── magnitude gate ────────────────────────────────────────────────────────────
MAG_BASE = 6.5        # naked-eye limit at t=0 (full sky)
MAG_PER_LEVEL = 1.9   # ≈ log10(9)/0.5: holds on-screen star density constant
FADE_MAG = 0.5        # soft knee below m_lim over which new stars fade in

# ── star rendering (TUNE to taste) ────────────────────────────────────────────
PSF_SIGMA = 0.8       # core point-spread, pixels
GLOW_SIGMA = 2.6      # wide halo around bright stars, pixels
GLOW_FRAC = 0.18      # halo weight relative to core
EXPOSURE = 0.16       # tone-map gain: limit-mag star peaks at ~0.05

CREDIT = 'AT-HYG star catalogue · David Nash · astronexus.com · CC BY-SA'
GAIA_CREDIT = ' · Gaia DR3 · ESA/Gaia/DPAC'
FULL_SKY_DEG2 = 4.0 * np.pi * (180.0 / np.pi) ** 2   # ≈ 41 252.96 deg²


# ── catalogue loading ─────────────────────────────────────────────────────────

def _load_athyg_ll(csv_path):
    """Parse athyg.csv → (lat, lon, mag, ci) float32 arrays sorted by mag.

    Cached in CACHE_DIR/athyg_ll.npz (flavour-independent).  `ra` is in
    hours in the AT-HYG format.  Sol and unparseable rows are dropped.
    """
    cache = os.path.join(CACHE_DIR, 'athyg_ll.npz')
    if os.path.exists(cache):
        z = np.load(cache)
        return z['lat'], z['lon'], z['mag'], z['ci']

    print(f'  parsing {csv_path} (one-time) …')
    lats, lons, mags, cis = [], [], [], []
    with open(csv_path, newline='') as f:
        rdr = csv.reader(f)
        hdr = next(rdr)
        i_ra, i_dec = hdr.index('ra'), hdr.index('dec')
        i_mag, i_ci = hdr.index('mag'), hdr.index('ci')
        for n, row in enumerate(rdr):
            try:
                ra, dec, m = float(row[i_ra]), float(row[i_dec]), float(row[i_mag])
            except (ValueError, IndexError):
                continue
            if m < -20.0:            # Sol
                continue
            lats.append(dec)
            lons.append((ra * 15.0 + 180.0) % 360.0 - 180.0)  # ra is in hours
            mags.append(m)
            try:
                cis.append(float(row[i_ci]))
            except (ValueError, IndexError):
                cis.append(np.nan)
            if (n + 1) % 500_000 == 0:
                print(f'    … {n + 1:,} rows')

    lat = np.asarray(lats, dtype=np.float32)
    lon = np.asarray(lons, dtype=np.float32)
    mag = np.asarray(mags, dtype=np.float32)
    ci = np.asarray(cis, dtype=np.float32)
    order = np.argsort(mag, kind='stable')
    lat, lon, mag, ci = lat[order], lon[order], mag[order], ci[order]
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(cache, lat=lat, lon=lon, mag=mag, ci=ci)
    print(f'  cached {len(mag):,} stars → {cache}')
    return lat, lon, mag, ci


def _load_gaia_ll(csv_path):
    """gaia_sgr_a.csv (ra,dec,g,bp_rp — ra in DEGREES, unlike AT-HYG's hours)
    → (lat, lon, mag, ci) float32 arrays.

    ci ≈ 0.75·bp_rp − 0.05 roughly maps Gaia BP−RP onto the B−V colour LUT;
    missing bp_rp stays NaN (neutral colour downstream).
    """
    data = np.genfromtxt(csv_path, delimiter=',', names=True)
    lat = data['dec'].astype(np.float32)
    lon = (((data['ra'] + 180.0) % 360.0) - 180.0).astype(np.float32)
    mag = data['g'].astype(np.float32)
    ci = (0.75 * data['bp_rp'] - 0.05).astype(np.float32)
    print(f'  gaia cone: {len(mag):,} stars from {csv_path}')
    return lat, lon, mag, ci


def _stars_in_noct(reg, n_oct, b_oct, g_gcd, flavour, csv_path):
    """Project the whole catalogue onto the net once.

    Returns (xy_n (N,2) float32, mag (N,), ci (N,)), sorted by mag so the
    per-frame magnitude gate is a prefix slice (np.searchsorted).  A Gaia POI
    cone (GAIA_CSV) is merged in when present — same arrays, same gate.
    Cached per flavour in CACHE_DIR/<tag>_noct_<flv>.npz.
    """
    gaia = os.path.exists(GAIA_CSV)
    flv = flavour.replace(':', '_')
    # Cache name tracks the Gaia csv so a re-targeted cone rebuilds cleanly.
    tag = ('athyg_' + os.path.splitext(os.path.basename(GAIA_CSV))[0]
           if gaia else 'athyg')
    cache = os.path.join(CACHE_DIR, f'{tag}_noct_{flv}.npz')
    if os.path.exists(cache):
        z = np.load(cache)
        return z['xy'], z['mag'], z['ci']

    lat, lon, mag, ci = _load_athyg_ll(csv_path)
    if gaia:
        g_lat, g_lon, g_mag, g_ci = _load_gaia_ll(GAIA_CSV)
        lat = np.concatenate([lat, g_lat])
        lon = np.concatenate([lon, g_lon])
        mag = np.concatenate([mag, g_mag])
        ci = np.concatenate([ci, g_ci])
        order = np.argsort(mag, kind='stable')
        lat, lon, mag, ci = lat[order], lon[order], mag[order], ci[order]
    print(f'  projecting {len(mag):,} stars → n_oct ({flavour}) …')
    stars_g = Points(np.stack([lat, lon], axis=1).astype(float), g_gcd)
    stars_b = reg.project(stars_g, [g_gcd, b_oct])
    ok = b_oct.valid(stars_b)
    stars_n = reg.project(stars_b.select(ok), [b_oct, n_oct])
    xy = stars_n.coords.astype(np.float32)
    mag, ci = mag[ok], ci[ok]
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(cache, xy=xy, mag=mag, ci=ci)
    print(f'  cached → {cache}')
    return xy, mag, ci


# ── star appearance ───────────────────────────────────────────────────────────

# B−V → linear-ish RGB anchors (blue ↔ white ↔ amber; luminance + blue/yellow).
_BV = np.array([-0.40, 0.00, 0.40, 0.80, 1.50, 2.50])
_BV_R = np.array([0.61, 0.79, 1.00, 1.00, 1.00, 1.00])
_BV_G = np.array([0.70, 0.85, 0.98, 0.92, 0.80, 0.64])
_BV_B = np.array([1.00, 1.00, 0.94, 0.83, 0.62, 0.34])


def _bv_rgb(ci):
    """(N,) B−V colour index → (N, 3) RGB.  NaN ci → neutral 0.5."""
    bv = np.where(np.isfinite(ci), ci, 0.5)
    return np.stack([np.interp(bv, _BV, _BV_R),
                     np.interp(bv, _BV, _BV_G),
                     np.interp(bv, _BV, _BV_B)], axis=1)


def _splat_starfield(xy, mag, rgb, vp, img_w, img_h, m_lim):
    """Render the starfield for one frame.

    Additive flux deposition (bincount) of every star with mag ≤ m_lim inside
    the viewport, zero-pointed at m_lim (auto-exposure), Gaussian core + wide
    halo PSF, then 1−exp tone-map (bright stars saturate to white).

    Returns (canvas (img_h, img_w, 4) float32, n_shown).
    """
    n_gate = int(np.searchsorted(mag, m_lim, side='right'))  # mag is sorted
    x, y, m = xy[:n_gate, 0], xy[:n_gate, 1], mag[:n_gate]
    sx = (img_w - 1) / (vp[1] - vp[0])
    sy = (img_h - 1) / (vp[3] - vp[2])
    px = np.rint((x - vp[0]) * sx).astype(np.int64)
    py = (img_h - 1) - np.rint((y - vp[2]) * sy).astype(np.int64)
    in_vp = (px >= 0) & (px < img_w) & (py >= 0) & (py < img_h)
    px, py, m = px[in_vp], py[in_vp], m[in_vp]
    col = rgb[:n_gate][in_vp]

    # Flux relative to the limit; soft fade-in over the last FADE_MAG.
    flux = 10.0 ** (0.4 * (m_lim - m))
    flux *= np.clip((m_lim - m) / FADE_MAG, 0.0, 1.0)

    idx = py * img_w + px
    canvas = np.zeros((img_h, img_w, 3), dtype=np.float32)
    flat = canvas.reshape(-1, 3)
    for ch in range(3):
        flat[:, ch] = np.bincount(idx, weights=flux * col[:, ch],
                                  minlength=img_h * img_w).astype(np.float32)
    for ch in range(3):
        core = gaussian_filter(canvas[:, :, ch], PSF_SIGMA)
        glow = gaussian_filter(canvas[:, :, ch], GLOW_SIGMA)
        canvas[:, :, ch] = core + GLOW_FRAC * glow
    canvas = 1.0 - np.exp(-EXPOSURE * canvas)

    out = np.ones((img_h, img_w, 4), dtype=np.float32)
    out[:, :, :3] = canvas
    return out, int(in_vp.sum())


# ── net / hex helpers (shared pattern with ex0098) ────────────────────────────

def _hex_at_layer(poi_n, layer, n_oct, b_oct, reg):
    """Centroid (n_oct) and polygon (1, 6, 2) for the hex at `layer` that
    geometrically contains poi_n.

    Ownership is picked by drawn containment over the local lattice, NOT by
    hex_reduce point-binning of the POI: at L0 that path ties hexagon
    centroids to uuids and does not return the owning cell (h9_bin_pts does).
    Containment keeps the highlight consistent with the overlay grid/labels.
    """
    d = 1.4 / (3.0 ** layer)
    box = Points(np.array([
        [poi_n[0] - d, poi_n[1] - d], [poi_n[0] + d, poi_n[1] - d],
        [poi_n[0] + d, poi_n[1] + d], [poi_n[0] - d, poi_n[1] + d],
    ]), n_oct)
    lattice_n = poly_net_field(box, layer)
    b_lattice, _ = _project_noct_pts(lattice_n.coords, n_oct, b_oct, reg)
    h_num, h_v, _, _ = hex_reduce(b_lattice, layer)
    h_par, h_oid, h_scale = hex_parents(b_oct, h_v, h_num)
    xc2, _, xpm = tail_unpack_reversible(h_v[:, -1])  # (c2, r_mo, p_mo)
    verts = hex_verts_in_noct(h_par, h_oid, xpm, xc2, h_scale,
                              n_oct).coords.reshape(-1, 6, 2)
    ctrs = verts.mean(axis=1)
    inside = np.array([inside_convex_polygon_cw(poi_n[None], p, tol=1e-9)[0]
                       for p in verts])
    if inside.any():
        k = int(np.argmax(inside))
    else:  # seam-adjacent fallback: nearest drawn centroid
        k = int(np.argmin(((ctrs - poi_n) ** 2).sum(axis=1)))
    return ctrs[k], verts[k:k + 1]


def _project_noct_pts(coords_n, n_oct, b_oct, reg, tol=0.0):
    """Assign face oids to arbitrary n_oct coordinates and project to b_oct.
    Returns (b_pts, idx) — idx maps outputs back to rows of coords_n."""
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


def _net_bounds(n_oct):
    """(xmin, xmax, ymin, ymax) over every face polygon of the net."""
    vs = np.concatenate([p for polys in n_oct.face_polys.values() for p in polys])
    return (float(vs[:, 0].min()), float(vs[:, 0].max()),
            float(vs[:, 1].min()), float(vs[:, 1].max()))


def _tier_lw(d):
    """Grid line-width as a continuous function of depth d = layer − t."""
    return float(np.interp(d, [0.0, 1.0, 2.0], [2.0, 1.0, 0.5]))


def _tier_alpha(d):
    """Grid edge alpha as a continuous function of depth d = layer − t."""
    if d >= 2.0 or d <= -1.0:
        return 0.0
    if d >= 1.55:
        return 0.5 * (2.0 - d) / 0.45
    if d <= -0.7:
        return 0.5 * (d + 1.0) / 0.3
    return 0.5


def _common_prefix(strs):
    """Longest common leading substring across strs."""
    if not strs:
        return ''
    lo, hi = min(strs), max(strs)
    for i, c in enumerate(lo):
        if c != hi[i]:
            return lo[:i]
    return lo


def fmt_solid_angle(layer, sig=4):
    """Per-cell solid angle at `layer`, in the nearest well-known sky unit."""
    deg2 = FULL_SKY_DEG2 / (12.0 * 9.0 ** layer)
    units = [('deg²', 1.0), ('arcmin²', 1.0 / 3600.0),
             ('arcsec²', 1.0 / 3600.0 ** 2), ('mas²', 1.0 / 3600.0 ** 2 / 1e6)]
    for sym, f in units:
        if deg2 >= f:
            return f'{deg2 / f:,.{sig}g} {sym}'
    sym, f = units[-1]
    return f'{deg2 / f:.{sig}g} {sym}'


def fmt_count(l: int):
    """12×9²² Cells"""
    sup = str(int(l)).translate(str.maketrans('0123456789-', '⁰¹²³⁴⁵⁶⁷⁸⁹⁻'))
    return f'12×9{sup}'


# ── main ─────────────────────────────────────────────────────────────────────

def run(*, reg=None, flavour='butterfly:0500', max_layer=4, csv_path=ATHYG_CSV,
        frames_per_level: int = 0, frame_interval: int = 1, max_frame=None,
        start_frame: int = 0, end_frame=None, skip_existing: bool = False):
    """Generate sky-zoom PNGs (frame indexing as in ex0098).

    Args:
        reg:              Registrar (a new one is created if None).
        flavour:          Net layout name.
        max_layer:        Highest zoom level (used when max_frame is not given).
        csv_path:         AT-HYG csv (only read on the first, cache-building run).
        frames_per_level: 0 → one PNG per integer level; N → frame i ↔ t = i/N.
        frame_interval:   Animate render stride (files are named by index).
        max_frame:        Animate: highest index.  Defaults to max_layer*N.
        start_frame:      Skip indices < this (resume window start).
        end_frame:        Stop after this index (inclusive); None → max_frame.
        skip_existing:    Skip frames whose PNG already exists on disk.
    """
    if reg is None:
        reg = Registrar()
    n_oct = reg.domain(f'n_oct:{flavour}')
    b_oct = reg.domain('b_oct')
    g_gcd = reg.domain('g_gcd')

    animate = frames_per_level > 0
    if animate:
        fpl = frames_per_level
        if max_frame is None:
            max_frame = max_layer * fpl
        max_layer = -(-max_frame // fpl)  # ceil

    img_w, img_h = 2560, 1440
    aspect = img_w / img_h

    # Star catalogue, projected once onto the net (mag-sorted).
    star_xy, star_mag, star_ci = _stars_in_noct(reg, n_oct, b_oct, g_gcd,
                                                flavour, csv_path)
    star_rgb = _bv_rgb(star_ci)

    # POI in b_oct
    poi_g = Points(np.array([[POI_LAT, POI_LON]]), g_gcd)
    poi_b = reg.project(poi_g, [g_gcd, b_oct])

    net_xmin, net_xmax, net_ymin, net_ymax = _net_bounds(n_oct)
    net_span_y = net_ymax - net_ymin

    # Viewport centres per integer level (as ex0098, locked to the POI).
    centres = [np.array([(net_xmin + net_xmax) / 2.0, (net_ymin + net_ymax) / 2.0])]
    poi_n = reg.project(poi_b, [b_oct, n_oct])
    poi_centre = poi_n.coords[0].copy()
    for _L in range(1, max_layer + 1):
        centres.append(poi_centre)

    def _viewport_t(t: float) -> list:
        """Continuous viewport at fractional zoom level t (ex0098 easing)."""
        L_lo = min(int(t), max_layer - 1)
        L_hi = L_lo + 1
        alpha = t - L_lo
        w = alpha * (3.0 ** (1.0 - alpha))   # uniform screen-space pan
        cx, cy = (1.0 - w) * centres[L_lo] + w * centres[L_hi]
        span_y = net_span_y / (3.0 ** t)
        span_x = span_y * aspect
        return [cx - span_x / 2, cx + span_x / 2, cy - span_y / 2, cy + span_y / 2]

    if not animate:
        frame_seq = [(L, float(L), f'L{L:02d}') for L in range(max_layer + 1)]
    else:
        idx_w = max(4, len(str(max_frame)))
        frame_seq = [(i, i / fpl, f'frame_{i:0{idx_w}d}')
                     for i in range(0, max_frame + 1, frame_interval)]

    flv = flavour.replace(':', '_')
    os.makedirs(CACHE_DIR, exist_ok=True)
    for frame_idx, t, label in frame_seq:
        if frame_idx < start_frame or (end_frame is not None and frame_idx > end_frame):
            continue
        L = min(int(t), max_layer)
        f_name = f'{CACHE_DIR}/{flv}_sky_{label}.png'
        if skip_existing and os.path.exists(f_name):
            print(f'  {label}  skip (exists)')
            continue
        m_lim = MAG_BASE + MAG_PER_LEVEL * t
        print(f'  {label}  (t={t:.3f}  L={L}  m≤{m_lim:.2f})')

        vp = _viewport_t(t)
        overlay_layers = [L, L + 1, L + 2] if (animate or L > 0) else [0, 1, 2]

        out, n_shown = _splat_starfield(star_xy, star_mag, star_rgb,
                                        vp, img_w, img_h, m_lim)

        fig = plt.figure(figsize=(img_w / 100, img_h / 100), dpi=100,
                         frameon=False, facecolor='black')
        fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        ax.set_facecolor('black')
        ax.set_xlim(vp[0], vp[1])
        ax.set_ylim(vp[2], vp[3])
        ax.imshow(out, extent=[vp[0], vp[1], vp[2], vp[3]],
                  alpha=1.0, origin='upper', aspect='auto')

        frac = t - int(t)
        ramp_alpha = 0.5 if not animate else min(frac / 0.9, 0.5)

        # Hex outlines — coarse (thick) → fine (thin), same tiering as ex0098.
        vp_pts = Points(np.array([
            [vp[0], vp[2]], [vp[1], vp[2]],
            [vp[1], vp[3]], [vp[0], vp[3]],
        ]), n_oct)
        lw = [2.0, 1.0, 0.5]
        coarse_hexes, coarse_v_k = None, None
        mid_hexes, mid_v_k = None, None
        for li, layer in enumerate(overlay_layers):
            lattice_n = poly_net_field(vp_pts, layer)
            b_lattice, _ = _project_noct_pts(lattice_n.coords, n_oct, b_oct, reg)
            if b_lattice is None:
                continue
            hex_num, hex_v_k, _, _ = hex_reduce(b_lattice, layer)
            if hex_v_k is None or len(hex_v_k) == 0:
                continue
            hex_par, hex_oid, hex_scale = hex_parents(b_oct, hex_v_k, hex_num)
            xc2, _, xpm = tail_unpack_reversible(hex_v_k[:, -1])
            verts_n = hex_verts_in_noct(hex_par, hex_oid, xpm, xc2, hex_scale, n_oct)
            hexes = verts_n.coords.reshape(-1, 6, 2)
            depth = layer - t
            ec_a = _tier_alpha(depth) if animate else 0.5
            lwv = _tier_lw(depth) if animate else lw[li]
            ax.add_collection(PolyCollection(
                hexes, facecolors='none', ec=(1, 1, 1, ec_a),
                linewidth=lwv, antialiaseds=True,
            ))
            if li == 0:
                coarse_hexes, coarse_v_k = hexes, hex_v_k
            elif li == 1:
                mid_hexes, mid_v_k = hexes, hex_v_k

        # Address labels with parent/child crossfade (as ex0098), but gated by
        # a radial screen-space falloff instead of ex0098's hard count caps.
        # The in-view count jumps ×9 the instant the coarse role ticks over a
        # level, so any count cap pops every label at once on nets that hold
        # more hexes per frame (butterfly ~16 never tripped it; windmill ~65
        # does).  Distance from the viewport centre is continuous through the
        # tick, and it labels the region the dive is actually about.
        def _label_vis(pos):
            rx = (pos[:, 0] - 0.5 * (vp[0] + vp[1])) / (0.5 * (vp[1] - vp[0]))
            ry = (pos[:, 1] - 0.5 * (vp[2] + vp[3])) / (0.5 * (vp[3] - vp[2]))
            return np.clip((1.05 - np.hypot(rx, ry)) / 0.35, 0.0, 1.0)

        def _draw_layer_labels(hexes, v_k, alpha, prefix):
            if hexes is None or alpha <= 0.0:
                return
            ctrs = np.mean(hexes, axis=1)
            bl_idx = np.argmin(hexes[:, :, 0] + hexes[:, :, 1], axis=1)
            bl_verts = hexes[np.arange(len(hexes)), bl_idx]
            label_pos = bl_verts + 0.03 * (ctrs - bl_verts)
            on_net = np.zeros(len(label_pos), dtype=bool)
            for _polys in n_oct.face_polys.values():
                for _poly in _polys:
                    on_net |= inside_convex_polygon_cw(label_pos, _poly, tol=0.0)
            vis = _label_vis(label_pos)
            strs = [''.join(f'{int(d):01x}' for d in row[:-1]) for row in v_k]
            for (lx, ly), lbl, on, vi in zip(label_pos, strs, on_net, vis):
                a = alpha * vi
                if not on or a <= 0.02:
                    continue
                # Only strip the prefix where it actually applies — edge labels
                # outside the central cluster may not share it.
                txt = lbl[len(prefix):] if prefix and lbl.startswith(prefix) else lbl
                ax.text(lx, ly, txt,
                        fontsize=12, ha='left', va='bottom',
                        color=(1.0, 1.0, 1.0, a),
                        path_effects=[pe.withStroke(
                            linewidth=1.2, foreground=(0, 0, 0, 0.5 * a))],
                        zorder=100, clip_on=True)

        hex_text = (f'Hex9 // Sky // L{L}: {fmt_count(L)} cells; '
                    f'{fmt_solid_angle(L)}/cell // m≤{m_lim:.1f} · {n_shown:,} stars')
        if coarse_hexes is not None:
            # Prefix from the central (labelled) cells, so strip + caption
            # describe the dive region rather than the whole frame.
            cen = _label_vis(np.mean(coarse_hexes, axis=1)) > 0.3
            sel = coarse_v_k[cen] if np.any(cen) else coarse_v_k
            lab_str = [''.join(f'{int(d):01x}' for d in row[:-1]) for row in sel]
            prefix = _common_prefix(lab_str)
            parent_a = min((1.0 - frac) / 0.3, 1.0) if animate else 1.0
            child_a = max(0.0, min((frac - 0.55) / 0.45, 1.0)) if animate else 0.0
            _draw_layer_labels(coarse_hexes, coarse_v_k, parent_a, prefix)
            _draw_layer_labels(mid_hexes, mid_v_k, child_a, prefix)
            if prefix:
                hex_text += f' // H9 Prefix: {prefix}'

        tablet = dict(boxstyle='round,pad=0.35', fc=(0, 0, 0, 0.45), ec='none')
        soft_halo = [pe.withStroke(linewidth=1.0, foreground=(0, 0, 0, 0.5))]
        ax.text(0.012, 0.012, hex_text,
                transform=ax.transAxes,
                fontsize=16, ha='left', va='bottom', family='monospace',
                color=(1.0, 0.85, 0.0, 1.0),
                path_effects=soft_halo, bbox=tablet,
                zorder=150, clip_on=False)

        # Gold highlight: hexes containing the POI at L, L+1, L+2.
        _, cur_poly = _hex_at_layer(poi_centre, L, n_oct, b_oct, reg)
        ax.add_collection(PolyCollection(
            cur_poly, facecolors='none',
            edgecolors=[(1.0, 0.85, 0.0, 1.0)], linewidth=2.0,
        ))
        _, gold_poly = _hex_at_layer(poi_centre, L + 1, n_oct, b_oct, reg)
        ax.add_collection(PolyCollection(
            gold_poly, facecolors='none',
            edgecolors=[(1.0, 0.85, 0.0, 1.0)], linewidth=1.5,
        ))
        _, gold_poly = _hex_at_layer(poi_centre, L + 2, n_oct, b_oct, reg)
        ax.add_collection(PolyCollection(
            gold_poly, facecolors='none',
            edgecolors=[(1.0, 0.85, 0.0, ramp_alpha)], linewidth=1.0,
        ))

        # Static catalogue credit (no per-frame source switching to scroll).
        credit = CREDIT + (GAIA_CREDIT if os.path.exists(GAIA_CSV) else '')
        ax.text(0.988, 0.010, credit,
                transform=ax.transAxes, fontsize=11, ha='right', va='bottom',
                color=(1.0, 1.0, 1.0, 0.9),
                path_effects=[pe.withStroke(linewidth=1.0, foreground=(0, 0, 0, 0.55))],
                bbox=dict(boxstyle='round,pad=0.3', fc=(0, 0, 0, 0.45), ec='none'),
                zorder=150, clip_on=False)

        fig.savefig(f_name, dpi=100, facecolor='black')
        plt.close(fig)
        print(f'  saved {f_name}')


if __name__ == '__main__':
    run(
        flavour='windmill_pacific',
        # flavour='butterfly:0500',
        frames_per_level=50,     # frame i ↔ zoom t = i/50
        max_frame=380,           # 4 levels · 50; with the Gaia cone ~375 (t=7.5 floor)
        frame_interval=1,
        skip_existing=True,
        # start_frame=0, end_frame=0,   # single-frame test window
    )
