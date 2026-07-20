# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Zoom sequence centred on a lat/lon point of interest.

Frame L shows:
  - Background pixel map zoomed to a 1/3^L sub-region of the full net.
  - Hex overlays: layers L, L+1, L+2 (uniformly, L=0 included — the 12 base
      cells are drawn and labelled at L0, just like any other level).
  - Middle-layer hex (L+1, for L>0) highlighted in gold.
  - Dashed yellow rectangle indicating the next frame's viewport (debug aid).

Background sources are swappable per-frame via bg_sources.  The builders live
in `hhg9.rendering.imagery`; each returns the standard sampler
(b_pts: Points) -> (N, C), so one source serves a frame background here, a hex
fill (LayerSpec.source), or a pixel backdrop (make_backdrop) interchangeably:
  make_pc_source   — PlateCarrée KDTree (the original approach, good for L=0)
  make_bm_source   — GDAL VRT sampler (Blue Marble or similar, for L>=1)
  make_xyz_source  — slippy XYZ tiles (Web Mercator or WGS84)
  make_wmts_source — WMTS, tile matrix set auto-detected from capabilities
  make_composite_source — alpha-composite a partial mosaic over a base

Sources carry their imagery credit (`.attribution`, or `.credits` for a
composite stack).  `_draw_credits` below reads those directly, since it also
scrolls the stack as sources change between frames; a static render would use
`hhg9.rendering.imagery.credits_of` and `plot_hex(credits=...)` instead.

Note: make_geotiff_sampler reads full raster bands into memory.  Use the
10800×10800 Blue Marble variant to keep RAM below ~400 MB.
Build a VRT first:
  gdalbuildvrt src/blue_marble.vrt src/world.200408.3x10800x10800.A*.tif

use geotag_image.py for adding in signs, butterflies, etc.
that will honour transparency, and then can use it to composite via
make_composite_source


Last Tested
16 Jul 2026 0.1.3a0 (passed)
28 Mar 2026 (new)
"""
import os

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import PolyCollection
from osgeo import gdal
from scipy.ndimage import distance_transform_edt

from hhg9 import Registrar, Points
from hhg9.algorithms.geometry import inside_convex_polygon_cw
from hhg9.h9 import H9O, H9P
from hhg9.h9.binning import hex_reduce, hex_parents
from hhg9.h9.grid import qa_grid, hex_verts_in_noct, poly_net_field, hex_props
from hhg9.h9.tail import tail_unpack_reversible
from hhg9.rendering.imagery import (make_pc_source, make_xyz_source,
                                    make_composite_source, make_wmts_source,
                                    make_bm_source)

POI_LON, POI_LAT = 1.640249982, 49.0987954416  # Butterfly demo.

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
    xc2, _, xpm = tail_unpack_reversible(h_v[:, -1])  # (c2, r_mo, p_mo)
    verts_n = hex_verts_in_noct(h_par, h_oid, xpm, xc2, h_scale, n_oct)
    ctr_b = Points(h_par, b_oct, oid=h_oid)
    ctr_n = reg.project(ctr_b, [b_oct, n_oct])
    return ctr_n.coords[0], verts_n.coords.reshape(-1, 6, 2)


def _viewport(L, centre_n, net_xmin, net_xmax, net_ymin, net_ymax, aspect):
    """Viewport [xmin, xmax, ymin, ymax] at zoom level L, with the given aspect.

    L=0 returns the full net height framed at `aspect` (pillarbox).  L>0 shrinks
    by 3^L in height and derives width from `aspect`, centred on centre_n.  The
    window aspect always matches the output canvas so the render never stretches.
    """
    if L == 0:
        span_y = net_ymax - net_ymin
        cx, cy = (net_xmin + net_xmax) / 2, (net_ymin + net_ymax) / 2
    else:
        span_y = (net_ymax - net_ymin) / (3.0 ** L)
        cx, cy = centre_n
    span_x = span_y * aspect
    return [cx - span_x / 2, cx + span_x / 2, cy - span_y / 2, cy + span_y / 2]


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
    yl = np.linspace(vp[3], vp[2], img_h)  # row 0 = top = high y
    xx, yy = np.meshgrid(xl, yl)
    coords = np.stack([xx.ravel(), yy.ravel()], axis=1)
    px_all = np.tile(np.arange(img_w), img_h)
    py_all = np.repeat(np.arange(img_h), img_w)
    tol = 0.5 * (vp[1] - vp[0]) / img_w  # half-pixel seam tolerance
    b_pts, idx = _project_noct_pts(coords, n_oct, b_oct, reg, tol=tol)
    return px_all[idx], py_all[idx], b_pts


def _pick_source(bg_sources, key_at):
    """Select the source callable active at `key_at` (frame index, or layer L in
    the legacy per-level path).

    bg_sources may be:
      - a single callable  → used for all frames
      - a dict {int: callable} → source with the greatest key ≤ key_at; falls
        back to the min key below it.
    """
    if callable(bg_sources):
        return bg_sources
    candidates = [k for k in bg_sources if k <= key_at]
    key = max(candidates) if candidates else min(bg_sources)
    return bg_sources[key]


def _draw_credits(ax, credit_list, frame_idx, bg_sources, fpl, animate):
    """Bottom-right imagery-credit stack: newest (deepest subject) at the bottom,
    at most 3 lines visible.

    When a new source composites in (a bg_sources key is crossed) the stack scrolls
    up one slot over ~fpl/5 indices — the new line rises into the bottom slot and
    the oldest clips off the top.  It is a pure function of frame_idx (no
    cross-frame state), so it is correct at any render stride or resume point.
    """
    credits = [c for c in credit_list if c]
    if not credits:
        return
    x, base_y, line_h, max_show = 0.988, 0.010, 0.018, 3

    delta, s = 0, 1.0
    if animate and isinstance(bg_sources, dict):
        keys = sorted(bg_sources)
        active = [k for k in keys if k <= frame_idx]
        entered_at = active[-1] if active else keys[0]
        below = [k for k in keys if k < entered_at]
        if below:                              # how many new credits this source adds
            prev = bg_sources[below[-1]]
            prev_credits = [c for c in (getattr(prev, 'credits', None)
                                        or [getattr(prev, 'attribution', '')]) if c]
            delta = max(0, len(credits) - len(prev_credits))
        s = min(1.0, (frame_idx - entered_at) / max(1, fpl // 5))

    for j in range(min(len(credits), max_show + delta)):
        slot = j - delta * (1.0 - s)           # 0 = bottom; slides up as s→1
        if slot < -1.0 or slot > max_show:
            continue
        vis = max(0.0, min(1.0, min(slot + 1.0, max_show - slot)))  # fade at both ends
        if vis <= 0.02:
            continue
        ax.text(x, base_y + slot * line_h, credits[j],
                transform=ax.transAxes, fontsize=11, ha='right', va='bottom',
                color=(1.0, 1.0, 1.0, 0.9 * vis),
                path_effects=[pe.withStroke(linewidth=1.0, foreground=(0, 0, 0, 0.55 * vis))],
                bbox=dict(boxstyle='round,pad=0.3', fc=(0, 0, 0, 0.45 * vis), ec='none'),
                zorder=150, clip_on=False)


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


def _tier_lw(d):
    """Grid line-width as a continuous function of depth d = layer - t.

    Coarse tiers (d≤0) are thick, fine tiers (d≥2) thin.  Because d moves smoothly
    with t, a tier no longer snaps wider when the integer level ticks over.
    """
    return float(np.interp(d, [0.0, 1.0, 2.0], [2.0, 1.0, 0.5]))


def _tier_alpha(d):
    """Grid edge alpha as a continuous function of depth d = layer - t.

    A newly-appearing fine tier fades in near d=2; an outgoing coarse tier fades
    out near d=-1; everything between holds at 0.5.  This is the old per-tier ramp
    re-expressed in d, so alpha AND width line up across every level boundary.
    """
    if d >= 2.0 or d <= -1.0:
        return 0.0
    if d >= 1.55:            # fine tier fading in  (frac 0 → 0.45)
        return 0.5 * (2.0 - d) / 0.45
    if d <= -0.7:            # coarse tier fading out (frac 0.7 → 1.0)
        return 0.5 * (d + 1.0) / 0.3
    return 0.5


# ── source builders ───────────────────────────────────────────────────────────

# Imagery source factories now live in hhg9.rendering.imagery — they return the
# standard (b_pts: Points) -> (N, C) sampler, so the same source drives a hex
# fill (LayerSpec.source) or a pixel backdrop (make_backdrop).


def fmt_area(area_m2, sig=4):
    """Format an area (given in m²) with the nearest well-known unit."""
    m2 = area_m2
    # (symbol, m² per unit) — largest → smallest
    units = [
        ('M km²', 1e12),  # million km²
        ('k km²', 1e9),   # thousand km²
        ('km²', 1e6),
        ('ha', 1e4),  # hectare
        ('m²', 1.0),
        ('cm²', 1e-4),
        ('mm²', 1e-6),
        ('k µm²', 1e-9),   # thousand µm²
        ('µm²', 1e-12),
        ('k nm²', 1e-15),  # thousand nm²
        ('nm²', 1e-18),
    ]

    for sym, f in units:
        if m2 >= f:
            return f'{m2 / f:,.{sig}g}{sym}'

    sym, f = units[-1]
    return f'{m2 / f:.{sig}g}{sym}'  # below smallest unit


def fmt_count(l: int):
    """12×9²² Cells"""
    sup = str(int(l)).translate(str.maketrans('0123456789-', '⁰¹²³⁴⁵⁶⁷⁸⁹⁻'))
    return f'12×9{sup}'


# ── main ─────────────────────────────────────────────────────────────────────

def run(*, reg=None, flavour='rhombus', scale=1201, max_layer=4, bg_sources,
        frames_per_level: int = 0, frame_interval: int = 1, max_frame=None,
        centre: str = 'poi', start_frame: int = 0, end_frame=None,
        skip_existing: bool = False):
    """Generate zoom-sequence PNGs.

    Frame indexing (animate, frames_per_level > 0) is decoupled from generation:
    index ``i`` always denotes zoom ``t = i / frames_per_level`` and the output
    file is named by that index (``frame_NNNNNN.png``), so a given index is the
    same viewpoint no matter the stride.  `frame_interval` is the render stride —
    render every Nth index (1 = every frame; = frames_per_level = one still per
    level).  `bg_sources` keys live in this same index space.

    Args:
        reg:              Registrar to use.  Pass the same instance used to build
                          bg_sources so domain/projection state is shared.  A new
                          Registrar is created if None.
        flavour:          Net layout name.
        scale:            Pixel density (triangle side in pixels) for the full grid.
        max_layer:        Highest zoom level (used when max_frame is not given).
        bg_sources:       Background colour source(s).  A single callable, or a dict
                          {index: callable} — the source with the greatest key ≤ the
                          current frame index is used (in the legacy per-level path
                          the key is the layer L).  A source that returns (N, 4) must
                          be an opaque composite (make_composite_source); a raw
                          with_coverage source belongs *inside* a composite, never here.
        frames_per_level: 0 → legacy: one PNG per integer level (static, full alpha).
                          N → animate: index resolution.  frame i ↔ t = i/N.
        frame_interval:   Animate render stride (default 1).  Files are named by index
                          (multiples of the stride), so the same index → same file.
        max_frame:        Animate: highest index to render.  Defaults to max_layer*N.
        centre:           'poi' (default) — lock viewport centre to the POI from L=1;
                          'hex' — drift between hex centroids each level (Ken Burns).
        start_frame:      Skip indices < this (resume window start).
        end_frame:        Stop after this index (inclusive); None → max_frame.
        skip_existing:    Skip any frame whose output PNG already exists on disk.
    """
    if reg is None:
        reg = Registrar()
    n_oct = reg.domain(f'n_oct:{flavour}')
    b_oct = reg.domain('b_oct')
    g_gcd = reg.domain('g_gcd')

    # Frame index space (animate): index i ↔ t = i/fpl; max_frame is the top index.
    # Derive max_layer from it so the centres precompute covers the deepest level.
    animate = frames_per_level > 0
    if animate:
        fpl = frames_per_level
        if max_frame is None:
            max_frame = max_layer * fpl
        max_layer = -(-max_frame // fpl)  # ceil

    # Output canvas is a fixed 16:9 frame, decoupled from `scale` (which now only
    # sets the L=0 net-grid density via qa_grid).  The viewport is anchored on
    # height at the same aspect (see _viewport/_viewport_t), so the render never
    # stretches.  Keep `scale` high enough that image_dims(scale).W >~ img_w or the
    # L=0 subsample gets gappy.
    img_w, img_h = 2560, 1440
    aspect = img_w / img_h

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
            placed_mode = prj.c2trans[c2][0] if prj.c2trans is not None else -1
            hhp = H9P.hh[mo, c2]
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
    net_span_y = net_ymax - net_ymin  # width is derived from `aspect`, not the net

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
        # Front-load the centre lerp so screen-space pan is uniform: the span
        # shrinks as 3^-t, so a linear-in-alpha centre would surge ~6x near each
        # level boundary (a throb). w = alpha*3^(1-alpha) makes screen motion ∝ alpha.
        w = alpha * (3.0 ** (1.0 - alpha))
        cx, cy = (1.0 - w) * centres[L_lo] + w * centres[L_hi]
        # Anchor the window on height and derive width from the canvas aspect, so
        # the viewport rectangle matches img_w/img_h and the render never stretches.
        span_y = net_span_y / (3.0 ** t)
        span_x = span_y * aspect
        # No clamp to net bounds: the 16:9 window is wider than the butterfly, so
        # clamping x would pin it to the net width (breaking aspect / isotropy) for
        # the first frames.  Off-net pixels sample nothing and stay transparent.
        return [cx - span_x / 2, cx + span_x / 2, cy - span_y / 2, cy + span_y / 2]

    # Build the sequence of (frame_index, t, label) tuples.
    # Legacy (fpl=0): one entry per integer level, labelled/keyed by L.
    # Animate (fpl>0): index i ↔ t = i/fpl, stepped by frame_interval; the file is
    #   named by the index, so it denotes the same viewpoint at any stride.
    if not animate:
        frame_seq = [(L, float(L), f'L{L:02d}') for L in range(max_layer + 1)]
    else:
        idx_w = max(4, len(str(max_frame)))
        frame_seq = [(i, i / fpl, f'frame_{i:0{idx_w}d}')
                     for i in range(0, max_frame + 1, frame_interval)]

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
        src_fn = _pick_source(bg_sources, frame_idx)  # key space = index (animate) / L (legacy)
        hp = hex_props(L)
        area_m2 = hp[0]
        area_str = fmt_area(area_m2)
        count = fmt_count(L)

        if animate or L > 0:
            vp = _viewport_t(t) if animate else _viewport(L, centres[L], net_xmin, net_xmax, net_ymin, net_ymax, aspect)
            overlay_layers = [L, L + 1, L + 2]
        else:
            vp = _viewport(0, None, net_xmin, net_xmax, net_ymin, net_ymax, aspect)
            overlay_layers = [0, 1, 2]

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

        # Hex outlines — coarse (thick) → fine (thin).  poly_net_field on the
        # viewport rectangle guarantees every hex in view at every level (L=0
        # included), so grid geometry + numbering are consistent across frames.
        vp_pts = Points(np.array([
            [vp[0], vp[2]], [vp[1], vp[2]],
            [vp[1], vp[3]], [vp[0], vp[3]],
        ]), n_oct)
        lw = [2.0, 1.0, 0.5]
        coarse_hexes, coarse_v_k, coarse_layer = None, None, overlay_layers[0]
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
            xc2, _, xpm = tail_unpack_reversible(hex_v_k[:, -1])  # (c2, r_mo, p_mo)
            verts_n = hex_verts_in_noct(hex_par, hex_oid, xpm, xc2, hex_scale, n_oct)
            hexes = verts_n.coords.reshape(-1, 6, 2)
            # Weight each tier by continuous depth d = layer - t, so line-width and
            # alpha vary smoothly and never snap when the integer level ticks over.
            # Static frames show every tier at full weight (no zoom to fade across).
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
            # Drop labels whose bottom-left anchor lands off the net (in a butterfly
            # notch or margin) — absent rather than mis-placed.
            on_net = np.zeros(len(label_pos), dtype=bool)
            for _polys in n_oct.face_polys.values():
                for _poly in _polys:
                    on_net |= inside_convex_polygon_cw(label_pos, _poly, tol=0.0)
            strs = [''.join(f'{int(d):01x}' for d in row[:-1]) for row in v_k]
            lbl_halo = [pe.withStroke(linewidth=1.2, foreground=(0, 0, 0, 0.5 * alpha))]
            for (lx, ly), lbl, on in zip(label_pos, strs, on_net):
                if not on:
                    continue
                ax.text(lx, ly, lbl[plen:],
                        fontsize=12, ha='left', va='bottom',
                        color=(1.0, 1.0, 1.0, alpha),
                        path_effects=lbl_halo,
                        zorder=100, clip_on=True)

        hex_text = f'Hex9 // Fly // L{L}: {count} cells; {area_str}/cell'
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

        # Dark tablet + a fine, soft halo → legible on any background without the
        # heavy outline demanding attention (no per-L colour hack).
        tablet = dict(boxstyle='round,pad=0.35', fc=(0, 0, 0, 0.45), ec='none')
        soft_halo = [pe.withStroke(linewidth=1.0, foreground=(0, 0, 0, 0.5))]
        ax.text(0.012, 0.012, hex_text,
                transform=ax.transAxes,
                fontsize=16, ha='left', va='bottom', family='monospace',
                color=(1.0, 0.85, 0.0, 1.0),
                path_effects=soft_halo, bbox=tablet,
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
        # Imagery-credit stack (lower-right); scrolls up as sources composite in.
        credit_list = getattr(src_fn, 'credits', None) or [getattr(src_fn, 'attribution', '')]
        _draw_credits(ax, credit_list, frame_idx, bg_sources, frames_per_level, animate)

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

    EOX_WMTS = 'https://tiles.maps.eox.at/wmts/1.0.0/WMTSCapabilities.xml'
    ESRI_WMTS = 'https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/WMTS/1.0.0/WMTSCapabilities.xml'
    OAM_URL = 'https://tiles.openaerialmap.org/6107d91f343da30006976e12/0/6107d91f343da30006976e13/{z}/{x}/{y}'

    # https://science.nasa.gov/earth/earth-observatory/blue-marble-next-generation/base-topography-bathymetry/
    _pc = make_pc_source('src/bm_3600x1800.png', _reg, _p_pix, _b_oct,
                         attribution='NASA Visible Earth: Blue Marble · visibleearth.nasa.gov')
    _s2 = make_wmts_source(EOX_WMTS, 's2cloudless-2020', _reg, _b_oct, gain=1.1, gamma=1.1,
                           attribution='Sentinel-2 cloudless 2020 · © EOX IT Services GmbH')
    _s3 = make_wmts_source(ESRI_WMTS, 'World_Imagery', _reg, _b_oct, gain=1.0, gamma=1.0,
                           attribution='Esri World Imagery · Esri, Maxar, Earthstar Geographics')
    # https://map.openaerialmap.org/#/1.641082763671875,49.09994811924072,15/square/12020223221032303/6107dae4343da30006976e14?_k=94noo8
    # OAM is a partial-footprint drone mosaic — composite it over Esri so its
    # out-of-footprint area shows Esri context instead of black.
    _s4_oam = make_xyz_source(OAM_URL, _reg, _b_oct, max_zoom=26, with_coverage=True,
                              gain=1.0, gamma=1.0,
                              attribution='Pierre d\'HUY ©2021  · OpenAerialMap, CC BY-SA 4.0')
    _s4 = make_composite_source(_s4_oam, _s3)
    # https://valdoise.observatoiredesarbres.fr/en/portail/583/observatoire/71781/cedre-du-liban-arboretum-de-la-roche-guyon-95.html
    _s5_src = make_bm_source('src/ex0098/sign.tif', _reg, _b_oct, gamma=1.0, with_coverage=True,
                         name='sign_wkt', attribution='© CAUE 95 · valdoise.observatoiredesarbres.fr (Research Only)')
    _s5 = make_composite_source(_s5_src, _s4)   # sign over OAM-over-Esri (not raw OAM)
    #
    # Composite of Two images.
    # https://commons.wikimedia.org/wiki/File:Alners_Gorse_Butterfly_Reserve,_Peacock_butterfly_%27Aglais_io%27_on_Common_fleabane_%27Pulicaria_dysenterica%27_-_geograph.org.uk_-_7875109.jpg
    # https://commons.wikimedia.org/wiki/File:2021-08-16_-_Peacock_butterfly_(Aglais_io)_-_eyespot_on_forewing_-_colourful_scales_-_DSG3404-1_(magnif._ratio_2.2x,_HiRes_focus_stack).jpg
    _s7_src = make_bm_source('src/ex0098/fly_55mm.tif', _reg, _b_oct, gamma=1.0, with_coverage=True,
                             name='fly_wkt', attribution='Michael Garlick ©2024 / Franz van Duns  ©2023 · Wikimedia Commons, CC BY 2.0')
    _s7 = make_composite_source(_s7_src, _s5)   # butterfly over sign-over-OAM-over-Esri

    # https://commons.wikimedia.org/wiki/File:Escama_de_ala_de_mariposa_SEM.jpg
    _s8_src = make_bm_source('src/ex0098/scale067.tif', _reg, _b_oct, gamma=1.0, with_coverage=True,
                             name='scale_wkt', attribution='Brandon Antonio Segura Torres & Priscilla Vieto Bonilla ©2025, Wikimedia Commons, CC BY 4.0')

    _s8 = make_composite_source(_s8_src, _s7)   # butterfly over sign-over-OAM-over-Esri


    run(
        reg=_reg,
        flavour='butterfly:0500',
        scale=750,
        frames_per_level=1000,   # index resolution: frame i ↔ zoom t = i/1000
        max_frame=25500,         # 24 levels · 1000
        bg_sources={             # keys are frame indices (introduce a source at a point)
            0: _s2,
            5430: _s3,
            5440: _s4,
            10900: _s5,
            12450: _s7,          # butterfly — tune the exact index by scrubbing frames
            18480: _s8
        },
        frame_interval=10,       # render stride: files 000000, 000010, … (=1000 → per level)
        skip_existing=True,      # resume: skip frames whose PNG is already on disk
        # start_frame=25490,       # window with start_frame / end_frame (index space)
        # end_frame=25500
    )
