# Part of the Hex9 (H9) Project
# Copyright ©2026, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Expose the underlying transport tilings visually (docs/dggs-transport-tilings.md).

Six panels spanning the taxonomy of docs/dggs-transport-tilings.md.
Top row — aligned at EVERY level:
  S2      — fully-nested quadtree: zones are their own transport (m=1).
  Hex9    — half-hex (d_cell), m=2, rep-9.  Native b_oct frame.
  A4-hex  — aperture-4 hexagonal (ISEA4H-class): half-hex, m=2, rep-4
            (machine-checked, see transport_check.py). Exact planar
            construction (no reference library installed; the structure
            is pure lattice). Split assignment propagates by role:
            centre child inherits the parent split, edge child splits
            along the shared coarse edge.
Bottom row — aligned at STRIDE 2 (the rotated intermediate level dashed):
  ISEA3H  — centre-triangle, m=6, rep-9. Alignment automatic: the
            ramified prime, (1-omega)^2 = 3*unit.
  H3      — centre-triangle, m=6, rep-49. Aligned because H3 alternates
            the conjugate rotations (measured ~22.0/41.9 degrees).
  ISEA7H  — (DGGAL) centre-triangle, m=6, rep-49. Also alternates
            (measured ~38.8/58.5 degrees). DGGRID's IGEO7 rotation
            policy is TO CHECK — if it compounds instead, it has no
            transport at any stride (forward-only theorem).

In each panel one coarse transport cell is outlined gold and the fine
transport cells it owns are shaded blue (they tile it exactly, which is
the point) — the counts (16/9/4 // 9/49/49) are printed as a soft
machine check. Zones are bold black; the transport is thin.

Run:
    python docs/dggs/dggs_transport.py            # -> docs/dggs/dggs_transport.png
"""
import argparse
import math
import os

import numpy as np
from shapely.geometry import Point as ShPoint, Polygon

try:                                                        # dggal ≥0.0.7rc1
    from dggal import *                                     # noqa: F403
    _dggal_app = Application(appGlobals=globals())          # noqa: F405
    pydggal_setup(_dggal_app)                               # noqa: F405
except ImportError:                                         # panel will report
    _dggal_app = None

GOLD, BLUE = '#efdca0', '#4f86c6'
GREY_MID, DARK = '#aaaaaa', '#222222'


def local_frame(lat0, lng0):
    k = math.cos(math.radians(lat0))
    def to_xy(la, lo):
        return ((((lo - lng0 + 180) % 360) - 180) * k, la - lat0)
    return to_xy


def spokes_and_triangles(centre, ring, to_xy):
    """A cell's centre-triangles: (centre, v_i, v_i+1) in panel coords."""
    c = to_xy(*centre)
    vs = [to_xy(la, lo) for la, lo in ring]
    return [Polygon([c, vs[i], vs[(i + 1) % len(vs)]]) for i in range(len(vs))]


def draw_poly(ax, coords, **kw):
    from matplotlib.patches import Polygon as MplPolygon
    ax.add_patch(MplPolygon(list(coords), closed=True, **kw))


# ── panels ───────────────────────────────────────────────────────────────────

def panel_hex9(ax, lat, lng, res):
    from hhg9 import Registrar, Points
    from hhg9.h9.grid import enmesh
    reg = Registrar()
    g_gcd, b_oct = reg.domain('g_gcd'), reg.domain('b_oct')

    # anchor point in b_oct; its home half-hex sets the length scale
    apts = reg.project(Points(np.array([[lat, lng]], float), g_gcd),
                       [g_gcd, b_oct])
    p0, oid0 = apts.coords[0], int(apts.oid[0])
    u0, _ = enmesh(apts, levels=res)
    w0 = float(max(np.ptp(u0[-1][:, 0]), np.ptp(u0[-1][:, 1])))   # layer-res half-hex

    # dense sample directly in b_oct (oid seeded from the anchor: one octant
    # frame, so the whole view fills even next to an octant seam)
    n = 64
    gx = np.repeat(np.linspace(p0[0] - 1.9 * w0, p0[0] + 1.9 * w0, n), n)
    gy = np.tile(np.linspace(p0[1] - 1.7 * w0, p0[1] + 1.7 * w0, n), n)
    pts = Points(np.column_stack([gx, gy]), b_oct, oid=oid0)
    uniq, refs = enmesh(pts, levels=res + 1)
    coarse = {tuple(np.round(uniq[i].ravel(), 12)): uniq[i]
              for i in np.unique(refs[:, res])}
    fine = {tuple(np.round(uniq[i].ravel(), 12)): uniq[i]
            for i in np.unique(refs[:, res + 1])}
    home = min(coarse.values(),
               key=lambda h: np.linalg.norm(h.mean(0) - p0))
    HP = Polygon(home)
    partner = min((h for h in coarse.values() if not np.allclose(h, home)),
                  key=lambda h: HP.symmetric_difference(Polygon(h)).area
                  if Polygon(h).intersection(HP.buffer(1e-9)).length >= 0 else 9,
                  default=None)
    # partner = coarse half-hex sharing the long (diameter) edge → max shared boundary
    partner = max((h for h in coarse.values() if not np.allclose(h, home)),
                  key=lambda h: Polygon(h).buffer(1e-12).intersection(
                      HP.buffer(1e-12)).area +
                      Polygon(h).exterior.intersection(HP.exterior).length)

    # reference zone fills: the anchor hexagon, and one fine hexagon
    # (a pair of fine half-hexes) inside the partner half
    draw_poly(ax, home, facecolor='#f2efe8', edgecolor='none', zorder=0)
    draw_poly(ax, partner, facecolor='#f2efe8', edgecolor='none', zorder=0)
    PP = Polygon(partner)
    f0 = min((h for h in fine.values() if PP.contains(ShPoint(h.mean(0)))),
             key=lambda h: np.linalg.norm(h.mean(0) - np.asarray(PP.centroid.coords[0])))
    f1 = max((h for h in fine.values() if not np.allclose(h, f0)),
             key=lambda h: Polygon(h).buffer(1e-12).intersection(
                 Polygon(f0).buffer(1e-12)).area +
                 Polygon(h).exterior.intersection(Polygon(f0).exterior).length)
    for f in (f0, f1):
        draw_poly(ax, f, facecolor='#dcdcdc', edgecolor='none', zorder=0.5)

    for h in fine.values():
        draw_poly(ax, h, fill=False, edgecolor='#888888', linewidth=0.6)
    for h in coarse.values():
        draw_poly(ax, h, fill=False, edgecolor=DARK, linewidth=1.4)

    owned = [h for h in fine.values() if HP.contains(ShPoint(h.mean(0)))]
    for h in owned:
        draw_poly(ax, h, facecolor=BLUE, edgecolor='#555555',
                  linewidth=0.5, alpha=.55)
    draw_poly(ax, home, fill=False, edgecolor='#c8991f', linewidth=3.0, zorder=5)
    hexagon = Polygon(home).union(Polygon(partner))
    xs, ys = hexagon.exterior.xy
    ax.plot(xs, ys, color=DARK, linewidth=2.6)

    w = max(np.ptp(home[:, 0]), np.ptp(home[:, 1]))
    cx, cy = home.mean(0)
    ax.set_xlim(cx - 1.6 * w, cx + 1.6 * w)
    ax.set_ylim(cy - 1.4 * w, cy + 1.4 * w)
    print(f'Hex9   L{res}: fine half-hexes in shaded coarse half-hex = '
          f'{len(owned)} (expect 9)')
    return len(owned)


def panel_h3(ax, lat, lng, res):
    import h3
    anchor = h3.latlng_to_cell(lat, lng, res)
    to_xy = local_frame(*h3.cell_to_latlng(anchor))

    def ring(c):
        return [to_xy(la, lo) for la, lo in h3.cell_to_boundary(c)]

    A = Polygon(ring(anchor))
    view = A.buffer(0.35 * math.sqrt(A.area))
    draw_poly(ax, ring(anchor), facecolor='#f2efe8', edgecolor='none', zorder=0)
    mid = h3.cell_to_children(anchor, res + 1)
    fine = [c for z in h3.grid_disk(anchor, 1)
            for c in h3.cell_to_children(z, res + 2)
            if view.intersects(Polygon(ring(c)))]

    fine_tris = []
    for c in fine:
        rg = ring(c)
        draw_poly(ax, rg, fill=False, edgecolor='#999999', linewidth=0.5)
        fine_tris += spokes_and_triangles(
            h3.cell_to_latlng(c), h3.cell_to_boundary(c), to_xy)
        for t in fine_tris[-len(rg):]:
            xs, ys = t.exterior.xy
            ax.plot(xs, ys, color='#bbbbbb', linewidth=0.4, zorder=1)
    for c in mid:
        draw_poly(ax, ring(c), fill=False, edgecolor='#8a8a8a',
                  linewidth=1.1, linestyle=(0, (4, 3)), zorder=4)

    coarse_tris = spokes_and_triangles(
        h3.cell_to_latlng(anchor), h3.cell_to_boundary(anchor), to_xy)
    T = coarse_tris[0]
    draw_poly(ax, list(T.exterior.coords)[:-1], fill=False,
              edgecolor='#c8991f', linewidth=3.0, zorder=5)
    owned = [t for t in fine_tris if T.contains(t.centroid)]
    for t in owned:
        draw_poly(ax, list(t.exterior.coords)[:-1], facecolor=BLUE,
                  edgecolor='#555555', linewidth=0.4, alpha=.55)
    for t in coarse_tris:
        xs, ys = t.exterior.xy
        ax.plot(xs, ys, color=DARK, linewidth=1.4)
    draw_poly(ax, ring(anchor), fill=False, edgecolor=DARK, linewidth=2.6)

    # reference fine-zone fill: a fine cell WELL inside the anchor (clear
    # margin, so containment is legible at figure scale), far from T
    inner = A.buffer(-0.6 * math.sqrt(Polygon(ring(fine[0])).area))
    far = max((c for c in fine if inner.contains(Polygon(ring(c)))),
              key=lambda c: Polygon(ring(c)).centroid.distance(T.centroid))
    draw_poly(ax, ring(far), facecolor='#dcdcdc', edgecolor='none', zorder=0.5)

    b = view.bounds
    ax.set_xlim(b[0], b[2]); ax.set_ylim(b[1], b[3])
    print(f'H3     r{res}: fine triangles in shaded coarse triangle = '
          f'{len(owned)} (expect 49)')
    return len(owned)


def panel_dggal(ax, lat, lng, res, dggrs_name, expect):
    """Generic DGGAL stride-2 panel (ISEA3H, ISEA7H): centre-triangle
    transport, rotated intermediate level dashed."""
    if _dggal_app is None:
        raise NotImplementedError('dggal unavailable (pip install dggal)')
    d = eval(dggrs_name)()                                  # noqa: F405,S307
    anchor = d.getZoneFromWGS84Centroid(res, GeoPoint(Degrees(lat), Degrees(lng)))  # noqa: F405
    ac = d.getZoneWGS84Centroid(anchor)
    to_xy = local_frame(float(ac.lat), float(ac.lon))

    def centre(z):
        c = d.getZoneWGS84Centroid(z)
        return float(c.lat), float(c.lon)

    def ring(z):
        return [(float(p.lat), float(p.lon)) for p in d.getZoneWGS84Vertices(z)]

    def xy_ring(z):
        return [to_xy(la, lo) for la, lo in ring(z)]

    nb_types = Array('<int>')                               # noqa: F405
    hood = [anchor] + list(d.getZoneNeighbors(anchor, nb_types))
    mid, fine = {}, {}
    for sz in d.getSubZones(anchor, 1):
        mid[d.getZoneTextID(sz)] = sz
    for z in hood:
        for sz in d.getSubZones(z, 2):
            fine[d.getZoneTextID(sz)] = sz

    fine_tris = []
    for z in fine.values():
        draw_poly(ax, xy_ring(z), fill=False, edgecolor='#999999', linewidth=0.5)
        tris = spokes_and_triangles(centre(z), ring(z), to_xy)
        fine_tris += tris
        for t in tris:
            xs, ys = t.exterior.xy
            ax.plot(xs, ys, color='#bbbbbb', linewidth=0.45, zorder=1)
    for z in mid.values():
        draw_poly(ax, xy_ring(z), fill=False, edgecolor='#8a8a8a',
                  linewidth=1.1, linestyle=(0, (4, 3)), zorder=4)

    coarse_tris = spokes_and_triangles(centre(anchor), ring(anchor), to_xy)
    T = coarse_tris[0]
    draw_poly(ax, list(T.exterior.coords)[:-1], fill=False,
              edgecolor='#c8991f', linewidth=3.0, zorder=5)
    owned = [t for t in fine_tris if T.contains(t.centroid)]
    for t in owned:
        draw_poly(ax, list(t.exterior.coords)[:-1], facecolor=BLUE,
                  edgecolor='#555555', linewidth=0.4, alpha=.55)
    for t in coarse_tris:
        xs, ys = t.exterior.xy
        ax.plot(xs, ys, color=DARK, linewidth=1.4)
    draw_poly(ax, xy_ring(anchor), fill=False, edgecolor=DARK, linewidth=2.6)

    A = Polygon(xy_ring(anchor))
    draw_poly(ax, xy_ring(anchor), facecolor='#f2efe8', edgecolor='none', zorder=0)
    z0 = next(iter(fine.values()))
    inner = A.buffer(-0.6 * math.sqrt(Polygon(xy_ring(z0)).area))
    far = max((z for z in fine.values() if inner.contains(Polygon(xy_ring(z)))),
              key=lambda z: Polygon(xy_ring(z)).centroid.distance(T.centroid))
    draw_poly(ax, xy_ring(far), facecolor='#dcdcdc', edgecolor='none', zorder=0.5)
    b = A.buffer(0.45 * math.sqrt(A.area)).bounds
    ax.set_xlim(b[0], b[2]); ax.set_ylim(b[1], b[3])
    print(f'{dggrs_name} L{res}: fine triangles in shaded coarse triangle = '
          f'{len(owned)} (expect {expect})')
    return len(owned)


def panel_s2(ax, lat, lng, res):
    """S2: fully-nested quadtree — the zones ARE their own transport (m=1).
    The intermediate level is dashed and ALIGNS (contrast the hex panels)."""
    import s2sphere as s2
    anchor = s2.CellId.from_lat_lng(s2.LatLng.from_degrees(lat, lng)).parent(res)
    c0 = anchor.to_lat_lng()
    to_xy = local_frame(c0.lat().degrees, c0.lng().degrees)

    def ring(cid):
        cell = s2.Cell(cid)
        pts = [s2.LatLng.from_point(cell.get_vertex(k)) for k in range(4)]
        return [to_xy(p.lat().degrees, p.lng().degrees) for p in pts]

    hood = [anchor] + list(anchor.get_all_neighbors(res))
    fine, mid = [], []
    for z in hood:
        f = [z]
        for _ in range(2):
            f = [c for cid in f for c in cid.children()]
        fine += f
        mid += list(z.children())

    for c in fine:
        draw_poly(ax, ring(c), fill=False, edgecolor='#999999', linewidth=0.5)
    for c in mid:
        draw_poly(ax, ring(c), fill=False, edgecolor='#8a8a8a',
                  linewidth=1.1, linestyle=(0, (4, 3)), zorder=4)
    A = Polygon(ring(anchor))
    draw_poly(ax, ring(anchor), facecolor='#f2efe8', edgecolor='none', zorder=0)
    owned = [c for c in fine if A.contains(Polygon(ring(c)).centroid)]
    for c in owned:
        draw_poly(ax, ring(c), facecolor=BLUE, edgecolor='#555555',
                  linewidth=0.4, alpha=.55)
    draw_poly(ax, ring(anchor), fill=False, edgecolor='#c8991f',
              linewidth=3.0, zorder=5)
    draw_poly(ax, ring(anchor), fill=False, edgecolor=DARK, linewidth=2.6)

    b = A.buffer(0.45 * math.sqrt(A.area)).bounds
    ax.set_xlim(b[0], b[2]); ax.set_ylim(b[1], b[3])
    print(f'S2     L{res}: fine cells in shaded coarse cell = '
          f'{len(owned)} (expect 16)')
    return len(owned)


def panel_a4_planar(ax):
    """Aperture-4 hexagonal (ISEA4H-class), exact planar construction —
    half-hex transport (m=2, rep-4, machine-checked in transport_check).
    Split assignment is dictated by role: a centre child inherits its
    parent's split; an edge child splits along the shared coarse edge."""
    import itertools

    def cs(a):
        return np.array([math.cos(math.radians(a)), math.sin(math.radians(a))])

    corners1 = np.array([cs(30 + 60 * k) for k in range(6)])
    # split direction (a corner-diameter, degrees) by fine-lattice parity;
    # coarse cells all split at 90 (the root choice, propagated)
    SPLIT = {(0, 0): 90, (1, 0): 90, (0, 1): 150, (1, 1): 30}

    def hex_cells(side, extent):
        step = side * math.sqrt(3)
        for a, b in itertools.product(range(-8, 9), repeat=2):
            c = np.array([step * (a + b / 2), 1.5 * side * b])
            if abs(c[0]) < extent and abs(c[1]) < extent:
                yield (a, b), c, c + corners1 * side

    def halves(c, side, theta):
        quad = [c + side * cs(theta + 60 * k) for k in range(4)]
        return [np.array(quad), np.array([2 * c - q for q in quad])]

    # reference zone fills: the anchor hexagon and its central fine child
    draw_poly(ax, corners1, facecolor='#f2efe8', edgecolor='none', zorder=0)
    draw_poly(ax, 0.5 * corners1, facecolor='#dcdcdc', edgecolor='none', zorder=0.5)

    fine_haves = []
    for (a, b), c, rg in hex_cells(0.5, 3.2):
        draw_poly(ax, rg, fill=False, edgecolor='#999999', linewidth=0.5)
        th = SPLIT[(a % 2, b % 2)]
        for h in halves(c, 0.5, th):
            fine_haves.append(Polygon(h))
        ax.plot([c[0] + .5 * cs(th)[0], c[0] - .5 * cs(th)[0]],
                [c[1] + .5 * cs(th)[1], c[1] - .5 * cs(th)[1]],
                color='#999999', linewidth=0.5, zorder=1)
    for (a, b), c, rg in hex_cells(1.0, 3.2):
        draw_poly(ax, rg, fill=False, edgecolor=DARK, linewidth=1.4)
        ax.plot([c[0], c[0]], [c[1] + 1, c[1] - 1],
                color=DARK, linewidth=1.4, zorder=2)     # coarse split (90)

    T = Polygon([cs(90), cs(150), cs(210), cs(270)])     # anchor left half-hex
    owned = [h for h in fine_haves if T.contains(h.centroid)]
    for h in owned:
        draw_poly(ax, list(h.exterior.coords)[:-1], facecolor=BLUE,
                  edgecolor='#555555', linewidth=0.5, alpha=.55)
    draw_poly(ax, list(T.exterior.coords)[:-1], fill=False,
              edgecolor='#c8991f', linewidth=3.0, zorder=5)
    draw_poly(ax, corners1, fill=False, edgecolor=DARK, linewidth=2.6)

    ax.set_xlim(-2.1, 2.1); ax.set_ylim(-1.85, 1.85)
    print(f'A4-hex planar: fine half-hexes in shaded coarse half-hex = '
          f'{len(owned)} (expect 4)')
    return len(owned)


def main(lat, lng, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(19, 13))
    titles = [
        'S2 — fully-nested: zones are their own transport\nm=1, rep-4 '
        '(every level; dashed intermediate level aligns)',
        'Hex9 — half-hex transport, m=2, rep-9 (every level)',
        'Aperture-4 hex (ISEA4H-class) — half-hex transport, m=2, rep-4\n'
        '(every level; exact planar construction, machine-checked)',
        'ISEA3H — centre-triangle transport, m=6, rep-9 (stride 2)\n'
        'dashed: the rotated (30°) intermediate level',
        'H3 — centre-triangle transport, m=6, rep-49 (stride 2)\n'
        'dashed: the rotated (19.1°) odd resolution the transport skips',
        'ISEA7H (DGGAL) — centre-triangle transport, m=6, rep-49 (stride 2)\n'
        'dashed: rotated (19.1°) intermediate — alternates like H3',
    ]
    panel_s2(axes[0, 0], lat, lng, res=8)
    panel_hex9(axes[0, 1], lat, lng, res=5)
    panel_a4_planar(axes[0, 2])
    panel_dggal(axes[1, 0], lat, lng, 11, 'ISEA3H', 9)
    panel_h3(axes[1, 1], lat, lng, res=5)
    panel_dggal(axes[1, 2], lat, lng, 6, 'ISEA7H', 49)
    for ax, t in zip(axes.flat, titles):
        ax.set_title(t, fontsize=11)
        ax.set_aspect('equal')
        ax.axis('off')
    fig.suptitle('The underlying transport — zones (bold black) as exact unions '
                 'of a fully-nested substructure (thin);\none coarse transport '
                 'cell (gold outline) tiled exactly by the fine transport cells '
                 'it owns (blue);\npale fills: reference zones at both scales — '
                 'note fine zones straddle the transport, transport cells never '
                 'straddle', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=3.0)
    fig.savefig(out_path, dpi=140)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--lat', type=float, default=40.0)
    ap.add_argument('--lng', type=float, default=-3.0)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__),
                                                  'dggs_transport.png'))
    a = ap.parse_args()
    main(a.lat, a.lng, a.out)
