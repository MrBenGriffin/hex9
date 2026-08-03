# Part of the Hex9 (H9) Project
# Copyright ©2026, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""E4H top-down residual descent (geometric, exact-edge).

Route (Ben, 2026-07-31): leaf-up ownership was only ever sound on a
common transport, and hex9 (/3) and E4H (/2) have none — the towers are
multiplicatively incommensurable below the attach layer. So the E4H
tail is DEFINED top-down instead:

    lat/lon or uuid --> encode to the transition layer A
      => h9 partial (host address + state) + b_oct residual
      => residual classified down the aperture-4 half-hex rep-4 carrier

The carrier is transport_check.a4_halfhex_canonical (rep-4 machine-
verified there): because the half-hex is an exact rep-tile, every level
has precise straight edges and nesting holds BY CONSTRUCTION — no
fractal ownership boundary, no moire. The price: below A the cells are
half-hex trapezoids, not hexagons (hexagons never nest anyway); the
hexagon reading survives only at the attach layer.

The descent is ONE classifier applied at every level: pull the residual
back through each candidate child's similarity (all |a| = 1/2 — the /2
tower), score signed distance to the canonical trapezoid, keep the
best; the pulled-back residual IS the next level's residual (the x2
renormalisation and the classification are the same operation). Digit 0
selects the half (hexagon = T0 u -T0); digits 1..B select rep-4
children (0 = central half-child, 1..3 = edge-child halves, order =
a4_halfhex_canonical piece order).

Conventions:
  * digit alphabet PINNED (Ben, 2026-07-31): 0 = centre half-child,
    2 = same-axis (180 deg) edge-child half, 1 / 3 = the wings
    (120 / 240 deg) — which is a4_halfhex_canonical's piece order
    verbatim. Digits are HOST-FRAME-RELATIVE: consistency across the
    plane and the octahedron comes from the hex9 rotation state
    (oid_th / rid), NOT from a global plane frame — the spike's shared
    single-octant frame from build_frame is the one remaining
    provisional stand-in, and wiring per-host frames to H9O state is
    the pinning step.
  * host enumeration = hex9's own, unchanged (E4H is a pure suffix on
    the h9 partial).
  * host = h9 address truncation (lineage), per the transition-record
    spec; the lineage-vs-hexagon discrepancy at layer A is REPORTED,
    not hidden (it is hex9's own boundary, not the descent's);
  * label form <h9-label>E<half><digits>; cid -> did/xid packing deferred.

Cone points (probe 2026-07-31, scratchpad cone_probe.py): the layer-A
grid is CORNER-REGISTERED at the 6 octahedron vertices — exactly 2
hosts ring each apex (2 x 120 deg = the 240 deg cone angle; 12
cone-touching hexes per layer). So every host, cone-touchers included,
is an intact hexagon with the apex on its BOUNDARY: interiors carry no
curvature, each host unfolds isometrically to the regular planar
hexagon, and this descent applies UNCHANGED at every host on the globe
— no pentagon cell, no special alphabet. The 2*pi/3 defect surfaces
only BETWEEN hosts (2-around-a-corner adjacency + frame holonomy),
i.e. in cross-host frame rotation — precisely what the hex9 rotations
already track.

Checks:
  1. PARTITION  — the 2*4^B depth-B leaves of a host tile the hexagon
                  exactly (union gap ~0, area sum exact, count 2*4^B).
  2. NESTING    — every cell lies inside its parent, all levels.
  3. ROUND-TRIP — every leaf centre re-descends to its own digits.
  4. PIPELINE   — real lat/lon: h9_encode -> layer A + residual ->
                  digits; the leaf trapezoid (mapped back to b_oct)
                  contains the point, for every in-hexagon residual;
                  lineage-vs-geometric host agreement reported.

Run:  python experimental/e4h_descent.py [--viz out.png]
"""
import argparse
import itertools
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'docs' / 'dggs'))
from transport_check import a4_halfhex_canonical
from e4h_forward import A, BOX, build_frame

B_E4 = 3          # descent depth (a4 digits after the half digit)
N_PTS = 800
S3 = math.sqrt(3)


def frames():
    """Canonical trapezoid as complex vertices, child similarity LUTs,
    and the unit-normalised CCW edge list used by the classifier."""
    T0, maps = a4_halfhex_canonical()
    z0 = T0[:, 0] + 1j * T0[:, 1]
    halves = [(1 + 0j, 0j), (-1 + 0j, 0j)]        # hexagon = T0 u -T0
    edges = []
    for i in range(4):
        p, q = z0[i], z0[(i + 1) % 4]
        e = (q - p) / abs(q - p)
        edges.append((p, e))
    maps = [(complex(a), complex(b)) for a, b in maps]
    return z0, maps, halves, edges


def score(w, edges):
    """Min signed distance of w to the canonical trapezoid (CCW edges;
    positive inside). The single classifier of the whole tower."""
    return min((e.conjugate() * (w - p)).imag for p, e in edges)


def classify(w, cand, edges):
    """One descent step: best child under pull-back; returns
    (digit, residual in the chosen child's own canonical frame)."""
    us = [(w - b) / a for a, b in cand]
    ss = [score(u, edges) for u in us]
    d = int(np.argmax(ss))
    return d, us[d]


def descend(w, maps, halves, edges, depth=B_E4):
    d, w = classify(w, halves, edges)
    digits = [d]
    for _ in range(depth):
        d, w = classify(w, maps, edges)
        digits.append(d)
    return digits


def fold(digits, maps, halves):
    """Compose the descent similarities: canonical T0 -> cell."""
    a, b = halves[digits[0]]
    for d in digits[1:]:
        a2, b2 = maps[d]
        a, b = a * a2, a * b2 + b
    return a, b


def cell_poly(digits, z0, maps, halves):
    a, b = fold(digits, maps, halves)
    return a * z0 + b


def check_canonical(z0, maps, halves, edges):
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    hexp = Polygon([(math.cos(math.radians(30 + 60 * k)),
                     math.sin(math.radians(30 + 60 * k))) for k in range(6)])
    ok = True

    # 1. PARTITION at depth B
    leaves = list(itertools.product(
        range(2), *[range(4)] * B_E4))
    polys = [Polygon(np.column_stack([v.real, v.imag]))
             for v in (cell_poly(d, z0, maps, halves) for d in leaves)]
    gap = hexp.symmetric_difference(unary_union(polys)).area
    asum = abs(sum(p.area for p in polys) - hexp.area)
    p_ok = gap < 1e-9 and asum < 1e-9 and len(polys) == 2 * 4 ** B_E4
    ok &= p_ok
    print(f'partition : {len(polys)} leaves (2*4^{B_E4}), union gap '
          f'{gap:.1e}, area-sum defect {asum:.1e}   '
          f'{"PASS" if p_ok else "FAIL"}')

    # 2. NESTING, every level
    bad = 0
    for lv in range(1, B_E4 + 1):
        for d in itertools.product(range(2), *[range(4)] * lv):
            par = Polygon(np.column_stack([v.real, v.imag])).buffer(1e-9) \
                if (v := cell_poly(d[:-1], z0, maps, halves)) is not None else None
            chd = cell_poly(d, z0, maps, halves)
            if Polygon(np.column_stack([chd.real, chd.imag])) \
                    .difference(par).area > 1e-12:
                bad += 1
    ok &= bad == 0
    print(f'nesting   : child-within-parent violations {bad} '
          f'(levels 1..{B_E4})   {"PASS" if bad == 0 else "FAIL"}')

    # 3. ROUND-TRIP from leaf centres
    bad = 0
    for d in leaves:
        v = cell_poly(d, z0, maps, halves)
        c = Polygon(np.column_stack([v.real, v.imag])).centroid
        if tuple(descend(complex(c.x, c.y), maps, halves, edges)) != d:
            bad += 1
    ok &= bad == 0
    print(f'round-trip: {len(leaves) - bad}/{len(leaves)} leaf centres '
          f're-descend to their own digits   {"PASS" if bad == 0 else "FAIL"}')
    return ok


def pipeline(z0, maps, halves, edges, viz_path=None):
    from shapely.geometry import Point as SPoint, Polygon
    from hhg9 import Points, Registrar
    from hhg9.h9.uuid_address import (h9_bin, h9_bin_pts, h9_dec, h9_encode,
                                      h9_label)
    reg = Registrar()
    b_oct = reg.domain('b_oct')
    rng = np.random.default_rng(9)
    full = h9_encode(rng.uniform(BOX[0], BOX[1], N_PTS),
                     rng.uniform(BOX[2], BOX[3], N_PTS), reg)
    hosts = h9_bin(full, A, reg)                   # lineage truncation
    pts = h9_dec(full, b_oct)
    xy = pts.coords[:, :2]
    c0, T, Tinv = build_frame(reg, b_oct)
    e1c = complex(T[0, 0], T[1, 0])
    k = S3 / e1c                                   # world -> canonical

    uniq = list(dict.fromkeys(hosts))
    hc = dict(zip((u.int for u in uniq),
                  h9_dec(uniq, b_oct).coords[:, :2]))
    hosts_geo = h9_bin_pts(Points(xy, domain=b_oct, oid=pts.oid), A)
    n_lin = sum(hu.int != hg.int for hu, hg in zip(hosts, hosts_geo))

    n_in, n_stray, n_cont, rows = 0, 0, 0, []
    for u, p in zip(hosts, xy):
        z = complex(*(p - hc[u.int]))
        w = z * k
        hex_side = max(min((e.conjugate() * (s * w - q)).imag
                           for q, e in edges)
                       for s in (1, -1))                    # in T0 u -T0
        inside = hex_side > -1e-9
        n_in += inside
        n_stray += not inside
        digits = descend(w, maps, halves, edges)
        v = cell_poly(digits, z0, maps, halves) / k
        leaf = Polygon(np.column_stack([v.real + hc[u.int][0],
                                        v.imag + hc[u.int][1]]))
        if inside and leaf.buffer(1e-9).contains(SPoint(*p)):
            n_cont += 1
        rows.append((u, digits, w))

    ok = n_cont == n_in
    i0 = 0
    u0, d0, w0 = rows[i0]
    lab = f'{h9_label(u0, with_tail=False)}E{"".join(map(str, d0))}'
    print(f'pipeline  : {N_PTS} pts -> layer {A}; residual in-hexagon '
          f'{n_in}, stray {n_stray} (lineage edge), lineage!=geometric '
          f'host {n_lin}')
    print(f'containmnt: leaf trapezoid contains point {n_cont}/{n_in} '
          f'in-hexagon pts   {"PASS" if ok else "FAIL"}')
    print(f'example   : {lab}   residual (host-pitch frame) '
          f'({w0.real / S3:+.3f}, {w0.imag / S3:+.3f})')

    if viz_path:
        visualise(viz_path, z0, maps, halves, rows[i0], hc, k, xy[i0], lab)
        print(f'viz       : {viz_path}')
    return ok


def visualise(path, z0, maps, halves, row, hc, k, p_xy, lab):
    """Descent cascade in the host's world frame: depth-1 partition as
    backdrop, then the point's chain refined level by level — the exact
    nesting is the picture."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MP
    BLUE, GOLD, RED = '#004488', '#DDAA33', '#BB5566'

    u, digits, _ = row
    hcx = hc[u.int]
    world = lambda v: np.column_stack([(v / k).real + hcx[0],
                                       (v / k).imag + hcx[1]])
    fig, ax = plt.subplots(figsize=(9, 9))
    hexv = np.array([np.exp(1j * math.radians(30 + 60 * t))
                     for t in range(7)])
    ax.plot(*world(hexv).T, color='0.3', lw=2.0)
    for d in itertools.product(range(2), range(4)):        # depth-1 backdrop
        ax.add_patch(MP(world(cell_poly(list(d), z0, maps, halves)),
                        closed=True, fc='none', ec='0.75', lw=0.8))
    for lv in range(1, B_E4 + 1):                          # cascade: siblings
        pre = digits[:lv]                                  # then chosen child
        for d4 in range(4):
            v = cell_poly(pre + [d4], z0, maps, halves)
            ax.add_patch(MP(world(v), closed=True, fc='none', ec='0.55',
                            lw=0.7))
        v = cell_poly(digits[:lv + 1], z0, maps, halves)
        last = lv == B_E4
        ax.add_patch(MP(world(v), closed=True,
                        fc=GOLD if last else 'none',
                        alpha=0.55 if last else 1.0,
                        ec=BLUE, lw=2.6 - 0.4 * lv))
    v = cell_poly(digits[:1], z0, maps, halves)
    ax.add_patch(MP(world(v), closed=True, fc='none', ec=BLUE, lw=2.6))
    ax.plot(*p_xy, marker='*', color=RED, ms=16, zorder=7)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'E4H top-down residual descent — exact rep-4 nesting\n'
                 f'{lab}   (half digit + {B_E4} a4 digits, /2 per level)',
                 fontsize=12)
    fig.savefig(path, dpi=140, bbox_inches='tight')


def main(viz_path=None):
    z0, maps, halves, edges = frames()
    assert score(-0.4 + 0j, edges) > 0             # CCW sanity: interior +ve
    ok = check_canonical(z0, maps, halves, edges)
    ok &= pipeline(z0, maps, halves, edges, viz_path)
    return ok


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--viz', metavar='PNG', help='write a one-point cascade')
    args = ap.parse_args()
    raise SystemExit(0 if main(args.viz) else 1)
