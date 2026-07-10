# Part of the Hex9 (H9) Project
# Copyright ©2026, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Machine checks for stride transports (docs/dggs-transport-tilings.md §3b).

Currently: H3's stride-2 centre-triangle transport. H3's resolutions
alternate Class II/III (±19.107°), so same-parity resolutions are aligned
and scale by exactly 7 per two levels. Splitting every cell into its 6
centre-triangles (cell centre + one boundary edge each) gives an exact
ownership relation at EVEN relative depths, using only the public API:

    owner(cell, R) = latlng_to_cell(interior point of cell's
                                    representative triangle, R)

Checks (cf. uber/h3#1095 and opengeospatial/ogcapi-dggs#108):

  1. COUNTS   — every non-pentagon anchor owns exactly 7^d cells at even
                relative depth d (partition property, N^d counts).
  2. WEIGHTS  — every owned cell straddling the anchor's boundary is
                split exactly in half (overlap fraction ½), because the
                anchor's boundary follows fine-triangle edges at
                matching parity. Optional: needs shapely.

Also: the aperture-4 half-hex rep-4 check (Sahr 2011 fig. 4; ISEA4H-class
m=2 transport) — the trapezoid subdivides exactly into 4 half-scale
direct-similar copies (rotations 0/120/180/240), iterated to depth 3.

Run:
    python transport_check.py                 # counts d=2 (40 anchors), d=4 (8) + rep-4
    python transport_check.py --weights       # also the shapely overlap check

Both checks are floating-point geometry over the library's own
projection; the triangle structure is lattice-native, so an exact
integer IJK implementation is possible and would upgrade this to an
exact proof. Odd relative depths are structurally excluded (the grids
differ by arctan(√3/5), an irrational multiple of π).
"""
import argparse
import math
import random


def _to_xyz(lat, lng):
    la, lo = math.radians(lat), math.radians(lng)
    return (math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la))


def _to_latlng(x, y, z):
    r = math.sqrt(x * x + y * y + z * z)
    return math.degrees(math.asin(z / r)), math.degrees(math.atan2(y, x))


def rep_triangle_point(cell):
    """Interior point of the cell's representative triangle (cell centre +
    first boundary edge). Vector-averaged: antimeridian/pole safe."""
    import h3
    pts = [h3.cell_to_latlng(cell)] + list(h3.cell_to_boundary(cell)[:2])
    vs = [_to_xyz(la, lo) for la, lo in pts]
    return _to_latlng(*(sum(c) / 3.0 for c in zip(*vs)))


def owner(cell, coarse_res):
    """The unique owning ancestor of `cell` at coarse_res.
    Exact when (res(cell) - coarse_res) is EVEN."""
    import h3
    return h3.latlng_to_cell(*rep_triangle_point(cell), coarse_res)


def random_anchors(n, res, seed):
    import h3
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        lat = math.degrees(math.asin(rng.uniform(-1.0, 1.0)))     # area-true
        lng = rng.uniform(-180.0, 180.0)
        c = h3.latlng_to_cell(lat, lng, res)
        if not h3.is_pentagon(c) and c not in out:
            out.append(c)
    return out


def owned_cells(anchor, depth):
    import h3
    res = h3.get_resolution(anchor)
    pool = set()
    for z in h3.grid_disk(anchor, 1):
        pool.update(h3.cell_to_children(z, res + depth))
    return [c for c in pool if owner(c, res) == anchor]


def check_counts(res, depth, n_anchors, seed):
    ok = True
    expect = 7 ** depth
    for a in random_anchors(n_anchors, res, seed):
        n = len(owned_cells(a, depth))
        if n != expect:
            print(f'  FAIL {a}: owned {n} != 7^{depth} = {expect}')
            ok = False
    print(f'counts  r{res} d={depth}: {n_anchors} anchors '
          f'{"all exactly" if ok else "NOT all"} 7^{depth} = {expect}'
          f'   {"PASS" if ok else "FAIL"}')
    return ok


def check_weights(res, depth, n_anchors, seed, tol=0.05):
    """Straddler overlap fractions == 1/2. Planar lon/lat polygons with
    longitudes unwrapped around the anchor (small cells: fine for area
    ratios away from the poles; polar anchors are skipped)."""
    import h3
    from shapely.geometry import Polygon
    ok, n_strad = True, 0
    for a in random_anchors(n_anchors, res, seed):
        c_lat, c_lng = h3.cell_to_latlng(a)
        if abs(c_lat) > 80:
            continue

        def poly(cell):
            ring = [(((lo - c_lng + 180) % 360) - 180 + c_lng, la)
                    for la, lo in h3.cell_to_boundary(cell)]
            return Polygon(ring)

        A = poly(a)
        for c in owned_cells(a, depth):
            P = poly(c)
            f = P.intersection(A).area / P.area
            if tol < f < 1 - tol:                       # a genuine straddler
                n_strad += 1
                if abs(f - 0.5) > tol:
                    print(f'  FAIL {c}: straddler fraction {f:.4f} != 1/2')
                    ok = False
    print(f'weights r{res} d={depth}: {n_strad} straddlers '
          f'{"all" if ok else "NOT all"} split at 1/2 (±{tol} fp tolerance)'
          f'   {"PASS" if ok else "FAIL"}')
    return ok


def a4_halfhex_canonical():
    """Canonical aperture-4 half-hex trapezoid and its intrinsic rep-4
    subdivision (docs/dggs-transport-tilings.md §2, Sahr 2011 fig. 4):
    half of the central child plus the inner corner-diameter halves of
    the three edge-children on that side. Returns (T0, maps) where each
    map is a complex direct similarity z -> a*z + b with |a| = 1/2."""
    import numpy as np

    def cs(a):
        return np.array([math.cos(math.radians(a)), math.sin(math.radians(a))])

    T0 = np.array([cs(90), cs(150), cs(210), cs(270)])      # left half-hex, side 1
    s3 = math.sqrt(3) / 2
    pieces = [0.5 * T0]
    for th, k0 in ((120, 210), (180, 270), (240, 330)):     # edge-children
        c = s3 * cs(th)
        pieces.append(np.array([c + .5 * cs(k0 + 60 * k) for k in range(4)]))
    z0 = T0[:, 0] + 1j * T0[:, 1]
    maps = []
    for p in pieces:
        zp = p[:, 0] + 1j * p[:, 1]
        a = (zp[1] - zp[0]) / (z0[1] - z0[0])
        b = zp[0] - a * z0[0]
        assert np.allclose(a * z0 + b, zp, atol=1e-12)      # direct similarity
        maps.append((a, b))
    return T0, maps


def check_a4_halfhex_rep4(levels=3, tol=1e-9):
    """The half-hex trapezoid is rep-4 under the aperture-4 child grid:
    at every subdivision level the pieces tile the parent exactly (no
    gaps, no overlaps, all congruent, direct similarities of scale 1/2)."""
    import numpy as np
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    T0, maps = a4_halfhex_canonical()
    Tp = Polygon(T0)
    z0 = T0[:, 0] + 1j * T0[:, 1]
    zs, ok = [z0], True
    for lv in range(1, levels + 1):
        zs = [a * z + b for z in zs for a, b in maps]
        polys = [Polygon(np.column_stack([z.real, z.imag])) for z in zs]
        gap = Tp.symmetric_difference(unary_union(polys)).area
        ov = max(polys[i].intersection(polys[j]).area
                 for i in range(len(polys)) for j in range(i + 1, len(polys)))
        lv_ok = gap < tol and ov < tol and len(polys) == 4 ** lv
        ok &= lv_ok
        print(f'a4 half-hex rep-4, level {lv}: {len(polys)} pieces, '
              f'gap {gap:.1e}, overlap {ov:.1e}   {"PASS" if lv_ok else "FAIL"}')
    return ok


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--res', type=int, default=5, help='anchor resolution')
    ap.add_argument('--seed', type=int, default=9)
    ap.add_argument('--weights', action='store_true',
                    help='also run the shapely straddler-weight check')
    args = ap.parse_args()

    passed = check_counts(args.res, 2, 40, args.seed)
    passed &= check_counts(args.res, 4, 8, args.seed)
    passed &= check_a4_halfhex_rep4()
    if args.weights:
        passed &= check_weights(args.res, 2, 12, args.seed)
        passed &= check_weights(args.res, 4, 4, args.seed)
    raise SystemExit(0 if passed else 1)
