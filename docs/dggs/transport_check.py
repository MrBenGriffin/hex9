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


def check_e4h_mode_ownership(tol=1e-9):
    """E4H mode-0 ownership balance (Hex9+E4H hybrid, 2026-07-29).

    Every aperture-4 edge child is bisected by the shared coarse edge into
    two transport half-hexes; mode = majority orientation (apex-up vs
    apex-down) of the trapezoid's three constituent fine lattice triangles,
    i.e. the triangle-parity class of the transport tiling. Claims:

      1. the two halves of every edge child have OPPOSITE mode;
      2. host-side modes alternate around each coarse cell (3 + 3);
      3. hence 'owner = host holding the mode-0 half' gives every host
         exactly 1 (centre) + 3 (edge) = 4 children — the aperture count —
         and every edge child exactly one owner.

    Which parity class is called mode-0 is a convention to pin to H9O
    oid_mo; the balance is independent of the choice."""
    s3 = math.sqrt(3)

    def cs(a):
        return (math.cos(math.radians(a)), math.sin(math.radians(a)))

    def add(p, q, f=1.0):
        return (p[0] + f * q[0], p[1] + f * q[1])

    def key(p):
        return (round(4 * p[0]), round(4 * p[1] / (s3 / 4)) )

    def tri_orient(a, b, c):
        """+1 apex-up / -1 apex-down for a fine lattice triangle (every
        lattice triangle has exactly one horizontal edge in this frame)."""
        for (p, q, r) in ((a, b, c), (b, c, a), (c, a, b)):
            if abs(p[1] - q[1]) < tol:
                return 1 if r[1] > p[1] else -1
        raise AssertionError('no horizontal edge — not a lattice triangle')

    def trap_mode(long1, long2, s1, s2):
        """Majority triangle orientation of the half-hex (long side = the
        coarse-edge diameter, s1 adjacent to long1, s2 to long2)."""
        m = add(long1, long2); m = (m[0] / 2, m[1] / 2)
        tris = [tri_orient(long1, m, s1), tri_orient(m, s1, s2),
                tri_orient(m, long2, s2)]
        assert abs(sum(tris)) == 1, 'expected a 2:1 orientation split'
        return 1 if sum(tris) > 0 else -1

    # coarse patch: radius-2 hex disc (19 cells), edge 1, centres spacing √3
    centres = [add((1.5 * a, s3 * (a / 2 + b)), (0, 0))
               for a in range(-2, 3) for b in range(-2, 3)
               if max(abs(a), abs(b), abs(a + b)) <= 2]
    interior = {key(c) for c in centres
                if max(abs(round(c[0] / 1.5)),
                       abs(round(c[1] / s3 - c[0] / 3)),
                       abs(round(c[0] / 1.5) + round(c[1] / s3 - c[0] / 3))) <= 1}

    edge_kids = {}                                 # key(mid) -> [(host, k, e1, e2)]
    for c in centres:
        V = [add(c, cs(60 * k)) for k in range(6)]
        for k in range(6):
            e1, e2 = V[k], V[(k + 1) % 6]
            m = ((e1[0] + e2[0]) / 2, (e1[1] + e2[1]) / 2)
            edge_kids.setdefault(key(m), []).append((key(c), c, k, e1, e2, m))

    ok, owners, owned_count = True, {}, {kc: 1 for kc in (key(c) for c in centres)}
    host_modes = {}                                # key(c) -> {k: mode}
    for km, hosts in edge_kids.items():
        if len(hosts) != 2:
            continue                               # patch-boundary child
        modes = {}
        for kc, c, k, e1, e2, m in hosts:
            fv = [add(m, cs(60 * j), 0.5) for j in range(6)]
            on = [v for v in fv if key(v) in (key(e1), key(e2))]
            assert len(on) == 2, 'coarse edge is not a fine main diagonal'
            n = ((m[0] - c[0]), (m[1] - c[1]))     # outward normal (unnormalised)
            inside = [v for v in fv
                      if (v[0] - m[0]) * n[0] + (v[1] - m[1]) * n[1] < -tol]
            assert len(inside) == 2
            if math.dist(inside[0], on[0]) > math.dist(inside[1], on[0]):
                inside.reverse()                   # s1 adjacent to on[0]
            modes[kc] = trap_mode(on[0], on[1], inside[0], inside[1])
            host_modes.setdefault(kc, {})[k] = modes[kc]
        (ka, ma), (kb, mb) = modes.items()
        if ma == mb:
            print(f'  FAIL edge child {km}: halves share mode {ma}')
            ok = False
        winner = ka if ma == 1 else kb             # mode-0 := apex-up majority
        owners[km] = winner
        owned_count[winner] = owned_count.get(winner, 0) + 1

    n_int = 0
    for kc, per_edge in host_modes.items():
        if kc not in interior or len(per_edge) != 6:
            continue
        n_int += 1
        seq = [per_edge[k] for k in range(6)]
        if any(seq[k] == seq[(k + 1) % 6] for k in range(6)):
            print(f'  FAIL host {kc}: host-side modes do not alternate: {seq}')
            ok = False
        if owned_count[kc] != 4:
            print(f'  FAIL host {kc}: owns {owned_count[kc]} != 1+3 = 4')
            ok = False
    print(f'e4h mode-0 ownership: {n_int} interior hosts, '
          f'{len(owners)} bisected edge children — halves opposite-mode, '
          f'host-side modes alternate, each host owns 1+3=4   '
          f'{"PASS" if ok else "FAIL"}')
    return ok


def check_triad_closure(tol=1e-9):
    """Triad closure lemma (Ben, 2026-07-31): three half-hexes onto a
    triangle give uniform edge enumeration, and the vertex twist that
    closes the octahedron is a symmetry of that decomposition.

      1. TRIAD  — the side-3 triangle (9 t_cells) is tiled by three
                  half-hex trapezoids related by ±120° rotation about
                  the centroid; exactly one long side per triangle
                  edge; the dissection exists in exactly two mirror
                  chiralities (cf. the note's two-chirality lemma).
      2. EDGE   — reflecting the face across any edge line (the hinge
                  development) unites the two edge-trapezoids into an
                  exact hexagon (the straddling edge child), the same
                  way on all three edges: uniform edge enumeration.
      3. VERTEX — the four hinge reflections around an octahedral
                  vertex compose to rotation by 240° = −120°, i.e. the
                  triad rotation itself (orientation-preserving, even
                  count); it maps the developed alternating-chirality
                  triads onto each other (face i → face i+2) and the
                  re-entrant face 5 lands exactly on rot240(face 1) —
                  enumeration is single-valued around the cone.
                  Corollary: FIVE hinges (icosahedral vertex) compose
                  to an orientation-REVERSING map — '5 is odd', no
                  consistent chirality/mode parity.
    """
    import numpy as np
    from shapely.geometry import LineString, Polygon
    from shapely.ops import unary_union
    s3 = math.sqrt(3)

    def rot(p, th, o=(0.0, 0.0)):
        c, s = math.cos(math.radians(th)), math.sin(math.radians(th))
        p = np.asarray(p, float) - o
        return np.column_stack([p[:, 0] * c - p[:, 1] * s,
                                p[:, 0] * s + p[:, 1] * c]) + o

    def refl(p, th):
        c, s = math.cos(math.radians(2 * th)), math.sin(math.radians(2 * th))
        p = np.asarray(p, float)
        return np.column_stack([p[:, 0] * c + p[:, 1] * s,
                                p[:, 0] * s - p[:, 1] * c])

    def refl_line(p, a, b):
        a = np.asarray(a, float)
        d = np.asarray(b, float) - a
        d /= np.linalg.norm(d)
        q = np.asarray(p, float) - a
        return a + 2 * np.outer(q @ d, d) - q

    def same(ps, qs):
        pol = [Polygon(q) for q in qs]
        return all(min(Polygon(p).symmetric_difference(q).area
                       for q in pol) < tol for p in ps)

    VA, VB, VC = (0.0, 0.0), (3.0, 0.0), (1.5, 1.5 * s3)
    tri = Polygon([VA, VB, VC])
    G = (1.5, 0.5 * s3)
    edges = [(VA, VB), (VB, VC), (VC, VA)]

    def triad(t0):
        t0 = np.asarray(t0, float)
        return [t0, rot(t0, 120, G), rot(t0, 240, G)]

    def tiles(ps):
        polys = [Polygon(p) for p in ps]
        gap = tri.symmetric_difference(unary_union(polys)).area
        ov = max(polys[i].intersection(polys[j]).area
                 for i in range(3) for j in range(i + 1, 3))
        return gap < tol and ov < tol

    def on_edge(p, e):
        """Both endpoints of the piece's long side (verts 0-1) on edge e."""
        from shapely.geometry import Point
        return max(LineString(e).distance(Point(v)) for v in p[:2]) < tol

    # 1. TRIAD — both chiralities tile; mirror-related; one long side/edge
    TA = triad([(0, 0), (2, 0), (1.5, s3 / 2), (0.5, s3 / 2)])
    TB = triad([(1, 0), (3, 0), (2.5, s3 / 2), (1.5, s3 / 2)])
    ok = tiles(TA) and tiles(TB)
    ok &= same([np.column_stack([3 - p[:, 0], p[:, 1]]) for p in TA], TB)
    for T in (TA, TB):
        used = [i for p in T for i, e in enumerate(edges) if on_edge(p, e)]
        ok &= sorted(used) == [0, 1, 2]
    print(f'triad     : side-3 triangle = 3 half-hexes (±120° about '
          f'centroid), both chiralities, one long side per edge   '
          f'{"PASS" if ok else "FAIL"}')

    # 2. EDGE — hinge development completes the edge child, all 3 edges
    ok2 = True
    for a, b in edges:
        piece = next(p for p in TA if on_edge(p, (a, b)))
        halves = [Polygon(piece), Polygon(refl_line(piece, a, b))]
        pair = unary_union(halves)
        ok2 &= (pair.convex_hull.symmetric_difference(pair).area < tol
                and abs(pair.area - 1.5 * s3) < tol
                and halves[0].intersection(halves[1]).area < tol)
    ok &= ok2
    print(f'edge      : hinge development completes an exact hexagon on '
          f'all 3 edges (uniform enumeration)   '
          f'{"PASS" if ok2 else "FAIL"}')

    # 3. VERTEX — 4 hinges = the triad rotation; single-valued closure
    F = [TA]
    for k in range(4):
        F.append([refl(p, 60 * (k + 1)) for p in F[-1]])
    v = np.eye(2)                                 # rows = mapped basis
    for k in range(4):
        v = refl(v, 60 * (k + 1))
    R240 = np.array([[math.cos(math.radians(240)),
                      -math.sin(math.radians(240))],
                     [math.sin(math.radians(240)),
                      math.cos(math.radians(240))]])
    ok3 = np.allclose(v, R240.T, atol=1e-12)      # v.T is the net map
    ok3 &= same(F[2], [rot(p, 120) for p in F[0]])          # face 3 = rot120(f1)
    ok3 &= same(F[3], [rot(p, 120) for p in F[1]])          # face 4 = rot120(f2)
    ok3 &= same(F[4], [rot(p, 240) for p in F[0]])          # re-entry single-valued
    w = np.eye(2)
    for k in range(5):
        w = refl(w, 60 * (k + 1))
    odd5 = np.linalg.det(np.asarray(w)) < 0
    ok &= ok3 and odd5
    print(f'vertex    : 4 hinges = rot 240° (triad rotation), faces map '
          f'i→i+2, re-entry exact; 5 hinges orientation-reversing '
          f'(icosahedron excluded)   '
          f'{"PASS" if ok3 and odd5 else "FAIL"}')
    return bool(ok)


def check_e4h_closure(depth=2, tol=1e-9):
    """E4H closure (2026-07-31): §4b's consequence (c) machine-checked —
    the triad/hinge/vertex closure survives INTO the aperture-4 tail.

    Placements of the canonical a4 half-hex (a4_halfhex_canonical) onto
    the triad trapezoids are anchored by the corner-end rule: each
    trapezoid long side has exactly one end on a face corner; that end
    is the anchor. With that rule fixed:

      1. TRIAD  — rot 120° about the face centroid carries placement k
                  onto placement k+1 EXACTLY (as similarity maps, all
                  three direct), so every descent level of every triad
                  piece maps cell-for-cell, digit-for-digit.
      2. EDGE   — the hinge-developed placement equals the hinge mirror
                  of the host placement; the depth-d dissections mate
                  exactly along the hinge (identical boundary segment
                  sets, no hanging nodes); conjugating to canonical
                  coords, the mirror is the trapezoid's own axis and
                  permutes digits by (1 3) — the wing-swap rule.
      3. VERTEX — around the octahedral vertex, rot 120° carries face
                  f's placements onto face f+2's and the re-entrant
                  face 5 equals rot 240° of face 1, per piece, as maps
                  — closure at ALL tail depths, not a sampled few.
    """
    import numpy as np
    from shapely.geometry import LineString, Point

    s3 = math.sqrt(3)
    T0, cmaps = a4_halfhex_canonical()
    z0 = T0[:, 0] + 1j * T0[:, 1]
    cmaps = [(complex(a), complex(b)) for a, b in cmaps]
    zsrc = np.array([z0[3], z0[0], z0[1], z0[2]])   # path from long-side end

    def rot(p, th, o=(0.0, 0.0)):
        c, s = math.cos(math.radians(th)), math.sin(math.radians(th))
        p = np.asarray(p, float) - o
        return np.column_stack([p[:, 0] * c - p[:, 1] * s,
                                p[:, 0] * s + p[:, 1] * c]) + o

    def refl(p, th):
        c, s = math.cos(math.radians(2 * th)), math.sin(math.radians(2 * th))
        p = np.asarray(p, float)
        return np.column_stack([p[:, 0] * c + p[:, 1] * s,
                                p[:, 0] * s - p[:, 1] * c])

    def refl_line(p, a, b):
        a = np.asarray(a, float)
        d = np.asarray(b, float) - a
        d /= np.linalg.norm(d)
        q = np.asarray(p, float) - a
        return a + 2 * np.outer(q @ d, d) - q

    G = (1.5, 0.5 * s3)
    TRI = np.array([(0.0, 0.0), (3.0, 0.0), (1.5, 1.5 * s3)])

    def triad(t0):
        t0 = np.asarray(t0, float)
        return [t0, rot(t0, 120, G), rot(t0, 240, G)]

    TA = triad([(0, 0), (2, 0), (1.5, s3 / 2), (0.5, s3 / 2)])

    def placement(quad, corners):
        """Corner-end-anchored similarity T0 -> quad; (a, b, refl?)."""
        q = quad[:, 0] + 1j * quad[:, 1]
        cz = corners[:, 0] + 1j * corners[:, 1]
        L = next(i for i in range(4)
                 if abs(abs(q[(i + 1) % 4] - q[i]) - 2) < tol)
        e = [L, (L + 1) % 4]
        anch = [i for i in e if np.min(np.abs(q[i] - cz)) < tol]
        assert len(anch) == 1, 'long side needs exactly one corner end'
        if anch[0] == L:                             # path from the anchor,
            idx = [(L + k) % 4 for k in range(4)]    # around the quad
        else:
            idx = [(L + 1 - k) % 4 for k in range(4)]
        path = q[idx]
        for rf in (False, True):
            src = np.conj(zsrc) if rf else zsrc
            a = (path[1] - path[0]) / (src[1] - src[0])
            b = path[0] - a * src[0]
            if np.max(np.abs(a * src + b - path)) < tol:
                return a, b, rf
        raise AssertionError('quad is not similar to the canonical T0')

    def apply(pl, z):
        a, b, rf = pl
        return a * (np.conj(z) if rf else z) + b

    def cells(pl, d):
        out = []
        import itertools as it
        for digs in it.product(range(4), repeat=d):
            z = z0
            for dg in reversed(digs):
                a, b = cmaps[dg]
                z = a * z + b
            out.append((digs, apply(pl, z)))
        return out

    def map_eq(f, pl1, pl2):
        return max(abs(f(apply(pl1, v)) - apply(pl2, v)) for v in z0) < tol

    gz = complex(*G)
    w120 = np.exp(2j * np.pi / 3)
    r120 = lambda z: gz + w120 * (z - gz)

    # 1. TRIAD — rot120 carries placement k -> k+1, all direct
    P = [placement(TA[k], TRI) for k in range(3)]
    ok = all(not p[2] for p in P)
    ok &= all(map_eq(r120, P[k], P[(k + 1) % 3]) for k in range(3))
    print(f'e4h triad : rot 120° carries a4 placements k→k+1 exactly '
          f'(all direct)   {"PASS" if ok else "FAIL"}')

    # 2. EDGE — hinge mirror placement; exact mating; digit rule (1 3)
    ok2 = True
    edges = [(TRI[0], TRI[1]), (TRI[1], TRI[2]), (TRI[2], TRI[0])]
    for a, b in edges:
        host = next(t for t in TA
                    if max(LineString((a, b)).distance(Point(v))
                           for v in t[:2]) < tol)
        pl_h = placement(host, TRI)
        pl_m = placement(refl_line(host, a, b), refl_line(TRI, a, b))
        az, dz = complex(*a), complex(*b) - complex(*a)
        dz /= abs(dz)
        mirr = lambda z: az + dz * np.conj((z - az) / dz)
        ok2 &= map_eq(mirr, pl_h, pl_m)
        segs = []
        for pl in (pl_h, pl_m):
            ss = set()
            for _, poly in cells(pl, depth):
                for i in range(4):
                    u, v = poly[i], poly[(i + 1) % 4]
                    if max(abs((u - az) / dz - np.real((u - az) / dz)),
                           abs((v - az) / dz - np.real((v - az) / dz))) < tol:
                        t0_, t1_ = sorted((round(np.real((u - az) / dz), 6),
                                           round(np.real((v - az) / dz), 6)))
                        ss.add((t0_, t1_))
            segs.append(ss)
        ok2 &= segs[0] == segs[1]
    cent = [np.mean(a * z0 + b) for a, b in cmaps]   # canonical mirror =
    perm = [int(np.argmin([abs(np.conj(cent[i]) - cent[j])   # T0's own axis
                           for j in range(4)])) for i in range(4)]
    ok2 &= perm == [0, 3, 2, 1]                      # digits: (0)(2)(1 3)
    ok &= ok2
    print(f'e4h edge  : hinge placement = mirror of host, depth-{depth} '
          f'dissections mate exactly on all 3 hinges, canonical digit '
          f'rule (1 3)   {"PASS" if ok2 else "FAIL"}')

    # 3. VERTEX — closure at all tail depths, per piece, as maps
    F, C = [TA], [TRI]
    for k in range(4):
        F.append([refl(p, 60 * (k + 1)) for p in F[-1]])
        C.append(refl(C[-1], 60 * (k + 1)))
    PL = [[placement(F[f][k], C[f]) for k in range(3)] for f in range(5)]
    flags = [PL[f][0][2] for f in range(5)]
    ok3 = all(PL[f][k][2] == flags[f] for f in range(5) for k in range(3))
    ok3 &= flags == [False, True, False, True, False]
    w = np.exp(2j * np.pi / 3)
    ok3 &= all(map_eq(lambda z: w * z, PL[f][k], PL[f + 2][k])
               for f in (0, 1) for k in range(3))
    ok3 &= all(map_eq(lambda z: w * w * z, PL[0][k], PL[4][k])
               for k in range(3))
    ok &= ok3
    print(f'e4h vertex: rot 120° carries face f→f+2 per piece, re-entry '
          f'= rot 240°, chirality alternates — closure at all tail '
          f'depths   {"PASS" if ok3 else "FAIL"}')
    return bool(ok)


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
    passed &= check_e4h_mode_ownership()
    passed &= check_triad_closure()
    passed &= check_e4h_closure()
    if args.weights:
        passed &= check_weights(args.res, 2, 12, args.seed)
        passed &= check_weights(args.res, 4, 4, args.seed)
    raise SystemExit(0 if passed else 1)
