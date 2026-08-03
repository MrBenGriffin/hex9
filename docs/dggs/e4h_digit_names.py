# Part of the Hex9 (H9) Project
# Copyright ©2026, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""Figure for E4H digit naming — the three-act journey
(docs/dggs-transport-tilings.md §4c-§4d).

  1. POSITIONAL — the first cycle (0 centre, 2 axis, 1/3 wings): the
     three centre-touching fine hexagons are wing|axis pairs (1|2) —
     mixed, ambiguous, structurally unfixable (the pinwheel puts
     adjacent cuts 120° apart).
  2. CLASSES — digit = direction class of the child's coarse edge:
     every fine hexagon on THIS FACE is one whole digit and the
     centre trio is {1,2,3}. Honesty note: on the sphere this scheme
     matches only within a host — cross-host pairs disagree totally
     (the §4c correction; adjacent hosts' state rings rotate with c2).
  3. FIVE SYMBOLS — the promoted enumeration (§4d), rendered on the
     REAL sphere around the cone point through the h9e API: digit =
     the octahedral axis of the neighbour octant the class points at
     (oid_nb); matched pairs are GLOBAL — solid hexagons across every
     host border, octant seam, and the vertex itself.

Run:  python docs/dggs/e4h_digit_names.py -> docs/dggs/e4h_digit_names.png
"""
import math
from pathlib import Path

import numpy as np

from e4h_closure import TA, TRI, cells, placement, refl, refl_line

S3 = math.sqrt(3)

# ---------------------------------------------------------------- colours
# CVD-safe scheme (r/g-safe): separation rides on LUMINANCE and the
# blue<->yellow axis only — never red-vs-green, never warm-on-warm at
# equal lightness. Base hues are the Tol high-contrast blue/gold pair.
#
# DIG: panels 1-2, indexed by the OLD digit alphabets (positional 0..3
# in panel 1, class 0..3 in panel 2). To play: keep neighbours in the
# list apart in lightness, and alternate blue-family / gold-family.
DIG = ['#2a5fa5',   # 0 · mid blue
       '#DDAA33',   # 1 · bright gold
       '#b7d0ee',   # 2 · light blue
       '#a97f14']   # 3 · dark gold ("the brown")
RED = '#BB5566'     # annotations, mismatch outlines, the cone-point star
DARK = '#1a1a1a'    # cell edges and the face outline

G = (1.5, 0.5 * S3)       # face centroid (the triad's rotation centre)
TOL = 1e-6


def face_pieces(quads, corners):
    out = []
    for k, q in enumerate(quads):
        for digs, z in cells(placement(q, corners), 1):
            out.append((digs[0], z))
    return out


def longside(z):
    for i in range(4):
        u, v = z[i], z[(i + 1) % 4]
        if abs(abs(u - v) - 1.0) < TOL:
            pts = sorted([(round(u.real, 4), round(u.imag, 4)),
                          (round(v.real, 4), round(v.imag, 4))])
            return tuple(pts)
    raise AssertionError


def assemble(pieces):
    """Pair half-hexes along long sides -> whole fine hexagons with
    positional pair digits and the class digit."""
    groups = {}
    for d, z in pieces:
        groups.setdefault(longside(z), []).append((d, z))
    out = []
    for key, ps in groups.items():
        if len(ps) != 2:
            continue
        (d1, z1), (d2, z2) = ps
        pts = np.concatenate([z1, z2])
        uniq = []
        for p in pts:
            if all(abs(p - u) > TOL for u in uniq):
                uniq.append(p)
        c = np.mean(uniq)
        uniq = sorted(uniq, key=lambda p: np.angle(p - c))
        if d1 == 0 and d2 == 0:
            dig = 0
        else:
            (p0, p1) = key
            ang = math.degrees(math.atan2(p1[1] - p0[1],
                                          p1[0] - p0[0])) % 180
            dig = 1 + int(round(ang / 60)) % 3
        out.append((np.array(uniq), c, dig, (d1, d2)))
    return out


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MP

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 6.6),
                             width_ratios=[1.0, 1.0, 1.35])

    edges = [(TRI[0], TRI[1]), (TRI[1], TRI[2]), (TRI[2], TRI[0])]
    face = face_pieces(TA, TRI)
    devs = []
    for a, b in edges:
        devs += face_pieces([refl_line(t, a, b) for t in TA],
                            refl_line(TRI, a, b))

    # ---------------- 1. BEFORE: positional halves ----------------
    ax = axes[0]
    for d, z in face:
        ax.add_patch(MP(np.column_stack([z.real, z.imag]), closed=True,
                        fc=DIG[d], ec=DARK, lw=0.8))
        c = z.mean()
        ax.annotate(str(d), (c.real, c.imag), fontsize=9, ha='center',
                    va='center', color='k',
                    bbox=dict(fc='white', alpha=0.55, ec='none', pad=0.7))
    for poly, c, dig, (d1, d2) in assemble(face):
        if d1 != d2:
            ax.add_patch(MP(np.column_stack([poly.real, poly.imag]),
                            closed=True, fc='none', ec=RED, lw=2.6))
    ax.annotate('the centre trio: every one\na mixed wing|axis pair (1|2)',
                (1.5, -0.55), ha='center', fontsize=10, color=RED,
                annotation_clip=False)
    ax.add_patch(MP(TRI, closed=True, fc='none', ec=DARK, lw=2.0))
    ax.set_title('1 · POSITIONAL — first cycle (0 centre, 2 axis,\n'
                 '1/3 wings): pairs mix, unfixably', fontsize=11.5)

    # ---------------- 2. AFTER: class digits, whole hexagons -------
    ax = axes[1]
    from shapely.geometry import Point, Polygon as SP
    tri_poly = SP(TRI).buffer(0.05)
    for poly, c, dig, _ in assemble(face + devs):
        if not tri_poly.contains(Point(c.real, c.imag)):
            continue
        ax.add_patch(MP(np.column_stack([poly.real, poly.imag]),
                        closed=True, fc=DIG[dig], ec=DARK, lw=0.9))
        ax.annotate(str(dig), (c.real, c.imag), fontsize=10, ha='center',
                    va='center', color='k',
                    bbox=dict(fc='white', alpha=0.55, ec='none', pad=0.7))
        if abs(c - complex(*G)) < 0.6:
            ax.add_patch(MP(np.column_stack([poly.real, poly.imag]),
                            closed=True, fc='none', ec=RED, lw=2.2))
    ax.annotate('centre trio now {1,2,3} — but this scheme only\n'
                'matches WITHIN a host on the sphere (§4c)',
                (1.5, -1.15), ha='center', fontsize=10, color=RED,
                annotation_clip=False)
    ax.add_patch(MP(TRI, closed=True, fc='none', ec=DARK, lw=2.0))
    ax.set_title('2 · CLASSES — digit = direction class of the\n'
                 'coarse edge: whole hexagons on this face', fontsize=11.5)

    # -------- 3. FIVE SYMBOLS: the real sphere, through the API -----
    # One deep encode is the whole panel: every pixel becomes a depth-2
    # E4H address, and everything drawn — colours, both line layers —
    # is read back off the address fields. Nothing is drawn as
    # geometry; all lines are GENERATED, i.e. they emerge wherever an
    # address field changes between neighbouring pixels.
    ax = axes[2]
    from hhg9 import Registrar
    from hhg9.h9.e4h import h9e_encode, h9e_split
    reg = Registrar()
    n, ext = 420, 0.11        # n: raster resolution; ext: half-width in
    g = np.linspace(-ext, ext, n)          # degrees around the cone point
    X, Y = np.meshgrid(g, g)
    us = h9e_encode(Y.ravel(), X.ravel(), 6, 2, reg)
    # Per unique leaf: host id, final digit, and the identity of its
    # depth-1 ANCESTOR HEXAGON. The hexagon identity uses a decode
    # probe at the canonical ORIGIN — the midpoint of the trapezoid's
    # long side — because BOTH halves of a fine hexagon map that point
    # to the same physical spot (the shared diameter midpoint, i.e.
    # the hexagon centre). Quantised, it is a true per-hexagon key,
    # host- and half-independent. (A digit-difference mask is NOT
    # enough here: S2 gives both halves' class-3 children the same
    # digit q, so distinct adjacent hexagons can share a digit and
    # their boundary would vanish — the "missing lines" Ben spotted.)
    from hhg9.h9.e4h import h9e_decode
    from hhg9.h9.uuid_address import (_batch_int_to_nibbles,
                                      batch_nibbles_to_int)
    import uuid as uuid_mod
    uniq = list({u.int: u for u in us}.values())
    pre1 = {}
    for u in uniq:                          # depth-2 leaf -> depth-1 prefix
        nib = _batch_int_to_nibbles([u.int], n=32)[0].copy()
        nib[6 + 4] = 0x0F                   # drop the last tail digit
        pre1[u.int] = uuid_mod.UUID(int=int(batch_nibbles_to_int(
            nib[None, :])[0]))
    p1u = list({p.int: p for p in pre1.values()}.values())
    cla, clo = h9e_decode(p1u, reg, _probe=0j)      # hexagon centres
    hexkey = {p.int: (round(a, 6), round(o, 6))
              for p, a, o in zip(p1u, cla, clo)}
    hxid = {}
    hid, cache = {}, {}
    dn = np.zeros(len(us), int)            # final digit    (colours)
    hx = np.zeros(len(us), np.int32)       # depth-1 hexagon (grey lines)
    host = np.zeros(len(us), np.int32)     # host id        (border lines)
    for i, u in enumerate(us):
        if u.int not in cache:
            h, hf, dg = h9e_split(u)
            k = hexkey[pre1[u.int].int]
            cache[u.int] = (hid.setdefault(h.int, len(hid)), dg[-1],
                            hxid.setdefault(k, len(hxid)))
        host[i], dn[i], hx[i] = cache[u.int]

    # pal6: fill colour per FINAL tail digit (0..5). Same CVD rules as
    # DIG above; digit 0 is deliberately near-white so the centre
    # children read as "background" against both line greys and the
    # dark gold (mid-grey clashed with the brown). Note S4 in the
    # image: each octant omits its own axis digit, so at most four of
    # digits 1..5 appear in any one octant — expect the mix to shift
    # visibly as you cross the black borders.
    pal6 = np.array([[int(c[j:j + 2], 16) for j in (1, 3, 5)] for c in
                     ['#e8e8e8',    # 0 · centre child (near-white)
                      '#86b4f0',    # 1 · mid blue   (the anchor digit) #2a5fa5
                      '#f2d280',    # 2 · bright gold
                      '#b7d0ee',    # 3 · light blue
                      '#80f2d9',    # 4 · dark gold
                      '#ed9daa']],  # 5 · red
                    float) / 255
    rgb = pal6[dn].reshape(n, n, 3)
    H = host.reshape(n, n)
    HX = hx.reshape(n, n)

    # Generated lines, layer 1 — the intermediate depth-1 hexagons.
    # No polygons are computed: lines emerge wherever the hexagon
    # identity field changes between neighbouring pixels. The same
    # trick draws any intermediate level from one deep encode (build
    # the level-k prefix and probe its diameter midpoint). Their
    # dramatic curving near the apex is real — the cone squash acting
    # on the intermediate level.
    h1edge = np.zeros((n, n), bool)
    h1edge[:, 1:] |= HX[:, 1:] != HX[:, :-1]
    h1edge[1:, :] |= HX[1:, :] != HX[:-1, :]

    # Generated lines, layer 2 — host borders, same idea on the host id.
    hedge = np.zeros((n, n), bool)
    hedge[:, 1:] |= H[:, 1:] != H[:, :-1]
    hedge[1:, :] |= H[1:, :] != H[:-1, :]

    # Line greys, painted dark-over-light (host borders win where the
    # two layers coincide). Play freely: 0.0 = black, 1.0 = white.
    rgb[h1edge] = 0.50        # depth-1 hexagons: mid-dark grey
    rgb[hedge] = 0.05         # host borders: near-black
    ax.imshow(rgb, origin='lower', extent=[-ext, ext, -ext, ext])
    ax.plot(0, 0, marker='*', color=RED, ms=8, zorder=6)
    ax.set_title('3 · FIVE SYMBOLS — the promoted rule (§4d), real\n'
                 'sphere, cone point: globally matched (grey lines =\n'
                 'the intermediate depth-1 hexagons)', fontsize=11.5)

    for ax in axes[:2]:
        ax.set_aspect('equal')
        ax.axis('off')
        ax.relim()
        ax.margins(0.14)
        ax.autoscale_view()
    axes[2].set_aspect('equal')
    axes[2].axis('off')
    fig.suptitle('E4H digit naming, the three-act journey (§4c-§4d): '
                 'positional (mixed) → classes (host-local) → five '
                 'symbols (globally matched, derived from oid_nb)',
                 fontsize=13, y=1.02)
    out = Path(__file__).with_name('e4h_digit_names.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print('->', out)


if __name__ == '__main__':
    main()
