# Part of the Hex9 (H9) Project
# Copyright ©2026, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""Figure for E4H digit naming (docs/dggs-transport-tilings.md §4c).

Three panels:

  1. BEFORE — the positional cycle (0 centre, 2 axis, 1/3 wings): the
     three centre-touching fine hexagons are wing|axis pairs (1|2) —
     mixed, ambiguous, structurally unfixable (the pinwheel puts
     adjacent cuts 120° apart).
  2. AFTER  — class digits (digit = direction class of the child's
     coarse edge; 0 = centre child): every fine hexagon is one whole
     digit, the centre trio is {1,2,3}, and every fine vertex is
     rainbow.
  3. VERTEX — the same class digits around the developed octahedral
     cone point: matched pairs hold across every hinge and the
     enumeration closes; alternating faces name their classes via the
     (1 2) swap — the mode law (H9O.oid_mo).

Run:  python docs/dggs/e4h_digit_names.py -> docs/dggs/e4h_digit_names.png
"""
import math
from pathlib import Path

import numpy as np

from e4h_closure import TA, TRI, cells, placement, refl, refl_line

S3 = math.sqrt(3)
DIG = ['#2a5fa5', '#DDAA33', '#b7d0ee', '#a97f14']
RED, DARK = '#BB5566', '#1a1a1a'
G = (1.5, 0.5 * S3)
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
    ax.set_title('1 · BEFORE — positional cycle\n'
                 '(0 centre, 2 axis, 1/3 wings): pairs mix', fontsize=11.5)

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
    ax.annotate('centre trio now {1,2,3};\nevery fine vertex rainbow',
                (1.5, -1.15), ha='center', fontsize=10, color=RED,
                annotation_clip=False)
    ax.add_patch(MP(TRI, closed=True, fc='none', ec=DARK, lw=2.0))
    ax.set_title('2 · AFTER — class digits (digit = direction\n'
                 'class of the coarse edge): whole hexagons', fontsize=11.5)

    # ---------------- 3. VERTEX: class digits around the cone ------
    ax = axes[2]
    F, C = [TA], [TRI]
    for k in range(4):
        F.append([refl(p, 60 * (k + 1)) for p in F[-1]])
        C.append(refl(C[-1], 60 * (k + 1)))
    pieces = []
    for f in range(4):
        pieces += face_pieces(F[f], C[f])
    for poly, c, dig, (d1, d2) in assemble(pieces):
        ax.add_patch(MP(np.column_stack([poly.real, poly.imag]),
                        closed=True, fc=DIG[dig], ec=DARK, lw=0.8))
        if abs(c) < 2.2:
            ax.annotate(str(dig), (c.real, c.imag), fontsize=8,
                        ha='center', va='center', color='k',
                        bbox=dict(fc='white', alpha=0.5, ec='none',
                                  pad=0.5))
    for f in range(4):
        ax.add_patch(MP(C[f], closed=True, fc='none', ec=DARK, lw=2.0))
        lab = np.array([(2.55, 1.35)])
        th = math.radians(60 * f)
        R = np.array([[math.cos(th), -math.sin(th)],
                      [math.sin(th), math.cos(th)]])
        p = (R @ lab.T).T[0]
        ax.annotate(f'f{f + 1}: ' + ('φ = id' if f % 2 == 0
                                     else 'φ = (1 2)'),
                    p, fontsize=9.5, ha='center', color=DARK,
                    bbox=dict(fc='white', alpha=0.75, ec='none', pad=1.0),
                    annotation_clip=False)
    ax.plot(0, 0, marker='*', color=RED, ms=17, zorder=6)
    ax.set_title('3 · around the cone point: matched pairs across\n'
                 'every hinge; alternating faces use the (1 2) swap '
                 '(mode law)', fontsize=11.5)

    for ax in axes:
        ax.set_aspect('equal')
        ax.axis('off')
        ax.relim()
        ax.margins(0.14)
        ax.autoscale_view()
    fig.suptitle('E4H digit naming (§4c) — from positional cycle to c2 '
                 'class digits: every fine hexagon one digit, globally '
                 '(e4h_digit_csp.py: SAT, mode law canonical)',
                 fontsize=13)
    out = Path(__file__).with_name('e4h_digit_names.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print('->', out)


if __name__ == '__main__':
    main()
