# Part of the Hex9 (H9) Project
# Copyright ©2026, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""Figure for E4H closure — enumeration matching through the a4 tail
(docs/dggs-transport-tilings.md §4b consequence (c), machine-checked by
transport_check.check_e4h_closure). Labels show the DIRECTION CLASSES
(c1..c3, centre = 0) carried by the placements; production tail digits
map each class through the five-symbol enumeration of §4d (digit =
the octahedral axis of the neighbour octant the class points at).

Three panels, one per check component:

  1. TRIAD  — the a4 dissection drawn in every triad trapezoid; the
              120° triad rotation carries the classes k→k+1 exactly.
  2. EDGE   — hinge development mirrors the host placement; depth-2
              dissections mate exactly along the hinge. Read in one
              frame, the wing classes SWAP under the reflection while
              the hinge-parallel class survives — the panel displays
              the very mechanism that makes class digits host-local
              (§4c) and motivates the five-symbol rule (§4d).
  3. VERTEX — around the octahedral vertex the classes are carried
              face f→f+2 by rot 120°, re-entry = rot 240°, chirality
              alternating — closure at all tail depths.

Run:  python docs/dggs/e4h_closure.py   ->  docs/dggs/e4h_closure.png
"""
import itertools
import math
from pathlib import Path

import numpy as np

from transport_check import a4_halfhex_canonical

S3 = math.sqrt(3)
# Fill colour per canonical PIECE INDEX (0 = centre half-child, 1..3 =
# edge-child halves in a4_halfhex_canonical order). CVD-safe (r/g-safe):
# blue/gold families only, neighbours separated by lightness.
DIG = ['#2a5fa5',   # piece 0 · mid blue
       '#DDAA33',   # piece 1 · bright gold
       '#b7d0ee',   # piece 2 · light blue
       '#a97f14']   # piece 3 · dark gold
RED = '#BB5566'     # rotation arrows, re-entry dashes, the apex star
DARK = '#1a1a1a'    # cell edges and face outlines
TOL = 1e-9

T0, CMAPS = a4_halfhex_canonical()
Z0 = T0[:, 0] + 1j * T0[:, 1]
CMAPS = [(complex(a), complex(b)) for a, b in CMAPS]
ZSRC = np.array([Z0[3], Z0[0], Z0[1], Z0[2]])


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


G = (1.5, 0.5 * S3)
TRI = np.array([(0.0, 0.0), (3.0, 0.0), (1.5, 1.5 * S3)])


def triad(t0):
    t0 = np.asarray(t0, float)
    return [t0, rot(t0, 120, G), rot(t0, 240, G)]


TA = triad([(0, 0), (2, 0), (1.5, S3 / 2), (0.5, S3 / 2)])


def placement(quad, corners):
    """Corner-end-anchored similarity T0 -> quad (see check_e4h_closure)."""
    q = quad[:, 0] + 1j * quad[:, 1]
    cz = corners[:, 0] + 1j * corners[:, 1]
    L = next(i for i in range(4)
             if abs(abs(q[(i + 1) % 4] - q[i]) - 2) < TOL)
    anch = [i for i in (L, (L + 1) % 4) if np.min(np.abs(q[i] - cz)) < TOL]
    if anch[0] == L:
        idx = [(L + k) % 4 for k in range(4)]
    else:
        idx = [(L + 1 - k) % 4 for k in range(4)]
    path = q[idx]
    for rf in (False, True):
        src = np.conj(ZSRC) if rf else ZSRC
        a = (path[1] - path[0]) / (src[1] - src[0])
        b = path[0] - a * src[0]
        if np.max(np.abs(a * src + b - path)) < TOL:
            return a, b, rf
    raise AssertionError('quad not similar to canonical T0')


def cells(pl, depth):
    a, b, rf = pl
    out = []
    for digs in itertools.product(range(4), repeat=depth):
        z = Z0
        for dg in reversed(digs):
            aa, bb = CMAPS[dg]
            z = aa * z + bb
        out.append((digs, a * (np.conj(z) if rf else z) + b))
    return out


def _cls_label(digs, z):
    """Direction-class label of a cell: '0' for the centre chain, else
    c1..c3 by the long side's face-frame direction (z runs v0..v3 with
    the long side v3->v0)."""
    if digs[-1] == 0:
        return '0'
    e = z[0] - z[3]
    th = math.degrees(math.atan2(e.imag, e.real)) % 180
    return f'c{int(round(th / 60.0)) % 3 + 1}'


def draw_cells(ax, pl, depth, alpha=1.0, label=False, lw=0.8, fill=True):
    from matplotlib.patches import Polygon as MP
    for digs, z in cells(pl, depth):
        v = np.column_stack([z.real, z.imag])
        ax.add_patch(MP(v, closed=True,
                        fc=DIG[digs[0]] if fill else 'none',
                        ec=DARK, lw=lw, alpha=alpha))
        if label:
            c = z.mean()
            ax.annotate(_cls_label(digs, z), (c.real, c.imag),
                        fontsize=9, ha='center', va='center', color='k',
                        bbox=dict(fc='white', alpha=0.55, ec='none',
                                  pad=0.8))


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Polygon as MP

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.6),
                             width_ratios=[1.0, 1.0, 1.3])

    # ---------------- 1. TRIAD ----------------
    ax = axes[0]
    for k in range(3):
        pl = placement(TA[k], TRI)
        draw_cells(ax, pl, 1, label=(k == 0))
    ax.add_patch(MP(TRI, closed=True, fc='none', ec=DARK, lw=2.2))
    g = np.asarray(G)
    ax.add_patch(FancyArrowPatch(g + (0.62, 0.05), g + (-0.36, 0.52),
                                 connectionstyle='arc3,rad=0.55',
                                 arrowstyle='-|>', mutation_scale=16,
                                 color=RED, lw=1.6))
    ax.annotate('120°', g + (0.55, 0.55), color=RED, fontsize=11,
                ha='center')
    ax.set_title('1 · the a4 dissection rides the triad:\n'
                 'rot 120° carries the classes k→k+1 exactly', fontsize=11.5)

    # ---------------- 2. EDGE ----------------
    ax = axes[1]
    host = TA[0]
    a, b = TRI[0], TRI[1]
    pl_h = placement(host, TRI)
    pl_m = placement(refl_line(host, a, b), refl_line(TRI, a, b))
    draw_cells(ax, pl_h, 1, label=True)
    draw_cells(ax, pl_m, 1, label=True, alpha=0.75)
    draw_cells(ax, pl_h, 2, fill=False, lw=0.4)
    draw_cells(ax, pl_m, 2, fill=False, lw=0.4)
    ax.plot([a[0] - 0.3, b[0] + 0.3], [a[1], b[1]], color=DARK, lw=1.0,
            ls='--')
    ax.annotate('hinge', (2.55, 0.07), fontsize=9, color='0.35')
    ax.annotate('mating is exact; in one frame the wing classes SWAP\n'
                'under reflection (c2|c3) while the hinge-parallel class\n'
                'survives (c1|c1) — why class digits are host-local (§4c)\n'
                'and the five-symbol rule exists (§4d)',
                (1.0, -1.75), ha='center', fontsize=9.5)
    ax.set_title('2 · hinge development mates the dissections\n'
                 'exactly at every depth (depth 2 outlined)', fontsize=11.5)

    # ---------------- 3. VERTEX ----------------
    ax = axes[2]
    F, C = [TA], [TRI]
    for k in range(4):
        F.append([refl(p, 60 * (k + 1)) for p in F[-1]])
        C.append(refl(C[-1], 60 * (k + 1)))
    for f in range(4):
        for k in range(3):
            draw_cells(ax, placement(F[f][k], C[f]), 1)
        ax.add_patch(MP(C[f], closed=True, fc='none', ec=DARK, lw=2.0))
        lab = rot(np.array([(2.35, 1.25)]), 60 * f)[0]
        ax.annotate(f'f{f + 1}', lab, fontsize=11, ha='center', color=DARK,
                    bbox=dict(fc='white', alpha=0.7, ec='none', pad=1.0))
    for k in range(3):                                 # re-entry dashed
        for _, z in cells(placement(F[4][k], C[4]), 1):
            ax.add_patch(MP(np.column_stack([z.real, z.imag]), closed=True,
                            fc='none', ec=RED, lw=1.0, ls='--'))
    ax.plot(0, 0, marker='*', color=RED, ms=17, zorder=6)
    ax.annotate('f1 re-entry = rot 240° (f1),\nclass-for-class  ✓',
                (2.75, -2.0), fontsize=10, ha='center', color=RED)
    ax.set_title('3 · vertex closure carries the enumeration:\n'
                 'faces f→f+2 by rot 120°, chirality alternating',
                 fontsize=11.5)

    for ax in axes:
        ax.set_aspect('equal')
        ax.axis('off')
        ax.relim()
        ax.autoscale_view()
    fig.suptitle('E4H closure — enumeration matching through the a4 tail '
                 '(machine-verified by transport_check.check_e4h_closure)',
                 fontsize=13)
    out = Path(__file__).with_name('e4h_closure.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print('->', out)


if __name__ == '__main__':
    main()
