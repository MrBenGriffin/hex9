# Part of the Hex9 (H9) Project
# Copyright ©2026, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""Figure for the triad closure lemma (docs/dggs-transport-tilings.md §4b).

Three panels, one per component of check_triad_closure:

  1. TRIAD  — the side-3 triangle tiled by three half-hexes related by
              ±120° rotation about the centroid; both chiralities; one
              trapezoid long side per triangle edge.
  2. EDGE   — hinge development (reflection across an edge line)
              completes each edge trapezoid to an exact hexagon — the
              straddling edge child — identically on all three edges.
  3. VERTEX — four hinge reflections around the octahedral vertex
              compose to rot 240° = −120°, the triad rotation; faces
              alternate chirality (2-colouring) and the re-entrant
              fifth face lands exactly on rot 240° of face 1.

Run:  python docs/dggs/triad_closure.py   ->  docs/dggs/triad_closure.png
"""
import math
from pathlib import Path

import numpy as np

S3 = math.sqrt(3)
# Tol high-contrast family + tints (CVD-safe: blue vs gold, luminance ramps)
BLUES = ['#2a5fa5', '#6f9fd8', '#b7d0ee']
GOLDS = ['#a97f14', '#DDAA33', '#f0d488']
RED, DARK = '#BB5566', '#1a1a1a'


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
TB = triad([(1, 0), (3, 0), (2.5, S3 / 2), (1.5, S3 / 2)])


def draw_triad(ax, T, org, cols, lw_long=3.2, alpha=1.0):
    from matplotlib.patches import Polygon as MP
    org = np.asarray(org, float)
    for p, c in zip(T, cols):
        q = p + org
        ax.add_patch(MP(q, closed=True, fc=c, ec=DARK, lw=0.9, alpha=alpha))
        ax.plot(q[:2, 0], q[:2, 1], color=DARK, lw=lw_long,
                solid_capstyle='round', alpha=alpha)


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Polygon as MP

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.4),
                             width_ratios=[1.15, 1.0, 1.25])

    # ---------------- 1. TRIAD, both chiralities ----------------
    ax = axes[0]
    for T, org, lab in ((TA, (0, 0), 'chirality A'),
                        (TB, (4.0, 0), 'chirality B (mirror)')):
        draw_triad(ax, T, org, BLUES if org[0] == 0 else GOLDS)
        g = np.asarray(org) + G
        ax.plot(*g, marker='o', color=DARK, ms=4)
        ax.add_patch(FancyArrowPatch(g + (0.62, 0.05), g + (-0.36, 0.52),
                                     connectionstyle='arc3,rad=0.55',
                                     arrowstyle='-|>', mutation_scale=16,
                                     color=RED, lw=1.6))
        ax.annotate('120°', g + (0.18, 0.62), color=RED, fontsize=11,
                    ha='center')
        ax.annotate(lab, np.asarray(org) + (1.5, -0.42), ha='center',
                    fontsize=11)
    ax.set_title('1 · side-3 triangle = three half-hexes\n'
                 '(±120° about centroid; long sides, one per edge, bold)',
                 fontsize=11.5)
    ax.set_xlim(-0.4, 7.4)

    # ---------------- 2. EDGE development ----------------
    ax = axes[1]
    draw_triad(ax, TA, (0, 0), BLUES)
    R = [refl_line(p, TRI[0], TRI[1]) for p in TA]     # develop bottom hinge
    for p in R[1:]:
        ax.add_patch(MP(p, closed=True, fc='none', ec='0.62', lw=0.9))
    ax.add_patch(MP(R[0], closed=True, fc=GOLDS[1], ec=DARK, lw=0.9,
                    alpha=0.9))
    hexa = np.vstack([TA[0], R[0]])
    hull = hexa[np.argsort(np.arctan2(hexa[:, 1] - 0, hexa[:, 0] - 1.0))]
    ax.add_patch(MP(hull, closed=True, fc='none', ec=RED, lw=2.6))
    ax.annotate('edge child completed\n(opposite-mode pair)', (1.0, -1.35),
                ha='center', fontsize=10, color=DARK)
    ax.plot([TRI[0][0], TRI[1][0]], [TRI[0][1], TRI[1][1]],
            color=DARK, lw=1.0, ls='--')
    ax.annotate('hinge', (2.62, -0.16), fontsize=9, color='0.35')
    ax.set_title('2 · hinge development completes an exact hexagon\n'
                 '(identical on all three edges)', fontsize=11.5)

    # ---------------- 3. VERTEX closure ----------------
    ax = axes[2]
    F, cols = [TA], [BLUES]
    for k in range(4):
        F.append([refl(p, 60 * (k + 1)) for p in F[-1]])
        cols.append(GOLDS if cols[-1] is BLUES else BLUES)
    for k in range(4):                                 # faces 1..4, solid
        draw_triad(ax, F[k], (0, 0), cols[k], lw_long=2.2)
        lab = rot(np.array([(1.9, 1.0)]), 60 * k)[0]
        ax.annotate(f'f{k + 1}', lab, fontsize=11, ha='center',
                    color=DARK)
    for p in F[4]:                                     # re-entry, dashed
        ax.add_patch(MP(p, closed=True, fc='none', ec=RED, lw=1.7,
                        ls='--'))
    ax.annotate('f1 re-entry\n= rot 240° (f1)  ✓', (2.45, -2.15),
                fontsize=10, ha='center', color=RED)
    ax.plot(0, 0, marker='*', color=RED, ms=17, zorder=6)
    ax.add_patch(FancyArrowPatch((1.15, -0.62), (-0.1, -1.28),
                                 connectionstyle='arc3,rad=-0.45',
                                 arrowstyle='-|>', mutation_scale=16,
                                 color=RED, lw=1.6))
    ax.annotate('4 hinges = −120°\n(the triad rotation)', (1.9, -1.5),
                fontsize=10, color=RED, ha='center')
    ax.set_title('3 · four hinge reflections close the octahedral vertex\n'
                 '(faces alternate chirality; 5 hinges would reverse '
                 'orientation)', fontsize=11.5)

    for ax in axes:
        ax.set_aspect('equal')
        ax.axis('off')
        ax.relim()
        ax.autoscale_view()
    fig.suptitle('The triad closure lemma — machine-verified by '
                 'transport_check.check_triad_closure (§4b)', fontsize=13)
    out = Path(__file__).with_name('triad_closure.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print('->', out)


if __name__ == '__main__':
    main()
