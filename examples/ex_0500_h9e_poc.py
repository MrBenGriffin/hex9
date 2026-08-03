# Part of the Hex9 (H9) Project
# Copyright ©2026, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""Hex9+E4H proof of concept, end to end.

Encodes points to E4H addresses (hex9 to the attach layer, 0xE marker,
half digit, aperture-4 class digits), decodes them back, demonstrates
the matched-pairs property through the partner verb, and renders a
patch around the octahedral cone point (0, 0) coloured by tail digits
— the acid test, since every seam and the vertex itself are in frame.

Run:  python examples/ex_0500_h9e_poc.py          -> h9e_poc.png + console table
"""
import numpy as np

from hhg9 import Registrar
from hhg9.h9.e4h import (h9e_decode, h9e_encode, h9e_label,
                         h9e_partner_point, h9e_split)

LAYER = 6


def table(reg, depth):
    pts = [('Thimphu', 27.4728, 89.6390),
           ('Greenwich', 51.4779, 0.0),
           ('Quito', -0.1807, -78.4678),
           ('cone point +', 0.02, 0.02),
           ('cone point -', -0.02, -0.02)]
    us = h9e_encode([p[1] for p in pts], [p[2] for p in pts],
                    LAYER, depth, reg)
    la, lo = h9e_decode(us, reg)
    pl, po = h9e_partner_point(us, reg)
    pus = h9e_encode(pl, po, LAYER, depth, reg)
    print(f'{"name":14s} {"address":16s} {"partner":16s} rep')
    for (name, _, _), u, pu, a, o in zip(pts, us, pus, la, lo):
        print(f'{name:14s} {h9e_label(u):16s} {h9e_label(pu):16s} '
              f'({a:+.5f}, {o:+.5f})')
        h1, _, d1 = h9e_split(u)
        h2, _, d2 = h9e_split(pu)
        assert d1[-1] == d2[-1], 'matched pairs broken!'
    # Matched pairs are GLOBAL under the five-symbol enumeration
    # (transport note §4d): every fine hexagon's two halves share the
    # final digit — same host, cross host, seams and cone points.
    print('matched pairs: ALL partners share the final digit (global)')


def render(reg, depth, path=None):
    """Two readings of the same patch around the cone point (0, 0):
    left, the ADDRESS TREE (hue = half digit, lightness = first class
    digit — the three split hexagons per host straddle the host's cut,
    their halves in opposite subtrees, as a correct trapezoid tree
    must); right, the CELL ENUMERATION (final class digit only —
    matched pairs make every fine hexagon solid, and no cell borders
    are drawn: boundaries emerge from the rainbow property)."""
    path = f'output/h9e_{depth}poc.png' if path is None else path
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n, ext = 2100, 1/3  # want to capture non-cone environment here also.
    g = np.linspace(-ext, ext, n)
    X, Y = np.meshgrid(g, g)
    us = h9e_encode(Y.ravel(), X.ravel(), LAYER, depth, reg)
    kid, hid = {}, {}
    key = np.zeros(len(us), np.int32)
    half = np.zeros(len(us), int)
    d1 = np.zeros(len(us), int)
    dn = np.zeros(len(us), int)
    host = np.zeros(len(us), np.int32)
    cache = {}
    for i, u in enumerate(us):
        if u.int not in cache:
            h, hf, dg = h9e_split(u)
            cache[u.int] = (hid.setdefault(h.int, len(hid)), hf, dg[0],
                            dg[-1], kid.setdefault(u.int, len(kid)))
        host[i], half[i], d1[i], dn[i], key[i] = cache[u.int]

    blues = ['#08213f', '#0b2e5a', '#2a5fa5', '#6f9fd8', '#a8c4e6',
             '#d8e4f4']
    golds = ['#3f2f05', '#6b4e0a', '#a97f14', '#DDAA33', '#ecc96f',
             '#f6e3ae']
    pal12 = np.array([[int(c[j:j + 2], 16) for j in (1, 3, 5)]
                      for c in blues + golds], float) / 255
    pal6 = np.array([[int(c[j:j + 2], 16) for j in (1, 3, 5)] for c in
                     ['#888888', '#2a5fa5', '#DDAA33', '#b7d0ee',
                      '#a97f14', '#BB5566']], float) / 255
    K, H = key.reshape(n, n), host.reshape(n, n)
    edge = np.zeros((n, n), bool)
    edge[:, 1:] |= K[:, 1:] != K[:, :-1]   # class-key change
    edge[1:, :] |= K[1:, :] != K[:-1, :]
    hedge = np.zeros((n, n), bool)
    hedge[:, 1:] |= H[:, 1:] != H[:, :-1]  # host-cell change
    hedge[1:, :] |= H[1:, :] != H[:-1, :]

    fig, axes = plt.subplots(1, 2, figsize=(18, 9.2), dpi=280)
    rgb = pal12[(half * 6 + d1)].reshape(n, n, 3)
    rgb[edge] = 0.55
    rgb[hedge] = 0.05
    axes[0].imshow(rgb, origin='lower', extent=[-ext, ext, -ext, ext])
    axes[0].set_title('address tree — hue = half digit, lightness = '
                      'first tail digit;\nsplit hexagons = pairs '
                      'straddling each host cut (two cells, one child)',
                      fontsize=11)
    rgb = pal6[dn].reshape(n, n, 3)
    rgb[hedge] = 0.05
    axes[1].imshow(rgb, origin='lower', extent=[-ext, ext, -ext, ext])
    axes[1].set_title('cell enumeration — final digit (five-symbol rule, '
                      '§4d);\nmatched pairs GLOBAL: solid hexagons across '
                      'every border', fontsize=11)
    for ax in axes:
        ax.plot(0, 0, marker='*', color='#BB5566', ms=18)
        ax.set_xlabel('longitude')
        ax.set_ylabel('latitude')
    fig.suptitle(f'Hex9+E4H (h9e_*, layer {LAYER} + E + half + {depth} '
                 'class digits) around the cone point (0,0)', fontsize=13)
    fig.savefig(path, dpi=280, bbox_inches='tight')
    print('render ->', path)


if __name__ == '__main__':
    reg = Registrar()
    for depth in range(1, 4):
        table(reg, depth)
        render(reg, depth)
