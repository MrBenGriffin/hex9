"""Render the verified two-shapes-plus-mirror SFC system and the
sigma-symmetric-tiling impossibility figure.

Figure 1 (h9_sfc_mirror.svg): shape A at depths 1-3. Cell fill shows
chirality (plain = cream, mirrored = pale blue); depth-1 cells carry hand
glyphs (mirrored hands drawn with a mirrored glyph, reversed hands primed).
Curve polyline through cell centroids, gold start / red end (CVD-safe).

Figure 2 (h9_sigma_tiling.svg): T1, T2 and the unique sigma-symmetric
tiling with its three self-symmetric pieces highlighted and the mirror
axis drawn — the visual of the self-mirror impossibility proof.
"""

from fractions import Fraction
from math import sqrt
import sys

sys.path.insert(0, "/Users/ben/Documents/Projects/PyCharm/hex9/experimental/sfc")
from sfc_grammar import TILINGS, PARENT, reconstruct, label_piece, tri_verts, GROUND, LIB
from sfc_mirror_verify import (build, compose, apply, SIGMA, I2, F,
                               E, X, PROD, plain, mirrored, CORNERS)

OUT = "/Users/ben/Documents/Projects/PyCharm/hex9/experimental/sfc"
CREAM, PALE = "#FFF7E0", "#DCE6F2"
BLUE, GOLD, RED, GREY = "#004488", "#DDAA33", "#BB5566", "#777777"


def xy(p):
    a, b = float(p[0]), float(p[1])
    return (a + b / 2.0, b * sqrt(3) / 2.0)


CENTROID = (F(7, 3), F(4, 3))


def expand_full(pieces, g, shape, d, par, depth, out, hand, rev):
    """Like verify's expand but records polygon, chirality parity, hand."""
    if depth == 0:
        e = apply(g, E[shape] if d == "f" else X[shape])
        x = apply(g, X[shape] if d == "f" else E[shape])
        poly = [apply(g, c) for c in CORNERS]
        cen = apply(g, CENTROID)
        out.append({"e": e, "x": x, "poly": poly, "cen": cen,
                    "par": par, "hand": hand, "rev": rev})
        return
    seq = PROD[shape]
    if d == "r":
        seq = [(i, k, ("r" if v == "f" else "f")) for (i, k, v) in reversed(seq)]
    for i, k, v in seq:
        h = compose(g, (pieces[i][0], pieces[i][1]))
        p2 = par
        if mirrored(k):
            h = compose(h, SIGMA)
            p2 = 1 - par
        expand_full(pieces, h, plain(k), v, p2, depth - 1, out,
                    (plain(k), p2), v)


def panel_svg(cells, S, show_labels):
    parts = []
    for c in cells:
        pts = " ".join(f"{x*S:.2f},{y*S:.2f}" for x, y in map(xy, c["poly"]))
        fill = PALE if c["par"] else CREAM
        parts.append(f'<polygon points="{pts}" fill="{fill}" '
                     f'stroke="#999" stroke-width="0.6"/>')
    pl = [xy(cells[0]["e"])] + [xy(c["cen"]) for c in cells] + [xy(cells[-1]["x"])]
    pts = " ".join(f"{x*S:.2f},{y*S:.2f}" for x, y in pl)
    w = 2.6 if len(cells) <= 81 else 1.1
    parts.append(f'<polyline points="{pts}" fill="none" stroke="{BLUE}" '
                 f'stroke-width="{w}" stroke-linejoin="round"/>')
    sx, sy = xy(cells[0]["e"]); ex, ey = xy(cells[-1]["x"])
    parts.append(f'<circle cx="{sx*S:.2f}" cy="{sy*S:.2f}" r="5" fill="{GOLD}"/>')
    parts.append(f'<circle cx="{ex*S:.2f}" cy="{ey*S:.2f}" r="5" fill="{RED}"/>')
    if show_labels:
        for c in cells:
            cx, cy = xy(c["cen"])
            name, par = c["hand"]
            glyph = name + ("′" if c["rev"] == "r" else "")
            tx = f'{cx*S:.1f}'
            t = (f'<text x="{tx}" y="{cy*S+6:.1f}" text-anchor="middle" '
                 f'font-size="20" font-weight="bold" fill="#222"')
            if par:  # mirrored hand: mirror the glyph itself
                t = (f'<text x="0" y="0" text-anchor="middle" font-size="20" '
                     f'font-weight="bold" fill="#222" transform='
                     f'"translate({cx*S:.1f},{cy*S+6:.1f}) scale(-1,1)"')
            parts.append(t + f'>{glyph}</text>')
    return parts


def fig_curve(pieces):
    S = 100.0
    pad = 26
    W = 6 * S + 2 * pad
    yh = 3 * sqrt(3) / 2 * S
    parts = [f'<rect width="100%" height="100%" fill="white"/>']
    y0 = pad
    for depth in (1, 2, 3):
        out = []
        expand_full(pieces, (I2, (F(0), F(0))), "A", "f", 0, depth, out,
                    ("A", 0), "f")
        parts.append(f'<g transform="translate({pad},{y0})">')
        parts.extend(panel_svg(out, S, show_labels=(depth == 1)))
        parts.append(f'<text x="{3*S:.0f}" y="{yh+30:.0f}" text-anchor="middle" '
                     f'font-size="16" fill="#222">shape A, level {depth} — '
                     f'{9**depth} cells (pale = mirrored dissection)</text>')
        parts.append('</g>')
        y0 += yh + 62
    H = y0 + pad
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.0f}" '
           f'height="{H:.0f}" viewBox="0 0 {W:.0f} {H:.0f}" '
           f'font-family="Helvetica,Arial">' + "\n".join(parts) + '</svg>')
    with open(f"{OUT}/h9_sfc_mirror.svg", "w") as f:
        f.write(svg)


def all_tilings():
    placements = set()
    for k, shape in LIB.items():
        for oy in range(-8, 9, 2):
            for ox in range(-8, 9):
                cells = frozenset((px + ox, py + oy) for (px, py) in shape)
                if cells <= GROUND:
                    placements.add(cells)
    placements = sorted(placements, key=lambda s: sorted(s))
    sols = []
    def cover(uncovered, chosen):
        if not uncovered:
            sols.append(frozenset(chosen))
            return
        cell = min(uncovered, key=lambda e: (e[1], e[0]))
        for cells in placements:
            if cell in cells and cells <= uncovered:
                chosen.append(cells)
                cover(uncovered - cells, chosen)
                chosen.pop()
    cover(frozenset(GROUND), [])
    return sols


def sig_cell_map():
    def sig(p):
        return (-p[0] - p[1] + 6, p[1])
    by = {frozenset(tri_verts(c)): c for c in GROUND}
    return {c: by[frozenset(sig(v) for v in tri_verts(c))] for c in GROUND}


def piece_poly(cells):
    """Boundary edges of a piece -> ordered polygon points."""
    from collections import Counter
    e = Counter()
    for c in cells:
        v = tri_verts(c)
        for i in range(3):
            e[frozenset((v[i], v[(i + 1) % 3]))] += 1
    edges = [tuple(k) for k, n in e.items() if n == 1]
    adj = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    start = edges[0][0]
    poly = [start]
    prev = None
    cur = start
    while True:
        nxt = [n for n in adj[cur] if n != prev]
        prev, cur = cur, nxt[0]
        if cur == start:
            break
        poly.append(cur)
    return poly


def fig_tilings():
    S = 60.0
    pad = 24
    sols = all_tilings()
    SIG = sig_cell_map()
    t1 = reconstruct(TILINGS["T1"])
    t2 = reconstruct(TILINGS["T2"])
    sym = [s for s in sols
           if frozenset(frozenset(SIG[c] for c in p) for p in s) == s][0]
    panels = [("T1 (canonical)", [set(c) for _, c in t1], None),
              ("T2 (mirror of T1)", [set(c) for _, c in t2], None),
              ("the unique σ-symmetric tiling (3 self-symmetric pieces)",
               [set(p) for p in sym],
               [p for p in sym if frozenset(SIG[c] for c in p) == p])]
    W = 6 * S + 2 * pad
    yh = 3 * sqrt(3) / 2 * S
    parts = ['<rect width="100%" height="100%" fill="white"/>']
    y0 = pad
    for title, pieces_cells, selfsym in panels:
        parts.append(f'<g transform="translate({pad},{y0})">')
        for cells in pieces_cells:
            poly = piece_poly(cells)
            pts = " ".join(f"{x*S:.2f},{y*S:.2f}" for x, y in map(xy, poly))
            fill = CREAM
            if selfsym and frozenset(cells) in [frozenset(p) for p in selfsym]:
                fill = GOLD
            parts.append(f'<polygon points="{pts}" fill="{fill}" '
                         f'fill-opacity="0.85" stroke="#333" stroke-width="1.1"/>')
        if selfsym:
            a1 = xy((3, 0)); a2 = xy((F(3, 2), 3))
            parts.append(f'<line x1="{a1[0]*S:.1f}" y1="{a1[1]*S:.1f}" '
                         f'x2="{a2[0]*S:.1f}" y2="{a2[1]*S:.1f}" '
                         f'stroke="{RED}" stroke-width="2" stroke-dasharray="7,5"/>')
        parts.append(f'<text x="{3*S:.0f}" y="{yh+26:.0f}" text-anchor="middle" '
                     f'font-size="14" fill="#222">{title}</text>')
        parts.append('</g>')
        y0 += yh + 52
    H = y0 + pad
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.0f}" '
           f'height="{H:.0f}" viewBox="0 0 {W:.0f} {H:.0f}" '
           f'font-family="Helvetica,Arial">' + "\n".join(parts) + '</svg>')
    with open(f"{OUT}/h9_sigma_tiling.svg", "w") as f:
        f.write(svg)


if __name__ == "__main__":
    pieces = build()
    fig_curve(pieces)
    fig_tilings()
    print("wrote h9_sfc_mirror.svg and h9_sigma_tiling.svg")
