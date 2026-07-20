# Part of the Hex9 (H9) Project
# Copyright ©2026, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""Lineage vs ownership, side by side, per system.

Companion to dggs_nesting.py (which compares systems under their own
native descendant relation). Here each NON-CONGRUENT system gets a row
with two panels over the same anchor:

  left  — LINEAGE descendants: the native index relation (cell_to_children
          / iterated parentage). Transitive by construction; decoheres —
          red cells lie entirely outside the ancestor they index under.
  right — OWNED descendants: unique geometric assignment — every zone at
          the target level whose owner (the coarse zone containing its
          centre/ownership point) is the anchor. Partitions the globe;
          nothing red, by construction.

For H3, A5 and ISEA3H the ownership rule used here is centre-point
containment (owner = encode(cell centre) at the anchor level) — definable
today with their public APIs, no library changes. ISEA3H's lineage is
DGGAL's primary-parent relation (parents[0] = upstream
getZonePrimaryParent), run through isea_probe.py in a subprocess. Note
their owned COUNTS vary per anchor (no carrier ⇒ no exact-count
property); Hex9's rep-9 d_cell carrier gives exactly 9^depth every time.
S2 and HEALPix are omitted: fully-nested, the two panels would be
identical.

The Hex9 row also shows the anchor's curve address beside its h9
address: the Hamiltonian-curve name for the same cell, a true bijection
(h9_curve_uuid ⇄ h9_curve_decode) kept alongside the h9 address, not in
place of it — the curve tree is the lineage tree, so curve-prefix
descendants are exactly the lineage panel, while ownership is the
geometric relation on the right.

Run:
    python docs/dggs/dggs_ownership.py            # -> docs/dggs/dggs_ownership.png
    python docs/dggs/dggs_ownership.py --gens 3 --lat 40 --lng -3
"""
import argparse
import os
import sys

import numpy as np
from shapely.geometry import Polygon

sys.path.insert(0, os.path.dirname(__file__))
from dggs_nesting import classify, COLOURS, EDGE_TOL, DEFAULT_RES  # noqa: E402


# ── per-system pairs ─────────────────────────────────────────────────────────
# Each returns (anchor_poly, naive_polys, owned_polys, label, owned_note).

def h3_pair(lat, lng, res, gens):
    import h3
    target = res + gens
    anchor = h3.latlng_to_cell(lat, lng, res)

    def poly(c):
        return Polygon([(lng_, lat_) for lat_, lng_ in h3.cell_to_boundary(c)])

    naive = [poly(c) for c in h3.cell_to_children(anchor, target)]
    cands = set()
    for z in h3.grid_disk(anchor, 1):
        cands.update(h3.cell_to_children(z, target))
    owned_ids = [c for c in cands
                 if h3.latlng_to_cell(*h3.cell_to_latlng(c), res) == anchor]
    owned = [poly(c) for c in owned_ids]
    return (poly(anchor), naive, owned, f'H3 r{res}→r{target}',
            f'{len(owned_ids)} owned (varies by anchor)')


def a5_pair(lat, lng, res, gens):
    import a5_fast as a5
    target = res + gens
    anchor = a5.lonlat_to_cell(float(lng), float(lat), res)

    def poly(c):
        return Polygon(a5.cell_to_boundary(c))            # ring of (lon, lat)

    naive = [poly(c) for c in a5.cell_to_children(anchor, target)]
    cands = set()
    for z in a5.grid_disk(anchor, 1):
        cands.update(a5.cell_to_children(z, target))
    owned_ids = []
    for c in cands:
        lon_c, lat_c = a5.cell_to_lonlat(c)
        if a5.lonlat_to_cell(float(lon_c), float(lat_c), res) == anchor:
            owned_ids.append(c)
    owned = [poly(c) for c in owned_ids]
    return (poly(anchor), naive, owned, f'A5 r{res}→r{target}',
            f'{len(owned_ids)} owned (varies by anchor)')


def isea3h_pair(lat, lng, res, gens):
    """ISEA3H via DGGAL (subprocess probe): lineage = primary-parent walk
    (parents[0] = upstream getZonePrimaryParent); owned = centre-point
    containment (zone whose WGS84 centroid re-encodes to the anchor) —
    definable today with DGGAL's public API. See isea_probe.py."""
    import json
    import subprocess
    from dggs_nesting import dggal_probe_cmd
    cmd = dggal_probe_cmd('--mode', 'ownership', '--lat', str(lat), '--lng', str(lng),
                          '--res', str(res), '--gens', str(gens))
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if out.returncode:
        raise NotImplementedError(f'isea_probe failed: {out.stderr.strip()[-200:]}')
    data = json.loads(out.stdout)
    return (Polygon(data['anchor']),
            [Polygon(r) for r in data['naive']],
            [Polygon(r) for r in data['owned']],
            data['label'].replace('->', '→'),
            f'{len(data["owned"])} owned (varies by anchor)')


def hex9_pair(lat, lng, res, gens):
    from shapely.validation import make_valid
    from hhg9 import Registrar, Points
    from hhg9.h9.grid import HexMesh
    from hhg9.h9.uuid_address import (h9_descendants, h9_bin_pts,
                                      h9_cell_parent, _anchor_hex_latlon,
                                      h9_label, h9_curve_uuid, h9_curve_label)
    target = res + gens
    reg = Registrar()
    g_gcd, b_oct = reg.domain('g_gcd'), reg.domain('b_oct')
    b_pts = reg.project(Points(np.array([[lat, lng]], float), g_gcd),
                        [g_gcd, b_oct])
    anchor = h9_bin_pts(b_pts, res)[0]

    def poly(u, lay):
        ring = _anchor_hex_latlon(u, lay, reg, inflate=1.0)
        P = Polygon([(lo, la) for la, lo in ring])
        return P if P.is_valid else make_valid(P)

    # owned: the h9_descendants contract (leaf-reified d_cell relation)
    owned_ids = h9_descendants(b_pts, res, gens, reg)[0]
    # naive: iterated one-level parentage over a generous enumeration
    clip = _anchor_hex_latlon(anchor, res, reg, inflate=2.4)
    cand = list(HexMesh.create_clipped([target], clip, reg).addr(target))
    cur = cand
    for _ in range(gens):
        cur = h9_cell_parent(cur, reg=reg)
    naive_ids = [c for c, a in zip(cand, cur) if a.int == anchor.int]
    # The anchor's two names: h9 address and curve address — a bijective
    # pair (h9_curve_uuid ⇄ h9_curve_decode), shown side by side.
    addresses = (f'anchor h9 {h9_label(anchor)} = '
                 f'curve {h9_curve_label(h9_curve_uuid([anchor])[0])}')
    return (poly(anchor, res),
            [poly(u, target) for u in naive_ids],
            [poly(u, target) for u in owned_ids],
            f'Hex9 L{res}→L{target}',
            f'{len(owned_ids)} owned (= 9^{gens}, every anchor)',
            addresses)


PAIRS = {'H3': h3_pair, 'A5': a5_pair, 'ISEA3H': isea3h_pair, 'Hex9': hex9_pair}


def main(lat, lng, gens, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    systems = list(PAIRS)
    fig, axes = plt.subplots(len(systems), 2, figsize=(11, 5 * len(systems)))
    print(f'anchor = ({lat}, {lng})   gens = {gens}')
    print(f'{"system":>6} {"relation":>9} {"cells":>6} {"inside":>7} '
          f'{"straddle":>9} {"OUTSIDE":>8}')

    for row, name in enumerate(systems):
        res = DEFAULT_RES[name]
        result = PAIRS[name](lat, lng, res, gens)
        anchor, naive, owned, label, note = result[:5]
        subtitle = result[5] if len(result) > 5 else None
        for col, (desc, rel) in enumerate([(naive, 'lineage'),
                                           (owned, 'owned')]):
            ax = axes[row, col]
            fracs, counts = classify(anchor, desc)
            ax.add_patch(MplPolygon(list(anchor.exterior.coords), closed=True,
                                    fill=False, edgecolor='#222222',
                                    linewidth=2.5, zorder=3))
            for d, f in zip(desc, fracs):
                key = ('out' if f < EDGE_TOL else
                       'in' if f > 1 - EDGE_TOL else 'straddle')
                ax.add_patch(MplPolygon(list(d.exterior.coords), closed=True,
                                        facecolor=COLOURS[key],
                                        edgecolor='#555555',
                                        linewidth=0.3, alpha=0.85, zorder=2))
            title = f'{label} — {rel}   (out {counts["out"]}/{len(desc)})'
            if subtitle:
                title += f'\n{subtitle}'
            ax.autoscale()
            ax.set_aspect('equal', adjustable='box')
            ax.axis('off')
            ax.set_title(title)
            print(f'{name:>6} {rel:>9} {len(desc):>6} {counts["in"]:>7} '
                  f'{counts["straddle"]:>9} {counts["out"]:>8}')
        print(f'{"":>6} {note}')
        if subtitle:
            print(f'{"":>6} {subtitle}')
        # Same anchor, same scale: share limits across the row so the two
        # relations are visually comparable.
        ax_l, ax_r = axes[row]
        x0 = min(ax_l.get_xlim()[0], ax_r.get_xlim()[0])
        x1 = max(ax_l.get_xlim()[1], ax_r.get_xlim()[1])
        y0 = min(ax_l.get_ylim()[0], ax_r.get_ylim()[0])
        y1 = max(ax_l.get_ylim()[1], ax_r.get_ylim()[1])
        for ax in (ax_l, ax_r):
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)

    fig.suptitle('Lineage vs ownership — the same anchor, two descendant '
                 'relations\nblue = inside   gold = straddle   red = '
                 'entirely outside the ancestor', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out_path, dpi=140)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--lat', type=float, default=40.0)
    ap.add_argument('--lng', type=float, default=-3.0)
    ap.add_argument('--gens', type=int, default=4)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__),
                                                  'dggs_ownership.png'))
    args = ap.parse_args()
    main(args.lat, args.lng, args.gens, args.out)
