# Part of the Hex9 (H9) Project
# Copyright ©2026, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Subprocess probe for DGGAL-backed DGGRS adapters (ISEA3H et al.).

Run by dggs_nesting.py in a separate interpreter because the dggal 0.0.6
macOS arm64 wheel ships x86_64 dylibs (upstream packaging bug): the probe is
executed under Rosetta from the repo's `.venv-x86` when a direct import fails.
Emits JSON on stdout:

    {"label": str, "anchor": [[lon, lat], ...], "descendants": [[[lon, lat], ...], ...]}

Lineage descendants use DGGAL's primary-parent convention — a child belongs to
`getZoneParents(child)[0]`, which is exactly upstream `getZonePrimaryParent`
(ecere/dggal 91983a7: `return zone.parent0`). This is DGGAL's own lineage
reading (cf. OGC API-DGGS issue #108 discussion); its native sub-zone listing
(`getSubZones`) is the coverage relation and deliberately NOT used here.
"""
import argparse
import json
import sys

from dggal import *                                         # noqa: F403


def make_dggrs(dggrs_name):
    app = Application(appGlobals=globals())                 # noqa: F405
    pydggal_setup(app)                                      # noqa: F405
    return eval(dggrs_name)()                               # e.g. ISEA3H / ISEA7H / ISEA9R


def commute(res, k, dggrs_name):
    """Read {"lats": [...], "lngs": [...]} on stdin; count points where direct
    coarse binning differs from binning at res+k then walking primary parents
    up k levels (DGGAL lineage, cf. getZonePrimaryParent)."""
    dggrs = make_dggrs(dggrs_name)
    data = json.load(sys.stdin)
    bad = 0
    for lat, lng in zip(data['lats'], data['lngs']):
        gp = GeoPoint(Degrees(lat), Degrees(lng))           # noqa: F405
        direct = dggrs.getZoneFromWGS84Centroid(res, gp)
        z = dggrs.getZoneFromWGS84Centroid(res + k, gp)
        for _ in range(k):
            z = dggrs.getZoneParents(z)[0]
        bad += (z != direct)
    print(json.dumps({'bad': int(bad)}))


def lineage_descendants(dggrs, anchor, gens):
    frontier = [anchor]
    for _ in range(gens):                  # lineage: keep children we are primary parent of
        frontier = [c for z in frontier for c in dggrs.getZoneChildren(z)
                    if dggrs.getZoneParents(c)[0] == z]
    return frontier


def ownership(lat, lng, res, gens, dggrs_name):
    """Lineage vs centre-point ownership over one anchor (cf. dggs_ownership.py).
    Owned = every zone at res+gens whose WGS84 centroid re-encodes to the anchor
    at res — definable today with DGGAL's public API, no geometry beyond the
    centroid. Candidates are lineage descendants of the anchor's 2-ring (drift
    is under one anchor width, so the 2-ring covers the anchor's region)."""
    dggrs = make_dggrs(dggrs_name)
    anchor = dggrs.getZoneFromWGS84Centroid(res, GeoPoint(Degrees(lat), Degrees(lng)))  # noqa: F405
    naive = lineage_descendants(dggrs, anchor, gens)

    ring = {dggrs.getZoneTextID(anchor): anchor}
    for _ in range(2):
        for z in list(ring.values()):
            nb_types = Array('<int>')                       # noqa: F405
            for nb in dggrs.getZoneNeighbors(z, nb_types):
                ring[dggrs.getZoneTextID(nb)] = nb
    cands = {}
    for z in ring.values():
        for c in lineage_descendants(dggrs, z, gens):
            cands[dggrs.getZoneTextID(c)] = c
    owned = [c for c in cands.values()
             if dggrs.getZoneFromWGS84Centroid(res, dggrs.getZoneWGS84Centroid(c)) == anchor]

    def ring_pts(zone):
        return [[float(p.lon), float(p.lat)] for p in dggrs.getZoneWGS84Vertices(zone)]

    print(json.dumps({
        'label': f'{dggrs_name} L{res}->L{res + gens}',
        'anchor': ring_pts(anchor),
        'naive': [ring_pts(z) for z in naive],
        'owned': [ring_pts(z) for z in owned],
    }))


def main(lat, lng, res, gens, dggrs_name):
    dggrs = make_dggrs(dggrs_name)
    anchor = dggrs.getZoneFromWGS84Centroid(res, GeoPoint(Degrees(lat), Degrees(lng)))  # noqa: F405
    frontier = lineage_descendants(dggrs, anchor, gens)

    def ring(zone):
        return [[float(p.lon), float(p.lat)] for p in dggrs.getZoneWGS84Vertices(zone)]

    print(json.dumps({
        'label': f'{dggrs_name} L{res}->L{res + gens}',
        'anchor': ring(anchor),
        'descendants': [ring(z) for z in frontier],
    }))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--mode', choices=['nesting', 'commute', 'ownership'],
                    default='nesting')
    ap.add_argument('--lat', type=float)
    ap.add_argument('--lng', type=float)
    ap.add_argument('--res', type=int, required=True)
    ap.add_argument('--gens', type=int, help='nesting: levels below the anchor')
    ap.add_argument('--k', type=int, help='commute: levels to truncate')
    ap.add_argument('--dggrs', default='ISEA3H')
    a = ap.parse_args()
    if a.mode == 'commute':
        commute(a.res, a.k, a.dggrs)
    elif a.mode == 'ownership':
        ownership(a.lat, a.lng, a.res, a.gens, a.dggrs)
    else:
        main(a.lat, a.lng, a.res, a.gens, a.dggrs)
