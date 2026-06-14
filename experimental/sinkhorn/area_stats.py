# Part of the Hex9 (H9) Project
# Copyright ©2026, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""
Canonical hex-level area statistics for a deployed warp (ex0081w-style).

Generates every depth-N hexagon, projects b_oct -> g_gcd under the given
registrar ellipsoid (which also selects the deployed {ell}_l5_warp_data.npz),
computes geodesic polygon areas, and reports the deviation stats recorded in
TRAINING_NOTES.md (min/max %dev, log-ratio sigma, |%dev| percentiles).

    python area_stats.py --ellipsoid Sphere --depth 5
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hhg9 import Registrar, Points
from hhg9.algorithms.distance import wgs84_area
from hhg9.h9 import H9_RA, H9O
from hhg9.h9.region import regions_xy
from hhg9.h9.polygon import hex_poly_layer
from experimental.sinkhorn import config


def chain_generator(initial_seed, depth, props=H9_RA.props, modes=H9_RA.modes):
    def _recurse(current_chain):
        if len(current_chain) - 2 == depth:
            yield current_chain
            return
        seed = current_chain[-1]
        for child in props[modes[seed]].flatten():
            yield from _recurse(current_chain + [child])
    yield from _recurse([initial_seed])


def get_data(reg: Registrar, depth):
    b_oct = reg.domain('b_oct')
    all_rgn = [
        list(chain_generator(H9_RA.proto[0], depth)),
        list(chain_generator(H9_RA.proto[1], depth)),
    ]
    rgn = H9_RA.rid2cell[np.array(all_rgn)]
    sides = []
    for oc in range(8):
        xym = regions_xy(rgn[H9O.oid_mo[oc]])
        sides.append(Points(xym[:, :-1], b_oct, oc))
    return Points.concat(sides)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ellipsoid', default='WGS84', choices=list(config.ELLIPSOIDS))
    ap.add_argument('--depth', type=int, default=5)
    args = ap.parse_args()

    reg = Registrar()
    config.apply_ellipsoid(reg, args.ellipsoid)
    print(f'ellipsoid: {reg.ellipsoid_name}  (a={reg.ellipsoid.a}, f={reg.ellipsoid.f})')
    print(f'surface area: {reg.ellipsoid_area:.6e} m²')

    pts, _ = hex_poly_layer(get_data(reg, args.depth), args.depth)
    bins = 12 * 9 ** args.depth
    ideal = reg.ellipsoid_area / bins
    print(f'depth {args.depth}: {bins} bins, ideal {ideal:.4f} m² per hex')

    # the depth-d region sampling emits each hex 3 times — dedupe on (xy, oid)
    xy = np.round(pts.coords.reshape(-1, 6, 2), 9)
    oid = pts.oid.reshape(-1, 6)
    key = np.concatenate([xy.reshape(-1, 12), oid.astype(float)], axis=1)
    _, keep = np.unique(key, axis=0, return_index=True)
    sel = np.repeat(keep[:, None] * 6, 6, axis=1) + np.arange(6)
    pts = Points(pts.coords[sel.ravel()], pts.domain, oid=pts.oid[sel.ravel()])
    print(f'{len(keep)} unique hexes after dedupe')

    g_pts = reg.project(pts, ['b_oct', 'g_gcd'])
    areas = wgs84_area(reg, g_pts)          # uses reg.ellipsoid
    if len(areas) != bins:
        print(f'WARNING: {len(areas)} hexes != expected {bins}; '
              'closure check below is the arbiter')

    ratio = np.abs(areas) / ideal
    dev = (ratio - 1.0) * 100.0
    log_ratio = np.log(ratio)
    p = lambda q: float(np.percentile(np.abs(dev), q))
    print(f'\nclosure: sum(areas)/surface - 1 = {np.abs(areas).sum()/reg.ellipsoid_area - 1:+.3e}')
    print(f'%dev mean {dev.mean():+.6f}  min {dev.min():+.5f}  max {dev.max():+.5f}')
    print(f'log-ratio sigma {log_ratio.std():.3e}')
    print(f'|%dev| p50 {p(50):.5f}  p99 {p(99):.5f}  p99.99 {p(99.99):.5f}')
