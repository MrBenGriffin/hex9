# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""
Local warp test harness — validates a trained warp before it is ported into
``hhg9/domains/octahedral_barycentric.py``.

Two checks:

1. b_oct round-trip:  do() then undo() over interior, lateral-edge and apex
   points. Lateral-edge and apex points must come back to machine precision
   (the warp is the identity there); interior points must come back tight.

2. Geodetic round-trip:  g_gcd → b_oct → g_gcd at the four index points from
   examples/try_warps.py. All four must be < 7 nm. This injects warp.py's
   AuthalicWarp into the live b_oct domain, so it tests the exact class that
   will be ported.

Usage:
    python test_warp.py --warp <file.npz> [--ellipsoid WGS84]
"""
import argparse
from pathlib import Path
import numpy as np

from hhg9 import Registrar, Points
from hhg9.algorithms.distance import wgs84
from experimental.sinkhorn import config
from experimental.sinkhorn.warp import AuthalicWarp


def boct_roundtrip(warp: AuthalicWarp):
    """do() then undo() on grid vertices, split by location."""
    grid = np.load(config.SINKHORN_DIR / 'grid_l5.npz', allow_pickle=True)
    src = np.asarray(grid['xy_vert'], dtype=np.float64)
    lat = config.lateral_edge_mask(src)

    b_oct = warp.do(src, mo=0)
    back  = warp.undo(b_oct, mo=0)
    err = np.linalg.norm(back - src, axis=1)

    print('--- b_oct round-trip (undo(do(p)) - p) ---')
    for label, m in [('interior     ', ~lat), ('lateral+apex ', lat)]:
        if np.any(m):
            print(f'  {label}: max {err[m].max():.3e}  mean {err[m].mean():.3e}  '
                  f'(n={int(m.sum())})')
    return float(err.max())


def geodetic_roundtrip(warp: AuthalicWarp, ellipsoid: str):
    """g_gcd → b_oct → g_gcd at the four index points; injects warp into b_oct."""
    reg = Registrar()
    config.apply_ellipsoid(reg, ellipsoid)
    g_gcd  = reg.domain('g_gcd')
    b_oct  = reg.domain('b_oct')

    b_oct.warp = warp
    warp.domain = b_oct
    for proj in b_oct.projs.values():
        proj.warp = warp

    print(f'--- geodetic round-trip ({ellipsoid}) ---')
    worst = 0.0
    for lat, lon in [(51.4778301090125, 1e-11), (51.4778301090125, -1e-11),
                     (89.99, 0.0), (45.0, 90.0001e-11)]:
        pt = Points(np.array([[lat, lon]]), g_gcd)
        b  = reg.project(pt, ['g_gcd', 'b_oct'])
        b2 = reg.project(b, ['b_oct', 'g_gcd'])
        err = wgs84(pt.coords, b2.coords)[0] * 1e9
        worst = max(worst, err)
        flag = '' if err < 7.0 else '   *** > 7 nm ***'
        print(f'  lat={lat:.4f} lon={lon:+.2e}: {err:.3f} nm{flag}')
    return worst


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Validate a trained warp.')
    ap.add_argument('--warp', type=Path, required=True, help='warp .npz to test')
    ap.add_argument('--ellipsoid', default=config.INDEX_ELLIPSOID,
                    choices=list(config.ELLIPSOIDS))
    args = ap.parse_args()

    warp = AuthalicWarp(args.warp)
    rt = boct_roundtrip(warp)
    gd = geodetic_roundtrip(warp, args.ellipsoid)

    print()
    ok = gd < 7.0
    print(f'b_oct round-trip max : {rt:.3e}')
    print(f'geodetic worst       : {gd:.3f} nm   {"PASS" if ok else "FAIL"}')
    raise SystemExit(0 if ok else 1)
