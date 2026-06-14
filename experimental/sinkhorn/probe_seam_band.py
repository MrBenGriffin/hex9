# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""
Probe the lateral-edge discontinuity band along non-equatorial seams.

Scans a b_oct-straight segment (the render path: b_oct -> g_gcd, as in
ex0263_89_seam_zoom.py) running from the lon-0 seam into the adjacent octant,
and reports the projected latitude as a function of ground distance from the
meridian. Under the legacy hard bypass (H9_WARP_EDGE=bypass) the projection
jumps at the band edge (|y - R3|x| + W| = 1e-7); under the F6 feather
(default) it should be continuous.

Run twice to compare:
    H9_WARP_EDGE=bypass  python probe_seam_band.py
    H9_WARP_EDGE=feather python probe_seam_band.py

User-measured reference (L22 render, legacy behaviour): band edge at
+/-0.434 m from the meridian at lat 51.48 (lon 6.26e-6 deg), jump ~0.2 m
almost purely meridional; worst case +/-89.1 near the 0/90 meridians.
"""
import os
import numpy as np
from hhg9 import Registrar
from hhg9.base.points import Points
from hhg9.h9 import H9K
from experimental.sinkhorn import config

M_PER_DEG = 111_319.49   # metres per degree of latitude (spherical approx)
BYPASS = 1e-7            # legacy hard-band threshold on the line value

reg = Registrar()
config.apply_ellipsoid(reg, os.environ.get('H9_PROBE_ELLIPSOID', 'WGS84'))
b_oct = reg.domain('b_oct')
g_gcd = reg.domain('g_gcd')

R3, W_ = H9K.radical.R3, H9K.Ẇ


def probe(lat, lon_hi, n=2001):
    """Scan b_oct segment from the seam (lon ~ 0+) out to lon_hi at given lat."""
    anchors = Points(np.array([[lat, 1e-9], [lat, lon_hi]]), domain=g_gcd)
    bp = reg.project(anchors, [g_gcd, b_oct])
    b, oids = bp.coords, bp.oid
    assert oids[0] == oids[1], f'anchors span octants: {oids}'

    t = np.linspace(0.0, 1.0, n)[:, None]
    seg = b[0] * (1 - t) + b[1] * t
    pts = Points(seg, domain=b_oct, oid=np.full(n, oids[0]))
    ll = reg.project(pts, [b_oct, g_gcd]).coords          # (lat, lon)

    lat_m = (ll[:, 0] - lat) * M_PER_DEG
    lon_m = ll[:, 1] * M_PER_DEG * np.cos(np.radians(lat))

    dlat, dlon = np.diff(lat_m), np.diff(lon_m)
    step = np.hypot(dlat, dlon)
    med = np.median(step)
    jumps = np.flatnonzero(step > 10 * med)

    expr = np.abs(np.abs(seg[:, 1]) - R3 * np.abs(seg[:, 0]) + W_)
    in_band = expr < BYPASS

    from hhg9.domains.octahedral_barycentric import LATERAL_EDGE_MODE
    print(f'\n=== lat {lat:+.2f}  oid {oids[0]}  mode {LATERAL_EDGE_MODE} ===')
    print(f'scan: {n} pts, lon (0, {lon_hi}] deg; median step {med*1000:.3f} mm; '
          f'{in_band.sum()} pts inside legacy band')
    if in_band.any():
        i1 = np.flatnonzero(in_band)[-1]
        print(f'legacy band edge at lon {ll[i1,1]:.3e} deg = {lon_m[i1]:+.4f} m')
    if len(jumps):
        for j in jumps:
            print(f'  JUMP at lon {lon_m[j]:+.4f} -> {lon_m[j+1]:+.4f} m: '
                  f'dlat {dlat[j]*1000:+9.2f} mm, dlon {dlon[j]*1000:+9.2f} mm')
    else:
        print('  no jumps > 10x median step — continuous')


if __name__ == '__main__':
    probe(51.48, 2e-5)
    probe(89.1, 2e-3)
    probe(-89.1, 2e-3)
