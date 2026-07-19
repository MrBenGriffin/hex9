"""Chain acceptance battery: WGS84 via-sphere vs classic, pure Python.

RT battery mirrors examples/ex0060u_addresses.py (g_gcd->b_oct->g_gcd and
uuid enc/dec round-trips, geodesic deltas in nm); equal-area battery is the
FD-Jacobian spread vs the WGS84 area element (seam-masked, corner subset).
Both chains are selected EXPLICITLY (robust to the registrar default).

Run from the repo root:  python experimental/battery.py
"""
import json
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
from hhg9 import Registrar, Points
from hhg9.algorithms.distance import wgs84
from hhg9.algorithms.pickers import gcd_rnd
from hhg9.h9.uuid_address import h9_enc_ext, h9_dec

A_WGS, E2 = 6378137.0, 0.00669437999014132
VERTICES = np.array([[90.0, 0.0], [-90.0, 0.0],
                     [0.0, 0.0], [0.0, 90.0], [0.0, -90.0], [0.0, 180.0]])


def build(via):
    reg = Registrar()
    reg.set_ellipsoid(a=6378137.0, inv_f=298.257223563, name='WGS84',
                      via_sphere=via)
    b = reg.domain('b_oct')
    b.no_lib()
    _ = b.warp                      # force the lazy CT build up front
    return reg, b


def rt_battery(reg, b_oct, label, pts):
    g = reg.domain('g_gcd')
    refs = Points(pts.copy(), g)
    t0 = time.time()
    b_rys = reg.project(refs, ['g_gcd', 'b_oct'])
    b_rtp = reg.project(b_rys, ['b_oct', 'g_gcd'])
    d = wgs84(refs.coords, b_rtp.coords) * 1e9
    oc, mo = b_rys.cm()
    ub = h9_enc_ext(b_rys, oc, mo)
    urt = h9_dec(ub, b_oct)
    up = reg.project(urt, ['b_oct', 'g_gcd'])
    ud = wgs84(refs.coords, up.coords) * 1e9
    print(f'[RT {label}] n={len(pts)} ({time.time()-t0:.0f}s)\n'
          f'    gcd  RT nm: med {np.median(d):7.3f}  p99 {np.percentile(d, 99):7.3f}  max {d.max():8.3f}\n'
          f'    uuid RT nm: med {np.median(ud):7.3f}  p99 {np.percentile(ud, 99):7.3f}  max {ud.max():8.3f}',
          flush=True)


def area_element(lat_deg):
    phi = np.radians(lat_deg)
    s2 = np.sin(phi) ** 2
    M = A_WGS * (1 - E2) / (1 - E2 * s2) ** 1.5
    N = A_WGS / (1 - E2 * s2) ** 0.5
    return M * N * np.cos(phi)


def corner_dist_deg(pts):
    lat, lon = np.radians(pts[:, 0]), np.radians(pts[:, 1])
    vlat, vlon = np.radians(VERTICES[:, 0]), np.radians(VERTICES[:, 1])
    cosd = (np.sin(lat)[:, None] * np.sin(vlat)[None, :]
            + np.cos(lat)[:, None] * np.cos(vlat)[None, :]
            * np.cos(lon[:, None] - vlon[None, :]))
    return np.degrees(np.arccos(np.clip(cosd.max(axis=1), -1.0, 1.0)))


def area_battery(reg, label, n=8000, h=2e-5, margin=0.5):
    rng = np.random.default_rng(4007)
    lon = rng.uniform(-180, 180, n)
    lat = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
    dlon = np.abs((lon + 45) % 90 - 45)
    keep = (np.abs(lat) > margin) & (dlon > margin) & (np.abs(lat) < 89 - margin)
    pts = np.column_stack([lat[keep], lon[keep]])
    allp = np.vstack([pts, pts + [0, h], pts + [h, 0]])   # (lat, lon) order
    g = reg.domain('g_gcd')
    t0 = time.time()
    out = reg.project(Points(allp, g), ['g_gcd', 'b_oct'])
    m = len(pts)
    oid = out.oid
    same = (oid[:m] == oid[m:2 * m]) & (oid[:m] == oid[2 * m:])
    o0, olon, olat = out.coords[:m], out.coords[m:2 * m], out.coords[2 * m:]
    hr = np.radians(h)
    J = np.abs((olon[:, 0] - o0[:, 0]) * (olat[:, 1] - o0[:, 1]) -
               (olat[:, 0] - o0[:, 0]) * (olon[:, 1] - o0[:, 1])) / hr ** 2
    rel = (J / area_element(pts[:, 0]))[same]
    rel = rel / np.median(rel)
    near = corner_dist_deg(pts[same]) < 3.0
    print(f'[EA {label}] n={same.sum()} ({time.time()-t0:.0f}s)  '
          f'p1 {np.percentile(rel, 1):.6f}  p99 {np.percentile(rel, 99):.6f}  '
          f'min {rel.min():.6f}  max {rel.max():.6f}', flush=True)
    if near.any():
        print(f'    corner<3°: n={near.sum()}  min {rel[near].min():.6f}  '
              f'max {rel[near].max():.6f}', flush=True)


if __name__ == '__main__':
    locs = json.load(open(Path(__file__).parents[1] / 'assets/locations.json'))
    ll = [tuple(map(float, v)) for region in locs.values() for v in region.values()]
    np.random.seed(4007)
    pts = np.vstack([np.array(ll), gcd_rnd(20_000)])
    print(f'{len(pts)} points (landmarks + seeded random)', flush=True)

    for via, label in ((False, 'WGS84 classic   '), (True, 'WGS84 via_sphere')):
        reg, b_oct = build(via)
        print(f'--- {label}  warp={str(b_oct.warp_file).rsplit("/", 1)[-1]}',
              flush=True)
        rt_battery(reg, b_oct, label, pts)
        area_battery(reg, label)
