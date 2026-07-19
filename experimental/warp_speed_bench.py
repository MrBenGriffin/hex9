"""Warp cost decomposition: build (load) vs per-point processing.

Serial chain only (parallel threshold raised), warp built up front and
timed separately. Four configs, each selected explicitly (robust to the
registrar default): classic WGS84-L5, via-sphere Sphere-L6 (2.39M-pt CT),
via-sphere Sphere-L5 (266k-pt CT), via-sphere no-warp (chain floor).

Run from the repo root:  python experimental/warp_speed_bench.py
"""
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
from hhg9 import Registrar, Points
from hhg9.algorithms.distance import wgs84

N = 4000
DATA = Path(__file__).parents[1] / 'hhg9' / 'data'

rng = np.random.default_rng(4007)
PTS = np.column_stack([np.degrees(np.arcsin(rng.uniform(-1, 1, N))),
                       rng.uniform(-180, 180, N)])


def bench(label, via, warp_file=None, no_warp=False):
    reg = Registrar()
    reg.set_ellipsoid(a=6378137.0, inv_f=298.257223563, name='WGS84',
                      via_sphere=via)
    b_oct = reg.domain('b_oct')
    b_oct.no_lib()
    reg.projection('gcd_brw').set_parallel(threshold=1_000_000)  # stay serial
    if warp_file:
        b_oct.set_warp(DATA / warp_file)
    if no_warp:
        b_oct.no_warp()
        t_build = 0.0
    else:
        t0 = time.time()
        w = b_oct.warp
        t_build = time.time() - t0
    g = reg.domain('g_gcd')
    t0 = time.time()
    fwd = reg.project(Points(PTS.copy(), g), [g, b_oct])
    t_enc = time.time() - t0
    t0 = time.time()
    back = reg.project(fwd, [b_oct, g])
    t_dec = time.time() - t0
    d = wgs84(PTS, back.coords) * 1e9
    npts = 0 if no_warp else len(w.src)
    print(f'[{label}] build {t_build:6.1f}s ({npts:>9,} pts)   '
          f'encode {t_enc*1000/N:6.2f} ms/pt   decode {t_dec*1000/N:6.2f} ms/pt   '
          f'RT med {np.median(d):6.2f} nm  max {d.max():8.2f} nm', flush=True)


if __name__ == '__main__':
    bench('classic  WGS84-L5   ', via=False)
    bench('via_sph  Sphere-L6  ', via=True)
    bench('via_sph  Sphere-L5  ', via=True, warp_file='Sphere_l5_warp_data.npz')
    bench('via_sph  no-warp    ', via=True, no_warp=True)
