# Part of the Hex9 (H9) Project
# Copyright ©2026, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""Wedge-CT acceptance: fold-evaluate vs unfolded-face CT on the same
fundamental-domain artifact (h9_proj L6_MOBIUS_WARP.md §3.7/§3.7b).

Builds the warp both ways — AuthalicWarp(npz, fold=True) (wedge chart +
mirror halos, queries folded in by the D3 group) and fold=False (legacy
unfold to the full face) — and compares: vertex exactness, field
agreement (interior + near-line bands), evaluation D3-equivariance,
seam tangency, displacement continuity across the fold line, round
trips, and bulk evaluation speed.

Expected: agreement at float noise everywhere except the few-lattice-
pitch discs at the three face corners (max ~3e-6 units), where the fold
build is exactly dihedral-symmetric and the face build patches estimated
gradients. See §3.7b for the accepted 2026-07-19 numbers.

Run from the repo root:  python experimental/wedge_fold_check.py [npz]
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from hhg9.h9 import H9K
from hhg9.domains.octahedral_barycentric import AuthalicWarp

NPZ = (sys.argv[1] if len(sys.argv) > 1
       else str(Path(__file__).parents[1] / 'hhg9/data/Sphere_l6_fund_warp_data.npz'))
rng = np.random.default_rng(42)

VF, VC, TR = H9K.limits.VF, H9K.limits.VC, H9K.limits.TR
R3, W = H9K.radical.R3, H9K.Ẇ


def face_sample(n):
    """Uniform random points in the mode-0 DOWN triangle (apex (0,VF))."""
    a, b = rng.random(n), rng.random(n)
    flip = a + b > 1.0
    a[flip], b[flip] = 1.0 - a[flip], 1.0 - b[flip]
    v0, v1, v2 = np.array([0.0, VF]), np.array([TR, VC]), np.array([-TR, VC])
    return v0 + np.outer(a, v1 - v0) + np.outer(b, v2 - v0)


def report(tag, d):
    d = np.linalg.norm(np.atleast_2d(d), axis=1)
    print(f'{tag:36s} max {d.max():.3e}  p99 {np.quantile(d, 0.99):.3e}  '
          f'mean {d.mean():.3e}')


def main():
    t0 = time.perf_counter()
    w_fold = AuthalicWarp(NPZ, fold=True)
    t_fold = time.perf_counter() - t0
    print(f'fold build   {t_fold:8.1f} s   (wedge pts {len(w_fold.src)})')
    t0 = time.perf_counter()
    w_face = AuthalicWarp(NPZ, fold=False)
    t_face = time.perf_counter() - t0
    print(f'face build   {t_face:8.1f} s   (face pts  {len(w_face.src)})')

    # 1. vertex exactness
    report('fold @ wedge vertices vs dst',
           w_fold.do(w_fold.src, mo=0) - w_fold.dst)
    report('face @ face vertices vs dst',
           w_face.do(w_face.src, mo=0) - w_face.dst)

    # 2. fold vs face: interior + near-line bands
    pts = face_sample(200_000)
    report('fold vs face, interior 200k',
           w_fold.do(pts, mo=0) - w_face.do(pts, mo=0))
    for tag, p in (
            ('near x=0 median', np.column_stack(
                [rng.uniform(-2e-4, 2e-4, 20_000),
                 rng.uniform(VF * 0.98, VC * 0.98, 20_000)])),
            ('near right lateral edge', (lambda t, e: np.column_stack(
                [t, R3 * t - W + e]))(rng.uniform(1e-3, TR - 1e-3, 20_000),
                                      rng.uniform(0.0, 2e-4, 20_000))),
            ('near equator (inside)', np.column_stack(
                [rng.uniform(-TR * 0.9, TR * 0.9, 20_000),
                 VC - rng.uniform(0.0, 2e-4, 20_000)])),
    ):
        report(f'fold vs face, {tag}', w_fold.do(p, mo=0) - w_face.do(p, mo=0))

    # 3. D3-equivariance of fold evaluation: do(T p) == T do(p)
    r120 = np.array([[-0.5, -R3 / 2.0], [R3 / 2.0, -0.5]])
    s1 = np.diag([-1.0, 1.0])
    p = face_sample(50_000)
    dp = w_fold.do(p, mo=0)
    worst = max(np.abs(w_fold.do(p @ T.T, mo=0) - dp @ T.T).max()
                for T in (r120, r120.T, s1, s1 @ r120, s1 @ r120.T))
    print(f'fold D3-equivariance worst           {worst:.3e}')

    # 4. seam tangency: on-edge points stay on edge
    t = rng.uniform(0.0, TR, 20_000)
    edge = np.column_stack([t, R3 * t - W])
    for tag, w in (('fold', w_fold), ('face', w_face)):
        q = w.do(edge, mo=0)
        resid = np.abs(q[:, 1] - R3 * q[:, 0] + W)
        print(f'{tag} on-edge normal residual           max {resid.max():.3e}')

    # 5. displacement continuity across the x=0 fold line (compare deltas,
    # not positions — the ±eps input separation would otherwise dominate)
    eps = 1e-10
    y = rng.uniform(VF * 0.98, VC * 0.98, 20_000)
    pl = np.column_stack([-np.full_like(y, eps), y])
    pr = np.column_stack([np.full_like(y, eps), y])
    report('fold δ-continuity across x=0',
           (w_fold.do(pl, mo=0) - pl) - (w_fold.do(pr, mo=0) - pr))

    # 6. round trips (FINE tolerance), interior + near-corner
    for tag, p in (('interior 50k', face_sample(50_000)),
                   ('near-corner C', np.array([0.0, VF])
                    + 1e-3 * (face_sample(5_000) - np.array([0.0, VF])))):
        report(f'fold RT {tag}', w_fold.undo(w_fold.do(p, mo=0), mo=0) - p)

    # 7. bulk evaluation speed
    p = face_sample(500_000)
    t0 = time.perf_counter()
    w_fold.do(p, mo=0)
    tf = time.perf_counter() - t0
    t0 = time.perf_counter()
    w_face.do(p, mo=0)
    tc = time.perf_counter() - t0
    print(f'do() 500k: fold {tf:.2f} s ({tf / 0.5:.2f} µs/pt)  '
          f'face {tc:.2f} s ({tc / 0.5:.2f} µs/pt)  ratio {tc / tf:.2f}×')


if __name__ == '__main__':
    main()
