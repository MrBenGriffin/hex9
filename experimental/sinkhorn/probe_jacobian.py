# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""
Dense Jacobian sweep of the forward warp — the "no cliffs anywhere" check.

validate_f6.py proves the field is clean *on the lateral edges* (CT-sampled) and
that L5 *vertices* have no neighbour jumps. But the original seam cliff was
sub-triangle (zero at vertices, a bulge between them), so a vertex-resolution
scan can miss it. This probe closes that gap: it evaluates the Jacobian of the
forward map  do(p) = p + δ(p)  on a fine barycentric grid covering the *whole*
octant interior (the lateral edges are the render seams, so this subsumes the
seam transects), and reports:

    det(J) > 0  everywhere   → no folds / inversions (a fold is the continuous
                               analogue of the cliff: where the map turns over).
    σ_max/σ_min  bounded      → no near-folds / shear spikes (anisotropy → ∞ is
                               the smooth precursor of a cliff).

The field is C1 (Clough–Tocher) in raw mode, so J is continuous and central
differences are valid even across mesh edges; a literal discontinuity can no
longer occur by construction — this bounds the remaining (continuous) risk.

Memory-slim: the grid is streamed one barycentric row at a time and only running
reductions are kept (worst det, worst anisotropy + their locations, fold/NaN
counts). Peak memory is one row × 4 finite-difference offsets, independent of N.

Usage:
    python probe_jacobian.py --warp WGS84_l5_warp_data.npz --n 600
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault('H9_WARP_EDGE', 'raw')      # must precede hhg9 import

import numpy as np

from hhg9.domains.octahedral_barycentric import AuthalicWarp, LATERAL_EDGE_MODE
from hhg9.h9 import H9K
from experimental.sinkhorn import config

UNIT_M = 7.076e6           # 1 b_oct unit ≈ 7,076 km (octant edge = √2)

FOLD_TOL = 0.0             # det(J) must be strictly > this
ANIS_TOL = 4.0            # σ_max/σ_min report gate (a near-fold blows past this)
CORNER_R = 0.05           # units; radius of the corner-local worst-case report


def corners():
    """Apex + the two equator corners of the mode-0 octant triangle."""
    tr, vf, vc = H9K.limits.TR, H9K.limits.VF, H9K.limits.VC
    return {'apex': np.array([0.0, vf]),
            'right': np.array([tr, vc]),
            'left':  np.array([-tr, vc])}


def jacobian_stats(warp, pts, h):
    """Per-point det(J) and anisotropy σ_max/σ_min via central differences.

    Returns (det, anis) arrays; non-finite where the interpolant is undefined.
    """
    ex = np.array([h, 0.0]); ey = np.array([0.0, h])
    # ∂(do)/∂x and ∂(do)/∂y, each a 2-vector [∂X, ∂Y] per point.
    jx = (warp.do(pts + ex, mo=0) - warp.do(pts - ex, mo=0)) / (2.0 * h)
    jy = (warp.do(pts + ey, mo=0) - warp.do(pts - ey, mo=0)) / (2.0 * h)
    a, c = jx[:, 0], jx[:, 1]      # column ∂/∂x  → [a; c]
    b, d = jy[:, 0], jy[:, 1]      # column ∂/∂y  → [b; d]
    det = a * d - b * c
    fro = a * a + b * b + c * c + d * d
    disc = np.sqrt(np.maximum(fro * fro - 4.0 * det * det, 0.0))
    s_max2 = 0.5 * (fro + disc)
    s_min2 = np.maximum(0.5 * (fro - disc), 0.0)
    anis = np.sqrt(s_max2 / np.maximum(s_min2, 1e-300))
    return det, anis


def main(warp_path: Path, n: int, h: float):
    assert LATERAL_EDGE_MODE == 'raw', f'need raw edge mode, got {LATERAL_EDGE_MODE}'
    print(f'probe_jacobian: {warp_path}  (edge mode {LATERAL_EDGE_MODE}, '
          f'grid n={n}, fd h={h:g})')
    warp = AuthalicWarp(warp_path)

    cs = corners()
    apex, rc, lc = cs['apex'], cs['right'], cs['left']

    # running reductions (memory-slim — never hold the whole grid)
    min_det, min_det_pt = np.inf, None
    max_anis, max_anis_pt = 0.0, None
    n_pts = n_fold = n_nan = 0
    corner_worst = {k: dict(det=np.inf, anis=0.0) for k in cs}

    for i in range(n + 1):                      # barycentric row a = i/n
        m = n - i + 1
        if m <= 0:
            continue
        a = i / n
        bs = np.arange(m) / n                   # b = j/n,  c = 1 - a - b
        row = (a * apex[None, :]
               + bs[:, None] * rc[None, :]
               + (1.0 - a - bs)[:, None] * lc[None, :])

        det, anis = jacobian_stats(warp, row, h)
        finite = np.isfinite(det) & np.isfinite(anis)
        n_pts += int(finite.sum())
        n_nan += int((~finite).sum())
        if not finite.any():
            continue
        det_f, anis_f, row_f = det[finite], anis[finite], row[finite]
        n_fold += int((det_f <= FOLD_TOL).sum())

        k = int(np.argmin(det_f))
        if det_f[k] < min_det:
            min_det, min_det_pt = float(det_f[k]), row_f[k].copy()
        k = int(np.argmax(anis_f))
        if anis_f[k] > max_anis:
            max_anis, max_anis_pt = float(anis_f[k]), row_f[k].copy()

        # corner-local worst case
        for name, cpt in cs.items():
            near = np.hypot(row_f[:, 0] - cpt[0], row_f[:, 1] - cpt[1]) < CORNER_R
            if near.any():
                cw = corner_worst[name]
                cw['det'] = min(cw['det'], float(det_f[near].min()))
                cw['anis'] = max(cw['anis'], float(anis_f[near].max()))

    print(f'\nswept {n_pts:,} interior points ({n_nan} non-evaluable)')
    print(f'  min det(J)        {min_det:+.5f}  at {min_det_pt}')
    print(f'  max anisotropy    {max_anis:8.3f}  at {max_anis_pt}')
    print(f'  folds (det<=0)    {n_fold}')
    print(f'\ncorner-local (r<{CORNER_R}):')
    for name in ('apex', 'right', 'left'):
        cw = corner_worst[name]
        print(f'  {name:<6} min det {cw["det"]:+.5f}   max anis {cw["anis"]:8.3f}')

    ok = (n_fold == 0) and (min_det > FOLD_TOL) and (max_anis <= ANIS_TOL)
    print(f'\n  {"PASS" if n_fold == 0 and min_det > FOLD_TOL else "FAIL"}'
          f'  no folds (det>0 everywhere)')
    print(f'  {"PASS" if max_anis <= ANIS_TOL else "WARN"}'
          f'  anisotropy <= {ANIS_TOL} (σ_max/σ_min)')
    print(f'\n=== Jacobian sweep: {"PASS" if ok else "REVIEW"} ===')
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--warp', type=Path,
                    default=config.SINKHORN_DIR / 'WGS84_l5_warp_data.npz')
    ap.add_argument('--n', type=int, default=600,
                    help='barycentric grid resolution (points ≈ (n+1)(n+2)/2)')
    ap.add_argument('--h', type=float, default=1e-5,
                    help='central-difference step in b_oct units')
    args = ap.parse_args()
    main(args.warp, args.n, args.h)
