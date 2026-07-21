# Part of the Hex9 (H9) Project
# Copyright ©2026, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
PKDO feasibility: can a compact analytic (polynomial) coefficient set
regenerate the canonical L6 fundamental-domain warp artifact,
hhg9/data/Sphere_l6_fund_warp_data.npz?

VERDICT (2026-07-21): NO — not at production fidelity. The 2026-04
conclusion in ex0065_octproj_c.py (global polynomial vs L5 CT) survives
the L6 fund field unchanged, and the numbers below make it stark.

METHOD
The fund wedge (399,675 pts, layer 6) is unfolded to the full mode-0
face (2,394,766 pts) via hhg9's own `_unfold_fund_face`. A PKDO basis
(Duffy-collapsed Jacobi, L2-orthonormal on the face triangle, built
from the actual face corners rather than ex0065's hardcoded map) is
ridge-fit to the displacement field on a ~160k sample containing every
boundary point and every point within 0.03 of a face corner, with
weights 1e4 (edges) / 1e6 (corner discs). Residuals are evaluated on
all 2.4M points. Metres via edge-arc scale: 1 b_oct unit ≈ 7,076 km
(octahedron edge subtends 90° on R = 6,371,007 m).

RESULTS (field magnitude for scale: |disp| max 68.5 km, mean 39.9 km)

  deg  coefs  size    max      p99      rms     vtx<.005  ring<.03  edge-leak
   12    182  1.4 KB  140.5 km  98.1 km 20.2 km   1.27 km  57.3 km  254 m
   20    462  3.6 KB   58.5 km  24.5 km  5.9 km   0.97 km  35.8 km  669 m
   32  1,122  8.8 KB   51.8 km  12.8 km  3.3 km   0.51 km  26.9 km  128 m
   44  2,070   16 KB   25.3 km   5.4 km  1.5 km   0.41 km  16.3 km   95 m
   56  3,306   26 KB    9.9 km   2.5 km  639 m    0.23 km   8.4 km   58 m

KEY FINDINGS
1. Convergence is algebraic, not spectral: deg 12→56 (18× coefficients)
   buys only ~14× in max error. Extrapolating to even 10 m max implies
   degree in the several hundreds — float64 conditioning dies first.
   The representation never reaches the 1e-16-parity class of the
   CT/fold pipeline.
2. Vertex loading: the displacement *magnitude* at the corners is
   moderate (7.1 km near C, 11.9 km near M, 2.1 km near G) — it is the
   curvature concentration there that a global polynomial cannot carry.
   With 1e6 corner weights the corner points themselves hold to ~230 m,
   but the error relocates to an ~8 km residual ring just outside the
   weighted disc (classic Runge redistribution). You choose *where* the
   polynomial is wrong near a corner, not *whether*.
3. Edge tangency: the fund data itself is edge-tangent to 2.6e-16
   b_oct (the slide-field gradient projection holds exactly). The
   polynomial leaks ~58 m normal to face edges even at deg 56 — this
   alone breaks cross-octant consistency.
4. Genuine remaining PKDO use is unchanged from ex0065: a smooth
   analytic Jacobian for metric-tensor / area-distortion analysis
   where differentiability matters more than authalic fidelity.

IF A COMPACT ANALYTIC FORM IS EVER TRULY NEEDED (research, not a
nice-to-have): corner-adapted basis functions (singular / r^α-type
terms matched to the smoothed cone-point character), per-subtriangle
spectral elements, or a wedge-only fit with explicit fold boundary
conditions. A plain global polynomial is the wrong tool.

ARTIFACT COMPRESSION NOTES (2026-07-21)
- The npz is now stored with np.savez_compressed: 13.2 → 7.2 MB. Zero
  new dependencies (zlib via stdlib zipfile); np.load reads both
  formats transparently; loaders never mmap. Arrays verified
  byte-identical across the swap; fold-warp build time unchanged
  (~12 s); do/undo round-trip 2.8e-17.
- Budget within the 7.2 MB: target_pts 5.87 MB (trained field, near
  incompressible), source_pts 1.30 MB. Two further options exist for
  source_pts, both sound but LEFT AS-IS (Ben, 2026-07-21 — real value
  in the artifact staying explicit and self-describing):
    (a) re-derive from the base grid: source_pts = the wedge subset of
        tri_mesh(layer, 0); `_unfold_fund_face` already KD-matches at
        1e-8 tolerance, so regeneration is safe by construction;
    (b) L6 UV encoding: source_pts lie exactly on the L6 UV lattice —
        x = i·(H9K.lattice.U/3^6), y = j·(H9K.lattice.V/3^6) with
        integer (i, j) to ~1e-12 of a step; i ∈ 0..1093,
        j ∈ −4374..0, so an int16 UV pair per point reconstructs
        them to ~1e-16 absolute.
  Either saves only ~1.2 MB of 7.2; the trained field dominates.

Last Tested
2026-07-21 (Python 3.12, numpy/scipy/sklearn; ~20 min full sweep)
"""

import time
import numpy as np
from importlib import resources as pkg_resources
from scipy.special import eval_jacobi
from sklearn.linear_model import Ridge

from hhg9.domains.octahedral_barycentric import _unfold_fund_face, H9K

R_SPHERE = 6371007.181
DEGREES = [12, 20, 32, 44, 56]


def load_fund():
    f = pkg_resources.files('hhg9.data').joinpath('Sphere_l6_fund_warp_data.npz')
    d = np.load(f)
    return d['source_pts'], d['target_pts'], int(d['layer'])


def face_corners():
    """Full-face triangle corners: A/B equatorial, C pole."""
    c = np.array([0.0, H9K.limits.VF])
    b = np.array([H9K.limits.TR, H9K.limits.VC])
    a = np.array([-H9K.limits.TR, H9K.limits.VC])
    return a, b, c


def pkdo_vandermonde(xy, degree, corners):
    """PKDO basis on the triangle spanned by `corners` (A, B, C).

    Barycentric → collapsed (r,s) → Duffy a; L2-orthonormal with
    C_{p,q} = sqrt((2p+1)(p+q+1)/2), as in ex0065 / the fossil
    _pkdo_vandermonde, but with the affine map derived from the actual
    corners instead of hardcoded."""
    A, B, C = corners
    T = np.linalg.inv(np.column_stack([B - A, C - A]))
    lam23 = (xy - A) @ T.T
    l2, l3 = lam23[:, 0], lam23[:, 1]
    l1 = 1.0 - l2 - l3
    r = -l1 + l2 - l3
    s = np.clip(-l1 - l2 + l3, -1.0, 1.0 - 1e-14)
    a = np.clip(2.0 * (1.0 + r) / (1.0 - s) - 1.0, -1.0, 1.0)
    nb = (degree + 1) * (degree + 2) // 2
    V = np.empty((len(r), nb))
    col = 0
    for p in range(degree + 1):
        Jph = eval_jacobi(p, 0, 0, a) * ((1.0 - s) / 2.0) ** p
        for q in range(degree + 1 - p):
            norm = np.sqrt((2 * p + 1) * (p + q + 1) / 2.0)
            V[:, col] = norm * Jph * eval_jacobi(q, 2 * p + 1, 0, s)
            col += 1
    return V


def main():
    f_src, f_tgt, layer = load_fund()
    print(f'wedge pts {f_src.shape[0]}  layer {layer}')
    src, dst = _unfold_fund_face(f_src, f_tgt, layer)
    disp = dst - src
    n = len(src)
    print(f'full face pts {n}')

    A, B, C = corners = face_corners()
    m_per_bu = R_SPHERE * (np.pi / 2) / np.linalg.norm(B - A)

    # barycentric edge/corner classification
    T = np.linalg.inv(np.column_stack([B - A, C - A]))
    lam23 = (src - A) @ T.T
    lams = [1.0 - lam23[:, 0] - lam23[:, 1], lam23[:, 0], lam23[:, 1]]
    edge_id = np.full(n, -1)
    for i, lam in enumerate(lams):          # 0: BC, 1: AC, 2: AB
        edge_id[np.abs(lam) < 1e-9] = i
    is_edge = edge_id >= 0
    cdist = np.min(np.linalg.norm(src[:, None, :] - np.array(corners)[None], axis=2), axis=1)

    edge_normals = []
    for (P, Q, O) in [(B, C, A), (A, C, B), (A, B, C)]:
        t = (Q - P) / np.linalg.norm(Q - P)
        nrm = np.array([-t[1], t[0]])
        edge_normals.append(-nrm if nrm @ (O - P) > 0 else nrm)
    data_leak = max(np.abs(disp[edge_id == i] @ edge_normals[i]).max() for i in range(3))
    print(f'data edge-normal leakage max {data_leak:.3e} b_oct')

    rng = np.random.default_rng(9)
    fit_idx = np.unique(np.concatenate([
        rng.choice(n, 150_000, replace=False),
        np.flatnonzero(is_edge),
        np.flatnonzero(cdist < 0.03)]))
    w = np.ones(len(fit_idx))
    w[is_edge[fit_idx]] = 1e4
    w[cdist[fit_idx] < 0.005] = 1e6
    print(f'fit sample {len(fit_idx)}')

    for degree in DEGREES:
        t0 = time.time()
        Vf = pkdo_vandermonde(src[fit_idx], degree, corners)
        cx = Ridge(alpha=1e-12, fit_intercept=False).fit(Vf, disp[fit_idx, 0], sample_weight=w).coef_
        cy = Ridge(alpha=1e-12, fit_intercept=False).fit(Vf, disp[fit_idx, 1], sample_weight=w).coef_
        del Vf

        res = np.empty(n)
        poly_leak = 0.0
        for a0 in range(0, n, 150_000):
            sl = slice(a0, min(a0 + 150_000, n))
            V = pkdo_vandermonde(src[sl], degree, corners)
            pd = np.column_stack([V @ cx, V @ cy])
            r = disp[sl] - pd
            res[sl] = np.hypot(r[:, 0], r[:, 1])
            for i in range(3):
                m = edge_id[sl] == i
                if m.any():
                    poly_leak = max(poly_leak, np.abs(pd[m] @ edge_normals[i]).max())

        nb = len(cx)
        print(f'deg {degree:3d} ({2 * nb} coef, {2 * nb * 8 / 1024:.1f} KB) '
              f'| max {res.max() * m_per_bu:9.1f} m '
              f'| p99 {np.percentile(res, 99) * m_per_bu:8.1f} m '
              f'| rms {np.sqrt(np.mean(res ** 2)) * m_per_bu:8.2f} m '
              f'| vtx<.005 {res[cdist < 0.005].max() * m_per_bu:7.1f} m '
              f'| ring<.03 {res[cdist < 0.03].max() * m_per_bu:8.1f} m '
              f'| edge-leak {poly_leak * m_per_bu:6.2f} m '
              f'| {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
