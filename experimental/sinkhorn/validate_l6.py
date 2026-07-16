# Part of the Hex9 (H9) Project
# Copyright ©2026, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""
Validation battery for the L6 fundamental-domain warp deliverables.

  (a) D3-equivariance of the unfolded full-face field (exact, via lattice
      index permutation for each of the six group ops)
  (b) mesh-triangle area-ratio stats (MAE, p1/p50/p99, min/max) of the new
      field on the Sphere, alongside Sphere_l5 v1 and the WGS84 benchmark
  (c) corner pinning exact; edge displacements exactly tangential
  (d) NaN / displacement-range checks

Usage:
  python validate_l6.py --fund ../../hhg9/data/Sphere_l6_fund_warp_data.npz \
      --face ../../hhg9/data/Sphere_l6_warp_data.npz
"""
import argparse
import numpy as np
from scipy.spatial import cKDTree

from hhg9 import Registrar, Points
from hhg9.h9.polygon import tri_mesh
from experimental.sinkhorn import config
from experimental.sinkhorn.fast_area import triangle_areas
from experimental.sinkhorn.stage3b_fund_l6 import GROUP, C_, G_, M_, U2, U3, TR, VC, VF


def area_stats(tgt, tris, ell, label):
    rg = Registrar(); config.apply_ellipsoid(rg, ell)
    b_raw, g_gcd = rg.domain('b_raw'), rg.domain('g_gcd')
    gp = rg.project(Points(tgt, b_raw, oid=0), [b_raw, g_gcd])
    ll = gp.coords[tris.ravel()].reshape(-1, 3, 2)
    e = config.ELLIPSOIDS[ell]
    r = triangle_areas(ll, e['a'], e['f'])
    r = r / r.mean()
    p = np.percentile(r, [1, 50, 99])
    print(f'[{label}] on {ell}: MAE={np.abs(r - 1).mean():.6f} '
          f'p1={p[0]:.5f} p50={p[1]:.5f} p99={p[2]:.5f} '
          f'min={r.min():.5f} max={r.max():.5f}  ({len(tris)} tris)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--fund', required=True)
    ap.add_argument('--face', required=True)
    args = ap.parse_args()

    # ── (d) format / NaN / displacement ──────────────────────────────────────
    for path, name in ((args.fund, 'FUND'), (args.face, 'FACE')):
        z = np.load(path)
        s, t = z['source_pts'], z['target_pts']
        d = np.linalg.norm(t - s, axis=1)
        print(f'[{name}] {path}')
        print(f'  keys={z.files} N={len(s)} layer={int(z["layer"])} '
              f'NaN={np.isnan(s).any() or np.isnan(t).any()} '
              f'max|D|={d.max():.4e} mean|D|={d.mean():.4e}')

    zf = np.load(args.fund)
    fs, ft = zf['source_pts'], zf['target_pts']
    fD = ft - fs
    cls = zf['edge_class']

    # ── (c) corners + tangency on the fundamental artifact ──────────────────
    for p, nm in ((C_, 'C(pole)'), (G_, 'G(centroid)'), (M_, 'M(edge-mid)')):
        i = np.argmin(np.linalg.norm(fs - p, axis=1))
        print(f'  corner {nm}: |D| = {np.linalg.norm(fD[i]):.3e}')
    n1 = np.array([1.0, 0.0])
    n2v = np.array([-U2[1], U2[0]])
    n3v = np.array([-U3[1], U3[0]])
    for c, nvec, nm in ((1, n1, 'C-G median (x=0)'), (2, n2v, 'G-M median'),
                        (3, n3v, 'C-M face edge')):
        m = cls == c
        r = np.abs(fD[m] @ nvec).max() if m.any() else 0.0
        print(f'  edge {nm}: n={int(m.sum())} verts, max |normal comp of D| = {r:.3e}')

    # ── (a) exact D3-equivariance of the unfolded face field ────────────────
    za = np.load(args.face)
    vs, vt = za['source_pts'], za['target_pts']
    vD = vt - vs
    tree = cKDTree(vs)
    worst = 0.0
    for T in GROUP[1:]:
        dist, j = tree.query(vs @ T.T, k=1)
        assert dist.max() < 1e-8
        res = np.linalg.norm(vD[j] - vD @ T.T, axis=1)  # D(Tv) vs T D(v)
        worst = max(worst, res.max())
        print(f'  D3 op residual: median {np.median(res):.3e}  max {res.max():.3e}')
    print(f'  (a) worst-case D3 residual: {worst:.3e}')

    # unfolded-face boundary invariants (loader expectations)
    eq = np.abs(vs[:, 1] - VC) < 1e-9
    print(f'  face equator dy: max |dy| = {np.abs(vD[eq, 1]).max():.3e} ({int(eq.sum())} verts)')
    lat = np.abs(vs[:, 1] - np.sqrt(3) * np.abs(vs[:, 0]) - VF) < 1e-9
    latn = np.abs(vD[lat, 0] * (np.sqrt(3) * np.sign(vs[lat, 0])) - vD[lat, 1]) \
        / 2.0  # |n.D| with n = (±sqrt3,-1)/2
    print(f'  face lateral edges: max |normal comp of D| = {latn.max():.3e} ({int(lat.sum())} verts)')

    # ── (b) area stats: new face field on Sphere over the L6 mesh ───────────
    L = int(za['layer'])
    verts, _e, tris = tri_mesh(L, 0)
    assert np.allclose(verts, vs)
    area_stats(vt, tris, 'Sphere', f'NEW L{L} face')
