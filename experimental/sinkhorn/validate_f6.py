# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""
F6 acceptance validation for a packed warp file (lateral-edge tangency).

Checks, per libhex9 docs/warp-retrain-brief.md, all in H9_WARP_EDGE=raw:

1. data tangency    — on-edge vertex deltas: normal component ~ 0.
2. CT tangency      — delta evaluated BETWEEN edge vertices via the production
                      Clough-Tocher path: |normal| <= ~1e-9 units along both
                      lateral edges (the shipped field failed this at ~3.2e-7
                      near the apex despite zero vertex deltas).
3. frame agreement  — dx antisymmetric / dy symmetric under x-mirror on the
                      edges, so adjacent octant frames agree about seam points.
4. equator symmetry — dy = 0 on the equator line (data + CT).
5. needle scan      — per-vertex |delta| and vertex-to-neighbour delta jumps
                      (gradient spikes) over the whole field.
6. round-trip       — production undo(do(p)) on L5 vertices, raw mode.
7. seam render scan — b_oct -> g_gcd transect at lat 51.48 / +-89.1 with the
                      candidate field injected (probe_seam_band logic).

Usage:
    python validate_f6.py --warp WGS84_l5_warp_data.npz
"""
import argparse
import os
from pathlib import Path

os.environ['H9_WARP_EDGE'] = 'raw'          # must precede hhg9 import

import numpy as np

from hhg9 import Registrar
from hhg9.base.points import Points
from hhg9.domains.octahedral_barycentric import AuthalicWarp, LATERAL_EDGE_MODE
from hhg9.h9 import H9K
from experimental.sinkhorn import config

R3, W_ = H9K.radical.R3, H9K.Ẇ
UNIT_M = 7.076e6           # 1 b_oct unit ~ 7,076 km (octant edge = sqrt(2))
M_PER_DEG = 111_319.49

TANGENCY_TOL = 2e-9        # units; brief asks ~1e-9 (~7 mm)


def edge_frames():
    """(corner_eq, corner_apex, unit tangent, unit normal) for right and left
    lateral edges of the mode-0 triangle."""
    tr, vf, vc = H9K.limits.TR, H9K.limits.VF, H9K.limits.VC
    apex = np.array([0.0, vf])
    out = {}
    for name, eq_corner in (('right', np.array([tr, vc])),
                            ('left',  np.array([-tr, vc]))):
        t = apex - eq_corner
        t = t / np.linalg.norm(t)
        n = np.array([-t[1], t[0]])
        out[name] = (eq_corner, apex, t, n)
    return out


def check(label, value, tol):
    flag = 'PASS' if value <= tol else f'FAIL (> {tol:.1e})'
    print(f'  {label:<46} {value:.3e}  {flag}')
    return value <= tol


def main(warp_path: Path, ellipsoid: str = 'WGS84'):
    print(f'validate_f6: {warp_path}   (edge mode: {LATERAL_EDGE_MODE}, '
          f'ellipsoid: {ellipsoid})')
    assert LATERAL_EDGE_MODE == 'raw'
    warp = AuthalicWarp(warp_path)
    src, dst = warp.src, warp.dst
    delta = dst - src
    ok = True

    # ── 1+3+4. data-level checks ─────────────────────────────────────────────
    print('\n[1] vertex-data tangency / symmetry')
    lat = config.lateral_edge_mask(src)
    fr = edge_frames()
    for name in ('right', 'left'):
        _, _, t, n = fr[name]
        sgn = 1.0 if name == 'right' else -1.0
        on = (np.abs(src[:, 1] - R3 * sgn * src[:, 0] + W_) < 1e-5) \
             & (sgn * src[:, 0] >= -1e-12)
        dn = np.abs(delta[on] @ n).max() if on.any() else np.nan
        ok &= check(f'on-{name}-edge max |normal delta| (units)', dn, TANGENCY_TOL)
        print(f'    ({int(on.sum())} vertices, max |tangential| '
              f'{np.abs(delta[on] @ t).max() * UNIT_M:.1f} m)')

    # mirror pairing
    from scipy.spatial import cKDTree
    mirror_q = np.column_stack([-src[:, 0], src[:, 1]])
    _, mi = cKDTree(src).query(mirror_q, k=1)
    ok &= check('mirror dx antisymmetry max |dx + dx_m|',
                np.abs(delta[:, 0] + delta[mi, 0]).max(), 1e-12)
    ok &= check('mirror dy symmetry     max |dy - dy_m|',
                np.abs(delta[:, 1] - delta[mi, 1]).max(), 1e-12)

    eq = np.abs(src[:, 1] - H9K.limits.VC) < 1e-9
    ok &= check(f'equator vertices ({int(eq.sum())}) max |dy|',
                np.abs(delta[eq, 1]).max(), 1e-12)

    # ── 2. CT-evaluated on-edge tangency (the real F6 criterion) ────────────
    print('\n[2] CT-evaluated on-edge tangency (raw, 20k samples/edge)')
    for name in ('right', 'left'):
        c_eq, apex, t, n = fr[name]
        s = np.linspace(0.0, 1.0, 20001)[:, None]
        pts = c_eq * (1 - s) + apex * s
        d = warp.do(pts, mo=0) - pts
        dn = d @ n
        k = int(np.argmax(np.abs(dn)))
        ok &= check(f'on-{name}-edge max |CT normal| (units)',
                    np.abs(dn).max(), TANGENCY_TOL)
        print(f'    worst at edge-param {s[k, 0]:.4f} (y={pts[k, 1]:+.4f}); '
              f'= {np.abs(dn).max() * UNIT_M * 1000:.3f} mm ground')

    # frame agreement on the seam: left-edge field vs mirrored right-edge field
    print('\n[3] cross-frame seam agreement (CT, 20k samples)')
    c_eq, apex, t, n = fr['left']
    s = np.linspace(0.0, 1.0, 20001)[:, None]
    pl = c_eq * (1 - s) + apex * s
    pr = np.column_stack([-pl[:, 0], pl[:, 1]])
    dl = warp.do(pl, mo=0) - pl
    dr = warp.do(pr, mo=0) - pr
    mism = np.hypot(dl[:, 0] + dr[:, 0], dl[:, 1] - dr[:, 1])
    ok &= check('max frame mismatch (units)', mism.max(), TANGENCY_TOL)
    print(f'    = {mism.max() * UNIT_M * 1000:.3f} mm ground')

    # equator CT dy
    tr, vc = H9K.limits.TR, H9K.limits.VC
    xs = np.linspace(-tr, tr, 20001)
    pe = np.column_stack([xs, np.full_like(xs, vc)])
    de = warp.do(pe, mo=0) - pe
    ok &= check('[4] equator-line max |CT dy| (units)',
                np.abs(de[:, 1]).max(), TANGENCY_TOL)

    # ── 5. needle / gradient-spike scan ──────────────────────────────────────
    print('\n[5] needle scan (vertex deltas + neighbour jumps)')
    grid = np.load(config.SINKHORN_DIR / 'grid_l5.npz', allow_pickle=True)
    tris = grid['grid']
    mag = np.linalg.norm(delta, axis=1)
    print(f'  |delta|: max {mag.max():.4e} units ({mag.max()*UNIT_M/1000:.1f} km), '
          f'p99.9 {np.quantile(mag, 0.999):.4e}')
    e01 = tris[:, [0, 1]]; e12 = tris[:, [1, 2]]; e20 = tris[:, [2, 0]]
    edges = np.vstack([e01, e12, e20])
    jump = np.linalg.norm(delta[edges[:, 0]] - delta[edges[:, 1]], axis=1)
    q = np.quantile(jump, [0.5, 0.999, 1.0])
    print(f'  neighbour jump: median {q[0]:.3e}, p99.9 {q[1]:.3e}, '
          f'max {q[2]:.3e} units (max/median {q[2]/max(q[0],1e-30):.1f}x)')
    spike = q[2] / max(q[1], 1e-30)
    ok &= check('max/p99.9 neighbour-jump ratio (spike check)', spike, 10.0)

    # ── 6. round-trip, raw mode ──────────────────────────────────────────────
    print('\n[6] round-trip undo(do(p)) on L5 vertices (raw Newton)')
    b = warp.do(src, mo=0)
    back = warp.undo(b, mo=0)
    err = np.linalg.norm(back - src, axis=1)
    for label, m in (('interior', ~lat), ('lateral-edge', lat)):
        print(f'  {label:<14} max {err[m].max():.3e}  mean {err[m].mean():.3e} '
              f'units (max {err[m].max()*UNIT_M*1e9:.1f} nm ground)')
    ok &= check('round-trip max (units)', err.max(), 1e-12)

    # ── 7. seam render transect ──────────────────────────────────────────────
    print('\n[7] seam render transect (b_oct -> g_gcd, candidate field injected)')
    reg = Registrar()
    config.apply_ellipsoid(reg, ellipsoid)
    bo = reg.domain('b_oct')
    gg = reg.domain('g_gcd')
    bo.set_warp(warp_path)
    for latd, lon_hi in ((51.48, 2e-5), (89.1, 2e-3), (-89.1, 2e-3)):
        anchors = Points(np.array([[latd, 1e-9], [latd, lon_hi]]), domain=gg)
        bp = reg.project(anchors, [gg, bo])
        bb, oids = bp.coords, bp.oid
        tt = np.linspace(0.0, 1.0, 2001)[:, None]
        seg = bb[0] * (1 - tt) + bb[1] * tt
        ll = reg.project(Points(seg, domain=bo, oid=np.full(2001, oids[0])),
                         [bo, gg]).coords
        dlat = np.diff((ll[:, 0] - latd) * M_PER_DEG)
        dlon = np.diff(ll[:, 1] * M_PER_DEG * np.cos(np.radians(latd)))
        step = np.hypot(dlat, dlon)
        ratio = step.max() / max(np.median(step), 1e-30)
        ok &= check(f'lat {latd:+.2f}: max/median transect step', ratio, 10.0)

    print(f'\n=== F6 validation: {"PASS" if ok else "FAIL"} ===')
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--warp', type=Path,
                    default=config.SINKHORN_DIR / 'WGS84_l5_warp_data.npz')
    ap.add_argument('--ellipsoid', default='WGS84',
                    choices=list(config.ELLIPSOIDS),
                    help='registrar ellipsoid for the transect check [7]; '
                         'checks [1]-[6] are ellipsoid-agnostic')
    args = ap.parse_args()
    main(args.warp, args.ellipsoid)
