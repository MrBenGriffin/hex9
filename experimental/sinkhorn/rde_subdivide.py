# Part of the Hex9 (H9) Project
# Copyright ©2026, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""
Adaptive 9-tree localisation of the residual density error (RDE).

The octant triangle subdivides exactly into 9 equal-b_oct-area children
(edge trisection), matching the t-grid hierarchy, so a parent's cell
average is exactly the mean of its children's. Two phases:

1. UNIFORM TABLE — evaluate pointwise density at every depth-(T+1)
   child centroid (9-point quadrature of every depth-T cell), aggregate
   upward, and print per-depth cell-average deviation stats: the
   "finer layers" figures without any polygon areas. (A depth-d
   triangle has 1.5x the b_oct area of a depth-d hex; distributions are
   comparable, not identical.)

2. ADAPTIVE REFINEMENT — refine a cell only while it is UNRESOLVED:
   its 9 child samples disagree by more than max(--tol, --rel x its own
   |dev|). Resolved cells become leaves whether hot or quiet, so a
   finite-width hot band is MEASURED (leaf areas above thresholds, band
   widths) rather than tiled at max depth; only genuinely singular
   structure (the corner cores, where spread ~ |dev| self-similarly)
   refines all the way down. Leaves are classified by feature (polar
   corner, equatorial corner, lateral seam, equator edge, interior).

    python rde_subdivide.py --ellipsoid Sphere --table-depth 6 \\
        --max-depth 10 --tol 5e-4 --rel 0.25

Outputs: output/rde_tree_<ELL>.npz + stdout report.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hhg9 import Registrar, Points
from experimental.sinkhorn import config
from experimental.sinkhorn.rde_field import (
    CORNERS, A_TRI, A_BOCT_TOTAL, TR, VC,
    density_jacobian, _child_centroid_weights)


def _child_vertex_weights() -> np.ndarray:
    """(9, 3, 3): barycentric weights of each child triangle's vertices."""
    def w(i, j):
        return [1.0 - (i + j) / 3.0, i / 3.0, j / 3.0]
    out = []
    for i in range(3):
        for j in range(3 - i):
            out.append([w(i, j), w(i + 1, j), w(i, j + 1)])          # upward
            if i + j < 2:
                out.append([w(i + 1, j), w(i, j + 1), w(i + 1, j + 1)])  # down
    arr = np.array(out)
    assert arr.shape == (9, 3, 3)
    return arr


W_VERT = _child_vertex_weights()        # (9 children, 3 verts, 3 parent wts)
W_CENT = _child_centroid_weights()      # (9 children, 3 parent wts)


def subdivide(tris: np.ndarray) -> np.ndarray:
    """(T,3,2) -> (9T,3,2), parent-major order."""
    return np.einsum('kwv,tvc->tkwc', W_VERT, tris).reshape(-1, 3, 2)


def child_centroids(tris: np.ndarray) -> np.ndarray:
    """(T,3,2) -> (T,9,2): centroids of each triangle's 9 children."""
    return np.einsum('kv,tvc->tkc', W_CENT, tris)


def classify(xy: np.ndarray, corner_r: float = 0.02,
             band_w: float = 0.02) -> np.ndarray:
    """Feature class per point: 0 polar corner, 1 equatorial corner,
    2 lateral seam, 3 equator edge, 4 interior.

    Physical halo radii (b_oct units): the boundary-layer structure
    extends a few L5 training cells (~2.7e-3) from each feature, so the
    default 0.02 swath comfortably contains it; corners override edges.
    """
    d_pole = np.linalg.norm(xy - CORNERS[2], axis=1)
    d_eq_c = np.minimum(np.linalg.norm(xy - CORNERS[0], axis=1),
                        np.linalg.norm(xy - CORNERS[1], axis=1))
    d_lat = config.lateral_distance(xy)
    d_equ = VC - xy[:, 1]
    cls = np.full(len(xy), 4)
    cls[d_equ < band_w] = 3
    cls[d_lat < band_w] = 2
    cls[d_eq_c < corner_r] = 1
    cls[d_pole < corner_r] = 0
    return cls


CLS_NAMES = ['polar-corner', 'equatorial-corner', 'lateral-seam',
             'equator-edge', 'interior']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ellipsoid', default='WGS84', choices=list(config.ELLIPSOIDS))
    ap.add_argument('--table-depth', type=int, default=6)
    ap.add_argument('--max-depth', type=int, default=8,
                    help='deepest refinement; below the L5 training lattice '
                         '(~2.7e-3 units) the pointwise field has structural '
                         'texture, so depths past ~8 multiply cells without '
                         'new information outside the corner cores')
    ap.add_argument('--tol', type=float, default=5e-4,
                    help='absolute child-spread floor for refinement')
    ap.add_argument('--rel', type=float, default=0.25,
                    help='relative spread/|dev| floor for refinement')
    args = ap.parse_args()

    reg = Registrar()
    config.apply_ellipsoid(reg, args.ellipsoid)
    ideal = reg.ellipsoid_area / A_BOCT_TOTAL
    print(f'ellipsoid {reg.ellipsoid_name}: ideal density {ideal:.6e} m²/unit²')

    # ── phase 1: uniform table ───────────────────────────────────────────────
    tris = CORNERS[None, :, :]
    for _ in range(args.table_depth):
        tris = subdivide(tris)
    T = len(tris)
    size0 = 2.0 * TR / 3 ** args.table_depth          # cell edge length
    print(f'uniform: {T} depth-{args.table_depth} cells; evaluating '
          f'{9 * T} child centroids...')
    cc = child_centroids(tris)                         # (T,9,2)
    h = max(1e-8, size0 / 60.0)
    dens9, _, _ = density_jacobian(reg, cc.reshape(-1, 2), h)
    dens9 = dens9.reshape(T, 9)

    print(f'\nper-depth cell-average deviation (triangle cells, ~1.5x the '
          f'hex area of the same depth):')
    print(f'{"depth":>5} {"cells/oct":>10} {"sigma(log)":>11} {"min%":>9} '
          f'{"max%":>9} {"p50|%|":>9} {"p99|%|":>9} {"p99.99|%|":>10}')
    flat = dens9.reshape(-1)
    for d in range(args.table_depth, -1, -1):
        avg = flat.reshape(9 ** d, -1).mean(axis=1)
        r = avg / ideal
        dev = (r - 1.0) * 100.0
        ad = np.abs(dev)
        print(f'{d:>5} {9**d:>10} {np.log(r).std():>11.3e} {dev.min():>+9.4f} '
              f'{dev.max():>+9.4f} {np.percentile(ad,50):>9.5f} '
              f'{np.percentile(ad,99):>9.5f} {np.percentile(ad,99.99):>10.5f}')

    # ── phase 2: adaptive refinement ─────────────────────────────────────────
    print(f'\nadaptive refinement: refine while child spread > '
          f'max({args.tol:g}, {args.rel:g}·|dev|), to depth {args.max_depth}')

    def split(tris_d, dev_d, depth):
        """Partition cells into (leaves, still-unresolved)."""
        spread = dev_d.max(axis=1) - dev_d.min(axis=1)
        amax = np.abs(dev_d).max(axis=1)
        unresolved = spread > np.maximum(args.tol, args.rel * amax)
        if depth >= args.max_depth:
            unresolved[:] = False
        return ~unresolved, unresolved

    leaves = []        # (depth, centroid, mean_dev, spread)

    def add_leaves(tris_d, dev_d, mask, depth):
        if mask.any():
            leaves.append((np.full(mask.sum(), depth),
                           tris_d[mask].mean(axis=1),
                           dev_d[mask].mean(axis=1),
                           dev_d[mask].max(axis=1) - dev_d[mask].min(axis=1)))

    depth = args.table_depth
    dev9 = dens9 / ideal - 1.0
    done, working = split(tris, dev9, depth)
    add_leaves(tris, dev9, done, depth)
    work = tris[working]
    print(f'  depth {depth}: {working.sum()}/{T} unresolved')

    while depth < args.max_depth and len(work):
        depth += 1
        tris_d = subdivide(work)
        size = 2.0 * TR / 3 ** depth
        cc = child_centroids(tris_d)
        h = max(1e-8, size / 60.0)
        dens, _, _ = density_jacobian(reg, cc.reshape(-1, 2), h)
        dev = dens.reshape(-1, 9) / ideal - 1.0
        done, working = split(tris_d, dev, depth)
        add_leaves(tris_d, dev, done, depth)
        print(f'  depth {depth}: {working.sum()}/{len(tris_d)} unresolved')
        work = tris_d[working]

    ld = np.concatenate([x[0] for x in leaves])
    lc = np.concatenate([x[1] for x in leaves])
    lm = np.concatenate([x[2] for x in leaves])
    ls = np.concatenate([x[3] for x in leaves])
    la = A_TRI / 9.0 ** ld                       # exact leaf b_oct areas
    assert abs(la.sum() / A_TRI - 1.0) < 1e-9    # leaves partition the octant
    cls = classify(lc)

    # ── inventory: where does the RDE live? ──────────────────────────────────
    unit_km = (np.pi / 2 * reg.ellipsoid.a / 1000.0) / (2.0 * TR)  # ~scale
    lat_len = 2.0 * 2.0 * TR                     # two lateral edges per octant
    equ_len = 2.0 * TR
    print(f'\nleaf inventory: {len(ld)} leaves, depths {int(ld.min())}–'
          f'{int(ld.max())}  (1 b_oct unit ≈ {unit_km:.0f} km)')
    print(f'{"threshold":>10} {"area frac":>10}  per-class area fraction / '
          f'band width')
    for thr in (args.tol, 1e-3, 5e-3, 1e-2, 0.04):
        m = np.abs(lm) > thr
        tot = la[m].sum() / A_TRI
        parts = []
        for c in range(5):
            mc = m & (cls == c)
            if not mc.any():
                continue
            afrac = la[mc].sum() / A_TRI
            txt = f'{CLS_NAMES[c]} {afrac:.2e}'
            if c == 2:
                txt += f' (width ~{la[mc].sum()/lat_len*unit_km:.1f} km)'
            if c == 3:
                txt += f' (width ~{la[mc].sum()/equ_len*unit_km:.1f} km)'
            parts.append(txt)
        print(f'{thr:>10g} {tot:>10.3e}  {";  ".join(parts)}')

    at_max = (ld == args.max_depth) & (ls > np.maximum(args.tol, args.rel * np.abs(lm)))
    if at_max.any():
        print(f'\nstill unresolved at max depth — sub-cell structure below '
              f'the sampling scale ({at_max.sum()} cells, '
              f'area {la[at_max].sum()/A_TRI:.3e} of octant):')
        for c in range(5):
            m = at_max & (cls == c)
            if m.any():
                print(f'  {CLS_NAMES[c]:<18} {m.sum():>7} cells '
                      f'(area {la[m].sum()/A_TRI:.2e}), dev range '
                      f'[{lm[m].min():+.4f}, {lm[m].max():+.4f}], '
                      f'spread up to {ls[m].max():.4f}')
    worst = int(np.argmax(np.abs(lm)))
    ll = reg.project(Points(lc[worst:worst+1], reg.domain('b_oct'), oid=0),
                     ['b_oct', 'g_gcd']).coords[0]
    print(f'worst leaf: depth {int(ld[worst])}, dev {lm[worst]:+.4f} at '
          f'b_oct ({lc[worst,0]:+.5f},{lc[worst,1]:+.5f}) '
          f'= lat {ll[0]:+.4f}, lon {ll[1]:+.4f} ({CLS_NAMES[cls[worst]]})')

    out = Path('output')
    out.mkdir(exist_ok=True)
    keep = np.abs(lm) > 0.5 * args.tol           # compact hot inventory
    np.savez_compressed(out / f'rde_tree_{reg.ellipsoid_name}.npz',
                        depth=ld[keep], centroid=lc[keep], mean_dev=lm[keep],
                        spread=ls[keep], cls=cls[keep],
                        tol=args.tol, rel=args.rel, ideal=ideal)
    print(f'\nwrote output/rde_tree_{reg.ellipsoid_name}.npz '
          f'({keep.sum()} hot leaves kept)')


if __name__ == '__main__':
    main()
