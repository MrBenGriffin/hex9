# Part of the Hex9 (H9) Project
# Copyright ©2026, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""
Pointwise residual-density-error (RDE) field of a deployed warp.

Per-hex area deviation at ANY layer is the cell average of one fixed
continuum field: the log ratio of achieved surface density (m² of
ellipsoid per unit b_oct area, through the production b_oct -> g_gcd
chain, warp included) to the ideal S_total / A_boct_total. This script
evaluates that field — plus the full Jacobian decomposition (Tissot:
anisotropy sigma1/sigma2) and the warp displacement strain — on a
uniform grid over octant 0, by central finite differences mapped onto
the authalic sphere (area-exact for any ellipsoid).

    python rde_field.py --ellipsoid Sphere --res 600 --spot-check 2000

Outputs: output/rde_field_<ELL>.npz, output/rde_field_<ELL>.png, stats
to stdout. --spot-check N anchors the cheap Jacobian path to
geographiclib polygon areas of N random depth-5 hexes.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hhg9 import Registrar, Points
from hhg9.h9 import H9K, H9O
from experimental.sinkhorn import config
from experimental.sinkhorn.fast_area import _authalic_lat

TR, VF, VC = H9K.limits.TR, H9K.limits.VF, H9K.limits.VC
R3, W_ = H9K.radical.R3, H9K.Ẇ
CORNERS = np.array([[-TR, VC], [TR, VC], [0.0, VF]])   # octant 0 (mode 0)
_e1, _e2 = CORNERS[1] - CORNERS[0], CORNERS[2] - CORNERS[0]
A_TRI = 0.5 * abs(_e1[0] * _e2[1] - _e1[1] * _e2[0])
A_BOCT_TOTAL = 8.0 * A_TRI


def authalic_xyz(latlon_deg: np.ndarray, a: float, f: float) -> np.ndarray:
    """Map geodetic lat/lon (deg) to 3D points on the authalic sphere.

    The authalic sphere has the same total area as the ellipsoid and the
    map is area-preserving, so tangent-vector cross products give exact
    ellipsoidal surface densities.
    """
    e2 = f * (2.0 - f)
    lat = np.deg2rad(latlon_deg[..., 0])
    lon = np.deg2rad(latlon_deg[..., 1])
    if e2 > 0.0:
        xi, q_p = _authalic_lat(lat, e2)
        r_q = a * np.sqrt(q_p / 2.0)
    else:
        xi, r_q = lat, a
    cx = np.cos(xi)
    return r_q * np.stack([cx * np.cos(lon), cx * np.sin(lon), np.sin(xi)], axis=-1)


def density_jacobian(reg, xy: np.ndarray, h: float, oid: int = 0,
                     chunk: int = 1_000_000):
    """Surface density (m²/b_oct-unit²) and tangent-Jacobian singular values.

    Central differences through the production chain b_oct -> g_gcd ->
    authalic sphere. Returns (density, sig1, sig2) with sig1 >= sig2 in
    metres per b_oct unit (density == sig1*sig2).
    """
    bo, gg = reg.domain('b_oct'), reg.domain('g_gcd')
    a, f = reg.ellipsoid.a, reg.ellipsoid.f
    n = len(xy)
    offsets = np.array([[h, 0.0], [-h, 0.0], [0.0, h], [0.0, -h]])
    dens = np.empty(n)
    s1 = np.empty(n)
    s2 = np.empty(n)
    for lo in range(0, n, chunk):
        sl = xy[lo:lo + chunk]
        pts = (sl[:, None, :] + offsets[None, :, :]).reshape(-1, 2)
        ll = reg.project(Points(pts, bo, oid=np.full(len(pts), oid)),
                         [bo, gg]).coords
        p = authalic_xyz(ll, a, f).reshape(-1, 4, 3)
        u = (p[:, 0] - p[:, 1]) / (2.0 * h)
        v = (p[:, 2] - p[:, 3]) / (2.0 * h)
        dens[lo:lo + chunk] = np.linalg.norm(np.cross(u, v), axis=1)
        # Gram matrix of [u v] -> singular values of the tangent Jacobian
        guu = np.einsum('ij,ij->i', u, u)
        gvv = np.einsum('ij,ij->i', v, v)
        guv = np.einsum('ij,ij->i', u, v)
        tr_half = 0.5 * (guu + gvv)
        disc = np.sqrt(np.maximum(0.25 * (guu - gvv) ** 2 + guv ** 2, 0.0))
        s1[lo:lo + chunk] = np.sqrt(np.maximum(tr_half + disc, 0.0))
        s2[lo:lo + chunk] = np.sqrt(np.maximum(tr_half - disc, 0.0))
    return dens, s1, s2


def warp_strain(reg, xy: np.ndarray, h: float):
    """Displacement-gradient invariants of the warp itself (octant 0).

    Returns (div, shear, emax): divergence tr(F), max shear (deviatoric
    principal of sym F), and largest |principal strain| of sym F, where
    F = grad d and d(x) = warp.do(x) - x. Dimensionless (b_oct/b_oct).
    """
    warp = reg.domain('b_oct').warp
    mo = int(H9O.oid_mo[0])
    offsets = np.array([[h, 0.0], [-h, 0.0], [0.0, h], [0.0, -h]])
    pts = (xy[:, None, :] + offsets[None, :, :]).reshape(-1, 2)
    d = (warp.do(pts, mo) - pts).reshape(-1, 4, 2)
    fx = (d[:, 0] - d[:, 1]) / (2.0 * h)        # [dxx, dyx]
    fy = (d[:, 2] - d[:, 3]) / (2.0 * h)        # [dxy, dyy]
    exx, eyy = fx[:, 0], fy[:, 1]
    exy = 0.5 * (fx[:, 1] + fy[:, 0])
    div = exx + eyy
    shear = np.sqrt(0.25 * (exx - eyy) ** 2 + exy ** 2)
    emax = np.maximum(np.abs(0.5 * div + shear), np.abs(0.5 * div - shear))
    return div, shear, emax


def octant_grid(res: int, margin: float) -> np.ndarray:
    """Uniform grid over the octant-0 triangle, margin inside all edges."""
    xs = np.linspace(-TR, TR, res)
    ys = np.linspace(VF, VC, int(res * (VC - VF) / (2 * TR)))
    gx, gy = np.meshgrid(xs, ys)
    xy = np.column_stack([gx.ravel(), gy.ravel()])
    inside = ((xy[:, 1] <= VC - margin)
              & (xy[:, 1] - R3 * np.abs(xy[:, 0]) + W_ >= 2.0 * margin))
    return xy[inside]


def spot_check(reg, n_hexes: int, h: float, rng=None):
    """Anchor the Jacobian density path to geographiclib polygon areas.

    Samples random depth-5 hexes; predicted area = (quadrature mean
    density over the hex) * (shoelace b_oct hex area) vs geodesic truth.
    """
    from hhg9.algorithms.distance import wgs84_area
    from hhg9.h9.polygon import hex_poly_layer
    from area_stats import get_data
    rng = rng or np.random.default_rng(42)

    pts, _ = hex_poly_layer(get_data(reg, 5), 5)
    hx = pts.coords.reshape(-1, 6, 2)
    ho = pts.oid.reshape(-1, 6)
    same_oid = (ho == ho[:, :1]).all(axis=1)          # skip octant-straddlers
    interior = config.lateral_distance(hx.reshape(-1, 2)).reshape(-1, 6).min(axis=1) > 4 * h
    interior &= (VC - hx[:, :, 1].max(axis=1)) > 4 * h
    cand = np.flatnonzero(same_oid & interior)
    pick = rng.choice(cand, size=min(n_hexes, len(cand)), replace=False)

    hx, ho = hx[pick], ho[pick, 0]
    # geodesic truth
    g_pts = reg.project(Points(hx.reshape(-1, 2), reg.domain('b_oct'),
                               oid=np.repeat(ho, 6)), ['b_oct', 'g_gcd'])
    true_area = np.abs(wgs84_area(reg, g_pts))
    # quadrature: fan-triangulate, 9 child centroids per fan triangle (54/hex)
    ctr = hx.mean(axis=1)
    fan = np.stack([np.repeat(ctr[:, None, :], 6, axis=1), hx,
                    np.roll(hx, -1, axis=1)], axis=2)        # (N,6,3,2)
    w = _child_centroid_weights()                            # (9,3)
    q = np.einsum('nftc,kt->nfkc', fan, w)                   # (N,6,9,2)
    q = q.reshape(len(hx), 54, 2)
    dens = np.empty((len(hx), 54))
    for oc in np.unique(ho):                # one projection batch per octant
        idx = np.flatnonzero(ho == oc)
        d, _, _ = density_jacobian(reg, q[idx].reshape(-1, 2), h, oid=int(oc))
        dens[idx] = d.reshape(-1, 54)
    shoelace = 0.5 * np.abs(np.einsum('nij,nij->n', hx,
                            np.roll(hx, -1, axis=1)[:, :, ::-1] * [1, -1]))
    pred_area = dens.mean(axis=1) * shoelace
    rel = pred_area / true_area - 1.0
    print(f'\n[spot-check] {len(hx)} depth-5 hexes, Jacobian-quadrature vs geodesic:')
    print(f'  rel diff median {np.median(np.abs(rel)):.3e}  '
          f'p99 {np.percentile(np.abs(rel), 99):.3e}  max {np.abs(rel).max():.3e}')
    return rel


def _child_centroid_weights() -> np.ndarray:
    """Barycentric weights of the 9 child-triangle centroids (3-subdivision)."""
    ws = []
    for i in range(3):
        for j in range(3 - i):
            ws.append([3 - i - j - 2/3, i + 1/3, j + 1/3])       # upward
            if i + j < 2:
                ws.append([3 - i - j - 4/3, i + 2/3, j + 2/3])   # downward
    w = np.array(ws) / 3.0
    assert w.shape == (9, 3) and np.allclose(w.sum(axis=1), 1.0)
    return w


def make_figure(xy, ell, anis, emax, name, h, out: Path):
    """Three-panel field figure; ℓ on a symlog scale (the structure is
    confined to thin boundary layers far below the linear range)."""
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.colors import SymLogNorm
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    m = np.abs(ell).max()
    panels = [(ell, 'RdBu_r', 'RDE ℓ = log(density/ideal)  [symlog]',
               SymLogNorm(linthresh=1e-5, vmin=-m, vmax=m)),
              (anis, 'viridis', 'anisotropy σ1/σ2', None),
              (emax, 'magma', 'warp strain |principal| max  [log]',
               matplotlib.colors.LogNorm(vmin=max(emax.min(), 1e-6),
                                         vmax=emax.max()))]
    for ax, (val, cm, title, norm) in zip(axes, panels):
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=val, s=1.2, cmap=cm,
                        linewidths=0, norm=norm)
        fig.colorbar(sc, ax=ax, shrink=0.8)
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.plot(*np.vstack([CORNERS, CORNERS[:1]]).T, 'k-', lw=0.5)
    fig.suptitle(f'{name} warp RDE field ({len(xy)} samples, h={h:g})')
    fig.savefig(out / f'rde_field_{name}.png', dpi=200, bbox_inches='tight')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ellipsoid', default='WGS84', choices=list(config.ELLIPSOIDS))
    ap.add_argument('--res', type=int, default=600,
                    help='samples across the octant width')
    ap.add_argument('--h', type=float, default=1e-6, help='FD step, b_oct units')
    ap.add_argument('--spot-check', type=int, default=0,
                    help='anchor against N geodesic depth-5 hex areas')
    args = ap.parse_args()

    reg = Registrar()
    config.apply_ellipsoid(reg, args.ellipsoid)
    ideal = reg.ellipsoid_area / A_BOCT_TOTAL
    print(f'ellipsoid {reg.ellipsoid_name}: S={reg.ellipsoid_area:.6e} m², '
          f'ideal density {ideal:.6e} m²/unit²')

    xy = octant_grid(args.res, margin=5 * args.h)
    print(f'grid: {len(xy)} samples, h={args.h:g}')

    dens, s1, s2 = density_jacobian(reg, xy, args.h)
    ell = np.log(dens / ideal)
    anis = s1 / s2
    div, shear, emax = warp_strain(reg, xy, args.h)

    half = xy[:, 0] >= 0.0          # fundamental domain (x-mirror) for stats
    l_h, a_h = ell[half], anis[half]
    pa = lambda v, q: float(np.percentile(np.abs(v), q))
    print(f'\nRDE ℓ (pointwise log density error), x>=0 half-octant:')
    print(f'  mean {l_h.mean():+.3e}  sigma {l_h.std():.3e}  '
          f'min {l_h.min():+.5f}  max {l_h.max():+.5f}')
    print(f'  |ℓ| p50 {pa(l_h,50):.3e}  p99 {pa(l_h,99):.3e}  '
          f'p99.99 {pa(l_h,99.99):.3e}')
    print(f'closure: grid-mean density / ideal - 1 = {dens.mean()/ideal - 1:+.3e}')
    print(f'anisotropy sigma1/sigma2: p50 {np.percentile(a_h,50):.4f}  '
          f'p99 {np.percentile(a_h,99):.4f}  max {a_h.max():.4f}')
    i = int(np.argmax(np.abs(ell)))
    ll = reg.project(Points(xy[i:i+1], reg.domain('b_oct'), oid=0),
                     ['b_oct', 'g_gcd']).coords[0]
    print(f'worst |ℓ| at b_oct ({xy[i,0]:+.4f},{xy[i,1]:+.4f}) '
          f'= lat {ll[0]:+.3f}, lon {ll[1]:+.3f}')
    print(f'warp strain: max |principal| {emax.max():.4f}  '
          f'max shear {shear.max():.4f}  max |div| {np.abs(div).max():.4f}')

    out = Path('output')
    out.mkdir(exist_ok=True)
    np.savez_compressed(out / f'rde_field_{reg.ellipsoid_name}.npz',
                        xy=xy, ell=ell, sig1=s1, sig2=s2,
                        div=div, shear=shear, emax=emax,
                        ideal=ideal, h=args.h)

    make_figure(xy, ell, anis, emax, reg.ellipsoid_name, args.h, out)
    print(f'\nwrote output/rde_field_{reg.ellipsoid_name}.npz / .png')

    if args.spot_check:
        spot_check(reg, args.spot_check, args.h)


if __name__ == '__main__':
    main()
