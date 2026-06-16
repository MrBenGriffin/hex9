# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
AK+Warp displacement visualisation — geographic domain.

Projects a regular octant-face grid through two pipelines:
  1. AK-only:   b_raw → b_oct (no warp) → c_oct → c_ell → g_gcd
  2. AK+Warp:   b_raw → b_oct (warp)    → c_oct → c_ell → g_gcd

Converts both to ECEF, computes the 3D displacement in metres.
This is the raw observed cost of the warp in surface-distance units.

Any future analytical approximation (combining AK with an explicit
curvature correction) can be dropped in for pipeline 1 and the residual
is immediately the same metric — metres on the ellipsoid surface.

Panels
──────
  Top    : World map (all 8 octants), points coloured by warp displacement (m)
  Bottom : Single mode-0 octant in b_raw space, coloured by warp displacement (m)

Usage:
  python ex0123.py

Last Tested
16 Jun 2026 0.1.3a0 (passed) 18.5s
23 Apr 2026
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

from hhg9 import Registrar, Points
from hhg9.algorithms.wgs84 import latlon_to_ecef
from hhg9.h9 import H9K, H9O, in_scope

os.makedirs('output', exist_ok=True)

Y_EQ = np.sqrt(6.0) / 6.0
TR   = H9K.limits.TR
VF   = H9K.limits.VF


# ── Grid generation ───────────────────────────────────────────────────────────

def face_grid(mode: int, n: int = 80) -> np.ndarray:
    """Regular grid of interior points for a single mode-0 or mode-1 face."""
    xs = np.linspace(-TR, TR, n)
    if mode == 0:
        ys = np.linspace(VF + 0.01, Y_EQ - 0.001, n)
    else:
        ys = np.linspace(-Y_EQ + 0.001, -VF - 0.01, n)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    mask = in_scope(H9K.radical.R3 * pts[:, 0], pts[:, 1], mode)
    return pts[mask]


def build_all_octants(n: int = 80):
    """Face-grid points for all 8 octants, with integer oid array."""
    all_pts, all_oid = [], []
    for oid in range(8):
        mode = int(H9O.oid_mo[oid])
        pts  = face_grid(mode, n)
        all_pts.append(pts)
        all_oid.append(np.full(len(pts), oid, dtype=np.int32))
    return np.vstack(all_pts), np.concatenate(all_oid)


# ── Projection ────────────────────────────────────────────────────────────────

def project_to_gcd(coords: np.ndarray, oids: np.ndarray, use_warp: bool) -> np.ndarray:
    """Project b_raw coords (all octants) through AK[+Warp] to (lat, lon) degrees."""
    rg    = Registrar()
    b_raw = rg.domain('b_raw')
    b_oct = rg.domain('b_oct')
    rg.domain('c_oct')
    rg.domain('c_ell')
    rg.domain('g_gcd')
    if not use_warp:
        b_oct.no_warp()
    pts = Points(coords.copy(), b_raw, oid=oids)
    result = rg.project(pts, ['b_raw', 'b_oct', 'c_oct', 'c_ell', 'g_gcd'])
    return result.coords   # (N, 2): [lat, lon]


def to_ecef(latlon: np.ndarray) -> np.ndarray:
    """(N, 2) lat/lon degrees → (N, 3) ECEF metres."""
    x, y, z = latlon_to_ecef(latlon[:, 0], latlon[:, 1])
    return np.column_stack([x, y, z])


# ── Main ──────────────────────────────────────────────────────────────────────

def run(n: int = 100):
    print('Building octant grids...')
    coords, oids = build_all_octants(n=n)
    print(f'  {len(coords)} points across 8 octants')

    print('Projecting AK-only (no warp)...')
    ll_ak   = project_to_gcd(coords, oids, use_warp=False)

    print('Projecting AK+Warp...')
    ll_warp = project_to_gcd(coords, oids, use_warp=True)

    print('Computing 3D displacements...')
    ecef_ak   = to_ecef(ll_ak)
    ecef_warp = to_ecef(ll_warp)
    dist = np.linalg.norm(ecef_warp - ecef_ak, axis=1)

    print(f'\n  Warp 3D displacement (m)')
    print(f'    min  = {dist.min():.1f}')
    print(f'    max  = {dist.max():.1f}')
    print(f'    mean = {dist.mean():.1f}')
    print(f'    p50  = {np.percentile(dist, 50):.1f}')
    print(f'    p95  = {np.percentile(dist, 95):.1f}')
    print(f'    p99  = {np.percentile(dist, 99):.1f}')

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('AK+Warp displacement — 3D surface distance (m)', fontsize=11)
    gs  = GridSpec(2, 2, figure=fig, wspace=0.3, hspace=0.35,
                   height_ratios=[1.6, 1.0])

    norm  = mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=dist.max())
    cmap  = 'plasma'

    # ── Panel A: world map (AK+Warp), coloured by displacement ───────────────
    ax_world = fig.add_subplot(gs[0, :])
    sc = ax_world.scatter(
        ll_warp[:, 1], ll_warp[:, 0],
        c=dist, s=1.0, cmap=cmap, norm=norm, rasterized=True
    )
    plt.colorbar(sc, ax=ax_world, label='3D warp displacement (m)', shrink=0.7)
    ax_world.set_xlim(-180, 180)
    ax_world.set_ylim(-90, 90)
    ax_world.set_xlabel('Longitude (°)')
    ax_world.set_ylabel('Latitude (°)')
    ax_world.set_title('AK+Warp projected positions, coloured by 3D warp displacement')
    ax_world.set_aspect('equal')
    ax_world.axhline(0,  color='grey', lw=0.4, ls='--')
    ax_world.axvline(0,  color='grey', lw=0.4, ls='--')
    ax_world.grid(True, lw=0.2, alpha=0.4)

    # ── Panel B: AK-only world map (baseline, grey) ───────────────────────────
    ax_ak = fig.add_subplot(gs[1, 0])
    ax_ak.scatter(ll_ak[:, 1], ll_ak[:, 0], c='steelblue', s=0.5,
                  alpha=0.5, rasterized=True)
    ax_ak.set_xlim(-180, 180)
    ax_ak.set_ylim(-90, 90)
    ax_ak.set_xlabel('Longitude (°)')
    ax_ak.set_ylabel('Latitude (°)')
    ax_ak.set_title('AK-only (no warp)')
    ax_ak.set_aspect('equal')
    ax_ak.grid(True, lw=0.2, alpha=0.4)

    # ── Panel C: single octant in b_raw space, coloured by displacement ───────
    mask0 = (oids == 0)   # octant 0 (mode 0)
    ax_face = fig.add_subplot(gs[1, 1])
    sc2 = ax_face.scatter(
        coords[mask0, 0], coords[mask0, 1],
        c=dist[mask0], s=2.0, cmap=cmap, norm=norm, rasterized=True
    )
    plt.colorbar(sc2, ax=ax_face, label='displacement (m)')
    # Draw face boundaries
    sqrt3 = np.sqrt(3.0)
    x_lat = np.array([0.0, TR])
    ax_face.plot( x_lat,  sqrt3 * x_lat + VF, color='orange', lw=0.8, ls='--')
    ax_face.plot(-x_lat,  sqrt3 * x_lat + VF, color='orange', lw=0.8, ls='--')
    ax_face.axhline(Y_EQ, color='red', lw=0.6, ls='--')
    ax_face.axhline(0.0,  color='grey', lw=0.4, ls=':')
    ax_face.set_xlabel('b_raw x')
    ax_face.set_ylabel('b_raw y')
    ax_face.set_title('Octant 0 (b_raw face), displacement (m)')
    ax_face.set_aspect('equal')

    out = 'output/ex0123_warp_displacement.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    run(n=100)
