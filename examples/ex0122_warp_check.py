# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Warp sanity check — CloughTocher continuity and round-trip accuracy.

Probes the AuthalicWarp for three failure modes before freezing a warp file:

  1. Round-trip accuracy    forward(src) → undo() ≈ src; max error reported
  2. Seam discontinuity     first-difference derivatives at y=0 (mode seam)
                            and y=±Y_EQ (equatorial edge with ghost rows)
  3. Folding                finite-difference Jacobian determinant — any det<0
                            means the warp is non-injective in that region

Usage:
  python ex0122_warp_check.py
  python ex0122_warp_check.py --warp ../hhg9/data/l5_boct_warp_data.npz
  python ex0122_warp_check.py --warp my_contender.npz

Outputs:
  Console report + output/ex0122_warp_check_<stem>.png  (displacement + det map)

Last Tested
21 Apr 2026
"""

import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

from hhg9.domains.octahedral_barycentric import AuthalicWarp
from hhg9.h9 import H9K

os.makedirs('output', exist_ok=True)

# ── Constants ────────────────────────────────────────────────────────────────

Y_EQ  = np.sqrt(6.0) / 6.0   # equatorial edge y  ≈ 0.4082
Y_MO1 = -Y_EQ                 # mode-1 equatorial edge
TR    = H9K.limits.TR          # x half-width at mode-0 equatorial edge

# ── Helpers ──────────────────────────────────────────────────────────────────

VF    = H9K.limits.VF    # polar vertex y  ≈ -0.8165  (face spans VF → Y_EQ)


def make_grid(n: int = 200, margin: float = 0.005):
    """Dense regular grid covering the mode-0 octant face interior.

    The octant face is a tall triangle: apex at (0, VF ≈ -0.8165),
    base from (-TR, Y_EQ) to (+TR, Y_EQ).  A *margin* (in b_oct units)
    is pulled back from the lateral edges to avoid finite-difference Jacobian
    artefacts: the CT convex hull extends into the mode-1 half of the data, so
    a FD probe that crosses the lateral edge can return a geometrically invalid
    displacement and produce a spurious Jacobian sign flip.
    """
    from hhg9.h9 import in_scope
    xs = np.linspace(-TR, TR, n)
    ys = np.linspace(VF + 0.01, Y_EQ - 0.001, n)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    # Strict interior: valid AND inside lateral edge by *margin*
    x_max = TR * (pts[:, 1] - VF) / (Y_EQ - VF)
    mask = in_scope(H9K.radical.R3 * pts[:, 0], pts[:, 1], 0) & (
        np.abs(pts[:, 0]) < x_max - margin
    )
    return pts[mask], xx.shape


def jac_det(warp: AuthalicWarp, pts: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Finite-difference Jacobian determinant at each point."""
    dx_pts  = pts + np.array([[eps, 0]])
    dy_pts  = pts + np.array([[0, eps]])
    f0  = warp.do(pts.copy())
    fdx = warp.do(dx_pts.copy())
    fdy = warp.do(dy_pts.copy())
    J00 = (fdx[:, 0] - f0[:, 0]) / eps  # df_x/dx
    J10 = (fdx[:, 1] - f0[:, 1]) / eps  # df_y/dx
    J01 = (fdy[:, 0] - f0[:, 0]) / eps  # df_x/dy
    J11 = (fdy[:, 1] - f0[:, 1]) / eps  # df_y/dy
    return J00 * J11 - J01 * J10


def mirror_symmetry_check(warp: AuthalicWarp, pts: np.ndarray) -> dict:
    """Check x=0 mirror symmetry of the warp on the interior grid.

    The mode-0 octant is symmetric about x=0, so the warp must satisfy:
      dx(-x, y) = -dx(x, y)   (antisymmetric)
      dy(-x, y) =  dy(x, y)   (symmetric)

    Uses only pts with x > 0, pairs each with its mirror, and reports
    the max and mean deviation from perfect symmetry.
    """
    mask = pts[:, 0] > 1e-8
    p     = pts[mask]
    p_mir = p.copy()
    p_mir[:, 0] *= -1.0

    f     = warp.do(p.copy())
    f_mir = warp.do(p_mir.copy())

    d     = f     - p        # displacement at  (x,  y)
    d_mir = f_mir - p_mir    # displacement at (-x,  y)

    # antisymmetry of dx: d_mir[:,0] should equal -d[:,0]
    err_dx = np.abs(d_mir[:, 0] + d[:, 0])
    # symmetry of dy: d_mir[:,1] should equal d[:,1]
    err_dy = np.abs(d_mir[:, 1] - d[:, 1])

    return {
        'max_err_dx': float(err_dx.max()),
        'mean_err_dx': float(err_dx.mean()),
        'max_err_dy': float(err_dy.max()),
        'mean_err_dy': float(err_dy.mean()),
    }


def lateral_seam_check(warp: AuthalicWarp, side: str = 'right',
                       n: int = 80, eps: float = 1e-6) -> dict:
    """One-sided warp check along a lateral octant edge (meridional seam).

    Samples points just inside the mode-0 triangle along the left or right
    lateral edge, reporting displacement magnitude and its inward-normal
    derivative. A high NaN fraction indicates CT fallback to nearest-neighbour
    near the boundary.

    Right edge: y =  sqrt(3)*x + VF  (x in [0, TR])
    Left edge:  y = -sqrt(3)*x + VF  (x in [-TR, 0])
    """
    sqrt3 = np.sqrt(3.0)
    t = np.linspace(0.05, 0.95, n)   # avoid degenerate apex/corner vertices

    if side == 'right':
        x_edge = t * TR
        y_edge = sqrt3 * x_edge + VF
        nx, ny = -sqrt3 / 2.0, 0.5   # inward normal (left-and-up)
    else:
        x_edge = -t * TR
        y_edge = -sqrt3 * x_edge + VF
        nx, ny = sqrt3 / 2.0, 0.5    # inward normal (right-and-up)

    p1 = np.column_stack([x_edge +     eps * nx, y_edge +     eps * ny])
    p2 = np.column_stack([x_edge + 2 * eps * nx, y_edge + 2 * eps * ny])

    f1 = warp.do(p1.copy())
    f2 = warp.do(p2.copy())

    displ    = np.linalg.norm(f1 - p1, axis=1)
    d_perp   = np.linalg.norm((f2 - f1) / eps, axis=1)
    nan_frac = np.isnan(f1).any(axis=1).mean()

    return {
        'max_displ':   float(np.nanmax(displ)),
        'max_d_perp':  float(np.nanmax(d_perp)),
        'mean_d_perp': float(np.nanmean(d_perp)),
        'nan_frac':    float(nan_frac),
    }


def seam_derivative_jump(warp: AuthalicWarp, seam_y: float,
                          n: int = 100, eps: float = 1e-6) -> dict:
    """Finite-difference derivative jump across a horizontal seam.

    Evaluates displacement and its y-derivative just above and below seam_y
    and returns max absolute differences.
    """
    # Sample x positions across the valid range at this y
    x_range = TR * max(0.0, 1.0 - abs(seam_y) / Y_EQ) * 0.9
    if x_range < 1e-8:
        return {'max_val_jump': np.nan, 'max_deriv_jump': np.nan}
    xs = np.linspace(-x_range, x_range, n)

    above = np.column_stack([xs, np.full(n, seam_y + eps)])
    below = np.column_stack([xs, np.full(n, seam_y - eps)])
    above2 = np.column_stack([xs, np.full(n, seam_y + 2 * eps)])
    below2 = np.column_stack([xs, np.full(n, seam_y - 2 * eps)])

    fa = warp.do(above.copy())
    fb = warp.do(below.copy())
    da = warp.do(above2.copy()) - fa   # one-sided dy step above
    db = fb - warp.do(below2.copy())   # one-sided dy step below

    val_jump   = np.abs(fa - fb).max()
    # derivative: (above2 - above) vs (below - below2), both divided by eps
    deriv_jump = np.abs((da - db) / eps).max()

    return {'max_val_jump': val_jump, 'max_deriv_jump': deriv_jump}


# ── Main check ───────────────────────────────────────────────────────────────

def run(warp_path: str) -> None:
    path = Path(warp_path)
    if not path.exists():
        raise FileNotFoundError(f'Warp file not found: {path}')

    print(f'\n=== Warp check: {path.name} ===')
    warp = AuthalicWarp(str(path))

    # ── 1. Source grid stats ──────────────────────────────────────────────────
    print(f'  VF={VF:.5f} (polar apex)  Y_EQ={Y_EQ:.5f} (equatorial)  TR={TR:.5f}')
    repo = np.load(str(path))
    src  = repo['source_pts']
    diff = repo['target_pts'] - src
    print(f'\n[Source grid]')
    print(f'  n_pts = {len(src)}')
    print(f'  y range: [{src[:,1].min():.5f}, {src[:,1].max():.5f}]   Y_EQ={Y_EQ:.5f}')
    print(f'  max |displacement|: dx={np.abs(diff[:,0]).max():.6f}  dy={np.abs(diff[:,1]).max():.6f}')

    # ── 2. Round-trip accuracy ────────────────────────────────────────────────
    print('\n[Round-trip: do() → undo()]')
    pts, _ = make_grid(n=150)
    warped   = warp.do(pts.copy())
    recovered = warp.undo(warped.copy())
    err = np.linalg.norm(pts - recovered, axis=1)
    print(f'  n_pts={len(pts)}  max_err={err.max():.3e}  mean_err={err.mean():.3e}  p99={np.percentile(err,99):.3e}')
    if err.max() > 1e-10:
        print(f'  WARNING: max round-trip error exceeds 1e-10')

    # ── 3. Seam derivative continuity ────────────────────────────────────────
    print('\n[Seam continuity (value jump / derivative jump)]')
    seams = {
        'y=0 (mode seam)': 0.0,
        'y=+Y_EQ (mode-0 equator)': Y_EQ * 0.9999,
        'y=-Y_EQ (mode-1 equator)': -Y_EQ * 0.9999,
    }
    for label, y in seams.items():
        r = seam_derivative_jump(warp, y)
        vj = r['max_val_jump']
        dj = r['max_deriv_jump']
        flag = '  OK' if (vj < 1e-5 and dj < 1e-2) else '  ** CHECK **'
        print(f'  {label}: val_jump={vj:.2e}  deriv_jump={dj:.2e}{flag}')

    # ── 4. Mirror symmetry (x=0) ─────────────────────────────────────────────
    print('\n[Mirror symmetry: x=0  (dx antisymmetric, dy symmetric)]')
    ms = mirror_symmetry_check(warp, pts)
    flag_dx = '  ** ASYMMETRIC **' if ms['max_err_dx'] > 1e-6 else '  OK'
    flag_dy = '  ** ASYMMETRIC **' if ms['max_err_dy'] > 1e-6 else '  OK'
    print(f'  dx: max_err={ms["max_err_dx"]:.3e}  mean_err={ms["mean_err_dx"]:.3e}{flag_dx}')
    print(f'  dy: max_err={ms["max_err_dy"]:.3e}  mean_err={ms["mean_err_dy"]:.3e}{flag_dy}')

    # ── 5. Lateral edge checks (one-sided, mode-0 interior) ──────────────────
    print('\n[Lateral edge checks (one-sided ∂/∂n, mode-0)]')
    for side in ('right', 'left'):
        r = lateral_seam_check(warp, side=side)
        flag = '  ** CHECK **' if (r['nan_frac'] > 0.05 or r['max_d_perp'] > 1.0) else '  OK'
        print(f'  {side:5s} edge: max|displ|={r["max_displ"]:.4f}  '
              f'max|∂/∂n|={r["max_d_perp"]:.4f}  '
              f'mean|∂/∂n|={r["mean_d_perp"]:.4f}  '
              f'NaN={r["nan_frac"]:.1%}{flag}')

    # ── 5. Jacobian determinant (fold check) ─────────────────────────────────
    print('\n[Jacobian determinant (folding check)]')
    det = jac_det(warp, pts)
    n_neg  = np.sum(det < 0)
    n_low  = np.sum((det > 0) & (det < 0.5))
    print(f'  min det={det.min():.4f}  max det={det.max():.4f}  mean det={det.mean():.4f}')
    print(f'  det < 0 (folded): {n_neg}  det < 0.5 (near-fold): {n_low}')
    if n_neg > 0:
        print(f'  ** FOLD DETECTED ** — warp is non-injective in {n_neg} grid cells')

    # ── 6. Plot ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f'Warp check: {path.name}', fontsize=11)
    gs  = GridSpec(2, 2, figure=fig, wspace=0.35, hspace=0.35)

    sqrt3 = np.sqrt(3.0)
    x_lat = np.array([0.0, TR])

    def add_boundary(ax, hline_color='red'):
        for y in [0.0, Y_EQ]:
            ax.axhline(y, color=hline_color, lw=0.5, ls='--')
        ax.plot( x_lat,  sqrt3 * x_lat + VF, color='orange', lw=0.5, ls='--')
        ax.plot(-x_lat,  sqrt3 * x_lat + VF, color='orange', lw=0.5, ls='--')

    # a) Displacement magnitude
    ax0 = fig.add_subplot(gs[0, 0])
    displ = np.linalg.norm(warp.do(pts.copy()) - pts, axis=1)
    sc = ax0.scatter(pts[:, 0], pts[:, 1], c=displ * 1000, s=1,
                     cmap='viridis', vmin=0)
    plt.colorbar(sc, ax=ax0, label='|displacement| ×10³')
    ax0.set_title('Displacement magnitude')
    ax0.set_aspect('equal')
    add_boundary(ax0)

    # b) Displacement quiver (coarse grid for readability)
    ax1 = fig.add_subplot(gs[0, 1])
    q_pts, _ = make_grid(n=40, margin=0.01)
    q_warped  = warp.do(q_pts.copy())
    q_disp    = q_warped - q_pts
    q_mag     = np.linalg.norm(q_disp, axis=1)
    qv = ax1.quiver(q_pts[:, 0], q_pts[:, 1],
                    q_disp[:, 0], q_disp[:, 1],
                    q_mag, cmap='plasma', scale=0.15, scale_units='xy',
                    width=0.003, headwidth=3)
    plt.colorbar(qv, ax=ax1, label='|displacement|')
    ax1.set_title('Displacement field (quiver)')
    ax1.set_aspect('equal')
    add_boundary(ax1, hline_color='blue')

    # c) Jacobian determinant
    ax2 = fig.add_subplot(gs[1, 0])
    norm = mcolors.TwoSlopeNorm(vmin=det.min(), vcenter=1.0, vmax=det.max())
    sc2 = ax2.scatter(pts[:, 0], pts[:, 1], c=det, s=1,
                      cmap='RdYlGn', norm=norm)
    plt.colorbar(sc2, ax=ax2, label='det(J)')
    ax2.set_title(f'Jacobian det  (folds={n_neg})')
    ax2.set_aspect('equal')
    add_boundary(ax2, hline_color='blue')

    # d) Round-trip error
    ax3 = fig.add_subplot(gs[1, 1])
    sc3 = ax3.scatter(pts[:, 0], pts[:, 1], c=np.log10(err + 1e-20), s=1,
                      cmap='hot_r', vmin=-16, vmax=-8)
    plt.colorbar(sc3, ax=ax3, label='log₁₀ round-trip error')
    ax3.set_title('Round-trip error (log₁₀)')
    ax3.set_aspect('equal')
    add_boundary(ax3, hline_color='blue')

    out = f'output/ex0122_warp_check_{path.stem}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--warp', default='../hhg9/data/WGS84_l5_warp_data.npz',
                        help='Path to .npz warp file (default: WGS84_l5_warp_data.npz)')
    args = parser.parse_args()
    run(args.warp)
