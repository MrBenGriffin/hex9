"""
Stage 2 (passthrough) — hand the Sinkhorn L4 result directly to stage 3.

Motivation
----------
The Adam-based stage 2 (stage2_gradient_l4.py) over-constrains the L4
displacement field: it tightens to its own local minimum and enforces full
interior mirror symmetry (via mirror_idx in snap_boundary_analytic).  When
stage 3 warm-starts from that, the L5 descent inherits a constraint surface
that traps it — observed as a -6.94% minimum at low coverage, far worse than
prior <0.5% worst-case runs.

This passthrough avoids that by repackaging the latest stage 1 (Sinkhorn)
checkpoint into a stage-3-compatible .npz, preserving the Sinkhorn target
positions exactly.  Stage 3 then has freedom to converge on the L5 mesh
from a softer warm start.

Usage
-----
    python stage2_passthrough.py
    python stage2_passthrough.py --ellipsoid WGS84
    python stage2_passthrough.py --sinkhorn output/q_WGS84_l4_iter99.npz

If --sinkhorn is omitted, the highest-iter q_{ellipsoid}_l4_iter*.npz
in ./output is used.

Output
------
    l4_tgrid_{ellipsoid}_sinkhorn.npz
        source_pts, target_pts, mae, layer  — readable by stage3_gradient_l5.py
"""

from pathlib import Path
import re
import numpy as np

from experimental.sinkhorn import config
from hhg9 import Registrar, Points
from experimental.sinkhorn.fast_area import fast_wgs84_area as wgs84_area
from hhg9.h9 import H9O, H9K
from hhg9.h9.classifier import location
from hhg9.h9.protocols import BaryLoc
from hhg9.h9.polygon import region_grid, H9P

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
OCTANT_ID = 0
LAYER     = 4

# If True, re-snap boundary vertices onto the analytical octant edges.
# Mirror symmetrisation (mirror_idx) is intentionally NOT applied here — that
# is the constraint we are trying to remove from the handover.
APPLY_BOUNDARY_SNAP = True
# ─────────────────────────────────────────────────────────────────────────────


def get_ideal_corners(mode: int):
    tr, vf, vc = H9K.limits.TR, H9K.limits.VF, H9K.limits.VC
    if int(mode) == 0:
        return np.array([[-tr, vc], [tr, vc], [0.0, vf]])
    else:
        return np.array([[-tr, vf], [tr, vf], [0.0, vc]])


def snap_boundary_no_mirror(xy, oc_edg, oc_vtx, mode):
    """Snap corners and edge vertices onto the analytical triangle boundary.
    Unlike stage 2 Adam version, no mirror_idx symmetrisation is applied."""
    out = xy.copy()
    corners = get_ideal_corners(mode)
    for idx in oc_vtx:
        out[idx] = corners[np.argmin(np.sum((corners - out[idx]) ** 2, axis=1))]
    if len(oc_edg) > 0:
        pts   = out[oc_edg]
        lines = [(corners[0], corners[1]),
                 (corners[1], corners[2]),
                 (corners[2], corners[0])]
        starts = np.array([l[0] for l in lines])
        vecs   = np.array([l[1] - l[0] for l in lines])
        lens2  = np.sum(vecs ** 2, axis=1)
        P, A, V = pts[:, None, :], starts[None, :, :], vecs[None, :, :]
        t = np.clip(np.sum((P - A) * V, axis=2) / lens2, 0.0, 1.0)
        projs  = A + t[:, :, None] * V
        out[oc_edg] = projs[
            np.arange(len(pts)),
            np.argmin(np.sum((P - projs) ** 2, axis=2), axis=1)
        ]
    boundary = np.array(list(oc_vtx) + list(oc_edg), dtype=int)
    on_mirror = boundary[np.abs(out[boundary, 0]) < 1e-10]
    out[on_mirror, 0] = 0.0
    return out


def latest_sinkhorn_checkpoint(ellipsoid: str, layer: int) -> Path:
    out_dir = Path('output')
    pattern = re.compile(rf'^q_{re.escape(ellipsoid)}_l{layer}_iter(\d+)\.npz$')
    candidates = []
    for p in out_dir.glob(f'q_{ellipsoid}_l{layer}_iter*.npz'):
        m = pattern.match(p.name)
        if m:
            candidates.append((int(m.group(1)), p))
    if not candidates:
        raise FileNotFoundError(
            f'No Sinkhorn checkpoints matching output/q_{ellipsoid}_l{layer}_iter*.npz'
        )
    candidates.sort()
    return candidates[-1][1]


def measure_areas(x_prime, t_grid, rg, b_raw, g_gcd):
    dpts      = Points(x_prime, b_raw, oid=0)
    gpts      = rg.project(dpts, [b_raw, g_gcd])
    t_pts_gcd = gpts.coords[t_grid.ravel()]
    areas     = wgs84_area(rg, Points(t_pts_gcd, g_gcd), 3)
    c_ideal   = float(np.mean(areas))
    ratios    = areas / c_ideal
    return areas, ratios, c_ideal


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Stage 2 (passthrough) — repackage Sinkhorn L4 for stage 3.')
    ap.add_argument('--ellipsoid', default=config.INDEX_ELLIPSOID,
                    choices=list(config.ELLIPSOIDS))
    ap.add_argument('--sinkhorn', default=None,
                    help='specific Sinkhorn checkpoint .npz to package '
                         '(default: latest iter in output/)')
    ap.add_argument('--no-snap', action='store_true',
                    help='skip boundary snap (pure passthrough)')
    args = ap.parse_args()

    rg = Registrar()
    ellipsoid = config.apply_ellipsoid(rg, args.ellipsoid)
    print(f'Ellipsoid: {ellipsoid}')
    b_raw = rg.domain('b_raw')
    g_gcd = rg.domain('g_gcd')
    mode  = H9O.oid_mo[OCTANT_ID]

    sh_path = Path(args.sinkhorn) if args.sinkhorn else latest_sinkhorn_checkpoint(ellipsoid, LAYER)
    print(f'Loading Sinkhorn checkpoint: {sh_path}')
    sh = np.load(sh_path, allow_pickle=True)
    target_pts = sh['target_pts'].copy()
    src_pts    = sh['source_pts'].copy() if 'source_pts' in sh.files else None
    sh_mae     = float(sh['mae']) if 'mae' in sh.files else float('nan')
    print(f'  Sinkhorn MAE (as saved): {sh_mae:.8f}')

    # Need source_pts + t_grid for measurement.  Pull from grid_l{LAYER}.npz when
    # not in the Sinkhorn checkpoint.
    grid_path = Path(f'grid_l{LAYER}.npz')
    grid = np.load(grid_path, allow_pickle=True)
    if src_pts is None:
        src_pts = grid['xy_vert'].copy()
    t_grid  = grid['grid']
    oc_vtx  = grid['oc_vtx']
    oc_edg  = grid['oc_edg']
    print(f'  L{LAYER}: {len(src_pts)} vertices, {len(t_grid)} triangles')

    if APPLY_BOUNDARY_SNAP and not args.no_snap:
        target_pts = snap_boundary_no_mirror(target_pts, oc_edg, oc_vtx, mode)
        # Re-pin lateral edges to identity (round-trip invertibility constraint).
        lat_mask = config.lateral_edge_mask(src_pts)
        target_pts[lat_mask] = src_pts[lat_mask]
        print(f'  Snapped boundary, pinned {int(lat_mask.sum())} lateral-edge vertices')

    # Measure current state (post-snap if applied).
    areas, ratios, c_ideal = measure_areas(target_pts, t_grid, rg, b_raw, g_gcd)
    mae = float(np.mean(np.abs(ratios - 1.0)))
    print(f'\n[Passthrough]  MAE: {mae:.8f} | '
          f'Min/Max: {ratios.min():.6f} / {ratios.max():.6f}')

    # Per-mode report for parity with stage 2 Adam output.
    tri_modes = np.array(
        [m for (_path, m, _origin, _scale) in region_grid(LAYER, mode, H9P)],
        dtype=np.int8,
    )
    pct_dev = (ratios - 1.0) * 100.0
    for m, sym in [(0, '∇ mode-0'), (1, '△ mode-1')]:
        mask = tri_modes == m
        print(f'  {sym}: {mask.sum():5d} tri  '
              f'mean {pct_dev[mask].mean():+.4f}%  std {pct_dev[mask].std():.4f}%')

    out_path = Path(f'l4_tgrid_{ellipsoid}_sinkhorn.npz')
    np.savez(
        out_path,
        layer=np.array(LAYER),
        source_pts=src_pts,
        target_pts=target_pts,
        mae=np.array(mae),
        source_checkpoint=str(sh_path),
    )
    print(f'\nSaved stage-3-compatible warm start: {out_path}')
    print('Run stage 3:')
    print(f'  python stage3_gradient_l5.py --ellipsoid {ellipsoid} '
          f'--warm-start {out_path}')
