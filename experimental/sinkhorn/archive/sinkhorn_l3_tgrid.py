"""
L3 triangle-grid authalic warp via Adam gradient descent.

Optimises per-triangle area equality at layer 3 (9^4 = 6561 triangles per
octant).  The resulting displacement field is coarser than the current L4
canonical warp, but serves as a cheaper, tighter-converged control point that
can be CT-interpolated to any finer layer.

Motivation
----------
The canonical L4 Sinkhorn warp has a systematic ~0.39% mode asymmetry
(mode-0 triangles slightly under-area, mode-1 slightly over-area) that is
NOT caused by alpha, but is a consequence of the right-isoceles octant geometry.
Because the Sinkhorn / gradient optimise total area variance rather than
explicitly mode-balanced area variance, the mode offset persists as the dominant
systematic in the ~0.1% residual.

An L3 solve, run fresh or warm-started from the canonical L4 warp, uses the
same per-triangle loss (Σ log r²) and the same Adam + shape-dampener approach
but on a 9× smaller vertex set.  Mode-split MAE is reported each iteration so
convergence on this specific metric can be tracked.

Usage
-----
Fresh start (identity initialisation):
    WARM_START_FROM = None

Warm-start from canonical L4 warp:
    WARM_START_FROM = Path('../../hhg9/data/l4_boct_warp_data.npz')

Resume an existing L3 run:
    L3_CHECKPOINT = Path('l3_tgrid_00500.npz')

Outputs
-------
    l3_tgrid_{iter:05d}.npz   — checkpoint every SAVE_EVERY iters and on best
    l3_tgrid_best.npz         — always current-best displacement field

The checkpoint format is compatible with gradient_l5log_froml4.py so the result
can be warm-started into an L4 or L5 Adam run if desired.
"""

from pathlib import Path
import numpy as np

from hhg9 import Registrar, Points
from hhg9.algorithms.distance import wgs84_area
from hhg9.h9 import H9O, H9K
from hhg9.h9.classifier import location
from hhg9.h9.protocols import BaryLoc
from hhg9.h9.polygon import tri_mesh, region_grid, H9P

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
OCTANT_ID = 0
LAYER = 4

# Warm-start: None = fresh (identity), or path to an existing .npz checkpoint
# whose source_pts / target_pts define a displacement field.  The field is
# CT-interpolated to the L3 vertex positions.
WARM_START_FROM: Path | None = None
# Example: WARM_START_FROM = Path('../../hhg9/data/l4_boct_warp_data.npz')

# Resume an existing L3 run (takes priority over WARM_START_FROM)
L3_CHECKPOINT: Path | None = None

ITERATIONS  = 5000
STEP_SIZE   = 0.0002    # Adam lr — L3 triangles are 9× larger than L4, so
                        # lr can be ~3× larger than the L5 value of 0.00001
STEP_MIN    = 1e-10
STEP_COOL   = 0.80      # lr multiplier when MAE plateaus
STEP_PATIENCE = 150     # iters of non-improvement before cooling
WARMUP_ITERS  = 300     # suppress cooling for first N iters

SHAPE_DAMPEN = 0.6      # gradient dampener for non-equilateral triangles
                        # (same semantics as gradient_l5log_froml4.py)
SAVE_EVERY   = 100      # save checkpoint every N iterations

# Adam hyper-parameters
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS   = 1e-8
# ─────────────────────────────────────────────────────────────────────────────


# ── helpers ───────────────────────────────────────────────────────────────────

def get_ideal_corners(mode: int):
    tr, vf, vc = H9K.limits.TR, H9K.limits.VF, H9K.limits.VC
    if int(mode) == 0:
        return np.array([[-tr, vc], [tr, vc], [0.0, vf]])
    else:
        return np.array([[-tr, vf], [tr, vf], [0.0, vc]])


def snap_boundary_analytic(xy, oc_edg, oc_vtx, mode):
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
    return out


def build_tri_modes(layer: int, octant_mode: int) -> np.ndarray:
    """Return (N_tri,) int8 mode array matching tri_mesh triangle order."""
    queue = region_grid(layer, octant_mode, H9P)
    return np.array([m for (_path, m, _origin, _scale) in queue], dtype=np.int8)


def measure_areas(x_prime, t_grid, rg, b_raw, g_gcd):
    """Return (areas, ratios, c_ideal) for all triangles."""
    dpts      = Points(x_prime, b_raw, oid=0)
    gpts      = rg.project(dpts, [b_raw, g_gcd])
    t_pts_gcd = gpts.coords[t_grid.ravel()]
    areas     = wgs84_area(rg, Points(t_pts_gcd, g_gcd), 3)
    c_ideal   = float(np.mean(areas))
    ratios    = areas / c_ideal
    return areas, ratios, c_ideal


def report_mode_split(ratios, tri_modes):
    """Print per-mode MAE and asymmetry; return asymmetry value."""
    pct_dev = (ratios - 1.0) * 100.0
    lines = []
    means = {}
    for m, sym in [(0, '∇ mode-0'), (1, '△ mode-1')]:
        mask = tri_modes == m
        mean_pct = pct_dev[mask].mean()
        std_pct  = pct_dev[mask].std()
        means[m] = mean_pct
        lines.append(f'  {sym}: {mask.sum():5d} tri  mean {mean_pct:+.4f}%  std {std_pct:.4f}%')
    asym = means[0] - means[1]
    lines.append(f'  Mode asymmetry (Δ mean): {asym:+.4f}%')
    if abs(asym) > 0.2:
        lines.append('  *** asymmetry > 0.2% ***')
    print('\n'.join(lines))
    return asym


COT_IDEAL = 1.0 / np.sqrt(3)   # cot(60°)


def compute_gradient(x_prime, t_grid, ratios, c_ideal, move_mask, oc_vtx,
                     shape_dampen=SHAPE_DAMPEN):
    """
    Vectorised gradient of Σ log(r_t)² w.r.t. vertex positions.
    Identical to gradient_l5log_froml4.compute_gradient.
    """
    v0 = t_grid[:, 0]
    v1 = t_grid[:, 1]
    v2 = t_grid[:, 2]
    p0 = x_prime[v0]
    p1 = x_prime[v1]
    p2 = x_prime[v2]

    e1 = p1 - p0
    e2 = p2 - p0
    flat_area = 0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    sign_f    = np.where(flat_area >= 0.0, 1.0, -1.0)
    abs_flat  = np.maximum(np.abs(flat_area), 1e-30)

    cot0 = np.sum((p1 - p0) * (p2 - p0), axis=1) / (2.0 * abs_flat)
    cot1 = np.sum((p0 - p1) * (p2 - p1), axis=1) / (2.0 * abs_flat)
    cot2 = np.sum((p0 - p2) * (p1 - p2), axis=1) / (2.0 * abs_flat)
    min_cot = np.minimum(np.minimum(cot0, cot1), cot2)

    polar_v = np.zeros(len(x_prime), dtype=bool)
    polar_v[oc_vtx] = True
    is_polar = polar_v[t_grid].any(axis=1)

    shape_dev    = (COT_IDEAL - min_cot) / COT_IDEAL
    shape_weight = np.clip(1.0 - shape_dampen * shape_dev, 0.0, 1.0)
    angle_weight = np.where(is_polar, 1.0, shape_weight)

    w = 2.0 * np.log(ratios) * sign_f / abs_flat * angle_weight

    grad = np.zeros_like(x_prime)
    for vi, pa, pb in ((v0, p1, p2), (v1, p2, p0), (v2, p0, p1)):
        opp = pb - pa
        dg  = 0.5 * w[:, None] * np.stack([-opp[:, 1], opp[:, 0]], axis=1)
        mask = move_mask[vi]
        np.add.at(grad, vi[mask], dg[mask])
    return grad


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from scipy.interpolate import CloughTocher2DInterpolator, NearestNDInterpolator

    rg    = Registrar()
    b_raw = rg.domain('b_raw')
    g_gcd = rg.domain('g_gcd')
    mode  = H9O.oid_mo[OCTANT_ID]

    # ── 1. Build L3 triangular grid ──────────────────────────────────────────
    print(f'Building L{LAYER} grid (octant {OCTANT_ID}, mode {mode})...')
    verts_boct, _edges, tris = tri_mesh(LAYER, mode)

    # tri_mesh returns b_oct coordinates.  At grid-generation time there is no
    # warp yet, so b_oct == b_raw (unwarped face coords).  The Sinkhorn/gradient
    # optimisation works entirely in this unwarped b_raw space, and the warp
    # displacement (target - source) is what gets stored.  This matches how
    # sinkhorn_toy_grid_gen.py stores xy_vert and how gradient scripts load it.
    l3_src = verts_boct.copy()            # (V, 2) unwarped face coords = b_raw

    t_grid  = tris                        # (T, 3) vertex index array
    n_verts = len(l3_src)
    n_tris  = len(t_grid)
    print(f'  {n_verts} vertices, {n_tris} triangles')

    # Boundary classification (location() expects x * R3, y in b_oct convention)
    locs   = location(verts_boct[:, 0] * H9K.R3, verts_boct[:, 1], mode)
    oc_vtx = np.flatnonzero(locs == BaryLoc.VTX)
    oc_edg = np.flatnonzero(locs == BaryLoc.EDG)
    print(f'  Boundary: {len(oc_vtx)} corner vertices, {len(oc_edg)} edge vertices')

    # Per-triangle mode (for diagnostic reporting)
    tri_modes = build_tri_modes(LAYER, mode)
    print(f'  Triangle modes: {(tri_modes==0).sum()} mode-0, {(tri_modes==1).sum()} mode-1')

    # ── 2. Initialise x_prime ────────────────────────────────────────────────
    if L3_CHECKPOINT is not None:
        print(f'\nResuming L3 checkpoint: {L3_CHECKPOINT}')
        ckpt    = np.load(L3_CHECKPOINT, allow_pickle=True)
        x_prime = ckpt['target_pts'].copy()
        print(f'  Resumed: MAE {float(ckpt["mae"]):.8f}')
    elif WARM_START_FROM is not None:
        print(f'\nWarm-starting from: {WARM_START_FROM}')
        ws      = np.load(WARM_START_FROM, allow_pickle=True)
        ws_src  = ws['source_pts']
        ws_tgt  = ws['target_pts']
        ws_disp = ws_tgt - ws_src
        ws_layer = int(ws['layer']) if 'layer' in ws.files else '?'
        ws_mae   = float(ws['mae'])  if 'mae'   in ws.files else float('nan')
        print(f'  Source warp: layer={ws_layer}, {len(ws_src)} pts, MAE={ws_mae:.8f}')
        print(f'  CT-interpolating displacement to {n_verts} L{LAYER} vertices...')
        ct      = CloughTocher2DInterpolator(ws_src, ws_disp)
        l3_disp = ct(l3_src)
        nan_mask = np.any(np.isnan(l3_disp), axis=1)
        if nan_mask.any():
            print(f'  NearestND fallback for {nan_mask.sum()} vertices outside hull')
            nn = NearestNDInterpolator(ws_src, ws_disp)
            l3_disp[nan_mask] = nn(l3_src[nan_mask])
        x_prime = l3_src + l3_disp
        x_prime = snap_boundary_analytic(x_prime, oc_edg, oc_vtx, mode)
        ckpt    = None
    else:
        print('\nFresh start: identity initialisation (no warm-start)')
        x_prime = l3_src.copy()
        ckpt    = None

    # ── 3. Move / fixed masks ────────────────────────────────────────────────
    move_mask = np.ones(n_verts, dtype=bool)
    move_mask[oc_vtx] = False        # octant corners are always fixed

    # ── 4. Baseline measurement ──────────────────────────────────────────────
    areas, ratios, c_ideal = measure_areas(x_prime, t_grid, rg, b_raw, g_gcd)
    mae = float(np.mean(np.abs(ratios - 1.0)))
    print(f'\n[Baseline]  MAE: {mae:.8f} | '
          f'Min/Max: {ratios.min():.6f} / {ratios.max():.6f}')
    report_mode_split(ratios, tri_modes)

    # ── 5. Adam state ────────────────────────────────────────────────────────
    best_mae     = mae
    best_min     = ratios.min()
    best_max     = ratios.max()
    best_x_prime = x_prime.copy()
    step         = STEP_SIZE
    no_improve   = 0
    prev_mae     = mae
    mae_window   = [mae] * 6

    if ckpt is not None and 'adam_t' in ckpt.files:
        adam_m = ckpt['adam_m'].copy()
        adam_v = ckpt['adam_v'].copy()
        adam_t = int(ckpt['adam_t'])
        step   = float(ckpt['step_size']) if 'step_size' in ckpt.files else STEP_SIZE
        start_iter = adam_t
        print(f'  Restored Adam state: t={adam_t}, lr={step:.2e}')
    else:
        adam_m     = np.zeros_like(x_prime)
        adam_v     = np.zeros_like(x_prime)
        adam_t     = 0
        start_iter = 0

    # ── 6. Adam gradient descent ─────────────────────────────────────────────
    print(f'\n--- L{LAYER} t-grid GRADIENT DESCENT ({ITERATIONS} iters, '
          f'lr={STEP_SIZE:.1e}, shape_dampen={SHAPE_DAMPEN}) ---\n')

    for k in range(start_iter, start_iter + ITERATIONS):
        if step < STEP_MIN:
            print(f'Step below minimum ({STEP_MIN:.1e}), stopping.')
            break

        grad    = compute_gradient(x_prime, t_grid, ratios, c_ideal,
                                   move_mask, oc_vtx)
        adam_t += 1
        adam_m  = ADAM_BETA1 * adam_m + (1.0 - ADAM_BETA1) * grad
        adam_v  = ADAM_BETA2 * adam_v + (1.0 - ADAM_BETA2) * grad ** 2
        m_hat   = adam_m / (1.0 - ADAM_BETA1 ** adam_t)
        v_hat   = adam_v / (1.0 - ADAM_BETA2 ** adam_t)
        delta   = step * m_hat / (np.sqrt(v_hat) + ADAM_EPS)

        x_prime[move_mask] -= delta[move_mask]
        x_prime = snap_boundary_analytic(x_prime, oc_edg, oc_vtx, mode)

        areas, ratios, c_ideal = measure_areas(x_prime, t_grid, rg, b_raw, g_gcd)
        mae      = float(np.mean(np.abs(ratios - 1.0)))
        curr_min = ratios.min()
        curr_max = ratios.max()

        mae_window.append(mae)
        mae_window.pop(0)
        window_min = min(mae_window)
        if prev_mae - window_min < 1e-7:
            no_improve += 1
        else:
            no_improve = 0
        if k >= start_iter + WARMUP_ITERS and no_improve >= STEP_PATIENCE:
            step      *= STEP_COOL
            no_improve = 0
            print(f'  → lr cooled to {step:.2e}')
        prev_mae = mae

        is_best = (mae < best_mae and curr_min > best_min and curr_max < best_max)
        if is_best:
            best_mae     = mae
            best_min     = curr_min
            best_max     = curr_max
            best_x_prime = x_prime.copy()

        print(f'[{k:5d}]  MAE: {mae:.8f} | lr={step:.1e} | '
              f'Min/Max: {curr_min:.6f}/{curr_max:.6f}'
              + (' ← best' if is_best else ''))

        # Mode-split report every 200 iters and on best
        if (k % 200 == 0) or is_best:
            report_mode_split(ratios, tri_modes)

        if (k % SAVE_EVERY == 0) or is_best:
            tag      = 'best' if is_best else f'{k:05d}'
            out_path = Path(f'l3_tgrid_{tag}.npz')
            np.savez(
                out_path,
                layer=np.array(LAYER),
                source_pts=l3_src,
                target_pts=best_x_prime,
                mae=np.array(best_mae),
                step_size=np.array(step),
                shape_dampen=np.array(SHAPE_DAMPEN),
                adam_m=adam_m,
                adam_v=adam_v,
                adam_t=np.array(adam_t),
            )
            if is_best:
                print(f'  Saved best: {out_path}  (MAE {best_mae:.8f})')
