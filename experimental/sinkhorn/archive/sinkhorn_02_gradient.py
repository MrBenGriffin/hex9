"""
Seam-strip gradient correction — fast alternative to Sinkhorn polish.

Loads a Sinkhorn checkpoint and applies gradient descent directly on vertex
positions to minimise area error in the seam strip. Interior (well-converged)
vertices are untouched; boundary vertices are re-snapped after each step.

Algorithm per iteration
-----------------------
1. Project x_prime → g_gcd, measure wgs84 area ratios for all triangles  O(N_tri)
2. Accumulate analytical gradient for each seam vertex:
       ∂f/∂p_i  =  Σ_{t∋i}  2(r_t - 1) · (r_t / A_t^flat) · 0.5·perp(opp_edge)
   where r_t = wgs84_area / c_ideal,  A_t^flat = 2-D area in b_raw space,
   perp(a,b) = (−b, a),  opp_edge = edge of t opposite to vertex i.
3. Step:  p_i  ←  p_i − α · ∂f/∂p_i    (seam interior only)
4. Re-snap boundary vertices onto triangle edges / corners.

Cost per iteration: O(N_tri) — seconds, not hours.
No transport matrices, no Sinkhorn, no finite-difference probes needed.

Tuning guide
------------
STEP_SIZE too large  → MAE oscillates / seam min gets worse each iter
STEP_SIZE too small  → very slow progress
SEAM_RINGS too small → strip too narrow, on-seam vertices unaffected
SEAM_RINGS too large → gradient "bleeds" into well-converged interior
"""

from pathlib import Path
import numpy as np

from hhg9 import Registrar, Points
from hhg9.algorithms.distance import wgs84_area
from hhg9.h9 import H9O

# ── CONFIGURATION ────────────────────────────────────────────────────────────
CHECKPOINT   = Path("l4_best.npz")
OCTANT_ID    = 0
GRID_FILE    = None      # None → auto-derive from checkpoint layer

ITERATIONS   = 10000
STEP_SIZE    = 0.00001    # Adam learning rate α — smaller reduces warm-up disturbance
STEP_MIN     = 1e-9       # stop when Adam lr cooled below this
STEP_COOL    = 0.80       # multiply lr by this when MAE improvement stalls
STEP_PATIENCE = 100       # Adam warm-up is non-monotone; wait for moments to stabilise
WARMUP_ITERS  = 200       # suppress cooling entirely for first N iters
SEAM_RINGS   = 32         # triangle-rings outward from oc_edg to include

# Adam hyper-parameters — standard defaults work well
ADAM_BETA1   = 0.9        # first-moment decay (momentum)
ADAM_BETA2   = 0.999      # second-moment decay (per-vertex step scale)
ADAM_EPS     = 1e-8       # denominator stabiliser
RESET_ADAM   = True       # True = ignore saved Adam state, restart moments from zero
# ─────────────────────────────────────────────────────────────────────────────


# ── helpers (minimal, standalone) ────────────────────────────────────────────

def get_ideal_corners(mode: int):
    from hhg9.h9 import H9K
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


def build_seam_strip(n_verts, oc_edg, t_grid, n_rings):
    # strip = set(int(v) for v in oc_edg)
    # for _ in range(n_rings):
    #     new = set()
    #     for tri in t_grid:
    #         tri_set = set(int(v) for v in tri)
    #         if strip & tri_set:
    #             new |= tri_set
    #     strip |= new
    # mask = np.zeros(n_verts, dtype=bool)
    # for v in strip:
    #     mask[v] = True
    mask = np.full(n_verts, True, dtype=bool)
    return mask


def measure_areas(x_prime, t_grid, rg, b_raw, g_gcd):
    """Return (areas, ratios, c_ideal) for all triangles."""
    dpts      = Points(x_prime, b_raw, oid=0)
    gpts      = rg.project(dpts, [b_raw, g_gcd])
    t_pts_gcd = gpts.coords[t_grid.ravel()]
    areas     = wgs84_area(rg, Points(t_pts_gcd, g_gcd), 3)
    c_ideal   = float(np.mean(areas))
    ratios    = areas / c_ideal
    return areas, ratios, c_ideal


def compute_gradient(x_prime, t_grid, ratios, c_ideal, move_mask):
    """
    Vectorised gradient of Σ(r_t−1)² w.r.t. vertex positions in b_raw space.

    For vertex i opposite edge (pa→pb) in triangle t:
        w_t       = 2(r_t−1) · r_t · sign(flat_area_t) / |flat_area_t|
        ∂f/∂v_i  += w_t · 0.5 · perp(pb − pa)   where perp(a,b) = (−b, a)

    Signed flat_area is essential: CW (inverted) triangles have negative area,
    flipping the gradient direction correctly for those cells.
    np.add.at handles the many-to-one vertex accumulation without buffering.
    """
    v0 = t_grid[:, 0]
    v1 = t_grid[:, 1]
    v2 = t_grid[:, 2]
    p0 = x_prime[v0]   # (N_tri, 2)
    p1 = x_prime[v1]
    p2 = x_prime[v2]

    e1 = p1 - p0
    e2 = p2 - p0
    flat_area = 0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    sign_f    = np.where(flat_area >= 0.0, 1.0, -1.0)
    abs_flat  = np.maximum(np.abs(flat_area), 1e-30)

    # combined per-triangle weight: 2(r−1) · r · sign / |area|
    w = 2.0 * (ratios - 1.0) * ratios * sign_f / abs_flat   # (N_tri,)

    grad = np.zeros_like(x_prime)

    # Each vertex's contribution: w * 0.5 * perp(opposite_edge)
    # opposite edges: v0↔(p1,p2),  v1↔(p2,p0),  v2↔(p0,p1)
    for vi, pa, pb in ((v0, p1, p2), (v1, p2, p0), (v2, p0, p1)):
        opp  = pb - pa                                               # (N_tri, 2)
        dg   = 0.5 * w[:, None] * np.stack([-opp[:, 1], opp[:, 0]], axis=1)
        mask = move_mask[vi]
        np.add.at(grad, vi[mask], dg[mask])

    return grad


# ── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # 1. Load checkpoint
    print(f"Loading checkpoint: {CHECKPOINT}")
    ckpt      = np.load(CHECKPOINT, allow_pickle=True)
    x_prime   = ckpt['target_pts'].copy()
    layer     = int(ckpt['layer'])
    if 'iteration' in ckpt:
        ck_iter   = int(ckpt['iteration'])
    else:
        ck_iter   = 0
    print(f"  Layer {layer}, iter {ck_iter}, "
          f"checkpoint MAE {float(ckpt['mae']):.8f}")

    # 2. Load grid
    grid_path = Path(GRID_FILE) if GRID_FILE else Path(f"grid_l{layer}.npz")
    print(f"Loading grid: {grid_path}")
    repo   = np.load(grid_path, allow_pickle=True)
    oc_vtx = repo['oc_vtx']
    oc_edg = repo['oc_edg']
    t_grid = repo['grid']

    # 3. Setup registrar / domains
    rg    = Registrar()
    b_raw = rg.domain('b_raw')
    g_gcd = rg.domain('g_gcd')
    mode  = H9O.oid_mo[OCTANT_ID]

    # 4. Seam strip mask (interior subset — boundary vertices never move)
    seam_mask = build_seam_strip(len(x_prime), oc_edg, t_grid, n_rings=SEAM_RINGS)
    # exclude boundary vertices from gradient update
    move_mask = seam_mask.copy()
    move_mask[oc_edg] = True  # allows edges to move...
    move_mask[oc_vtx] = False

    seam_tri_mask = np.array([any(seam_mask[v] for v in tri) for tri in t_grid])

    print(f"Seam strip : {seam_mask.sum()} / {len(seam_mask)} vertices  "
          f"({move_mask.sum()} moveable)")
    print(f"Seam tris  : {seam_tri_mask.sum()} / {len(t_grid)}")

    # 5. Baseline measurement
    areas, ratios, c_ideal = measure_areas(x_prime, t_grid, rg, b_raw, g_gcd)
    mae      = float(np.mean(np.abs(ratios - 1.0)))
    seam_r   = ratios[seam_tri_mask]
    print(f"\n[Baseline]  MAE: {mae:.8f} | "
          f"All Min/Max: {ratios.min():.6f}/{ratios.max():.6f} | "
          f"Seam Min/Max: {seam_r.min():.8f}/{seam_r.max():.8f}")

    # 6. Gradient descent loop
    print(f"\n--- GRADIENT DESCENT ({ITERATIONS} iters, "
          f"step={STEP_SIZE}, rings={SEAM_RINGS}) ---")

    best_mae      = mae
    best_x_prime  = x_prime.copy()
    step          = STEP_SIZE
    no_improve    = 0
    prev_mae      = mae
    mae_window    = [mae] * 6   # rolling window — detects period-2 oscillation

    # Adam state — restored from checkpoint if available, else zero-initialised.
    # Adam state: restore from checkpoint unless RESET_ADAM=True (e.g. after over-cooling).
    files = ckpt.files
    if not RESET_ADAM and 'adam_t' in files:
        adam_m = ckpt['adam_m'].copy()
        adam_v = ckpt['adam_v'].copy()
        adam_t = int(ckpt['adam_t'])
        step   = float(ckpt['step_size']) if 'step_size' in files else STEP_SIZE
        print(f"  Restored Adam state: t={adam_t}, lr={step:.2e}")
    else:
        adam_m = np.zeros_like(x_prime)
        adam_v = np.zeros_like(x_prime)
        adam_t = 0
        if RESET_ADAM and 'adam_t' in files:
            print(f"  Adam state reset (RESET_ADAM=True); lr={step:.2e}")

    for k in range(ITERATIONS):
        if step < STEP_MIN:
            print(f"Step below minimum ({STEP_MIN:.1e}), stopping.")
            break

        grad = compute_gradient(x_prime, t_grid, ratios, c_ideal, move_mask)

        # Adam update — bias-corrected per-vertex adaptive step sizes.
        # Vertices with consistently small gradients (house-icon cells near oc_vtx)
        # accumulate moment and get progressively larger effective steps; well-
        # converged vertices plateau naturally as their gradient magnitude → 0.
        adam_t += 1
        adam_m = ADAM_BETA1 * adam_m + (1.0 - ADAM_BETA1) * grad
        adam_v = ADAM_BETA2 * adam_v + (1.0 - ADAM_BETA2) * grad ** 2
        m_hat  = adam_m / (1.0 - ADAM_BETA1 ** adam_t)
        v_hat  = adam_v / (1.0 - ADAM_BETA2 ** adam_t)
        delta  = step * m_hat / (np.sqrt(v_hat) + ADAM_EPS)

        x_prime[move_mask] -= delta[move_mask]
        x_prime = snap_boundary_analytic(x_prime, oc_edg, oc_vtx, mode)

        areas, ratios, c_ideal = measure_areas(x_prime, t_grid, rg, b_raw, g_gcd)
        mae    = float(np.mean(np.abs(ratios - 1.0)))
        seam_r = ratios[seam_tri_mask]

        # Cooling: trigger when window minimum hasn't improved — catches both
        # genuine stalls and period-2 oscillation (where alternate iterations
        # improve/worsen, so consecutive-improvement check never fires).
        mae_window.append(mae)
        mae_window.pop(0)
        window_min = min(mae_window)
        if prev_mae - window_min < 1e-6:
            no_improve += 1
        else:
            no_improve = 0
        if k >= WARMUP_ITERS and no_improve >= STEP_PATIENCE:
            step      *= STEP_COOL
            no_improve = 0
            print(f"  → cooling step to {step:.2e}")
        prev_mae = mae

        print(f"[Iter {k+1:3d}]  MAE: {mae:.8f} | lr={step:.1e} | "
              f"All Min/Max: {ratios.min():.6f}/{ratios.max():.6f} | "
              f"Seam Min/Max: {seam_r.min():.8f}/{seam_r.max():.8f}")

        if mae < best_mae:
            best_mae     = mae
            best_x_prime = x_prime.copy()

        if k % 100 == 0:
            # 7. Save best result — carry ALL original checkpoint fields forward so the
            # output is a valid RESUME_FROM target for sinkhorn_01_l4.py (v_prev_corr,
            # strength, mae_prev etc. are preserved even if this script didn't use them).
            stem     = CHECKPOINT.stem
            out_path = CHECKPOINT.parent / f"{stem}_grad_{SEAM_RINGS}_s{STEP_SIZE}_{k:05d}.npz"
            save_dict = {o: ckpt[o] for o in ckpt.files}   # copy all original fields
            save_dict.update(
                mae=best_mae,
                target_pts=best_x_prime,
                grad_iterations=ITERATIONS,
                step_size=step,            # current (possibly cooled) lr
                seam_rings=SEAM_RINGS,
                adam_m=adam_m,
                adam_v=adam_v,
                adam_t=np.array(adam_t),
            )
            np.savez(out_path, **save_dict)
            print(f"\nBest MAE: {best_mae:.8f}")
            print(f"Saved: {out_path}")
