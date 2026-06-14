"""
L5 gradient descent authalic refinement.

Warm-starts from an L4 warp checkpoint by CT-interpolating the L4 displacement
field to L5 vertex positions, then runs Adam gradient descent to minimise area
error across the full L5 mesh.

Pipeline
--------
1. Load L4 warp (target_pts - source_pts = displacement field).
2. CT-interpolate displacement to L5 b_raw vertex positions → warm-start x_prime.
3. Snap boundary vertices onto the octant triangle edges/corners.
4. Run Adam gradient descent: O(N_tri_L5) per iter ≈ 9× L4 cost.

Resume: set L5_CHECKPOINT to an existing L5 .npz output to continue a run.
Fresh:  set L5_CHECKPOINT = None; L4_CHECKPOINT provides the warm start.

Tuning guide
------------
STEP_SIZE too large  → MAE oscillates or triangles fold
STEP_SIZE too small  → very slow progress
WARMUP_ITERS        → suppress cooling during Adam moment build-up (~200 iters)
STEP_PATIENCE       → iters of non-improvement before lr cooling fires
"""

from pathlib import Path
import numpy as np

from hhg9 import Registrar, Points
from hhg9.algorithms.distance import wgs84_area
from hhg9.h9 import H9O

# ── CONFIGURATION ────────────────────────────────────────────────────────────
L4_CHECKPOINT = Path("l4_best_grad_32_s1e-05_01500.npz")   # L4 warp used for CT warm-start
L5_CHECKPOINT = Path("l5_grad_32_s1e-05_00100.npz")   # Set to resume an existing L5 run; None = fresh start
OCTANT_ID     = 0
LAYER         = 5
GRID_FILE     = None                   # None → auto-derive as grid_l{LAYER}.npz

ITERATIONS    = 10000
STEP_SIZE     = 0.00001   # Adam lr — L5 triangles are ~3× smaller so use ~½ of L4 value
STEP_MIN      = 1e-9      # stop when Adam lr cooled below this
STEP_COOL     = 0.80      # multiply lr by this when MAE improvement stalls
STEP_PATIENCE = 100       # Adam warm-up is non-monotone; wait for moments to stabilise
WARMUP_ITERS  = 200       # suppress cooling entirely for first N iters
SEAM_RINGS    = 32        # triangle-rings outward from oc_edg to include

# Adam hyper-parameters
ADAM_BETA1    = 0.9       # first-moment decay (momentum)
ADAM_BETA2    = 0.999     # second-moment decay (per-vertex step scale)
ADAM_EPS      = 1e-8      # denominator stabiliser
RESET_ADAM    = True      # True = ignore saved Adam state, restart moments from zero
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

    # combined per-triangle log weight: 2(r−1) · r · sign / |area|
    # w = 2.0 * (ratios - 1.0) * ratios * sign_f / abs_flat   # (N_tri,)
    w = 2.0 * np.log(ratios) * sign_f / abs_flat

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
    from scipy.interpolate import CloughTocher2DInterpolator, NearestNDInterpolator

    # 1. Setup registrar / domains
    rg    = Registrar()
    b_raw = rg.domain('b_raw')
    g_gcd = rg.domain('g_gcd')
    mode  = H9O.oid_mo[OCTANT_ID]

    # 2. Load L5 grid
    grid_path = Path(GRID_FILE) if GRID_FILE else Path(f"grid_l{LAYER}.npz")
    print(f"Loading grid: {grid_path}")
    repo   = np.load(grid_path, allow_pickle=True)
    l5_src = repo['xy_vert']    # L5 b_raw vertex positions (unwarped)
    oc_vtx = repo['oc_vtx']
    oc_edg = repo['oc_edg']
    t_grid = repo['grid']
    print(f"  L5 grid: {len(l5_src)} vertices, {len(t_grid)} triangles")

    # 3. Initialise x_prime — either resume L5 run or warm-start from L4
    if L5_CHECKPOINT is not None:
        print(f"Resuming L5 checkpoint: {L5_CHECKPOINT}")
        ckpt    = np.load(L5_CHECKPOINT, allow_pickle=True)
        x_prime = ckpt['target_pts'].copy()
        l5_src  = ckpt['source_pts']      # restore L5 source for consistency
        print(f"  Resumed: MAE {float(ckpt['mae']):.8f}")
    else:
        print(f"Warm-starting from L4 checkpoint: {L4_CHECKPOINT}")
        l4_ckpt = np.load(L4_CHECKPOINT, allow_pickle=True)
        l4_tgt  = l4_ckpt['target_pts']
        # L4 source: prefer checkpoint field, fall back to L4 grid file
        if 'source_pts' in l4_ckpt.files:
            l4_src = l4_ckpt['source_pts']
        else:
            l4_layer = int(l4_ckpt['layer']) if 'layer' in l4_ckpt.files else 4
            l4_src = np.load(Path(f"grid_l{l4_layer}.npz"), allow_pickle=True)['xy_vert']
        l4_disp = l4_tgt - l4_src
        print(f"  L4 displacement: {len(l4_src)} pts, MAE {float(l4_ckpt['mae']):.8f}")

        print(f"  CT-interpolating to {len(l5_src)} L5 vertices...")
        ct       = CloughTocher2DInterpolator(l4_src, l4_disp)
        l5_disp  = ct(l5_src)
        nan_mask = np.any(np.isnan(l5_disp), axis=1)
        if nan_mask.any():
            print(f"  NearestND fallback for {nan_mask.sum()} vertices outside hull")
            nn = NearestNDInterpolator(l4_src, l4_disp)
            l5_disp[nan_mask] = nn(l5_src[nan_mask])

        x_prime = l5_src + l5_disp
        x_prime = snap_boundary_analytic(x_prime, oc_edg, oc_vtx, mode)
        ckpt    = None   # no L4 fields carried into L5 saves

    # 4. Seam / move masks
    seam_mask     = build_seam_strip(len(x_prime), oc_edg, t_grid, n_rings=SEAM_RINGS)
    move_mask     = seam_mask.copy()
    move_mask[oc_edg] = True
    move_mask[oc_vtx] = False
    seam_tri_mask = np.array([any(seam_mask[v] for v in tri) for tri in t_grid])
    print(f"Seam strip : {seam_mask.sum()} / {len(seam_mask)} vertices  "
          f"({move_mask.sum()} moveable)")
    print(f"Seam tris  : {seam_tri_mask.sum()} / {len(t_grid)}")

    # 5. Baseline measurement
    areas, ratios, c_ideal = measure_areas(x_prime, t_grid, rg, b_raw, g_gcd)
    mae    = float(np.mean(np.abs(ratios - 1.0)))
    seam_r = ratios[seam_tri_mask]
    print(f"\n[Baseline]  MAE: {mae:.8f} | "
          f"All Min/Max: {ratios.min():.6f}/{ratios.max():.6f} | "
          f"Seam Min/Max: {seam_r.min():.8f}/{seam_r.max():.8f}")

    # 6. Adam state
    best_mae     = mae
    best_min     = ratios.min()
    best_max     = ratios.max()
    best_x_prime = x_prime.copy()
    step         = STEP_SIZE
    no_improve   = 0
    prev_mae     = mae
    mae_window   = [mae] * 6

    files = ckpt.files if ckpt is not None else []
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

    # 7. Gradient descent loop
    print(f"\n--- L{LAYER} GRADIENT DESCENT ({ITERATIONS} iters, "
          f"lr={STEP_SIZE}, rings={SEAM_RINGS}) ---")

    for k in range(ITERATIONS):
        if step < STEP_MIN:
            print(f"Step below minimum ({STEP_MIN:.1e}), stopping.")
            break

        grad   = compute_gradient(x_prime, t_grid, ratios, c_ideal, move_mask)
        adam_t += 1
        adam_m  = ADAM_BETA1 * adam_m + (1.0 - ADAM_BETA1) * grad
        adam_v  = ADAM_BETA2 * adam_v + (1.0 - ADAM_BETA2) * grad ** 2
        m_hat   = adam_m / (1.0 - ADAM_BETA1 ** adam_t)
        v_hat   = adam_v / (1.0 - ADAM_BETA2 ** adam_t)
        delta   = step * m_hat / (np.sqrt(v_hat) + ADAM_EPS)

        x_prime[move_mask] -= delta[move_mask]
        x_prime = snap_boundary_analytic(x_prime, oc_edg, oc_vtx, mode)

        areas, ratios, c_ideal = measure_areas(x_prime, t_grid, rg, b_raw, g_gcd)
        mae    = float(np.mean(np.abs(ratios - 1.0)))
        seam_r = ratios[seam_tri_mask]

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
        curr_min = ratios.min()
        curr_max = ratios.max()

        print(f"[Iter {k:4d}]  MAE: {mae:.8f} | lr={step:.1e} | "
              f"All Min/Max: {curr_min:.8f}/{curr_max:.8f}")
              # f"Seam Min/Max: {seam_r.min():.8f}/{seam_r.max():.8f}")

        best = False
        if mae < best_mae and curr_min > best_min and curr_max < best_max:
            print(f"  New best MAE: {mae:.8f}  (from {best_mae:.8f})")
            best = True
            best_mae     = mae
            best_x_prime = x_prime.copy()

        if (k % 23 == 0) or best:
            out_path = Path(f"l5_grad_all_s{STEP_SIZE}_{k:05d}.npz")
            np.savez(
                out_path,
                layer=LAYER,
                source_pts=l5_src,
                target_pts=best_x_prime,
                mae=best_mae,
                step_size=step,
                seam_rings=SEAM_RINGS,
                grad_iterations=ITERATIONS,
                adam_m=adam_m,
                adam_v=adam_v,
                adam_t=np.array(adam_t),
            )
            print(f"\nBest MAE: {best_mae:.8f}  Saved: {out_path}")
