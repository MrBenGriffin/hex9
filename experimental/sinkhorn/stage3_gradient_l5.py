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
from experimental.sinkhorn.fast_area import fast_wgs84_area as wgs84_area_coarse
from hhg9.algorithms.distance import wgs84_area
from hhg9.h9 import H9O
from hhg9.h9.polygon import region_grid, H9P
from experimental.sinkhorn import config

# ── CONFIGURATION ────────────────────────────────────────────────────────────
L4_CHECKPOINT = None   # stage-2 output to warm-start from; overridable with --warm-start
L5_CHECKPOINT = None   # set, or pass --resume, to continue an existing L5 run
OCTANT_ID     = config.OCTANT_ID
LAYER         = 5
GRID_FILE     = None                   # None → auto-derive as grid_l{LAYER}.npz

ITERATIONS    = 10000
STEP_SIZE     = 0.00002   # Adam lr — L5 triangles are ~3× smaller so use ~½ of L4 value
STEP_MIN      = 1e-10     # stop when Adam lr cooled below this
STEP_COOL     = 0.95      # multiply lr by this when MAE improvement stalls
STEP_PATIENCE_LOCAL  = 300   # iters of "no descent in recent window" before cool.
                             # Catches pure local plateau / noise-band stagnation.
STEP_PATIENCE_GLOBAL = 600   # iters since last new global best before cool.
                             # Catches the case where Adam keeps finding tiny
                             # local dips that reset local patience but never
                             # produces a real new best.
WARMUP_ITERS  = 200       # suppress cooling entirely for first N iters
SHAPE_DAMPEN  = 0.8       # gradient dampener for non-equilateral triangles.
                          # 0 = no dampening; 1 = full suppression at 90°.
                          # Triangles with all angles near 60° (equilateral → regular hexagons)
                          # get full weight; weight falls as the largest angle exceeds 60°.
                          # Polar triangles (oc_vtx) are always exempt.
MB_LAMBDA     = 0.5       # mode-balance correction strength (0 = off).
                          # Each iteration, the effective log-ratio used in the gradient
                          # is shifted by ±(MB_LAMBDA × half_asymmetry) so that mode-0 and
                          # mode-1 triangles receive a corrective nudge proportional to the
                          # current inter-mode offset.  MB_LAMBDA=1.0 applies the full
                          # half-asymmetry correction; 0.5 applies half of it per step.
                          # Does not change the gradient structure, only per-mode weighting.

# Adam hyper-parameters
ADAM_BETA1    = 0.9       # first-moment decay (momentum)
ADAM_BETA2    = 0.999     # second-moment decay (per-vertex step scale)
ADAM_EPS      = 1e-8      # denominator stabiliser
RESET_ADAM    = True      # True = ignore saved Adam state, restart moments from zero
OUTPUT_STR    = 'output/stage3'   # all checkpoints land here
# ─────────────────────────────────────────────────────────────────────────────


# ── helpers (minimal, standalone) ────────────────────────────────────────────

def get_ideal_corners(mode: int):
    from hhg9.h9 import H9K
    tr, vf, vc = H9K.limits.TR, H9K.limits.VF, H9K.limits.VC
    if int(mode) == 0:
        return np.array([[-tr, vc], [tr, vc], [0.0, vf]])
    else:
        return np.array([[-tr, vf], [tr, vf], [0.0, vc]])


def snap_boundary_analytic(xy, oc_edg, oc_vtx, mode, mirror_idx=None):
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
    # On-mirror boundary points (polar apex + base-edge midpoint) must stay at x=0
    boundary = np.array(list(oc_vtx) + list(oc_edg), dtype=int)
    on_mirror = boundary[np.abs(out[boundary, 0]) < 1e-10]
    out[on_mirror, 0] = 0.0
    # Full interior mirror symmetry: dx antisymmetric, dy symmetric
    if mirror_idx is not None:
        dx = (out[:, 0] - out[mirror_idx, 0]) / 2.0
        out[:, 0]           =  dx
        out[mirror_idx, 0]  = -dx
        dy = (out[:, 1] + out[mirror_idx, 1]) / 2.0
        out[:, 1]           =  dy
        out[mirror_idx, 1]  =  dy
    return out


def measure_areas(x_prime, t_grid, rg, b_raw, g_gcd, area_fn):
    """Return (areas, ratios, c_ideal) for all triangles.

    area_fn selects between wgs84_area_coarse (fast vectorised approximation,
    typical relative error ~1e-4) and wgs84_area (strict geodesic per call,
    ~10-100× slower but accurate to geographiclib precision).
    """
    dpts      = Points(x_prime, b_raw, oid=0)
    gpts      = rg.project(dpts, [b_raw, g_gcd])
    t_pts_gcd = gpts.coords[t_grid.ravel()]
    areas     = area_fn(rg, Points(t_pts_gcd, g_gcd), 3)
    c_ideal   = float(np.mean(areas))
    ratios    = areas / c_ideal
    return areas, ratios, c_ideal


COT_IDEAL = 1.0 / np.sqrt(3)   # cot(60°) — target for equilateral triangles / regular hexagons


def symmetrize_pairs(vec, mirror_idx, anti_x=True):
    """Enforce x=0 mirror symmetry on a per-vertex (N,2) array.

    Use anti_x=True for vector-like quantities where x flips sign under
    reflection (gradients, displacements, Adam first moment m).
    Use anti_x=False for sign-invariant quantities where both components are
    symmetric under reflection (squared gradients, Adam second moment v).

    For self-paired (on-axis) vertices mirror_idx[i] == i; the anti_x=True
    formula yields x=0 automatically. mirror_idx must be an involution.
    """
    other = vec[mirror_idx]
    out = np.empty_like(vec)
    if anti_x:
        out[:, 0] = (vec[:, 0] - other[:, 0]) / 2.0
    else:
        out[:, 0] = (vec[:, 0] + other[:, 0]) / 2.0
    out[:, 1] = (vec[:, 1] + other[:, 1]) / 2.0
    return out


def compute_gradient(x_prime, t_grid, ratios, c_ideal, move_mask, oc_vtx,
                     shape_dampen=SHAPE_DAMPEN,
                     tri_modes=None, mb_lambda=0.0):
    """
    Vectorised gradient of Σ log(r_t)² w.r.t. vertex positions in b_raw space.

    For vertex i opposite edge (pa→pb) in triangle t:
        w_t       = 2 log(r_t) · sign(flat_area_t) / |flat_area_t|
        ∂f/∂v_i  += w_t · 0.5 · perp(pb − pa)   where perp(a,b) = (−b, a)

    Signed flat_area is essential: CW (inverted) triangles have negative area,
    flipping the gradient direction correctly for those cells.
    np.add.at handles the many-to-one vertex accumulation without buffering.

    Shape dampener: each hexagon interior angle is 120° when its constituent
    triangles are equilateral (all angles 60°, cot = 1/√3). As the largest
    triangle angle grows beyond 60°, min_cot falls below COT_IDEAL — a signal
    that the hexagon is becoming non-round. The dampener reduces the gradient
    weight continuously as min_cot falls below COT_IDEAL, reaching
    (1 − shape_dampen) at 90° and zero at (1 + 1/shape_dampen)× COT_IDEAL below
    zero. Polar triangles (oc_vtx) are always exempt — their angles are
    topologically constrained by the 4-valent octahedral vertex.

    Mode-balance correction (mb_lambda > 0, tri_modes provided):
    The AK projection has an intrinsic inter-mode area bias: mode-0 (∇) triangles
    are systematically under-area and mode-1 (△) are over-area.  The shared-vertex
    gradient cannot eliminate this bias unaided.  When mb_lambda > 0, the effective
    log(r_t) used in the gradient is shifted by ±(mb_lambda × half_asymmetry) for
    each mode, amplifying the corrective signal for the lagging mode.

    Sign convention: if mode-0 is under-area (log(r_0) < 0, m0 < m1):
      asym = m0 - m1 < 0
      mode-0 shift = -asym * mb_lambda/2 > 0   → effective_log more negative → expands ✓
      mode-1 shift =  asym * mb_lambda/2 < 0   → effective_log more positive → contracts ✓
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

    # Shape dampener — cotangent of each interior angle:
    #   cot(θ_i) = dot(edge_a, edge_b) / (2 · |flat_area|)
    #   cot(60°) = 1/√3 ≈ 0.577  (equilateral, ideal)
    #   cot(90°) = 0              (right angle)
    #   cot < 0                   (obtuse)
    cot0 = np.sum((p1 - p0) * (p2 - p0), axis=1) / (2.0 * abs_flat)
    cot1 = np.sum((p0 - p1) * (p2 - p1), axis=1) / (2.0 * abs_flat)
    cot2 = np.sum((p0 - p2) * (p1 - p2), axis=1) / (2.0 * abs_flat)
    min_cot = np.minimum(np.minimum(cot0, cot1), cot2)

    # Polar triangle mask — oc_vtx triangles have topologically mandated angles.
    polar_v = np.zeros(len(x_prime), dtype=bool)
    polar_v[oc_vtx] = True
    is_polar = polar_v[t_grid].any(axis=1)   # (N_tri,)

    # Continuous dampener: full weight at equilateral (min_cot = COT_IDEAL);
    # falls linearly as the largest angle exceeds 60°; clips to 0 for very obtuse.
    shape_dev = (COT_IDEAL - min_cot) / COT_IDEAL   # 0 at ideal, 1 at 90°, >1 obtuse
    shape_weight = np.clip(1.0 - shape_dampen * shape_dev, 0.0, 1.0)
    angle_weight = np.where(is_polar, 1.0, shape_weight)

    # Effective log-ratio: optionally shifted per-mode to correct inter-mode bias.
    log_r = np.log(ratios)
    if mb_lambda > 0.0 and tri_modes is not None:
        m0       = log_r[tri_modes == 0].mean()
        m1       = log_r[tri_modes == 1].mean()
        asym     = m0 - m1          # negative when mode-0 is under-area
        shift    = asym * mb_lambda * 0.5
        # mode-0: subtract -shift (= add |shift|) → more negative effective_log
        # mode-1: subtract +shift (= subtract |shift|) → more positive effective_log
        mode_shift = np.where(tri_modes == 0, -shift, shift)
        log_r      = log_r - mode_shift

    # Combined per-triangle log weight, modulated by angle guard.
    w = 2.0 * log_r * sign_f / abs_flat * angle_weight

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
    import argparse
    from scipy.interpolate import CloughTocher2DInterpolator, NearestNDInterpolator

    ap = argparse.ArgumentParser(description='Stage 3 — L5 Adam gradient refinement.')
    ap.add_argument('--ellipsoid', default=config.INDEX_ELLIPSOID,
                    choices=list(config.ELLIPSOIDS))
    ap.add_argument('--resume', default=L5_CHECKPOINT,
                    help='L5 checkpoint .npz to resume from')
    ap.add_argument('--warm-start', default=L4_CHECKPOINT,
                    help='stage-2 (L4) .npz to warm-start from')
    ap.add_argument('--iterations', type=int, default=ITERATIONS,
                    help='Cap on Adam iterations (default %(default)s).')
    ap.add_argument('--output_dir', type=str, default=OUTPUT_STR,
                    help='Cap on Adam iterations (default %(default)s).')
    ap.add_argument('--lr', type=float, default=STEP_SIZE,
                    help='Adam learning rate (default %(default)s). Drop for '
                         'near-converged warm-starts to avoid early transient.')
    ap.add_argument('--strict-best', action=argparse.BooleanOptionalAction,
                    default=True,
                    help='Require non-regression on min/max tails for a save. '
                         'Default is on (L5 mesh is expected to be able to '
                         'tighten both tails). Use --no-strict-best to fall '
                         'back to MAE-only saving if Adam stops finding any '
                         'tail-preserving improvements.')
    ap.add_argument('--area', choices=['coarse', 'strict'], default='coarse',
                    help='Area integrator. "coarse" uses the fast vectorised '
                         'approximation (~1e-4 relative error, default). '
                         '"strict" uses geodesic per-triangle solves '
                         '(~10-100× slower, geographiclib precision). '
                         'Switch to strict for the polish phase once MAE '
                         'descends near the coarse error floor. On resume the '
                         'baseline is re-measured with the active area_fn so '
                         'best_mae is on the right scale.')
    args = ap.parse_args()
    L4_CHECKPOINT = Path(args.warm_start) if args.warm_start else None
    L5_CHECKPOINT = Path(args.resume)     if args.resume     else None
    ITERATIONS = args.iterations
    STEP_SIZE  = args.lr
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Setup registrar / domains
    rg    = Registrar()
    ellipsoid = config.apply_ellipsoid(rg, args.ellipsoid)
    print(f"Ellipsoid: {ellipsoid}")
    b_raw = rg.domain('b_raw')
    g_gcd = rg.domain('g_gcd')
    mode  = H9O.oid_mo[OCTANT_ID]

    # Bind area integrator from the --area flag. measure_areas takes this as a
    # parameter so the same function is used for the baseline AND every iter,
    # guaranteeing best_mae is always comparable to current mae.
    area_fn = wgs84_area if args.area == 'strict' else wgs84_area_coarse
    print(f"Area integrator: {args.area}")

    # 2. Load L5 grid
    grid_path = Path(GRID_FILE) if GRID_FILE else Path(f"grid_l{LAYER}.npz")
    print(f"Loading grid: {grid_path}")
    repo   = np.load(grid_path, allow_pickle=True)
    l5_src = repo['xy_vert']    # L5 b_raw vertex positions (unwarped)
    oc_vtx = repo['oc_vtx']
    oc_edg = repo['oc_edg']
    t_grid = repo['grid']
    print(f"  L5 grid: {len(l5_src)} vertices, {len(t_grid)} triangles")

    # Precompute mirror_idx: for each vertex (x,y), find index of (-x,y) partner.
    from scipy.spatial import cKDTree
    mirror_query = np.column_stack([-l5_src[:, 0], l5_src[:, 1]])
    _, mirror_idx = cKDTree(l5_src).query(mirror_query, k=1)
    print(f"  Mirror symmetry: max mirror-pair distance = "
          f"{np.linalg.norm(l5_src[mirror_idx] - mirror_query, axis=1).max():.3e}")

    on_axis = np.where(mirror_idx == np.arange(len(l5_src)))[0]
    print(f"  On-axis vertices (x=0): {len(on_axis)}")

    # Per-triangle mode (0=∇, 1=△) — used by mode-balance correction
    tri_modes = np.array(
        [m for (_path, m, _origin, _scale) in region_grid(LAYER, mode, H9P)],
        dtype=np.int8,
    )
    print(f"  Triangle modes: {(tri_modes==0).sum()} mode-0, {(tri_modes==1).sum()} mode-1")

    # 3. Initialise x_prime — either resume L5 run or warm-start from L4
    if L5_CHECKPOINT is not None:
        print(f"Resuming L5 checkpoint: {L5_CHECKPOINT}")
        ckpt    = np.load(L5_CHECKPOINT, allow_pickle=True)
        x_prime = ckpt['target_pts'].copy()
        l5_src  = ckpt['source_pts']      # restore L5 source for consistency
        if 'mae' in ckpt:
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
        x_prime = snap_boundary_analytic(x_prime, oc_edg, oc_vtx, mode, mirror_idx)
        ckpt    = None   # no L4 fields carried into L5 saves

    # 4. Move mask — all vertices except fixed octant corners and lateral edges.
    #    Lateral edges are pinned to zero displacement so the warp is invertible
    #    at the seam/apex.
    lat_mask = config.lateral_edge_mask(l5_src)
    move_mask = np.ones(len(x_prime), dtype=bool)
    move_mask[oc_vtx]   = False
    move_mask[lat_mask] = False
    # No lateral-edge identity reset: preserve the warm-started slide-along-the-
    # edge positions from stage 2 (or stage 1 via passthrough). move_mask blocks
    # Adam from touching them; snap_boundary_analytic keeps them on the edge line.
    print(f"Moveable vertices: {move_mask.sum()} / {len(move_mask)} "
          f"({int(lat_mask.sum())} lateral-edge vertices pinned)")

    # 5. Baseline measurement (always uses the active area_fn, so best_mae
    #    starts on the correct scale even on resume after an --area switch).
    areas, ratios, c_ideal = measure_areas(x_prime, t_grid, rg, b_raw, g_gcd, area_fn)
    mae    = float(np.mean(np.abs(ratios - 1.0)))
    pct_dev = (ratios - 1.0) * 100.0
    m0_mean = pct_dev[tri_modes == 0].mean()
    m1_mean = pct_dev[tri_modes == 1].mean()
    print(f"\n[Baseline]  MAE: {mae:.8f} | "
          f"All Min/Max: {ratios.min():.6f}/{ratios.max():.6f} | "
          f"mode Δ: {m0_mean:+.4f}% / {m1_mean:+.4f}%")

    # 6. Adam state
    best_mae     = mae
    best_min     = ratios.min()
    best_max     = ratios.max()
    best_x_prime = x_prime.copy()
    step         = STEP_SIZE
    no_improve_local  = 0
    no_improve_global = 0
    prev_mae     = mae
    mae_window   = [mae] * 6

    files = ckpt.files if ckpt is not None else []
    if not RESET_ADAM and 'adam_t' in files:
        adam_m = ckpt['adam_m'].copy()
        adam_v = ckpt['adam_v'].copy()
        adam_t = int(ckpt['adam_t'])
        # Flush any asymmetric drift carried in checkpointed moments.
        adam_m = symmetrize_pairs(adam_m, mirror_idx, anti_x=True)
        adam_v = symmetrize_pairs(adam_v, mirror_idx, anti_x=False)
        step   = float(ckpt['step_size']) if 'step_size' in files else STEP_SIZE
        print(f"  Restored Adam state: t={adam_t}, lr={step:.2e}")
    else:
        adam_m = np.zeros_like(x_prime)
        adam_v = np.zeros_like(x_prime)
        adam_t = 0
    STEP_SIZE = args.lr

    # 7. Gradient descent loop
    print(f"\n--- L{LAYER} GRADIENT DESCENT ({ITERATIONS} iters, lr={STEP_SIZE}) ---")

    for k in range(ITERATIONS):
        if step < STEP_MIN:
            print(f"Step below minimum ({STEP_MIN:.1e}), stopping.")
            break

        grad   = compute_gradient(x_prime, t_grid, ratios, c_ideal, move_mask, oc_vtx,
                                   tri_modes=tri_modes, mb_lambda=MB_LAMBDA)
        # Anti-symmetric in x: dE/dx_i = -dE/dx_j for mirrored pair (i, j).
        # Keeps adam_m / adam_v structurally symmetric for all subsequent iters.
        grad   = symmetrize_pairs(grad, mirror_idx, anti_x=True)
        adam_t += 1
        adam_m  = ADAM_BETA1 * adam_m + (1.0 - ADAM_BETA1) * grad
        adam_v  = ADAM_BETA2 * adam_v + (1.0 - ADAM_BETA2) * grad ** 2
        m_hat   = adam_m / (1.0 - ADAM_BETA1 ** adam_t)
        v_hat   = adam_v / (1.0 - ADAM_BETA2 ** adam_t)
        # Linear lr warmup over WARMUP_ITERS to absorb Adam's bias-correction
        # transient at t=1 (when m_hat = grad/0.1, v_hat = grad²/0.001 produce
        # a near-sign-step of magnitude lr·0.316). From a near-converged warm-
        # start, that first step randomises the very state we wanted to keep —
        # ramping the effective lr from 0 → step over WARMUP_ITERS lets the
        # moments fill up at near-zero displacement before lr engages.
        warmup_scale  = min(1.0, adam_t / max(WARMUP_ITERS, 1))
        effective_step = step * warmup_scale
        delta   = effective_step * m_hat / (np.sqrt(v_hat) + ADAM_EPS)

        x_prime[move_mask] -= delta[move_mask]
        x_prime = snap_boundary_analytic(x_prime, oc_edg, oc_vtx, mode, mirror_idx)
        # No lateral-edge identity reset: move_mask blocks Adam from updating
        # them, and snap_boundary_analytic keeps them on the edge line.

        areas, ratios, c_ideal = measure_areas(x_prime, t_grid, rg, b_raw, g_gcd, area_fn)
        mae    = float(np.mean(np.abs(ratios - 1.0)))

        # Local patience: windowed plateau detector.
        mae_window.append(mae)
        mae_window.pop(0)
        window_min = min(mae_window)
        if prev_mae - window_min < 1e-6:
            no_improve_local += 1
        else:
            no_improve_local = 0
        prev_mae = mae
        curr_min = ratios.min()
        curr_max = ratios.max()

        # --strict-best (default): require non-regression on both tails. Avoids
        # the min/max ratchet-down bug AND ensures the saved best is a genuine
        # tightening of both bulk and tails — appropriate for L5 where the
        # finer mesh has the degrees of freedom to handle the octahedral
        # vertex tension that L4 cannot.
        # --no-strict-best: MAE-only (fallback if Adam stalls on the tighter
        # criterion).
        if args.strict_best:
            no_regress = (curr_min >= best_min) and (curr_max <= best_max)
            strict_one = (curr_min > best_min) or (curr_max < best_max)
            best = (mae < best_mae and no_regress) or (mae == best_mae and no_regress and strict_one)
        else:
            best = mae < best_mae

        # Global patience: iters since last new best. Independent of local.
        # Cool if EITHER patience trips. Reset both on cool (treat as one event).
        cool_local = no_improve_local >= STEP_PATIENCE_LOCAL
        cool_global = no_improve_global >= STEP_PATIENCE_GLOBAL
        if best:
            no_improve_global = 0
            cool_local = 0
        else:
            no_improve_global += 1


        if k >= WARMUP_ITERS and (cool_local or cool_global):
            step      *= STEP_COOL
            reason    = 'local' if cool_local else 'global'
            no_improve_local  = 0
            no_improve_global = 0
            print(f"  → cooling step to {step:.2e}  ({reason} patience tripped)")

        if best:
            pct_dev = (ratios - 1.0) * 100.0
            m0_mean = pct_dev[tri_modes == 0].mean()
            m1_mean = pct_dev[tri_modes == 1].mean()
            print(f"[Iter {k:4d}]  MAE: {mae:.8f} (from {best_mae:.8f}) | "
                  f"lr={step:.1e} | "
                  f"Min/Max: {curr_min:.8f}/{curr_max:.8f} | "
                  f"mode Δ: {m0_mean:+.4f}% / {m1_mean:+.4f}% | "
                  f"patience L:{STEP_PATIENCE_LOCAL-no_improve_local} "
                  f"G:{STEP_PATIENCE_GLOBAL-no_improve_global}")
            best_mae     = mae
            best_min     = curr_min
            best_max     = curr_max
            best_x_prime = x_prime.copy()
            out_path = OUTPUT_DIR / f"l05_{ellipsoid}_best.npz"
            np.savez(
                out_path,
                layer=LAYER,
                source_pts=l5_src,
                target_pts=best_x_prime,
                mae=best_mae,
                step_size=step,
                grad_iterations=ITERATIONS,
                adam_m=adam_m,
                adam_v=adam_v,
                adam_t=np.array(adam_t),
            )
            print(f"\nBest MAE: {best_mae:.8f}  Saved: {out_path}")
        else:
            # Non-best iters: terse line so patience countdown is visible.
            print(f"[Iter {k:4d}]  MAE: {mae:.8f} | lr={step:.1e} | "
                  f"Min/Max: {curr_min:.6f}/{curr_max:.6f} | "
                  f"patience L:{STEP_PATIENCE_LOCAL-no_improve_local} "
                  f"G:{STEP_PATIENCE_GLOBAL-no_improve_global}")

