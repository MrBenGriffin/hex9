"""
Seam-strip polish — loads any checkpoint from sinkhorn_01_l4.py and applies
targeted boundary correction without re-running the full Sinkhorn feedback loop.

Usage:
    Edit CHECKPOINT below to point at any output/q_lN_iterM.npz file, then run.

Why separate from the main loop:
    The main Sinkhorn feedback loop converges the interior well but leaves a
    systematic distortion band 1-2 rings inside each octant edge:
        on-seam row:      slightly oversized  (~+8% at L5)
        next 1-2 rows in: undersized          (~-15% at L5)

    This is a boundary artefact from snap_boundary_analytic pinning edge vertices.
    The transport cannot move mass through the pinned wall, so it compensates by
    pulling from the immediately adjacent interior ring.

    The polish re-runs Sinkhorn with weight corrections restricted to the seam strip
    only, leaving the well-converged interior untouched. Separating it means polish
    parameters (SEAM_RINGS, POLISH_STRENGTH, DIFFUSION, POLISH_ITERATIONS) can be
    tuned quickly without re-running the expensive main loop.

Octant symmetry note:
    A cross-octant joint solve is not required. The octahedral faces are arranged
    so that ellipsoidal (WGS84) distortion is identical under every face and the
    seam is a perfect mirror axis. Each octant solves to the same boundary condition
    independently; fixing the strip on one face automatically fixes the adjacent face.
"""

from pathlib import Path
import numpy as np
import ot
import warnings

from hhg9 import Registrar, Points
from hhg9.algorithms.distance import wgs84_area
from hhg9.h9 import H9O

# --- CONFIGURATION ---
CHECKPOINT = Path("output/q_l4_iter50.npz")   # checkpoint to resume from
OCTANT_ID = 0
GRID_FILE = None    # override grid file path; None = auto-derive from checkpoint layer

POLISH_ITERATIONS = 3    # number of seam-polish Sinkhorn iterations
POLISH_STRENGTH = 0.15   # weight correction strength; reduce if interior degrades
DIFFUSION = 0.5          # momentum blend to suppress oscillation between iterations
SEAM_RINGS = 5           # triangle-rings outward from oc_edg included in strip
# #   Tuning guide:
# #     seam min/max barely moves  → increase SEAM_RINGS (strip too narrow)
# #     all min/max degrades       → reduce POLISH_STRENGTH
#     oscillation in seam ratios → increase DIFFUSION toward 0.7


# --- shared helpers (duplicated from sinkhorn_01_l4.py to keep this script standalone) ---

def get_ideal_corners(mode: int):
    from hhg9.h9 import H9K
    tr, vf, vc = H9K.limits.TR, H9K.limits.VF, H9K.limits.VC
    if int(mode) == 0:
        return np.array([[-tr, vc], [tr, vc], [0.0, vf]])
    else:
        return np.array([[-tr, vf], [tr, vf], [0.0, vc]])


def create_ghost_padding(pts, weights, mode, limit=0.05):
    corners = get_ideal_corners(mode)
    lines = [(corners[0], corners[1]), (corners[1], corners[2]), (corners[2], corners[0])]
    super_pts, super_w, masks = [pts], [weights], [np.ones(len(pts), dtype=bool)]
    for p_start, p_end in lines:
        line_vec = p_end - p_start
        normal = np.array([-line_vec[1], line_vec[0]])
        normal /= np.linalg.norm(normal)
        dist_perp = np.dot(pts - p_start, normal)
        if limit is not None:
            mask_halo = np.abs(dist_perp) < limit
            if not np.any(mask_halo): continue
            pts_r, dist_r, w_r = pts[mask_halo], dist_perp[mask_halo], weights[mask_halo]
        else:
            pts_r, dist_r, w_r = pts, dist_perp, weights
        super_pts.append(pts_r - 2 * normal * dist_r[:, None])
        super_w.append(w_r)
        masks.append(np.zeros(len(pts_r), dtype=bool))
    return np.vstack(super_pts), np.concatenate(super_w), np.concatenate(masks)


def snap_boundary_analytic(xy, oc_edg, oc_vtx, mode):
    out = xy.copy()
    corners = get_ideal_corners(mode)
    for idx in oc_vtx: out[idx] = corners[np.argmin(np.sum((corners - out[idx]) ** 2, axis=1))]
    if len(oc_edg) > 0:
        pts, lines = out[oc_edg], [(corners[0], corners[1]), (corners[1], corners[2]), (corners[2], corners[0])]
        starts = np.array([l[0] for l in lines])
        vecs = np.array([l[1] - l[0] for l in lines])
        lens2 = np.sum(vecs ** 2, axis=1)
        P, A, V = pts[:, None, :], starts[None, :, :], vecs[None, :, :]
        t = np.clip(np.sum((P - A) * V, axis=2) / lens2, 0.0, 1.0)
        projs = A + t[:, :, None] * V
        out[oc_edg] = projs[np.arange(len(pts)), np.argmin(np.sum((P - projs) ** 2, axis=2), axis=1)]
    return out


def build_seam_strip(n_verts, oc_edg, t_grid, n_rings=4):
    # Dilates the set of octant-edge vertices outward by n_rings triangle-rings
    # using triangle adjacency, returning a boolean vertex mask.
    #
    # Used by the seam-strip polish phase to restrict weight corrections to the
    # boundary compensation band (the ~2-3 row region inward from oc_edg that
    # shows reduced area due to the transport wall effect). Interior vertices —
    # already well-converged by the main feedback loop — are left untouched.
    strip = set(int(v) for v in oc_edg)
    for _ in range(n_rings):
        new = set()
        for tri in t_grid:
            tri_set = set(int(v) for v in tri)
            if strip & tri_set:
                new |= tri_set
        strip |= new
    mask = np.zeros(n_verts, dtype=bool)
    for v in strip:
        mask[v] = True
    return mask


if __name__ == '__main__':
    # --- 1. Load checkpoint ---
    print(f"Loading checkpoint: {CHECKPOINT}")
    ckpt = np.load(CHECKPOINT, allow_pickle=True)
    source_pts = ckpt['source_pts']      # original b_raw vertex positions (a_p)
    x_prime = ckpt['target_pts']         # warped positions at end of main loop
    current_b_w = ckpt['target_weights'].copy()
    REG = float(ckpt['reg'])
    bandwidth = float(ckpt['bandwidth'])
    layer = int(ckpt['layer'])
    start_iter = int(ckpt['iteration'])
    print(f"  Layer {layer}, iteration {start_iter}, "
          f"checkpoint MAE {float(ckpt['mae']):.8f}, "
          f"REG={REG}, bandwidth={bandwidth:.4f}")

    # --- 2. Load grid ---
    grid_path = Path(GRID_FILE) if GRID_FILE else Path(f"grid_l{layer}.npz")
    print(f"Loading grid: {grid_path}")
    repo = np.load(grid_path, allow_pickle=True)
    oc_vtx = repo['oc_vtx']
    oc_edg = repo['oc_edg']
    t_grid = repo['grid']

    # --- 3. Reconstruct transport matrices ---
    rg = Registrar()
    b_raw = rg.domain('b_raw')
    g_gcd = rg.domain('g_gcd')
    mode = H9O.oid_mo[OCTANT_ID]

    a_p = source_pts
    t_p = source_pts.copy()   # target coords initialised from source (same as main loop)
    num = len(a_p)
    a_w = np.ones(num, dtype=np.float64) / num

    print(f"Rebuilding transport matrices (bandwidth={bandwidth:.4f})...")
    ga, gaw_static, _ = create_ghost_padding(a_p, a_w, mode, limit=bandwidth)
    gt_coords, _, ghost_mask = create_ghost_padding(t_p, current_b_w, mode, limit=bandwidth)
    M_cache = ot.dist(ga.astype(np.float32), gt_coords.astype(np.float32), metric="sqeuclidean")
    M_cache /= M_cache.max()
    gaw_use = (gaw_static / gaw_static.sum()).astype(np.float32)

    # --- 4. Build seam strip ---
    seam_mask = build_seam_strip(len(current_b_w), oc_edg, t_grid, n_rings=SEAM_RINGS)
    seam_tri_mask = np.array([any(seam_mask[v] for v in tri) for tri in t_grid])
    print(f"Seam strip: {seam_mask.sum()} / {len(seam_mask)} vertices, "
          f"{seam_tri_mask.sum()} / {len(t_grid)} triangles")

    # --- 5. Measure baseline (checkpoint state, before any polish) ---
    dpts = Points(x_prime, b_raw, oid=0)
    gpts = rg.project(dpts, [b_raw, g_gcd])
    t_pts_gcd = np.array([gpts.coords[v] for t in t_grid for v in t])
    areas = wgs84_area(rg, Points(t_pts_gcd, g_gcd), 3)
    c_ideal = np.mean(areas)
    ratios = areas / c_ideal
    seam_r = ratios[seam_tri_mask]
    print(f"[Baseline]  MAE: {np.mean(np.abs(ratios - 1.0)):.8f} | "
          f"All Min/Max: {ratios.min():.6f}/{ratios.max():.6f} | "
          f"Seam Min/Max: {seam_r.min():.6f}/{seam_r.max():.6f}")

    # --- 6. Seam-strip polish loop ---
    # Re-runs Sinkhorn each iteration with weight corrections restricted to the
    # seam strip. Interior weights stay frozen (v_apply=1 outside strip) so the
    # well-converged bulk is not disturbed.
    print(f"\n--- SEAM POLISH ({POLISH_ITERATIONS} iters, "
          f"strength={POLISH_STRENGTH}, diffusion={DIFFUSION}, rings={SEAM_RINGS}) ---")
    v_prev_smooth = np.ones(len(current_b_w))
    mae = 0.0

    for k in range(POLISH_ITERATIONS):
        # Sinkhorn step
        _, full_b_w, _ = create_ghost_padding(t_p, current_b_w, mode, limit=bandwidth)
        gbw_use = (full_b_w / full_b_w.sum()).astype(np.float32)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always", UserWarning)
            gamma = ot.sinkhorn(gaw_use, gbw_use, M_cache, reg=REG,
                                numItermax=50000, stopThr=1e-7, verbose=False, warn=True)
        row_sum = gamma.sum(axis=1, keepdims=True)
        x_prime = ((gamma @ gt_coords) / np.maximum(row_sum, 1e-30))[ghost_mask]
        x_prime = snap_boundary_analytic(x_prime, oc_edg, oc_vtx, mode)

        # Measure
        dpts = Points(x_prime, b_raw, oid=0)
        gpts = rg.project(dpts, [b_raw, g_gcd])
        t_pts_gcd = np.array([gpts.coords[v] for t in t_grid for v in t])
        areas = wgs84_area(rg, Points(t_pts_gcd, g_gcd), 3)
        c_ideal = np.mean(areas)
        ratios = areas / c_ideal
        mae = np.mean(np.abs(ratios - 1.0))
        seam_r = ratios[seam_tri_mask]
        print(f"[Polish {k + 1:2d}] MAE: {mae:.8f} | "
              f"All Min/Max: {ratios.min():.6f}/{ratios.max():.6f} | "
              f"Seam Min/Max: {seam_r.min():.6f}/{seam_r.max():.6f}")

        # Vertex corrections from triangle ratios
        v_corr_raw = np.zeros(len(current_b_w))
        v_cnt = np.zeros(len(current_b_w))
        for t_idx, tri in enumerate(t_grid):
            v_corr_raw[tri] += ratios[t_idx]
            v_cnt[tri] += 1
        active = v_cnt > 0
        v_corr_raw[active] /= v_cnt[active]
        v_corr_raw[~active] = 1.0

        # Momentum smoothing
        v_smooth = (1.0 - DIFFUSION) * v_corr_raw + DIFFUSION * v_prev_smooth
        v_prev_smooth = v_smooth.copy()

        # Apply only to seam strip
        v_apply = np.ones(len(current_b_w))
        v_apply[seam_mask] = v_smooth[seam_mask]
        current_b_w *= np.power(v_apply, POLISH_STRENGTH)
        current_b_w /= current_b_w.sum()

    # --- 7. Save polished result ---
    stem = CHECKPOINT.stem
    out_path = CHECKPOINT.parent / f"{stem}_polished_r{SEAM_RINGS}_s{POLISH_STRENGTH}.npz"
    np.savez(
        out_path,
        mae=mae,
        source_pts=a_p,
        target_pts=x_prime,
        target_weights=current_b_w,
        reg=REG,
        bandwidth=bandwidth,
        layer=layer,
        iteration=start_iter,
        polish_iterations=POLISH_ITERATIONS,
        polish_strength=POLISH_STRENGTH,
        seam_rings=SEAM_RINGS,
    )
    print(f"\nSaved: {out_path}")