import pickle
from pathlib import Path
import numpy as np
import ot
import warnings

from hhg9 import Registrar, Points
from hhg9.algorithms.distance import (wgs84_area)
from hhg9.h9 import H9O, H9K
# from hhg9.domains.octahedral_barycentric import AuthalicWarp

# --- CONFIGURATION ---
LAYER = 4          # Set to 3 for logic testing; change to 4 for production
REG = 0.00012      # Sinkhorn regularisation. L3 test value; tighten (e.g. 0.00008) for L4
FEEDBACK_ITERATIONS = 100
START_STRENGTH = 0.78  # Lower start strength to prevent initial ringing
# RESUME_FROM = 'output/q_l4t_iter50_tweaked.npz'  # Set to e.g. Path("output/q_l4_iter23.npz") to resume after a crash
RESUME_FROM = None

def load_grid(layer: int):
    """
    Cached loader for the grid NPZ for a given hex_layer.
    This avoids re-reading the same file from disk when `run` is called
    repeatedly for the same (hex_layer,e) pair.
    """
    f_name = Path(f"grid_l{layer}.npz")
    repo = np.load(f_name, allow_pickle=True)
    cmp = repo['cmp']
    xy_vert = repo['xy_vert']
    v_ell = repo['v_ell']
    oc_vtx = repo['oc_vtx']
    oc_edg = repo['oc_edg']
    grid = repo['grid']   # This is the triangle grid.
    locs = repo['locs'] if 'locs' in repo.files else None
    return cmp, xy_vert, v_ell, oc_vtx, oc_edg, grid, locs


# --- HELPERS ---
def get_ideal_corners(mode: int):
    tr, vf, vc = H9K.limits.TR, H9K.limits.VF, H9K.limits.VC
    if int(mode) == 0:
        return np.array([[-tr, vc], [tr, vc], [0.0, vf]])
    else:
        return np.array([[-tr, vf], [tr, vf], [0.0, vc]])


def create_ghost_padding(pts, weights, mode, limit=0.05):
    # Ghost padding mirrors a thin halo of points across each octant edge via
    # perpendicular reflection. This makes the Sinkhorn transport behave as if the
    # domain continues beyond the boundary, preventing mass from piling up against
    # the pinned edge vertices.
    #
    # Root cause of seam distortion: snap_boundary_analytic forces edge vertices
    # onto the triangle boundary, creating a hard wall. Without the ghost halo the
    # transport "pays" for that wall by under-filling the 1-2 ring of interior
    # vertices just inside the seam (producing the characteristic low-area band next
    # to an oversized on-seam row). Widening `limit` (bandwidth) deepens the halo
    # and reduces this compensation artefact.
    #
    # Note: a cross-octant joint solve is NOT required here. The octahedral faces are
    # arranged so that ellipsoidal (WGS84) distortion is identical under every face
    # and the seam is a perfect mirror axis. Each octant therefore solves to the same
    # boundary condition independently; fixing the distortion on one face
    # automatically fixes the corresponding strip on the adjacent face.
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
    # Projects octant-edge and corner vertices back onto the exact analytical
    # triangle boundary after each Sinkhorn reconstruction step.
    # This is the "hard wall" that causes the seam compensation artefact: edge
    # vertices cannot move, so the transport gradient steepens just inside them.
    # The ghost padding and seam-strip polish together address this artefact.
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


if __name__ == '__main__':
    rg = Registrar()
    b_raw = rg.domain('b_raw')
    g_gcd = rg.domain('g_gcd')
    octant_id = 0
    mode = H9O.oid_mo[octant_id]
    layer = LAYER

    # LAYER = 4
    # REG = 0.00015  # Slightly lower for L4 precision

    marker = 'sh__l4'

    print(f"--- STARTING L{LAYER} MASTER RUN ---")

    # 1. Load Data
    _, xy_vert, v_ell, oc_vtx, oc_edg, t_grid, locs = load_grid(layer=LAYER)

    pts = Points(xy_vert, b_raw, oid=0)
    a_p, t_p = pts.coords, pts.coords.copy()

    # 2. Weights & Boosts
    # v_ell encodes the ellipsoidal area distortion at each vertex relative to the
    # ideal equal-area projection. Exponentiating and normalising gives the target
    # weight distribution that the OT solve will try to match.
    num = len(pts)
    a_w = np.ones(num, dtype=np.float64) / num
    b_w_raw = np.exp(v_ell - np.median(v_ell))
    b_w_base = np.clip(b_w_raw, np.percentile(b_w_raw, 2), np.percentile(b_w_raw, 98))
    # --- INSERT  CENTER/CORNER BOOST LOGIC HERE IF NEEDED ---
    current_b_w = b_w_base / b_w_base.sum()

    # 3. Pre-Calc Matrices (Float32 + Halo)
    # bandwidth controls how deep the ghost halo reaches across each octant edge.
    # Too narrow (e.g. 5*sqrt(REG) ≈ 0.061 at L4) and the halo doesn't reach the
    # 2nd interior ring, leaving the compensation artefact visible. Widening to
    # 10*sqrt(REG) is cheap at L3 and is worth trying to see how much of the seam
    # distortion it eliminates before polish even runs.
    bandwidth = 10.0 * np.sqrt(REG)
    print(f"Generating Halo (Limit={bandwidth:.4f})...")
    ga, gaw_static, _ = create_ghost_padding(a_p, a_w, mode, limit=bandwidth)
    # ghost_mask selects the real (non-ghost) rows back out of the padded arrays
    gt_coords, _, ghost_mask = create_ghost_padding(t_p, current_b_w, mode, limit=bandwidth)

    print("Calculating Distance Matrix...")
    # M_cache is the squared-Euclidean cost matrix between ghost-padded source and
    # target point sets. Computed once and reused every iteration; normalised to
    # [0,1] to keep REG numerically stable across layers.
    M_cache = ot.dist(ga.astype(np.float32), gt_coords.astype(np.float32), metric="sqeuclidean")
    M_cache /= M_cache.max()
    gaw_use = (gaw_static / gaw_static.sum()).astype(np.float32)
    x_prime = None
    strength = START_STRENGTH
    mae_prev, mae = 50.0, 10.0
    v_prev_corr = np.ones(num)
    start_iter = 0

    # Resume from checkpoint if requested — restores loop state so the cooling
    # schedule and momentum continue smoothly from where the run was interrupted.
    # M_cache is always rebuilt from scratch (not saved) since it depends only on
    # the fixed source positions and bandwidth, both of which are in the checkpoint.
    if RESUME_FROM is not None:
        print(f"Resuming from {RESUME_FROM} ...")
        ckpt = np.load(Path(RESUME_FROM), allow_pickle=True)
        x_prime     = ckpt['target_pts']
        current_b_w = ckpt['target_weights'].copy()
        # Fields may be absent if the checkpoint came from sinkhorn_02_gradient
        # (older gradient outputs) — fall back to clean defaults in that case.
        v_prev_corr = ckpt['v_prev_corr'] if 'v_prev_corr' in ckpt.files else np.ones(num)
        strength    = float(ckpt['strength'])  if 'strength'  in ckpt.files else START_STRENGTH
        mae_prev    = float(ckpt['mae_prev'])  if 'mae_prev'  in ckpt.files else 50.0
        mae         = float(ckpt['mae'])
        start_iter  = int(ckpt['iteration']) + 1
        print(f"  Resuming at iter {start_iter}, strength={strength:.3f}, MAE={mae:.8f}")

    # 4. Feedback Loop
    # Each iteration: solve OT → reconstruct warped positions → measure area ratios
    # → update target weights to push over-large cells smaller and vice versa.
    # Strength is annealed down so early iterations do coarse shaping and later
    # iterations fine-tune without oscillating.
    for i in range(start_iter, FEEDBACK_ITERATIONS):
        # A. Cooling Schedule
        # Replace fixed schedule with MAE-gated cooldown
        if mae_prev - mae < 0.0001:
            strength = max(strength * 0.85, 0.10)
        mae_prev = mae
        print(f"\n[Iter {i + 1}] Solving (Strength={strength:.2f})...")

        # B. Generate Target Weights (Dynamic)
        _, full_b_w, _ = create_ghost_padding(t_p, current_b_w, mode, limit=bandwidth)
        gbw_use = (full_b_w / full_b_w.sum()).astype(np.float32)

        # C. Sinkhorn
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always", UserWarning)
            gamma = ot.sinkhorn(gaw_use, gbw_use, M_cache, reg=REG,
                                # , method='sinkhorn_log',
                                numItermax=50000, stopThr=1e-7, verbose=False, warn=True)

        # D. Reconstruct: barycentre of transport plan, strip ghost rows, snap boundary
        row_sum = gamma.sum(axis=1, keepdims=True)
        x_prime = ((gamma @ gt_coords) / np.maximum(row_sum, 1e-30))[ghost_mask]
        x_prime = snap_boundary_analytic(x_prime, oc_edg, oc_vtx, mode)

        # E. Measure
        dpts = Points(x_prime, b_raw, oid=0)
        gpts = rg.project(dpts, [b_raw, g_gcd])
        t_pts_gcd = t_grid[gpts.coords]
        # t_pts_gcd = np.array([gpts.coords[v] for t in t_grid for v in t])
        areas = wgs84_area(rg, Points(t_pts_gcd, g_gcd), 3)
        c_ideal = np.mean(areas)
        ratios = areas / c_ideal
        mae = np.mean(np.abs(ratios - 1.0))
        print(f"   MAE: {mae:.8f} | Min/Max: {ratios.min():.6f} / {ratios.max():.6f}")

        # F. Correction with Diffusion (Anti-Ringing)
        # Map per-triangle area ratios back to vertices (each vertex receives the
        # mean ratio of its incident triangles), then blend with the previous
        # iteration's correction via momentum to suppress checkerboard oscillation.
        v_corr = np.zeros(len(current_b_w))
        v_cnt = np.zeros(len(current_b_w))
        for t_idx, tri in enumerate(t_grid):
            v_corr[tri] += ratios[t_idx]
            v_cnt[tri] += 1
        mask_v = v_cnt > 0
        v_corr[mask_v] /= v_cnt[mask_v]
        v_corr[~mask_v] = 1.0

        if i > 0:
            v_corr = 0.75 * v_corr + 0.3 * v_prev_corr
        v_prev_corr = v_corr.copy()

        current_b_w *= np.power(v_corr, strength)
        current_b_w /= current_b_w.sum()

        # Save checkpoint — full loop state so either the polish script or a
        # resume can pick up exactly where this left off.
        np.savez(
            f"output/q_l{LAYER}_iter{i}.npz",
            mae=mae,
            source_pts=a_p,
            target_pts=x_prime,
            target_weights=current_b_w,
            v_prev_corr=v_prev_corr,
            strength=strength,
            mae_prev=mae_prev,
            reg=REG,
            bandwidth=bandwidth,
            layer=LAYER,
            iteration=i
        )
