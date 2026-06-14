import time
from pathlib import Path

import numpy as np
import ot
import warnings

from hhg9 import Points, Registrar
from hhg9.algorithms.distance import wgs84_area
from hhg9.h9 import H9K, H9O

# --- CONFIGURATION ---
LAYER = 4
REG = 0.00015  # Slightly lower for L4 precision
FEEDBACK_ITERATIONS = 50  # Give it time to converge
FEEDBACK_STRENGTH = 0.75  # Safe damping

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
    grid = repo['grid']
    locs = repo['locs'] if 'locs' in repo.files else None
    return cmp, xy_vert, v_ell, oc_vtx, oc_edg, grid, locs


# --- HELPERS (Analytic Geometry) ---
def get_ideal_corners(mode: int):
    tr, vf, vc = H9K.limits.TR, H9K.limits.VF, H9K.limits.VC
    if int(mode) == 0:
        return np.array([[-tr, vc], [tr, vc], [0.0, vf]])
    else:
        return np.array([[-tr, vf], [tr, vf], [0.0, vc]])


def create_ghost_padding(pts, weights, mode, limit=0.05):
    """Generates 'Halo' ghosts within 'limit' distance."""
    corners = get_ideal_corners(mode)
    lines = [(corners[0], corners[1]), (corners[1], corners[2]), (corners[2], corners[0])]
    super_pts, super_w, masks = [pts], [weights], [np.ones(len(pts), dtype=bool)]

    for p_start, p_end in lines:
        line_vec = p_end - p_start
        normal = np.array([-line_vec[1], line_vec[0]])
        normal /= np.linalg.norm(normal)
        dist_perp = np.dot(pts - p_start, normal)

        # HALO FILTER
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


if __name__ == '__main__':
    rg = Registrar()
    b_oct = rg.domain('b_oct')
    g_gcd = rg.domain('g_gcd')
    octant_id = 0
    mode = H9O.oid_mo[octant_id]
    layer = LAYER

    # LAYER = 4
    # REG = 0.00015  # Slightly lower for L4 precision
    # FEEDBACK_ITERATIONS = 15  # Give it time to converge
    # FEEDBACK_STRENGTH = 0.6  # Safe damping

    marker = 'g4_afl_4'

    # --- MAIN RUN ---
    print(f"--- STARTING L{LAYER} OPTIMIZATION (Halo + Feedback) ---")

    # 1. Load Data
    cmp, xy_vert, v_ell, oc_vtx, oc_edg, t_grid, locs = load_grid(layer=LAYER)
    pts = Points(xy_vert, b_oct, cmp)
    a_p, t_p = pts.coords, pts.coords.copy()

    # 2. Setup Weights
    num = len(pts)
    a_w = np.ones(num, dtype=np.float64) / num
    b_w_raw = np.exp(v_ell - np.median(v_ell))
    b_w_base = np.clip(b_w_raw, np.percentile(b_w_raw, 2), np.percentile(b_w_raw, 98))
    # ... (Apply your center/corner boosts here if needed) ...
    current_b_w = b_w_base / b_w_base.sum()

    # 3. Pre-Calculate Matrix (With Halo)
    bandwidth = 5.0 * np.sqrt(REG)  # Safe Halo size
    print(f"Generating Static Halo (Limit={bandwidth:.4f})...")
    ga, gaw_static, _ = create_ghost_padding(a_p, a_w, mode, limit=bandwidth)
    gt_coords, _, mask = create_ghost_padding(t_p, current_b_w, mode, limit=bandwidth)

    print(f"Matrix Size: {len(ga)} points. Pre-calculating distances...")
    # Float32 Cast for Memory Safety
    M_cache = ot.dist(ga.astype(np.float32), gt_coords.astype(np.float32), metric="sqeuclidean")
    M_cache /= M_cache.max()
    gaw_static = gaw_static.astype(np.float32)

    # --- FEEDBACK LOOP ---
    for i in range(FEEDBACK_ITERATIONS):
        print(f"\n[Iter {i + 1}] Solving...")

        # A. Generate Weights (Dynamic)
        _, full_b_w, _ = create_ghost_padding(t_p, current_b_w, mode, limit=bandwidth)

        # B. STABILITY FIX: Renormalize sums!
        # This prevents the "UNSTABLE" error caused by Halo slicing
        gaw_use = gaw_static / gaw_static.sum()
        gbw_use = full_b_w.astype(np.float32)
        gbw_use /= gbw_use.sum()

        # C. Solve
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", UserWarning)
            gamma_log = ot.sinkhorn(
                gaw_use, gbw_use, M_cache,
                reg=REG,
                # method='sinkhorn_log',
                numItermax=50000, stopThr=1e-7, verbose=False, warn=True
            )
            if len(w) > 0 or np.any(np.isnan(gamma_log)):
                print("   !!! UNSTABLE SOLVE. Stopping.")
                break

        # D. Reconstruct
        row_sum = gamma_log.sum(axis=1, keepdims=True)
        x_super = (gamma_log @ gt_coords) / np.maximum(row_sum, 1e-30)
        x_prime = x_super[mask]
        x_prime = snap_boundary_analytic(x_prime, oc_edg, oc_vtx, mode)

        # E. Measure Error
        dpts = Points(x_prime, b_oct, cmp)
        gpts = rg.project(dpts, [b_oct, g_gcd])
        t_coords_gcd = np.array([gpts.coords[v] for t in t_grid for v in t])
        areas = wgs84_area(rg, Points(t_coords_gcd, g_gcd), 3)
        c_ideal = np.sum(areas) / len(areas)
        ratios = areas / c_ideal
        mae = np.mean(np.abs(ratios - 1.0))

        print(f"   MAE: {mae:.6f} | Min/Max: {ratios.min():.3f} / {ratios.max():.3f}")

        # F. Update Weights
        v_corr = np.zeros_like(current_b_w)
        v_cnt = np.zeros_like(current_b_w)
        for t_idx, tri in enumerate(t_grid):
            v_corr[tri] += ratios[t_idx]
            v_cnt[tri] += 1

        mask_v = v_cnt > 0
        v_corr[mask_v] /= v_cnt[mask_v]
        v_corr[~mask_v] = 1.0

        current_b_w *= np.power(v_corr, FEEDBACK_STRENGTH)
        current_b_w /= current_b_w.sum()

        # Save checkpoint
        np.savez(f"output/L{LAYER}_iter{i}.npz", grid=x_prime, mae=mae)

    print("L4 Optimization Complete.")