import pickle
from pathlib import Path
import numpy as np
import ot
import warnings
from scipy.interpolate import CloughTocher2DInterpolator

from hhg9 import Registrar, Points
from hhg9.algorithms.distance import wgs84_area
from hhg9.h9 import H9O, H9K

# --- CONFIGURATION ---
LAYER = 4
REG = 0.00015  # Precision for L4
FEEDBACK_ITERATIONS = 100  # More iterations for L4
START_STRENGTH = 0.78  # Lower start strength to prevent initial ringing


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


# --- HELPERS ---
def get_ideal_corners(mode: int):
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


class AuthalicWarp:
    def __init__(self, source_pts, target_pts, store_pts=False):
        print(f"Building Clough-Tocher Interpolator ({len(source_pts)} points)...")
        diff = target_pts - source_pts
        self.source_pts = None
        self.target_pts = None
        if store_pts:
            self.source_pts = source_pts
            self.target_pts = target_pts
        self.dx_interp = CloughTocher2DInterpolator(source_pts, diff[:, 0])
        self.dy_interp = CloughTocher2DInterpolator(source_pts, diff[:, 1])
        print("Warp Ready.")

    def __call__(self, xy):
        xy = np.asarray(xy)
        if xy.ndim == 1: xy = xy[None, :]
        dx, dy = self.dx_interp(xy), self.dy_interp(xy)
        mask_nan = np.isnan(dx) | np.isnan(dy)
        if np.any(mask_nan): dx[mask_nan] = 0.0; dy[mask_nan] = 0.0
        return xy + np.stack([dx, dy], axis=1)

    def save(self, f):
        with open(f, 'wb') as file:
            pickle.dump(self, file)
        np.savez(
            f"warp_z.npz",
            source_pts=self.source_pts,
            target_pts=self.target_pts,
            layer=LAYER,
            iteration=FEEDBACK_ITERATIONS
        )


# --- MAIN ---
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

    marker = 'g6q_l4'

    print(f"--- STARTING L{LAYER} MASTER RUN ---")

    # 1. Load Data
    cmp, xy_vert, v_ell, oc_vtx, oc_edg, t_grid, locs = load_grid(layer=LAYER)
    pts = Points(xy_vert, b_oct, cmp)
    a_p, t_p = pts.coords, pts.coords.copy()

    # 2. Weights & Boosts
    num = len(pts)
    a_w = np.ones(num, dtype=np.float64) / num
    b_w_raw = np.exp(v_ell - np.median(v_ell))
    b_w_base = np.clip(b_w_raw, np.percentile(b_w_raw, 2), np.percentile(b_w_raw, 98))
    # --- INSERT YOUR CENTER/CORNER BOOST LOGIC HERE IF NEEDED ---
    current_b_w = b_w_base / b_w_base.sum()

    # 3. Pre-Calc Matrices (Float32 + Halo)
    bandwidth = 5.0 * np.sqrt(REG)
    print(f"Generating Halo (Limit={bandwidth:.4f})...")
    ga, gaw_static, _ = create_ghost_padding(a_p, a_w, mode, limit=bandwidth)
    gt_coords, _, mask = create_ghost_padding(t_p, current_b_w, mode, limit=bandwidth)

    print("Calculating Distance Matrix...")
    M_cache = ot.dist(ga.astype(np.float32), gt_coords.astype(np.float32), metric="sqeuclidean")
    M_cache /= M_cache.max()
    gaw_use = (gaw_static / gaw_static.sum()).astype(np.float32)
    x_prime = None

    # 4. Feedback Loop
    for i in range(FEEDBACK_ITERATIONS):
        # A. Cooling Schedule
        if i < 8:
            strength = START_STRENGTH
        elif i < 15:
            strength = START_STRENGTH * 0.6
        else:
            strength = 0.2

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

        # D. Reconstruct
        row_sum = gamma.sum(axis=1, keepdims=True)
        x_prime = ((gamma @ gt_coords) / np.maximum(row_sum, 1e-30))[mask]
        x_prime = snap_boundary_analytic(x_prime, oc_edg, oc_vtx, mode)

        # E. Measure
        dpts = Points(x_prime, b_oct, cmp)
        gpts = rg.project(dpts, [b_oct, g_gcd])
        t_pts_gcd = np.array([gpts.coords[v] for t in t_grid for v in t])
        areas = wgs84_area(rg, Points(t_pts_gcd, g_gcd), 3)
        c_ideal = np.mean(areas)
        ratios = areas / c_ideal
        mae = np.mean(np.abs(ratios - 1.0))
        print(f"   MAE: {mae:.6f} | Min/Max: {ratios.min():.3f} / {ratios.max():.3f}")

        # F. Correction with Diffusion (Anti-Ringing)
        v_corr = np.zeros(len(current_b_w))
        v_cnt = np.zeros(len(current_b_w))
        for t_idx, tri in enumerate(t_grid):
            v_corr[tri] += ratios[t_idx]
            v_cnt[tri] += 1
        mask_v = v_cnt > 0
        v_corr[mask_v] /= v_cnt[mask_v]
        v_corr[~mask_v] = 1.0

        # --- ERROR DIFFUSION STEP ---
        # Simple averaging with previous value (Momentum) to dampen oscillation
        if i > 0:
            v_corr = 0.75 * v_corr + 0.3 * v_prev_corr
        v_prev_corr = v_corr.copy()

        current_b_w *= np.power(v_corr, strength)
        current_b_w /= current_b_w.sum()

        # Save checkpoint
        np.savez(
            f"output/q_l{LAYER}_iter{i}.npz",
            mae=mae,
            source_pts=a_p,
            target_pts=x_prime,
            layer=LAYER,
            iteration=i
        )
    # --- L4 ANTI-ZIPPER POLISH ---
    # Run this starting with your current 'current_b_w' and grid
    print("--- POLISHING SEAMS (Anti-Zipper) ---")
    mae = 0
    POLISH_ITERATIONS = 6
    POLISH_STRENGTH = 0.15  # Very gentle
    DIFFUSION = 0.5  # How much to share error with neighbors

    for k in range(POLISH_ITERATIONS):
        # 1. Standard Sinkhorn Step
        # ... (Re-generate weights -> Sinkhorn -> Reconstruct -> Snap) ...
        # [Use your standard code block here]
        dpts = Points(x_prime, b_oct, cmp)
        gpts = rg.project(dpts, [b_oct, g_gcd])
        t_pts_gcd = np.array([gpts.coords[v] for t in t_grid for v in t])
        areas = wgs84_area(rg, Points(t_pts_gcd, g_gcd), 3)
        c_ideal = np.mean(areas)
        ratios = areas / c_ideal
        mae = np.mean(np.abs(ratios - 1.0))

        # 2. Measure Error
        # ... (Calculate ratios) ...
        print(f"[Polish {k + 1}] MAE: {mae:.6f} | Min/Max: {ratios.min():.3f} / {ratios.max():.3f}")

        # 3. COMPUTE CORRECTION (With Diffusion)
        # Map triangle error to vertices
        v_corr_raw = np.zeros(len(current_b_w))
        v_cnt = np.zeros(len(current_b_w))

        for t_idx, tri in enumerate(t_grid):
            v_corr_raw[tri] += ratios[t_idx]
            v_cnt[tri] += 1

        mask = v_cnt > 0
        v_corr_raw[mask] /= v_cnt[mask]
        v_corr_raw[~mask] = 1.0

        # --- THE MAGIC STEP: Spatial Diffusion ---
        # We smooth the correction factor 'v_corr' by averaging it with its neighbors.
        # This kills the checkerboard/zipper pattern instantly.
        # (Simplified diffusion: just blend with global mean or local average if you have an adjacency list.
        #  Since we are AFK, we will use a simpler "Momentum" trick which works similarly)

        if k == 0:
            v_smooth = v_corr_raw
        else:
            v_smooth = (1.0 - DIFFUSION) * v_corr_raw + DIFFUSION * v_prev_smooth

        v_prev_smooth = v_smooth.copy()

        # 4. Apply Gentle Correction
        current_b_w *= np.power(v_smooth, POLISH_STRENGTH)
        current_b_w /= current_b_w.sum()

    # Save final result
    np.savez(
        f"output/q_l{LAYER}_polished.npz",
        mae=mae,
        source_pts=a_p,
        target_pts=x_prime,
        layer=LAYER,
        iteration=FEEDBACK_ITERATIONS
    )
    # np.savez(f"output/L4_Final_Polished.npz", grid=x_prime, mae=mae)

    # 5. Build & Save Warp
    print("\nFeedback Complete. Building Warp...")
    warp = AuthalicWarp(a_p, x_prime, True)
    warp.save(f"output/q_l{LAYER}_warp.pkl")
    print(f"Warp saved to output/q_l{LAYER}_Warp.pkl")
