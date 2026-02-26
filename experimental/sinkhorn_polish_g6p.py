import numpy as np
from pathlib import Path
from collections import defaultdict
from hhg9 import Points, Registrar
from hhg9.algorithms.distance import wgs84_area
from hhg9.h9 import H9K, H9O

# --- CONFIGURATION ---
LAYER = 4
SMOOTHING_STRENGTH = 0.5  # Stronger for the iron (0.5 = halfway to average)
ITERATIONS = 120  # More passes to really kill the zipper


# --- HELPERS ---
def load_grid_structure(layer: int):
    """Loads the connectivity (triangles) and original geometry."""
    f_name = Path(f"grid_l{layer}.npz")
    repo = np.load(f_name, allow_pickle=True)
    # Return: Component map, Verts, weights, boundary indices, boundary edges, TRIANGLES
    return (repo['cmp'], repo['xy_vert'], repo['v_ell'],
            repo['oc_vtx'], repo['oc_edg'], repo['grid'])


def get_ideal_corners(mode: int):
    tr, vf, vc = H9K.limits.TR, H9K.limits.VF, H9K.limits.VC
    return np.array([[-tr, vc], [tr, vc], [0.0, vf]]) if int(mode) == 0 else \
        np.array([[-tr, vf], [tr, vf], [0.0, vc]])


def snap_boundary_analytic(xy, oc_edg, oc_vtx, mode):
    """Forces points back onto the ideal mathematical lines."""
    out = xy.copy()
    corners = get_ideal_corners(mode)

    # 1. Snap Vertices (Corner Points)
    for idx in oc_vtx:
        # Find closest corner
        dists = np.sum((corners - out[idx]) ** 2, axis=1)
        out[idx] = corners[np.argmin(dists)]

    # 2. Snap Edges (Lines)
    if len(oc_edg) > 0:
        pts = out[oc_edg]
        # Define the 3 lines of the triangle
        lines = [(corners[0], corners[1]), (corners[1], corners[2]), (corners[2], corners[0])]
        starts = np.array([l[0] for l in lines])
        vecs = np.array([l[1] - l[0] for l in lines])
        lens2 = np.sum(vecs ** 2, axis=1)

        # Vectorized projection: Project every point onto every line
        P = pts[:, None, :]  # (N, 1, 2)
        A = starts[None, :, :]  # (1, 3, 2)
        V = vecs[None, :, :]  # (1, 3, 2)

        # t = dot(P-A, V) / dot(V,V)
        t = np.clip(np.sum((P - A) * V, axis=2) / lens2, 0.0, 1.0)
        projs = A + t[:, :, None] * V

        # Find distance to each line
        dists = np.sum((P - projs) ** 2, axis=2)  # (N, 3)
        best_line = np.argmin(dists, axis=1)  # (N,)

        # Assign best projection
        out[oc_edg] = projs[np.arange(len(pts)), best_line]

    return out


# --- MAIN ---
if __name__ == '__main__':
    print(f"--- GEOMETRIC IRON (L{LAYER}) ---")
    rg = Registrar()
    b_oct, g_gcd = rg.domain('b_oct'), rg.domain('g_gcd')
    mode = H9O.oid_mo[0]

    # 1. Load Structure (Connectivity)
    print("Loading Grid Structure...")
    cmp, xy_vert, v_ell, oc_vtx, oc_edg, grid_orig = load_grid_structure(layer=LAYER)

    # 'grid_orig' is the triangulation indices [59049, 3]
    t_grid = grid_orig

    # 2. Load Deformed State (Coordinates)
    # Replace with your actual L4 output file
    DATA_FILE = "output/q_l4_iter29.npz"
    print(f"Loading Deformed State: {DATA_FILE}")
    data = np.load(DATA_FILE)
    x_curr = data['target_pts'].copy()  # These are the [N_verts, 2] coordinates
    x_orig = data['source_pts'].copy()  # These are the [N_verts, 2] coordinates

    # 3. Build Adjacency Graph (Who is next to whom?)
    print("Building Adjacency Map...")
    neighbors = defaultdict(set)  # Use set to avoid duplicates

    # Iterate over the triangle indices to build the graph
    # (Fast enough for 60k triangles)
    for t in t_grid:
        v0, v1, v2 = t[0], t[1], t[2]
        neighbors[v0].add(v1)
        neighbors[v0].add(v2)
        neighbors[v1].add(v0)
        neighbors[v1].add(v2)
        neighbors[v2].add(v0)
        neighbors[v2].add(v1)

    # Convert to list for faster indexing later
    adj_list = {k: list(v) for k, v in neighbors.items()}

    # 4. Measure Start State
    dpts = Points(x_curr, b_oct, cmp)
    gpts = rg.project(dpts, [b_oct, g_gcd])
    # Reconstruct triangle coordinates for area calc
    t_pts_gcd = np.array([gpts.coords[v] for t in t_grid for v in t])
    areas = wgs84_area(rg, Points(t_pts_gcd, g_gcd), 3)
    mae_start = np.mean(np.abs(areas / np.mean(areas) - 1.0))
    print(f"Start MAE: {mae_start:.8f}")

    # 5. The Ironing Loop
    print(f"Ironing Seams (Strength={SMOOTHING_STRENGTH})...")

    # We only touch the boundary vertices (oc_edg)
    target_indices = np.array(oc_edg)

    for k in range(ITERATIONS):
        x_next = x_curr.copy()
        shifts = []

        # For every point on the boundary...
        for idx in target_indices:
            nbs = adj_list[idx]
            if not nbs: continue

            # Calculate average position of neighbors
            # Note: Neighbors include internal points, which anchors the smoothing
            center = np.mean(x_curr[nbs], axis=0)

            # Move current point towards center
            new_pos = (1.0 - SMOOTHING_STRENGTH) * x_curr[idx] + SMOOTHING_STRENGTH * center
            x_next[idx] = new_pos

        # CRITICAL: Snap back to the analytic line
        # This turns the "Average" into a "1D Average along the edge"
        x_next = snap_boundary_analytic(x_next, oc_edg, oc_vtx, mode)

        # Stats
        max_shift = np.max(np.linalg.norm(x_next - x_curr, axis=1))
        x_curr = x_next
        print(f"   Pass {k + 1}: Max Shift = {max_shift:.9f}")

    # 6. Measure End State
    dpts = Points(x_curr, b_oct, cmp)
    gpts = rg.project(dpts, [b_oct, g_gcd])
    t_pts_gcd = np.array([gpts.coords[v] for t in t_grid for v in t])
    areas = wgs84_area(rg, Points(t_pts_gcd, g_gcd), 3)
    mae_end = np.mean(np.abs(areas / np.mean(areas) - 1.0))

    print(f"End MAE: {mae_end:.8f} (Delta: {mae_end - mae_start:.8f})")

    # 7. Save
    np.savez(
        f"output/l{LAYER}_polished.npz",
        mae=mae_end,
        source_pts=x_orig,
        target_pts=x_curr,
        layer=LAYER,
    )
    # np.savez("output/data/L4_Final_Ironed.npz", grid=x_curr, mae=mae_end)
    print("Done.")
