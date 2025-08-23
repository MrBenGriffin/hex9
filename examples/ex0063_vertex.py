"""
Part of the H9 project - uses H9 addresses and round-trips them.
Set points at the vertices of the octahedron to verify fidelity.
Vertices are, like seams, special edge cases.
Testing and validated against Geodetic Conversions.
Last Tested 23 August 2025 -
# ⚠️ 'nm' == NANOMETRES, not NAUTICAL MILES. SUB-MILLIMETRE fidelity checks.
"""
import matplotlib.pyplot as plt
import numpy as np
from hhg9 import Registrar, H9Engine, Points
from hhg9.algorithms import wgs84
from hhg9.domains import GeneralGCD, OctahedralCartesian, OctahedralBarycentric, EllipsoidCartesian
from hhg9.projections import EllipsoidGCD, AKOctahedralEllipsoid, GCDBary
from hhg9.formats import OctahedralH9, DMS, DecimalDegrees, DecimalCartesian
from geographiclib.geodesic import Geodesic

WGS84 = Geodesic.WGS84  # or custom ellipsoid


# --- diagnostics config -----------------------------------------------------
# Set RAW_MODE=True to remove most training wheels (minimal filtering).
RAW_MODE = True
ALIAS_CAP_NM = None  # e.g., 10 for 1 m; None disables
ELIDE_VERTICES = 0 if RAW_MODE else 1
BIG = 6.0  # nm threshold from prior work

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def plot_bullring_kde(radii, ring_id, theta_pt, nm_pt, *,
                      grid_resolution=200j, bw_method=None,
                      title="KDE Heatmap", cmap='viridis'):
    """
    Plots a polar heatmap using a weighted Kernel Density Estimate (KDE)
    to avoid binning artifacts and ensure perfect registration.

    Args:
        grid_resolution (complex): The number of points for the evaluation grid.
                                   e.g., 200j means a 200x200 grid.
        bw_method (str or float): The bandwidth for the KDE. Controls smoothness.
                                  Defaults to 'scott'. Smaller values are spikier.
    """

    # --- 1. Data Preparation: Convert Polar to Cartesian ---
    # Get the radius for each individual data point
    point_radii = radii[ring_id]

    # Convert each point's (r, theta) to (x, y)
    x = point_radii * np.cos(theta_pt)
    y = point_radii * np.sin(theta_pt)

    # --- 2. KDE Calculation ---
    # Prepare the dataset for scipy's KDE: shape is (2, n_points)
    dataset = np.vstack([x, y])

    # Create the weighted KDE. Each point's contribution is weighted by its nm_pt value.
    print("Calculating weighted KDE...")
    kde = gaussian_kde(dataset, weights=nm_pt, bw_method=bw_method)
    print("KDE calculation complete.")

    # --- 3. Create Evaluation Grid ---
    # Get the min/max extents of the data to create a grid
    padding = np.max(radii) * 0.1  # Add a little padding
    xmin, xmax = x.min() - padding, x.max() + padding
    ymin, ymax = y.min() - padding, y.max() + padding

    # Create a regular Cartesian grid
    grid_x, grid_y = np.mgrid[xmin:xmax:grid_resolution, ymin:ymax:grid_resolution]

    # --- 4. Evaluate KDE on the Grid ---
    # The KDE is a function; call it on the grid points to get the Z values.
    print("Evaluating KDE on the grid...")
    grid_z = kde(np.vstack([grid_x.ravel(), grid_y.ravel()]))
    # Reshape the result into a 2D grid
    grid_z = grid_z.reshape(grid_x.shape)
    print("Evaluation complete.")

    # --- 5. Plotting ---
    fig = plt.figure(figsize=(12, 12), facecolor='black')
    ax = fig.add_subplot(111, projection='polar', facecolor='black')

    # Use contourf to plot the smooth KDE surface.
    # We provide the Cartesian grid; the polar projection handles the rest.
    pc = ax.contourf(grid_x, grid_y, grid_z, levels=150, cmap=cmap)

    # Add the scatter overlay to verify perfect registration
    ax.scatter(theta_pt, point_radii, s=15, c=nm_pt, cmap='hot', alpha=0.6, edgecolors='none')

    # Styling (same as before)
    cbar = fig.colorbar(pc, ax=ax, pad=0.1, shrink=0.9)
    cbar.set_label("KDE of ∂ values", color='white')
    cbar.ax.tick_params(labelcolor='white')

    ax.set_title(title, color='white')
    ax.tick_params(axis='both', labelcolor='white')
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(-1)
    ax.grid(color='white', linestyle='--', linewidth=0.5, alpha=0.7)

    # Set the radial limits to match your data's extent
    ax.set_rmax(np.max(radii))
    ax.set_rmin(np.min(radii) * 0.5)  # Or your desired inner limit
    ax.set_yscale('log')  # Re-apply log scale if needed

    plt.show()


def generate_vertex_bullrings(v, radii_m, n_per_ring=720, *, a=6378137.0, b=6356752.314245179):
    """
    Concentric ellipsoidal rings around direction `v` (ECEF), radii in meters.
    Returns:
      ring_xyz : (Σn, 3) ECEF points on the WGS-84 ellipsoid
      ring_id  : (Σn,) int ring index per point (0..len(radii)-1)
    """
    v = np.asarray(v, dtype=np.float64)
    v = v / np.linalg.norm(v)

    # Tangent frame (u, w) ⟂ v
    up = np.array([0.0, 0.0, 1.0]) if abs(v[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(up, v); u /= np.linalg.norm(u)
    w = np.cross(v, u)  # unit

    radii_m = np.asarray(radii_m, dtype=float)
    m = radii_m.size

    thetas = np.linspace(0, 2*np.pi, n_per_ring, endpoint=False)  # (n,)
    cos_t = np.cos(thetas)[None, :]  # (1,n)
    sin_t = np.sin(thetas)[None, :]  # (1,n)

    # Build unit direction perturbations for each ring
    # dir = v*cosα + (u cosθ + w sinθ)*sinα  where α ≈ r / R_local
    # Use spherical small-angle for direction; project to ellipsoid by ray/quadric intersection.
    # Ray: x(t) = 0 + t*dir  ; ellipsoid: x^2/a^2 + y^2/a^2 + z^2/b^2 = 1
    # Solve t from (dx^2+dy^2)/a^2 + dz^2/b^2 = 1  →  t = 1 / sqrt( (dx^2+dy^2)/a^2 + dz^2/b^2 )

    # Choose local radius ~ a (small α); good enough for unit dirs before scaling to ellipsoid
    alpha = (radii_m / a)[:, None]                      # (m,1)
    cos_a = np.cos(alpha)                               # (m,1)
    sin_a = np.sin(alpha)                               # (m,1)

    U = u[None, None, :] * cos_t[:, :, None]            # (1,n,3)
    W = w[None, None, :] * sin_t[:, :, None]            # (1,n,3)
    tang = (U + W)                                      # (1,n,3)

    base = (cos_a[:, None] * v[None, None, :])          # (m,1,3)
    dirs = base + sin_a[:, None] * tang                 # (m,n,3)

    dx, dy, dz = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    inv_a2 = 1.0 / (a*a)
    inv_b2 = 1.0 / (b*b)
    denom = (dx*dx + dy*dy) * inv_a2 + (dz*dz) * inv_b2
    t = 1.0 / np.sqrt(denom)                            # (m,n)

    xyz = dirs * t[..., None]                           # (m,n,3)
    ring_xyz = xyz.reshape(-1, 3)
    ring_id = np.repeat(np.arange(m, dtype=int), n_per_ring)
    theta_pt = np.tile(thetas, m)                       # (m*n,)
    return ring_xyz, ring_id, theta_pt


def polot_bullring_heatmap(radii, ring_id, theta_pt, nm_pt, *,
                          metric='median', n_theta=720,
                          title="heatmap", cmap='viridis'):
    """
    n_theta=1080
    metric  : 'median' or 'p95'
    """
    m = radii.size
    edges = np.empty(m + 1, dtype=float)
    if m == 1:
        edges[0] = radii[0] / np.sqrt(10.0)
        edges[1] = radii[0] * np.sqrt(10.0)
    else:
        edges[1:-1] = np.sqrt(radii[1:] * radii[:-1])
        edges[0]  = radii[0]**2 / edges[1]
        edges[-1] = radii[-1]**2 / edges[-2]

    # azimuth bins (1D array of n_theta + 1 edges)
    th_edges = np.linspace(0, 2*np.pi, n_theta+1, endpoint=True)

    # accumulator Z[m, n_theta]
    Z = np.full((m, n_theta), np.nan, dtype=float)
    p_theta = nm_pt.reshape(m, -1).shape[1]
    P = np.full((m, p_theta), np.nan, dtype=float)

    # fill per ring/az bin
    th_idx = np.digitize(theta_pt % (2*np.pi), th_edges) - 1
    th_idx[th_idx == n_theta] = n_theta - 1

    for r in range(m):
        sel_r = (ring_id == r)
        if not np.any(sel_r):
            continue
        rr_th = th_idx[sel_r]
        rr_nm = nm_pt[sel_r]
        # group by theta bin
        for k in range(n_theta):
            b_sel = (rr_th == k)
            if not np.any(b_sel):
                continue
            match metric:
                case 'p95':
                    Z[r, k] = np.percentile(rr_nm[b_sel], 95.0)
                case 'median':
                    Z[r, k] = np.median(rr_nm[b_sel])
                case 'mean':
                    Z[r, k] = np.mean(rr_nm[b_sel])
                case 'max':
                    Z[r, k] = np.max(rr_nm[b_sel])
                case _:
                    Z[r, k] = np.median(rr_nm[b_sel])
                    print('unknown metric:', metric, 'using median.')
        for k in range(p_theta):
            b_sel = (rr_th == k)
            if not np.any(b_sel):
                continue
            match metric:
                case 'p95':
                    P[r, k] = np.percentile(rr_nm[b_sel], 95.0)
                case 'median':
                    P[r, k] = np.median(rr_nm[b_sel])
                case 'mean':
                    P[r, k] = np.mean(rr_nm[b_sel])
                case 'max':
                    P[r, k] = np.max(rr_nm[b_sel])
                case _:
                    P[r, k] = np.median(rr_nm[b_sel])
                    print('unknown metric:', metric, 'using median.')

    # PLOTTING SECTION FOR GOURAUD SHADING
    # 1. Calculate the coordinates of the CENTER of each bin
    r_centers = (edges[:-1] + edges[1:]) / 2.0
    th_centers = (th_edges[:-1] + th_edges[1:]) / 2.0

    # 2. Create 2D grids from the center coordinates to match Z's shape
    #    Note the order: r_centers first, to match Z's (m, n_theta) shape
    R_cen, TH_cen = np.meshgrid(r_centers, th_centers, indexing='ij')

    # plot
    fig = plt.figure(figsize=(18, 16), dpi=300, frameon=False)
    ax  = fig.add_subplot(111, projection='polar')
    ax.set_yscale('log')
    Z = np.nan_to_num(Z)

    # --- NEW: Pad the data for seamless triangulation ---
    # Number of columns to use for padding
    pad_cols = 2

    # Pad right side (add first columns to the end, shifted by +2*pi)
    TH_pad_right = TH_cen[:, 0:pad_cols] + 2 * np.pi
    R_pad_right = R_cen[:, 0:pad_cols]
    Z_pad_right = Z[:, 0:pad_cols]

    # Pad left side (add last columns to the beginning, shifted by -2*pi)
    TH_pad_left = TH_cen[:, -pad_cols:] - 2 * np.pi
    R_pad_left = R_cen[:, -pad_cols:]
    Z_pad_left = Z[:, -pad_cols:]

    # Combine original data with padded data
    TH_full = np.hstack([TH_pad_left, TH_cen, TH_pad_right])
    R_full = np.hstack([R_pad_left, R_cen, R_pad_right])
    Z_full = np.hstack([Z_pad_left, Z, Z_pad_right])
    # --- END PADDING SECTION ---

    pc = ax.tricontourf(
        TH_full.flatten(),
        R_full.flatten(),
        Z_full.flatten(),
        levels=150,
        cmap=cmap, vmax = 6.0
    )

    point_radii = radii[ring_id]
    ax.scatter(theta_pt, point_radii, c=P, ec='none', s=5, cmap=cmap, vmax=6.0, alpha=1.0)

    cbar = fig.colorbar(pc, ax=ax, pad=0.05, shrink=0.99)
    cbar.set_label(f"{metric} (nm)", fontsize=20)
    ax.set_title(f'{title} – {metric}', fontsize=20)
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(-1)
    ax.tick_params(axis='both', labelcolor='white')  # Set grid label color
    ax.grid(color='green', linestyle='-', linewidth=0.5, alpha=0.5)
    plt.show()

def run():
    """
        Convert
    """
    reg = Registrar()  # Manage Domains & Projections
    g_gcd = GeneralGCD(reg)  # GCD Spherical Domain (latitude/longitude)
    c_ell = EllipsoidCartesian(reg)  # Cartesian Ellipsoid (xyz)
    c_oct = OctahedralCartesian(reg)  # Cartesian Octahedron (xyz)
    b_oct = OctahedralBarycentric(reg, c_oct)  # 2d Flat for addressing.

    h9f = OctahedralH9()  # formatter.
    h9e = H9Engine()
    g_gcd.register_format(DMS())
    g_gcd.register_format(DecimalDegrees())
    c_ell.register_format(DecimalCartesian())
    b_oct.register_format(h9f)

    # Projections/Transforms. Bary and Net are loaded by the domains.
    EllipsoidGCD(reg)  # g_sph <=> c_sph
    ak = AKOctahedralEllipsoid(reg)  # c_sph <=> (c_oct <=> b_oct)
    ak.set_accuracy(1e-9)
    GCDBary(reg)

    a = c_ell.a
    b = c_ell.b
    vertices = {
        "Null Island": np.array([a, 0.0, 0.0], dtype=float),
        "Null Antipodes": np.array([-a, 0.0, 0.0], dtype=float),
        "East Equator": np.array([0.0, a, 0.0], dtype=float),
        "West Equator": np.array([0.0, -a, 0.0], dtype=float),
        "North Pole": np.array([0.0, 0.0, b], dtype=float),
        "South Pole": np.array([0.0, 0.0, -b], dtype=float),
    }

    radii = np.array([1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e+1, 1e+2])
    for name, val in vertices.items():
        ring_xyz, ring_id, theta_xy = generate_vertex_bullrings(val, radii, n_per_ring=360, a=a, b=b)
        pts = Points(ring_xyz, c_ell)
        pll = reg.project(pts, [c_ell, c_oct])
        prt = reg.project(pll, [c_oct, c_ell])
        d_m = np.linalg.norm(prt.coords - ring_xyz, axis=1)
        nm_pt = d_m * 1e9  # nm
        metric = 'p95'    # 'p95'
        cmap = 'plasma'
        plot_bullring_heatmap(radii, ring_id, theta_xy, nm_pt, title=f'{name} c_ell->c_oct->c_ell ∂', cmap=cmap, metric=metric)


if __name__ == '__main__':
    run()
