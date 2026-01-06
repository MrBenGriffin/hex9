"""
Part of the H9 project - Preparation 0005
Load the GCD quadrilateral, project onto Barycentric Coordinates.
Calculate an optimal theta to the boundary.
Display it, and store it as a single value, along with
the octahedral rectangle coordinates.
Last Tested 14 October 2025 √
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pathlib import Path
from bounds_utils import load_presets, resolve_bounds, needs_run
from hhg9 import Registrar, Points


def north_arrow(ax, centroid, poly, color, scale=0.12, label="layer"):
    """Draw a north-pointing arrow above the given polygon’s centroid.
    The arrow length scales with the polygon’s extent so it remains visible for tiny patches.
    """
    # centroid can be (1,2) or (2,); normalise to scalars
    cx = float(np.atleast_1d(centroid)[0]) if np.ndim(centroid) == 1 else float(centroid[0, 0])
    cy = float(np.atleast_1d(centroid)[1]) if np.ndim(centroid) == 1 else float(centroid[0, 1])

    poly = np.asarray(poly, float)
    extent = np.max(poly, axis=0) - np.min(poly, axis=0)
    length = float(scale * np.max(extent)) if np.all(np.isfinite(extent)) else 0.1
    if not np.isfinite(length) or length == 0.0:
        length = 0.1

    start = (cx, cy + 0.5 * length)
    end   = (cx, cy + 1.5 * length)

    ax.annotate(label,
                xy=start, xytext=end,
                arrowprops=dict(arrowstyle="-|>", lw=1.2, facecolor=color, edgecolor=color),
                ha='center', va='bottom', fontsize=10, color=color,
                zorder=10, clip_on=False)
# --- Rotation estimators -----------------------------------------------------

def axis_mean_rotation(corners: np.ndarray) -> float:
    """
    Robust axis angle (mod π) via circular mean of 2θ, length-weighted.
    Treats opposite directions as identical.
    Returns angle in radians.
    """
    v = np.roll(corners, -1, axis=0) - corners
    w = np.linalg.norm(v, axis=1) + 1e-15
    th = np.arctan2(v[:, 1], v[:, 0])
    C = np.sum(w * np.cos(2.0 * th))
    S = np.sum(w * np.sin(2.0 * th))
    return 0.5 * np.arctan2(S, C)


def pca_rotation(corners: np.ndarray) -> float:
    """
    Angle of the first principal component of the four corners.
    Ambiguous up to π (fine for axis alignment).
    """
    X = corners - corners.mean(axis=0, keepdims=True)
    # SVD of centered coords; first right-singular vector is dominant axis
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    v = Vt[0]
    return np.arctan2(v[1], v[0])


def opposite_edges_mean_rotation(corners: np.ndarray) -> float:
    """
    Average the directions of opposite edges (0↔2 vs 1↔3),
    then pick the pair with larger total length. Circular mean modulo π.
    """
    v = np.roll(corners, -1, axis=0) - corners
    L2 = np.einsum('ij,ij->i', v, v)
    # choose the stronger opposite pair by summed length
    i, j = (0, 2) if (L2[0] + L2[2] >= L2[1] + L2[3]) else (1, 3)
    th_i, th_j = np.arctan2(v[i,1], v[i,0]), np.arctan2(v[j,1], v[j,0])
    C = np.cos(2*th_i) + np.cos(2*th_j)
    S = np.sin(2*th_i) + np.sin(2*th_j)
    return 0.5 * np.arctan2(S, C)


def _rot2d(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s,  c]])


def _rectangularity_score(rotated: np.ndarray) -> float:
    """
    Lower is better. We use |cov_xy| of rotated points as a simple,
    rotation-invariant “off-axis” measure (0 for a perfectly axis-aligned rectangle).
    """
    X = rotated - rotated.mean(axis=0, keepdims=True)
    C = (X.T @ X) / max(len(X)-1, 1)
    return abs(C[0, 1])


def _prefer_north_up(theta: float, corners: np.ndarray) -> float:
    """
    Flip axis direction (add π) if the dominant axis points downward.
    We examine the first principal component in the *rotated* frame and
    ensure its y-component is non-negative.
    """
    R = _rot2d(-theta)          # we rotate data by -θ in your pipeline
    X = (corners - corners.mean(axis=0, keepdims=True)) @ R.T
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    major_axis = Vt[0]          # axis in the rotated frame
    if major_axis[1] < 0:
        theta = (theta + np.pi) % (2*np.pi)
    # normalize to (-π/2, π/2] for readability
    if theta > np.pi/2:
        theta -= np.pi
    return theta


def _orient_theta_by_north(theta: float, north_vec_boct: np.ndarray) -> float:
    """Choose among θ + k·(π/2) so that the rotated north vector is most 'up'.
    We rotate data by -θ in plotting, so here we evaluate v' = R(-(θ+kπ/2))·n.
    Select k that maximizes (v'_y, -|v'_x|) lexicographically with v'_y≥0 preference.
    """
    n = np.asarray(north_vec_boct, float).reshape(2)
    best_theta = theta
    best_key = (-np.inf, -np.inf)
    for k in (0, 1, 2, 3):
        th = theta + k * (np.pi / 2.0)
        R = _rot2d(-th)
        v = R @ n
        key = (v[1], -abs(v[0]))  # prefer larger up (y), then smaller sideways (|x|)
        if v[1] < 0:
            # punish pointing down a bit to avoid choosing it unless all are down
            key = (v[1] - 1e6, -abs(v[0]))
        if key > best_key:
            best_key = key
            best_theta = th
    # normalize angle to (-π, π]
    if best_theta <= -np.pi:
        best_theta += 2*np.pi
    elif best_theta > np.pi:
        best_theta -= 2*np.pi
    return best_theta


def compute_north_vector_boct(centroid_boct: np.ndarray, reg: Registrar,
                               g_gcd: GeneralGCD, c_ell: EllipsoidCartesian,
                               c_oct: OctahedralCartesian, b_oct: OctahedralBarycentric,
                               comp_row: np.ndarray,
                               dlat: float = 1e-6) -> np.ndarray:
    """Compute the local geographic north vector in b_oct coordinates at the given centroid.
    We inverse-project the centroid to GCD, nudge latitude by +dlat (same lon), and project back.
    Returns an unnormalized 2D vector in b_oct space; caller may pixels it for plotting.
    comp_row: (3,) octant tuple to use for constructing b_oct Points
    """
    # normalize centroid shape to (1,2)
    c = np.atleast_2d(centroid_boct)
    # inverse: b_oct -> c_oct -> c_ell -> g_gcd
    comp = np.repeat(np.atleast_2d(comp_row), c.shape[0], axis=0)
    gcd_pt = reg.project(Points(c, b_oct, components=comp), [b_oct, c_oct, c_ell, g_gcd])
    lat0, lon0 = float(gcd_pt.coords[0, 0]), float(gcd_pt.coords[0, 1])
    # forward with small +lat step
    lat1 = max(-90.0, min(90.0, lat0 + dlat))
    north_gcd = Points(np.array([[lat1, lon0]]), g_gcd)
    north_boct = reg.project(north_gcd, [g_gcd, c_ell, c_oct, b_oct])
    vec = north_boct.coords[0] - c[0]
    return vec


def draw_north_arrow(ax, origin: np.ndarray, vec: np.ndarray, poly: np.ndarray,
                      color: str, scale: float = 0.12, label: str = "layer") -> None:
    """Draw a north arrow from origin along `vec`, scaled to polygon extent.
    - `origin`: (1,2) array-like in b_oct coords
    - `vec`: geographic north direction at origin in b_oct coords
    - `poly`: polygon (Nx2) to compute visual pixels
    """
    o = np.atleast_2d(origin)[0]
    v = np.asarray(vec, float)
    # pixels length to polygon extent so it’s visible for tiny patches
    extent = np.max(poly, axis=0) - np.min(poly, axis=0)
    L = float(scale * np.max(extent))
    if not np.isfinite(L) or L <= 0:
        L = 0.1
    # normalize vec; if degenerate, fall back to pure up in axes coords
    nrm = np.linalg.norm(v)
    if not np.isfinite(nrm) or nrm == 0.0:
        v = np.array([0.0, 1.0])
        nrm = 1.0
    v = v / nrm
    # force arrow to point upward in the plot for readability (visual convention)
    if v[1] < 0:
        v = -v
    start = o + 0.25 * L * v
    end   = o + 1.25 * L * v
    ax.annotate(label, xy=start, xytext=end,
                arrowprops=dict(arrowstyle="-|>", lw=1.2, facecolor=color, edgecolor=color),
                ha='center', va='bottom', fontsize=10, color=color,
                zorder=10, clip_on=False)


def choose_rotation_and_visualize(corners: np.ndarray, centroid: np.ndarray,
                                  reg: Registrar, g_gcd: GeneralGCD,
                                  c_ell: EllipsoidCartesian, c_oct: OctahedralCartesian,
                                  b_oct: OctahedralBarycentric, comp_row: np.ndarray, ax=None):
    methods = {
        "axis-mean": axis_mean_rotation(corners),
        "pca":       pca_rotation(corners),
        "opp-edges": opposite_edges_mean_rotation(corners),
    }

    # compute geographic north direction at the (unrotated) centroid in b_oct space
    north_vec_boct = compute_north_vector_boct(centroid, reg, g_gcd, c_ell, c_oct, b_oct, comp_row)

    # apply two-stage orientation per method:
    #  1) flip by π if needed so major axis points "up" in the rotated frame
    #  2) choose among θ + k·(π/2) so that geographic north is as close to +Y as possible
    methods = {
        name: _orient_theta_by_north(_prefer_north_up(theta, corners), north_vec_boct)
        for name, theta in methods.items()
    }

    # score each method by rectangularity after rotation
    scores = {}
    rotated = {}
    for name, theta in methods.items():
        R = _rot2d(-theta)
        rot = (corners - centroid) @ R.T + centroid
        rotated[name] = rot
        scores[name] = _rectangularity_score(rot)

    # pick the lowest score
    best = min(scores, key=scores.get)
    theta_best = methods[best]

    # viz
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=150)
    ax.clear()
    ax.set_aspect('equal', 'box')
    ax.grid(True, linestyle='--', alpha=0.6)

    # original
    ax.add_patch(patches.Polygon(corners, facecolor='none', edgecolor='k', linewidth=2, alpha=0.6, label='original'))

    # overlays
    colors = {"axis-mean":"tab:blue", "pca":"tab:green", "opp-edges":"tab:orange"}
    for name, rot in rotated.items():
        ax.add_patch(patches.Polygon(rot, facecolor='none', edgecolor=colors[name], linewidth=2, alpha=0.9,
                                     label=f"{name}: {np.degrees(methods[name]):.2f}°, score={scores[name]:.3e}"))
        # rotate the north vector by the same R that produced this overlay
        R = _rot2d(-methods[name])
        nvec_rot = (R @ np.atleast_2d(north_vec_boct).T).T[0]
        cent_rot = np.atleast_2d(rot.mean(axis=0))
        draw_north_arrow(ax, cent_rot, nvec_rot, rot, colors[name])

    ax.scatter([centroid[0,0]], [centroid[0,1]], s=30, c='k', zorder=3)

    ax.legend(loc='best', fontsize=9)
    ax.set_title("Barycentric quad rotation: original + three estimators (lower score = more rectangular)")

    return theta_best, best, scores


def edge_based_rotation(corners):
    """
    Return rotation of a quadrilateral by
    averaging the angles of its two longest edges.
    """
    edge_vectors = np.roll(corners, -1, axis=0) - corners
    edge_lengths_sq = np.sum(edge_vectors ** 2, axis=1)
    longest_edge_indices = np.argsort(edge_lengths_sq)[-2:]
    longest_vectors = edge_vectors[longest_edge_indices]
    angle1 = np.arctan2(longest_vectors[0, 1], longest_vectors[0, 0])
    angle2 = np.arctan2(longest_vectors[1, 1], longest_vectors[1, 0])
    return (angle1 + angle2) / 2.0


def stage(file: str,
          src_dir: str | Path = "src",
          dst_dir: str | Path = "src",
          bounds: tuple[float, float, float, float] | None = None,
          preset: str | None = None,
          tag: str | None = None,
          dpi: int = 150,
          show: bool = False,
          save_plot: bool = False,
          force: bool = False,
          **_):
    """
    Stage 4: load GCD bounds for a dataset, project to b_oct, compute optimal θ (radians)
    with north-up preference, write θ, centroid, and borders to disk (signatured filenames).

    Returns: tuple of output Paths (theta, centroid, bry, rot_bry)
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Resolve which bounds file to use + base name signature
    sig = None
    presets = load_presets()
    if preset:
        extent, sig = resolve_bounds(file, presets, preset=preset)
        base = f"{file}__{tag or sig}"
        bounds_file = src_dir / f"{base}_lat_lon_bounds.npy"
    elif bounds is not None:
        extent, sig = resolve_bounds(file, presets, explicit=bounds)
        base = f"{file}__{tag or sig}"
        bounds_file = src_dir / f"{base}_lat_lon_bounds.npy"
    else:
        # Fallback to an existing signatured bounds file, else the default one
        cand = list(src_dir.glob(f"{file}__*_lat_lon_bounds.npy"))
        if cand:
            cand.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            bounds_file = cand[0]
            base = bounds_file.name.replace("_lat_lon_bounds.npy", "")
            extent = np.load(bounds_file).astype(float)
        else:
            bounds_file = src_dir / f"{file}_lat_lon_bounds.npy"
            if not bounds_file.exists():
                raise FileNotFoundError(
                    f"Bounds file not found: {bounds_file}. Provide --preset/--bounds or run stage 2b.")
            base = file
            extent = np.load(bounds_file).astype(float)

    # Ensure extent shape and correctness
    extent = np.asarray(extent, dtype=float).reshape(4,)
    min_lat, min_lon, max_lat, max_lon = extent
    if min_lat > max_lat:
        min_lat, max_lat = max_lat, min_lat
    if min_lon > max_lon:
        min_lon, max_lon = max_lon, min_lon

    # Output paths
    theta_file    = dst_dir / f"{base}_theta.npy"
    centroid_file = dst_dir / f"{base}_centroid.npy"
    bry_file      = dst_dir / f"{base}_bry_border.npy"
    bryc_file     = dst_dir / f"{base}_bry_border_cmp.npy"
    rot_bry_file  = dst_dir / f"{base}_rot_bry_border.npy"

    if not force and not needs_run([bounds_file], [theta_file, centroid_file, bry_file, bryc_file, rot_bry_file]):
        print(f"[pr0004] up-to-date: {base}")
        return theta_file, centroid_file, bry_file, rot_bry_file

    # Registrar / domains
    reg = Registrar()
    g_gcd = GeneralGCD(reg)
    c_ell = EllipsoidCartesian(reg)
    c_oct = OctahedralCartesian(reg)
    b_oct = OctahedralBarycentric(reg, c_oct)
    EllipsoidGCD(reg)
    ak = AKOctahedralEllipsoid(reg)
    ak.set_accuracy(0.0000000001)

    # Build GCD rectangle and project to b_oct
    gcd_r = np.array([
        [min_lat, min_lon], [max_lat, min_lon], [max_lat, max_lon], [min_lat, max_lon]
    ], dtype=float)
    b_gcd = Points(gcd_r, g_gcd)
    b_data = reg.project(b_gcd, [g_gcd, c_ell, c_oct, b_oct])
    poly = b_data.coords.copy()
    centroid = np.atleast_2d(poly.mean(axis=0))
    comps = b_data.components.copy()  # (4,3) per-corner components
    comp_row = comps[0]

    # Compute θ and optionally visualize
    # We don’t need a figure unless saving/showing
    ax = None
    if save_plot or show:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=dpi)
    theta, method, scores = choose_rotation_and_visualize(poly, centroid, reg, g_gcd, c_ell, c_oct, b_oct, comp_row, ax=ax)

    # Save arrays
    np.save(theta_file, theta)
    np.save(centroid_file, centroid)
    np.save(bry_file, poly)
    np.save(bryc_file, comps)

    # Save rotated border and optional plot
    R = _rot2d(-theta)
    squared = (poly - centroid) @ R.T + centroid
    np.save(rot_bry_file, squared)

    if save_plot and ax is not None:
        out_png = dst_dir / f"{base}_theta_debug.png"
        plt.savefig(out_png, bbox_inches='tight', dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close('all')

    print(f"[pr0004] Chosen method: {method}, θ={np.degrees(theta):.4f}°, scores={scores}")
    return theta_file, centroid_file, bry_file, rot_bry_file


if __name__ == '__main__':
    import argparse
    script_dir = Path(__file__).parent
    p = argparse.ArgumentParser(description='Stage 4: compute theta/centroid from GCD bounds')
    p.add_argument('--file', type=str, default='gbr', help="dataset id, e.g. 'ggy', 'btn', 'jpn'")
    p.add_argument('--src', type=Path, default=script_dir / 'src')
    p.add_argument('--dst', type=Path, default=script_dir / 'src')

    p.add_argument('--bounds', type=float, nargs=4, metavar=('MIN_LAT','MIN_LON','MAX_LAT','MAX_LON'))
    p.add_argument('--preset', type=str, help="named bounds preset (per-dataset, see bounds_presets.yaml)")
    p.add_argument('--tag', type=str, help='explicit signature tag to embed in output filenames')
    p.add_argument('--dpi', type=int, default=300)
    p.add_argument('--show', action='store_true')
    p.add_argument('--save-plot', default=True, action='store_true')
    p.add_argument('--force', action='store_true')
    args = p.parse_args()

    stage(args.file, src_dir=args.src, dst_dir=args.dst,
          bounds=tuple(args.bounds) if args.bounds else None,
          preset=args.preset, tag=args.tag,
          dpi=args.dpi, show=args.show, save_plot=args.save_plot,
          force=args.force)
