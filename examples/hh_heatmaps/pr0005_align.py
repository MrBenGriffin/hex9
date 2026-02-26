"""
Part of the H9 project - Preparation 0005
Load the GCD quadrilateral, project onto Barycentric Net Coordinates.
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


def north_arrow(ax, centroid, poly, color, scale=0.12, label="hex_layer"):
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


def draw_north_arrow(ax, origin: np.ndarray, vec: np.ndarray, poly: np.ndarray,
                      color: str, scale: float = 0.12, label: str = "hex_layer") -> None:
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
          dpi: int = 300,
          show: bool = False,
          save_plot: bool = False,
          force: bool = False,
          **_):
    """
    Stage 4: load GCD bounds for a dataset, project to n_oct
    write θ, centroid, and borders to disk (signatured filenames).
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
                    f"Bounds file not found: {bounds_file}. Provide --preset/--bounds or run stage 3.")
            base = file
            extent = np.load(bounds_file).astype(float)
    if file in presets:
        settings = presets[file]
        net_name, net_theta = settings["net"]
    else:
        net_name, net_theta = 'mortar', '30'
        print(f"Add network-map, global rotation in (degrees) for {file} to presets.yaml")
        print(f'currently using {net_name} with {net_theta}.')

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
    net_file      = dst_dir / f"{base}_net_border.npy"
    rot_net_file  = dst_dir / f"{base}_rot_net_border.npy"

    if not force and not needs_run([bounds_file], [theta_file, centroid_file, net_file, rot_net_file]):
        print(f"[pr0005] up-to-date: {base}")
        return theta_file, centroid_file, net_file, rot_net_file

    # Registrar / domains
    reg = Registrar()
    g_gcd = reg.domain('g_gcd')
    b_oct = reg.domain('b_oct')
    b_oct.set_warp('../src/l4_polished.npz')
    n_oct = reg.domain(f'n_oct:{net_name}')

    # Build GCD rectangle and project to n_oct
    gcd_r = np.array([
        [min_lat, min_lon], [max_lat, min_lon], [max_lat, max_lon], [min_lat, max_lon]
    ], dtype=float)
    b_gcd = Points(gcd_r, g_gcd)
    b_data = reg.project(b_gcd, [g_gcd, b_oct, n_oct])
    poly = b_data.coords.copy()
    centroid = np.atleast_2d(poly.mean(axis=0))
    theta = np.deg2rad(net_theta)

    # Save arrays
    np.save(theta_file, theta)
    np.save(centroid_file, centroid)
    np.save(net_file, poly)

    # Save rotated border and optional plot
    R = _rot2d(theta)
    squared = (poly - centroid) @ R.T + centroid
    np.save(rot_net_file, squared)

    print(f"[pr0005] Chosen net projection: {net_name}; global rotation: θ={np.degrees(theta):.4f}°")
    return theta_file, centroid_file, net_file, rot_net_file


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
