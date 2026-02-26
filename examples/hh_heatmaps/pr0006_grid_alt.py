"""
Part of the H9 project - Preparation 0006
Project the plate-carree imagemap onto barycentric via sampling.
Last Tested 19 October 2025 √
"""

import numpy as np
from matplotlib import image
from scipy.spatial import KDTree
from hhg9 import Registrar, Points
from hhg9.algorithms.grid import qa_grid
from PIL import Image  # Pillow for clean image saving
from pathlib import Path
from bounds_utils import load_presets, resolve_bounds, needs_run


def stage(file: str,
          src_dir: str | Path = "src",
          dst_dir: str | Path = "src",
          bounds: tuple[float, float, float, float] | None = None,
          preset: str | None = None,
          tag: str | None = None,
          width_px: int = 6000,
          force: bool = False,
          **_):
    """
    Stage 5: Sample a plate-carree background into a barycentric grid using θ and centroid.
    Reads signatured inputs from previous stages and writes a signatured grid image.

    Inputs (preferred):
      - <src>/<file>__<sig>_lat_lon_bounds.npy        (from stage 3)
      - <src>/<file>__<sig>_gcd.png                   (from stage 4)
      - <src>/<file>__<sig>_net_border.npy            (from stage 5)
      - <src>/<file>__<sig>_theta.npy                 (from stage 5)
      - <src>/<file>__<sig>_centroid.npy              (from stage 5)

    Output:
      - <dst>/<file>__<sig>_grid.png
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Resolve signature + bounds (use explicit/preset → fallback to latest signatured bounds)
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
        cand = list(src_dir.glob(f"{file}__*_lat_lon_bounds.npy"))
        if cand:
            cand.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            bounds_file = cand[0]
            base = bounds_file.name.replace("_lat_lon_bounds.npy", "")
            extent = np.load(bounds_file).astype(float)
        else:
            bounds_file = src_dir / f"{file}_lat_lon_bounds.npy"
            base = file
            if not bounds_file.exists():
                raise FileNotFoundError(
                    f"Bounds file not found: {bounds_file}. Provide --preset/--bounds or run stage 3.")
            extent = np.load(bounds_file).astype(float)

    net_name, net_theta = 'mortar', '30'
    if file in presets:
        settings = presets[file]
        net_name, net_theta = settings['net']
    else:
        print(f"Add network-map, global rotation in (degrees) for {file} to presets.yaml")
        print(f'currently using {net_name} with {net_theta}.')

    # Ensure extent ordering [min_lat, min_lon, max_lat, max_lon]
    extent = np.asarray(extent, dtype=float).reshape(4,)
    min_lat, min_lon, max_lat, max_lon = extent
    if min_lat > max_lat:
        min_lat, max_lat = max_lat, min_lat
    if min_lon > max_lon:
        min_lon, max_lon = max_lon, min_lon

    # Signatured input/output paths
    gcd_img  = src_dir / f"{base}_gcd.png"
    b_border = src_dir / f"{base}_net_border.npy"
    theta_np = src_dir / f"{base}_theta.npy"
    cent_np  = src_dir / f"{base}_centroid.npy"
    rot_net  = src_dir / f"{base}_rot_net_border.npy"
    out_grid = dst_dir / f"{base}_grid.png"

    # Up-to-date check (skip unless force)
    inputs = [bounds_file, gcd_img, b_border, theta_np, cent_np, rot_net]
    if not force and not needs_run(inputs, [out_grid]):
        print(f"[pr0006] up-to-date: {out_grid}")
        return out_grid

    # Registrar & domains
    # util = Util()
    reg = Registrar()
    g_gcd = reg.domain('g_gcd')  # latitude longitude
    p_pix = reg.domain('p_pix')  # plate carrée pixel
    b_oct = reg.domain('b_oct')
    b_oct.set_warp('../src/l4_polished.npz')

    n_oct = reg.domain(f'n_oct:{net_name}')
    pix_gcd = reg.projection('pix_gcd')  # convert plate carrée to gcd

    # Load inputs: use the *rotated* border so the grid is tight in its own extent
    rot_border = np.load(rot_net)  # polygon corners in rotated n_oct (shape (4,2))
    img_w = int(width_px)
    (ww, hh, pxl, msk, (grid_org, (sx, sy))) = qa_grid(rot_border, img_w)
    np.save(src_dir / f"{base}_bg_extent.npy", grid_org)
    theta = float(np.load(theta_np))
    c_th, s_th = np.cos(theta), np.sin(theta)
    rot = np.array([[c_th, -s_th], [s_th, c_th]])
    centroid = np.load(cent_np)
    dnet = (pxl[msk] - centroid) @ rot.T + centroid  # inverse rotation
    pts = Points(dnet, n_oct)
    bry = reg.project(pts, [n_oct, b_oct])
    refs = reg.project(bry, [b_oct, g_gcd])
    # Load plate-carree image and build a pixel domain instance bound to the extent
    if not gcd_img.exists():
        raise FileNotFoundError(f"{gcd_img} not found. Run stage 4.")
    img = image.imread(str(gcd_img), 'png')
    pc_px = p_pix.adopt(img)  # create pixel cloud from image
    # Order for set_dim is (lon_min, lon_max, lat_min, lat_max)
    pix_gcd.set_dim(pc_px, (min_lon, max_lon, min_lat, max_lat))
    pc_sp = reg.project(pc_px, [p_pix, g_gcd])

    # KDTree sample
    src_tree = KDTree(pc_sp.coords)
    _, idx = src_tree.query(refs, workers=-1)
    samples = pc_sp.samples[idx]
    if samples.shape[1] == 3:
        rgba = np.hstack((samples, np.ones((samples.shape[0], 1), dtype=samples.dtype)))
    else:
        rgba = samples

    # Paint into raster
    pvx = np.round((pxl[:, 0] - grid_org[0]) * sx).astype(int)
    pvy = np.round((pxl[:, 1] - grid_org[1]) * sy).astype(int)
    pvy = hh - pvy  # flip y so top row is y=0
    out = np.zeros((hh + 1, ww + 1, 4), dtype=float)
    # Sanity check: only draw samples that fall within the raster bounds
    vx = pvx[msk]
    vy = pvy[msk]
    valid = (vx >= 0) & (vx <= ww) & (vy >= 0) & (vy <= hh)
    if not np.all(valid):
        dropped = int((~valid).sum())
        if dropped:
            print(f"[pr0006] warning: {dropped} samples fell outside raster and were dropped")
    out[vy[valid], vx[valid]] = rgba[valid]

    image_uint8 = (out * 255).astype(np.uint8)
    Image.fromarray(image_uint8).save(out_grid)
    print(f"[pr0005] Saved {out_grid}  size={ww+1}x{hh+1}")
    return out_grid


if __name__ == '__main__':
    import argparse
    script_dir = Path(__file__).parent
    p = argparse.ArgumentParser(description='Stage 5: sample plate-carree into barycentric grid')
    p.add_argument('--file', type=str, default='gbr', help="dataset id, e.g. 'ggy', 'btn', 'jpn'")
    p.add_argument('--src', type=Path, default=script_dir / 'src')
    p.add_argument('--dst', type=Path, default=script_dir / 'src')

    p.add_argument('--bounds', type=float, nargs=4, metavar=('MIN_LAT','MIN_LON','MAX_LAT','MAX_LON'))
    p.add_argument('--preset', type=str, help="named bounds preset (per-dataset, see bounds_presets.yaml)")
    p.add_argument('--tag', type=str, help='explicit signature tag to embed in output filenames')
    p.add_argument('--width', type=int, default=6000, help='grid width in pixels')
    p.add_argument('--force', action='store_true')
    args = p.parse_args()

    stage(args.file, src_dir=args.src, dst_dir=args.dst,
          bounds=tuple(args.bounds) if args.bounds else None,
          preset=args.preset, tag=args.tag,
          width_px=args.width, force=args.force)
