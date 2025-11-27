"""
Part of the H9 project - Preparation 0003
Load a boundary (in GCD) and retrieve an image from OpenStreetMap.
Last Tested 07 August 2025 √
"""
import io
import math

import numpy as np
import cartopy.crs as ccrs  # Import the Cartopy Coordinate Reference Systems
import pyproj
from cartopy.io.img_tiles import OSM
from matplotlib import pyplot as plt, patches
from hhg9.algorithms.distance import wgs84_angular_ratio
from PIL import Image  # Use Pillow for clean image saving
# from shapely.geometry import Polygon
# import pyproj
# import math

from pathlib import Path
from bounds_utils import load_presets, resolve_bounds, needs_run


def calculate_dpi_aware_zoom(bounds, figsize_inches, dpi):
    """
    Calculates the zoom level needed to match a target DPI.
    Args:
        bounds (list): [lon_min, lon_max, lat_min, lat_max] in WGS84 degrees.
        figsize_inches (tuple): (width, height) of the matplotlib figure in inches.
        dpi (int): The target dots per inch (DPI) of the output image.

    Returns:
        int: The calculated integer zoom level.
    """
    # 1. Calculate required output dimensions in pixels
    fig_w_px = figsize_inches[0] * dpi
    fig_h_px = figsize_inches[1] * dpi

    # 2. Project geographic bounds to Web Mercator meters
    lon_min, lon_max, lat_min, lat_max = bounds
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    merc_min_x, merc_min_y = transformer.transform(lon_min, lat_min)
    merc_max_x, merc_max_y = transformer.transform(lon_max, lat_max)
    merc_w_meters = merc_max_x - merc_min_x
    merc_h_meters = merc_max_y - merc_min_y

    # 3. Determine the required resolution in meters/pixel
    resolution_w = merc_w_meters / fig_w_px
    resolution_h = merc_h_meters / fig_h_px
    required_resolution = min(resolution_w, resolution_h)
    TILE_SIZE = 256  # Standard tile size in pixels
    EARTH_RADIUS_M = 6378137.0
    try:
        zoom_float = math.log2((2 * math.pi * EARTH_RADIUS_M) / (required_resolution * TILE_SIZE))
    except (ValueError, ZeroDivisionError):
        zoom_float = 0  # Handle cases with zero size
    return int(np.ceil(zoom_float))

def stage(file: str,
          src_dir: str | Path = "src",
          dst_dir: str | Path = "src",
          bounds: list[float] | tuple[float, float, float, float] | None = None,
          preset: str | None = None,
          tag: str | None = None,
          dpi: int = 100,
          width_px: int = 6000,
          zoom_margin: int = 2,
          fmt: str = "png",
          force: bool = False,
          **_):
    """
    Render an OSM background image for the dataset bounds.

    Inputs (one of):
      - explicit `bounds` (min_lat, min_lon, max_lat, max_lon)
      - `preset` (looked up in bounds_presets.yaml for this dataset)
      - fallback: load bounds from `<src>/<file>__*_lat_lon_bounds.npy` or `<src>/<file>_lat_lon_bounds.npy`

    Output:
      `<dst>/<file>__<sig>_gcd.<fmt>` (sig from preset/bbox when applicable)
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    # 1) Resolve bounds + base name signature
    sig = None
    extent = None  # (min_lat, min_lon, max_lat, max_lon)
    presets = load_presets()

    if preset or bounds is not None:
        if preset:
            extent, sig = resolve_bounds(file, presets, preset=preset)
        elif bounds is not None:
            extent, sig = resolve_bounds(file, presets, explicit=tuple(bounds))
        base = f"{file}__{tag or sig}"
        bounds_file = src_dir / f"{base}_lat_lon_bounds.npy"
    else:
        # Fallback: try signatured first, then default
        cand = list(src_dir.glob(f"{file}__*_lat_lon_bounds.npy"))
        if cand:
            # pick most recent
            cand.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            bounds_file = cand[0]
            base = bounds_file.name.replace("_lat_lon_bounds.npy", "")
        else:
            bounds_file = src_dir / f"{file}_lat_lon_bounds.npy"
            base = file
        if not bounds_file.exists():
            raise FileNotFoundError(f"Bounds file not found: {bounds_file}\n"
                                    f"Provide --preset/--bounds or generate bounds in stage 2b.")
        # file stores [min_lat, min_lon, max_lat, max_lon]
        extent = np.load(bounds_file).astype(float)

    # Ensure extent is ndarray of shape (4,)
    extent = np.asarray(extent, dtype=float).reshape(4,)
    min_lat, min_lon, max_lat, max_lon = extent

    # Sanity: swap if needed
    if min_lat > max_lat:
        min_lat, max_lat = max_lat, min_lat
    if min_lon > max_lon:
        min_lon, max_lon = max_lon, min_lon

    # Prepare output path and up-to-date check
    out_img = dst_dir / f"{base}_gcd.{fmt.lower()}"
    inputs = [bounds_file] if 'bounds_file' in locals() and bounds_file.exists() else []
    if not force and not needs_run(inputs, [out_img]):
        print(f"[pr0003] up-to-date: {out_img}")
        return out_img

    # Compute pixel geometry and zoom
    lon_lat = np.array([min_lon, max_lon, min_lat, max_lat], dtype=float)
    ratio = wgs84_angular_ratio(lon_lat)
    img_w_pix = int(width_px)
    img_h_pix = max(1, int(img_w_pix * ratio))
    fig_w_in, fig_h_in = img_w_pix / dpi, img_h_pix / dpi
    zoom = calculate_dpi_aware_zoom(lon_lat, (fig_w_in, fig_h_in), dpi) - 1
    level = max(0, int(zoom) - int(zoom_margin))

    # Render with Cartopy + OSM
    imagery = OSM()
    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi, frameon=False)
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], ccrs.PlateCarree())
    ax.axis('off')
    ax.add_image(imagery, level, regrid_shape=(img_w_pix, img_h_pix), interpolation='spline36')

    # Save to disk via a raw buffer (avoids white borders)
    with io.BytesIO() as buff:
        fig.savefig(buff, format='raw')
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    plt.close(fig)
    im = data.reshape((img_h_pix, img_w_pix, -1))
    Image.fromarray(im).save(out_img)

    print(f"[pr0003] Saved {out_img}  size={img_w_pix}x{img_h_pix}px  zoom≈{zoom} level={level}")
    return out_img


if __name__ == '__main__':
    import argparse
    script_dir = Path(__file__).parent
    p = argparse.ArgumentParser(description='Stage 3: render OSM background for dataset bounds')
    p.add_argument('--file', type=str, default='gbr', help="dataset id, e.g. 'btn', 'jpn'")
    p.add_argument('--src', type=Path, default=script_dir / 'src')
    p.add_argument('--dst', type=Path, default=script_dir / 'src')

    p.add_argument('--bounds', type=float, nargs=4, metavar=('MIN_LAT','MIN_LON','MAX_LAT','MAX_LON'))
    p.add_argument('--preset', type=str, help="named bounds preset (per-dataset, see bounds_presets.yaml)")
    p.add_argument('--tag', type=str, help='explicit signature tag to embed in output filenames')
    p.add_argument('--dpi', type=int, default=100)
    p.add_argument('--width', type=int, default=10000, help='output width in pixels')
    p.add_argument('--zoom-margin', type=int, default=0, help='subtract from calculated zoom')
    p.add_argument('--fmt', type=str, default='png', choices=['png','jpg','jpeg','tif','tiff'])
    p.add_argument('--force', action='store_true')
    args = p.parse_args()

    stage(args.file, src_dir=args.src, dst_dir=args.dst,
          bounds=tuple(args.bounds) if args.bounds else None,
          preset=args.preset, tag=args.tag,
          dpi=args.dpi, width_px=args.width, zoom_margin=args.zoom_margin,
          fmt=args.fmt, force=args.force)
