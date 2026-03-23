"""
Part of the H9 project - Preparation 0003
(1) Load the population numpy data file
(2) Derive or resolve a boundary and project it onto the Barycentric Octahedral Net.
"""
import os
import numpy as np
from pathlib import Path
from bounds_utils import load_presets, resolve_bounds
from hhg9 import Registrar, Points


def stage(file: str,
          src_dir: str | Path = "src",
          dst_dir: str | Path = "src",
          bounds: tuple[float, float, float, float] | None = None,
          preset: str | None = None,
          tag: str | None = None,
          force: bool = False,
          **_):
    """
    Extract population vector, resolve boundary (explicit/preset/data-driven),
    and project the boundary min/max corners to barycentric, saving artifacts with a signature.
    """
    script_dir = Path(__file__).parent  # for pretty-printing relative paths only
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    np_file = src_dir / f"{file}_lat_lon_pop.npy"
    if not np_file.exists():
        raise FileNotFoundError(f"Source npy not found: {np_file}")

    print(f"Loading {np_file}")
    # Load data
    data = np.load(np_file)
    lat_lon = data[:, :2]
    del data

    # Resolve bounds + signature
    presets = load_presets()
    if preset:
        bounds, sig = resolve_bounds(file, presets, preset=preset)
    elif bounds:
        bounds, sig = resolve_bounds(file, presets, explicit=bounds)
    else:
        bounds, sig = resolve_bounds(file, presets, data_lat_lon=lat_lon, pad_frac=0.025)
    base = f"{file}__{tag or sig}"

    if preset:
        lat, lon = lat_lon[:, 0], lat_lon[:, 1]
        mask = (lat >= bounds[0]) & (lat <= bounds[2]) & (lon >= bounds[1]) & (lon <= bounds[3])
        lat_lon = lat_lon[mask]

    # Outputs
    bd_file  = dst_dir / f"{base}_lat_lon_bounds.npy"
    br_file  = dst_dir / f"{base}_bounds_bry.npy"
    brc_file = dst_dir / f"{base}_bounds_bry_cmp.npy"

    # Up-to-date check (boundary and pop depend only on the src npy)
    from bounds_utils import needs_run as _needs
    if not force and not _needs([np_file], [bd_file, br_file, brc_file]):
        print(f"[pr0003] up-to-date: {base}")
        return bd_file, br_file, brc_file

    # Save bounds extent (min_lat, min_lon, max_lat, max_lon)
    min_lat_lon = np.array([bounds[0], bounds[1]], dtype=float)
    max_lat_lon = np.array([bounds[2], bounds[3]], dtype=float)
    extent = np.array([*min_lat_lon, *max_lat_lon], dtype=float)
    np.save(bd_file, extent)

    # Project the 2 boundary corners to barycentric
    reg = Registrar()
    g_gcd = reg.domain('g_gcd')
    b_oct = reg.domain('b_oct')

    mm_gcd = Points(np.vstack((min_lat_lon, max_lat_lon)), g_gcd)
    mm_data = reg.project(mm_gcd, [g_gcd, b_oct])
    mm_bry = mm_data.coords.T.reshape(-1)
    mm_bry_c = mm_data.oid
    np.save(br_file, mm_bry)
    np.save(brc_file, mm_bry_c)

    print("[pr0003] Wrote:",
          bd_file.relative_to(script_dir),
          br_file.relative_to(script_dir),
          brc_file.relative_to(script_dir))
    return bd_file, br_file, brc_file


if __name__ == '__main__':
    import argparse
    script_dir = Path(__file__).parent
    p = argparse.ArgumentParser(description='Stage 3: boundary projection')
    p.add_argument('--file', type=str, default='gbr', help="dataset id, e.g. 'btn', 'jan'")
    p.add_argument('--src', type=Path, default=script_dir / 'src')
    p.add_argument('--dst', type=Path, default=script_dir / 'src')
    p.add_argument('--bounds', type=float, nargs=4, metavar=('MIN_LAT','MIN_LON','MAX_LAT','MAX_LON'))
    p.add_argument('--preset', type=str, help="named bounds preset (per-dataset, see bounds_presets.yaml)")
    p.add_argument('--tag', type=str, help='explicit signature tag to embed in output filenames')
    p.add_argument('--force', action='store_true')
    args = p.parse_args()
    stage(args.file, src_dir=args.src, dst_dir=args.dst,
          bounds=tuple(args.bounds) if args.bounds else None,
          preset=args.preset, tag=args.tag, force=args.force)
