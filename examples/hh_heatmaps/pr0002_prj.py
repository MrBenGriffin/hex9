from __future__ import annotations
import time
from pathlib import Path
import numpy as np
from hhg9 import Registrar, Points
from bounds_utils import load_presets, resolve_bounds, apply_bounds, needs_run

ACCURACY = 0.000001  # cm


def stage(file: str,
          src_dir: str | Path = "src",
          dst_dir: str | Path = "src",
          bounds: tuple[float, float, float, float] | None = None,
          preset: str | None = None,
          tag: str | None = None,
          accuracy: float = ACCURACY,
          force: bool = False,
          **_):
    """
    Project CSV-derived lat/lon to barycentric coords and components
    Inputs:
      <src>/<file>_lat_lon_pop.npy  (columns: lat,lon,pop)

    Outputs:
      <dst>/<file>__<sig>_bry.npy
      <dst>/<file>__<sig>_bry_cmp.npy
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    src_file = src_dir / f"{file}_lat_lon_pop.npy"

    if not src_file.exists():
        raise FileNotFoundError(f"Source npy not found: {src_file}")

    # Load data
    data = np.load(src_file)
    smp = data[:, 2]
    lat_lon = data[:, :2]
    del data

    # Resolve bounds + signature
    presets = load_presets()
    if preset:
        # YAML per-dataset preset
        bounds, sig = resolve_bounds(file, presets, preset=preset)
    elif bounds:
        # explicit numeric bounds
        bounds, sig = resolve_bounds(file, presets, explicit=bounds)
    else:
        # data-driven with padding
        bounds, sig = resolve_bounds(file, presets, data_lat_lon=lat_lon, pad_frac=0.025)

    base = f"{file}__{tag or sig}"
    bry_file = dst_dir / f"{base}_bry.npy"
    cmp_file = dst_dir / f"{base}_bry_cmp.npy"
    smp_file = dst_dir / f"{base}_smp.npy"
    if smp_file.exists():
        smp_file.unlink()
    np.save(smp_file, smp)
    del smp

    if not force and not needs_run([src_file], [bry_file, cmp_file]):
        print(f"[pr0002] up-to-date: {bry_file}, {cmp_file}")
        return bry_file, cmp_file

    # Apply crop before projection
    lat_lon = apply_bounds(lat_lon, bounds)
    size = lat_lon.shape[0]
    if size == 0:
        print("[pr0002] No points after bounds filter; nothing to do.")
        return bry_file, cmp_file

    bry_file.parent.mkdir(parents=True, exist_ok=True)
    cmp_file.parent.mkdir(parents=True, exist_ok=True)
    reg = Registrar()
    b_oct = reg.domain('b_oct')
    g_gcd = reg.domain('g_gcd')
    pts = Points(lat_lon, g_gcd)
    del lat_lon
    reg.projection('gcd_bry').set_accuracy(accuracy)
    start_time = time.perf_counter()
    bry = reg.project(pts, [g_gcd, b_oct])
    seconds = time.perf_counter() - start_time
    print(f'{seconds:.6f} seconds to process {len(bry)} points at {accuracy}m accuracy.')
    np.save(bry_file, bry.coords)
    np.save(cmp_file, bry.components)
    return bry_file, cmp_file


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description='Project lat/lon to barycentric')
    p.add_argument('--file', type=str, default='gbr', help="dataset id, e.g.'ggy', ''btn', 'jpn'")
    p.add_argument('--src', type=Path, default=Path('src'))
    p.add_argument('--dst', type=Path, default=Path('src'))
    p.add_argument('--bounds', type=float, nargs=4, metavar=('MIN_LAT','MIN_LON','MAX_LAT','MAX_LON'))
    p.add_argument('--preset', type=str, help="named bounds preset (per-dataset, see bounds_presets.yaml)")
    p.add_argument('--tag', type=str, help='explicit signature tag to embed in output filenames')
    p.add_argument('--accuracy', type=float, default=ACCURACY, help='accuracy in metres')
    p.add_argument('--force', action='store_true')
    args = p.parse_args()
    stage(args.file, src_dir=args.src, dst_dir=args.dst,
          bounds=tuple(args.bounds) if args.bounds else None,
          preset=args.preset, tag=args.tag,
          accuracy=args.accuracy, force=args.force)