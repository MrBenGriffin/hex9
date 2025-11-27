from __future__ import annotations
import time
from pathlib import Path
from multiprocessing import Pool
import numpy as np

from bounds_utils import load_presets, resolve_bounds, apply_bounds, needs_run

# --- adjust these imports to your project structure ---
from hhg9.base.registrar import Registrar
from hhg9.base.points import Points
from hhg9.domains.general_gcd import GeneralGCD
from hhg9.domains.ellipsoid_cartesian import EllipsoidCartesian
from hhg9.domains.octahedral_cartesian import OctahedralCartesian
from hhg9.domains.octahedral_barycentric import OctahedralBarycentric
from hhg9.projections.ellipsoid_gcd import EllipsoidGCD
from hhg9.projections.ak_octahedral import AKOctahedralEllipsoid
# ------------------------------------------------------

# Default knobs (can be overridden via CLI)
BATCH_SIZE = 10000
N_WORKERS = 18
ACCURACY = 0.01  # cm


def _init_worker(accuracy_cm: float):
    # construct fresh registrar per worker
    global reg, g_gcd, c_ell, c_oct, b_oct
    reg = Registrar()
    g_gcd = GeneralGCD(reg)
    c_ell = EllipsoidCartesian(reg)
    c_oct = OctahedralCartesian(reg)
    b_oct = OctahedralBarycentric(reg, c_oct)
    EllipsoidGCD(reg)
    ak = AKOctahedralEllipsoid(reg)
    ak.set_accuracy(accuracy_cm)


def _process_batch(args):
    start_idx, lat_lon_batch = args
    p_gcd = Points(lat_lon_batch, g_gcd)
    result = reg.project(p_gcd, [g_gcd, c_ell, c_oct, b_oct])
    return start_idx, result.coords, result.components


def stage(file: str,
          src_dir: str | Path = "src",
          dst_dir: str | Path = "src",
          bounds: tuple[float, float, float, float] | None = None,
          preset: str | None = None,
          tag: str | None = None,
          batch_size: int = BATCH_SIZE,
          n_workers: int = N_WORKERS,
          accuracy_cm: float = ACCURACY,
          force: bool = False,
          **_):
    """
    Project CSV-derived lat/lon to barycentric coords and components, memmap-friendly.

    Inputs:
      <src>/<file>_lon_lat_pop.npy  (columns: lon,lat,pop)

    Outputs:
      <dst>/<file>__<sig>_bry.npy
      <dst>/<file>__<sig>_bry_cmp.npy
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    src_file = src_dir / f"{file}_lat_lon_pop.npy"

    if not src_file.exists():
        raise FileNotFoundError(f"Source NPY not found: {src_file}")

    # Load and order to (lat,lon)
    g_data = np.load(src_file)
    lat_lon = g_data[:, :2]
    del g_data

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

    # Early exit if up-to-date
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

    # (Re)create memmaps
    if bry_file.exists(): bry_file.unlink()
    if cmp_file.exists(): cmp_file.unlink()
    b_coords = np.lib.format.open_memmap(bry_file, mode='w+', dtype=np.float64, shape=(size, 2))
    b_comps  = np.lib.format.open_memmap(cmp_file, mode='w+', dtype=np.int8,    shape=(size, 3))
    b_coords[:] = np.nan
    b_comps[:]  = -1

    # Build batches
    idx = np.arange(size)
    batches = []
    for i in range(0, size, batch_size):
        batch_idx = idx[i:i + batch_size]
        batches.append((batch_idx[0], lat_lon[batch_idx]))

    print(f"[pr0002] Total points: {size}")
    print(f"[pr0002] Starting projection on {n_workers} workers...")

    t0 = time.time()
    with Pool(processes=n_workers, initializer=_init_worker, initargs=(accuracy_cm,)) as pool:
        for batch_num, (start_idx, coords, comps) in enumerate(pool.imap(_process_batch, batches), start=1):
            end_idx = start_idx + coords.shape[0]
            b_coords[start_idx:end_idx] = coords
            b_comps[start_idx:end_idx] = comps
            if batch_num % 10 == 0:
                elapsed = time.time() - t0
                print(f"[pr0002] batch {batch_num}: {elapsed:.1f}s")

    print(f"[pr0002] Done in {time.time() - t0:.1f}s → {bry_file}, {cmp_file}")
    return bry_file, cmp_file


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description='Project lat/lon to barycentric (memmap)')
    p.add_argument('--file', type=str, default='gbr', help="dataset id, e.g.'ggy', ''btn', 'jpn'")
    p.add_argument('--src', type=Path, default=Path('src'))
    p.add_argument('--dst', type=Path, default=Path('src'))
    p.add_argument('--bounds', type=float, nargs=4, metavar=('MIN_LAT','MIN_LON','MAX_LAT','MAX_LON'))
    p.add_argument('--preset', type=str, help="named bounds preset (per-dataset, see bounds_presets.yaml)")
    p.add_argument('--tag', type=str, help='explicit signature tag to embed in output filenames')
    p.add_argument('--batch', type=int, default=BATCH_SIZE)
    p.add_argument('--workers', type=int, default=N_WORKERS)
    p.add_argument('--accuracy', type=float, default=ACCURACY)
    p.add_argument('--force', action='store_true')
    args = p.parse_args()
    stage(args.file, src_dir=args.src, dst_dir=args.dst,
          bounds=tuple(args.bounds) if args.bounds else None,
          preset=args.preset, tag=args.tag,
          batch_size=args.batch, n_workers=args.workers,
          accuracy_cm=args.accuracy, force=args.force)