import os
import time
import gc
import numpy as np
from multiprocessing import Pool, cpu_count

from hhg9 import Registrar, Points
from hhg9.domains import GeneralGCD, EllipsoidCartesian, OctahedralCartesian, OctahedralBarycentric
from hhg9.projections import EllipsoidGCD, AKOctahedralEllipsoid


# ---- CONFIG ----
BATCH_SIZE = 10000
N_WORKERS = 18  # or cpu_count()
ACCURACY = 0.01  # cm


def init_projection():
    """Setup projection pipeline objects for each worker."""
    global reg, g_gcd, c_ell, c_oct, b_oct
    reg = Registrar()
    g_gcd = GeneralGCD(reg)
    c_ell = EllipsoidCartesian(reg)
    c_oct = OctahedralCartesian(reg)
    b_oct = OctahedralBarycentric(reg, c_oct)
    EllipsoidGCD(reg)
    ak = AKOctahedralEllipsoid(reg)
    ak.set_accuracy(ACCURACY)


def process_batch(args):
    """Process a batch of indices."""
    start_idx, end_idx, lat_lon_batch = args
    p_gcd = Points(lat_lon_batch, g_gcd)
    result = reg.project(p_gcd, [g_gcd, c_ell, c_oct, b_oct])
    return start_idx, result.coords, result.components


def ensure_memmap(path, shape, dtype=np.float64):
    """Create a memmap-friendly .npy file if not already existing."""
    if not os.path.exists(path):
        arr = np.full(shape, np.nan, dtype=dtype)
        np.save(path, arr)
    # Now open in r+ mode for read/write
    return np.load(path, mmap_mode='r+')


if __name__ == '__main__':
    # Kyushu
    min_lat_lon = np.array([30.973930071123313, 129.37257774562536])
    max_lat_lon = np.array([34.867227645401115, 132.06213814072615])

    FILE = 'jpn'
    SRC_FILE = f'src/{FILE}_lon_lat_pop.npy'
    BRY_FILE = f'src/{FILE}_bry.npy'
    CMP_FILE = f'src/{FILE}_bry_cmp.npy'

    # ---- Load source ----
    print(f"Loading {SRC_FILE}")
    g_data = np.load(SRC_FILE)
    lon_lat = g_data[:, :2]
    lat_lon = lon_lat[:, [1, 0]]
    mask = (
            (lat_lon[:, 0] >= min_lat_lon[0]) &
            (lat_lon[:, 0] <= max_lat_lon[0]) &
            (lat_lon[:, 1] >= min_lat_lon[1]) &
            (lat_lon[:, 1] <= max_lat_lon[1])
    )
    lat_lon = lat_lon[mask]
    size = lat_lon.shape[0]
    del g_data, lon_lat
    gc.collect()  # force garbage collection

    # ---- Ensure output files exist & open as memmaps ----
    b_coords = ensure_memmap(BRY_FILE, (size, 2))
    b_comps = ensure_memmap(CMP_FILE, (size, 3))

    # ---- Figure out unprocessed indices ----
    mask_done = ~np.isnan(b_coords[:, 0])
    todo_indices = np.where(~mask_done)[0]
    total_batches = (len(todo_indices) + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"Total points: {size}")
    print(f"Already done: {np.sum(mask_done)}")
    print(f"Remaining: {len(todo_indices)} points in {total_batches} batches.")

    if len(todo_indices) == 0:
        print("Nothing to do — exiting.")
        exit(0)

    # ---- Prepare batches ----
    batches = []
    for i in range(0, len(todo_indices), BATCH_SIZE):
        batch_idx = todo_indices[i:i + BATCH_SIZE]
        batches.append((batch_idx[0], batch_idx[-1] + 1, lat_lon[batch_idx]))

    # ---- Parallel processing ----
    print(f"Starting projection on {N_WORKERS} workers...")
    start_time_all = time.time()

    with Pool(processes=N_WORKERS, initializer=init_projection) as pool:
        for batch_num, (start_idx, coords, comps) in enumerate(pool.imap(process_batch, batches), start=1):
            # Store results directly into memmap
            end_idx = start_idx + coords.shape[0]
            b_coords[start_idx:end_idx] = coords
            b_comps[start_idx:end_idx] = comps

            # Logging
            elapsed = time.time() - start_time_all
            batch_time = elapsed / batch_num
            remaining_batches = total_batches - batch_num
            eta = remaining_batches * batch_time
            print(f"[{batch_num}/{total_batches}] "
                  f"Processed {coords.shape[0]} points in ~{batch_time:.2f}s/batch | "
                  f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")

    print(f"Done. Total time: {time.time() - start_time_all:.1f}s")
