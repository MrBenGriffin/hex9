# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
ak_parallel improves processing of the work intensive c_ell->c_oct
"""

import numpy as np
from hhg9 import Registrar, Points
from hhg9.base.projection import Projection
from concurrent.futures import ProcessPoolExecutor
from functools import partial


# --- WORKER FUNCTIONS (Must be top-level for pickling) ---

def _worker_project_gcd_to_bary(chunk_ll: np.ndarray, warp_file: str = None, accuracy: int = 38):
    """
    Worker process:
    1. Rebuilds the minimal stack (Registrar + Domains).
    2. Loads the specific Warp File (L4 Gold) into the b_oct domain.
    3. Runs the projection chain.
    """
    # 1. Rebuild Stack
    reg = Registrar()
    # Ensure all domains are initialized so 'project' finds the path
    _ = reg.domain('g_gcd')
    _ = reg.domain('c_ell')
    _ = reg.domain('c_oct')
    b_oct = reg.domain('b_oct')
    reg.projection('oct_ell').accuracy = accuracy

    # 2. INJECT THE WARP
    if warp_file:
        # Assuming b_oct has a method to load the warp from disk/path
        # If your API is b_oct.warp.load(file), do that.
        # Based on previous context, it might be:
        b_oct.set_warp(warp_file)
        # b_oct.warp.default_iterations = 200

    # 3. Project
    # Force contiguous to prevent memory fragmentation warnings in tight loops
    chunk_ll = np.ascontiguousarray(chunk_ll, dtype=np.float64)
    pts = Points(chunk_ll, 'g_gcd')

    # Run the chain: GCD -> Ellipsoid -> Octant -> Barycentric (Warped)
    bc = reg.project(pts, ['g_gcd', 'c_ell', 'c_oct', b_oct])

    # Return raw data to minimize serialization overhead
    return (np.ascontiguousarray(bc.coords, dtype=np.float64),
            np.ascontiguousarray(bc.components))


def _project_gcd_to_boct_parallel(ll_array: np.ndarray, workers=0, chunk=8_000, warp_file=None, accuracy=1e-9):
    """Orchestrates the parallel map"""
    # Auto-detect workers if not specified
    if workers <= 0:
        import os
        cpu_count = os.cpu_count() or 2
        # Don't spin up 16 cores for 100 points. Clamp workers based on chunk size.
        needed_workers = int(np.ceil(len(ll_array) / chunk))
        workers = min(cpu_count, max(1, needed_workers))

    # Split data
    chunks = [np.ascontiguousarray(ll_array[i:i + chunk])
              for i in range(0, len(ll_array), chunk)]

    # Freeze the warp_file argument so we can just map over chunks
    worker_func = partial(_worker_project_gcd_to_bary, warp_file=warp_file, accuracy=accuracy)

    out_coords, out_comps = [], []

    with ProcessPoolExecutor(max_workers=workers) as ex:
        try:
            # ex.map guarantees results come back in the same order as inputs
            for xy, cmp in ex.map(worker_func, chunks):
                out_coords.append(xy)
                out_comps.append(cmp)
        finally:
            ex.shutdown(cancel_futures=True)

    if not out_coords:
        return np.empty((0, 2)), np.empty((0, 3))

    return np.vstack(out_coords), np.vstack(out_comps)


# --- THE PROJECTION CLASS ---

class GCDBary(Projection):
    """
    Parallel-Aware Direct Hop: g_gcd -> b_oct

    Automatically switches between in-process serial execution (for small queries)
    and ProcessPool execution (for heavy batch processing).
    """

    def __init__(self, registrar, *,
                 workers: int | None = None,
                 chunk: int = 8_000,
                 threshold: int = 8_000):

        super().__init__(registrar, 'gcd_bry', 'g_gcd', 'b_oct')
        self.reg = registrar
        self.workers = workers
        self.chunk = int(chunk)
        self.threshold = int(threshold)

        # Cache references to domains
        self.b_oct = registrar.domain('b_oct')

        # Grab the filename from the CURRENT registrar's b_oct
        # This ensures the workers load the same L4 Gold file that the main process has.
        self.warp_file = getattr(self.b_oct.warp, 'file_name', None)

        # Set accuracy on the underlying iterative solver (Octant->Ellipsoid)
        self.accuracy = registrar.projection('oct_ell').accuracy

    def set_parallel(self, *, workers=None, chunk=None, threshold=None):
        """Set parallel frame"""
        if workers is not None: self.workers = workers
        if chunk is not None: self.chunk = int(chunk)
        if threshold is not None: self.threshold = int(threshold)
        return self

    def forward(self, arr: Points) -> Points:
        coords = arr.coords
        n = 0 if coords is None else len(coords)

        # 1. Edge Case: Empty
        if n == 0:
            return Points(np.empty((0, 2)), self.fwd_cs,
                          components=np.empty((0, 3), dtype=np.int8),
                          samples=arr.samples)

        # 2. Small Batch: Serial (In-Process)
        # Avoids IPC overhead for single-point lookups or small arrays
        if n < self.threshold:
            return self.reg.project(arr, ['g_gcd', 'c_ell', 'c_oct', 'b_oct'])

        # 3. Large Batch: Parallel
        # Pass self.warp_file explicitly so workers can load the grid
        xy, cmp = _project_gcd_to_boct_parallel(
            coords,
            workers=self.workers or 0,
            chunk=self.chunk,
            warp_file=self.warp_file,
            accuracy=self.accuracy
        )
        return Points(xy, self.fwd_cs, components=cmp, samples=arr.samples)

    def backward(self, arr: Points) -> Points:
        # Backward path (b_oct -> g_gcd) is usually fast enough in serial,
        # or you can implement a similar parallel strategy if needed.
        return self.reg.project(arr, ['b_oct', 'c_oct', 'c_ell', 'g_gcd'])
