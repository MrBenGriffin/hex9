# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
ak_parallel improves processing of the work intensive c_ell->c_oct
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from hhg9 import Registrar, Points
from hhg9.base.projection import Projection


def _init_stack():
    reg = Registrar()
    g_gcd = reg.domain('g_gcd')
    c_ell = reg.domain('c_ell')
    c_oct = reg.domain('c_oct')
    b_oct = reg.domain('b_oct')
    ake = reg.projection('oct_ell')   #AKOctahedralEllipsoid(reg)
    return reg, g_gcd, c_ell, c_oct, b_oct, ake


def _worker_project_gcd_to_bary(chunk_ll: np.ndarray, accuracy: float):
    # Rebuild lightweight stack per process to avoid pickle issues
    reg, g_gcd, c_ell, c_oct, b_oct, ake = _init_stack()
    ake.set_accuracy(accuracy)
    pts = Points(chunk_ll, g_gcd)
    bc = reg.project(pts, [g_gcd, c_ell, c_oct, b_oct])  # current path; robust but rootfindy
    # Return 2D coords + components (uint8) for minimal IPC
    return bc.coords, bc.components


def project_gcd_to_boct_parallel(ll_array: np.ndarray, accuracy: float, workers=0, chunk=8_000):
    """Project g_gcd to b_oct using parallel workers."""
    if workers <= 0:
        # sensible default: min(physical cores, chunks)
        import os
        workers = min(os.cpu_count() or 2, max(1, int(np.ceil(len(ll_array)/chunk))))
    futures, out_coords, out_comps = [], [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i in range(0, len(ll_array), chunk):
            ch = ll_array[i:i+chunk]
            futures.append(ex.submit(_worker_project_gcd_to_bary, ch, accuracy))
        for fut in as_completed(futures):
            xy, cmp = fut.result()
            out_coords.append(xy)
            out_comps.append(cmp)
    return np.vstack(out_coords), np.vstack(out_comps)


# === Direct g_gcd → b_oct hop with parallel fallback ===
class GCDToBOctParallel(Projection):
    """Direct g_gcd → b_oct projection.

    - For small batches (n < threshold), falls back to the standard chain
      [g_gcd → c_ell → c_oct → b_oct] in-process.
    - For large batches, uses a process pool to parallelise the chain and
      returns b_oct Points with components.

    Tune with set_parallel(workers, chunk, threshold) and set_accuracy(acc).
    Registering this class provides a direct hop in the registrar so callers
    can simply use reg.project(pts, [g_gcd, b_oct]).
    """
    def __init__(self, registrar, *, workers: int | None = None, chunk: int = 8_000, threshold: int = 50_000, accuracy: float = 1e-9):
        super().__init__(registrar, 'g2b_par', 'g_gcd', 'b_oct')
        # local knobs
        self.workers = workers
        self.chunk = int(chunk)
        self.threshold = int(threshold)
        self.accuracy = float(accuracy)
        self.reg = registrar
        # lightweight stack for fallback path and backward()
        self.g_gcd = registrar.domain('g_gcd')
        self.c_ell = registrar.domain('c_ell')
        self.c_oct = registrar.domain('c_oct')
        self.b_oct = registrar.domain('b_oct')
        self.ake = registrar.projection('oct_ell')  # AKOctahedralEllipsoid(reg)


    # Convenience setters
    def set_parallel(self, *, workers: int | None = None, chunk: int | None = None, threshold: int | None = None):
        """Set the parallel environment: workers, chunk, threshold"""
        if workers is not None:
            self.workers = workers
        if chunk is not None:
            self.chunk = int(chunk)
        if threshold is not None:
            self.threshold = int(threshold)
        return self

    def set_accuracy(self, accuracy: float):
        """Set the normal AK accuracy value (in m^2)"""
        self.accuracy = float(accuracy)
        return self

    # Forward: g_gcd → b_oct
    def forward(self, arr: Points) -> Points:
        """g_gcd → b_oct process"""
        coords = arr.coords
        n = 0 if coords is None else int(coords.shape[0])
        if n == 0:
            return Points(np.empty((0, 2), dtype=np.float64), self.fwd_cs, components=np.empty((0, 3), dtype=np.int8), samples=arr.samples)
        if n < self.threshold:
            # small: run the normal chain in-process
            bc = self.reg.project(arr, [self.g_gcd, self.c_ell, self.c_oct, self.b_oct])
            return bc
        # large: parallel
        xy, cmp = project_gcd_to_boct_parallel(coords, accuracy=self.accuracy, workers=self.workers or 0, chunk=self.chunk)
        return Points(xy, self.fwd_cs, components=cmp, samples=arr.samples)

    # Backward: b_oct → g_gcd (serial path; direct hop registered by this class suffices)
    def backward(self, arr: Points) -> Points:
        """b_oct → g_gcd: The normal route is fast and efficient."""
        return self.reg.project(arr, [self.b_oct, self.c_oct, self.c_ell, self.g_gcd])


# Optional: helper to register the parallel hop if not already present
def register_g2b_parallel(registrar, **kwargs):
    """Register the parallel g_gcd→b_oct hop if not present; return the instance."""
    try:
        # if another projection already provides this hop, reuse registrar mapping
        proj = registrar.projection('g_gcd', 'b_oct')
        return proj
    except Exception:
        pass
    return GCDToBOctParallel(registrar, **kwargs)
