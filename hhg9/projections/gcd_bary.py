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
    _ = reg.projection('ell_gcd')  # EllipsoidGCD
    ak = reg.projection('oct_ell')

    return reg, g_gcd, c_ell, c_oct, b_oct, ak


def _worker_project_gcd_to_bary(chunk_ll: np.ndarray, accuracy: float):
    """Rebuild lightweight stack per process to avoid pickle issues"""
    reg, g_gcd, c_ell, c_oct, b_oct, ak = _init_stack()
    ak.set_accuracy(accuracy)
    chunk_ll = np.ascontiguousarray(chunk_ll).copy()
    pts = Points(chunk_ll, g_gcd)
    bc = reg.project(pts, [g_gcd, c_ell, c_oct, b_oct])  # current path; robust but rootfindy
    # Return 2D coords + components (uint8) for minimal IPC
    return (np.ascontiguousarray(bc.coords).copy(),
            np.ascontiguousarray(bc.components).copy())


def _project_gcd_to_boct_parallel(ll_array: np.ndarray, accuracy: float, workers=0, chunk=8_000):
    """Paralleled work"""
    if workers <= 0:
        import os
        workers = min(os.cpu_count() or 2, max(1, int(np.ceil(len(ll_array) / chunk))))

    # Build chunk list in input order
    # build contiguous, detached chunks
    chunks = [np.ascontiguousarray(ll_array[i:i+chunk]).copy()
              for i in range(0, len(ll_array), chunk)]
    # chunks = [ll_array[i:i + chunk] for i in range(0, len(ll_array), chunk)]

    out_coords, out_comps = [], []
    from itertools import repeat
    # ctx = mp.get_context("forkserver")  # or "spawn"
    with ProcessPoolExecutor(max_workers=workers) as ex:
        # executor.map preserves input order
        try:
            for xy, cmp in ex.map(_worker_project_gcd_to_bary, chunks, repeat(accuracy)):
                out_coords.append(xy)
                out_comps.append(cmp)
        finally:
            # avoid leaked semaphores on interrupts
            ex.shutdown(cancel_futures=True)
    return np.vstack(out_coords), np.vstack(out_comps)

# def _init_stack():
#     reg = Registrar()
#     g_gcd = GeneralGCD(reg)
#     c_ell = EllipsoidCartesian(reg)
#     c_oct = OctahedralCartesian(reg)
#     b_oct = OctahedralBarycentric(reg, c_oct)
#     EllipsoidGCD(reg)
#     ak = AKOctahedralEllipsoid(reg)
#     return reg, g_gcd, c_ell, c_oct, b_oct, ak
#
#
# def _worker_project_gcd_to_bary(chunk_ll: np.ndarray, accuracy: float):
#     # Rebuild lightweight stack per process to avoid pickle issues
#     reg, g_gcd, c_ell, c_oct, b_oct, ak = _init_stack()
#     ak.set_accuracy(accuracy)
#     pts = Points(chunk_ll, g_gcd)
#     bc = reg.project(pts, [g_gcd, c_ell, c_oct, b_oct])  # current path; robust but rootfindy
#     # Return 2D coords + components (uint8) for minimal IPC
#     return bc.coords, bc.components
#
#
# def project_gcd_to_boct_parallel(ll_array: np.ndarray, accuracy: float, workers=0, chunk=8_000):
#     if workers <= 0:
#         # sensible default: min(physical cores, chunks)
#         import os
#         workers = min(os.cpu_count() or 2, max(1, int(np.ceil(len(ll_array)/chunk))))
#     futures, out_coords, out_comps = [], [], []
#     with ProcessPoolExecutor(max_workers=workers) as ex:
#         for i in range(0, len(ll_array), chunk):
#             ch = ll_array[i:i+chunk]
#             futures.append(ex.submit(_worker_project_gcd_to_bary, ch, accuracy))
#         for fut in as_completed(futures):
#             xy, cmp = fut.result()
#             out_coords.append(xy)
#             out_comps.append(cmp)
#     return np.vstack(out_coords), np.vstack(out_comps)


# === Direct g_gcd → b_oct hop with parallel fallback EllipsoidGCDRad ===
class GCDBary(Projection):
    """Direct g_gcd → b_oct projection.

    - For small batches (n < threshold), falls back to the standard chain
      [g_gcd → c_ell → c_oct → b_oct] in-process.
    - For large batches, uses a process pool to parallelise the chain and
      returns b_oct Points with components.

    Tune with set_parallel(workers, chunk, threshold) and set_accuracy(acc).
    Registering this class provides a direct hop in the registrar so callers
    can simply use reg.project(pts, [g_gcd, b_oct]).
    """
    def __init__(self, registrar, *, workers: int | None = None, chunk: int = 8_000, threshold: int = 8_000, accuracy: float = 1e-10):
        super().__init__(registrar, 'gcd_bry', 'g_gcd', 'b_oct')
        from hhg9.domains import GeneralGCD, OctahedralCartesian, OctahedralBarycentric, EllipsoidCartesian
        from hhg9.projections import EllipsoidGCD, AKOctahedralEllipsoid
        self.reg = registrar

        # local knobs
        self.workers = workers
        self.chunk = int(chunk)
        self.threshold = int(threshold)
        self.accuracy = float(accuracy)
        # lightweight stack for fallback path and backward()
        self.g_gcd = registrar.domain('g_gcd')
        self.c_ell = registrar.domain('c_ell')
        self.c_oct = registrar.domain('c_oct')
        self.b_oct = registrar.domain('b_oct')
        self.ake = registrar.projection('oct_ell')


    # Convenience setters
    def set_parallel(self, *, workers: int | None = None, chunk: int | None = None, threshold: int | None = None):
        """Set the parallelisation knobs"""
        if workers is not None:
            self.workers = workers
        if chunk is not None:
            self.chunk = int(chunk)
        if threshold is not None:
            self.threshold = int(threshold)
        return self

    def set_accuracy(self, meters: float):
        """set the accuracy and return the levels indicator"""
        idx = np.searchsorted(self.ake.diameters[::-1], meters, side='right')
        self.accuracy = len(self.ake.diameters) - idx
        return self.accuracy

    # Forward: g_gcd → b_oct
    def forward(self, arr: Points) -> Points:
        coords = arr.coords
        n = 0 if coords is None else int(coords.shape[0])
        if n == 0:
            return Points(np.empty((0, 2), dtype=np.float64), self.fwd_cs, components=np.empty((0, 3), dtype=np.int8), samples=arr.samples)
        if n < self.threshold:
            # small: run the normal chain in-process
            bc = self.reg.project(arr, [self.g_gcd, self.c_ell, self.c_oct, self.b_oct])
            return bc
        xy, cmp = _project_gcd_to_boct_parallel(coords, accuracy=self.accuracy, workers=self.workers or 0, chunk=self.chunk)
        return Points(xy, self.fwd_cs, components=cmp, samples=arr.samples)

    # Backward: b_oct → g_gcd (serial path; direct hop registered by this class suffices)
    def backward(self, arr: Points) -> Points:
        return self.reg.project(arr, ['b_oct', 'c_oct', 'c_ell', 'g_gcd'])

