# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
GCDBary: parallel-aware direct hop g_gcd -> b_oct.

Outside PostgreSQL: uses ProcessPoolExecutor so each worker runs its own Python
interpreter and GIL, giving true CPU parallelism for the compute-heavy
c_ell->c_oct->b_oct chain (including scipy's CT warp solver).

Inside PostgreSQL (plpython3u): process spawning is not permitted, so the large-
batch path falls back to in-process serial execution.  Hex-fill queries are
bounded in size so the performance cost is acceptable.
"""

import numpy as np
from hhg9 import Points
from hhg9.base.projection import Projection
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# Detect PostgreSQL plpython3u environment once at import time.
try:
    import plpy as _plpy        # noqa: F401
    _IN_POSTGRES = True
except ImportError:
    _IN_POSTGRES = False


# --- WORKER (top-level for pickling) ---

def _worker_project_gcd_to_bary(chunk_ll: np.ndarray, warp_file: str = None, accuracy: float = 1e-9,
                                ellipsoid=None):
    """
    Process worker: rebuilds the minimal projection stack, injects the warp
    file, and projects one chunk g_gcd -> b_oct.
    Each worker runs in a separate process with its own GIL.
    ellipsoid is the parent registrar's (a, f, name) — without it a fresh
    Registrar() would silently compute WGS84 whatever the parent was using.
    """
    from hhg9 import Registrar, Points
    reg = Registrar()
    if ellipsoid is not None:
        a, f, name = ellipsoid
        reg.set_ellipsoid(a=a, f=f, name=name)
    cart = 'c_sph'
    _ = reg.domain('g_gcd')
    _ = reg.domain(cart)
    _ = reg.domain('c_oct')
    b_oct = reg.domain('b_oct')
    reg.projection('oct_sph').accuracy = accuracy
    if warp_file:
        b_oct.set_warp(warp_file)
    pts = Points(np.ascontiguousarray(chunk_ll, dtype=np.float64), 'g_gcd')
    bc = reg.project(pts, ['g_gcd', cart, 'c_oct', b_oct])
    return (np.ascontiguousarray(bc.coords, dtype=np.float64),
            np.ascontiguousarray(bc.oid))


def _project_gcd_to_boct_parallel(ll_array: np.ndarray, warp_file, accuracy, workers=0, chunk=8_000,
                                  ellipsoid=None):
    """Orchestrate process-parallel projection over chunks of ll_array."""
    if workers <= 0:
        import os
        cpu_count = os.cpu_count() or 2
        needed_workers = int(np.ceil(len(ll_array) / chunk))
        workers = min(cpu_count, max(1, needed_workers))

    chunks = [np.ascontiguousarray(ll_array[i:i + chunk])
              for i in range(0, len(ll_array), chunk)]

    worker_func = partial(_worker_project_gcd_to_bary, warp_file=warp_file, accuracy=accuracy,
                          ellipsoid=ellipsoid)

    out_coords, out_oids = [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        try:
            for xy, oid in ex.map(worker_func, chunks):
                out_coords.append(xy)
                out_oids.append(oid)
        finally:
            ex.shutdown(cancel_futures=True)

    if not out_coords:
        return np.empty((0, 2)), np.empty(0, dtype=np.uint8)

    return np.vstack(out_coords), np.concatenate(out_oids)


# --- THE PROJECTION CLASS ---

class GCDBary(Projection):
    """
    Parallel-Aware Direct Hop: g_gcd -> b_oct

    Switches between serial execution (small batches or PostgreSQL) and
    ProcessPoolExecutor (large batches outside PostgreSQL).

    NOTE — engine tension (read this if g_gcd↔b_oct seems to bypass GCDBary):
    GCDBary is an OPT-IN direct hop; nothing instantiates it by default, so the
    default resolver for ('g_gcd','b_oct') is either the libhex9 resolver
    GcdBoctLib (registered by OctahedralBarycentric when libhex9 loads) or, with
    no lib, the g_gcd→b_raw→b_oct bridge. If GCDBary IS instantiated alongside
    GcdBoctLib, both register ('g_gcd','b_oct'); Registrar.project picks the
    first inserted, so GcdBoctLib (registered at b_oct construction) wins. Use
    b_oct.no_lib() to fall back to the pure-Python bridge. See
    hhg9/projections/gcd_boct_lib.py and OctahedralBarycentric.
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
        self.b_oct = registrar.domain('b_oct')
        # Cached for process workers (they can't share the parent's objects).
        # Use the cheap warp_file property — reading .warp would force the lazy
        # CT/Newton build (see OctahedralBarycentric.warp).
        self.warp_file = self.b_oct.warp_file
        # The engine always runs on the unit authalic sphere (c_sph), never
        # the source ellipsoid ECEF (c_ell) — Registrar.via_sphere is constant.
        self._cart = 'c_sph'
        self.accuracy = registrar.projection('oct_sph').accuracy
        geo = registrar.ellipsoid
        self._ell = (geo.a, geo.f, registrar.ellipsoid_name)

    def set_parallel(self, *, workers=None, chunk=None, threshold=None):
        """Configure parallel execution."""
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
                          oid=np.empty(0, dtype=np.uint8),
                          samples=arr.samples)

        # 2. Serial: small batch, or always serial inside PostgreSQL
        if n < self.threshold or _IN_POSTGRES:
            return self.reg.project(arr, ['g_gcd', self._cart, 'c_oct', 'b_oct'])

        # 3. Large batch outside PostgreSQL: process pool
        xy, oid = _project_gcd_to_boct_parallel(
            coords,
            warp_file=self.warp_file,
            accuracy=self.accuracy,
            workers=self.workers or 0,
            chunk=self.chunk,
            ellipsoid=self._ell,
        )
        return Points(xy, self.fwd_cs, oid=oid, samples=arr.samples)

    def backward(self, arr: Points) -> Points:
        return self.reg.project(arr, ['b_oct', 'c_oct', self._cart, 'g_gcd'])
