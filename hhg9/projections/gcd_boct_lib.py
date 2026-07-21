# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
GcdBoctLib: direct g_gcd ↔ b_oct via the libhex9 backend.

Registered by OctahedralBarycentric AHEAD of the g_gcd→b_raw→b_oct bridge when
libhex9 loads (a direct projection beats a bridge in Registrar._check_chain), so
the lib becomes the default engine for the common hop. The lib owns the whole
"AK + [warp]" process in C (one OpenMP, GIL-released call per direction), and is
verified bit-identical to the pure-Python pipeline.

Engine orthogonality (blue/green + regression): when disabled at runtime via
b_oct.no_lib(), forward/backward fall through to the EXPLICIT bridge chain
['g_gcd','b_raw','b_oct'] — the canonical Python reference — rather than this
resolver, so the two engines stay independently selectable and comparable.

Warp on/off is b_oct.warp_enabled, read live and passed to BOTH directions, so a
forward/inverse pair can never desync (the do/undo hazard: a warp-ON b_oct and a
warp-OFF BRAW are different numbers for the same point).

Axis order: g_gcd is [lat, lon]; the libhex9 ABI is (lon, lat) — one column swap
here at the boundary.
"""
import numpy as np
from hhg9 import Points
from hhg9.base.projection import Projection


class GcdBoctLib(Projection):

    def __init__(self, registrar):
        super().__init__(registrar, 'gcd_bct_lib', 'g_gcd', 'b_oct')
        self.reg = registrar
        self.b_oct = registrar.domain('b_oct')

    def _use_lib(self, n: int) -> bool:
        bo = self.b_oct
        return bool(n) and bo.lib_enabled and bo._lib is not None

    def forward(self, arr: Points) -> Points:
        """g_gcd [lat,lon] → b_oct (cx,cy,oid). Lib, else the bridge reference."""
        coords = arr.coords
        n = 0 if coords is None else len(coords)
        if not self._use_lib(n):
            if n == 0:
                return Points(np.empty((0, 2)), self.fwd_cs,
                              oid=np.empty(0, dtype=np.uint8), samples=arr.samples)
            return self.reg.project(arr, ['g_gcd', 'b_raw', 'b_oct'])
        lon = np.ascontiguousarray(coords[:, 1])   # [lat,lon] → (lon,lat)
        lat = np.ascontiguousarray(coords[:, 0])
        cx, cy, oid = self.b_oct._lib.project_many(
            lon, lat, use_warp=self.b_oct.warp_enabled, via_sphere=True)
        return Points(np.column_stack([cx, cy]), self.fwd_cs,
                      oid=oid.astype(np.uint8), samples=arr.samples)

    def backward(self, arr: Points) -> Points:
        """b_oct (cx,cy,oid) → g_gcd [lat,lon]. Lib, else the bridge reference."""
        coords = arr.coords
        n = 0 if coords is None else len(coords)
        if not self._use_lib(n):
            if n == 0:
                return Points(np.empty((0, 2)), self.rev_cs, samples=arr.samples)
            return self.reg.project(arr, ['b_oct', 'b_raw', 'g_gcd'])
        cx = np.ascontiguousarray(coords[:, 0])
        cy = np.ascontiguousarray(coords[:, 1])
        oid = np.ascontiguousarray(np.asarray(arr.oid), dtype=np.int32)
        lon, lat = self.b_oct._lib.unproject_many(
            cx, cy, oid, use_warp=self.b_oct.warp_enabled, via_sphere=True)
        return Points(np.column_stack([lat, lon]), self.rev_cs, samples=arr.samples)
