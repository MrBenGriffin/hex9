# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
BrawBoct: vectorised b_raw <-> b_oct warp projection.

b_raw carries geometric (unwarped) barycentric coordinates.
b_oct carries authalic (warped) barycentric coordinates.

The only difference between the two is AuthalicWarp.do / undo.
The per-octant net_mode flip (y-sign) is handled via a batch np.where
so the warp interpolator is called once per direction, not 8 times.
"""
import numpy as np
from hhg9 import Points
from hhg9.base.projection import Projection
from hhg9.h9.constants import H9O


class BrawBoct(Projection):
    """
    Vectorised b_raw -> b_oct  (forward)  and  b_oct -> b_raw  (backward).
    """

    def __init__(self, registrar):
        super().__init__(registrar, 'brw_bct', 'b_raw', 'b_oct')
        self.b_oct = registrar.domain('b_oct')

    def _apply(self, xy: np.ndarray, oids: np.ndarray, warp_fn) -> np.ndarray:
        """Apply warp_fn (undo or do) with per-octant net_mode-y-flip, batched."""
        safe_oids = np.clip(oids, 0, 7)  # guard against OID_INVALID (255) on seam vertices
        mf = np.where(H9O.oid_mo[safe_oids] == 0, 1.0, -1.0)   # (N,) ±1
        xy_v = xy.copy()
        xy_v[:, 1] *= mf          # normalise all rows to net_mode-0 convention
        result = warp_fn(xy_v)    # single call, no internal net_mode flip (mo=0)
        result[:, 1] *= mf        # restore

        return result

    def forward(self, arr: Points) -> Points:
        """b_raw (geometric) -> b_oct (authalic): apply warp.undo"""
        # pts = arr.domain.valid(arr)
        warp = self.b_oct.warp
        xy = arr.coords
        if warp is None:
            return Points(xy.copy(), self.fwd_cs, oid=arr.oid, samples=arr.samples)
        result = self._apply(xy, arr.oid, warp.undo)
        return Points(result, self.fwd_cs, oid=arr.oid, samples=arr.samples)

    def backward(self, arr: Points) -> Points:
        """b_oct (authalic) -> b_raw (geometric): apply warp.do"""
        warp = self.b_oct.warp
        xy = arr.coords
        if warp is None:
            return Points(xy.copy(), self.rev_cs, oid=arr.oid, samples=arr.samples)
        result = self._apply(xy, arr.oid, warp.do)
        return Points(result, self.rev_cs, oid=arr.oid, samples=arr.samples)
