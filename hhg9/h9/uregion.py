# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Experimental
uregion.py differs from region.py by method - using lattice+residual methods.
This is currently unsupported as it lacks the additional controls
offered by xy_regions, etc.
"""
from dataclasses import dataclass
from typing import Literal
import numpy as np
from numpy.typing import NDArray
from hhg9.h9.region import H9Context, H9CTX, StepEvent


@dataclass(slots=True)
class StepEventUV(StepEvent):
    space: Literal['uv'] = 'uv'


def xyu_regions_iter(xy, mode=None, depth=36, ctx: H9Context = H9CTX):
    """Generator mirroring xyu_regions but yielding per-layer StepEventUV.
    Yields two events per layer (pre/post) and returns the final addresses at StopIteration.
    """
    import hhg9.h9.classifier as clf
    h9k, h9cl, h9c, cr = ctx.k, ctx.cl, ctx.c, ctx.r
    invalid_region = cr.invalid_region
    proto_up, proto_dn = cr.proto_up, cr.proto_dn
    r3 = h9k.radical.R3
    r6 = np.sqrt(6.0)
    ẋ0 = xy[:, 0] * r3
    xu = ẋ0 * r6
    yv = 3.0 * xy[:, 1] * r6
    xu = snap_int(xu, k=3.5)
    yv = snap_int(yv, k=1.5)
    off_uv_f = h9c.off_uv.astype(np.float64, copy=False)
    addresses = np.full((xy.shape[0], depth + 2), invalid_region, dtype=np.uint8)
    addresses[:, 0] = np.where(mode == 1, proto_up, proto_dn)
    for i in range(depth + 1):
        cid = clf.classify_cell_uv(xu, yv)
        bad = ~cr.is_in[cid]
        yield StepEventUV(i=i, phase='pre', addresses=addresses, cid=cid, bad=bad, y=yv, ẋ=xu, pmo=None)
        if np.any(bad):
            # Pass 1: snap V
            yv[bad] = np.rint(yv[bad]).astype(yv.dtype, copy=False)
            cid_bad = clf.classify_cell_uv(xu[bad], yv[bad])
            still_bad = ~cr.is_in[cid_bad]
            # Pass 2: snap U
            if np.any(still_bad):
                # idx_sb = np.nonzero(still_bad)[0]
                xu_sub = xu[bad][still_bad]
                xu_sub = np.rint(xu_sub).astype(xu.dtype, copy=False)
                cid_bad2 = clf.classify_cell_uv(xu_sub, yv[bad][still_bad])
                cid_bad[still_bad] = cid_bad2
                xu[bad][still_bad] = xu_sub
                still_bad = ~cr.is_in[cid_bad]
            # Pass 3: metric fallback
            if np.any(still_bad):
                # Final fallback: choose the closest child in the CURRENT supercell (UV, float64)
                pm_bd = h9c.mode[addresses[bad, i]]
                pm_sb = pm_bd[still_bad]
                cand_mat = np.stack([cr.downs, cr.ups], axis=0)
                cand_sb = cand_mat[pm_sb]
                off_cand = off_uv_f[cand_sb]
                uv_sb = np.stack([xu[bad][still_bad], yv[bad][still_bad]], axis=1)
                d = off_cand - uv_sb[:, None, :]
                d2 = d[..., 0]*d[..., 0] + d[..., 1]*d[..., 1]
                pick = np.argmin(d2, axis=1)
                chosen = cand_sb[np.arange(pick.size), pick]
                cid_bad[np.nonzero(still_bad)[0]] = chosen
                still_bad = np.zeros_like(still_bad, dtype=bool)
            # Merge
            cid[bad] = cid_bad
            bad_next = np.zeros_like(bad)
            bad_next[bad] = still_bad
            bad = bad_next

        good = ~bad
        if np.any(good):
            g_id = cid[good]
            uvo_g = off_uv_f[g_id]
            xu[good] = 3.0 * xu[good] - 3.0 * uvo_g[:, 0]
            yv[good] = 3.0 * yv[good] - 3.0 * uvo_g[:, 1]
            addresses[good, i + 1] = g_id

        yield StepEventUV(i=i, phase='post', addresses=addresses, cid=cid, bad=bad, y=yv, ẋ=xu)
    return addresses


def snap_int(z, k: float = 1.0, ulp_floor: float | None = None):
    """Snap values that are within ~k ULPs of an integer to that integer."""
    zi = np.rint(z)
    diff = np.abs(z - zi)
    # per-element ULP at the integer (distance to next float)
    ulp = np.abs(np.nextafter(zi, np.inf) - zi)
    if ulp_floor is None:
        ulp_floor = np.finfo(z.dtype).eps  # ~2.22e-16 for float64
    ulp = np.maximum(ulp, ulp_floor)
    max_diff = k * ulp
    mask = diff <= max_diff
    return np.where(mask, zi, z)


def xyu_regions(xy, mode=None, depth=36, ctx: H9Context = None) -> NDArray[np.uint8]:
    """Calculate regions via UV array - more integer math, mainly handling residuals"""
    # Use the iterator but materialize the result for compatibility.
    it = xyu_regions_iter(xy, mode=mode, depth=depth, ctx=ctx)
    addresses = None
    for ev in it:
        pass
    addresses = ev.addresses if 'ev' in locals() else None
    return addresses
