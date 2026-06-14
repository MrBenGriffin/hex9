# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
H9 Hex Binning.

Functions for aggregating points into hexagonal grid cells and building
per-hex polygon geometry.

Moved here from polygon.py; polygon.py re-exports these for backward
compatibility with existing callers.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Protocol

from hhg9.h9 import H9C, H9K, H9R
from hhg9.h9.addressing import hex_layer as _hex_layer, hex_key, hex_digits_reg
from hhg9.h9.tail import TailStyle, tail_unpack_reversible
from hhg9.h9.classifier import location


class HexReducer(Protocol):
    """Callable signature for aggregating per-point data into per-hex values."""

    def __call__(
        self,
        inv_hex: NDArray[np.int64],
        counts: NDArray[np.int64],
        pts,
    ) -> NDArray[np.float64]:
        ...


def hex_reduce(pts, layer):
    """Given a set of points, find the hexes."""
    h_val = _hex_layer(pts, layer=layer, tail_style=TailStyle.reversible)
    h_key = hex_key(h_val)
    hex_k, hex_idx, hex_inv = np.unique(h_key, axis=0, return_index=True, return_inverse=True)
    hex_num = hex_k.shape[0]
    hex_v = h_val[hex_idx]
    return hex_num, hex_v, hex_inv, hex_idx


def hex_parents(dom, hex_v, hex_num, layer=None):
    """Hex parent centroids — return the b_oct points, octants, and remaining scale."""
    if layer is None:
        layer = hex_v.shape[1] - 2
    hex_xy = np.zeros((hex_num, 2), dtype=float)
    hex_oid, hex_rgn = hex_digits_reg(dom, hex_v)
    scale = 1.0
    for i in range(1, layer + 1):
        hex_xy += H9C.off_xy[hex_rgn[:, i]] * scale
        scale /= 3.0
    return hex_xy, hex_oid, scale


def ctr_from_pars(dom, hex_par, hex_oid, scale, tail):
    """Build hex centroids from parent info. Returns Points (one per hex).

    hex_par is already the barycentric centre of each hex (accumulated by
    hex_parents); vertex-averaging is unnecessary and wrong for hexes whose
    vertices 4/5 cross an octant boundary.
    """
    from hhg9 import Points
    from hhg9.h9.polygon import H9P
    xpm, xc2, xrm, rgn = tail_unpack_reversible(tail)
    hex_all = H9P.hx[xpm, xc2]
    hex_pts = hex_par[:, None, :] + hex_all * scale  # (H, 6, 2)
    hex_ctr = np.mean(hex_pts, axis=1)
    return Points(hex_ctr, dom, oid=hex_oid)


def hex_from_pars(dom, hex_par, hex_oid, scale, tail):
    """Build hex polygon vertices from parent info. dom=b_oct Returns Points (hex*6)."""
    from hhg9 import Points
    from hhg9.h9 import H9O
    from hhg9.h9.polygon import H9P
    xpm, xc2, xrm, rgn = tail_unpack_reversible(tail)
    hex_all = H9P.hx[xpm, xc2]
    hex_pts = hex_par[:, None, :] + hex_all * scale  # (H, 6, 2)

    oc_poly6 = np.repeat(hex_oid[:, None], 6, axis=1)
    nbr_oid = H9O.oid_nb[hex_oid, xc2]

    hex_ẋ = H9K.R3 * hex_pts[..., 0].ravel()
    hex_y = hex_pts[..., 1].ravel()
    oc_mo = np.repeat(xrm, 6)
    types = location(hex_ẋ, hex_y, oc_mo)
    locs = types.reshape(-1, 6)

    ex4 = locs[:, 4] == 1
    ex5 = locs[:, 5] == 1
    ext_hex = ex4 & ex5

    hex_pts[ext_hex, 4] = hex_pts[ext_hex, 2] * [1, -1]
    hex_pts[ext_hex, 5] = hex_pts[ext_hex, 1] * [1, -1]

    n_oct = nbr_oid[ext_hex]
    oc_poly6[ext_hex, 4] = n_oct
    oc_poly6[ext_hex, 5] = n_oct

    hx_coords = hex_pts.reshape([-1, 2])
    hx_oc = oc_poly6.reshape([-1])
    return Points(hx_coords, dom, oid=hx_oc)


def hex_poly_groups(pts, layers: int = 10):
    """Return hex polygons plus grouping needed to aggregate arbitrary per-point data.

    Returns:
        tuple: (hx_pts, inv_hex, counts, idx)
            hx_pts: Points of hex polygon vertices (H*6, 2) with per-vertex octant components.
            inv_hex: (N,) mapping each input point -> hex index in [0, H).
            counts: (H,) population count per hex.
            idx: (H,) indices of representative points for each hex.
    """
    dom = pts.domain
    if dom.name[1:5] != '_oct':
        raise ValueError('hex_poly_groups requires pts to be in b_oct/n_oct domain')

    hex_num, hex_v, hex_inv, hex_idx = hex_reduce(pts, layers)
    hex_xy, hex_oid, scale = hex_parents(dom, hex_v, hex_num, layers)
    hx_pts = hex_from_pars(dom, hex_xy, hex_oid, scale, hex_v[:, -1])

    counts = np.bincount(hex_inv, minlength=hex_num).astype(np.int64, copy=False)
    return hx_pts, hex_inv.astype(np.int64, copy=False), counts, hex_idx


def hex_poly_layer(pts, layers: int = 10, reducer: Optional[HexReducer] = None):
    """Return hex polygons and an aggregated value per hex.

    Args:
        pts: Input points in b_oct.
        layers: Hex layer depth.
        reducer: Optional callable to aggregate per-point data into per-hex values.
            Signature: reducer(inv_hex, counts, pts) -> (H,) float array.
            Default: mean of pts.samples per hex.

    Returns:
        tuple: (hx_pts, values)
    """
    hx_pts, inv_hex, counts, idx = hex_poly_groups(pts, layers=layers)

    if reducer is None:
        sum_wt = np.bincount(inv_hex, weights=pts.samples, minlength=counts.shape[0])
        values = np.divide(sum_wt, counts, out=np.zeros_like(sum_wt, dtype=float), where=counts > 0)
    else:
        values = reducer(inv_hex, counts, pts)
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (counts.shape[0],):
            raise ValueError(f"reducer must return shape ({counts.shape[0]},), got {values.shape}")

    return hx_pts, values


def hh_layer(pts, layer: int = 10):
    """Calculate Half-Hexagon polygons for a batch of points.

    Returns:
        tuple: (polygon, pops, inv, oc_poly4)
            polygon: (H, 4, 2) Half-hex vertices.
            pops: Population count per half-hex.
            inv: Inverse mapping.
            oc_poly4: Octant IDs for the 4 vertices.
    """
    from hhg9.h9.region import xy_regions
    from hhg9.h9.polygon import H9P
    num_pts = len(pts)
    if num_pts == 0:
        return

    ocf, mode_f = pts.cm()
    rgx = xy_regions(pts.coords, mode_f, layer)
    poi = rgx[:, -2]
    smo = H9C.mode[poi]
    c2i = rgx[:, -1]
    c2 = H9R.mcc2[smo, c2i]
    key = np.concatenate([ocf[:, None], rgx[:, 1:layer + 1], c2[:, None]], axis=1)
    uniq, first_idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    pops = np.bincount(inv)
    rgs = rgx[first_idx]
    oc = ocf[first_idx]
    c2 = c2[first_idx]
    num_hh = rgs.shape[0]
    acc_xy = np.zeros((num_hh, 2), dtype=np.float64)
    scale = 1.0
    for i in range(1, layer + 1):
        acc_xy += H9C.off_xy[rgs[:, i]] * scale
        scale /= 3.0
    poi = rgs[:, -2]
    smo = H9C.mode[poi]
    polygon = scale * H9P.hh[smo, c2] + acc_xy[:, None]
    oc_poly4 = np.repeat(oc[:, None], 4, axis=1)
    return polygon, pops, inv, oc_poly4
