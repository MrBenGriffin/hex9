# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
H9 Polygon Generation.

This module generates the geometric shapes (polygons) associated with H9 grid cells.
It uses Look-Up Tables (LUTs) to define the vertices of triangles, half-hexagons,
and full hexagons relative to a cell center.

**Key Features:**

* **Shape LUTs:** Pre-calculated vertex offsets for all orientations (Mode 0/1, C2 0-2).
* **Mesh Generation:** Creating triangle meshes (`tri_mesh`) for surface plots.
* **Hex Binning:** Aggregating points into specific resolution layers (`hex_layer`).
* **Boundary Handling:** Complex logic to "stitch" hexagons that straddle the edges of the octahedron faces.


**Coordinate System:**
Shapes are defined using offsets of :math:`(U, 3V)` relative to the cell center.
All polygons are defined in **Clockwise** order.
"""

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, List, Optional, Callable, Protocol

from hhg9.h9 import H9C, H9K
from hhg9.h9.addressing import reg_hex_digits, hex_digits_reg, HEX_LUTS, TailStyle, hex_layer, tail_unpack_reversible, \
    hex_key
from hhg9.h9.classifier import location
from hhg9.h9.protocols import H9ConstLike, H9PolygonLike


@dataclass(frozen=True, slots=True)
class H9Polygon:
    """
    Immutable container for Polygon LUTs.

    Attributes:
        hh (NDArray[float64]): Half-hex vertices. Shape (2, 3, 4, 2) -> [Mode, C2, Vertices, XY].
        hx (NDArray[float64]): Full-hex vertices. Shape (2, 3, 6, 2).
        tx (NDArray[float64]): Cell triangle vertices. Shape (2, 3, 3, 3, 2).
        se (NDArray[float64]): Supercell edge points. Shape (2, 9, 2).
        sv (NDArray[float64]): Supercell vertices. Shape (2, 3, 2).
        gd (NDArray[float64]): Unshared points of a cell excluding (0,0).
    """
    hh: NDArray[np.float64]
    hx: NDArray[np.float64]
    tx: NDArray[np.float64]
    se: NDArray[np.float64]
    sv: NDArray[np.float64]
    gd: NDArray[np.float64]


def _h9_polygon(h9k: Optional[H9ConstLike] = None) -> H9Polygon:
    """
    Factory to build the H9Polygon singleton.

    Defines the relative vertices for all standard grid shapes (clockwise order).
    """
    pts = {
        # Clockwise.
        (0, 0): [  # c2 half-hexagons mode 0
            [(3, 1), (2, 0), (0, 0), (-1, 1)],
            [(0, -2), (-1, -1), (0, 0), (2, 0)],
            [(-3, 1), (-1, 1), (0, 0), (-1, -1)]
        ],
        (0, 1): [  # c2 half-hexagons mode 1
            [(-1, -1), (0, 0), (2, 0), (3, -1)],
            [(-1, 1), (0, 0), (-1, -1), (-3, -1)],
            [(2, 0), (0, 0), (-1, 1), (0, 2)]
        ],
        (1, 0): [  # c2 hexagons mode 0;  final 2 pts in opp. mode.
            # If exterior, they are idx 2,1 (eg (0,0) (2,0)).
            [(3, 1), (2, 0), (0, 0), (-1, 1), (0, 2), (2, 2)],        # (0, 2), (2, 2)
            [(0, -2), (-1, -1), (0, 0), (2, 0), (3, -1), (2, -2)],    # (3, -1), (2, -2)
            [(-3, 1), (-1, 1), (0, 0), (-1, -1), (-3, -1), (-4, 0)]   # (-3, -1), (-4, 0)
        ],
        (1, 1): [  # c2 hexagons mode 1;  final 2 pts in opp. mode - exterior.
            [(-1, -1), (0, 0), (2, 0), (3, -1), (2, -2), (0, -2)],    # (2, -2), (0, -2)
            [(-1, 1), (0, 0), (-1, -1), (-3, -1), (-4, 0), (-3, 1)],  # (-4, 0), (-3, 1)
            [(2, 0), (0, 0), (-1, 1), (0, 2), (2, 2), (3, 1)]         # (2, 2), (3, 1)
        ],
        (2, 0): [  # region triangles mode 0
            [  # 0x26, 0x2a, 0x2b: c0 VΛV
                [(0, 0), (-1, 1), (1, 1)],  # 26
                [(1, 1), (2, 0), (0, 0)],  # 2a
                [(2, 0), (1, 1), (3, 1)],  # 2b
            ], [  # 0x3a, 0x39, 0x49:   c1 VΛV
                [(1, -1), (0, 0), (2, 0)],  # 3a
                [(0, 0), (1, -1), (-1, -1)],  # 39
                [(0, -2), (-1, -1), (1, -1)],  # 49
            ], [  # 0x35, 0x25, 0x21:   c2 VΛV
                [(-1, -1), (-2, 0), (0, 0)],  # 35
                [(-1, 1), (0, 0), (-2, 0)],  # 25
                [(-2, 0), (-3, 1), (-1, 1)],  # 21
            ]
        ],
        (2, 1): [  # region triangles mode 1
            [  # 0x39, 0x3a, 0x3e: c0  ΛVΛ
                [(0, 0), (1, -1), (-1, -1)],  # 39
                [(1, -1), (0, 0), (2, 0)],  # 3a
                [(2, 0), (3, -1), (1, -1)],  # 3e
            ], [  # 0x25, 0x35, 0x34: c1 ΛVΛ
                [(-1, 1), (0, 0), (-2, 0)],  # 25
                [(-1, -1), (-2, 0), (0, 0)],  # 35
                [(-2, 0), (-1, -1), (-3, -1)],  # 34
            ], [  # 0x2a, 0x26, 0x16: c2 ΛVΛ
                [(1, 1), (2, 0), (0, 0)],  # 2a
                [(0, 0), (-1, 1), (1, 1)],  # 26
                [(0, 2), (1, 1), (-1, 1)],  # 16
            ]
        ],
        (3, 0): [  # clockwise, c0,c1,c2 - edges of super-region
            [
                (-3, 1), (-1, 1), (1, 1),
                (3, 1), (2, 0), (1, -1),
                (0, -2), (-1, -1), (-2, 0),
            ]
        ],
        (3, 1): [  # clockwise, c0,c1,c2 - edges of super-region
            [
                (3, -1), (1, 1), (-1, -1),
                (-3, -1), (-2, 0), (-1, 1),
                (0, 2), (1, 1), (2, 0),
            ]
        ],
        (4, 0): [  # clockwise, c0,c1,c2 - vertexes of super-region
            [  # =  <-c0->  <-c1->   <-c2->
                (-3, 1), (3, 1), (0, -2),  # (-3, 1)
            ]
        ],
        (4, 1): [  # clockwise, c0,c1,c2 - vertexes of super-region
            [  # =  <-c0->    <-c1->  <-c2->
                (3, -1), (-3, -1), (0, 2)
            ]
        ],
        # Unshared points of cell excluding (0,0) use only on one mode
        # The other mode will be just the (0,0)
        (5, 0): [[(2, 0), (1, -1), (-1, -1), (-2, 0), (-1, 1), (1, 1)]]
    }

    if h9k is None:
        from hhg9.h9.constants import H9K
        h9k = H9K
    uv = np.array([h9k.lattice.U, 3 * h9k.lattice.V])
    hh = np.zeros((2, 3, 4, 2), dtype=np.float64)
    hx = np.zeros((2, 3, 6, 2), dtype=np.float64)
    tx = np.zeros((2, 3, 3, 3, 2), dtype=np.float64)
    te = np.zeros((2, 9, 2), dtype=np.float64)
    sr = np.zeros((2, 3, 2), dtype=np.float64)
    gd = np.zeros((6, 2), dtype=np.float64)
    for (kind, mode), c2s in pts.items():
        for c2, poly in enumerate(c2s):
            arr = np.asarray(poly, dtype=np.float64) * uv
            match kind:
                case 0:
                    hh[mode, c2] = arr
                case 1:
                    hx[mode, c2] = arr
                case 2:
                    tx[mode, c2] = arr
                case 3:
                    te[mode] = arr
                case 4:
                    sr[mode] = arr
                case 5:
                    gd = arr
    return H9Polygon(hh=hh, hx=hx, tx=tx, se=te, sv=sr, gd=gd)


H9P = _h9_polygon()


def region_grid(levels: int = 3, mode: int = 0, h9p: H9Polygon = H9P) -> List[Tuple]:
    """
    Generate a grid of points recursively.

    Returns:
        List of [address_path, current_mode, origin_xy, scale].
    """
    from hhg9.h9 import H9C, H9R
    h9c, h9r = H9C, H9R
    modes = [h9r.downs, h9r.ups]
    kids = modes[mode]
    queue = [(k, h9c.mode[k], h9c.off_xy[k], 1.0 / 3.0) for k in kids]

    for depth in range(levels):
        next_q = []
        for path, mode, origin, scale in queue:
            kids = modes[mode]  # shape (9,) indices
            for k in kids:
                mo_k = h9c.mode[k]
                off_k = h9c.off_xy[k]
                path_k = np.append(path, k)
                origin_k = origin + off_k * scale
                scale_k = scale / 3.0
                next_q.append((path_k, mo_k, origin_k, scale_k))
        queue = next_q
    return queue


def tri_grid(levels: int = 5, mode: int = 0, h9p: H9Polygon = H9P) -> NDArray[np.float64]:
    """
    Generate all triangle centroids for an octant at a given depth.

    Returns:
        NDArray: Shape (layer, 3, 2) array of triangle vertices.
    """
    queue = region_grid(levels, mode, h9p)
    pts = np.empty((len(queue), 3, 2), dtype=np.float64)
    for i, (path, mode, origin, scale) in enumerate(queue):
        pts[i] = origin + h9p.sv[mode] * scale
    return pts


def _unique_rows_tol(xy: np.ndarray, tol: float = 1e-12):
    """Deduplicate 2D points by absolute tolerance `tol`.

    Uses power-of-two quantization (via `np.ldexp`) to avoid large-magnitude division.
    Returns original representatives (first occurrence) plus an inverse map.

    Args:
        xy: (N, 2) float64 points.
        tol: absolute tolerance in the same units as xy.

    Returns:
        verts: (V, 2) representative original values (one per bucket)
        inv: (N,) indices mapping each input row -> bucket index
    """
    tol = float(tol)
    if tol <= 0:
        raise ValueError("tol must be > 0")

    # Choose k such that 2**(-k) <= tol (quantization step ~ 2**(-k)).
    k = int(np.ceil(-np.log2(tol)))
    key = np.rint(np.ldexp(xy, k)).astype(np.int64)

    _, idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    verts = xy[idx]  # representative original (first occurrence)
    return verts, inv


def tri_mesh(levels: int = 5, mode: int = 0, h9p: H9Polygon = H9P):
    """
    Return unique vertices and edges for the triangular mesh at a given level.
    Useful for creating Matplotlib Triangulations.

    Args:
        levels: Subdivision depth.
        mode: Root mode (0 or 1).
        h9p: Polygon LUT.

    Returns:
        tuple: (verts, edges, tris)
            verts: (V, 2) Unique vertex coordinates.
            edges: (E, 2) Edge indices into verts.
            tris: (T, 3) Triangle indices into verts.
    """
    tris_xy = tri_grid(levels=levels, mode=mode, h9p=h9p)  # (T, 3, 2)
    num_tris = tris_xy.shape[0]
    # Flatten all triangle vertices and deduplicate
    flat = tris_xy.reshape(-1, 2)  # (T*3, 2)
    # By layer 7, true vertex spacing is ~O(1/m) (~1.5e-4), so tol can be
    # relaxed substantially without collapsing genuine neighbours. A tol of 1e-10
    # should still be far below true spacing, but will merge the rare epsilon-drift
    # duplicates that show up at very high layers.
    tol = 1e-10 if levels >= 6 else 1e-12
    verts, inv = _unique_rows_tol(flat, tol=tol)

    # Map each triangle to indices into the unique vertex array
    tris = inv.reshape(num_tris, 3)

    # Hard-stop: ensure verts contains no exact duplicates (should usually be a no-op)
    verts_u, inv_u = np.unique(verts, axis=0, return_inverse=True)
    if verts_u.shape[0] != verts.shape[0]:
        tris = inv_u[tris]
        verts = verts_u

    # Build undirected edge list from triangle connectivity
    e01 = tris[:, [0, 1]]
    e12 = tris[:, [1, 2]]
    e20 = tris[:, [2, 0]]
    edges_all = np.concatenate([e01, e12, e20], axis=0)

    # Sort each edge's endpoints so that (i, j) and (j, i) dedupe
    edges_sorted = np.sort(edges_all, axis=1)
    edges = np.unique(edges_sorted, axis=0)
    return verts, edges, tris


def enmesh(pts, levels: int = 35, shape=None):
    """
    Builds the half-hex polygons for a batch of points up to `levels` depth.

    Walks the hierarchy for each point and generates the geometry for every layer.

    Returns:
        tuple: (uniques, refs)
            uniques: (U, 4, 2) Unique polygons in global coordinates.
            refs: (layer, depth) Indices into `uniques` for each input point/layer.
    """
    from hhg9.h9 import H9C
    from hhg9.h9.region import H9R, xy_regions_iter

    # Modes per point
    oc, mo = pts.cm()
    coords = pts.coords
    num_pts = len(pts)
    depth = levels + 1

    # Offsets and shapes (float64)
    offs = H9C.off_xy.astype(np.float64, copy=False)  # (R,2)
    hh = H9P.hh  # (2,3,4,2)

    # Accumulators
    parent_xy = np.zeros((num_pts, 2), dtype=np.float64)
    scale = 1.0

    # We'll collect per-layer polygons flattened as (layer*D, 4, 2)
    polys = np.empty((num_pts * depth, 4, 2), dtype=np.float64)
    flat_idx = np.empty((num_pts, depth), dtype=np.int64)  # indices into `polys`

    for ev in xy_regions_iter(coords, mode=mo, depth=levels):
        if ev.phase != 'pre':
            continue
        i = ev.i  # 0..D-1

        # Child c2 for each row under the parent mode
        c2 = H9R.mcc2[ev.pmo, ev.cid]  # (layer,)
        hh_shape = hh[ev.pmo, c2]  # pmo

        # Translate/scale each triangle to global coords
        polygon = parent_xy[:, None, :] + hh_shape * scale  # (layer,4,2)

        # Store flattened by layer, keeping a stable mapping back to rows
        start = i * num_pts
        polys[start:start + num_pts] = polygon
        flat_idx[:, i] = np.arange(start, start + num_pts, dtype=np.int64)

        # Step parent origin for next layer
        parent_xy = parent_xy + offs[ev.cid] * scale
        scale /= 3.0

    if num_pts > 1:
        # Deduplicate exactly: reshape to (M, 8) and unique over rows
        mass = num_pts * depth
        flat = polys.reshape(mass, 8)
        uniq, first_idx, inv = np.unique(flat, axis=0, return_index=True, return_inverse=True)
        uniques = uniq.reshape(-1, 4, 2)

        # Map each (address, layer) to its unique polygon index
        refs = inv[flat_idx]
        return uniques, refs
    else:
        return polys, np.array(list[range(num_pts)])


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
    """Given a set of points, find the hex"""
    h_val = hex_layer(pts, layer=layer, tail_style=TailStyle.reversible)
    h_key = hex_key(h_val)
    hex_k, hex_idx, hex_inv = np.unique(h_key, axis=0, return_index=True, return_inverse=True)
    hex_num = hex_k.shape[0]
    hex_v = h_val[hex_idx]
    return hex_num, hex_v, hex_inv, hex_idx


def hex_parents(dom, hex_v, hex_num, layer):
    """Hex parent centroids - return the b_oct points, octants, and remaining scale"""
    hex_xy = np.zeros((hex_num, 2), dtype=float)
    hex_oid, hex_rgn = hex_digits_reg(dom, hex_v)
    scale = 1.0
    for i in range(1, layer + 1):
        hex_xy += H9C.off_xy[hex_rgn[:, i]] * scale
        scale /= 3.0
    return hex_xy, hex_oid, scale


def ctr_from_pars(dom, hex_par, hex_oid, scale, tail):
    """Build hex polygons, return Points (layer)"""
    from hhg9 import Points
    xpm, xc2, xrm, rgn = tail_unpack_reversible(tail)
    hex_all = H9P.hx[xpm, xc2]  # given the parent, and modes, c2 of hexes we want...
    hex_pts = hex_par[:, None, :] + hex_all * scale  # (H,6,2)
    hex_ctr = np.mean(hex_pts, axis=1)
    hx_pts = Points(hex_ctr, dom, components=hex_oid)
    return hx_pts


def hex_from_pars(dom, hex_par, hex_oid, scale, tail):
    """Build hex polygons, return Points (layer*6)"""
    from hhg9 import Points
    from hhg9.h9 import H9O
    xpm, xc2, xrm, rgn = tail_unpack_reversible(tail)
    hex_all = H9P.hx[xpm, xc2]  # given the parent, and modes, c2 of hexes we want...
    hex_pts = hex_par[:, None, :] + hex_all * scale  # (H,6,2)

    # Set the basic octant id for each hexagon
    oc_poly6 = np.repeat(hex_oid[:, None], 6, axis=1)  # (H, 6)
    nbr_oid = H9O.oid_nb[hex_oid, xc2]

    hex_ẋ = H9K.R3 * hex_pts[..., 0].ravel()  # Classifier ẋ
    hex_y = hex_pts[..., 1].ravel()           # Classifier y
    oc_mo = np.repeat(xrm, 6)          # Set the octant mode for each hexagon
    types = location(hex_ẋ, hex_y, oc_mo)
    locs = types.reshape(-1, 6)

    ex4 = locs[:, 4] == 1
    ex5 = locs[:, 5] == 1
    ext_hex = (ex4 & ex5)

    # Stitch exterior vertices across face boundaries
    hex_pts[ext_hex, 4] = hex_pts[ext_hex, 2] * [1, -1]
    hex_pts[ext_hex, 5] = hex_pts[ext_hex, 1] * [1, -1]

    n_oct = nbr_oid[ext_hex]
    oc_poly6[ext_hex, 4] = n_oct
    oc_poly6[ext_hex, 5] = n_oct

    hx_coords = hex_pts.reshape([-1, 2])
    hx_oc = oc_poly6.reshape([-1])
    hx_pts = Points(hx_coords, dom, components=hx_oc)
    return hx_pts


def hex_poly_groups(pts, layers: int = 10):
    """Return hex polygons plus the grouping needed to aggregate arbitrary per-point data.

    Returns:
        tuple: (hx_pts, inv_hex, counts, idx)
            hx_pts: Points of hex polygon vertices (H*6,2) with per-vertex octant components.
            inv_hex: (layer,) mapping each input point -> hex index in [0, H)
            counts: (H,) population count per hex
            idx: (H,) indices of representative points for each hex (as returned by np.unique)

    Notes:
        - This function does NOT aggregate values; callers can compute means/sums/modes/medians etc.
        - `idx` can be used to recover per-hex address metadata from the representative point.
    """

    dom = pts.domain
    if dom.name[1:5] != '_oct':
        raise ValueError('hex_poly_groups requires pts to be in b_oct/n_oct domain')

    # reduce all points to hex ids, preserving reversible (needed for pts construction)
    hex_num, hex_v, hex_inv, hex_idx = hex_reduce(pts, layers)
    hex_xy, hex_oid, scale = hex_parents(dom, hex_v, hex_num, layers)   # (HP,2)
    hx_pts = hex_from_pars(dom, hex_xy, hex_oid, scale, hex_v[:, -1])

    counts = np.bincount(hex_inv, minlength=hex_num).astype(np.int64, copy=False)
    return hx_pts, hex_inv.astype(np.int64, copy=False), counts, hex_idx


def hex_poly_layer(pts, layers: int = 10, reducer: Optional[HexReducer] = None):
    """Return hex polygons and an aggregated value per hex.

    By default, this reproduces the previous behaviour: mean of `pts.samples` per hex.

    Args:
        pts: Input points in b_oct.
        layers: Hex layer.
        reducer: Optional callable to aggregate per-point data into per-hex values.
            Signature: reducer(inv_hex, counts, pts) -> (H,) float array.

    Returns:
        tuple: (hx_pts, values)
            hx_pts: Points of hex polygon vertices (H*6,2)
            values: (H,) aggregated value per hex (float64)
    """
    hx_pts, inv_hex, counts, idx = hex_poly_groups(pts, layers=layers)

    if reducer is None:
        # Default: mean of pts.samples per hex (previous behaviour)
        sum_wt = np.bincount(inv_hex, weights=pts.samples, minlength=counts.shape[0])
        values = np.divide(sum_wt, counts, out=np.zeros_like(sum_wt, dtype=float), where=counts > 0)
    else:
        values = reducer(inv_hex, counts, pts)
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (counts.shape[0],):
            raise ValueError(f"reducer must return shape ({counts.shape[0]},), got {values.shape}")

    return hx_pts, values


def hh_layer(pts, layer: int = 10):
    """
    Calculates Half-Hexagon polygons for a batch of points.

    Returns:
        tuple: (polygon, pops, inv, oc_poly4)
            polygon: (H, 4, 2) Half-hex vertices.
            pops: Population count per half-hex.
            inv: Inverse mapping.
            oc_poly4: Octant IDs for the 4 vertices.
    """
    from hhg9.h9.region import xy_regions
    from hhg9.h9 import H9C, H9R
    num_pts = len(pts)
    if num_pts == 0:
        return

    ocf, mode_f = pts.cm()  # return the octant/mode
    rgx = xy_regions(pts.coords, mode_f, layer)
    poi = rgx[:, -2]
    smo = H9C.mode[poi]  # We will need it's mode (smo=self_mode)
    c2i = rgx[:, -1]
    c2 = H9R.mcc2[smo, c2i]  # Now grab the c2 of the current region - which neighbour?!
    key = np.concatenate([ocf[:, None], rgx[:, 1:layer + 1], c2[:, None]], axis=1)
    uniq, first_idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    pops = np.bincount(inv)
    rgs = rgx[first_idx]  # This gives us the indicative regions.
    oc = ocf[first_idx]  # and the indicative octants for each region.
    c2 = c2[first_idx]
    num_hh = rgs.shape[0]  # and the number of half-hexagons found.
    acc_xy = np.zeros((num_hh, 2), dtype=np.float64)
    scale = 1.0
    for i in range(1, layer + 1):
        acc_xy += (H9C.off_xy[rgs[:, i]] * scale)
        scale /= 3.0
    poi = rgs[:, -2]
    smo = H9C.mode[poi]  # We will need its mode (smo=self_mode)
    polygon = scale * H9P.hh[smo, c2] + acc_xy[:, None]
    oc_poly4 = np.repeat(oc[:, None], 4, axis=1)  # 4 points of half-hex
    return polygon, pops, inv, oc_poly4