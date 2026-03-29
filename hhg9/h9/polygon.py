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
from typing import Tuple, List, Optional

from hhg9.h9 import H9C, H9K
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
    poly_eps = 1 - 1e-16
    pts = {
        # Clockwise.
        (0, 0): [  # c2 half-hexagons net_mode 0
            [(3, 1), (2, 0), (0, 0), (-1, 1)],
            [(0, -2), (-1, -1), (0, 0), (2, 0)],
            [(-3, 1), (-1, 1), (0, 0), (-1, -1)]
        ],
        (0, 1): [  # c2 half-hexagons net_mode 1
            [(-1, -1), (0, 0), (2, 0), (3, -1)],
            [(-1, 1), (0, 0), (-1, -1), (-3, -1)],
            [(2, 0), (0, 0), (-1, 1), (0, 2)]
        ],
        (1, 0): [  # c2 hexagons net_mode 0;  final 2 pts in opp. net_mode.
            # If exterior, they are idx 2,1 (eg (0,0) (2,0)).
            [(3, 1), (2, 0), (0, 0), (-1, 1), (0, 2), (2, 2)],        # (0, 2), (2, 2)
            [(0, -2), (-1, -1), (0, 0), (2, 0), (3, -1), (2, -2)],    # (3, -1), (2, -2)
            [(-3, 1), (-1, 1), (0, 0), (-1, -1), (-3, -1), (-4, 0)]   # (-3, -1), (-4, 0)
        ],
        (1, 1): [  # c2 hexagons net_mode 1;  final 2 pts in opp. net_mode - exterior.
            [(-1, -1), (0, 0), (2, 0), (3, -1), (2, -2), (0, -2)],    # (2, -2), (0, -2)
            [(-1, 1), (0, 0), (-1, -1), (-3, -1), (-4, 0), (-3, 1)],  # (-4, 0), (-3, 1)
            [(2, 0), (0, 0), (-1, 1), (0, 2), (2, 2), (3, 1)]         # (2, 2), (3, 1)
        ],
        (2, 0): [  # region triangles net_mode 0
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
        (2, 1): [  # region triangles net_mode 1
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
        # Unshared points of cell excluding (0,0) use only on one net_mode
        # The other net_mode will be just the (0,0)
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
            bas = np.asarray(poly, dtype=np.float64) * uv
            ctr = np.mean(bas, axis=0)
            arr = ((bas - ctr) * poly_eps) + ctr
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


H9P: H9Polygon = _h9_polygon()


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
        NDArray: Shape (hex_layer, 3, 2) array of triangle vertices.
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
    Return unique vertices and edges for the triangular mesh at a given layer.
    Useful for creating Matplotlib Triangulations.

    Args:
        levels: Subdivision depth.
        mode: Root net_mode (0 or 1).
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
    # By hex_layer 7, true vertex spacing is ~O(1/m) (~1.5e-4), so tol can be
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


# Binning functions have moved to binning.py; re-exported here for backward compatibility.
from hhg9.h9.binning import (
    HexReducer,
    hex_reduce,
    hex_parents,
    ctr_from_pars,
    hex_from_pars,
    hex_poly_groups,
    hex_poly_layer,
    hh_layer,
)