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

from hhg9.h9 import H9C, H9K, H9O
from hhg9.h9.classifier import location
from hhg9.h9.protocols import H9ConstLike, H9PolygonLike, BaryLoc


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
        hi (NDArray[float64]): Full-hex UV, including centroids. Shape  (3, 2) -> [C2,xy]
    """
    hh: NDArray[np.float64]
    hx: NDArray[np.float64]
    tx: NDArray[np.float64]
    se: NDArray[np.float64]
    sv: NDArray[np.float64]
    gd: NDArray[np.float64]
    hi: NDArray[np.int16]


def _h9_polygon(h9k: Optional[H9ConstLike] = None) -> H9Polygon:
    """
    Factory to build the H9Polygon singleton.

    Defines the relative vertices for all standard grid shapes (clockwise order).
    """
    poly_eps = 1 - 1e-16
    pts = {
        # Clockwise.
        (0, 0): [  # c2 half-hexagons net_mode 0 - starting at oct. vertex
            [(3, 1), (2, 0), (0, 0), (-1, 1)],    # c2=0 gradient: flat
            [(0, -2), (-1, -1), (0, 0), (2, 0)],  # c2=1 gradient: forward
            [(-3, 1), (-1, 1), (0, 0), (-1, -1)]  # c2=2 gradient: back
        ],
        (0, 1): [  # c2 half-hexagons net_mode 1- starting at oct. vertex
            [(3, -1), (-1, -1), (0, 0), (2, 0)],
            [(-3, -1), (-1, 1), (0, 0), (-1, -1)],
            [(0, 2), (2, 0), (0, 0), (-1, 1)]
        ],
        (1, 0): [  # c2 hexagons net_mode 0;  final 2 pts in opp. net_mode.
            # If exterior, they are idx 2,1 (eg (0,0) (2,0)).
            #་ Mode 0 hexagon centroids, by C2 are (1,1), (1,-1), (-2,0)
            [(3, 1), (2, 0), (0, 0), (-1, 1), (0, 2), (2, 2)],        # (0, 2), (2, 2)
            [(0, -2), (-1, -1), (0, 0), (2, 0), (3, -1), (2, -2)],    # (3, -1), (2, -2)
            [(-3, 1), (-1, 1), (0, 0), (-1, -1), (-3, -1), (-4, 0)]   # (-3, -1), (-4, 0)
        ],
        (1, 1): [  # c2 hexagons net_mode 1;  final 2 pts in opp. net_mode - exterior.
            # Mode 1 hexagon centroids, by C2 are (1,-1), (-2,0), (1,-1),
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
        (3, 0): [  # clockwise, c0,c1,c2 - edges of super-region 0: (0,0 is centre)
            [
                (-1, 1), (1, 1), (3, 1),
                (2, 0), (1, -1), (0, -2),
                (-1, -1), (-2, 0), (-3, 1)
            ]
        ],
        (3, 1): [  # clockwise, c0,c1,c2 - edges of super-region 1
            [
                (3, -1), (1, -1), (-1, -1),
                (-3, -1), (-2, 0), (-1, 1),
                (0, 2), (1, 1), (2, 0),
            ]
        ],
        (4, 0): [  # clockwise, c0,c1,c2 - mode 0 vertices of super-region
            [  # <-c0->  <-c1->   <-c2->
                (3, 1), (0, -2), (-3, 1),  # (-3, 1)
            ]
        ],
        (4, 1): [  # clockwise, c0,c1,c2 - mode 1 vertices of super-region
            [  # <-c0->  <-c1->   <-c2->
                (3, -1), (-3, -1), (0, 2)
            ]
        ],
        # Unshared points of cell excluding (0,0) use only on one net_mode
        # The other net_mode will be just the (0,0)
        (5, 0): [[(2, 0), (1, -1), (-1, -1), (-2, 0), (-1, 1), (1, 1)]],
        # Hex centroids.
        (6, None): [(1, 1), (1, -1), (-2, 0)]
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
    hi = np.zeros((2, 3, 7, 2), dtype=np.int16)  # hexagon index with centroid as uv integers
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
                case 6:
                    hi[0, c2] = np.array(pts[1, 0][c2] + [poly])
                    hi[1, c2] = np.array(pts[1, 1][c2] + [poly])


    return H9Polygon(hh=hh, hx=hx, tx=tx, se=te, sv=sr, gd=gd, hi=hi)


H9P: H9Polygon = _h9_polygon()

# H9P.sv corner k of a mode-m octant face lies on this ECEF axis
# (0=x/AP, 1=y/EW, 2=z/NS); on the sv edge OPPOSITE corner k that axis's
# coordinate is exactly 0 (an octant seam). Verified by projecting every
# face's sv corners through b_oct -> g_gcd.
SV_CORNER_AXIS = ((0, 2, 1), (0, 1, 2))


def fold_to_octant(verts: NDArray, oid: int) -> Tuple[NDArray, NDArray]:
    """Fold template vertices that overhang their octant into the octant
    that actually contains them.

    Hex templates near an octant edge legitimately extend past the sv
    triangle, but b_oct coordinates are only meaningful inside their own
    octant's frame: projecting an overhanging vertex "strictly in-octant"
    wraps to nonsense (the grid_face_vertex_oid_bug family). Unfolding is
    exact — the neighbour's face continued into this frame is the mirror
    image across the shared edge — so a vertex beyond the edge opposite
    corner k belongs to ``oid ^ (1 << SV_CORNER_AXIS[mode][k])``, and its
    coordinates in that neighbour's own frame follow by matching the three
    axis-labelled corners (an exact isometry between congruent triangles).
    Repeats (≤3) for vertices beyond a corner, where two seams are crossed;
    at the 6 octahedral vertices the fold order is the inherent cone-angle
    ambiguity and either result is a faithful representative.

    Args:
        verts: (N, 2) b_oct coordinates in ``oid``'s frame.
        oid:   the frame's octant id.
    Returns:
        (coords (N, 2), oids (N,)) — each vertex in its containing octant.
    """
    out = np.array(verts, dtype=np.float64, copy=True)
    oids = np.full(len(out), int(oid), dtype=np.uint8)
    for i in range(len(out)):
        v = out[i]
        o = int(oid)
        for _ in range(3):
            mode = int(H9O.oid_mo[o])
            tri = H9P.sv[mode]
            k_hit = None
            for k in range(3):
                a, b = tri[(k + 1) % 3], tri[(k + 2) % 3]
                cross = ((b[0] - a[0]) * (v[1] - a[1]) -
                         (b[1] - a[1]) * (v[0] - a[0]))
                if cross > 1e-12:          # CW polygon: inside is cross <= 0
                    k_hit = k
                    break
            if k_hit is None:
                break
            k = k_hit
            a, b = tri[(k + 1) % 3], tri[(k + 2) % 3]
            e = b - a
            e = e / np.hypot(e[0], e[1])

            def _reflect(p):
                d = p - a
                return a + 2.0 * float(d @ e) * e - d

            mode_b = 1 - mode
            tri_b = H9P.sv[mode_b]
            axis_b = SV_CORNER_AXIS[mode_b]
            # source corners (unfolded-neighbour triangle in this frame) and
            # target corners (the neighbour's canonical frame), axis-matched
            src = np.array([a, b, _reflect(tri[k])])
            axes = (SV_CORNER_AXIS[mode][(k + 1) % 3],
                    SV_CORNER_AXIS[mode][(k + 2) % 3],
                    SV_CORNER_AXIS[mode][k])
            dst = np.array([tri_b[axis_b.index(ax)] for ax in axes])
            m = np.linalg.solve(np.column_stack([src, np.ones(3)]), dst)
            v = np.array([v[0], v[1], 1.0]) @ m
            o ^= 1 << SV_CORNER_AXIS[mode][k]
        out[i] = v
        oids[i] = o
    return out, oids


def uv_grid_debug(levels: int = 3, mode: int = 0) -> NDArray:
    """
    Generate a grid of points recursively.
    Returns: np.array of [u, v, scale, mode]
    """
    from hhg9.h9 import H9C, H9R
    h9c, h9r = H9C, H9R
    modes = [h9r.downs, h9r.ups]
    kids = modes[mode]
    scale = 3**levels
    queue = [(k, h9c.mode[k], h9c.off_uv[k], scale) for k in kids]

    for depth in range(levels):
        next_q = []
        for path, mode, origin, scale in queue:
            kids = modes[mode]  # shape (9,) indices
            for k in kids:
                mo_k = h9c.mode[k]
                off_k = h9c.off_uv[k]
                path_k = np.append(path, k)
                origin_k = origin + off_k * scale
                next_q.append((path_k, mo_k, origin_k, scale // 3))
        queue = next_q
    return np.array([[res[2][0], res[2][1], res[3], res[1]] for res in queue], dtype=np.int32)

def uv_grid(levels: int = 3, mode: int = 0, debug=True) -> NDArray:
    """
    Generate a grid of points recursively.
    Returns: np.array of [u, v, scale, mode]
    """
    if debug:
        return uv_grid_debug(levels, mode)
    from hhg9.h9 import H9C, H9R
    h9c, h9r = H9C, H9R
    modes = [h9r.downs, h9r.ups]
    kids = modes[mode]
    scale = 3**levels
    queue = [(h9c.mode[k], h9c.off_uv[k], scale) for k in kids]

    for depth in range(levels):
        next_q = []
        for mode, origin, scale in queue:
            kids = modes[mode]  # shape (9,) indices
            for k in kids:
                mo_k = h9c.mode[k]
                off_k = h9c.off_uv[k]
                origin_k = origin + off_k * scale
                next_q.append((mo_k, origin_k, scale // 3))
        queue = next_q
    return np.array([[res[1][0], res[1][1], res[2], res[0]] for res in queue], dtype=np.int32)


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


def uv_grid(levels: int = 3, mode: int = 0, flatten=True, h9p: H9Polygon = H9P) -> List[Tuple]:
    """
    Generate a grid of points in UV recursively.
    If flatten=True, returns a list of (mode, u, v) tuples (where scale = 1)
    If flatten=False, returns a list of lists of (mode, u, v) tuples.
    Returns:
        List of [current_mode, origin_xy, scale].
    """
    from hhg9.h9 import H9C, H9R   # H9_RA
    h9c, h9r = H9C, H9R
    mode_cells = [h9r.downs, h9r.ups]
    mode_ofs = h9c.off_uv[mode_cells].astype(np.int32)
    mode_ofs[:, :, 0] *= 3   # Polygon Grid shape: U is 3U
    k_mos = mode_cells[mode]
    k_ofs = mode_ofs[mode]
    scale = 3**levels
    queue = [(h9c.mode[k], o * scale, scale) for k, o in zip(k_mos, k_ofs)]
    layers = []
    if not flatten:
        layers.append(np.array([[m, u, v, s] for (m, (u, v), s) in queue], dtype=np.int32))
    for depth in range(levels):
        next_q = []
        for mode, origin, scale in queue:
            k_scale = scale // 3
            k_mos = mode_cells[mode]
            k_ofs = mode_ofs[mode]
            for k, o in zip(k_mos, k_ofs):
                mo_k = h9c.mode[k]
                origin_k = origin + (o * k_scale)
                next_q.append((mo_k, origin_k, k_scale))
        queue = next_q
        if not flatten:
            layers.append(np.array([[m, u, v, s] for (m, (u, v), s) in queue], dtype=np.int32))
    if flatten:
        return np.array([[m, u, v, s] for (m, (u, v), s) in queue], dtype=np.int32)
    else:
        return layers


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


def octant_grid(levels: int = 3, octant_id: int = 0, h9p: H9Polygon = H9P):
    """Triangular mesh for one octant, with its boundary vertices classified.

    The mesh holds 9**(levels+1) triangles: 9 per octant at levels 0, and 9 per
    triangle at each subsequent level.

    Args:
        levels: Subdivision depth.
        octant_id: Octant index (0-7); selects the root mode and components.
        h9p: Polygon LUT.

    Returns:
        tuple: (verts, tris, oc_vtx, oc_edg, cmp)
            verts: (V, 2) b_oct vertex coordinates.
            tris: (T, 3) triangle indices into verts.
            oc_vtx: indices of verts on an octant vertex (3 of them).
            oc_edg: indices of verts on an octant seam.
            cmp: the octant's component signature, for Points(...).

    Raises:
        ValueError: if the mesh does not have the structure the level implies —
            the counts below are exact, so a mismatch means the mesh is wrong,
            not merely unexpected.
    """
    if not (0 <= octant_id < 8):
        raise ValueError(f'octant_id must be in [0, 7], got {octant_id}')
    mode = H9O.oid_mo[octant_id]
    cmp = H9O.oid_cmp[octant_id]
    verts, _, tris = tri_mesh(levels, mode, h9p=h9p)

    # Classify each vertex: on an octant corner (VTX), on a seam (EDG), or interior.
    x3, y = verts[:, 0] * H9K.R3, verts[:, 1]
    locs = location(x3, y, mode)
    oc_vtx = np.flatnonzero(locs == BaryLoc.VTX)
    oc_edg = np.flatnonzero(locs == BaryLoc.EDG)

    m = 3 ** (levels + 1)
    for got, want, what in (
        (verts.shape[0], (m + 1) * (m + 2) // 2, 'vertices'),
        (tris.shape[0], 9 ** (levels + 1), 'triangles'),
        (oc_vtx.shape[0], 3, 'octant corner vertices'),
        (oc_edg.shape[0], 3 * m - 3, 'octant seam vertices'),
    ):
        if got != want:
            raise ValueError(f'octant_grid(levels={levels}, octant_id={octant_id}): '
                             f'{what} count {got} is not {want}')
    return verts, tris, oc_vtx, oc_edg, cmp


def net_polys(reg, n_oct):
    """Return the face-triangle vertices for a net as Points in n_oct.

    .. deprecated::
        For face membership / seam testing prefer ``n_oct.pt_face(coords)``
        (``OctahedralNet.pt_face``), which handles all layouts including
        rhombus and returns a per-point octant OID directly.
        This function does not support the rhombus layout.
    """
    from hhg9 import Points
    if n_oct.name.split(":")[1] == 'rhombus':
        raise NotImplementedError("rhombus not implemented - yet!")
    b_oct = n_oct.b_oct
    o = np.array([np.repeat(o, 3) for o in range(8)])
    v = np.array([H9P.sv[mo] for mo in H9O.oid_mo])  # 8,3,2
    pts = Points(v.reshape(-1, 2), domain=b_oct, oid=o.reshape(-1))
    return reg.project(pts, [b_oct, n_oct])


# Quick helper to round a point for safe matching
def _match_hx_pt(pt):
    return tuple(round(v, 5) for v in pt)


def net_poly(reg, n_oct):
    """Given an octahedral net (ONLY triangular! Not rhombus, yet), return the list of bounding polygons.

    .. deprecated::
        See ``net_polys`` — prefer ``n_oct.pt_face(coords)`` for membership tests.
    """
    npt = net_polys(reg, n_oct)

    # convert back to vertices and store as lines.
    boundary_edges = dict()
    for tri in npt.coords.reshape([8, 3, 2]):
        for i in range(3):
            v1, v2 = tri[i], tri[(i + 1) % 3]
            # Pair the current vertex with the next one (wrapping around)
            p1 = tuple(np.round(v1, 5).tolist())
            p2 = tuple(np.round(v2, 5).tolist())
            edge = tuple(sorted([p1, p2]))
            if edge in boundary_edges:
                del boundary_edges[edge]
            else:
                boundary_edges[edge] = [v1, v2]

    # ---------------------------------------------------------
    # Phase 2: Upgraded Chaining (Safe Float Comparison)
    # ---------------------------------------------------------
    edges_pool = list(boundary_edges.values())
    polygons = []

    while edges_pool:
        start_edge = edges_pool.pop(0)
        current_outline = [start_edge[0], start_edge[1]]

        while True:
            last_pt = current_outline[-1]
            first_pt = current_outline[0]

            # 1. Safe Loop Closure: Compare rounded points
            if len(current_outline) > 2 and _match_hx_pt(last_pt) == _match_hx_pt(first_pt):
                current_outline.pop()
                break

            found_connection = False
            for i, edge in enumerate(edges_pool):
                # 2. Safe Edge Connection: Compare rounded points, but append the RAW payload
                if _match_hx_pt(edge[0]) == _match_hx_pt(last_pt):
                    current_outline.append(edge[1])
                    edges_pool.pop(i)
                    found_connection = True
                    break
                elif _match_hx_pt(edge[1]) == _match_hx_pt(last_pt):
                    current_outline.append(edge[0])
                    edges_pool.pop(i)
                    found_connection = True
                    break

            if not found_connection:
                break

        polygons.append(current_outline)
    return polygons


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