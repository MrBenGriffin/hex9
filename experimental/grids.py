# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
import numpy as np
from hhg9.h9 import H9O, H9K
from hhg9.h9.classifier import H9CL, in_down, in_up
from hhg9.h9.polygon import H9P, tri_grid
from hhg9.h9.protocols import BaryLoc


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


def tri_mesh(levels: int = 5, mode: int = 0):
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
    # Use existing tri_grid to get all triangle vertices at this layer
    h9p = H9P
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


def location(ẋ, y, mode=0):
    """
    from hhg9.h9.classifier
    Here everything will be in_down.
    Classify the location of (ẋ, y) as "internal", "edge", "vertex", or "external"
    with respect to supercell boundaries, using barycentric inclusion.
    Vectorized for scalar or array input, returns array of strings.
    """
    h9c = H9CL
    # NOTE: some boundary constants (e.g. VC/ΛF) can be ~0, in which case np.isclose
    # relies almost entirely on `atol`. This needs to be comfortably above float64
    # epsilon-scale noise and consistent with the mesh de-dup tolerance.
    a_eps = 2e-14
    r_eps = 1e-12
    if not isinstance(mode, np.ndarray):
        mode = np.full(ẋ.shape[0], mode, dtype=np.uint8)
    ups = np.flatnonzero(mode)
    dns = np.flatnonzero(~mode)
    ẋu, yu = ẋ[ups], y[ups]
    ẋd, yd = ẋ[dns], y[dns]
    in_d, in_u = in_down(ẋd, yd, h9c), in_up(ẋu, yu, h9c)

    # Solve C2=0 (flat edge)
    on0_d = np.isclose(yd, H9K.limits.VC, rtol=r_eps, atol=a_eps)  # uppermost point is flat
    on0_u = np.isclose(yu, H9K.limits.ΛF, rtol=r_eps, atol=a_eps)  # lowermost point is flat
    # Solve C2=1 (forward edge)
    ẇ = H9K.derived.Ẇ
    on1_d = np.isclose(yd - ẋd, -ẇ, rtol=r_eps, atol=a_eps)  # y-ẋ == -ẇ
    on1_u = np.isclose(yu - ẋu,  ẇ, rtol=r_eps, atol=a_eps)  # y-ẋ == ẇ
    # Solve C2=1 (backward edge)
    on2_d = np.isclose(yd + ẋd, -ẇ, rtol=r_eps, atol=a_eps)  # y+ẋ == -ẇ
    on2_u = np.isclose(yu + ẋu,  ẇ, rtol=r_eps, atol=a_eps)  # y+ẋ == ẇ
    close_d = on0_d.astype(int) + on1_d.astype(int) + on2_d.astype(int)
    close_u = on0_u.astype(int) + on1_u.astype(int) + on2_u.astype(int)

    # If a point is within tolerance of a boundary line, treat it as "inside" for
    # classification purposes. This prevents epsilon-scale drift from turning true
    # edge/vertex points into EXT.
    in_d_eff = in_d | (close_d > 0)
    in_u_eff = in_u | (close_u > 0)

    result = np.full(ẋ.shape, BaryLoc.EXT, dtype=int)
    result[in_d_eff & (close_d == 2)] = BaryLoc.VTX  # Vertex: two or more boundaries within eps
    result[in_d_eff & (close_d == 1)] = BaryLoc.EDG  # Edge: exactly one boundary within eps
    result[in_d_eff & (close_d == 0)] = BaryLoc.INT  # Internal: inside and no boundary within eps
    result[in_u_eff & (close_u == 2)] = BaryLoc.VTX  # Vertex: two or more boundaries within eps
    result[in_u_eff & (close_u == 1)] = BaryLoc.EDG  # Edge: exactly one boundary within eps
    result[in_u_eff & (close_u == 0)] = BaryLoc.INT  # Internal: inside and no boundary within eps
    return result


def grid(layer: int = 3, octant_id: int = 0):
    """
    Triangular grid will be 9 triangles per octant at hex_layer 0.
    At each subsequent hex_layer, the number of triangles will increase by 9 per triangle.
    So the number of triangles will be 9**(1+hex_layer) per octant.
    """
    mode = H9O.oid_mo[octant_id]
    m = 3**(layer+1)
    pts_expected = int((m+1)*(m+2)/2)
    tri_expected = 9**(layer+1)
    vtx_expected = 3
    edg_expected = 3*m-3
    verts, _, trx = tri_mesh(layer, mode)
    ẋ, y = verts[:, 0] * H9K.R3, verts[:, 1]
    locs = location(ẋ, y, mode)
    oc_vtx = np.flatnonzero(locs == BaryLoc.VTX)  # On an octant vertex (1 of 3)
    oc_edg = np.flatnonzero(locs == BaryLoc.EDG)  # On the octant seam
    if verts.shape[0] != pts_expected:
        print(f"{layer}: octant points {verts.shape[0]} is not {pts_expected}")
    if trx.shape[0] != tri_expected:
        print(f"{layer}: octant triangles {trx.shape[0]} is not {tri_expected}")
    if oc_vtx.shape[0] != vtx_expected:
        print(f"{layer}: octant vert_pts {oc_vtx.shape[0]} is not {vtx_expected}")
    if oc_edg.shape[0] != edg_expected:
        print(f"{layer}: octant seam_pts {oc_edg.shape[0]} is not {edg_expected}")


if __name__ == '__main__':
    octant_id = 0
    for layer in [6]:  # range(4, 6):
        grid(layer, octant_id)
        print(f"Generated grid for layer {layer} and octant {octant_id}")
