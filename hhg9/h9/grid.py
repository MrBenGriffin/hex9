# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Grid Methods - eg for composing rectilinear pixel grids for projected sampling.
"""
from functools import cache
import numpy as np
from hhg9.algorithms.geometry import inside_convex_polygon_cw
from hhg9.h9 import H9K, H9O, H9P
from hhg9.h9.classifier import in_scope, location
from hhg9.h9.polygon import region_grid, uv_grid
from hhg9.algorithms.distance import ellipsoid_area_wgs84
from hhg9 import Points, Registrar


# Octahedral face adjacency: each face has exactly 3 neighbours.
# Derived from layer-0 hex boundary corrections in _raw_uv.
# Used by _get_verts to restrict seam-vertex fallback to geometrically
# nearby faces and prevent grabbing vertices from non-adjacent (far) faces.
_FACE_ADJ: tuple[frozenset, ...] = (
    frozenset({1, 2, 4}),  # 0
    frozenset({0, 3, 5}),  # 1
    frozenset({0, 3, 6}),  # 2
    frozenset({1, 2, 7}),  # 3
    frozenset({0, 5, 6}),  # 4
    frozenset({1, 4, 7}),  # 5
    frozenset({2, 4, 7}),  # 6
    frozenset({3, 5, 6}),  # 7
)


@cache
def _hex_areas(earth_area: float = None):
    """Return characteristic lengths for an ideal regular hexagon at each layer.
    Returns:
        np.array (max_depth, 5):
            area: area in m^2
            side: side length s (aka) circumradius: (vertex-centre)
            inradius: r (flat-centre)
            flat_diameter: 2*r aka 'height'
            point_diameter: 2*side
    """
    # Level 0 area ≈ 42_505_468 km^2 (area of Earth / 12 hexes)
    max_depth = 64  # way over the top.
    earth = earth_area if earth_area is not None else float(ellipsoid_area_wgs84())
    hex_0 = earth / 12
    l_hex = hex_0
    h_areas = np.zeros((max_depth, 5), dtype=np.float64)
    for i in range(max_depth):
        area = l_hex
        side = np.sqrt((2.0 * area) / (3.0 * np.sqrt(3.0)))  # for hex, side aka big_r
        r = (np.sqrt(3.0) / 2.0) * side
        in_radius = r
        flat_diameter = 2.0 * r
        point_diameter = 2.0 * side
        h_areas[i] = [area, side, in_radius, flat_diameter, point_diameter]
        l_hex /= 9
    return h_areas


@cache
def _tri_areas(earth_area: float = None):
    """Return characteristic lengths for an ideal equilateral triangle at `layer`.
    Derived from area A = (sqrt(3)/4) * a^2.
    returns np.array (max_depth, 5):
        area: area in m^2
        side: side length a
        inradius: r
        height: altitude
        circumradius: R
    """
    # Level 0 area ≈ 63_758_203 km^2 (area of Earth / 8 faces)
    max_depth = 64  # way over the top.
    earth = earth_area if earth_area is not None else float(ellipsoid_area_wgs84())
    tri_0 = earth / 8
    l_tri = tri_0
    t_areas = np.zeros((max_depth, 5), dtype=np.float64)
    for i in range(max_depth):
        area = l_tri
        a = np.sqrt((4.0 * area) / np.sqrt(3.0))  # side
        r = (np.sqrt(3.0) / 6.0) * a  # inradius
        h = (np.sqrt(3.0) / 2.0) * a  # height
        big_r = (np.sqrt(3.0) / 3.0) * a
        side = a
        in_radius = r
        height = h
        t_areas[i] = [area, side, in_radius, height, big_r]
        l_tri /= 9
    return t_areas


@cache
def hex_props(level=None, earth_area: float = None):
    """return (ideal) hex_grid properties for layer"""
    if level is None:
        return _hex_areas(earth_area)
    return _hex_areas(earth_area)[level]


@cache
def tri_props(level, earth_area: float = None):
    """return (ideal) tri_grid properties for layer"""
    return _tri_areas(earth_area)[level]


def densify_step_for_layer(level, kind: str = 'hex', factor: float = 1.0, safety: float = 0.9) -> float:
    """
    Suggest a geodesic densify step (metres) from hex_layer semantics.
    Intended for boundary densification when building a bbox for clipping.

    Args:
        level: grid hex_layer.
        kind: 'hex' or 'tri'.
        factor: multiplier applied to a conservative characteristic span.
        safety: shrink factor (<=1) to hedge against local authalic shrink.

    Returns:
        step length in metres.

    Notes:
        This errs on the side of more boundary samples.
    """
    if kind == 'tri':
        props = tri_props(level)
    else:
        props = hex_props(level)
    base = float(props[2])  # inradius
    return float(base) * float(safety) / float(factor)


def fit(pts, img_w, img_h):
    """fit/scale pnts to the grid size"""
    x = pts.coords[:, 0]
    y = pts.coords[:, 1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    nx = x_max - x_min
    ny = y_max - y_min
    sx = (img_w - 1) / nx if nx > 0 else 1.0
    sy = (img_h - 1) / ny if ny > 0 else 1.0
    pvx = np.rint((x - x_min) * sx).astype(int)
    pvy = (img_h - 1) - np.rint((y - y_min) * sy).astype(int)
    pvx = np.clip(pvx, 0, img_w - 1)
    pvy = np.clip(pvy, 0, img_h - 1)
    return pvx, pvy


def sq_grid(scale: float = 1000, mode: int = 1):
    """
    Generate a rectilinear grid of points within a square centered at (cx, cy) in pixel space.
    """
    wid = max(1, int(round(scale)))
    hgt = max(1, int(np.ceil(scale * H9K.derived.RH)))
    fl, cl = (H9K.limits.ΛF, H9K.limits.ΛC) if mode == 1 else (H9K.limits.VC, H9K.limits.VF)
    xl = np.linspace(H9K.limits.TL, H9K.limits.TR, num=wid)
    yl = np.linspace(fl, cl,     num=hgt)

    # IMPORTANT: one meshgrid for everything; keep indexing="xy"
    xx, yy = np.meshgrid(xl, yl, indexing="xy")                   # shapes (hgt, wid)
    rec    = np.stack((xx.ravel(), yy.ravel()), axis=1)     # hex_layer = hgt*wid

    ii, jj = np.meshgrid(np.arange(wid), np.arange(hgt), indexing="xy")
    px_idx = ii.ravel().astype(np.int32)                        # x (col)
    py_idx = (hgt - 1 - jj.ravel()).astype(np.int32)            # y (row), flipped once

    # Edge-inclusive mask
    dx = (H9K.limits.TR - H9K.limits.TL) / max(wid - 1, 1)
    dy = (cl - fl) / max(hgt - 1, 1)
    pix = 1 - max(abs(dx), abs(dy))
    trx = in_scope(H9K.radical.R3 * rec[:, 0] * pix, rec[:, 1] * pix, mode)
    return wid, hgt, rec, trx, px_idx, py_idx


def sq_grid_vx(scale: float = 1000, mode: int = 0):
    """
    Return a rectilinear grid of points within an equilateral triangle centered
    at (cx, cy) in pixel space conforming to the barycentric projection of the side of a unit octahedron.
    When calling this for a net, remember to use the net's ΛV not the barycentric!

    """
    wid = int(scale)
    if wid < 1:
        wid = 1
    hgt = max(1, int(round(scale * H9K.derived.RH)))

    # generate a covering rectangle (inclusive of endpoints)
    fl, cl = (H9K.limits.ΛF, H9K.limits.ΛC) if mode == 1 else (H9K.limits.VF, H9K.limits.VC)
    yl = np.linspace(fl, cl, num=hgt).astype(np.float64)
    xl = np.linspace(H9K.limits.TL, H9K.limits.TR, num=wid)
    xx, yy = np.meshgrid(xl, yl)
    rec = np.stack((xx.ravel(), yy.ravel()), axis=1, dtype=np.float64)
    mo = np.full(rec.shape[0], mode, dtype=np.uint8)

    # Integer pixel indices for each sample (column x, row y)
    jj, ii = np.meshgrid(np.arange(hgt), np.arange(wid), indexing='ij')
    pix_x = ii.ravel()
    pix_y = (hgt - 1 - jj.ravel())  # flip Y once, globally

    # Robust edge inclusion: evaluate validity on a slightly in-set copy so that
    # points that lie numerically on the sloped edges are treated as inside.
    # Pixel size in coordinate space
    dx = (H9K.limits.TR - H9K.limits.TL) / max(wid - 1, 1)
    dy = (cl - fl) / max(hgt - 1, 1)
    pix = max(abs(dx), abs(dy))
    tx = rec[:, 0] * H9K.radical.R3
    trx = in_scope(tx, rec[:, 1], mo)  # ~half a pixel in coord space
    return wid, hgt, rec, trx, pix_x, pix_y


def qa_grid(quad, scale: float = 200, affine=None, tol: float = 0.5, net_mode: int = -1):
    """
    Return a rectilinear grid of points within a quadrilateral.
    Also returns the mask and scales.

    Parameters
    ----------
    quad : array-like, shape (4, 2)
        Quadrilateral vertices in barycentric coordinate space.
    scale : float
        Pixels per coordinate unit (e.g. 200 → 200 px over a 1-unit span).
        Grid dimensions are derived from the placed bounding box:
            wid = round(scale * bbox_width)
            hgt = round(scale * bbox_height)
    affine : tuple (matrix, offset) or None
        Affine transform mapping quad from barycentric to net space.
        grid and mask are both computed in the post-transform space.
    tol : float
        Edge-expansion tolerance in pixels (default 0.5).  Passed to
        :func:`~hhg9.algorithms.geometry.inside_convex_polygon_cw` as
        ``tol / scale`` (distance units), so each edge is widened by half a
        pixel outward.  Adjacent quads therefore overlap by one pixel at
        shared edges, eliminating tears between c2 / half-hex regions.
        Pass ``tol=0`` to restore the original strict boundary.
    net_mode : int
        Placed orientation of the sub-triangle in net space (-1 = unspecified).
        When mode=1 (Λ, apex-up), the flat top edge (cmaxy) is a seam that
        belongs to the adjacent mode-0 quad by convention; those grid rows are
        excluded from the mask.  When mode=0 or -1, no adjustment is made.

    Returns
    -------
    wid, hgt : int
        Pixel dimensions of the bounding-rectangle grid.
    grid : ndarray (wid*hgt, 2)
        Coordinates of all grid points in placed space (Y top→bottom).
    trx : ndarray bool (wid*hgt,)
        Interior mask — True where the point lies inside the placed quad.
    (bbox, (sx, sy)) : tuple
        bbox = [cminx, cminy, cmaxx, cmaxy]; sx/sy are pixels-per-unit.
    """
    quad = np.asarray(quad, dtype=float)
    if affine is not None:
        matrix, offset = affine
        placed = quad @ matrix + np.asarray(offset, dtype=float)
    else:
        placed = quad
    cminx, cmaxx = placed[:, 0].min(), placed[:, 0].max()
    cminy, cmaxy = placed[:, 1].min(), placed[:, 1].max()
    cw = cmaxx - cminx
    ch = cmaxy - cminy
    wid = max(1, int(np.rint(scale * cw)))
    hgt = max(1, int(np.rint(scale * ch)))
    yl = np.linspace(cmaxy, cminy, num=hgt)
    xl = np.linspace(cminx, cmaxx, num=wid)
    xx, yy = np.meshgrid(xl, yl)
    grid = np.stack((xx.ravel(), yy.ravel()), axis=1)
    dist_tol = tol / scale if (tol > 0.0 and scale > 0.0) else 0.0
    trx = inside_convex_polygon_cw(grid, placed, tol=dist_tol)
    if net_mode == 1:
        trx &= ~np.isclose(grid[:, 1], cmaxy, atol=0.4 / scale)
    sx = (wid - 1) / cw if wid > 1 else 1.0
    sy = (hgt - 1) / ch if hgt > 1 else 1.0
    return wid, hgt, grid, trx, (np.array([cminx, cminy, cmaxx, cmaxy]), (sx, sy))


def _calculate_intersections(poly_grid, v):
    """
    Finds u-coordinates where the horizontal line at v crosses polygon edges.

    Args:
        poly_grid: (m, 2) array of polygon vertices in scaled UV space.
        v: The current scanline (integer row index).
    """
    p1 = poly_grid
    p2 = np.roll(poly_grid, -1, axis=0)
    mask = ((p1[:, 1] <= v) & (p2[:, 1] > v)) | ((p2[:, 1] <= v) & (p1[:, 1] > v))
    if not np.any(mask):
        return np.array([])
    e1 = p1[mask]
    e2 = p2[mask]
    u_ints = e1[:, 0] + (v - e1[:, 1]) * (e2[:, 0] - e1[:, 0]) / (e2[:, 1] - e1[:, 1])
    return np.sort(u_ints)


def poly_net_field(poly_n, layer: int):
    """
    Generate a dense set of H9 axial grid points (in n_oct space) covering a polygon.

    Uses a scanline fill in hex axial (skew UV) coordinates to enumerate every
    grid point at the given layer that falls inside ``poly_n``.

    Args:
        poly_n: Points in n_oct whose convex hull defines the region of interest.
        layer:  H9 hex layer; controls grid spacing (sn = W * 3^-(layer+1)).

    Returns:
        Points in n_oct, filtered to those inside the domain.
    """
    sn = H9K.radical.W * (3 ** -(layer + 1))

    x, y = poly_n.coords[:, 0], poly_n.coords[:, 1]
    v_skew = (y * 2 / H9K.R3) / sn
    u_skew = (x / sn) - (v_skew / 2)
    poly_grid = np.stack([u_skew, v_skew], axis=1)

    v_min = int(np.floor(poly_grid[:, 1].min()))
    v_max = int(np.ceil(poly_grid[:, 1].max()))
    u_acc, v_acc = [], []

    for v in range(v_min, v_max + 1):
        u_ints = _calculate_intersections(poly_grid, v)
        for i in range(0, len(u_ints), 2):
            if i + 1 >= len(u_ints):
                break
            u_start = int(np.ceil(u_ints[i]))
            u_end = int(np.floor(u_ints[i + 1]))
            if u_end >= u_start:
                u_range = np.arange(u_start, u_end + 1)
                u_acc.append(u_range)
                v_acc.append(np.full_like(u_range, v))

    if not u_acc:
        avg = np.mean(poly_grid, axis=0)
        u_final = np.array([np.round(avg[0])])
        v_final = np.array([np.round(avg[1])])
    else:
        u_final = np.concatenate(u_acc)
        v_final = np.concatenate(v_acc)

    y_n = (v_final - (1 / 3)) * sn * (H9K.R3 / 2)
    x_n = (u_final + (v_final / 2)) * sn
    uv = np.stack([x_n, y_n], axis=1)
    pts = Points(uv, domain=poly_n.domain)
    return poly_n.domain.filter(pts)


def hex_verts_in_noct(hex_par, hex_oid, xpm, xc2, scale, n_oct):
    """
    Build hex polygon vertices directly in n_oct space using per-face rigid transforms.

    Args:
        hex_par:  (H, 2) hex parent centre coordinates in b_oct.
        hex_oid:  (H,)   octant face id for each hex (from hex_parents).
        xpm:      (H,)   net_mode index (0 or 1).
        xc2:      (H,)   c2 index (0, 1, or 2).
        scale:    scalar scale factor for this layer (from hex_parents).
        n_oct:    OctahedralNet domain instance.

    Returns:
        Points of shape (H*6, 2) in n_oct.
        (1 - 1e-15) 0
        (1 - 2e-13) 4
    """
    hex_verts_n = np.zeros((len(hex_par), 6, 2))
    sc = 2.2e-15  # np.nextafter(scale, 0) is too small.
    # scale_in = scale - 1.65e-15  # Tiny bit smaller to stop fp errors
    for side, proj in n_oct.projs.items():
        oid = n_oct.sides[side].oid
        fm = (hex_oid == oid)
        if not np.any(fm):
            continue
        if proj.c2trans is None:
            par_n = hex_par[fm] @ proj.matrix + proj.offset
            hex_verts_n[fm] = par_n[:, None, :] + H9P.hx[xpm[fm], xc2[fm]] * (scale-sc) @ proj.matrix
        else:
            from hhg9.h9.classifier import classify_cell
            from hhg9.h9.region import H9R
            bary_mode = proj.rev_cs.mode
            fm_idx = np.where(fm)[0]
            face_cid = classify_cell(H9K.R3 * hex_par[fm, 0], hex_par[fm, 1])
            face_c2 = H9R.mcc2[bary_mode, face_cid]
            for c2v in range(3):
                local_mask = face_c2 == c2v
                if not np.any(local_mask):
                    continue
                c2m = fm_idx[local_mask]
                matr, off_c2 = proj.c2_affine(c2v)
                par_n = hex_par[c2m] @ matr + off_c2 + proj.offset
                hex_verts_n[c2m] = par_n[:, None, :] + H9P.hx[xpm[c2m], xc2[c2m]] * (scale-sc) @ matr
    pending = Points(hex_verts_n.reshape(-1, 2), n_oct)

    return pending


def enmesh(pts, levels: int = 35, shape=None):
    """
    Builds the half-hex polygons for a batch of points up to `levels` depth.
    Walks the hierarchy for each point and generates the geometry for every hex_layer.

    Returns:
        tuple: (uniques, refs)
            uniques: (U, 4, 2) Unique polygons in global coordinates.
            refs: (hex_layer, depth) Indices into `uniques` for each input point/hex_layer.
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

    # We'll collect per-hex_layer polygons flattened as (hex_layer*D, 4, 2)
    polys = np.empty((num_pts * depth, 4, 2), dtype=np.float64)
    flat_idx = np.empty((num_pts, depth), dtype=np.int64)  # indices into `polys`

    for ev in xy_regions_iter(coords, mode=mo, depth=levels):
        if ev.phase != 'pre':
            continue
        i = ev.i  # 0..D-1

        # Child c2 for each row under the parent net_mode
        c2 = H9R.mcc2[ev.pmo, ev.cid]  # (hex_layer,)
        hh_shape = hh[ev.pmo, c2]  # pmo

        # Translate/scale each triangle to global coords
        polygon = parent_xy[:, None, :] + hh_shape * scale  # (hex_layer,4,2)

        # Store flattened by hex_layer, keeping a stable mapping back to rows
        start = i * num_pts
        polys[start:start + num_pts] = polygon
        flat_idx[:, i] = np.arange(start, start + num_pts, dtype=np.int64)

        # Step parent origin for next hex_layer
        parent_xy = parent_xy + offs[ev.cid] * scale
        scale /= 3.0

    if num_pts > 1:
        # Deduplicate exactly: reshape to (M, 8) and unique over rows
        mass = num_pts * depth
        flat = polys.reshape(mass, 8)
        uniq, first_idx, inv = np.unique(flat, axis=0, return_index=True, return_inverse=True)
        uniques = uniq.reshape(-1, 4, 2)

        # Map each (address, hex_layer) to its unique polygon index
        refs = inv[flat_idx]
        return uniques, refs
    else:
        return polys, np.array([range(num_pts)])


# ---------- b_oct-native global hex mesh ------------------------------------

class HexMesh:
    """Shared-vertex hex mesh spanning one or more H9 layers.

    The vertex pool is sized to the **finest** layer requested.  Coarser-layer
    face arrays index into the same pool — their vertices are a strict subset
    of the finest lattice (every L-scale lattice point is also a (L+δ)-scale
    point, by construction of the H9 ternary hierarchy).

    Construction is dict-based: each unique vertex is keyed by its exact
    integer lattice coordinates ``(ia, ib, oid)`` — no ``np.unique`` and no
    floating-point comparison.  Edge ownership between layers is derivable
    from the same integer keys via pure arithmetic (no collinearity test).

    Access
    ------
    ``mesh.pts``       → unique vertex pool, ``Points`` in b_oct.
    ``mesh.faces``     → ``(N, 6)`` face array for the finest layer (property).
    ``mesh[L]``        → ``(N_L, 6)`` face array for layer *L*.
    ``mesh.layers``    → sorted tuple of available layer indices.
    ``mesh.fine``      → finest layer index.
    ``mesh.ia``, ``mesh.ib`` → per-vertex integer lattice keys (finest scale).

    Usage
    -----
    Single layer::

        mesh   = HexMesh.create(3, reg)
        ll     = reg.project(mesh.pts, [b_oct, g_gcd])
        hexes  = ll.coords[mesh.faces]          # (N, 6, 2)

    Multi-layer — coarser layers densified by the fine vertex pool::

        mesh   = HexMesh.create(range(4), reg)
        ll     = reg.project(mesh.pts, [b_oct, g_gcd])
        coarse = ll.coords[mesh[0]]             # 12 hexes, 6 verts each
        fine   = ll.coords[mesh[3]]             # 12*729 hexes

    Edge densification (structural, no lerp)::

        dense  = mesh.densify(0)                # L0 edges ↦ fine vert indices
        polys  = ll.coords[dense]               # (12, 6*(3^δ+1), 2)
    """

    def __init__(self, pts, faces_dict, fine, ia, ib, vert_dict, addrs_dict=None):
        self.pts  = pts                          # unique b_oct Points
        self._fl  = faces_dict                   # {layer: (N_L, 6) int32}
        self.fine = fine                         # finest layer index
        self.ia   = ia                           # (M,) int64, finest-scale
        self.ib   = ib                           # (M,) int64, finest-scale
        self._vd  = vert_dict                    # (ia, ib, oid) → vertex idx
        self._ad  = addrs_dict or {}             # {layer: [uuid.UUID, ...]}
        self.alt = None
        self.uv_o = np.array(list(self._vd.keys()), dtype=np.int64)

    @property
    def faces(self) -> np.ndarray:
        """Face index array for the finest layer."""
        return self._fl[self.fine]

    @property
    def addrs(self) -> np.ndarray:
        """UUID address list for the finest layer."""
        return self._ad.get(self.fine, [])

    def addr(self, L: int) -> list:
        """UUID address list for layer L."""
        return self._ad.get(int(L), [])

    @property
    def layers(self) -> tuple:
        """Sorted tuple of layer indices present in this mesh."""
        return tuple(sorted(self._fl))

    def __getitem__(self, L: int) -> np.ndarray:
        """Face index array for layer *L*."""
        return self._fl[int(L)]

    def __repr__(self) -> str:
        n = len(self.pts) if self.pts is not None else 0
        h = {L: len(f) for L, f in self._fl.items()}
        return f'HexMesh(verts={n}, hexes={h})'

    def _get_verts(self, idx_a: int, idx_b: int, lev: int):
        """Recursively find vertices between idx_a and idx_b.

        Returns a list of vertex indices from idx_a to idx_b inclusive.

        For same-oid or same-mode edges, raw (ia, ib) interpolation is used.
        For cross-mode edges (one mo==0 endpoint, one mo==1 endpoint), the
        mode-0 endpoint's ib is negated to convert it to mode-1 space before
        interpolating; all intermediate vertices live on the mode-1 face and
        are stored under the mode-1 oid with negative ib.
        """
        if lev >= self.fine:
            return [idx_a, idx_b]

        oa = int(self.pts.oid[idx_a])
        ob = int(self.pts.oid[idx_b])
        ua, va = int(self.ia[idx_a]), int(self.ib[idx_a])
        ub, vb = int(self.ia[idx_b]), int(self.ib[idx_b])
        du = (ub - ua) // 3

        if oa == ob:
            dv = (vb - va) // 3
            u1, v1 = ua + du, va + dv
            u2, v2 = ua + 2 * du, va + 2 * dv
            idx_m1 = self._vd.get((u1, v1, oa))
            if idx_m1 is None:
                idx_m1 = self._vd.get((u1, v1, ob))
            idx_m2 = self._vd.get((u2, v2, ob))
            if idx_m2 is None:
                idx_m2 = self._vd.get((u2, v2, oa))
        else:
            mo_a = int(H9O.oid_mo[oa])
            mo_b = int(H9O.oid_mo[ob])
            if mo_a == mo_b:
                # Same mode (both mo==0 or both mo==1), different octant.
                dv = (vb - va) // 3
                u1, v1 = ua + du, va + dv
                u2, v2 = ua + 2 * du, va + 2 * dv
                idx_m1 = self._vd.get((u1, v1, oa))
                if idx_m1 is None:
                    idx_m1 = self._vd.get((u1, v1, ob))
                idx_m2 = self._vd.get((u2, v2, ob))
                if idx_m2 is None:
                    idx_m2 = self._vd.get((u2, v2, oa))
            elif mo_a == 0:
                # a is mo==0, b is mo==1.  Convert a to mode-1 space: (ua, -va).
                # All intermediates lie on the mode-1 face (ob).
                va_m1 = -va
                dv = (vb - va_m1) // 3
                u1, v1 = ua + du, va_m1 + dv
                u2, v2 = ua + 2 * du, va_m1 + 2 * dv
                idx_m1 = self._vd.get((u1, v1, ob))
                if idx_m1 is None:
                    idx_m1 = self._vd.get((u1, v1, oa))
                idx_m2 = self._vd.get((u2, v2, ob))
                if idx_m2 is None:
                    idx_m2 = self._vd.get((u2, v2, oa))
            else:
                # a is mo==1, b is mo==0.  Convert b to mode-1 space: (ub, -vb).
                # All intermediates lie on the mode-1 face (oa).
                vb_m1 = -vb
                dv = (vb_m1 - va) // 3
                u1, v1 = ua + du, va + dv
                u2, v2 = ua + 2 * du, va + 2 * dv
                idx_m1 = self._vd.get((u1, v1, oa))
                if idx_m1 is None:
                    idx_m1 = self._vd.get((u1, v1, ob))
                idx_m2 = self._vd.get((u2, v2, oa))
                if idx_m2 is None:
                    idx_m2 = self._vd.get((u2, v2, ob))

        if idx_m1 is None or idx_m2 is None:
            raise KeyError(
                f"densify: missing vertex at lev={lev} between "
                f"idx={idx_a}(oid={oa}, ia={ua}, ib={va}) and "
                f"idx={idx_b}(oid={ob}, ia={ub}, ib={vb}): "
                f"1/3=({u1},{v1}) found={idx_m1}  "
                f"2/3=({u2},{v2}) found={idx_m2}"
            )

        path1 = self._get_verts(idx_a,  idx_m1, lev + 1)
        path2 = self._get_verts(idx_m1, idx_m2, lev + 1)
        path3 = self._get_verts(idx_m2, idx_b,  lev + 1)

        return path1[:-1] + path2[:-1] + path3

    def densify(self, layer: int) -> np.ndarray:
        """Return densified face indices for layer *L* using fine-layer verts."""
        delta = self.fine - layer
        if delta <= 0:
            return self._fl[layer]

        # Every edge at 'layer' contains 3^delta segments at 'fine'
        v_per_e = 3 ** delta
        fl = self._fl[layer]
        n_hex = len(fl)

        # We return a loop of vertices for each hex.
        # Total vertices in the densified loop = 6 edges * v_per_e segments
        out = np.empty((n_hex, 6 * v_per_e), dtype=np.int32)

        for hi in range(n_hex):
            hx = fl[hi]
            full_loop = []
            for e in range(6):
                idx_start = hx[e]
                idx_end = hx[(e + 1) % 6]

                # Get the indices along the edge (start...end)
                edge_indices = self._get_verts(idx_start, idx_end, layer)

                # Add all points except the last one to maintain a continuous loop
                full_loop.extend(edge_indices[:-1])

            out[hi] = full_loop

        return out

    # def _get_verts(self, a, b, lev: int):
    #     if lev < self.fine:
    #         ud, vd = (b[0] - a[0]) // 3, (b[1] - a[1]) // 3
    #         p = a[0] + ud, a[1] + vd
    #         if p in self.uv_o[:, :-1]:  # if p[0] in self.ia and p[1] in self.ib:
    #             lft = self._get_verts(a, p, lev + 1)
    #             rgt = self._get_verts(p, b, lev + 1)
    #             return lft + rgt[1:]
    #         p = a[0] + (ud << 1), a[1] + (vd << 1)
    #         if p in self.uv_o[:, :-1]:  # if p[0] in self.ia and p[1] in self.ib:
    #             lft = self._get_verts(a, p, lev + 1)
    #             rgt = self._get_verts(p, b, lev + 1)
    #             return lft + rgt[1:]
    #     return a, b
    #
    # # ---- edge densification -------------------------------------------
    # def densify(self, layer: int) -> np.ndarray:
    #     """Return densified face indices for layer *L* using fine-layer verts."""
    #     delta  = self.fine - layer
    #     if delta <= 0:
    #         return self._fl[layer]
    #     v_per_e = int(3 ** delta)
    #     fl     = self._fl[layer]                          # (N_L, 6)
    #     n_hex  = len(fl)
    #     out    = np.empty((n_hex, 6, v_per_e), dtype=np.int32)
    #     for hi, hx in enumerate(fl):
    #         for e in range(6):
    #             j = (e + 1) % 6
    #             xe, xj = hx[e], hx[j]
    #             idx_auv = self.ia[xe], self.ib[xe],                       # (N_L,) start vertex
    #             idx_buv = self.ia[xj], self.ib[xj],                       # (N_L,) start vertex
    #             vrts = self._get_verts(idx_auv, idx_buv, layer)
    #             out[hi, e] = vrts
    #     return out

    # ---- supercell helpers ---------------------------------------------
    @staticmethod
    def _supercell_uvs(layer: int, mode: int, reduce: int = 0):
        """Return integer UV origins for all mode-*reduce* supercells at *layer*.

        Each row is ``[ia, ib, scale]`` where ``(ia, ib)`` is the supercell
        centre in the integer UV coordinate system used by ``_raw_uv``, and
        ``scale`` is always 1 (leaf positions — the hex template is at unit
        scale).

        Integer UV ↔ float b_oct conversion (used in ``create``):
            rx = ia * U1 / 3^layer
            ry = ib * V3 / 3^layer   (V3 = 3 * H9K.lattice.V)

        The two coordinate axes have *different* denominators because the
        hex lattice uses U = 2·cos(30°)·V for the horizontal spacing, while
        the polygon template stores v-coordinates in V3 = 3·V units to keep
        all values as small integers.

        Implementation note — why ``region_grid`` and not ``uv_grid``:
        ``region_grid(layer-1, mode)`` returns leaf positions as float xy in
        b_oct face units, accumulated top-down so that the root offset
        contributes at the highest scale (3^(layer-1)) and each successive
        child at one power of 3 lower.  Converting via ``ia = origin_x *
        3^layer / U1`` therefore gives the correct integer weight to every
        level of the path.

        ``uv_grid`` accumulates bottom-up (the initial queue entries carry
        scale = 3^levels, so *child* offsets get high weight and *parent*
        offsets are added at scale 1).  Multiplying afterwards by 3 fixes the
        single-level case (layer 1) but is wrong for layer 2+, where parent
        and child weights are swapped relative to the correct values.
        """
        if layer == 0:
            if mode == reduce:
                return np.array([[0, 0, 1]], dtype=np.int32)  # [ia, ib, scale]
            return np.empty((0, 3), dtype=np.int32)
        uvg = uv_grid(layer-1, mode, flatten=True)
        mask = uvg[:, 0] == reduce
        return uvg[mask, 1:]
        # result = [g[g[:, 0] == reduce, 1:] for g in uvg]
        # return result


    @classmethod
    def create(cls, layers, reg) -> 'HexMesh':
        """Build a shared-vertex mesh for one or more layers.

        Parameters
        ----------
        layers : int or iterable of int
            Layer(s) to include.  The finest layer determines the vertex pool.
        reg : Registrar

        Returns
        -------
        HexMesh
        """
        if isinstance(layers, int):
            layers = [layers]
        layers = sorted(set(int(L) for L in layers))
        finest = layers[-1]
        b_oct  = reg.domain('b_oct')

        # Prepare UV→xy conversion for the finest layer
        # Layer 0 = 12 hexagons.
        div_f = float(3 ** finest)
        U1    = H9K.lattice.U
        V3    = 3.0 * H9K.lattice.V

        # --- build finest layer — UV integer keys, centroid in pool -----
        alt = cls._alt_uv(finest)
        uvo_list, alt_hexes = alt

        N_f      = len(alt_hexes)
        vert_dict: dict[tuple, int] = {}
        vert_xy:   list = []
        vert_oid:  list = []
        ia_list:   list = []
        ib_list:   list = []
        face_idx   = np.empty((N_f, 6), dtype=np.int32)

        poly_eps = 1.0 - 1e-14  # pull polygon verts slightly off triangle edges as needed.

        def _add_or_alias(ia, ib, o, cx, cy):
            """Look up (ia, ib, o) in vert_dict; create if absent.

            Seam hex verts 0-3 are stored under the mo==0 oid; verts 4-5
            (the mo==1 side's interior points) are stored under their mo==1
            oid with mo==1 UV coords.  _get_verts handles the Y-flip when
            searching across the seam.  Returns the vertex index.
            """
            key = (ia, ib, o)
            idx = vert_dict.get(key)
            if idx is None:
                idx = len(vert_xy)
                vert_dict[key] = idx
                rx = ia * U1 / div_f
                ry = ib * V3 / div_f
                rx = cx + (rx - cx) * poly_eps
                ry = cy + (ry - cy) * poly_eps
                vert_xy.append((rx, ry))
                vert_oid.append(o)
                ia_list.append(ia)
                ib_list.append(ib)
            return idx

        for h, hverts in enumerate(alt_hexes):
            # Float centroid = average of the 6 polygon vertices.
            cx = sum(uvo_list[v][0] for v in hverts) * U1 / (6.0 * div_f)
            cy = sum(uvo_list[v][1] for v in hverts) * V3 / (6.0 * div_f)
            for vi, gv in enumerate(hverts):
                ia, ib, o = uvo_list[gv]
                face_idx[h, vi] = _add_or_alias(ia, ib, o, cx, cy)

        # --- cache coarser-layer UV data (needed for both synthetic verts and face arrays)
        coarser_uv = {}
        for L in layers[:-1]:
            coarser_uv[L] = cls._raw_uv(L)

        # --- synthetic edge-interior vertices for densification ---
        # For a coarser layer L, the 1/3, 2/3, … positions along each hex edge
        # are required by _get_verts but may not coincide with any fine-grid hex
        # vertex.  Add them to the vertex pool here (before freezing arrays) so
        # that _get_verts can always find them by (ia, ib, oid) lookup.
        #
        # For cross-mode edges (one mo==0 endpoint, one mo==1 endpoint), the
        # interpolation is done in mode-1 space: the mo==0 endpoint's ib is
        # negated before computing the step.  All intermediate vertices are
        # stored under the mode-1 oid with a negative ib.
        for L in layers[:-1]:
            factor = int(3 ** (finest - L))
            uv7_c, oc7_c = coarser_uv[L]
            for h in range(len(uv7_c)):
                for e in range(6):
                    e_next = (e + 1) % 6
                    ia_a = int(uv7_c[h, e,      0]) * factor
                    ib_a = int(uv7_c[h, e,      1]) * factor
                    oa   = int(oc7_c[h, e])
                    ia_b = int(uv7_c[h, e_next, 0]) * factor
                    ib_b = int(uv7_c[h, e_next, 1]) * factor
                    ob   = int(oc7_c[h, e_next])
                    dia  = ia_b - ia_a

                    if oa == ob:
                        ib_start, dib, o_mid = ib_a, ib_b - ib_a, oa
                    else:
                        mo_a_v = int(H9O.oid_mo[oa])
                        mo_b_v = int(H9O.oid_mo[ob])
                        if mo_a_v == mo_b_v:
                            ib_start, dib, o_mid = ib_a, ib_b - ib_a, oa
                        elif mo_a_v == 0:
                            # a is mo==0, b is mo==1: convert a to mode-1 space
                            ib_start = -ib_a
                            dib      = ib_b - ib_start
                            o_mid    = ob
                        else:
                            # a is mo==1, b is mo==0: convert b to mode-1 space
                            ib_start = ib_a
                            dib      = (-ib_b) - ib_start
                            o_mid    = oa

                    for k in range(1, factor):
                        ia_m  = ia_a     + k * dia  // factor
                        ib_m  = ib_start + k * dib  // factor
                        key   = (ia_m, ib_m, o_mid)
                        if key in vert_dict:
                            continue
                        alt_o   = ob if o_mid == oa else oa
                        alt_key = (ia_m, ib_m, alt_o)
                        if alt_key in vert_dict:
                            continue
                        # Vertex genuinely absent — add synthetic entry.
                        idx = len(vert_xy)
                        vert_dict[key] = idx
                        rx = ia_m * U1 / div_f
                        ry = ib_m * V3 / div_f
                        vert_xy.append((rx, ry))
                        vert_oid.append(o_mid)
                        ia_list.append(ia_m)
                        ib_list.append(ib_m)

        pts    = Points(np.array(vert_xy, dtype=np.float64), b_oct, oid=np.array(vert_oid, dtype=np.int32))
        ia_arr = np.array(ia_list, dtype=np.int64)
        ib_arr = np.array(ib_list, dtype=np.int64)

        faces_dict = {finest: face_idx}

        # --- coarser layers: scale UV keys by 3^(finest-L), look up in pool
        for L in layers[:-1]:
            factor       = int(3 ** (finest - L))
            uv7_c, oc7_c = coarser_uv[L]
            face_c       = np.empty((len(uv7_c), 6), dtype=np.int32)
            for h in range(len(uv7_c)):
                for vi in range(6):
                    ia = int(uv7_c[h, vi, 0]) * factor
                    ib = int(uv7_c[h, vi, 1]) * factor
                    o  = int(oc7_c[h, vi])
                    key = (ia, ib, o)
                    idx = vert_dict.get(key)
                    if idx is None:
                        # Seam vertex: stored under canonical (mo==0) oid.
                        for nb in H9O.oid_nb[o].tolist():
                            idx = vert_dict.get((ia, ib, nb))
                            if idx is not None:
                                break
                    face_c[h, vi] = idx
            faces_dict[L] = face_c

        # --- H9 UUID addresses for each hex at each layer -------------------
        from hhg9.h9.uuid_address import h9_bin_pts as _h9_bin_pts

        addrs_dict = {}
        for L in layers:
            fl = faces_dict[L]

            # Mode-safe centroid: seam hexes span mode-0 and mode-1 octants.
            # Averaging mixed-mode b_oct coords (where mode-1 has y negated) puts
            # the centroid outside any valid octant, causing the encoder to
            # return garbage nibbles (including 0x0F in body positions).
            # Fix: use only vertices whose mode matches the first vertex.
            oid_v        = pts.oid[fl]                           # (N, 6)
            mo_v         = H9O.oid_mo[oid_v]                    # (N, 6)
            primary_mo   = mo_v[:, 0]                            # (N,)
            match_mo     = (mo_v == primary_mo[:, None])         # (N, 6) bool
            coords_v     = pts.coords[fl]                        # (N, 6, 2)
            sum_xy       = (coords_v * match_mo[:, :, None]).sum(axis=1)
            count        = match_mo.sum(axis=1, keepdims=True).astype(float)
            centroid_xy  = sum_xy / count                        # (N, 2)
            centroid_oid = oid_v[:, 0].astype(np.int32)

            centroid_pts = Points(centroid_xy, b_oct, oid=centroid_oid)
            addrs_dict[L] = _h9_bin_pts(centroid_pts, L)

        offspring = cls(pts, faces_dict, finest, ia_arr, ib_arr, vert_dict, addrs_dict)
        offspring.alt = alt
        return offspring

    # ---- polygon-clipped construction ---------------------------------
    @classmethod
    def _nodes_overlap(cls, queue, oid, reg, b_oct, g_gcd, div_f, U1, V3, poly_bbox):
        """Bool array: does each descent node's lat/lon footprint hit the poly bbox.

        A node is a (mode, origin_uv, scale) supercell.  An H9 supercell is
        self-similar — its three mode-template hexes at the node's own scale
        tile the whole node region exactly — so the 18 hex vertices give an
        exact conservative footprint for pruning.
        """
        n = len(queue)
        if n == 0:
            return np.zeros(0, dtype=bool)
        all_uv = np.empty((n, 18, 2), dtype=np.float64)
        for i, (m, origin, scale) in enumerate(queue):
            verts = (H9P.hi[m][:, :6, :].astype(np.int64) * scale
                     + origin).reshape(18, 2)
            all_uv[i] = verts
        flat = all_uv.reshape(-1, 2)
        rx = flat[:, 0] * U1 / div_f
        ry = flat[:, 1] * V3 / div_f
        node_pts = Points(np.stack([rx, ry], axis=1), b_oct,
                          oid=np.full(len(flat), oid, dtype=np.int32))
        ll = reg.project(node_pts, [b_oct, g_gcd]).coords.reshape(n, 18, 2)
        lat, lon = ll[:, :, 0], ll[:, :, 1]
        with np.errstate(invalid='ignore'):
            latmin, latmax = np.nanmin(lat, axis=1), np.nanmax(lat, axis=1)
            lonmin, lonmax = np.nanmin(lon, axis=1), np.nanmax(lon, axis=1)
        plat0, plat1, plon0, plon1 = poly_bbox
        keep = ((latmin <= plat1) & (latmax >= plat0) &
                (lonmin <= plon1) & (lonmax >= plon0))
        return np.nan_to_num(keep, nan=False)

    @classmethod
    def _clip_descent(cls, L, oid, reg, b_oct, g_gcd, poly_bbox, U1, V3):
        """Pruned hierarchical descent: mode-0 supercell origins for *oid* at *L*.

        Subdivides the H9 supercell tree breadth-first, discarding any branch
        whose footprint misses the polygon bounding box.  The surviving queue
        is bounded by the polygon's area, not the globe — so deep layers over
        a small region stay tractable (no 12·9**L global enumeration).
        """
        from hhg9.h9 import H9C, H9R
        mode = int(H9O.oid_mo[oid])
        if L == 0:
            return [(0, 0)] if mode == 0 else []
        levels = L - 1
        mode_cells = [np.asarray(H9R.downs), np.asarray(H9R.ups)]
        mode_ofs = H9C.off_uv[[H9R.downs, H9R.ups]].astype(np.int64)
        mode_ofs[:, :, 0] *= 3                       # polygon grid: U is 3U
        div_f = float(3 ** L)
        scale0 = 3 ** levels
        queue = [(int(H9C.mode[k]), o * scale0, scale0)
                 for k, o in zip(mode_cells[mode], mode_ofs[mode])]
        for depth in range(levels + 1):
            keep = cls._nodes_overlap(queue, oid, reg, b_oct, g_gcd,
                                      div_f, U1, V3, poly_bbox)
            queue = [q for q, k in zip(queue, keep) if k]
            if depth == levels or not queue:
                break
            nxt = []
            for m, origin, scale in queue:
                ks = scale // 3
                for k, o in zip(mode_cells[m], mode_ofs[m]):
                    nxt.append((int(H9C.mode[k]), origin + o * ks, ks))
            queue = nxt
        return [(int(origin[0]), int(origin[1]))
                for m, origin, scale in queue if m == 0]

    @classmethod
    def _assemble_clipped(cls, kept_by_oid, L):
        """Build (uv7, oid7) hex arrays from surviving mode-0 supercells.

        Mirrors the ``_raw_uv`` per-supercell template placement and the
        verts 4/5 boundary reflection, restricted to the kept supercells.
        """
        template = H9P.hi[0, :3]                     # (3, 7, 2) mode-0
        parts_uv, parts_oid, parts_c2 = [], [], []
        for oid, origins in kept_by_oid.items():
            if not origins:
                continue
            org = np.array(origins, dtype=np.int64)  # (S, 2)
            hex_uv = (template[None, :, :, :] + org[:, None, None, :]).reshape(-1, 7, 2)
            n = len(hex_uv)
            parts_uv.append(hex_uv)
            parts_oid.append(np.full((n, 7), oid, dtype=np.int32))
            parts_c2.append(np.tile(np.arange(3, dtype=np.int32), len(org)))
        if not parts_uv:
            return np.empty((0, 7, 2), np.int64), np.empty((0, 7), np.int32)
        out_uv = np.concatenate(parts_uv)
        out_oid = np.concatenate(parts_oid)
        c2 = np.concatenate(parts_c2)
        face_oid = out_oid[:, 0].copy()
        # Mode-0 octants own the seam reflection; mode-1 octants leave their
        # v4/v5 in place and share vertices via the pooled dedup.  Without
        # this guard the mode-0 edge conditions below misfire on mode-0 sub-
        # hexes inside mode-1 octants (whose face triangle is inverted), and
        # ``H9O.oid_nb[face_oid, c2]`` returns the wrong neighbour.
        is_m0 = (H9O.oid_mo[face_oid] == 0)

        s = 3 ** L
        u4, v4 = out_uv[:, 4, 0], out_uv[:, 4, 1]
        u5, v5 = out_uv[:, 5, 0], out_uv[:, 5, 1]
        ext4 = (v4 > s) | (u4 - v4 > 2 * s) | (u4 + v4 < -2 * s)
        ext5 = (v5 > s) | (u5 - v5 > 2 * s) | (u5 + v5 < -2 * s)
        ext = is_m0 & ext4 & ext5
        if ext.any():
            out_uv[ext, 4, 0] = out_uv[ext, 2, 0]
            out_uv[ext, 4, 1] = -out_uv[ext, 2, 1]
            out_uv[ext, 5, 0] = out_uv[ext, 1, 0]
            out_uv[ext, 5, 1] = -out_uv[ext, 1, 1]
            nb = H9O.oid_nb[face_oid[ext], c2[ext]].astype(np.int32)
            out_oid[ext, 4] = nb
            out_oid[ext, 5] = nb
        return out_uv.astype(np.int64), out_oid

    @classmethod
    def create_clipped(cls, layers, ll_polygon, reg) -> 'HexMesh':
        """Build a hex mesh clipped to a lat/lon polygon — no global enumeration.

        Unlike :meth:`create`, which enumerates every hex on the globe, this
        descends the H9 supercell hierarchy while pruning branches outside the
        polygon's bounding box, then keeps only hexes whose centroid lies
        inside the polygon.  Cost scales with the polygon's area, so deep
        layers over a bounded region remain feasible.

        Parameters
        ----------
        layers : int or iterable of int
            Layer(s) to include.  Coarser layers form the ancestry "nest".
        ll_polygon : (P, 2) array
            Polygon vertices as ``[lat, lon]`` in WGS84 degrees.
        reg : Registrar

        Returns
        -------
        HexMesh
            ``mesh[L]`` gives the clipped face array for each requested layer.
            ``mesh.densify`` is unavailable on a clipped mesh (no synthetic
            edge vertices) — use the per-layer face arrays directly.
        """
        from matplotlib.path import Path as _Path
        from hhg9.h9.uuid_address import h9_bin_pts as _h9_bin_pts

        if isinstance(layers, int):
            layers = [layers]
        layers = sorted(set(int(L) for L in layers))
        finest = layers[-1]
        b_oct = reg.domain('b_oct')
        g_gcd = reg.domain('g_gcd')
        U1 = H9K.lattice.U
        V3 = 3.0 * H9K.lattice.V
        div_f = float(3 ** finest)

        poly = np.asarray(ll_polygon, dtype=np.float64)        # (P, 2) [lat, lon]
        poly_path = _Path(np.column_stack([poly[:, 1], poly[:, 0]]))   # (lon, lat)
        poly_bbox = (poly[:, 0].min(), poly[:, 0].max(),
                     poly[:, 1].min(), poly[:, 1].max())

        # --- per-layer pruned descent + exact centroid clip ----------------
        clip_uv7 = {}
        for L in layers:
            kept = {oid: cls._clip_descent(L, oid, reg, b_oct, g_gcd,
                                           poly_bbox, U1, V3)
                    for oid in range(8)}
            uv7, oid7 = cls._assemble_clipped(kept, L)
            if len(uv7):
                # Filter by the *match-mo* centroid — the same one the UUID
                # encoder uses below.  ``uv7[:, 6]`` is the template centroid
                # in the face-oid's b_oct frame; for seam-straddling hexes it
                # falls exactly on the octant edge and projects to lon=±180
                # (or any seam meridian), causing the polygon containment test
                # to reject the hex even though its geometric centroid is
                # well inside the bbox.  The match-mo centroid uses only the
                # face-oid-frame vertices (v0..v3 for a reflected hex), so it
                # stays inside the face and projects to a representative
                # interior lat/lon.
                vu = uv7[:, :6, 0].astype(np.float64) * U1 / (3 ** L)
                vv = uv7[:, :6, 1].astype(np.float64) * V3 / (3 ** L)
                coords_v = np.stack([vu, vv], axis=-1)             # (n, 6, 2)
                mo_v = H9O.oid_mo[oid7[:, :6]]
                match_mo = (mo_v == mo_v[:, :1])
                weights = match_mo[:, :, None].astype(np.float64)
                cen_xy = (coords_v * weights).sum(axis=1) / weights.sum(axis=1)
                cpts = Points(cen_xy, b_oct,
                              oid=oid7[:, 0].astype(np.int32))
                cll = reg.project(cpts, [b_oct, g_gcd]).coords
                inside = poly_path.contains_points(
                    np.column_stack([cll[:, 1], cll[:, 0]]))
                uv7, oid7 = uv7[inside], oid7[inside]
            clip_uv7[L] = (uv7, oid7)

        # --- shared vertex pool (finest first, coarser keys scaled up) -----
        vert_dict, vert_xy, vert_oid, ia_list, ib_list = {}, [], [], [], []

        def _add(ia, ib, o):
            key = (ia, ib, o)
            idx = vert_dict.get(key)
            if idx is None:
                idx = len(vert_xy)
                vert_dict[key] = idx
                vert_xy.append((ia * U1 / div_f, ib * V3 / div_f))
                vert_oid.append(o)
                ia_list.append(ia)
                ib_list.append(ib)
            return idx

        faces_dict = {}
        for L in [finest] + [x for x in layers if x != finest]:
            uv7, oid7 = clip_uv7[L]
            fct = int(3 ** (finest - L))
            face = np.empty((len(uv7), 6), dtype=np.int32)
            for h in range(len(uv7)):
                for vi in range(6):
                    face[h, vi] = _add(int(uv7[h, vi, 0]) * fct,
                                       int(uv7[h, vi, 1]) * fct,
                                       int(oid7[h, vi]))
            faces_dict[L] = face

        pts = Points(np.array(vert_xy, dtype=np.float64).reshape(-1, 2), b_oct,
                     oid=np.array(vert_oid, dtype=np.int32))
        ia_arr = np.array(ia_list, dtype=np.int64)
        ib_arr = np.array(ib_list, dtype=np.int64)

        # --- H9 UUID addresses per hex per layer ---------------------------
        addrs_dict = {}
        for L in layers:
            fl = faces_dict[L]
            if len(fl) == 0:
                addrs_dict[L] = []
                continue
            oid_v = pts.oid[fl]
            mo_v = H9O.oid_mo[oid_v]
            match_mo = (mo_v == mo_v[:, :1])
            coords_v = pts.coords[fl]
            centroid_xy = ((coords_v * match_mo[:, :, None]).sum(axis=1) /
                           match_mo.sum(axis=1, keepdims=True).astype(float))
            centroid_pts = Points(centroid_xy, b_oct, oid=oid_v[:, 0].astype(np.int32))
            addrs_dict[L] = _h9_bin_pts(centroid_pts, L)

        return cls(pts, faces_dict, finest, ia_arr, ib_arr, vert_dict, addrs_dict)

    @classmethod
    def _alt_results(cls, layer):
        """Generate corrected hex verts at *layer* in integer UV space.

        Each hex is assembled as:
            hex_uv[h, vi] = scale * template[c2, vi] + origin_uv[h]
        where scale = 1 for all supercell leaf nodes (see ``_supercell_uvs``).
        """
        template = H9P.hi[:, :, :4, :]  # mode, c2, 4 pts.
        if layer == 0:
            result = []
            for mode in range(2):
                pts = template[mode]
                coords = []
                hhm = []
                for hhp in pts:
                    hh = []
                    for (u, v) in hhp:
                        pt = (u, v)
                        try:
                            idx = coords.index(pt)
                        except ValueError:
                            idx = len(coords)
                            coords.append(pt)
                        hh.append(idx)
                    hhm.append(hh)
                result.append((np.array(coords, dtype=np.int32), [], np.array(hhm, dtype=np.int32)))
            return result
        offsets = np.array([  # scale is 1. so we carve off the redundant 1.
            uv_grid(layer - 1, 0)[:, :3],
            uv_grid(layer - 1, 1)[:, :3]
        ])
        modes = offsets[:, :, 0].astype(np.int32)
        xy_offsets = offsets[:, :, 1:]
        coords = template[modes] + xy_offsets[:, :, np.newaxis, np.newaxis, :]
        coords = coords.reshape((2, -1, 4, 2))  # Drop the c2.
        results = []
        for group in coords:
            all_pts = group.reshape((-1, 2))
            unique_pts, inverse = np.unique(all_pts, axis=0, return_inverse=True)
            row_ids = inverse.reshape(-1, 4)  # Shape: [N, 4]
            endpoints = row_ids[:, [0, 3]]  # Get start and end IDs for every row
            sorted_endpoints = np.sort(endpoints, axis=1)
            edge_keys = sorted_endpoints[:, 0].astype(np.int64) * (unique_pts.shape[0] + 1) + sorted_endpoints[:, 1]
            # Get indices of sorted keys to find neighbors
            sort_idx = np.argsort(edge_keys)
            sorted_keys = edge_keys[sort_idx]
            is_match = sorted_keys[:-1] == sorted_keys[1:]
            idx_a = sort_idx[:-1][is_match]
            idx_b = sort_idx[1:][is_match]
            flip_mask = row_ids[idx_a, 0] != row_ids[idx_b, 3]
            hex_indices = np.zeros((len(idx_a), 6), dtype=int)
            hex_indices[:, 0:4] = row_ids[idx_a]

            b_mids = row_ids[idx_b, 1:3]
            b_mids[flip_mask] = b_mids[flip_mask, ::-1]  # Flip [p5, p4] to [p4, p5]
            hex_indices[:, 4:6] = b_mids

            # all_row_indices = np.arange(len(row_ids))
            matched_mask = np.zeros(len(row_ids), dtype=bool)
            matched_mask[idx_a] = True
            matched_mask[idx_b] = True
            orphan_rows = row_ids[~matched_mask]

            results.append([unique_pts, hex_indices, orphan_rows])
        return results

    @classmethod
    def _stitch_seam(cls, m0_bundle, m1_bundle, c2):
        m0_orphans, pts0, val0 = m0_bundle
        m1_orphans, pts1, val1 = m1_bundle

        # Transform mode 1 into mode 0 coordinate space pts
        val1_trans = val1 * [1, -1]

        # Match endpoints
        keys0 = cls._get_endpoint_keys(val0)
        keys1 = cls._get_endpoint_keys(val1_trans)
        map1 = {k: i for i, k in enumerate(keys1)}

        matches = []
        for i0, k0 in enumerate(keys0):
            if k0 in map1:
                matches.append((i0, map1[k0]))

        if not matches: return None, {}
        matches = np.array(matches)
        idx0, idx1 = matches[:, 0], matches[:, 1]

        # Assemble with proper winding
        hexs = np.zeros((len(matches), 6), dtype=np.int32)
        hexs[:, 0:4] = m0_orphans[idx0]

        # WINDING FIX:
        # Check if Mode 0 Start matches Mode 1 Transformed Start
        # If it matches the End instead, we must flip the middle points
        m1_orphans_matched = m1_orphans[idx1]
        # Corrected comparison: Does Start match Start?
        start_match = np.all(val0[idx0, 0] == val1_trans[idx1, 0], axis=1)
        b_mids = m1_orphans_matched[:, 1:3]
        # If it's a Start-to-End match (not start-to-start),
        # the points p1, p2 are already in the "correct" return order.
        # If it's a Start-to-Start match, we must flip them.
        for i in range(len(b_mids)):
            if start_match[i]:
                hexs[i, 4:6] = b_mids[i, ::-1]  # Reverse for return trip
            else:
                hexs[i, 4:6] = b_mids[i]

        # Create the Remap Table
        remap = {}
        for i, (i0, i1) in enumerate(matches):
            p0_m0, p3_m0 = m0_orphans[idx0[i], [0, 3]]
            p0_m1, p3_m1 = m1_orphans[idx1[i], [0, 3]]
            if start_match[i]:
                remap[p0_m1], remap[p3_m1] = (p0_m0, c2), (p3_m0, c2)
            else:
                remap[p0_m1], remap[p3_m1] = (p3_m0, c2), (p0_m0, c2)

        return hexs, remap

    @classmethod
    def _alt_uv(cls, layer):
        # s2 = (3 ** layer) << 1
        s1 = 3 ** layer
        s2 = s1 << 1

        results = cls._alt_results(layer)
        m0_pts, m0_hex, m0_orphans = results[0]
        m1_pts, m1_hex, m1_orphans = results[1]
        m0_op, m1_op = m0_pts[m0_orphans], m1_pts[m1_orphans]

        # Mode 0 Masks
        m0c2_0 = (m0_op[:, 0, 1] == s1) & (m0_op[:, 3, 1] == s1)  # Y = s1
        m0c2_1 = (m0_op[:, 0, 0] - m0_op[:, 0, 1] == s2) & (m0_op[:, 3, 0] - m0_op[:, 3, 1] == s2)    # X - Y = s2
        m0c2_2 = (m0_op[:, 0, 0] + m0_op[:, 0, 1] == -s2) & (m0_op[:, 3, 0] + m0_op[:, 3, 1] == -s2)  # X + Y = -s2

        # Mode 1 Masks (Mirrored)
        m1c2_0 = (m1_op[:, 0, 1] == -s1) & (m1_op[:, 3, 1] == -s1)  # Y = -s1
        m1c2_1 = (m1_op[:, 0, 0] - m1_op[:, 0, 1] == -s2) & (m1_op[:, 3, 0] - m1_op[:, 3, 1] == -s2)  # X - Y = -s2
        m1c2_2 = (m1_op[:, 0, 0] + m1_op[:, 0, 1] == s2) & (m1_op[:, 3, 0] + m1_op[:, 3, 1] == s2)    # X + Y = s2

        masks = [
            (m0c2_0, m1c2_0),  # Equator: Both endpoints have Y = +/- s1
            (m0c2_1, m1c2_2),  # Forward: Both endpoints satisfy X - Y = +/- s2 (after Y-flip logic)
            (m0c2_2, m1c2_1),  # Backward: Both endpoints satisfy X + Y = +/- s2
        ]
        seam_hexs = []
        for seam_type in range(3):
            m0_mask, m1_mask = masks[seam_type]
            m0_bundle = (m0_orphans[m0_mask], m0_pts, m0_op[m0_mask])
            m1_bundle = (m1_orphans[m1_mask], m1_pts, m1_op[m1_mask])
            hexs, remap = cls._stitch_seam(m0_bundle, m1_bundle, seam_type)
            if hexs is not None:
                seam_hexs.append(hexs)
            # g_map = {tuple(m1_pts[k].tolist()): tuple((m0_pts[v].tolist(), c2)) for k, (v, c2) in remap.items()}
        # seam_hexs are in c2 order (consider oid_nb here!)
        # Need to return uvo_points, uv_faces.
        uvo = []
        uvo_lookup = {}
        hexes = []

        def _add_to_uvo(pts, o, repo):
            for pt in pts:
                u, v = int(pt[0]), int(pt[1])
                gp = (u, v, o)
                if gp not in uvo_lookup:
                    uvo_lookup[gp] = len(uvo)
                    uvo.append(gp)
                repo[gp] = uvo_lookup[gp]

        for oid in range(8):
            onb = H9O.oid_nb[oid].tolist()
            l_to_g = {}
            mo = H9O.oid_mo[oid]
            if mo == 0:
                for hx in m0_hex:
                    hx_0 = [m0_pts[p].tolist() for p in hx]
                    _add_to_uvo(hx_0, oid, l_to_g)
                    hexes.append([l_to_g[(u, v, oid)] for (u, v) in hx_0])
                for c2, nb in enumerate(onb):
                    oi = [oid, oid, oid, oid, nb, nb]
                    for hx in seam_hexs[c2]:
                        hx_0 = [m0_pts[p].tolist() for p in hx[:4]]
                        hx_1 = [m1_pts[p].tolist() for p in hx[4:]]
                        _add_to_uvo(hx_0, oid, l_to_g)
                        _add_to_uvo(hx_1, nb, l_to_g)
                        hexes.append([l_to_g[(u, v, o)] for (u, v), o in zip(hx_0 + hx_1, oi)])
            else:
                for hx in m1_hex:
                    hx_1 = [m1_pts[p].tolist() for p in hx]
                    _add_to_uvo(hx_1, oid, l_to_g)
                    hexes.append([l_to_g[(u, v, oid)] for (u, v) in hx_1])
        return uvo, hexes

    @classmethod
    def get_canonical_gp(cls, u, v, oid, s1, s2):
        """
        Finds the lowest-ID octant that shares this physical coordinate
        and returns the point in that octant's coordinate space.
        """
        # 1. Identify which seams this point sits on
        seams = []
        if abs(v) == s1: seams.append(0)  # Equator
        if abs(u - v) == s2: seams.append(1)  # Forward Seam
        if abs(u + v) == s2: seams.append(2)  # Backward Seam

        if not seams:
            return (u, v, oid)  # Internal point, no sharing

        # 2. Find all sharing octants using the LUT
        # Start with the current octant
        sharing_oids = {oid}
        for s_type in seams:
            neighbor_oid = int(H9O.oid_nb[oid][s_type])
            sharing_oids.add(neighbor_oid)

            # At corners, a neighbor's neighbor might also share this point.
            # This catch-all handles the 4-way pole junctions.
            second_neighbor = H9O.oid_nb[neighbor_oid][(s_type + 1) % 3]
            # (This logic varies slightly based on your LUT structure,
            # but min() is the ultimate decider)

        master_oid = min(sharing_oids)

        if master_oid == oid:
            return (u, v, oid)  # We are the owner

        # 3. Coordinate Transformation
        # If we are handing ownership to a neighbor of a DIFFERENT mode,
        # apply the Y-flip. If same mode, keep current (u, v).
        target_u, target_v = u, v
        if H9O.oid_mo[oid] != H9O.oid_mo[master_oid]:
            target_v = -v

        return (target_u, target_v, master_oid)

    @classmethod
    def _get_endpoint_keys(cls, pts):
        # pts shape: [N, 4, 2]
        # We take point 0 and point 3 as the "bridge"
        endpoints = pts[:, [0, 3], :].reshape(-1, 4)  # [N, (x0, y0, x3, y3)]
        # We sort the start/end pairs so direction (p0->p3 vs p3->p0) doesn't matter
        # p0 = endpoints[:, :2]
        # p3 = endpoints[:, 2:]

        # Simple Cantor-like pairing or just a view-based comparison
        # Sorting ensures that a [p0, p3] and [p3, p0] match
        sorted_ends = np.sort(endpoints.reshape(-1, 2, 2), axis=1).reshape(-1, 4)
        return [tuple(row) for row in sorted_ends]

    @classmethod
    def _raw_uv(cls, layer):
        """Generate corrected hex verts at *layer* in integer UV space.

        Returns ``(N, 7, 2)`` int64 and ``(N, 7)`` int32.
        Verts 0-5 are the hex polygon vertices (clockwise); vert 6 is the
        centroid, used for vertex deduplication in ``create``.
        N == 12 * 9**layer (all hexes across all 8 octahedral faces).

        Coordinate system
        -----------------
        Integer UV (ia, ib) map to b_oct float (rx, ry) as:
            rx = ia * U1 / 3^layer
            ry = ib * V3 / 3^layer
        where U1 = H9K.lattice.U and V3 = 3 * H9K.lattice.V.

        Template
        --------
        ``H9P.hi[0, c2]`` holds the 6 polygon vertices + centroid for a
        unit-scale mode-0 hex of c2-type, stored as raw integer lattice
        coordinates (the same integers as in polygon.py key (1,0), i.e.
        values like 3, 2, 0, −1 …).  Multiplied by U1 or V3 and divided by
        3^layer they yield the same float positions as ``H9P.hx[0, c2]``
        used in ``_raw``.

        Each hex is assembled as:
            hex_uv[h, vi] = scale * template[c2, vi] + origin_uv[h]
        where scale = 1 for all supercell leaf nodes (see ``_supercell_uvs``).

        Boundary correction
        -------------------
        Hexes on the edge of a triangular face have verts 4 and 5 placed in
        the neighbouring face.  The face domain for a mode-0 face at
        scale s = 3^layer is the triangle with integer UV vertices
        (−3s/3, s/3), (3s/3, s/3), (0, −2s/3) → scaled up: (−3,1),(3,1),(0,−2)
        times s.  A vertex (u,v) lies outside (EXT) when any edge condition
        is violated:
            v > s          (above top edge)
            u − v > 2s     (right of lower-right edge)
            u + v < −2s    (left of lower-left edge)

        When *both* verts 4 and 5 of a mode-0 hex are EXT, the hex straddles
        the face boundary.  The correction reflects them across the shared
        edge (negate v, keep u) and reassigns their oid to the neighbour face
        via H9O.oid_nb[face_oid, c2].

        Mode-1 oid hexes are always interior at every layer and are never
        corrected (the is_m0 guard prevents it).
        """
        # H9P.hi[0, :3] — (3, 7, 2) integer UV template for mode-0, c2=0,1,2.
        # Axis 0 indexes c2 (hex sub-type within the supercell).
        # Axis 1 indexes vertex 0-5 then centroid (6).
        # Axis 2 is (u, v).
        template = H9P.hi[0, :3]

        parts_uv, parts_oid, parts_c2, parts_m0 = [], [], [], []
        for oid in range(8):
            mode = int(H9O.oid_mo[oid])
            # _supercell_uvs returns only mode-0 (reduce=0) supercell origins.
            # Mode-1 supercells within a mode-0 face are always interior and
            # do not need the boundary correction below, so we skip them here.
            uvs  = cls._supercell_uvs(layer, mode, 0)
            if not len(uvs):
                continue
            sc = len(uvs)   # number of mode-0 supercells for this oid

            # Broadcast: for each supercell, place all 3 c2-typed hexes.
            # uvs shape: (sc, 3) — columns [ia, ib, scale]
            # template shape: (3, 7, 2)
            # hex_uv shape before reshape: (sc, 3, 7, 2)
            hex_uv = (uvs[:, 2, None, None, None]
                      * template[None, :, :, :]
                      + uvs[:, None, None, :2]).reshape(-1, 7, 2)
            N = len(hex_uv)   # sc * 3
            parts_uv.append(hex_uv)
            parts_oid.append(np.full((N, 7), oid, dtype=np.int32))
            # c2 cycles 0,1,2 for every supercell
            parts_c2.append(np.tile(np.arange(3, dtype=np.int32), sc))
            parts_m0.append(np.full(N, mode == 0))

        out_uv   = np.concatenate(parts_uv)    # (total_N, 7, 2)
        out_oid  = np.concatenate(parts_oid)   # (total_N, 7)
        c2_flat  = np.concatenate(parts_c2)    # (total_N,)
        is_m0    = np.concatenate(parts_m0)    # (total_N,) bool

        # --- Boundary correction for verts 4 and 5 ---
        # The mode-0 triangular face domain at layer L has scale s = 3^layer.
        # Vertices taken from polygon.py key (4,0): (−3,1),(3,1),(0,−2) × s.
        # A vertex (u,v) is EXT when it violates any of the three edge half-planes.
        face_oid = out_oid[:, 0].copy()
        s  = 3 ** layer
        u4, v4 = out_uv[:, 4, 0], out_uv[:, 4, 1]
        u5, v5 = out_uv[:, 5, 0], out_uv[:, 5, 1]
        ext4 = (v4 > s) | (u4 - v4 > 2*s) | (u4 + v4 < -2*s)
        ext5 = (v5 > s) | (u5 - v5 > 2*s) | (u5 + v5 < -2*s)
        # Only correct when *both* outer verts are EXT and the hex is mode-0.
        ext  = is_m0 & ext4 & ext5
        if ext.any():
            # Reflect across the shared face edge: negate v, keep u.
            # Vert 4 mirrors vert 2; vert 5 mirrors vert 1 (from _raw convention).
            out_uv[ext, 4, 0] =  out_uv[ext, 2, 0]
            out_uv[ext, 4, 1] = -out_uv[ext, 2, 1]
            out_uv[ext, 5, 0] =  out_uv[ext, 1, 0]
            out_uv[ext, 5, 1] = -out_uv[ext, 1, 1]
            # Corrected verts belong to the neighbouring face.
            nb = H9O.oid_nb[face_oid[ext], c2_flat[ext]].astype(np.int32)
            out_oid[ext, 4] = nb
            out_oid[ext, 5] = nb

        return out_uv.astype(np.int64), out_oid


def clipped_ll(arr_ll: np.ndarray) -> list:
    """Simple convex polygon clip to 180/dateline."""
    result = []
    for row in arr_ll:
        # Clean out NaNs/Inf
        not_oob = np.isfinite(row).all(axis=1)  # Use .all() to ensure both x and y are finite
        verts = row[not_oob]

        if len(verts) == 0:
            continue

        # Calculate spans between subsequent points
        lons = verts[:, 1]
        spans = np.abs(np.diff(lons))

        jumps = np.where(spans > 180.0)[0]
        if jumps.size > 0:
            first_jump_idx = jumps[0] + 1
            result.append(verts[:first_jump_idx])
        else:
            result.append(verts)
    return result


def valid_ll(arr_ll: np.ndarray) -> np.ndarray:
    """True for rows where all coords are finite and lon span <= 180°.

    Parameters
    ----------
    arr_ll : (N, K, 2) float  —  last axis is [lat, lon].
    Returns
    -------
    [N, (...)] reduced co-oordinates.
    """
    fin   = np.isfinite(arr_ll).all(axis=(1, 2))
    spans = arr_ll[..., 1].max(axis=-1) - arr_ll[..., 1].min(axis=-1)
    ok = fin & (spans <= 180.0)
    return ok


def h9_postgis_hexagons(
        min_lon: float,
        min_lat: float,
        max_lon: float,
        max_lat: float,
        layer: int,
        max_samples: int = 1_000_000,
        reg=None,
) -> list[tuple[str, str, str]]:
    """
    Generate H9 hexagon polygons covering a bounding box at the given layer.

    Intended as the Python backend for the PostGIS ``h9_bbox_hexagons`` function.
    The SQL wrapper ``h9_fill_polygon`` then clips the results to an exact polygon
    using ``ST_Intersects``, so no geometry library is required here.

    Parameters
    ----------
    min_lon, min_lat, max_lon, max_lat : float
        Bounding box in WGS84 degrees.
    layer : int
        H9 hex layer (0..UUID_DEPTH).  Determines hexagon size.
    max_samples : int
        Safety cap on the number of grid sample points.  If the bbox/layer
        combination would require more, the step is coarsened automatically.
        Default 1_000_000.
    reg : Registrar, optional

    Returns
    -------
    list of (bin_uuid_str, wkt_polygon) tuples.
        hex9_uuid_str  : canonical UUID for the H9 Point
        bin_uuid_str  : canonical UUID string for the H9 bin at ``layer``.
        wkt_polygon   : WKT POLYGON with (lon lat) vertex order, ring closed.
    """
    from hhg9 import Registrar, Points
    from hhg9.h9.binning import hex_reduce, hex_parents, hex_from_pars
    from hhg9.h9.uuid_address import h9_encode, batch_nibbles_to_int
    import uuid as uuid_mod

    if reg is None:
        reg = Registrar()

    g_gcd = reg.domain('g_gcd')
    b_oct = reg.domain('b_oct')

    # Sampling step: hex inradius converted to degrees (equatorial approximation,
    # intentionally conservative — finer grid than needed at low latitudes).
    inradius_m = float(hex_props(layer)[2])
    step_deg = inradius_m / 111_320.0

    # Coarsen step if it would exceed the sample cap.
    lat_span = max_lat - min_lat
    lon_span = max_lon - min_lon
    n_lat = max(1, int(np.ceil(lat_span / step_deg)))
    n_lon = max(1, int(np.ceil(lon_span / step_deg)))
    if n_lat * n_lon > max_samples:
        step_deg = max(lat_span, lon_span) / int(max_samples ** 0.5)
        n_lat = max(1, int(np.ceil(lat_span / step_deg)))
        n_lon = max(1, int(np.ceil(lon_span / step_deg)))

    # Regular lat/lon sample grid.
    lats = np.linspace(min_lat, max_lat, n_lat)
    lons = np.linspace(min_lon, max_lon, n_lon)
    lon_g, lat_g = np.meshgrid(lons, lats)
    sample_lats = lat_g.ravel()
    sample_lons = lon_g.ravel()

    # Project to b_oct and bin into hexagons.
    coords = np.column_stack([sample_lats, sample_lons])
    b_pts = reg.project(Points(coords, g_gcd), [g_gcd, b_oct])

    # Use the lower-level binning steps so we can access hex_v directly.
    # hex_v shape: (H, layer+2) — H unique hexes, each with reversible address.
    hex_num, hex_v, _inv, _idx = hex_reduce(b_pts, layer)
    hx = int(hex_num)

    hex_xy, hex_oid, scale = hex_parents(b_oct, hex_v, hx, layer)
    hx_pts = hex_from_pars(b_oct, hex_xy, hex_oid, scale, hex_v[:, -1])

    # Derive bin UUIDs directly from hex_v — no approximation, no round-trip.
    # hex_v[:, :-1] = body (L+1 nibble cols), hex_v[:, -1] = reversible tail byte.
    body_l = hex_v[:, :-1].astype(np.uint8)          # (H, layer+1)
    tail_bytes = hex_v[:, -1].astype(np.uint8)        # (H,)
    key_tail = ((tail_bytes >> 4) & 0x0F)             # (p_c2 << 1) | r_mo

    uuid_nibs = np.zeros((hx, 32), dtype=np.uint8)
    uuid_nibs[:, :layer + 1] = body_l
    uuid_nibs[:, layer + 1] = key_tail
    bin_uuids = [uuid_mod.UUID(int=v) for v in batch_nibbles_to_int(uuid_nibs)]

    # Full depth-30 UUID: encode the representative sample for each unique hex.
    # hex_v only covers layer+2 nibbles so we re-encode from lat/lon.
    full_uuids = h9_encode(sample_lats[_idx], sample_lons[_idx], reg=reg)

    # Project hexagon vertices (H*6 points) back to lon/lat.
    g_hex = reg.project(hx_pts, [b_oct, g_gcd])
    # g_gcd coords: column 0 = lat, column 1 = lon.
    hex_latlons = g_hex.coords.reshape(hx, 6, 2)

    results = []
    for i in range(hx):
        verts = hex_latlons[i]           # (6, 2): [lat, lon]
        ring = ', '.join(f'{v[1]:.9f} {v[0]:.9f}' for v in verts)
        closing = f'{verts[0, 1]:.9f} {verts[0, 0]:.9f}'
        wkt = f'POLYGON(({ring}, {closing}))'
        results.append((str(full_uuids[i]), str(bin_uuids[i]), wkt))
    return results

