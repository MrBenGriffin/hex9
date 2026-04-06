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
from hhg9.h9.protocols import BaryLoc
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
def _hex_areas():
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
    earth = 510_065_621_724_154.6
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
def _tri_areas():
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
    earth = 510_065_621_724_154.6
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
def hex_props(level):
    """return (ideal) hex_grid properties for layer"""
    h_areas = _hex_areas()
    return h_areas[level]


@cache
def tri_props(level):
    """return  (ideal)  tri_grid properties for layer"""
    t_areas = _tri_areas()
    return t_areas[level]


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

    def __init__(self, pts, faces_dict, fine, ia, ib, vert_dict):
        self.pts  = pts                          # unique b_oct Points
        self._fl  = faces_dict                   # {layer: (N_L, 6) int32}
        self.fine = fine                         # finest layer index
        self.ia   = ia                           # (M,) int64, finest-scale
        self.ib   = ib                           # (M,) int64, finest-scale
        self._vd  = vert_dict                    # (ia, ib, oid) → vertex idx
        self.uv_o = np.array(list(self._vd.keys()), dtype=np.int32)

    @property
    def faces(self) -> np.ndarray:
        """Face index array for the finest layer."""
        return self._fl[self.fine]

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
        For cross-face edges the ib-axis is reflected across the shared face
        boundary, so the delta is computed in face-a's local frame.
        """
        if lev >= self.fine:
            return [idx_a, idx_b]

        ua, va = int(self.ia[idx_a]), int(self.ib[idx_a])
        ub, vb = int(self.ia[idx_b]), int(self.ib[idx_b])
        oa, ob = int(self.pts.oid[idx_a]), int(self.pts.oid[idx_b])
        cross = (oa != ob)

        du = (ub - ua) // 3
        if cross:
            # Boundary correction in _raw_uv negates ib when reflecting across
            # the shared face edge, so vb is in face-b's frame.  Reflect it back
            # into face-a's frame before computing the step.
            dv = (-vb - va) // 3
        else:
            dv = (vb - va) // 3

        u1, v1 = ua + du, va + dv       # 1/3 point (face-a frame)
        u2, v2 = ua + 2 * du, va + 2 * dv  # 2/3 point (face-a frame)

        s = 3 ** self.fine   # face-scale in fine-layer UV units

        def _lookup(u, v):
            """Look up (u,v) preferring the endpoint faces.

            For interior edges only oa is tried.

            For cross-face edges the step direction (dv) determines which
            face owns the intermediate vertex:

            prefer_reflect: (dv<0 and 0<v≤s) or (dv>0 and v<0)

            Case 1 — dv<0, 0<v≤s:
              We started at or beyond the face-oa boundary (va≥s) and are
              stepping inward.  The intermediate lies in face-ob's domain
              despite having positive v in face-oa's frame; the actual fine
              vertex is stored as (u,−v,ob).

            Case 2 — dv>0, v<0:
              The start vertex has negative v (already in face-ob's reflected
              domain), and we are stepping toward face-ob's origin (vb=0).
              The actual vertex is in face-ob's frame at (u,−v,ob).

            In all other cases (v<0 with dv<0, v>s with dv>0, etc.) the
            boundary-reflected vertex is correctly stored under oa's oid and
            the standard (u,v,oa) lookup succeeds first.
            """
            if not cross:
                return self._vd.get((u, v, oa))
            prefer_reflect = (dv < 0 and 0 < v <= s) or (dv > 0 and v < 0)
            if prefer_reflect:
                keys = [(u, -v, ob), (u, -v, oa), (u, v, ob), (u, v, oa)]
            else:
                keys = [(u, v, oa), (u, v, ob), (u, -v, ob), (u, -v, oa)]
            for key in keys:
                idx = self._vd.get(key)
                if idx is not None:
                    return idx
            return None

        idx_m1 = _lookup(u1, v1)
        idx_m2 = _lookup(u2, v2)

        # Sparse mode-1 edges occasionally lack the 1/3 vertex; fall back to
        # the start/end so the output length stays 3^delta.
        if idx_m1 is None:
            idx_m1 = idx_a
        if idx_m2 is None:
            idx_m2 = idx_b

        path1 = self._get_verts(idx_a, idx_m1, lev + 1)
        path2 = self._get_verts(idx_m1, idx_m2, lev + 1)
        path3 = self._get_verts(idx_m2, idx_b, lev + 1)

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
                return np.array([[0, 0, 1]], dtype=np.int32)  # [ia, ib, scale=1]
            return np.empty((0, 3), dtype=np.int32)
        U1    = H9K.lattice.U
        V3    = 3.0 * H9K.lattice.V
        div_f = float(3 ** layer)
        rows  = []
        for _, sc_mode, sc_origin, sc_scale in region_grid(layer - 1, mode):
            if sc_mode != reduce:
                continue
            # Convert float b_oct origin to integer UV.  The products are
            # always exact integers (off_uv entries are small integers,
            # powers of 3 are exact in float64).
            ia = int(round(sc_origin[0] * div_f / U1))
            ib = int(round(sc_origin[1] * div_f / V3))
            rows.append([ia, ib, 1])
        if not rows:
            return np.empty((0, 3), dtype=np.int32)
        return np.array(rows, dtype=np.int32)

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
        uv7_f, oc7_f = cls._raw_uv(finest, b_oct)

        # debug - check values are the same
        # xy6_f, oc6_f = cls._raw(finest, b_oct)          # (N, 7, 2), (N, 7)
        # xy6_f_uv = uv7_f[:, :6, :] * [U1, V3] / div_f
        # _debug_bp = None  # Looks good. But points are missing.

        N_f          = len(uv7_f)
        pts_p_hx = uv7_f.shape[1]

        vert_dict: dict[tuple, int] = {}
        vert_xy:   list = []
        vert_oid:  list = []
        ia_list:   list = []
        ib_list:   list = []
        face_idx   = np.empty((N_f, 6), dtype=np.int32)

        poly_eps = 1.0 - 1e-14  # pull polygon verts slightly off triangle edges as needed.
        for h in range(N_f):
            # Centroid xy — used to shrink polygon verts 0-5 toward interior
            cx = int(uv7_f[h, 6, 0]) * U1 / div_f
            cy = int(uv7_f[h, 6, 1]) * V3 / div_f
            for vi in range(pts_p_hx):           # include centroid (vi=6) in pool
                ia = int(uv7_f[h, vi, 0])
                ib = int(uv7_f[h, vi, 1])
                o  = int(oc7_f[h, vi])
                key = (ia, ib, o)
                idx = vert_dict.get(key)
                if idx is None:
                    idx = len(vert_xy)
                    vert_dict[key] = idx
                    rx = ia * U1 / div_f
                    ry = ib * V3 / div_f
                    if vi < 6:  # shrink polygon verts; centroid stays exact
                        rx = cx + (rx - cx) * poly_eps
                        ry = cy + (ry - cy) * poly_eps
                    vert_xy.append((rx, ry))
                    vert_oid.append(o)
                    ia_list.append(ia)
                    ib_list.append(ib)
                if vi < 6:
                    face_idx[h, vi] = idx

        pts    = Points(np.array(vert_xy, dtype=np.float64), b_oct, oid=np.array(vert_oid, dtype=np.int32))
        ia_arr = np.array(ia_list, dtype=np.int64)
        ib_arr = np.array(ib_list, dtype=np.int64)

        faces_dict = {finest: face_idx}

        # --- coarser layers: scale UV keys by 3^(finest-L), look up in pool
        for L in layers[:-1]:
            factor       = int(3 ** (finest - L))
            uv7_c, oc7_c = cls._raw_uv(L, b_oct)
            face_c       = np.empty((len(uv7_c), 6), dtype=np.int32)
            for h in range(len(uv7_c)):
                for vi in range(6):
                    ia = int(uv7_c[h, vi, 0]) * factor
                    ib = int(uv7_c[h, vi, 1]) * factor
                    o  = int(oc7_c[h, vi])
                    face_c[h, vi] = vert_dict[(ia, ib, o)]
            faces_dict[L] = face_c

        return cls(pts, faces_dict, finest, ia_arr, ib_arr, vert_dict)

    @classmethod
    def _raw_uv(cls, layer, b_oct):
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

    # @classmethod
    # def _raw(cls, layer, b_oct):
    #     """Generate corrected hex verts at *layer*, no dedup.
    #
    #     Returns ``(verts6, oc6)`` — arrays of shape ``(N, 6, 2)`` and
    #     ``(N, 6)`` ready for dict insertion.  Mode-1 boundary hexes are
    #     already excluded.
    #     """
    #     all_v6  = []
    #     all_oc6 = []
    #
    #     for side, sdom in b_oct.sides.items():
    #         oid  = int(sdom.oid)
    #         mode = int(H9O.oid_mo[oid])
    #
    #         sc0 = [(o, s) for m, o, s in cls._supercells(layer, mode) if m == 0]
    #         if not sc0:
    #             continue
    #
    #         rows   = [(o + H9P.hx[0, c2] * s, c2) for o, s in sc0 for c2 in range(3)]
    #         verts6 = np.array([r[0] for r in rows])          # (N, 6, 2)
    #         c2_arr = np.array([r[1] for r in rows], dtype=int)
    #         N      = len(verts6)
    #
    #         flat   = verts6.reshape(-1, 2)
    #         mo_arr = np.full(N * 6, mode, dtype=np.uint8)
    #         locs   = location(H9K.R3 * flat[:, 0], flat[:, 1], mo_arr).reshape(N, 6)
    #         ext    = (locs[:, 4] == BaryLoc.EXT) & (locs[:, 5] == BaryLoc.EXT)
    #
    #         v6  = verts6.copy()
    #         oc6 = np.full((N, 6), oid, dtype=int)
    #
    #         if ext.any():
    #             if mode == 0:
    #                 v6[ext, 4] = verts6[ext, 2] * [1.0, -1.0]
    #                 v6[ext, 5] = verts6[ext, 1] * [1.0, -1.0]
    #                 for c2v in range(3):
    #                     mask = ext & (c2_arr == c2v)
    #                     if mask.any():
    #                         oc6[mask, 4] = int(H9O.oid_nb[oid, c2v])
    #                         oc6[mask, 5] = int(H9O.oid_nb[oid, c2v])
    #             else:
    #                 v6  = v6[~ext]
    #                 oc6 = oc6[~ext]
    #
    #         if len(v6):
    #             all_v6.append(v6)
    #             all_oc6.append(oc6)
    #
    #     if not all_v6:
    #         return np.empty((0, 6, 2)), np.empty((0, 6), dtype=int)
    #     return np.concatenate(all_v6), np.concatenate(all_oc6)

# @staticmethod
# def _supercells(layer: int, mode: int):
#     """Yield ``(sc_mode, sc_origin_xy, sc_scale)`` for every supercell.
#
#     Used by ``_raw`` (float b_oct path).  For the integer UV path see
#     ``_supercell_uvs``.
#     """
#     if layer == 0:
#         yield mode, np.zeros(2, dtype=float), 1.0
#     else:
#         for _, sc_mode, sc_origin, sc_scale in region_grid(layer - 1, mode):
#             yield sc_mode, sc_origin, sc_scale

# ---- raw hex generation per layer ----------------------------------

    # ---- factory -------------------------------------------------------

    def claude_densify(self, L: int) -> np.ndarray:
        """Return densified face indices for layer *L* using fine-layer verts.

        For each of the 6 edges of each L-hex, the intermediate fine-layer
        vertices are determined by integer lattice interpolation — pure
        arithmetic, no collinearity test.

        Returns
        -------
        ndarray, shape ``(N_L, 6 * (3**δ + 1))``
            δ = ``self.fine - L``.  Each row lists vertex indices walking
            around the densified hex perimeter: ``3**δ + 1`` verts per edge
            (start vertex + intermediates; final vertex is start of next edge).
        """
        delta  = self.fine - L
        if delta <= 0:
            return self._fl[L]
        factor = int(3 ** delta)
        f_L    = self._fl[L]                          # (N_L, 6)
        n_hex  = len(f_L)
        vpere  = factor                           # verts per edge (incl. start)
        out    = np.empty((n_hex, 6 * vpere), dtype=np.int32)

        for e in range(6):
            j       = (e + 1) % 6
            idx_a   = f_L[:, e]                       # (N_L,) start vertex
            idx_b   = f_L[:, j]                       # (N_L,) end vertex
            ia_a    = self.ia[idx_a]
            ib_a    = self.ib[idx_a]
            dia     = self.ia[idx_b] - ia_a           # (N_L,) Δia over edge
            dib     = self.ib[idx_b] - ib_a
            oid_a   = self.pts.oid[idx_a]
            oid_b   = self.pts.oid[idx_b]

            for t in range(vpere):
                ia_t  = ia_a + dia * t // factor
                ib_t  = ib_a + dib * t // factor
                # Primary oid: start-side for first half, end-side for second.
                # For interior edges (oid_a == oid_b) this always hits.
                # For boundary edges the halfway rule may not match what
                # _raw_uv assigned, so fall back to the other oid on a miss.
                oid_t = np.where(t * 2 < factor, oid_a, oid_b)
                alt_t = np.where(t * 2 < factor, oid_b, oid_a)
                col   = e * vpere + t
                def _lk(ia, ib, o1, o2):
                    v = self._vd.get((ia, ib, o1))
                    return v if v is not None else self._vd[(ia, ib, o2)]
                out[:, col] = np.array([
                    _lk(int(ia_t[h]), int(ib_t[h]), int(oid_t[h]), int(alt_t[h]))
                    for h in range(n_hex)
                ], dtype=np.int32)

        return out


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
    full_uuids, _ = h9_encode(sample_lats[_idx], sample_lons[_idx], reg=reg)

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

