# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Grid Methods - eg for composing rectilinear pixel grids for projected sampling.
"""
from functools import cache
import numpy as np

from hhg9.algorithms.geometry import inside_convex_polygon_cw
from hhg9.h9 import H9K, H9P
from hhg9.h9.classifier import in_scope
from hhg9 import Points


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

def qa_grid(quad, scale: float = 200, affine=None, tol: float = 0.5):
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
        xpm:      (H,)   mode index (0 or 1).
        xc2:      (H,)   c2 index (0, 1, or 2).
        scale:    scalar scale factor for this layer (from hex_parents).
        n_oct:    OctahedralNet domain instance.

    Returns:
        Points of shape (H*6, 2) in n_oct.
    """
    hex_verts_n = np.zeros((len(hex_par), 6, 2))
    for side, proj in n_oct.projs.items():
        oid = n_oct.sides[side].oid
        fm = (hex_oid == oid)
        if not np.any(fm):
            continue
        if proj.c2trans is None:
            par_n = hex_par[fm] @ proj.matrix + proj.offset
            hex_verts_n[fm] = par_n[:, None, :] + H9P.hx[xpm[fm], xc2[fm]] * scale @ proj.matrix
        else:
            for c2v in range(3):
                c2m = fm & (xc2 == c2v)
                if not np.any(c2m):
                    continue
                matr, off_c2 = proj.c2_affine(c2v)
                par_n = hex_par[c2m] @ matr + off_c2 + proj.offset
                hex_verts_n[c2m] = par_n[:, None, :] + H9P.hx[xpm[c2m], c2v] * scale @ matr
    return Points(hex_verts_n.reshape(-1, 2), n_oct)


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

        # Child c2 for each row under the parent mode
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
