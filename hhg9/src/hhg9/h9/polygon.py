"""
Part of the H9 project
Author: Ben
base/h9/polygon.py
`polygon.py` generates luts for defining shapes within barycentric lattice.
"""
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from hhg9.h9.addressing import reg_hex_digits
from hhg9.h9.classifier import location
from hhg9.h9.protocols import H9ConstLike, H9PolygonLike


@dataclass(frozen=True, slots=True)
class H9Polygon:
    """Polygons under Lattice"""
    hh: NDArray[np.float64]  # half-hex (mode, c2) 4 pts (x,y)
    hx: NDArray[np.float64]  # hex (mode, c2) 6 pts (x,y)
    tx: NDArray[np.float64]  # cell (mode, c2, ord) 3 pts (x,y)
    se: NDArray[np.float64]  # supercell edges (mode) 9 pts (x,y)
    sv: NDArray[np.float64]  # supercell vertices (mode) 3 pts (x,y)
    gd: NDArray[np.float64]  # unshared points of a cell excluding (0,0).


def _h9_polygon(h9k: H9ConstLike = None) -> H9Polygon:
    """build H9Polygons = all are clockwise!"""
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
        (1, 0): [  # c2 hexagons mode 0;  final 2 pts are exterior to the region
            [(3, 1), (2, 0), (0, 0), (-1, 1), (0, 2), (2, 2)],
            [(0, -2), (-1, -1), (0, 0), (2, 0), (3, -1), (2, -2)],
            [(-3, 1), (-1, 1), (0, 0), (-1, -1), (-3, -1), (-4, 0)]
        ],
        (1, 1): [  # c2 hexagons mode 1;  final 2 pts are exterior to the region
            [(-1, -1), (0, 0), (2, 0), (3, -1), (2, -2), (0, -2)],
            [(-1, 1), (0, 0), (-1, -1), (-3, -1), (-4, 0), (-3, 1)],
            [(2, 0), (0, 0), (-1, 1), (0, 2), (2, 2), (3, 1)]
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


def region_grid(levels: int = 3, mode=0, h9p=H9P):
    """
    Return grid of points with root mode=mode at depth = levels.
    returned grid is a list of [address, origin, scale]
    """
    from hhg9.h9 import H9C, H9R
    h9c, h9r = H9C, H9R
    modes = [h9r.downs, h9r.ups]
    kids = modes[mode]
    queue = [(k, h9c.mode[k], h9c.off_xy[k], 1.0/3.0) for k in kids]

    for depth in range(levels):
        next_q = []
        for path, mode, origin, scale in queue:
            kids = modes[mode]   # shape (9,) indices
            for k in kids:
                mo_k = h9c.mode[k]
                off_k = h9c.off_xy[k]
                path_k = np.append(path, k)
                origin_k = origin + off_k * scale
                scale_k = scale / 3.0
                next_q.append((path_k, mo_k, origin_k, scale_k))
        queue = next_q
    return queue


def tri_grid(levels: int = 5, mode=0, h9p=H9P):
    """given a layer - generate the entire set of triangles at that layer (for an octant)"""
    queue = region_grid(levels, mode, h9p)
    pts = np.empty((len(queue), 3, 2), dtype=np.float64)
    for i, (path, mode, origin, scale) in enumerate(queue):
        pts[i] = origin + h9p.sv[mode] * scale
    return pts


def tri_mesh(levels: int = 5, mode: int = 0, h9p=H9P):
    """Return unique vertices and edges for the triangular mesh at a given level.
    Parameters
    ----------
    levels : int
        Subdivision depth (same semantics as ``tri_grid``).
    mode : int
        Root mode (0 or 1) for the octant.
    h9p : H9Polygon
        Polygon LUT (defaults to global ``H9P``).

    Returns
    -------
    verts : (V, 2) float64
        Unique vertex coordinates in global (x, y).
    edges : (E, 2) int64
        Undirected edges as pairs of vertex indices into ``verts``.
    tris : (T, 3) int64
        Triangle vertex indices into ``verts`` (for matplotlib Triangulation, etc.).
    """
    # Use existing tri_grid to get all triangle vertices at this level
    tris_xy = tri_grid(levels=levels, mode=mode, h9p=h9p)  # (T, 3, 2)
    num_tris = tris_xy.shape[0]

    # Flatten all triangle vertices and deduplicate
    flat = tris_xy.reshape(-1, 2)  # (T*3, 2)
    verts, inv = np.unique(flat, axis=0, return_inverse=True)

    # Map each triangle to indices into the unique vertex array
    tris = inv.reshape(num_tris, 3)

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
    Given a batch of barycentric points (`pts`), walk `depth` layers and build
    the half‑hex polygon for each address/layer. Return **unique** polygons and
    an index map back to those uniques for each (address, layer).

    Returns
    -------
    uniques : (U, 4, 2) float64
        Unique polygons in global (x,y), deduplicated exactly.
    refs : (N, depth) int64
        For each input address (N) and layer (depth), the index into `uniques`.
    """
    from hhg9.h9 import H9C
    from hhg9.h9.region import H9R, xy_regions_iter

    # Modes per point
    oc, mo = pts.cm()
    coords = pts.coords
    num_pts = len(pts)
    depth = levels+1

    # Offsets and shapes (float64)
    offs = H9C.off_xy.astype(np.float64, copy=False)   # (R,2)
    hh   = H9P.hh                                      # (2,3,4,2)

    # Accumulators
    parent_xy = np.zeros((num_pts, 2), dtype=np.float64)
    scale = 1.0

    # We'll collect per-layer polygons flattened as (N*D, 4, 2)
    polys = np.empty((num_pts * depth, 4, 2), dtype=np.float64)
    flat_idx = np.empty((num_pts, depth), dtype=np.int64)  # indices into `polys`

    for ev in xy_regions_iter(coords, mode=mo, depth=levels):
        if ev.phase != 'pre':
            continue
        i = ev.i  # 0..D-1
        # Here parent is the level for which we are generating the
        # hexagon, I believe.
        # c2 = rgn.H9R.mcc2[pmo, region]
        # hhp = hh_polys[smo, c2]
        # Child c2 for each row under the parent mode
        c2 = H9R.mcc2[ev.pmo, ev.cid]             # (N,)
        hh_shape = hh[ev.pmo, c2]                 # pmo

        # Translate/scale each triangle to global coords
        polygon = parent_xy[:, None, :] + hh_shape * scale  # (N,4,2)

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


def hex_layer(pts, layer: int = 10):
    """
    Hex aggregation using region_neighbours from region.py directly:
      1) Address points to layer+1 using xy_regions
      2) Identify mode-1 parents at layer L
      3) Use region_neighbours (from region.py) to find mode-0 equivalents
      4) Handle octant hops properly
      5) Build hex identity WITHOUT c2 (hex = 2 half-hexes that differ only by c2)

    Returns: polys (h,6,2), count (h,), inv (n,), oc_poly6 (h,6)
    Expected: 12*(9**layer) unique hexes
    """
    from hhg9.h9.region import xy_regions, region_neighbours as region_neighbours
    from hhg9.h9 import H9K, H9C, H9R, H9P, in_scope

    n = len(pts)
    if n == 0:
        return (np.empty((0, 6, 2), float),
                np.empty((0,), int),
                np.empty((0,), int),
                np.empty((0,), int))

    b_oct = pts.domain
    if b_oct.name != 'b_oct':
        raise ValueError('pts must be barycentric')
    oc, oct_mode = pts.cm()  # oc=octant id of each pt, oct_mode = octant modes.

    # Address to layer+1 to get parent(L) and child(L+1)
    rg = xy_regions(pts.coords, oct_mode, layer)

    # Identify which parents are mode-1 and need coalescing
    par = rg[:, layer]
    pmo = H9C.mode[par]

    move = (pmo == 1)
    if np.any(move):
        move_idx = np.flatnonzero(move)
        og_root = rg[move, 0]
        rg_neig, c2_neig = region_neighbours(rg[move])
        n_root = rg_neig[:, 0]
        root_changed = (n_root != og_root)
        keep_local = ~root_changed  # (M,)
        keep_global = move_idx[keep_local]  # (K,) global indices
        non_hop_nb = rg_neig[keep_local]
        rg[keep_global] = non_hop_nb

        if np.any(root_changed):
            sel = move_idx[root_changed]
            oc[sel] = b_oct.oid_nb[oc[sel], c2_neig[root_changed]]
            coords_flip = pts.coords[sel].copy()
            coords_flip[:, 1] *= -1.0
            mode_flip = b_oct.oid_mo[oc[sel]]
            ogx = xy_regions(coords_flip, mode_flip, layer)
            rg[sel] = ogx

    key_hex = reg_hex_digits(rg, oc, b_oct)
    hex_u, hex_first, inv_hex = np.unique(key_hex, axis=0, return_index=True, return_inverse=True)
    count = np.bincount(inv_hex, minlength=hex_u.shape[0])

    # Geometry from unique representatives
    hex_rgn = rg[hex_first]                         # list of unique hexagon addresses for each hex
    hex_oid = oc[hex_first]                         # octant ids of each of those.
    hex_omo = b_oct.oid_mo[hex_oid]                 # modes of each octant for each hexagon
    hex_num = hex_rgn.shape[0]                      # number of hexagons.

    # For geometry, compute c2 from the representative (coalesced) addresses
    hex_par = hex_rgn[:, layer]                     # get the parent at the hex level
    hex_chd = hex_rgn[:, layer + 1]                 # get the child
    hex_pmo = H9C.mode[hex_par]                     # local mode.
    hex_xc2 = H9R.mcc2[hex_pmo, hex_chd]            # c2-relation/orientation of the hexagon.
    hex_all = H9P.hx[hex_pmo, hex_xc2]              # The h,6,2 points.
    hex_xy = np.zeros((hex_num, 2), dtype=float)

    # Now calculate neighbour components also.  These are used for external/octant crossing hexagons.
    nbr_rgn, nbr_xc2 = region_neighbours(hex_rgn)   # nbr equivalent of hex_rgn, and the c2-relation eq. of hex_xc2.
    nbr_oid = b_oct.oid_nb[hex_oid, nbr_xc2]        # nbr equivalent of hex_oid
    # nbr_pmo = 1 - hex_pmo                         # neighbours are always opposite mode of local.
    # nbr_all = H9P.hx[nbr_pmo, nbr_xc2]            # The h,6,2 points.
    nbr_xy = np.zeros((hex_num, 2), dtype=float)

    # Now compose the origin of every hexagon, along with the scale.
    scale = 1.0
    for i in range(1, layer + 1):
        hex_xy += H9C.off_xy[hex_rgn[:, i]] * scale
        nbr_xy += H9C.off_xy[nbr_rgn[:, i]] * scale
        scale /= 3.0

    # The parent mode, and c2 give us what we need to generate the hexes.
    # This gives us the offset and scale
    hex_pts = hex_xy[:, None, :] + hex_all * scale  # (n,6,2)
    nbr_pts = hex_xy[:, None, :] + hex_all * scale  # (n,6,2)

    # Set the basic octant id for each hexagon
    oc_poly6 = np.repeat(hex_oid[:, None], 6, axis=1)  # (H, 6)

    # Now resolve externally impinged hexagons.
    # This is necessary because our Projection guarantees no accuracy outside the boundaries
    # of the equilateral. The are points needed in their local context, hence the neighbour work.
    hex_ẋ = H9K.R3 * hex_pts[..., 0].ravel()   # Classifier ẋ
    hex_y = hex_pts[..., 1].ravel()            # Classifier y
    oc_mo = np.repeat(hex_omo, 6)       # Set the octant mode for each hexagon
    locs = location(hex_ẋ, hex_y, oc_mo).reshape(-1, 6)  # identify locations in octant of each pt.
    # locs are:  0: undefined, 1: external, 2: internal, 3:edge, 4:vertex
    ext_mask = (locs == 1)
    if np.any(ext_mask):
        n_ext = int(ext_mask.sum())
        n_hex = int(ext_mask.any(axis=1).sum())
        print(f"[hex_layer] layer={layer}: {n_ext} external vertices across {n_hex} hexagons")
        # Additional diagnostic: hexagons with externals not at vertices 4/5
        ext_any = ext_mask.any(axis=1)
        ext_45 = ext_mask[:, 4] | ext_mask[:, 5]
        weird = ext_any & ~ext_45
        n_weird = int(weird.sum())
        if n_weird:
            print(f"[hex_layer] layer={layer}: {n_weird} hexagons with externals outside vertices 4/5")
        # Check for non-standard external patterns
        ext_hex = ext_mask.any(axis=1)
        expected = np.array([False, False, False, False, True, True], dtype=bool)
        patterns = ext_mask[ext_hex]
        if patterns.shape[0] > 0:
            ok = np.all(patterns == expected, axis=1)
            if not np.all(ok):
                bad = ~ok
                n_bad = int(bad.sum())
                bad_patterns = np.unique(patterns[bad], axis=0)
                print(f"[hex_layer] layer={layer}: {n_bad} hexagons with non-standard external pattern; examples: {bad_patterns}")
    else:
        ext_hex = np.any(ext_mask, axis=1)

    wnb_pts = nbr_pts[ext_hex]             # gather the related neighbour point.
    wlx_pts = hex_pts[ext_hex]             # gather the related local point.
    w_oct = oc_poly6[ext_hex]              # these octants are currently set to the central one.
    n_oct = nbr_oid[ext_hex]               # grab the related neighbour octant id.

    # Use location mask to update only those vertices that are external (locs == 1).
    # ext_mask is over all hexes; restrict it to the external subset.
    locs_ext = ext_mask[ext_hex]  # (Nh, 6) bool
    for idx in (4, 5):
        sel = locs_ext[:, idx]  # which external-hex rows have an external at this vertex
        if not np.any(sel):
            continue
        # Restrict patterns to those rows, then determine which of them are "standard" in the last two positions.
        pats_sel = patterns[sel]
        mnx_sel = np.any(pats_sel[:, -2:], axis=1)
        if not np.any(mnx_sel):
            continue
        # Build a full-length mask over the Nh external hexes.
        mask = np.zeros(locs_ext.shape[0], dtype=bool)
        mask[sel] = mnx_sel
        inv_idx = (6 - idx) % 6
        # Substitute neighbour vertex and flip its y coordinate for the selected rows.
        wlx_pts[mask, idx, :] = wnb_pts[mask, inv_idx, :]
        wlx_pts[mask, idx, 1] *= -1.0
        # Update octant id for those vertices.
        w_oct[mask, idx] = n_oct[mask]
    oc_poly6[ext_hex] = w_oct
    hex_pts[ext_hex] = wlx_pts

    return hex_pts, count.astype(int), inv_hex.astype(int), oc_poly6


def hh_layer(pts, layer: int = 10):
    """
    Given a batch of barycentric points (`pts`), and a level,
    calculate the half-hexagons for that level, based upon those points.
    Also return the number of points within each half-hexagon.
    """
    from hhg9.h9.region import xy_regions
    from hhg9.h9 import H9C, H9R
    num_pts = len(pts)
    if num_pts == 0:
        return

    ocf, mode_f = pts.cm()  # return the octant/mode
    rgx = xy_regions(pts.coords, mode_f, layer)
    poi = rgx[:, -2]
    smo = H9C.mode[poi]    # We will need it's mode (smo=self_mode)
    c2i = rgx[:, -1]
    c2 = H9R.mcc2[smo, c2i]  # Now grab the c2 of the current region - which neighbour?!
    key = np.concatenate([ocf[:, None], rgx[:, 1:layer + 1], c2[:, None]], axis=1)
    uniq, first_idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    pops = np.bincount(inv)
    rgs = rgx[first_idx]   # This gives us the indicative regions.
    oc = ocf[first_idx]    # and the indicative octants for each region.
    c2 = c2[first_idx]
    num_hh = rgs.shape[0]  # and the number of half-hexagons found.
    acc_xy = np.zeros((num_hh, 2), dtype=np.float64)
    scale = 1.0
    for i in range(1, layer+1):
        acc_xy += (H9C.off_xy[rgs[:, i]] * scale)
        scale /= 3.0
    poi = rgs[:, -2]
    smo = H9C.mode[poi]    # We will need its mode (smo=self_mode)
    polygon = scale * H9P.hh[smo, c2] + acc_xy[:, None]
    oc_poly4 = np.repeat(oc[:, None], 4, axis=1)  # 4 points of half-hex
    return polygon, pops, inv, oc_poly4

