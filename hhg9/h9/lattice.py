"""
Part of the H9 project
Author: Ben
base/h9/lattice.py
`lattice.py` constructs the UV rectilinear lattice **once** in float64 and freezes.
lattice.py defines the [h,p,d,m] <-> [u,v] luts.
This module depends upon the H9K constants and the H9CL classifier to generate
a set of cell barycentres that can be used to fix each of the 42 geometric cells
of the barycentric domain with an integer (u,v) co-ordinate.
These may then be collapsed into 12 legal layer **regions**, the 9 cells that belong to
the up-supercell, the 9 that belong to the down-supercell, and the c2 memberships from
which we determine the half-hexagons that are central to H9, and from which the hexagons emerge.
"""
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from .protocols import H9ConstLike, H9ClassifierLike


# Define the Cell/Supercell constants, modes, and membership LUTs
# Decode has several non-geometric cells (04,05,06, etc.) which we need to mask.
# (1) identify the 42 valid geometric ids.
# (2) to determine the mode of each geometric cell
# Because the H-layer is downward, rather than traditional
# upward, we invert the parity of each address accordingly.
# (3) identify the Lattice-related U,V indices of each
# of the 42 geometrically valid cell that determine their triangle-centroids.
# and generate luts for cell<->uv
# (3) identify supercell membership for each supercell mode -
# (this may be done via the _in_up / _in_dn methods) for each geometric cell.
# (4) identify the in_scope membership - the logical OR of (3) across both modes.
# (5) Maybe to identify the 3 groups of 3 C2 memberships for each supercell mode.
#    \     \     \    /     /     /
#  00 \  01 \  02 \03/ 07  / 0b  / 0f
# _____\_____\ ____\/____ /____ /______
#       \     \    /\    /     /
#     10 \  11 \12/16\17/ 1b  / 1f
# ________\_____\/____\/____ /_________
#          \    /\    /\    /
#        20 \21/25\26/2a\2b/ 2f
# ___________\/____\/____\/____________
#            /\    /\    /\
#        30 /34\35/39\3a/3e\ 3f
# _________/____\/____\/____\__________
#         /     /\    /\     \
#     40 /  44 /48\49/4d\ 4e  \ 4f
# ______/____ /____\/____\ ____\________
#      /     /     /\     \     \
#  50 / 54  / 58  /5c\  5d \  5e \ 5f

# First, define (N,P,H,M) as an affine cube-like coordinate system with one linear constraint over the planar triangular lattice
# where N is the -ve slope, P the +ve slope, H the horizontal slope thresholds, and M the triangle mode.
# We can fix (N0,P0,H0,M0)=(1,2,3,1); such that n-p+h+m:=3 for every (n,p,h,m) on the lattice.
# This N,P,H,M bears strong relation to the compose_luts encode/decode structure and lends itself heavily to the
# relations derived from the bands of the inequalities as depicted above.
# A U,V lattice is a rectangular lattice over the same plane, such that the U interval is 1/2 (triangle width),
# and V is 1/3 the triangle height. This will place every barycentre onto an integer u,v.
# The purpose here is to identify and place those cells in encode/decode that are valid geometric cells.
# cell_size = len(decode)
# u_scp = n_count + p_count


# @dataclass(frozen=True, slots=True)
# class H9Lattice:
#     """Cell-Lattice Properties"""
#     # ids: np.ndarray  # cell ids
#     count: int         # count of cells in classifier.
#     mode: np.ndarray  # cell mode (0=down, 1=up)
#     uv: np.array       # uv offset
#     offsets: np.ndarray  # cell offsets from origin
#     in_scopes: np.ndarray  # array of in-scope cell ids
#     is_dn: np.ndarray  # bools indicating if cell is in the down supercell
#     is_up: np.ndarray  # bools indicating if cell is in the up supercell
#     downs: np.ndarray  # array of cells in down supercell
#     ups: np.ndarray  # array of cells in up supercell


@dataclass(frozen=True, slots=True)
class H9Cell:
    """Cell Properties (derived via H9Lattice)"""
    count: int  # count of cells in classifier.
    mode: NDArray[np.uint8]  # cell mode (0=down, 1=up)
    off_uv: NDArray[np.int8]  # uv co-ordinate. uv ∈ [-9..9]
    off_xy: NDArray[np.float64]  # metric barycentric co-ordinate (x, y)
    off_ẋy: NDArray[np.float64]  # metric √3x co-ordinate (ẋ, y)
    in_scope: NDArray[np.uint8]  # array of in-scope cell ids
    in_mode: NDArray[bool]
    in_dn: NDArray[bool]  # (count) bools indicating if cell is in the down supercell
    in_up: NDArray[bool]  # (count) bools indicating if cell is in the up supercell
    downs: NDArray[np.uint8]  # array of cells in down supercell
    ups: NDArray[np.uint8]  # array of cells in up supercell
    c2: NDArray[np.uint8]  # [2, 3, 3] cells for each mode/c2 cluster


def _c2_groups(cell_ids, offsets, supercell_mode):
    """
    Split the 9 cells of one supercell into 3 C2 wedges (size 3) by angular sectors.
    Returns (N,2): [c2 in {0,1,2}, cell_id]
    """
    c2x = [
        [[0x26, 0x2a, 0x2b], [0x3a, 0x39, 0x49], [0x35, 0x25, 0x21]],  # mode 0
        [[0x39, 0x3a, 0x3e], [0x25, 0x35, 0x34], [0x2a, 0x26, 0x16]]   # mode 1
    ]
    return np.array(c2x[supercell_mode]).astype(np.uint8)
    # nudge pushes non-collinear cells to align with favoured group.
    # base aligns labels so C2=0 matches canon: Λ→base=0, V→base=2
    # nudge, base = (1, 0) if supercell_mode else (-1, 2)
    # pts = offsets[cell_ids].astype(np.float64)  # (N,2)
    # theta6 = np.arctan2(pts[:, 1], pts[:, 0]) / (np.pi / 3)  # 60° units, (half-integer)
    # sector = np.rint(6.5 + theta6).astype(np.int32) % 6  # 30° shift; 6.5 + half-integer -> 0...5
    # counts = np.bincount(sector, minlength=6)  # find singletons
    # singleton_mask = counts[sector] == 1  # gather singleton mask
    # sector[singleton_mask] += nudge  # merge singletons (will % 6 next)
    # c2 = (((base - sector) % 6) >> 1).astype(np.int8)  # align to C2 canon, %6 and 0..2
    # vals = np.column_stack((c2, cell_ids.astype(np.int16)))
    # order = np.lexsort((vals[:, 1], vals[:, 0]))
    # return vals[order][:, 1].reshape(3, 3)


def h9_cell_lattice(h9k: H9ConstLike = None, h9cl: H9ClassifierLike = None) -> H9Cell:
    """
    H9GConst factory method.
    :return: H9GConst
    """
    import hhg9.h9.classifier as clf
    if h9k is None:
        from hhg9.h9.constants import H9K
        h9k = H9K
    if h9cl is None:
        h9cl = clf.H9CL

    c_dec = h9cl.decode
    s_count = h9cl.p_levels.shape[0]  # p and n share the same shape.
    h_count = h9cl.h_levels.shape[0]
    cell_size = len(c_dec)

    # define the lattice to cover enough area as defined by the classifier.
    u_scp = 2 * s_count  # u = cell.width/2 - will need 1 outlier.
    v_scp = 3 * (h_count - 2)  # v = cell.height/3 - won't need more than 1 outlier.

    # define a rectilinear lattice.
    u_vals = np.arange(-u_scp, u_scp, dtype=np.int16)
    v_vals = np.arange(-v_scp, v_scp, dtype=np.int16)

    # However, we are solely interested in cell barycentres here.
    # So we will eliminate those points that do not rest on a barycentre.
    # even u: v ≡ 2 or 4 (mod 6)  → sequences r, r+6, ... within -v_scp, v_scp
    # odd  u: v ≡ 1 or 5 (mod 6)  → sequences r, r+6, ... within -v_scp, v_scp
    parity_match = ((v_vals[None, :] & 1) == (u_vals[:, None] & 1))
    not_mult3 = (v_vals[None, :] % 3) != 0
    bary_mask = parity_match & not_mult3

    # Pull pairs
    iu, iv = np.nonzero(bary_mask)
    uv = np.column_stack((u_vals[iu], v_vals[iv])).astype(np.int16, copy=False)

    uvf = uv.astype(np.float64, copy=False)
    uvx = uvf.copy()
    # metric scaling (needed for classification and distance):
    uvf *= (h9k.U, h9k.V)
    uvx *= (h9k.Ü, h9k.V)

    rx = clf.classify_cell(uvx[:, 0], uvx[:, 1], h9cl)
    # Now we need to select a reference barycentre for those regions which have more than 1.
    # Choose a per‑cell reference barycentre: closest to origin
    # Compute squared Euclidean distance in metric space (uvf) to avoid sqrt.
    r2 = uvf[:, 0] * uvf[:, 0] + uvf[:, 1] * uvf[:, 1]
    order = np.lexsort((r2, rx))  # Sort primarily by cell id (rx), secondarily by distance r2 (ascending).
    first_idx_in_sorted = np.unique(rx[order], return_index=True)[1]
    ref_rows = order[first_idx_in_sorted]
    ref_cells = rx[ref_rows]

    # These 42 ref_cells are all correctly placed on the grid.
    # It follows, via, classification that the hex9 cell plane is carved into 42 distinct regions.
    # This is the *actual* meaning of life, the universe, and everything. Don't tell the mice.
    ref_uv = uv[ref_rows]

    ref_xy = uvf[ref_rows]  # still in order?
    ref_ẋy = uvx[ref_rows]
    in_up_bools = clf.in_up(ref_ẋy[:, 0], ref_xy[:, 1], h9cl)
    in_dn_bools = clf.in_down(ref_ẋy[:, 0], ref_xy[:, 1], h9cl)

    # ups, downs
    ups = ref_cells[in_up_bools]
    dns = ref_cells[in_dn_bools]
    assert len(ups) == len(dns) == 9, f'Count of [Ups({len(ups)}) | Downs({len(dns)})] != 9'

    # in_scope
    in_scope = ref_cells[in_up_bools | in_dn_bools]
    assert len(in_scope) == 12, f'{len(in_scope)} != 12'

    # Build a dense per‑cell offsets table (NaN for non‑geometric / unused ids).
    # Store offsets in integer lattice units (uv), which are stable and compact.
    offs_xy = np.full((cell_size, 2), np.nan, dtype=np.float64)
    offs_xy[ref_cells] = ref_xy
    bary_xy = np.full((cell_size, 2), np.nan, dtype=np.float64)
    bary_xy[ref_cells] = ref_ẋy

    # in_ups, in_dns
    in_ups = np.full((cell_size,), False, dtype=bool)
    in_dns = np.full((cell_size,), False, dtype=bool)
    in_ups[ref_cells] = in_up_bools
    in_dns[ref_cells] = in_dn_bools
    in_mode = np.vstack([in_dns, in_ups])

    # uv indexes
    uv_idx = np.full((cell_size, 2), (99, 99), dtype=np.int8)
    uv_idx[ref_cells] = uv[ref_rows]

    # modes
    mode = np.zeros((cell_size,), dtype=np.uint8)
    i_dec = c_dec[ref_cells]
    mode[ref_cells] = (1 ^ (i_dec.sum(axis=1) & 1)).astype(np.uint8)  # 0=V, 1=Λ

    # c2_groups
    d_c2 = _c2_groups(dns, offs_xy, 0)
    u_c2 = _c2_groups(ups, offs_xy, 1)
    c2 = np.stack([d_c2, u_c2])  # shape 2,3,3

    return H9Cell(cell_size, mode, uv_idx, offs_xy, bary_xy, in_scope, in_mode, in_dns, in_ups, dns, ups, c2)


H9C = h9_cell_lattice()


# sanity checks.
# uv indices should be integers in range
assert np.allclose(H9C.off_uv, H9C.off_uv.astype(int))

# xy and ẋy should correspond where they are non-nan.
mask = ~np.isnan(H9C.off_xy[:, 0])
assert np.allclose(H9C.off_ẋy[mask, 0], np.sqrt(3) * H9C.off_xy[mask, 0])
assert np.allclose(H9C.off_ẋy[mask, 1], H9C.off_xy[mask, 1])
