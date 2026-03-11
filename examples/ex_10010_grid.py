# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
hh_grid: Pixel grid generation for half-hexagon (c2) regions.

Provides two entry points:

``hh_grid(mode, c2, scale)``
    Generate a tight pixel grid for one half-hexagon sub-region (c2=0,1,2)
    of an octant face.  Returns a subset of the global face lattice, so pixel
    indices are face-level and compositing is index-exact.

``hh_grid_face(mode, scale)``
    Generate all three c2 sub-grids for a full octant face in one call.
    Pixel sets are disjoint and together tile the face triangle exactly.

Both functions use the same ``scale`` convention as ``sq_grid_vx``:
``scale`` = number of pixels spanning the full triangle width in x.

Design
------
Global face lattice
    A single shared lattice underlies all three c2 sub-grids of one face::

        step_x = W / scale      (same as sq_grid_vx uses)
        step_y = H / face_hgt   (same as sq_grid_vx uses)

    Pixel (i, j) in the *face image* has its centre at::

        x = face_x0 + (i + 0.5) * step_x
        y = face_y0 + (j + 0.5) * step_y

    Because all three sub-grids are carved from this one lattice, every
    pixel has exactly one well-defined spatial location.  ``hh_grid``
    simply selects the face pixels whose centres fall inside the given
    half-hex quadrilateral.

Half-integer centering
    The ``+ 0.5`` ensures pixel centres never land on the triangular lattice
    itself.  Half-hex interior edges connect (0,0) to supercell vertices along
    irrational metric directions (slope ±√3), so no pixel centre ever falls
    exactly on a shared edge.  ``in_convex_poly`` therefore returns an
    unambiguous result for every pixel — zero overlaps, zero gaps.

Tight bounding box
    ``hh_grid`` only allocates and tests pixels within the axis-aligned
    bounding box of the half-hex (a 4×1 or 3×2 UV-unit rectangle), not the
    full face.  This is Option B from the design discussion: roughly 2/3 of
    face pixels are skipped without ever being tested.

Option A (``hh_grid_from_triangle``)
    Generate the full-face triangle via ``sq_grid_vx``, then carve by
    ``in_convex_poly`` per c2.  Identical pixel sets to ``hh_grid_face``
    (both use the same lattice), useful as a cross-check.  Less efficient
    because it first allocates the full face grid.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, NamedTuple
from hhg9.h9 import H9K
from hhg9.h9.grid import in_convex_poly

# ---------------------------------------------------------------------------
# Half-hex vertex tables  (UV integer units; origin = supercell barycentre)
# ---------------------------------------------------------------------------
# Shape (2, 3, 4, 2): [mode, c2, vertex_index, (x_UV, y_UV)]
# Clockwise order, consistent with H9P.hh and inside_triangle_cw.
# UV units: one x-step = U_metric = sqrt(2)/6
#           one y-step = 3V_metric = sqrt(6)/6  (= sqrt(3) * U_metric)

_HH_UV: NDArray[np.int8] = np.array([
    [   # mode 0 (V, down-pointing supercell)
        [( 3, 1), ( 2, 0), ( 0, 0), (-1, 1)],   # c2=0
        [( 0,-2), (-1,-1), ( 0, 0), ( 2, 0)],   # c2=1
        [(-3, 1), (-1, 1), ( 0, 0), (-1,-1)],   # c2=2
    ],
    [   # mode 1 (L, up-pointing supercell)
        [(-1,-1), ( 0, 0), ( 2, 0), ( 3,-1)],   # c2=0
        [(-1, 1), ( 0, 0), (-1,-1), (-3,-1)],   # c2=1
        [( 2, 0), ( 0, 0), (-1, 1), ( 0, 2)],   # c2=2
    ],
], dtype=np.int8)

# Bounding box (xmin, xmax, ymin, ymax) in UV integer units.
_HH_BBOX: NDArray[np.int8] = np.array([
    [(-1, 3,  0, 1),    # mode 0, c2=0  ->  4 x 1 UV
     (-1, 2, -2, 0),    # mode 0, c2=1  ->  3 x 2 UV
     (-3, 0, -1, 1)],   # mode 0, c2=2  ->  3 x 2 UV
    [(-1, 3, -1, 0),    # mode 1, c2=0  ->  4 x 1 UV
     (-3, 0, -1, 1),    # mode 1, c2=1  ->  3 x 2 UV
     (-1, 2,  0, 2)],   # mode 1, c2=2  ->  3 x 2 UV
], dtype=np.int8)

# Full-face bounding box (xmin, xmax, ymin, ymax) in UV integer units.
# mode 0 (V):  x in [-3, 3],  y in [-2, 1]   (3 UV units tall)
# mode 1 (L):  x in [-3, 3],  y in [-1, 2]   (3 UV units tall)
_FACE_BBOX: NDArray[np.int8] = np.array([
    (-3, 3, -2, 1),     # mode 0
    (-3, 3, -1, 2),     # mode 1
], dtype=np.int8)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _face_dims(scale: float) -> Tuple[int, int, float, float]:
    """
    Pixel dimensions and step sizes for the full face image.

    Returns
    -------
    face_wid, face_hgt : int
        Image dimensions (consistent with sq_grid_vx).
    step_x, step_y : float
        Pixel step in metric barycentric units (square-ish pixels).
    """
    V3 = H9K.lattice.V * 3      # = sqrt(6)/6

    face_wid = max(1, int(scale))
    face_hgt = max(1, round(scale * H9K.derived.RH))
    step_x   = H9K.radical.W / scale   # = (TR - TL) / face_wid
    step_y   = 3.0 * V3 / face_hgt     # = H / face_hgt
    return face_wid, face_hgt, step_x, step_y


# ---------------------------------------------------------------------------
# Named return type
# ---------------------------------------------------------------------------

class HHGridResult(NamedTuple):
    """
    Pixel grid for one half-hexagon.

    Attributes
    ----------
    wid, hgt : int
        Pixel dimensions of the half-hex bounding rectangle.
    coords : NDArray, shape (N, 2)
        Metric barycentric (x, y) coordinates of each valid pixel centre.
    mask : NDArray[bool], shape (wid * hgt,)
        Half-hex interior mask over the flat bounding-rectangle grid.
    px_idx : NDArray[int], shape (N,)
        **Face-level** column index for each valid pixel (0 = leftmost).
    py_idx : NDArray[int], shape (N,)
        **Face-level** row index for each valid pixel (0 = bottom row).
    px_off : int
        Column offset of the bounding rectangle's left edge in the face.
    py_off : int
        Row offset of the bounding rectangle's bottom edge in the face.
    """
    wid:    int
    hgt:    int
    coords: NDArray
    mask:   NDArray
    px_idx: NDArray
    py_idx: NDArray
    px_off: int
    py_off: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def hh_grid(
    mode:  int,
    c2:    int,
    scale: float,
) -> HHGridResult:
    """
    Pixel grid for one half-hexagon, carved from the global face lattice.

    Parameters
    ----------
    mode : int
        Supercell mode: 0 = V (down), 1 = L (up).
    c2 : int
        C2 sector: 0, 1, or 2.
    scale : float
        Pixels across the full triangle width.  Same convention as
        ``sq_grid_vx`` and ``sq_grid``.

    Returns
    -------
    HHGridResult
        Named tuple — see class docstring for field descriptions.
        ``px_idx``/``py_idx`` are **face-level** and index directly into a
        buffer of size ``face_image_dims(scale)``.
    """
    U  = H9K.lattice.U
    V3 = H9K.lattice.V * 3

    face_wid, face_hgt, step_x, step_y = _face_dims(scale)

    # Face lower-left corner in metric barycentric (mode-dependent y)
    fx0_UV = int(_FACE_BBOX[mode][0])   # always -3
    fy0_UV = int(_FACE_BBOX[mode][2])   # -2 (mode 0) or -1 (mode 1)
    face_x0 = fx0_UV * U
    face_y0 = fy0_UV * V3

    # Half-hex bounding box in UV integer units
    hx0, hx1, hy0, hy1 = (int(v) for v in _HH_BBOX[mode, c2])

    # Integer pixel offsets of the half-hex bbox lower-left in the face image.
    # Derived from UV integer offsets and the integer face pixel dimensions.
    # Both face_wid/6 and face_hgt/3 may be non-integer, but the round() here
    # is applied once and consistently for all sub-grids of the same face.
    px_off = round((hx0 - fx0_UV) * face_wid / 6)
    py_off = round((hy0 - fy0_UV) * face_hgt / 3)

    # Pixel dimensions of the bounding rectangle
    wid = max(1, round((hx1 - hx0) * face_wid / 6))
    hgt = max(1, round((hy1 - hy0) * face_hgt / 3))

    # Global face pixel indices for each column and row in this bounding box.
    # j=0 is the BOTTOM row of the face image (y = face_y0).
    i_face = np.arange(px_off, px_off + wid, dtype=np.int32)   # global col (x)
    j_face = np.arange(py_off, py_off + hgt, dtype=np.int32)   # global row (y, 0=bottom)

    # Metric barycentric coordinates of pixel centres (half-integer offset).
    xi = face_x0 + (i_face.astype(np.float64) + 0.5) * step_x
    yi = face_y0 + (j_face.astype(np.float64) + 0.5) * step_y

    xx, yy = np.meshgrid(xi, yi, indexing='xy')          # (hgt, wid)
    rec    = np.stack([xx.ravel(), yy.ravel()], axis=1)  # (wid*hgt, 2)

    # Face-level pixel indices for every point in the bounding rectangle.
    # px_face = i_face[ii_local]  (global column)
    # py_face = (face_hgt - 1) - j_face[jj_local]  (image-convention flip applied once,
    #           at the global level so that sub-grids sharing rows agree).
    jj_loc, ii_loc = np.meshgrid(np.arange(hgt), np.arange(wid), indexing='ij')
    px_face = i_face[ii_loc.ravel()].astype(np.int32)
    py_face = (face_hgt - 1 - j_face[jj_loc.ravel()]).astype(np.int32)

    # Interior mask: half-hex vertices in metric barycentric
    verts_m = _HH_UV[mode, c2].astype(np.float64) * np.array([U, V3])
    mask    = in_convex_poly(rec, verts_m)

    return HHGridResult(
        wid    = wid,
        hgt    = hgt,
        coords = rec[mask],
        mask   = mask,
        px_idx = px_face[mask],
        py_idx = py_face[mask],
        px_off = px_off,
        py_off = py_off,
    )


def hh_grid_face(
    mode:  int,
    scale: float,
) -> Tuple[HHGridResult, HHGridResult, HHGridResult]:
    """
    All three c2 half-hex grids for one octant face.

    Returns a 3-tuple of ``HHGridResult`` (c2=0, c2=1, c2=2).  All pixel
    indices are face-level and the three pixel sets are exactly disjoint.
    """
    return (
        hh_grid(mode, 0, scale),
        hh_grid(mode, 1, scale),
        hh_grid(mode, 2, scale),
    )


def hh_grid_from_triangle(
    mode:  int,
    scale: float,
) -> Tuple[HHGridResult, HHGridResult, HHGridResult]:
    """
    Option A: carve c2 sub-grids from the full-triangle lattice.

    Calls ``sq_grid_vx`` once (identical lattice to ``hh_grid``), then masks
    each c2 quadrilateral via ``in_convex_poly``.  Pixel indices are
    face-level and match those from ``hh_grid_face`` exactly.

    This is less efficient than ``hh_grid_face`` (allocates the full face
    grid) but is useful as a correctness cross-check.
    """
    from hhg9.h9.grid import sq_grid_vx
    U  = H9K.lattice.U
    V3 = H9K.lattice.V * 3

    face_wid, face_hgt, _, _ = _face_dims(scale)
    _, _, rec, face_mask, px_all, py_all = sq_grid_vx(scale, mode)

    valid_rec = rec[face_mask]
    valid_px  = px_all[face_mask]
    valid_py  = py_all[face_mask]

    fx0_UV = int(_FACE_BBOX[mode][0])
    fy0_UV = int(_FACE_BBOX[mode][2])

    results = []
    for c2 in range(3):
        verts_m = _HH_UV[mode, c2].astype(np.float64) * np.array([U, V3])
        c2_mask = in_convex_poly(valid_rec, verts_m)

        hx0, hx1, hy0, hy1 = (int(v) for v in _HH_BBOX[mode, c2])
        sub_wid = max(1, round((hx1 - hx0) * face_wid / 6))
        sub_hgt = max(1, round((hy1 - hy0) * face_hgt / 3))
        px_off  = round((hx0 - fx0_UV) * face_wid / 6)
        py_off  = round((hy0 - fy0_UV) * face_hgt / 3)

        results.append(HHGridResult(
            wid    = sub_wid,
            hgt    = sub_hgt,
            coords = valid_rec[c2_mask],
            mask   = c2_mask,
            px_idx = valid_px[c2_mask],
            py_idx = valid_py[c2_mask],
            px_off = px_off,
            py_off = py_off,
        ))
    return tuple(results)


def face_image_dims(scale: float) -> Tuple[int, int]:
    """
    ``(wid, hgt)`` of the face image, consistent with ``sq_grid_vx``.

    Face-level pixel indices from ``hh_grid`` / ``hh_grid_face`` index
    directly into a buffer of this size.
    """
    w, h, _, _ = _face_dims(scale)
    return w, h


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    SCALE = 60.0
    print(f'hh_grid self-test  scale={SCALE}')
    print(f'Face image: {face_image_dims(SCALE)}')
    print()

    # Option B
    print('Option B  hh_grid_face:')
    for mode in [0, 1]:
        grids = hh_grid_face(mode, SCALE)
        for c2, g in enumerate(grids):
            n     = g.coords.shape[0]
            fill  = n / (g.wid * g.hgt) * 100
            print(f'  m{mode}c{c2}: {g.wid:2d}x{g.hgt:2d}  valid={n:5d} ({fill:4.1f}%)'
                  f'  offset=({g.px_off:+3d},{g.py_off:+3d})'
                  f'  face_x=[{g.px_idx.min():3d},{g.px_idx.max():3d}]'
                  f'  face_y=[{g.py_idx.min():3d},{g.py_idx.max():3d}]')
        print()

    # Zero-overlap check
    print('Zero-overlap / coverage check:')
    for mode in [0, 1]:
        grids  = hh_grid_face(mode, SCALE)
        seen   = {}
        for c2, g in enumerate(grids):
            for px, py in zip(g.px_idx.tolist(), g.py_idx.tolist()):
                k = (px, py)
                if k in seen:
                    print(f'  OVERLAP m{mode}c{c2} vs c{seen[k]} at {k}')
                seen[k] = c2
        total = sum(g.coords.shape[0] for g in grids)
        status = 'OK' if len(seen) == total else f'MISMATCH {len(seen)} unique != {total} total'
        print(f'  mode {mode}: {len(seen)} unique face pixels from {total} samples — {status}')
    print()

    # Cross-check Option A vs Option B
    print('Cross-check Option A == Option B:')
    for mode in [0, 1]:
        b = hh_grid_face(mode, SCALE)
        a = hh_grid_from_triangle(mode, SCALE)
        for c2 in range(3):
            b_set = set(zip(b[c2].px_idx.tolist(), b[c2].py_idx.tolist()))
            a_set = set(zip(a[c2].px_idx.tolist(), a[c2].py_idx.tolist()))
            diff  = b_set.symmetric_difference(a_set)
            print(f'  m{mode}c{c2}: {"MATCH" if not diff else f"DIFF {len(diff)} pixels"}')
    print()
    print('Done.')
