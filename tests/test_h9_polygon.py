# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""
Invariant tests for the PRODUCTION polygon LUTs and mesh helpers
(``hhg9.h9.polygon``).

These bind to the public ``H9P`` singleton and the grid/mesh generators, and
assert geometric/topological invariants (LUT shapes, clockwise winding,
cross-LUT consistency, recursion counts, mesh Euler characteristic) rather than
literal vertex coordinates — so they survive a refactor of the underlying
construction.
"""
import numpy as np
import pytest

from hhg9.h9.polygon import (
    H9P, tri_grid, tri_mesh, region_grid, uv_grid, _unique_rows_tol,
)
from hhg9.h9.constants import H9K


def _signed_area(poly):
    """Shoelace signed area: > 0 counter-clockwise, < 0 clockwise."""
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


# ---------------------------------------------------------------------------
# 1. LUT shapes (the documented contract on H9Polygon)
# ---------------------------------------------------------------------------

def test_lut_shapes():
    assert H9P.hh.shape == (2, 3, 4, 2)   # half-hex: 4 verts
    assert H9P.hx.shape == (2, 3, 6, 2)   # full hex: 6 verts
    assert H9P.tx.shape == (2, 3, 3, 3, 2)  # 3 triangles × 3 verts
    assert H9P.se.shape == (2, 9, 2)
    assert H9P.sv.shape == (2, 3, 2)
    assert H9P.gd.shape == (6, 2)
    assert H9P.hi.shape == (2, 3, 7, 2)   # 6 hex UV verts + centroid


# ---------------------------------------------------------------------------
# 2. Winding — all polygons are clockwise (per module docstring)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("c2", [0, 1, 2])
def test_full_hex_is_clockwise(mode, c2):
    assert _signed_area(H9P.hx[mode, c2]) < 0


@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("c2", [0, 1, 2])
@pytest.mark.parametrize("tri", [0, 1, 2])
def test_region_triangle_is_clockwise_and_nondegenerate(mode, c2, tri):
    area = _signed_area(H9P.tx[mode, c2, tri])
    assert area < 0                      # clockwise
    assert abs(area) > 1e-12             # non-degenerate


@pytest.mark.parametrize("mode", [0, 1])
def test_supercell_vertex_triangle_is_clockwise(mode):
    assert _signed_area(H9P.sv[mode]) < 0


# ---------------------------------------------------------------------------
# 3. Cross-LUT consistency: hi (integer UV) reconstructs hx (metric)
# ---------------------------------------------------------------------------

def test_hi_integer_uv_reconstructs_hx():
    """The first 6 entries of ``hi`` are the hex vertices in integer (U, 3V)
    lattice units; scaling by (U, 3V) must reproduce the metric ``hx`` LUT."""
    uv = np.array([H9K.lattice.U, 3 * H9K.lattice.V])
    recon = H9P.hi[:, :, :6].astype(np.float64) * uv
    np.testing.assert_allclose(recon, H9P.hx, atol=1e-12)


# ---------------------------------------------------------------------------
# 4. Recursive grids — counts follow the 9-fold subdivision
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("levels", [0, 1, 2])
@pytest.mark.parametrize("mode", [0, 1])
def test_region_grid_count(levels, mode):
    assert len(region_grid(levels, mode)) == 9 ** (levels + 1)


@pytest.mark.parametrize("levels", [0, 1, 2])
@pytest.mark.parametrize("mode", [0, 1])
def test_tri_grid_shape(levels, mode):
    pts = tri_grid(levels, mode)
    assert pts.shape == (9 ** (levels + 1), 3, 2)


# ---------------------------------------------------------------------------
# 5. tri_mesh — valid, deduplicated triangulation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("levels,mode", [(1, 0), (2, 0), (2, 1)])
def test_tri_mesh_indices_in_range(levels, mode):
    verts, edges, tris = tri_mesh(levels, mode)
    n = len(verts)
    assert tris.shape == (9 ** (levels + 1), 3)
    assert tris.min() >= 0 and tris.max() < n
    assert edges.min() >= 0 and edges.max() < n


@pytest.mark.parametrize("levels,mode", [(1, 0), (2, 0), (2, 1)])
def test_tri_mesh_edges_are_undirected_unique(levels, mode):
    _, edges, _ = tri_mesh(levels, mode)
    # sorted endpoints, no self-loops, no duplicate undirected edges
    assert np.all(edges[:, 0] < edges[:, 1])
    assert len(np.unique(edges, axis=0)) == len(edges)


@pytest.mark.parametrize("levels,mode", [(1, 0), (2, 0), (2, 1)])
def test_tri_mesh_euler_characteristic(levels, mode):
    """A triangulated simply-connected patch satisfies V - E + T = 1
    (Euler characteristic of a disk).  This pins shared-vertex/edge correctness
    independent of the exact coordinates."""
    verts, edges, tris = tri_mesh(levels, mode)
    assert len(verts) - len(edges) + len(tris) == 1


# ---------------------------------------------------------------------------
# 6. _unique_rows_tol — tolerance dedup helper
# ---------------------------------------------------------------------------

def test_unique_rows_tol_merges_within_tolerance():
    pts = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0 + 1e-15, 1.0]])
    verts, inv = _unique_rows_tol(pts, tol=1e-9)
    assert verts.shape[0] == 2
    np.testing.assert_array_equal(inv, [0, 0, 1, 1])


def test_unique_rows_tol_rejects_nonpositive_tol():
    with pytest.raises(ValueError):
        _unique_rows_tol(np.zeros((3, 2)), tol=0.0)


# ---------------------------------------------------------------------------
# 7. uv_grid — flatten vs layered output
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("levels", [1, 2])
def test_uv_grid_flatten_shape(levels):
    g = uv_grid(levels, 0)            # flatten=True default
    assert g.ndim == 2 and g.shape[1] == 4   # [mode, u, v, scale]
    assert g.shape[0] == 9 ** (levels + 1)


@pytest.mark.parametrize("levels", [1, 2])
def test_uv_grid_layers_match_flatten(levels):
    flat = uv_grid(levels, 0, flatten=True)
    layers = uv_grid(levels, 0, flatten=False)
    assert len(layers) == levels + 1
    np.testing.assert_array_equal(layers[-1], flat)
