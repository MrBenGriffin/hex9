# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""
Tests for the dynamic / local net unfolding: OctahedralNet.local_layout + place.

A bounded region spanning >1 octant has no single b_oct plot.  local_layout lays
the touched octants flat by hinging them out from a fundamental octant, reusing
each octant's own proj.matrix and recomputing only the offset.  Because each
octant is a √2 equilateral preserved in n_oct, the hinge across a shared edge is
a pure translation, so a *compatible* set of octants closes to machine epsilon.

Checks:
  1. An edge-adjacent pair (Britain spans octants 0 & 2) closes exactly.
  2. The fundamental octant gets the identity offset and its own matrix.
  3. Per-octant matrices are isometric (|det| == 1).
  4. place() maps absent octants to NaN.
  5. The residual guard discriminates: a compatible pair ≈0; the whole
     octahedron (which butterfly's fixed matrices cannot unfold by translation
     alone) reports a large residual.
  6. End-to-end through a HexMesh: shared seam verts coincide after place(), and
     hex edge lengths are preserved (the unfolding is an isometry).
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.spatial import cKDTree

# Britain–Belgium NON-CONVEX polygon, [lat, lon]; spans octant 0 & 2.
B_DLL = np.array([
    [51.084464524954795, -3.4462861844929585],
    [52.68223344923537, -2.189458562294672],
    [52.60858425701964, -1.432594162686184],
    [51.648685168766754, -2.0537941887799422],
    [51.52446468552721, -0.8970768988122543],
    [51.82555423461282, -0.747132064927554],
    [51.684114370958646, 0.5309691381848917],
    [51.40435883726121, 0.47384729670500597],
    [51.29190823165293, 5.4859054058927266],
    [50.72448838777691, 5.4044368907702074],
])


@pytest.fixture(scope="module")
def reg():
    from hhg9 import Registrar
    return Registrar()


@pytest.fixture(scope="module")
def n_oct(reg):
    return reg.domain('n_oct:butterfly')


def test_adjacent_pair_closes_exactly(n_oct):
    ll = n_oct.local_layout([2, 0])
    assert ll.unreached == []
    assert set(ll.layout) == {0, 2}
    assert ll.residual < 1e-12          # machine-epsilon seam closure


def test_fundamental_is_identity_offset(n_oct):
    from hhg9.h9 import H9O
    ll = n_oct.local_layout([2, 0], fundamental=2)
    mtx, off = ll.layout[2]
    assert_allclose(off, np.zeros(2), atol=1e-15)
    proj = n_oct.projs[H9O.oid_str[2]]
    assert_allclose(mtx, np.asarray(proj.matrix), atol=1e-15)


def test_matrices_are_isometric(n_oct):
    ll = n_oct.local_layout([0, 2])
    for mtx, _ in ll.layout.values():
        assert abs(abs(np.linalg.det(mtx)) - 1.0) < 1e-12


def test_place_absent_octant_is_nan(n_oct):
    ll = n_oct.local_layout([0, 2])
    coords = np.array([[0.1, 0.1], [0.2, 0.2]])
    oids = np.array([0, 5])             # 5 not in the layout
    out = n_oct.place(coords, oids, ll.layout)
    assert not np.any(np.isnan(out[0]))
    assert np.all(np.isnan(out[1]))


def test_guard_discriminates(n_oct):
    """Compatible pair ≈0; whole octahedron trips the guard (approach-(a) limit)."""
    pair = n_oct.local_layout([0, 2]).residual
    whole = n_oct.local_layout(range(8), fundamental=0)
    assert pair < 1e-12
    assert whole.unreached == []        # all reachable via oid_nb...
    assert whole.residual > 1e-3        # ...but translation-only cannot close them


def test_compositor_local_matches_fixed_and_closes(reg, n_oct):
    """Compositor(local=True) keeps inclusion identical to the fixed path and
    closes the seam; placed hexes stay regular."""
    from hhg9 import Points
    from hhg9.rendering.composition import LayerSpec, Compositor
    b_oct = reg.domain('b_oct')
    g_gcd = reg.domain('g_gcd')

    polygon_n = reg.project(Points(B_DLL, g_gcd), [g_gcd, b_oct, n_oct])
    specs = [LayerSpec(level=5, kind='outline'), LayerSpec(level=6, kind='outline')]

    fixed = Compositor(reg, b_oct, n_oct, specs).run(polygon_n)
    comp = Compositor(reg, b_oct, n_oct, specs, local=True)
    local = comp.run(polygon_n)

    assert [cl.count for cl in fixed] == [cl.count for cl in local]   # same inclusion
    assert comp.local_residual < 1e-12                                # seam closed

    polys = local[-1].verts.coords.reshape(-1, 6, 2)                  # regular hexes
    edges = np.linalg.norm(polys - polys[:, [1, 2, 3, 4, 5, 0]], axis=2).ravel()
    assert_allclose(edges, np.median(edges), rtol=1e-9)


def test_end_to_end_hexmesh_seam_and_isometry(reg, n_oct):
    from hhg9.h9.grid import HexMesh
    b_oct = reg.domain('b_oct')
    g_gcd = reg.domain('g_gcd')

    # L6: fine enough that the clip carries duplicated seam verts (one per
    # adjacent octant frame) to test coincidence after placement.
    mesh = HexMesh.create_clipped([6], B_DLL, reg)
    oids = np.asarray(mesh.pts.oid)
    octs = sorted(set(oids.tolist()))
    assert octs == [0, 2]               # the polygon genuinely spans two octants

    ll = n_oct.local_layout(octs)
    assert ll.residual < 1e-12
    world = n_oct.place(mesh.pts.coords, oids, ll.layout)
    assert not np.any(np.isnan(world))

    # Shared seam verts (same sphere point under both octant frames) coincide.
    gll = reg.project(mesh.pts, [b_oct, g_gcd]).coords
    ia = np.where(oids == 0)[0]
    ib = np.where(oids == 2)[0]
    d, j = cKDTree(gll[ib]).query(gll[ia], k=1)
    shared = d < 1e-6
    assert shared.sum() >= 2
    gap = np.linalg.norm(world[ia[shared]] - world[ib[j[shared]]], axis=1)
    assert gap.max() < 1e-9            # tear-free across the seam

    # Placed hexes are regular -- every edge of every hex equal, including the
    # seam-straddling hexes (whose raw b_oct cross-octant edges are meaningless).
    # That equality across the seam is exactly what the local net buys.
    pw = world[mesh[6]]                                  # (N, 6, 2) placed
    edges = np.linalg.norm(pw - pw[:, [1, 2, 3, 4, 5, 0]], axis=2).ravel()
    assert edges.min() > 0
    assert_allclose(edges, np.median(edges), rtol=1e-9)
