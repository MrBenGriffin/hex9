# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""
Tests for the dynamic / local net unfolding: OctahedralNet.local_layout + place.

A bounded region spanning >1 octant has no single b_oct plot.  local_layout lays
the touched octants flat by keeping the fundamental in its native b_oct frame
(identity) and hinging each other octant onto its parent by a det +1 rotation
that closes the shared edge exactly.  It is net-independent (it does NOT reuse
the host net's per-face matrices), so a connected octant tree always closes to
machine epsilon; the real obstruction (a cone-vertex cycle) shows up only in
cut_residual.

Checks:
  1. An edge-adjacent pair (Britain spans octants 0 & 2) closes exactly.
  2. The fundamental octant gets the identity offset and identity (native) matrix.
  3. Per-octant matrices are isometric (|det| == 1).
  4. place() maps absent octants to NaN.
  5. The whole octahedron's tree closes (residual ≈0) but its cone-vertex cycles
     cannot, so cut_residual trips.
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
    # Pinned to the CLASSIC chain: the expectations below (octant spans of
    # clipped meshes, seam-vertex flag sets) were derived under the
    # WGS84-trained warp; the via-sphere default shifts cell membership at
    # clip boundaries by the warp-field difference. The local-net logic is
    # chain-agnostic — revisit expectations when local-net work resumes.
    r = Registrar()
    r.set_ellipsoid(a=6378137.0, inv_f=298.257223563, name='WGS84',
                    via_sphere=False)
    return r


@pytest.fixture(scope="module")
def n_oct(reg):
    return reg.domain('n_oct:butterfly')


def test_adjacent_pair_closes_exactly(n_oct):
    ll = n_oct.local_layout([2, 0])
    assert ll.unreached == []
    assert set(ll.layout) == {0, 2}
    assert ll.residual < 1e-12          # machine-epsilon seam closure


def test_fundamental_is_native_identity(n_oct):
    # Net-independent: the fundamental keeps its native b_oct frame (identity),
    # not the host net's per-face matrix.
    ll = n_oct.local_layout([2, 0], fundamental=2)
    mtx, off = ll.layout[2]
    assert_allclose(off, np.zeros(2), atol=1e-15)
    assert_allclose(mtx, np.eye(2), atol=1e-15)


def test_matrices_are_isometric(n_oct):
    ll = n_oct.local_layout([0, 2])
    for mtx, _ in ll.layout.values():
        assert abs(abs(np.linalg.det(mtx)) - 1.0) < 1e-12


def test_cut_residual_flags_vertex_complete_sets(n_oct):
    """cut_residual ≈0 when the octant set tiles flat; large when it is
    vertex-complete (wraps a cone vertex), even if the BFS tree residual is ~0."""
    pair = n_oct.local_layout([0, 2])                # two octants, no cycle
    assert pair.residual < 1e-12 and pair.cut_residual < 1e-9

    pole = n_oct.local_layout([0, 1, 2, 3])          # all four N faces -> N pole
    assert pole.residual < 1e-12                     # tree still closes cleanly
    assert pole.cut_residual > 0.1                   # but the cycle seam cannot
    ll = n_oct.local_layout([0, 2])
    coords = np.array([[0.1, 0.1], [0.2, 0.2]])
    oids = np.array([0, 5])             # 5 not in the layout
    out = n_oct.place(coords, oids, ll.layout)
    assert not np.any(np.isnan(out[0]))
    assert np.all(np.isnan(out[1]))


def test_guard_discriminates(n_oct):
    """Compatible pair closes; the whole octahedron's tree also closes now (the
    net-independent rotation hinge), but its cone-vertex cycles trip cut_residual."""
    pair = n_oct.local_layout([0, 2]).residual
    whole = n_oct.local_layout(range(8), fundamental=0)
    assert pair < 1e-12
    assert whole.unreached == []        # all reachable via oid_nb
    assert whole.residual < 1e-9        # tree closes (rotation hinge), net-independent
    assert whole.cut_residual > 0.1     # but wrapping cone vertices needs cuts


def test_local_layout_is_net_independent(reg):
    # The unfold no longer inherits the host net's orientation: calling on
    # different net flavours yields the same layout, and a pair that NO net
    # places adjacent (octants 4 & 6, identical matrices in every net) now
    # closes by the rotation hinge.
    bfly = reg.domain('n_oct:butterfly')
    mort = reg.domain('n_oct:mortar')
    for octs in ([0, 1], [4, 6]):
        a = bfly.local_layout(octs, fundamental=octs[0])
        b = mort.local_layout(octs, fundamental=octs[0])
        assert a.residual < 1e-9                       # closes (incl. the 4&6 holdout)
        for o in octs:
            assert_allclose(a.layout[o][0], b.layout[o][0], atol=1e-12)
            assert_allclose(a.layout[o][1], b.layout[o][1], atol=1e-12)


def test_compositor_local_accepts_g_gcd_and_closes(reg, n_oct):
    """Compositor(local=True) takes a g_gcd polygon (no n_oct flavour baking),
    clips it, closes the seam, and yields regular hexes."""
    from hhg9 import Points
    from hhg9.rendering.composition import LayerSpec, Compositor
    b_oct = reg.domain('b_oct')
    g_gcd = reg.domain('g_gcd')

    specs = [LayerSpec(level=5, kind='outline'), LayerSpec(level=6, kind='outline')]
    comp = Compositor(reg, b_oct, n_oct, specs, local=True)
    local = comp.run(Points(B_DLL, g_gcd))           # frame-neutral input

    assert local and all(cl.count > 0 for cl in local)
    assert comp.local_residual < 1e-12               # seam closed
    assert comp.local_cut < 1e-9                      # two octants, no cone vertex

    polys = local[-1].verts.coords.reshape(-1, 6, 2)  # regular hexes
    edges = np.linalg.norm(polys - polys[:, [1, 2, 3, 4, 5, 0]], axis=2).ravel()
    assert_allclose(edges, np.median(edges), rtol=1e-9)


def test_compositor_local_cband_clean_and_flags_vertex(reg, n_oct):
    """The 4-octant C-band via the frame-neutral create_clipped path renders
    clean — centroid clipping excludes the open-Pacific straddler that the old
    convex-hull scan produced — while cut_residual still flags the wrapped pole.
    """
    from hhg9 import Points
    from hhg9.rendering.composition import LayerSpec, Compositor
    b_oct = reg.domain('b_oct')
    g_gcd = reg.domain('g_gcd')

    # C-band over the four northern octants, open over the Pacific.
    c_band = np.array([[68, -170], [68, -90], [68, 0], [68, 90], [68, 115],
                       [42, 115], [42, 90], [42, 0], [42, -90], [42, -170]])

    specs = [LayerSpec(level=1, kind='outline'),
             LayerSpec(level=2, kind='outline'),
             LayerSpec(level=3, kind='outline')]
    comp = Compositor(reg, b_oct, n_oct, specs, local=True)
    layers = comp.run(Points(c_band, g_gcd))         # g_gcd input

    assert sorted(comp.layout.layout) == [0, 1, 2, 3]
    assert comp.local_cut > 0.1                       # wraps the north pole vertex
    for cl in layers:                                 # all hexes compact (no slivers)
        polys = cl.verts.coords.reshape(-1, 6, 2)
        edges = np.linalg.norm(polys - polys[:, [1, 2, 3, 4, 5, 0]], axis=2)
        assert edges.max() <= 3.0 * np.median(edges)


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
