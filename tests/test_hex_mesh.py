# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""
Structural tests for HexMesh.create() and HexMesh.densify().

Checks:
  1. Hex count == 12 * 9**layer for layers 0-2.
  2. All vertex oids are in [0, 7] — oid=255 indicates OOB.
  3. All face vertex indices are in range.
  4. Interior hexes (verts 4,5 same oid as 0-3) have equal edge lengths in b_oct.
  5. After projection to g_gcd, all hexes have equal geodesic edge lengths
     (confirms boundary hexes are correct globally, not just locally).
  6. densify(L) — shape, index range, and no missing _vd keys.
     Parametrised over (coarse_layer, finest_layer) pairs.  By the fractal
     nature of H9, densify(k) with finest=k+1 (delta=1, factor=3) is
     structurally identical at every layer; similarly for delta=2 etc.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def reg():
    from hhg9 import Registrar
    return Registrar()


@pytest.fixture(scope="module")
def mesh_l0(reg):
    from hhg9.h9.grid import HexMesh
    return HexMesh.create(0, reg)


@pytest.fixture(scope="module")
def mesh_l1(reg):
    from hhg9.h9.grid import HexMesh
    return HexMesh.create(1, reg)


@pytest.fixture(scope="module")
def mesh_l2(reg):
    from hhg9.h9.grid import HexMesh
    return HexMesh.create(2, reg)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _edge_lengths_boct(mesh):
    """Return (N, 6) array of cyclic edge lengths for each hex in b_oct xy."""
    xy = mesh.pts.coords          # (V, 2)
    faces = mesh.faces            # (N, 6)
    hv = xy[faces]                # (N, 6, 2)
    edges = np.roll(hv, -1, axis=1) - hv   # (N, 6, 2)
    return np.linalg.norm(edges, axis=2)    # (N, 6)


def _is_interior(mesh):
    """True for hexes whose verts 4,5 share the same oid as verts 0-3."""
    oids = mesh.pts.oid[mesh.faces]   # (N, 6)
    return (oids[:, 4] == oids[:, 0]) & (oids[:, 5] == oids[:, 0])


def _haversine_deg(ll1, ll2):
    """Great-circle distance in degrees between (lat,lon) pairs, vectorised."""
    lat1, lon1 = np.radians(ll1[..., 0]), np.radians(ll1[..., 1])
    lat2, lon2 = np.radians(ll2[..., 0]), np.radians(ll2[..., 1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))))


# ---------------------------------------------------------------------------
# 1. Hex count
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("layer,mesh_fix", [
    (0, "mesh_l0"),
    (1, "mesh_l1"),
    (2, "mesh_l2"),
])
def test_hex_count(layer, mesh_fix, request):
    mesh = request.getfixturevalue(mesh_fix)
    expected = 12 * 9**layer
    assert len(mesh.faces) == expected, (
        f"Layer {layer}: expected {expected} hexes, got {len(mesh.faces)}")


# ---------------------------------------------------------------------------
# 2. OID validity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mesh_fix", ["mesh_l0", "mesh_l1", "mesh_l2"])
def test_oids_valid(mesh_fix, request):
    mesh = request.getfixturevalue(mesh_fix)
    oids = mesh.pts.oid
    assert not np.any(oids == 255), \
        f"OOB oid (255) found at indices {np.where(oids == 255)[0]}"
    bad = np.unique(oids[~np.isin(oids, np.arange(8))])
    assert len(bad) == 0, f"Unexpected oid values: {bad}"


# ---------------------------------------------------------------------------
# 3. Face index range
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mesh_fix", ["mesh_l0", "mesh_l1", "mesh_l2"])
def test_face_indices_in_range(mesh_fix, request):
    mesh = request.getfixturevalue(mesh_fix)
    n_verts = len(mesh.pts.coords)
    assert mesh.faces.min() >= 0
    assert mesh.faces.max() < n_verts, \
        f"Face index {mesh.faces.max()} >= vertex pool size {n_verts}"


# ---------------------------------------------------------------------------
# 4. Equal edge lengths for interior hexes (b_oct xy)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mesh_fix", ["mesh_l0", "mesh_l1", "mesh_l2"])
def test_interior_hex_edges_equal(mesh_fix, request):
    mesh = request.getfixturevalue(mesh_fix)
    interior = _is_interior(mesh)
    if not interior.any():
        pytest.skip("No interior hexes at this layer (all hexes are boundary hexes)")
    lengths = _edge_lengths_boct(mesh)[interior]   # (K, 6)
    # All 6 edges of each interior hex should be equal
    std = np.std(lengths, axis=1)                  # (K,)
    worst_hex = np.argmax(std)
    assert std[worst_hex] < 1e-10, (
        f"Non-uniform edges in interior hex {worst_hex}: "
        f"lengths={lengths[worst_hex].round(6)}, std={std[worst_hex]:.2e}")


# ---------------------------------------------------------------------------
# 5. Geodesic edge lengths after projection (all hexes, including boundary)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mesh_fix,tol_deg", [
    ("mesh_l0", 5.0),   # ~4° variation is inherent to octahedral projection at L0
    ("mesh_l1", 5.0),   # ~1.75° at L1; broken correction gives std > 50°
])
def test_geodesic_edges_equal(mesh_fix, tol_deg, reg, request):  # noqa: E302
    from hhg9.h9.grid import valid_ll
    mesh = request.getfixturevalue(mesh_fix)
    b_oct = reg.domain('b_oct')
    g_gcd = reg.domain('g_gcd')

    ll_pts  = reg.project(mesh.pts, [b_oct, g_gcd])   # (V, 2) lat/lon
    ll_hexes = ll_pts.coords[mesh.faces]               # (N, 6, 2)

    ok = valid_ll(ll_hexes)                            # drop antimeridian hexes
    ll = ll_hexes[ok]                                  # (M, 6, 2)

    # 6 cyclic edges per hex
    nxt = np.roll(ll, -1, axis=1)                      # (M, 6, 2)
    edge_deg = _haversine_deg(ll, nxt)                 # (M, 6) geodesic degrees

    std = np.std(edge_deg, axis=1)                     # (M,)
    worst = np.argmax(std)
    assert std[worst] < tol_deg, (
        f"Non-uniform geodesic edges in hex {worst}: "
        f"edges={edge_deg[worst].round(6)} deg, std={std[worst]:.2e}")


# ---------------------------------------------------------------------------
# Fixtures — multi-layer meshes for densify tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mesh_d1_fine1(reg):
    """finest=1, layers=[0,1] — delta=1 densify(0) test case."""
    from hhg9.h9.grid import HexMesh
    return HexMesh.create([0, 1], reg)


@pytest.fixture(scope="module")
def mesh_d1_fine2(reg):
    """finest=2, layers=[0,1,2] — delta=1 densify(1) and delta=2 densify(0)."""
    from hhg9.h9.grid import HexMesh
    return HexMesh.create([0, 1, 2], reg)


@pytest.fixture(scope="module")
def mesh_d1_fine3(reg):
    """finest=3, layers=[0,1,2,3] — delta=1 densify(2), delta=2 densify(1), etc."""
    from hhg9.h9.grid import HexMesh
    return HexMesh.create(range(4), reg)


# ---------------------------------------------------------------------------
# 6. densify — helper and tests
# ---------------------------------------------------------------------------

def _densify_scan(mesh, L):
    """Run densify(L) and check all returned indices are in range.

    Returns a list of dicts, one per out-of-range index:
        hex, col, index, n_verts
    """
    delta = mesh.fine - L
    if delta <= 0:
        return []
    n_verts = len(mesh.pts.coords)
    try:
        d = mesh.densify(L)
    except Exception as exc:
        return [dict(hex=-1, col=-1, index=-1, n_verts=n_verts, error=str(exc))]
    misses = []
    for h in range(len(d)):
        for col in range(d.shape[1]):
            idx = int(d[h, col])
            if idx < 0 or idx >= n_verts:
                misses.append(dict(hex=h, col=col, index=idx, n_verts=n_verts))
    return misses


def _densify_report(misses, coarse_L, finest):
    """Format a concise failure message from _densify_scan output."""
    if misses and 'error' in misses[0]:
        return f"densify({coarse_L}, finest={finest}) raised: {misses[0]['error']}"
    n = len(misses)
    first = misses[0]
    return (
        f"densify({coarse_L}, finest={finest}): {n} out-of-range indices. "
        f"First: hex={first['hex']} col={first['col']} index={first['index']} "
        f"n_verts={first['n_verts']}"
    )


# Each tuple: (fixture_name, coarse_L, expect_pass)
# expect_pass=True  → assert shape/range valid and no missing keys.
# expect_pass=False → currently failing; xfail with diagnostic so suite stays green.
#
# Recursive _get_verts correctly handles cross-face edges at all layers.
_DENSIFY_CASES = [
    # delta=1 (factor=3)
    ("mesh_d1_fine1", 0, True),   # 0←1  ✓
    ("mesh_d1_fine2", 1, True),   # 1←2  ✓
    ("mesh_d1_fine3", 2, True),   # 2←3  ✓
    # delta=2 (factor=9)
    ("mesh_d1_fine2", 0, True),   # 0←2  ✓
    ("mesh_d1_fine3", 1, True),   # 1←3  ✓
    # delta=3 (factor=27)
    ("mesh_d1_fine3", 0, True),   # 0←3  ✓
]


@pytest.mark.parametrize("fix_name,coarse_L,expect_pass", _DENSIFY_CASES)
def test_densify_no_missing_keys(fix_name, coarse_L, expect_pass, request):
    """Every interpolated edge position must be present in _vd (both oids tried)."""
    mesh   = request.getfixturevalue(fix_name)
    finest = mesh.fine
    misses = _densify_scan(mesh, coarse_L)
    if expect_pass:
        assert not misses, _densify_report(misses, coarse_L, finest)
    else:
        pytest.xfail(_densify_report(misses, coarse_L, finest) if misses
                     else f"densify({coarse_L}, finest={finest}) unexpectedly passed")


@pytest.mark.parametrize("fix_name,coarse_L,expect_pass", _DENSIFY_CASES)
def test_densify_shape_and_range(fix_name, coarse_L, expect_pass, request):
    """densify(L) must return shape (N_L, 6*3^delta) with all indices in range."""
    mesh   = request.getfixturevalue(fix_name)
    finest = mesh.fine
    n_L    = len(mesh[coarse_L])
    n_v    = len(mesh.pts.coords)
    delta  = finest - coarse_L
    factor = int(3 ** delta)

    def _check():
        d = mesh.densify(coarse_L)
        assert d.shape == (n_L, 6 * factor), \
            f"shape: expected ({n_L}, {6 * factor}), got {d.shape}"
        assert d.min() >= 0 and d.max() < n_v, \
            f"index out of range [0, {n_v})"

    if expect_pass:
        _check()
    else:
        with pytest.raises(Exception):
            _check()


# ---------------------------------------------------------------------------
# 7. densify — UV linearity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fix_name,coarse_L,expect_pass", _DENSIFY_CASES)
def test_densify_uv_linear(fix_name, coarse_L, expect_pass, request):
    """Every intermediate densified vertex must lie on the UV line between its
    edge endpoints, with equal spacing (no fallback-to-endpoint collapse).

    For cross-mode edges (one mo==0 endpoint, one mo==1 endpoint) the
    interpolation is done in mode-1 space — the mo==0 endpoint's ib is negated
    before computing the expected step — matching _get_verts behaviour.
    """
    from hhg9.h9 import H9O

    mesh   = request.getfixturevalue(fix_name)
    finest = mesh.fine
    delta  = finest - coarse_L
    factor = int(3 ** delta)

    def _check():
        d    = mesh.densify(coarse_L)     # (N, 6*factor)
        fl   = mesh[coarse_L]             # (N, 6) coarse face indices
        ia   = mesh.ia                    # integer U per vertex
        ib   = mesh.ib                    # integer V per vertex
        oids = mesh.pts.oid               # oid per vertex

        failures = []
        for hi in range(len(fl)):
            hx   = fl[hi]
            loop = d[hi]
            for e in range(6):
                v_start = int(hx[e])
                v_end   = int(hx[(e + 1) % 6])
                ia_a, ib_a = int(ia[v_start]), int(ib[v_start])
                ia_b, ib_b = int(ia[v_end]),   int(ib[v_end])

                # For cross-mode edges interpolate in mode-1 space
                oid_a = int(oids[v_start])
                oid_b = int(oids[v_end])
                if oid_a != oid_b:
                    mo_a = int(H9O.oid_mo[oid_a])
                    mo_b = int(H9O.oid_mo[oid_b])
                    if mo_a != mo_b:
                        if mo_a == 0:
                            ib_a = -ib_a   # convert mo==0 start to mode-1 space
                        else:
                            ib_b = -ib_b   # convert mo==0 end to mode-1 space

                dia = ia_b - ia_a
                dib = ib_b - ib_a

                # Corner vertex must match coarse face index
                assert int(loop[e * factor]) == v_start, (
                    f"hex {hi} edge {e}: loop corner != coarse vert")

                # Intermediate vertices (k = 1 … factor-1)
                for k in range(1, factor):
                    exp_ia = ia_a + k * dia // factor
                    exp_ib = ib_a + k * dib // factor
                    v = int(loop[e * factor + k])
                    act_ia = int(ia[v])
                    act_ib = int(ib[v])
                    if act_ia != exp_ia or act_ib != exp_ib:
                        failures.append(
                            dict(hex=hi, edge=e, k=k,
                                 exp=(exp_ia, exp_ib),
                                 act=(act_ia, act_ib)))
                        if len(failures) >= 5:
                            break
                if len(failures) >= 5:
                    break
            if len(failures) >= 5:
                break

        assert not failures, (
            f"densify({coarse_L}, finest={finest}): {len(failures)} non-linear "
            f"intermediate verts. First: {failures[0]}")

    if expect_pass:
        _check()
    else:
        pytest.xfail(f"densify({coarse_L}, finest={finest}) UV linearity not yet passing")


# ---------------------------------------------------------------------------
# 8. densify — no repeated vertex within a single hex loop
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fix_name,coarse_L,expect_pass", _DENSIFY_CASES)
def test_densify_no_repeated_vertices(fix_name, coarse_L, expect_pass, request):
    """No vertex index may appear more than once in the densified loop of a hex."""
    mesh   = request.getfixturevalue(fix_name)
    finest = mesh.fine

    def _check():
        d  = mesh.densify(coarse_L)   # (N, 6*factor)
        failures = []
        for hi in range(len(d)):
            loop = d[hi].tolist()
            seen = set()
            dups = []
            for idx in loop:
                if idx in seen:
                    dups.append(idx)
                seen.add(idx)
            if dups:
                failures.append(dict(hex=hi, duplicates=dups, loop=loop))
                if len(failures) >= 3:
                    break
        assert not failures, (
            f"densify({coarse_L}, finest={finest}): repeated vertex indices in "
            f"{len(failures)} hex(es). First: {failures[0]}")

    if expect_pass:
        _check()
    else:
        pytest.xfail(f"densify({coarse_L}, finest={finest}) repeated-vertex check not yet passing")


# ---------------------------------------------------------------------------
# 9. densify — shared-edge consistency (neighbour reversal)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fix_name,coarse_L,expect_pass", _DENSIFY_CASES)
def test_densify_edge_reversal(fix_name, coarse_L, expect_pass, request):
    """For every shared edge, the densified vertex sequence of one hex must be
    the exact reverse of its neighbour's sequence for the same edge."""
    mesh   = request.getfixturevalue(fix_name)
    finest = mesh.fine
    delta  = finest - coarse_L
    factor = int(3 ** delta)
    n_loop = 6 * factor

    def _check():
        d  = mesh.densify(coarse_L)   # (N, n_loop)
        fl = mesh[coarse_L]           # (N, 6)

        # Build edge_key → list of (hi, e, full_edge_sequence including both endpoints)
        edge_map: dict = {}
        for hi in range(len(fl)):
            loop = d[hi]
            for e in range(6):
                v_start = int(fl[hi, e])
                v_end   = int(fl[hi, (e + 1) % 6])
                # Full edge sequence: factor+1 vertices (start … end inclusive)
                seq = tuple(int(loop[(e * factor + k) % n_loop])
                            for k in range(factor + 1))
                key = (min(v_start, v_end), max(v_start, v_end))
                edge_map.setdefault(key, []).append((hi, e, seq))

        failures = []
        for key, entries in edge_map.items():
            if len(entries) != 2:
                continue  # boundary edge — no neighbour to compare
            (hi_a, e_a, seq_a), (hi_b, e_b, seq_b) = entries
            if seq_a != seq_b[::-1]:
                failures.append(dict(
                    key=key,
                    hex_a=hi_a, edge_a=e_a, seq_a=seq_a[:4],
                    hex_b=hi_b, edge_b=e_b, seq_b=seq_b[:4],
                ))
                if len(failures) >= 5:
                    break

        assert not failures, (
            f"densify({coarse_L}, finest={finest}): {len(failures)} edge-reversal "
            f"mismatches. First: {failures[0]}")

    if expect_pass:
        _check()
    else:
        pytest.xfail(f"densify({coarse_L}, finest={finest}) edge reversal not yet passing")
