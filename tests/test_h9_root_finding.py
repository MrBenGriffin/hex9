# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""
Tests for the vectorised beam-search root finder (``hhg9.h9.root_finding``).

``find_coords`` searches the H9 region tree for the barycentric xy whose
projection is closest to a target.  It is load-bearing for the inverse
projection in ``projections.ak_octahedral``.

These tests use a self-contained *identity* projector (target space == the
off_xy barycentric space) so the harness needs no projection chain, yet still
exercises the full machinery: per-mode child expansion, top-k beam selection,
path extension, the chunked-recursion branch, and early-exit convergence.  An
identity projector reduces the search to "find the deepest lattice cell
containing the target", so the recovered xy must equal the target to fp
precision and the path must be a structurally valid region address.
"""
import numpy as np
import pytest

from hhg9 import Registrar, Points
from hhg9.h9.root_finding import find_coords
from hhg9.h9.lattice import H9C


def _identity(xy, octants):
    return xy


def _euclid(a, b):
    return np.linalg.norm(a - b, axis=-1)


@pytest.fixture(scope="module")
def targets():
    """b_oct points (octant-local barycentric xy) + their modes/oids."""
    reg = Registrar()
    g, b = reg.domain('g_gcd'), reg.domain('b_oct')
    ll = np.array([[51.5, -0.12], [40.7, -74.0], [-33.9, 151.2], [10.0, 20.0]])
    pb = reg.project(Points(ll, domain=g), [g, b])
    oid, mode = pb.cm()
    return pb.coords.copy(), mode, oid


def test_find_coords_recovers_target(targets):
    xy_t, mode, oid = targets
    xy, paths = find_coords(xy_t, mode, oid, H9C, _identity, _euclid,
                            depth=30, beam_width=6)
    assert np.abs(xy - xy_t).max() < 1e-12
    assert paths.shape == (len(xy_t), 31)        # depth + 1 (root + 30 layers)


def test_find_coords_root_matches_mode(targets):
    xy_t, mode, oid = targets
    _, paths = find_coords(xy_t, mode, oid, H9C, _identity, _euclid,
                           depth=12, beam_width=6)
    expected_root = np.where(mode == 1, 0x16, 0x49).astype(np.uint8)
    np.testing.assert_array_equal(paths[:, 0], expected_root)


def test_find_coords_path_is_structurally_valid(targets):
    """Every step must be a child of its parent under the parent's net_mode."""
    xy_t, mode, oid = targets
    _, paths = find_coords(xy_t, mode, oid, H9C, _identity, _euclid,
                           depth=12, beam_width=6)
    ups, downs = set(H9C.ups.tolist()), set(H9C.downs.tolist())
    for r in range(paths.shape[0]):
        for i in range(1, paths.shape[1]):
            parent, cur = int(paths[r, i - 1]), int(paths[r, i])
            legal = ups if H9C.mode[parent] == 1 else downs
            assert cur in legal, f"row {r} step {i}: {hex(cur)} not a child of {hex(parent)}"


def test_find_coords_chunking_is_equivalent(targets):
    """The chunked-recursion path (num_pts > chunk) must match the unchunked run."""
    xy_t, mode, oid = targets
    full = find_coords(xy_t, mode, oid, H9C, _identity, _euclid, depth=20, beam_width=6)
    chunked = find_coords(xy_t, mode, oid, H9C, _identity, _euclid,
                          depth=20, beam_width=6, chunk=2)
    np.testing.assert_allclose(full[0], chunked[0])
    np.testing.assert_array_equal(full[1], chunked[1])


@pytest.mark.parametrize("beam_width", [1, 3, 6])
def test_find_coords_converges_for_any_beam_width(targets, beam_width):
    xy_t, mode, oid = targets
    xy, _ = find_coords(xy_t, mode, oid, H9C, _identity, _euclid,
                        depth=30, beam_width=beam_width)
    assert np.abs(xy - xy_t).max() < 1e-9
