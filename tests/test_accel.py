# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""
Tests for the optional native backend (``hhg9.accel.libhex9``).

The C library (libhex9.dylib/.so) may or may not be built in a given
environment, so the functional tests skip cleanly when ``backend()`` is None.
``backend()`` itself is always exercised (it must never raise and must be
idempotent — the result is cached behind a sentinel).
"""
import numpy as np
import pytest

import hhg9.accel.libhex9 as L


def test_backend_is_idempotent_and_never_raises():
    b = L.backend()
    assert b is None or type(b).__name__ == "_Lib"
    assert L.backend() is b          # cached behind the sentinel


@pytest.fixture(scope="module")
def lib():
    b = L.backend()
    if b is None:
        pytest.skip("native libhex9 not built in this environment")
    return b


def test_version_is_nonempty_string(lib):
    assert isinstance(lib.version, str) and len(lib.version) > 0


def test_project_unproject_roundtrip(lib):
    """lon/lat → b_oct → lon/lat must recover the input when the same use_warp
    flag is used on both legs (the do/undo warp hazard noted in the wrapper)."""
    lon = np.array([-0.12, -74.0, 151.2, 0.0, 90.0])
    lat = np.array([51.5, 40.7, -33.9, 0.0, -45.0])
    cx, cy, oid = lib.project_many(lon, lat, use_warp=False)
    assert oid.shape == lon.shape
    assert np.all((oid >= 0) & (oid <= 7))
    lon2, lat2 = lib.unproject_many(cx, cy, oid, use_warp=False)
    assert np.abs(lon2 - lon).max() < 1e-6
    assert np.abs(lat2 - lat).max() < 1e-6
