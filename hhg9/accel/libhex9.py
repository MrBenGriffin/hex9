# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
hhg9.accel.libhex9 — optional libhex9 acceleration for g_gcd ↔ b_oct.

libhex9 (the bundled C DGGS core) computes the octant coordinate b_oct
(cx, cy, oid) from (lon, lat), and its inverse, in single OpenMP-parallel,
GIL-released calls. Verified bit-identical to hhg9's pure-Python pipeline on the
forward map (warp-off 2e-15, warp-on f6 2.3e-15, 50k pts) and equivalent to
hhg9's Bowring backward on the inverse (same ~cm round-trip floor).

The lib OWNS the whole "AK + [warp]" hop; the warp is a toggle (hex9_set_use_warp)
that both directions read, so forward/inverse never desync (the do/undo hazard).
This backend is engine-orthogonal to pure Python by design — both paths stay
selectable for blue/green and regression, even though they agree.

Discovery (first hit wins): HEX9_LIBHEX9 (explicit file), HEX9_LIBHEX9_DIR, then
a few conventional locations relative to this repo / a sibling libhex9 build.

Axis order: hhg9 g_gcd is [lat, lon]; the libhex9 ABI is (lon, lat). The single
column swap lives in the caller (GCDBary), not here.
"""
from __future__ import annotations

import ctypes
import os
import sys
import warnings
from pathlib import Path

import numpy as np

_DLL_NAMES = ("libhex9.dylib", "libhex9.so") if sys.platform == "darwin" \
    else ("libhex9.so", "libhex9.dylib")


def _candidate_paths():
    """Yield candidate shared-library paths in priority order."""
    env = os.environ.get("HEX9_LIBHEX9")
    if env:
        yield Path(env)
    dirs = []
    env_dir = os.environ.get("HEX9_LIBHEX9_DIR")
    if env_dir:
        dirs.append(Path(env_dir))
    here = Path(__file__).resolve()
    dirs += [
        here.parents[2] / "libhex9" / "build",
        here.parents[3] / "libhex9" / "build",
        Path.home() / "Documents/Projects/libhex9/build",
    ]
    for d in dirs:
        for name in _DLL_NAMES:
            yield d / name


class _Lib:
    """Thin ctypes wrapper over libhex9's continuous projection ABI."""

    def __init__(self, path: Path):
        self.path = str(path)
        self.lib = ctypes.CDLL(self.path)
        dp = ctypes.POINTER(ctypes.c_double)
        ip = ctypes.POINTER(ctypes.c_int)

        self.lib.hex9_version.restype = ctypes.c_char_p
        self.lib.hex9_warp_init.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        self.lib.hex9_warp_init.restype = ctypes.c_int
        self.lib.hex9_set_use_warp.argtypes = [ctypes.c_int]
        self.lib.hex9_project_many.argtypes = [dp, dp, ctypes.c_size_t, dp, dp, ip]
        self.lib.hex9_project_many.restype = ctypes.c_int
        self.lib.hex9_unproject_many.argtypes = [dp, dp, ip, ctypes.c_size_t, dp, dp]
        self.lib.hex9_unproject_many.restype = ctypes.c_int

        err = ctypes.create_string_buffer(256)
        if self.lib.hex9_warp_init(err, 256) != 0:
            raise RuntimeError(f"hex9_warp_init failed: {err.value.decode()}")
        self._dp, self._ip = dp, ip

        # Canonical cell roll-up (address-space mode-0 fold), when this
        # build exports it. Callers must gate on has_cell_ancestor.
        u8p = ctypes.POINTER(ctypes.c_uint8)
        self._u8p = u8p
        try:
            self.lib.hex9_cell_ancestor_many.argtypes = [
                u8p, ctypes.c_int, ctypes.c_size_t, u8p]
            self.lib.hex9_cell_ancestor_many.restype = ctypes.c_int
            self.has_cell_ancestor = True
        except AttributeError:
            self.has_cell_ancestor = False

        # Via-sphere chain (authalic-series front-end + unit-sphere core +
        # Sphere-L6 wedge-fold warp), when this build exports it. Callers
        # must gate on has_via_sphere.
        try:
            self.lib.hex9_set_via_sphere.argtypes = [
                ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t]
            self.lib.hex9_set_via_sphere.restype = ctypes.c_int
            self.has_via_sphere = True
        except AttributeError:
            self.has_via_sphere = False
        self._via_active = False

    def set_via_sphere(self, on: bool):
        """Flip the lib's via-sphere mode (first enable lazily builds the
        sphere warp state, ~1 s). No-op when already in the wanted mode."""
        on = bool(on)
        if on == self._via_active:
            return
        if on and not self.has_via_sphere:
            raise RuntimeError("libhex9 build has no hex9_set_via_sphere")
        err = ctypes.create_string_buffer(256)
        if self.lib.hex9_set_via_sphere(1 if on else 0, err, 256) != 0:
            raise RuntimeError(
                f"hex9_set_via_sphere failed: {err.value.decode()}")
        self._via_active = on

    @property
    def version(self) -> str:
        return self.lib.hex9_version().decode()

    def project_many(self, lon, lat, use_warp: bool, via_sphere: bool = False):
        """(lon, lat)° → (cx, cy, oid) b_oct arrays. oid int32 0..7."""
        lo = np.ascontiguousarray(lon, dtype=np.float64)
        la = np.ascontiguousarray(lat, dtype=np.float64)
        n = lo.size
        cx = np.empty(n, dtype=np.float64)
        cy = np.empty(n, dtype=np.float64)
        oid = np.empty(n, dtype=np.int32)
        self.set_via_sphere(via_sphere)
        self.lib.hex9_set_use_warp(1 if use_warp else 0)
        rc = self.lib.hex9_project_many(
            lo.ctypes.data_as(self._dp), la.ctypes.data_as(self._dp), n,
            cx.ctypes.data_as(self._dp), cy.ctypes.data_as(self._dp),
            oid.ctypes.data_as(self._ip))
        if rc != 0:
            raise RuntimeError(f"hex9_project_many rc={rc}")
        return cx, cy, oid

    def cell_ancestor_many(self, uuids_u8, layer: int):
        """(N, 16) uint8 bin UUIDs -> (N, 16) canonical layer-``layer``
        ancestors (the mode-0 d_cell fold). Requires has_cell_ancestor."""
        a = np.ascontiguousarray(uuids_u8, dtype=np.uint8)
        out = np.empty_like(a)
        rc = self.lib.hex9_cell_ancestor_many(
            a.ctypes.data_as(self._u8p), int(layer), a.shape[0],
            out.ctypes.data_as(self._u8p))
        if rc != 0:
            raise RuntimeError(f"hex9_cell_ancestor_many rc={rc}")
        return out

    def unproject_many(self, cx, cy, oid, use_warp: bool,
                       via_sphere: bool = False):
        """(cx, cy, oid) b_oct → (lon, lat)° arrays. Must use the SAME use_warp
        (and via_sphere) the coordinate was produced with, or the round-trip
        drifts by one warp displacement (the do/undo hazard)."""
        cxx = np.ascontiguousarray(cx, dtype=np.float64)
        cyy = np.ascontiguousarray(cy, dtype=np.float64)
        oidd = np.ascontiguousarray(oid, dtype=np.int32)
        n = cxx.size
        lon = np.empty(n, dtype=np.float64)
        lat = np.empty(n, dtype=np.float64)
        self.set_via_sphere(via_sphere)
        self.lib.hex9_set_use_warp(1 if use_warp else 0)
        rc = self.lib.hex9_unproject_many(
            cxx.ctypes.data_as(self._dp), cyy.ctypes.data_as(self._dp),
            oidd.ctypes.data_as(self._ip), n,
            lon.ctypes.data_as(self._dp), lat.ctypes.data_as(self._dp))
        if rc != 0:
            raise RuntimeError(f"hex9_unproject_many rc={rc}")
        return lon, lat


_backend: object = ...  # sentinel: not yet attempted


def backend() -> "_Lib | None":
    """Return a loaded libhex9 backend, or None if unavailable. Cached."""
    global _backend
    if _backend is not ...:
        return _backend
    for p in _candidate_paths():
        try:
            if p and Path(p).exists():
                _backend = _Lib(Path(p))
                return _backend
        except Exception as e:  # wrong arch, missing symbol, init failure …
            warnings.warn(f"libhex9 found at {p} but failed to load: {e}")
    _backend = None
    return None
