# Part of the Hex9 (H9) Project
# Copyright ©2026, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""Census-grade regression tests for hhg9.h9.e4h (promoted from the
spike experimental/e4h/e4h_symbolic.py, 2026-08-05).

The 856-point global census — lat/lon grid over all octants, equator
seam band, cone-point rings around the six octahedron vertices, poles
— is run through four (layer, depth) regimes and asserts, per regime:

  * BYTE PARITY against the frozen geometric-probe oracle
    (tests/_e4h_geometric_ref.py = hhg9/h9/e4h.py @ 1ba4401): the two
    implementations are independent derivations (least-squares ring
    fits + atan2 class probes vs constant state frames + integer
    rotation accumulators), identical everywhere OFF cut lines.
  * the knife-edge doctrine: points ON a cut line (classification
    margin below KNIFE_TOL in canonical units) may legitimately fall
    either side, so they are excluded from byte parity — but their
    count is pinned per regime (measured 2026-08-05: on-cut margins
    are <= ~4e-13 and the nearest off-cut margin is ~1e-5, so the
    1e-9 tolerance sits in the middle of an ~8-order gap and the
    counts are stable). Of the excluded points, 35 in total actually
    disagree (1/14/14/6), matching the recorded census in the module
    docstring of hhg9.h9.e4h.
  * decode parity oracle-vs-symbolic <= 1e-8 deg (observed <= 4e-10);
  * symbolic decode -> re-encode identity on ALL census points, knife
    edges included (decode returns leaf centroids, which are interior).

Everything is deterministic: the census is a fixed lattice (no RNG,
no wall clock); margins and counts are pure FP functions of it.
Whole file runs in ~10 s. A cross-implementation bridge against the
installed `hex9` wheel is included but skips until a wheel >= 2.3.0
ships e4h bindings.
"""
import math
import uuid as uuid_mod

import numpy as np
import pytest

from hhg9 import Points, Registrar
from hhg9.h9 import e4h
from hhg9.h9.e4h import h9e_decode, h9e_encode
from tests import _e4h_geometric_ref as e4h_geo

REGIMES = ((3, 3), (6, 2), (6, 5), (4, 1))

# Knife-edge doctrine: margin below this (canonical units) = ON a cut.
KNIFE_TOL = 1e-9
# Pinned exclusion counts per regime (small, stable — see docstring).
KNIFE_COUNT = {(3, 3): 55, (6, 2): 66, (6, 5): 71, (4, 1): 44}


# ------------------------------------------------------------ census set
def _census_points():
    """The spike's 856-point global census: grid + equator seam band +
    cone-point rings + poles. Deterministic lattice, no RNG."""
    pts = []
    for lat in range(-80, 81, 10):
        for lon in range(-170, 180, 20):
            pts.append((lat + 0.37, lon + 0.53))
    for lon in np.arange(-179.5, 180, 7.0):        # equator seam band
        for lat in (0.0, 0.21, -0.21, 1.7, -1.7):
            pts.append((lat, lon))
    verts = [(90, 0), (-90, 0), (0, 0), (0, 90), (0, 180), (0, -90)]
    for vlat, vlon in verts:                        # cone-point rings
        for r in (0.4, 2.5, 8.0):
            for th in np.arange(0, 360, 22.5):
                la = vlat + r * math.cos(math.radians(th))
                lo = vlon + r * math.sin(math.radians(th))
                la = min(89.99, max(-89.99, la))
                pts.append((la, ((lo + 180) % 360) - 180))
    pts.append((89.999, 45.0))
    pts.append((-89.999, -135.0))
    arr = np.array(pts)
    return arr[:, 0], arr[:, 1]


def _min_cut_margin(lats, lons, layer, depth, reg):
    """Per-point minimum classification margin over the whole descent:
    at the half cut and at each of the `depth` rep-4 cuts, margin =
    best minus runner-up signed-distance score in canonical units.
    ~1e-13 means the point sits ON a cut line (either side valid);
    generic census points score >= ~1e-5. Mirrors h9e_encode's descent
    using the same e4h internals (frames, unfolds, classifier)."""
    from hhg9.h9.uuid_address import h9_bin_pts
    g_gcd, b_oct = reg.domain('g_gcd'), reg.domain('b_oct')
    bp = reg.project(Points(np.column_stack([lats, lons]), g_gcd),
                     [g_gcd, b_oct])
    P, O = bp.coords[:, :2], np.asarray(bp.oid)
    hosts = h9_bin_pts(bp, layer)
    infos, frames = {}, {}
    out = np.empty(len(hosts))
    for i, (u, p, g) in enumerate(zip(hosts, P, O)):
        if u.int not in infos:
            infos[u.int] = e4h._host_info(u, b_oct)
            frames[u.int] = e4h._host_frame(infos[u.int])
        _c, o, _c2, _mo, _lay = infos[u.int]
        if int(g) != o:
            p = np.array([p[0], p[1], 1.0]) @ e4h._unfolds()[o, int(g)]
        w = e4h._fwd(frames[u.int], complex(p[0], p[1]))
        m = math.inf
        for cand in (e4h._HALVES,) + (e4h._MAPS,) * depth:
            ss = sorted(e4h._score((w - b) / a) for a, b in cand)
            m = min(m, ss[-1] - ss[-2])
            _k, w = e4h._classify(w, cand)
        out[i] = m
    return out


# ---------------------------------------------------------------- fixtures
@pytest.fixture(scope='module')
def reg():
    return Registrar()


@pytest.fixture(scope='module')
def census():
    return _census_points()


@pytest.fixture(scope='module')
def sym(reg, census):
    """Memoised symbolic encodes per regime (shared across tests)."""
    lats, lons = census
    cache = {}

    def get(regime):
        if regime not in cache:
            cache[regime] = h9e_encode(lats, lons, *regime, reg)
        return cache[regime]
    return get


# ------------------------------------------------------------------ tests
@pytest.mark.parametrize('regime', REGIMES, ids=lambda r: f'A{r[0]}B{r[1]}')
def test_census_byte_parity_vs_geometric_oracle(reg, census, sym, regime):
    """Symbolic vs frozen geometric oracle: byte-identical everywhere
    off cut lines; knife-edge points (margin < KNIFE_TOL, either side
    valid) are excluded and their count pinned."""
    lats, lons = census
    ours = sym(regime)
    oracle = e4h_geo.h9e_encode(lats, lons, *regime, reg)
    margins = _min_cut_margin(lats, lons, *regime, reg)
    knife = margins < KNIFE_TOL
    assert int(knife.sum()) == KNIFE_COUNT[regime], (
        f'knife-edge exclusion count drifted: {int(knife.sum())} points '
        f'with margin < {KNIFE_TOL} (expected {KNIFE_COUNT[regime]})')
    bad = [i for i in range(len(ours))
           if not knife[i] and ours[i].int != oracle[i].int]
    assert not bad, (
        f'{len(bad)} off-cut byte mismatches vs geometric oracle, '
        f'first at ({lats[bad[0]]:.3f},{lons[bad[0]]:.3f}) '
        f'margin {margins[bad[0]]:.2e}')
    # any disagreement must be a knife-edge point (~35 total, recorded)
    mism = [i for i in range(len(ours)) if ours[i].int != oracle[i].int]
    assert all(knife[i] for i in mism)


@pytest.mark.parametrize('regime', REGIMES, ids=lambda r: f'A{r[0]}B{r[1]}')
def test_census_decode_parity_vs_geometric_oracle(reg, census, sym, regime):
    """Both implementations decode the SAME addresses to the same leaf
    representative (<= 1e-8 deg; observed <= 4e-10). No knife-edge
    doctrine applies: decode is address -> point."""
    ours = sym(regime)
    la_s, lo_s = h9e_decode(ours, reg)
    la_g, lo_g = e4h_geo.h9e_decode(ours, reg)
    dmax = max(np.max(np.abs(la_s - la_g)), np.max(np.abs(lo_s - lo_g)))
    assert dmax < 1e-8, f'decode parity max |dlat,dlon| = {dmax:.2e}'


@pytest.mark.parametrize('regime', REGIMES, ids=lambda r: f'A{r[0]}B{r[1]}')
def test_census_symbolic_roundtrip(reg, census, sym, regime):
    """decode -> re-encode identity on ALL 856 census points (100%,
    knife edges included: decode returns leaf centroids, interior by
    construction)."""
    ours = sym(regime)
    la, lo = h9e_decode(ours, reg)
    rt = h9e_encode(la, lo, *regime, reg)
    bad = sum(a.int != b.int for a, b in zip(ours, rt))
    assert bad == 0, f'symbolic roundtrip {len(ours) - bad}/{len(ours)}'


def test_census_wheel_bridge(reg, census):
    """Cross-implementation bridge: hhg9's h9e_encode vs the installed
    `hex9` wheel's e4h_encode on the census, byte-compared. Skips
    cleanly until a wheel >= 2.3.0 ships e4h bindings (2.2.1 predates
    them). Points within 1e-5 of a cut are excluded — same doctrine as
    the census, wider tolerance for the C port's independent FP path."""
    hex9 = pytest.importorskip('hex9')
    if not hasattr(hex9, 'e4h_encode'):
        pytest.skip('installed hex9 wheel predates e4h (needs >= 2.3.0)')
    layer, depth = 6, 2
    lats, lons = census
    keep = _min_cut_margin(lats, lons, layer, depth, reg) >= 1e-5
    la, lo = lats[keep], lons[keep]
    ours = h9e_encode(la, lo, layer, depth, reg)
    # wheel convention is (lon, lat) order, cf. hex9.encode
    theirs = hex9.e4h_encode(np.ascontiguousarray(lo),
                             np.ascontiguousarray(la), layer, depth)

    def as_int(v):
        if isinstance(v, uuid_mod.UUID):
            return v.int
        if isinstance(v, (bytes, bytearray)):
            return uuid_mod.UUID(bytes=bytes(v)).int
        if isinstance(v, np.ndarray):               # (16,) uint8 row
            return uuid_mod.UUID(bytes=v.tobytes()).int
        return uuid_mod.UUID(str(v)).int

    bad = [i for i, (a, b) in enumerate(zip(ours, theirs))
           if a.int != as_int(b)]
    assert not bad, (f'{len(bad)}/{len(ours)} wheel-bridge byte '
                     f'mismatches, first at ({la[bad[0]]:.3f},'
                     f'{lo[bad[0]]:.3f})')
