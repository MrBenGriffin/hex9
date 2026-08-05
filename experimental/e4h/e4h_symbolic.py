# Part of the Hex9 (H9) Project
# Copyright ©2026, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""Symbolic/structural E4H encode-decode: oracle byte-parity harness.

PROMOTED 2026-08-05: hhg9/h9/e4h.py now IS the symbolic implementation,
so running this file compares symbolic-to-symbolic (expect 100%). The
recorded oracle run below was against the GEOMETRIC PoC (this commit's
parent): 856-pt census x 4 (layer, depth) regimes — byte-identical
except 35 knife-edge points exactly ON a cut line (piece margin ~1e-13,
frames agree to 2.6e-13; either side valid), decode parity <=4e-10,
symbolic roundtrip 856/856, two-seam unfolds needed: 0.

The geometric PoC retained three geometric probes:
  1. `_fit`: per-(host, octant) least-squares similarity fit on decoded
     float ring corners, reflection search, 0.9-inset chirality hack;
  2. `_class_of`: atan2 read of the piece's diameter direction;
  3. decode's digit->piece inversion via candidate-composition probing.

This spike replaces all three with structure derived from H9P/H9O
constants at import time, and byte-compares against the PoC (oracle
methodology, cf. h9_cell_ancestor):

  * ROT: the rep-4 child maps rotate by (0, 120, 180, 240) degrees;
    class is an integer rotation accumulator, no angles measured.
  * A[p_mo, c2]: the state frame is a constant similarity (reflection
    structural: H9P.hx is clockwise, the canonical ring CCW) times
    3^layer, translated by the host centre. No fitting.
  * UNFOLD[o][k]: the exact inverse of fold_to_octant's per-seam
    isometry; points in a neighbour octant are unfolded into the
    host's frame, so one frame per host suffices (local-net doctrine:
    residuals are taken in the host's unfolded frame).

Run: python experimental/e4h/e4h_symbolic.py
"""
import math
import sys
import uuid as uuid_mod
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hhg9.h9.e4h import (_CEN, _HALVES, _MAPS, _TGT, _classify, _digit5,
                         _fwd, _host_layer, _inv, E_MARK,
                         h9e_decode, h9e_encode, h9e_split)

# ------------------------------------------------ symbolic constants


def _rotations():
    """Rotation (units of 60 deg) of each rep-4 child map: (0, 2, 3, 4).
    Asserted against the canonical maps at import."""
    rot = []
    for a, _b in _MAPS:
        th = math.degrees(math.atan2(a.imag, a.real)) % 360
        u = th / 60.0
        assert abs(u - round(u)) < 1e-9, f'child map rotation {th} off-lattice'
        rot.append(int(round(u)) % 6)
    assert rot == [0, 2, 3, 4], f'unexpected child rotations {rot}'
    return tuple(rot)


ROT = _rotations()
_ROT3_PIECE = {2: 1, 0: 2, 1: 3}   # rot mod 3 -> edge piece (distinct)


def _cls(s):
    """Direction class (1..3) of a composition with accumulated rotation
    s (units of 60 deg; half-1 contributes 3)."""
    return ((s - 1) % 3) + 1


def _state_frames():
    """A[p_mo, c2]: unit-scale similarity constant taking the CENTRED
    H9P.hx ring (host chart, clockwise) onto the canonical ring _TGT
    (CCW) with the structural reflection. Verified on all 6 corners."""
    from hhg9.h9 import H9P
    A = np.zeros((2, 3), dtype=np.complex128)
    for mo in range(2):
        for c2 in range(3):
            hx = H9P.hx[mo, c2]
            rel = hx - hx.mean(axis=0)
            z = rel[:, 0] + 1j * rel[:, 1]
            a = _TGT[0] / np.conj(z[0])
            assert np.max(np.abs(a * np.conj(z) - _TGT)) < 1e-9, \
                f'hx[{mo},{c2}] ring is not the canonical ring under conj'
            A[mo, c2] = a
    return A


_A = _state_frames()


def _unfolds():
    """UNFOLD[o][k]: 3x2 affine taking the canonical frame of neighbour
    octant g = o ^ (1 << SV_CORNER_AXIS[mode][k]) into o's EXTENDED
    frame — the exact inverse of fold_to_octant's per-seam isometry
    (same corner-matching construction, solved backwards)."""
    from hhg9.h9 import H9P
    from hhg9.h9.constants import H9O
    from hhg9.h9.polygon import SV_CORNER_AXIS
    tab = {}
    for o in range(8):
        mode = int(H9O.oid_mo[o])
        tri = H9P.sv[mode]
        ax = SV_CORNER_AXIS[mode]
        mode_b = 1 - mode
        tri_b = H9P.sv[mode_b]
        axis_b = SV_CORNER_AXIS[mode_b]
        for k in range(3):
            a, b = tri[(k + 1) % 3], tri[(k + 2) % 3]
            e = b - a
            e = e / np.hypot(e[0], e[1])
            d = tri[k] - a
            refl = a + 2.0 * float(d @ e) * e - d
            src = np.array([a, b, refl])
            axes = (ax[(k + 1) % 3], ax[(k + 2) % 3], ax[k])
            dst = np.array([tri_b[axis_b.index(x)] for x in axes])
            m = np.linalg.solve(np.column_stack([dst, np.ones(3)]), src)
            g = o ^ (1 << ax[k])
            tab[o, g] = m
    return tab


_UNFOLD = _unfolds()


# --------------------------------------------------- symbolic host info
def _host_info(host, reg, b_oct):  # noqa: shadows promoted e4h._host_info
    """(centre, oid, c2, p_mo, layer) — one h9_dec per host (exact fold),
    tail-nibble state unpack; no ring corners, no folding."""
    from hhg9.h9.tail import tail_unpack_reversible
    from hhg9.h9.uuid_address import h9_dec
    layer = _host_layer(host)
    dpts = h9_dec([host], b_oct)
    c = dpts.coords[0][:2]
    oid = int(dpts.oid[0])
    c2, _r, p_mo = tail_unpack_reversible(
        np.array([host.int & 0xF], dtype=np.uint8))
    return c, oid, int(c2[0]), int(p_mo[0]), layer


def _host_frame(info):
    """State frame of a host in its own octant: fr = (a, b, rf=True),
    a = A[p_mo, c2] * 3^layer, b = -a * conj(centre)."""
    c, _oid, c2, p_mo, layer = info
    a = _A[p_mo, c2] * (3.0 ** layer)
    b = -a * np.conj(complex(c[0], c[1]))
    return a, b, True


# ------------------------------------------------------- symbolic API
TWO_SEAM = []   # census: (host.int, g, o) needing a 2-seam unfold


def sym_encode(lats, lons, layer=6, depth=2, reg=None):
    from hhg9 import Points, Registrar
    from hhg9.h9.uuid_address import (_batch_int_to_nibbles,
                                      batch_nibbles_to_int, h9_bin_pts)
    if layer + 2 + depth > 30:
        raise ValueError('depth does not fit the nibble budget')
    reg = reg or Registrar()
    g_gcd, b_oct = reg.domain('g_gcd'), reg.domain('b_oct')
    lats = np.asarray(lats, float).ravel()
    lons = np.asarray(lons, float).ravel()
    bp = reg.project(Points(np.column_stack([lats, lons]), g_gcd),
                     [g_gcd, b_oct])
    P = bp.coords[:, :2]
    O = np.asarray(bp.oid)
    hosts = h9_bin_pts(bp, layer)

    infos, frames = {}, {}
    out = [None] * len(hosts)
    for i, (u, p, g) in enumerate(zip(hosts, P, O)):
        if u.int not in infos:
            infos[u.int] = _host_info(u, reg, b_oct)
            frames[u.int] = _host_frame(infos[u.int])
        c, o, c2_h, _mo, _lay = infos[u.int]
        g = int(g)
        if g != o:
            if (o, g) not in _UNFOLD:
                TWO_SEAM.append((u.int, g, o))
                out[i] = None
                continue
            p = np.array([p[0], p[1], 1.0]) @ _UNFOLD[o, g]
        w = _fwd(frames[u.int], complex(p[0], p[1]))
        half, w = _classify(w, _HALVES)
        s = 3 * half
        digits = []
        for _ in range(depth):
            k, w = _classify(w, _MAPS)
            s += ROT[k]
            digits.append(0 if k == 0 else _digit5(o, c2_h, half, _cls(s)))
        nib = _batch_int_to_nibbles([u.int], n=32)[0].copy()
        nib[layer + 1] = E_MARK
        nib[layer + 2] = half
        for j, d in enumerate(digits):
            nib[layer + 3 + j] = d
        out[i] = uuid_mod.UUID(int=int(batch_nibbles_to_int(nib[None, :])[0]))
    return out


def sym_decode(uuids, reg=None, _probe=_CEN):
    from hhg9 import Points, Registrar
    from hhg9.h9.polygon import fold_to_octant
    reg = reg or Registrar()
    g_gcd, b_oct = reg.domain('g_gcd'), reg.domain('b_oct')
    infos, frames = {}, {}
    lats, lons = np.zeros(len(uuids)), np.zeros(len(uuids))
    for i, u in enumerate(uuids):
        host, half, digits = h9e_split(u)
        if host.int not in infos:
            infos[host.int] = _host_info(host, reg, b_oct)
            frames[host.int] = _host_frame(infos[host.int])
        _c, oid, c2h, _mo, _lay = infos[host.int]
        fr = frames[host.int]
        comp = _HALVES[half]
        s = 3 * half
        d2c = {_digit5(oid, c2h, half, cl): cl for cl in (1, 2, 3)}
        for d in digits:
            if d == 0:
                k = 0
            else:
                if d not in d2c:
                    raise ValueError(f'digit {d} invalid in this context')
                k = _ROT3_PIECE[(d2c[d] - s) % 3]
            a2, b2 = _MAPS[k]
            comp = (comp[0] * a2, comp[0] * b2 + comp[1])
            s += ROT[k]
        rep = _inv(fr, comp[0] * _probe + comp[1])
        xy, roid = fold_to_octant(np.array([[rep.real, rep.imag]]), oid)
        gp = reg.project(Points(xy, b_oct, oid=np.asarray(roid)),
                         [b_oct, g_gcd])
        lats[i], lons[i] = gp.coords[0, 0], gp.coords[0, 1]
    return lats, lons


# ------------------------------------------------------------- census
def _census_points():
    """Global grid + equator seam + octahedron vertices + poles."""
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
                if la > 89.99:
                    la = 89.99
                if la < -89.99:
                    la = -89.99
                pts.append((la, ((lo + 180) % 360) - 180))
    pts.append((89.999, 45.0))
    pts.append((-89.999, -135.0))
    arr = np.array(pts)
    return arr[:, 0], arr[:, 1]


def main():
    from hhg9 import Registrar
    reg = Registrar()
    lats, lons = _census_points()
    print(f'census: {len(lats)} points')
    total_mism = 0
    for layer, depth in ((3, 3), (6, 2), (6, 5), (4, 1)):
        oracle = h9e_encode(lats, lons, layer=layer, depth=depth, reg=reg)
        mine = sym_encode(lats, lons, layer=layer, depth=depth, reg=reg)
        mism = [(i, a, b) for i, (a, b) in enumerate(zip(oracle, mine))
                if b is None or a.int != b.int]
        total_mism += len(mism)
        print(f'A={layer} B={depth}: encode parity '
              f'{len(lats) - len(mism)}/{len(lats)}'
              + (f'  MISMATCH {len(mism)}' if mism else '  PASS'))
        for i, a, b in mism[:5]:
            print(f'   pt ({lats[i]:.3f},{lons[i]:.3f}): '
                  f'oracle {a} vs {b}')
        # decode parity on the oracle addresses
        la_o, lo_o = h9e_decode(oracle, reg=reg)
        la_s, lo_s = sym_decode(oracle, reg=reg)
        dmax = max(np.max(np.abs(la_o - la_s)), np.max(np.abs(lo_o - lo_s)))
        print(f'          decode parity max |dlat,dlon| = {dmax:.2e}'
              + ('  PASS' if dmax < 1e-8 else '  FAIL'))
        # roundtrip through the symbolic pair
        rt = sym_encode(la_s, lo_s, layer=layer, depth=depth, reg=reg)
        bad = sum(1 for a, b in zip(oracle, rt) if b is None or a.int != b.int)
        print(f'          symbolic roundtrip {len(oracle) - bad}/{len(oracle)}'
              + ('  PASS' if bad == 0 else f'  FAIL {bad}'))
    print(f'two-seam unfolds needed: {len(TWO_SEAM)}'
          + ('' if not TWO_SEAM else f'  e.g. {TWO_SEAM[:3]}'))
    print('ALL PASS' if total_mism == 0 and not TWO_SEAM else 'DIFFS FOUND')


if __name__ == '__main__':
    main()
