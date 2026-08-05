# Part of the Hex9 (H9) Project
# Copyright ©2026, Ben Griffin
# Licensed under the Apache License, Version 2.0

# FROZEN GEOMETRIC REFERENCE — DO NOT MODERNISE.
# This is hhg9/h9/e4h.py as of commit 1ba4401, the geometric-probe PoC
# that served as the oracle when the symbolic implementation was
# promoted (2026-08-05). tests/test_h9e_census.py byte-compares the
# live symbolic hhg9.h9.e4h against this module over the 856-point
# global census; the two derivations are independent (least-squares
# ring fits + atan2 class probes here vs integer rotation accumulators
# + constant state frames there), so parity is a real regression pin.
# Any edit here weakens the oracle: fix bugs in hhg9/h9/e4h.py, never
# in this file.

"""Hex9+E4H proof of concept: the aperture-4 structural tail (h9e_*).

An E4H address is a hex9 address truncated at the attach layer, an 0xE
break marker ("extended" — not a valid h9 digit, so self-describing),
one HALF digit (0/1: which state-cut trapezoid of the host hexagon),
and a run of TAIL digits (0 = centre child, 1..5 = the five-symbol
enumeration) descending the aperture-4 half-hex rep-4 carrier:

    nibbles:  [h9 digits 0..A] [0xE] [half] [d1 .. dB] [0xF ...] [tail]
    label  :  <h9-label>E<half><d1..dB>          e.g.  0031586E1213

Design provenance (2026-07-31, machine-verified):
  * cells are DEFINED top-down by residual classification on the
    half-hex rep-4 carrier — exact nesting, straight edges at every
    level; truncation = containment below the E (suffix-local, unlike
    the a9 digits). Spike: experimental/e4h/e4h_descent.py, all PASS.
  * global closure over the octahedron incl. cone points:
    docs/dggs-transport-tilings.md §4b/§4c + transport_check.py
    (triad closure, e4h closure, digit CSP — all PASS/SAT).
  * digit semantics — THE FIVE-SYMBOL ENUMERATION (transport note
    §4d, promoted 2026-08-03): digit 0 = centre child; digits 1..5 =
    _digit5(oid, c2, half, class), where class (1..3) is the child's
    direction class in the host's canonical state frame and the rule
    is: 1 anchors at class (1 + half); class 3 carries q in both
    halves; (p, q, r) rotates right with the host's c2; the bases
    derive from H9O.oid_nb — the digit of an edge child NAMES THE
    OCTAHEDRAL AXIS of the neighbour octant its class points at. The
    sole gauge is the global axis→digit bijection (_AXIS_DIGIT).
  * matched pairs are GLOBAL: the two halves of every fine hexagon —
    same host, cross host, across octant seams, around cone points —
    share their final digit, and around every fine vertex the three
    cells differ. Machine provenance: rule uniqueness mod gauge +
    CLEAN vs 41,367 constraints to depth 4
    (experimental/e4h/e4h_closed_form.py, e4h_canonical.py; journey in
    §4c-§4d of the transport note). Hexagon-level canonical naming
    (one address per hexagon) remains the ownership pin (mode-0
    half); h9e_partner_point walks the pair.
  * the half digit carries one bit in a nibble: canonical form keeps
    nibble alignment (dense packing belongs to the optional u64 layer);
    values 2..0xD after the E are invalid — a free validity check.

Provisional pins (PoC): the canonical half-0 side is the T0-image of
the state ring (H9P.hx corner order, corner-0 anchored); frames for
seam-overhang octants are fitted with 0.9-inset ring points to break
the two-corner chirality degeneracy.

Public API: h9e_encode, h9e_decode, h9e_label, h9e_split,
h9e_partner_point.
"""
import math
import uuid as uuid_mod

import numpy as np

E_MARK = 0xE
_TGT = np.exp(1j * (np.pi / 6 + np.pi / 3 * np.arange(6)))  # canonical ring
_TOL = 1e-9


# ---------------------------------------------------------------- canonical
def _canonical():
    """The canonical half-hex trapezoid, its rep-4 similarity maps, the
    two half maps, and the CCW edge list for the signed-distance
    classifier. Same dissection as transport_check.a4_halfhex_canonical
    (Sahr 2011 fig. 4), rep-4 machine-verified there."""
    def cs(a):
        return np.array([math.cos(math.radians(a)), math.sin(math.radians(a))])

    T0 = np.array([cs(90), cs(150), cs(210), cs(270)])
    s3 = math.sqrt(3) / 2
    pieces = [0.5 * T0]
    for th, k0 in ((120, 210), (180, 270), (240, 330)):
        c = s3 * cs(th)
        pieces.append(np.array([c + .5 * cs(k0 + 60 * k) for k in range(4)]))
    z0 = T0[:, 0] + 1j * T0[:, 1]
    maps = []
    for p in pieces:
        zp = p[:, 0] + 1j * p[:, 1]
        a = (zp[1] - zp[0]) / (z0[1] - z0[0])
        maps.append((a, zp[0] - a * z0[0]))
    halves = [(1 + 0j, 0j), (-1 + 0j, 0j)]
    edges = []
    for i in range(4):
        p, q = z0[i], z0[(i + 1) % 4]
        edges.append((p, (q - p) / abs(q - p)))
    return z0, maps, halves, edges


_Z0, _MAPS, _HALVES, _EDGES = _canonical()
# exact centroid of the canonical trapezoid (shoelace): x̄ = −2/(3√3), ȳ = 0
_CEN = complex(-2.0 / (3.0 * math.sqrt(3.0)), 0.0)


def _score(w, edges=_EDGES):
    return min(((e.conjugate() * (w - p)).imag for p, e in edges))


def _classify(w, cand):
    us = [(w - b) / a for a, b in cand]
    ss = [_score(u) for u in us]
    k = int(np.argmax(ss))
    return k, us[k]


# ------------------------------------------------------------------- state
def _ring(host, reg, b_oct):
    """State ring of a host: corners folded to true octants, plus the
    host centre and primary octant. Mirrors _anchor_hex_latlon."""
    from hhg9.h9 import H9P
    from hhg9.h9.polygon import fold_to_octant
    from hhg9.h9.tail import tail_unpack_reversible
    from hhg9.h9.uuid_address import h9_dec
    layer = _host_layer(host)
    dpts = h9_dec([host], b_oct)
    c = dpts.coords[0][:2]
    oid = int(dpts.oid[0])
    c2, _r, p_mo = tail_unpack_reversible(
        np.array([host.int & 0xF], dtype=np.uint8))
    hx = H9P.hx[int(p_mo[0]), int(c2[0])]
    ring_rel = (hx - hx.mean(axis=0)) * (3.0 ** -layer)
    fxy, foid = fold_to_octant(c[None, :] + ring_rel, oid)
    ixy, ioid = fold_to_octant(c[None, :] + 0.9 * ring_rel, oid)
    return (fxy, np.asarray(foid), ixy, np.asarray(ioid), c, oid,
            int(c2[0]))


def _frame(ring, g):
    """Similarity chart-of-octant-g -> canonical for a host's ring: fit
    on the ring corners in g plus the 0.9-inset points (the insets
    break the 2-point chirality degeneracy for seam-overhang frames)."""
    fxy, foid, ixy, ioid = ring[0], ring[1], ring[2], ring[3]
    m, mi = foid == g, ioid == g
    if m.sum() + mi.sum() < 3:
        raise NotImplementedError(
            'octant shares too little of the host ring for a frame')
    src = np.vstack([fxy[m], ixy[mi]])
    tgt = np.concatenate([_TGT[m], 0.9 * _TGT[mi]])
    return _fit(src, tgt)


def _fit(src_xy, tgt):
    """Similarity (reflection allowed) src -> tgt; ((a, b, refl))."""
    s = src_xy[:, 0] + 1j * src_xy[:, 1]
    best = None
    for rf in (False, True):
        ss = np.conj(s) if rf else s
        M = np.column_stack([ss, np.ones(len(ss))])
        sol = np.linalg.lstsq(M, tgt, rcond=None)[0]
        r = np.max(np.abs(M @ sol - tgt))
        if best is None or r < best[0]:
            best = (r, (sol[0], sol[1], rf))
    assert best[0] < 1e-6, 'state ring is not similar to the canonical ring'
    return best[1]


def _fwd(fr, z):
    a, b, rf = fr
    return a * (np.conj(z) if rf else z) + b


def _inv(fr, w):
    a, b, rf = fr
    u = (w - b) / a
    return np.conj(u) if rf else u


def _host_layer(host):
    from hhg9.h9.uuid_address import _batch_int_to_nibbles
    nib = _batch_int_to_nibbles([host.int], n=32)[0]
    sent = np.flatnonzero(nib[:31] == 0x0F)
    return int(sent[0]) - 1 if len(sent) else 30


def _class_of(comp):
    """Direction class of the current piece from its diameter direction
    in the CANONICAL (state ring) frame. Ring edges i / i+3 run at
    (150 + 60 i) mod 180 in canonical coordinates; the piece's long
    side is parallel to exactly one pair. Class = i + 1 (1..3).
    Frame-independent, fold-free, suffix-local, depth-invariant."""
    e = comp[0] * (_Z0[0] - _Z0[3])
    th = math.degrees(math.atan2(e.imag, e.real)) % 180
    i = int(round((th - 150.0) / 60.0)) % 3
    assert abs((th - 150.0 - 60.0 * ((round((th - 150.0) / 60.0)))) ) < 1.0, \
        f'diameter direction {th} off the canonical lattice'
    return i + 1


# ---- the five-symbol enumeration (transport note §4d, canonical) ----
_AXIS_DIGIT = {0: 4, 1: 5, 2: 2, 3: 3}   # the sole gauge: axis -> digit
_BASES = None


def _bases():
    """base(o) = reversed(axis-digit of oid_nb[o][slot], slot 0..2):
    each edge class points at a neighbour octant, and the digit names
    that neighbour's octahedral axis. Derived from H9O alone;
    machine-verified unique mod gauge (experimental/e4h/e4h_canonical.py)."""
    global _BASES
    if _BASES is None:
        from hhg9.h9.constants import H9O
        anti = {}
        for o in range(8):
            c = tuple(-int(v) for v in H9O.oid_cmp[o])
            anti[o] = int(H9O.cmp_oid[c])
        ax = {o: min(o, anti[o]) for o in range(8)}
        _BASES = {o: tuple(_AXIS_DIGIT[ax[int(n)]]
                           for n in reversed(H9O.oid_nb[o]))
                  for o in range(8)}
    return _BASES


def _digit5(oid, c2, half, cls):
    """The globally matched digit (S1-S3 over the oid_nb bases): 1
    anchors at class (1 + half); class 3 carries q in both halves;
    (p, q, r) rotates right with the host's c2."""
    if cls == 1 + half:
        return 1
    b = _bases()[oid]
    if cls == 3:
        return b[(1 - c2) % 3]
    return b[(0 - c2) % 3] if half == 0 else b[(2 - c2) % 3]


# -------------------------------------------------------------------- API
def h9e_split(u):
    """(host_uuid, half, digits) from an E4H uuid; validates grammar."""
    from hhg9.h9.uuid_address import (_batch_int_to_nibbles,
                                      batch_nibbles_to_int)
    nib = _batch_int_to_nibbles([u.int], n=32)[0].copy()
    epos = np.flatnonzero(nib[:31] == E_MARK)
    if len(epos) != 1:
        raise ValueError('not an E4H address (need exactly one 0xE)')
    e = int(epos[0])
    half = int(nib[e + 1])
    if half > 1:
        raise ValueError(f'invalid half digit {half}')
    tail = nib[e + 2:31]
    stop = np.flatnonzero(tail == 0x0F)
    n = int(stop[0]) if len(stop) else len(tail)
    digits = [int(d) for d in tail[:n]]
    if any(d > 5 for d in digits) or np.any(tail[n:] != 0x0F):
        raise ValueError('invalid tail digits')
    nib[e:31] = 0x0F
    host = uuid_mod.UUID(int=int(batch_nibbles_to_int(nib[None, :])[0]))
    return host, half, digits


def h9e_label(u):
    """Label form <h9-label>E<half><digits>."""
    from hhg9.h9.uuid_address import h9_label
    host, half, digits = h9e_split(u)
    return (h9_label(host, with_tail=False) + 'E' + str(half)
            + ''.join(map(str, digits)))


def h9e_encode(lats, lons, layer=6, depth=2, reg=None):
    """Encode points to Hex9+E4H addresses: h9 to the attach layer,
    then `depth` aperture-4 class digits below the half cut."""
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

    rings, frames = {}, {}
    out = [None] * len(hosts)
    for i, (u, p, g) in enumerate(zip(hosts, P, O)):
        kf = (u.int, int(g))
        if kf not in frames:
            if u.int not in rings:
                rings[u.int] = _ring(u, reg, b_oct)
            frames[kf] = _frame(rings[u.int], int(g))
        fr = frames[kf]
        oid_h, c2_h = rings[u.int][5], rings[u.int][6]
        w = _fwd(fr, complex(p[0], p[1]))
        half, w = _classify(w, _HALVES)
        comp = _HALVES[half]                       # canonical -> canonical
        digits = []
        for _ in range(depth):
            k, w = _classify(w, _MAPS)
            a2, b2 = _MAPS[k]
            comp = (comp[0] * a2, comp[0] * b2 + comp[1])
            digits.append(0 if k == 0 else
                          _digit5(oid_h, c2_h, half, _class_of(comp)))
        nib = _batch_int_to_nibbles([u.int], n=32)[0].copy()
        nib[layer + 1] = E_MARK
        nib[layer + 2] = half
        for j, d in enumerate(digits):
            nib[layer + 3 + j] = d
        out[i] = uuid_mod.UUID(int=int(batch_nibbles_to_int(nib[None, :])[0]))
    return out


def h9e_decode(uuids, reg=None, _probe=_CEN):
    """Decode E4H addresses to representative lat/lon (leaf trapezoid
    centroids). Exact suffix-local fold; one h9_dec per unique host."""
    from hhg9 import Points, Registrar
    from hhg9.h9.polygon import fold_to_octant
    reg = reg or Registrar()
    g_gcd, b_oct = reg.domain('g_gcd'), reg.domain('b_oct')
    rings, frames = {}, {}
    lats, lons = np.zeros(len(uuids)), np.zeros(len(uuids))
    for i, u in enumerate(uuids):
        host, half, digits = h9e_split(u)
        if host.int not in rings:
            rings[host.int] = _ring(host, reg, b_oct)
            frames[host.int] = _frame(rings[host.int],
                                      rings[host.int][5])
        oid = rings[host.int][5]
        c2h = rings[host.int][6]
        fr = frames[host.int]
        comp = _HALVES[half]
        for d in digits:
            if d == 0:
                a2, b2 = _MAPS[0]
            else:
                want = [cl for cl in (1, 2, 3)
                        if _digit5(oid, c2h, half, cl) == d]
                if len(want) != 1:
                    raise ValueError(f'digit {d} invalid in this context')
                pick = None
                for k in (1, 2, 3):
                    a2, b2 = _MAPS[k]
                    t = (comp[0] * a2, comp[0] * b2 + comp[1])
                    if _class_of(t) == want[0]:
                        assert pick is None, 'ambiguous class'
                        pick = (a2, b2)
                assert pick is not None, f'class {want[0]} matches no child'
                a2, b2 = pick
            comp = (comp[0] * a2, comp[0] * b2 + comp[1])
        rep = _inv(fr, comp[0] * _probe + comp[1])
        xy, roid = fold_to_octant(np.array([[rep.real, rep.imag]]), oid)
        gp = reg.project(Points(xy, b_oct, oid=np.asarray(roid)),
                         [b_oct, g_gcd])
        lats[i], lons[i] = gp.coords[0, 0], gp.coords[0, 1]
    return lats, lons


def h9e_partner_point(uuids, reg=None):
    """Representative lat/lon of each leaf's PARTNER half — the mirror
    trapezoid across the leaf's long side, i.e. the other half of the
    same fine hexagon. Encoding the result gives an address that
    differs in half/prefix but shares the final class digit (the §4c
    matched-pairs property); for centre children (final digit 0) the
    partner is the host's other half."""
    return h9e_decode(uuids, reg, _probe=complex(-_CEN.real, _CEN.imag))
