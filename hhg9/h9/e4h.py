# Part of the Hex9 (H9) Project
# Copyright ©2026, Ben Griffin
# Licensed under the Apache License, Version 2.0

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

SYMBOLIC/STRUCTURAL (2026-08-05): no geometric probes remain — every
constant below derives from H9P/H9O tables at import/first use:
  * the state frame of a host is a CONSTANT similarity per
    (p_mo, c2) — the centred H9P.hx ring is exactly the canonical
    ring under conjugation (H9P.hx winds clockwise, the canonical
    ring counter-clockwise, so the reflection is structural) — scaled
    by 3^layer and translated by the host centre. No fitting, no
    chirality search, no inset points; one exact h9_dec per host.
  * points in a neighbouring octant are UNFOLDED into the host's
    frame by the exact inverse of fold_to_octant's per-seam isometry
    (residuals are taken in the host's unfolded frame — the local-net
    doctrine). Binned points never sit two seams from their host
    (census: experimental/e4h/e4h_symbolic.py).
  * class is an integer rotation accumulator: the rep-4 child maps
    rotate by (0, 120, 180, 240) degrees, the half-1 map by 180, and
    class = ((s - 1) mod 3) + 1 in 60-degree units s. No angles are
    measured; decode inverts digit→class→piece by the same integers.
Byte-parity vs the geometric PoC: 856-point global census (all
octants, equator seam, cone-point rings, poles) × four (layer, depth)
regimes — identical everywhere off cut lines; the only diffs are
knife-edge points ON a cut (piece margin ~1e-13, frames agree to
2.6e-13) where either side is a valid answer. Decode parity ≤4e-10,
symbolic roundtrip 100%.

Public API: h9e_encode, h9e_decode, h9e_label, h9e_split,
h9e_partner_point.
"""
import math
import uuid as uuid_mod

import numpy as np

E_MARK = 0xE
_TGT = np.exp(1j * (np.pi / 6 + np.pi / 3 * np.arange(6)))  # canonical ring


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


def _rotations():
    """Rotation (units of 60°) of each rep-4 child map: (0, 2, 3, 4),
    asserted against the canonical maps. Class transport is these
    integers — no angle is ever measured at runtime."""
    rot = []
    for a, _b in _MAPS:
        u = (math.degrees(math.atan2(a.imag, a.real)) % 360) / 60.0
        assert abs(u - round(u)) < 1e-9, 'child map rotation off-lattice'
        rot.append(int(round(u)) % 6)
    assert rot == [0, 2, 3, 4], f'unexpected child rotations {rot}'
    return tuple(rot)


_ROT = _rotations()
_ROT3_PIECE = {2: 1, 0: 2, 1: 3}   # rot mod 3 -> edge piece (distinct mod 3)


def _cls(s):
    """Direction class (1..3) of a composition whose accumulated
    rotation is s units of 60° (the half-1 map contributes 3)."""
    return ((s - 1) % 3) + 1


def _score(w, edges=_EDGES):
    return min(((e.conjugate() * (w - p)).imag for p, e in edges))


def _classify(w, cand):
    us = [(w - b) / a for a, b in cand]
    ss = [_score(u) for u in us]
    k = int(np.argmax(ss))
    return k, us[k]


# ------------------------------------------------------------------- state
_A = None


def _state_frames():
    """A[p_mo, c2]: the constant similarity taking the CENTRED H9P.hx
    ring (host chart, clockwise) onto the canonical ring _TGT (CCW)
    under conjugation — the reflection is structural, not searched.
    Verified on all 6 corners at first use."""
    global _A
    if _A is None:
        from hhg9.h9 import H9P
        A = np.zeros((2, 3), dtype=np.complex128)
        for mo in range(2):
            for c2 in range(3):
                hx = H9P.hx[mo, c2]
                rel = hx - hx.mean(axis=0)
                z = rel[:, 0] + 1j * rel[:, 1]
                a = _TGT[0] / np.conj(z[0])
                assert np.max(np.abs(a * np.conj(z) - _TGT)) < 1e-9, \
                    f'hx[{mo},{c2}] ring is not canonical under conj'
                A[mo, c2] = a
        _A = A
    return _A


_UNFOLD = None


def _unfolds():
    """UNFOLD[o, g]: 3x2 affine taking neighbour octant g's canonical
    frame into o's EXTENDED frame — the exact inverse of
    fold_to_octant's per-seam isometry (same corner-matching
    construction, solved backwards)."""
    global _UNFOLD
    if _UNFOLD is None:
        from hhg9.h9 import H9P
        from hhg9.h9.constants import H9O
        from hhg9.h9.polygon import SV_CORNER_AXIS
        tab = {}
        for o in range(8):
            mode = int(H9O.oid_mo[o])
            tri = H9P.sv[mode]
            ax = SV_CORNER_AXIS[mode]
            tri_b = H9P.sv[1 - mode]
            axis_b = SV_CORNER_AXIS[1 - mode]
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
                tab[o, o ^ (1 << ax[k])] = m
        _UNFOLD = tab
    return _UNFOLD


def _host_info(host, b_oct):
    """(centre, oid, c2, p_mo, layer) of a host: one exact h9_dec plus
    the tail-nibble state unpack. No ring corners, no folding."""
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
    """State frame of a host in its own octant chart:
    fr = (a, b, rf=True) with a = A[p_mo, c2]·3^layer, b = −a·conj(c)."""
    c, _oid, c2, p_mo, layer = info
    a = _state_frames()[p_mo, c2] * (3.0 ** layer)
    return a, -a * np.conj(complex(c[0], c[1])), True


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


def _digit_to_class(oid, c2, half):
    """Inverse of _digit5 for a context: {digit: class}. The three
    classes always carry three distinct digits (rule property)."""
    d2c = {_digit5(oid, c2, half, cl): cl for cl in (1, 2, 3)}
    assert len(d2c) == 3, 'digit rule lost injectivity'
    return d2c


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

    infos, frames = {}, {}
    out = [None] * len(hosts)
    for i, (u, p, g) in enumerate(zip(hosts, P, O)):
        if u.int not in infos:
            infos[u.int] = _host_info(u, b_oct)
            frames[u.int] = _host_frame(infos[u.int])
        _c, o, c2_h, _mo, _lay = infos[u.int]
        g = int(g)
        if g != o:
            m = _unfolds().get((o, g))
            if m is None:
                raise ValueError(
                    'point lies two seams from its host — unreachable '
                    'for binned points (census: e4h_symbolic.py)')
            p = np.array([p[0], p[1], 1.0]) @ m
        w = _fwd(frames[u.int], complex(p[0], p[1]))
        half, w = _classify(w, _HALVES)
        s = 3 * half
        digits = []
        for _ in range(depth):
            k, w = _classify(w, _MAPS)
            s += _ROT[k]
            digits.append(0 if k == 0 else _digit5(o, c2_h, half, _cls(s)))
        nib = _batch_int_to_nibbles([u.int], n=32)[0].copy()
        nib[layer + 1] = E_MARK
        nib[layer + 2] = half
        for j, d in enumerate(digits):
            nib[layer + 3 + j] = d
        out[i] = uuid_mod.UUID(int=int(batch_nibbles_to_int(nib[None, :])[0]))
    return out


def h9e_decode(uuids, reg=None, _probe=_CEN):
    """Decode E4H addresses to representative lat/lon (leaf trapezoid
    centroids). Exact suffix-local fold; one h9_dec per unique host;
    digit→piece inversion is pure integers."""
    from hhg9 import Points, Registrar
    from hhg9.h9.polygon import fold_to_octant
    reg = reg or Registrar()
    g_gcd, b_oct = reg.domain('g_gcd'), reg.domain('b_oct')
    infos, frames = {}, {}
    lats, lons = np.zeros(len(uuids)), np.zeros(len(uuids))
    for i, u in enumerate(uuids):
        host, half, digits = h9e_split(u)
        if host.int not in infos:
            infos[host.int] = _host_info(host, b_oct)
            frames[host.int] = _host_frame(infos[host.int])
        _c, oid, c2h, _mo, _lay = infos[host.int]
        fr = frames[host.int]
        comp = _HALVES[half]
        s = 3 * half
        d2c = _digit_to_class(oid, c2h, half)
        for d in digits:
            if d == 0:
                k = 0
            else:
                if d not in d2c:
                    raise ValueError(f'digit {d} invalid in this context')
                k = _ROT3_PIECE[(d2c[d] - s) % 3]
            a2, b2 = _MAPS[k]
            comp = (comp[0] * a2, comp[0] * b2 + comp[1])
            s += _ROT[k]
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
