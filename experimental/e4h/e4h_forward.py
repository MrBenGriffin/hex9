# Part of the Hex9 (H9) Project
# Copyright ©2026, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""Forward transcode spike: H9 (deep) -> d_cells -> Hex9+E4H.

Route (Ben, 2026-07-29): the d-cell layer is the substrate — a deep H9
cell's canonical representative is its mode-0 d_cell interior point
(hhg9's own `_mode0_interior_pts`), and the E4H tail is recovered by
walking the aperture-4 ownership tree bottom-up from the fine owner of
that point (leaf-only reification, as with h9_cell_ancestor's deep
re-bin doctrine). This spike is the geometric-oracle harness: floats
with lattice snapping, one octant, planar b_oct frame. The fully
symbolic implementation (d_adr digits all the way down, no coordinates)
must reproduce it byte-for-byte, the same methodology by which
h9_cell_ancestor was verified against its geometric oracle.

Provisional conventions, to be pinned by the d-cell layer (H9O.oid_mo):
  * owned edge-child directions = even direction-index in the host frame
    (P0 = 0). The transport_check.py mode-alternation theorem guarantees
    any such parity choice IS the mode-0 ownership rule geometrically.
  * a4 digit alphabet: 0 = centre child, 1..3 = owned edge children in
    direction order (digit = 1 + t//2).
  * label form: <h9-label>E<digits>; uuid nibble packing (0xE break,
    tail interplay) deferred to the grammar spec.

Checks:
  1. FOLD     — walk-up(fine) then fold-down(digits) returns the same
                fine cell for every sample (address is self-consistent).
  2. COUNTS   — every interior host owns exactly 4^B fine descendants
                (the E4H count law; cf. transport_check check_counts).
  3. VERBS    — E4H walk-up host vs h9_cell_ancestor(deep, A) vs the
                d_adr lineage cut (x_adr_to_d_adr truncation): agreement
                rates, the lineage-vs-ownership signature in numbers.

Run:  python experimental/e4h_forward.py
"""
import numpy as np

A = 6            # host layer (E4H attach level)
L_DEEP = 9       # source H9 layer (3 levels below the hosts)
B = 2            # E4H tail depth (a4 digits)
P0 = 0           # provisional mode-0 direction parity (see docstring)
N_PTS = 1200
BOX = (19.5, 20.5, 34.5, 35.5)          # lat0, lat1, lon0, lon1 — octant interior
UNITS = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]   # axial, 60° order


def build_frame(reg, b_oct):
    """Host-centre lattice frame from hhg9's own decoded centres, using a
    dense probe grid around the box centre so the full neighbour ring of
    the origin host is present: origin at the most central host, basis
    e1/e2 = adjacent neighbour offsets."""
    from hhg9.h9.uuid_address import h9_encode, h9_bin, h9_dec
    la0, lo0 = (BOX[0] + BOX[1]) / 2, (BOX[2] + BOX[3]) / 2
    g = np.linspace(-0.2, 0.2, 40)
    glat, glon = np.meshgrid(la0 + g, lo0 + g)
    probe = h9_bin(h9_encode(glat.ravel(), glon.ravel(), reg), A, reg)
    uniq = list(dict.fromkeys(probe))
    ctrs = h9_dec(uniq, b_oct).coords[:, :2]
    mid = ctrs.mean(axis=0)
    c0 = ctrs[np.argmin(np.linalg.norm(ctrs - mid, axis=1))]
    d = np.linalg.norm(ctrs - c0, axis=1)
    pitch = d[d > 1e-12].min()
    nb = ctrs[(d > 1e-12) & (d < 1.5 * pitch)] - c0
    assert len(nb) == 6, f'central host has {len(nb)} neighbours, expected 6'
    nb = nb[np.argsort(np.arctan2(nb[:, 1], nb[:, 0]))]
    e1, e2 = nb[0], nb[1]
    T = np.column_stack([e1, e2])
    Tinv = np.linalg.inv(T)
    lat = (ctrs - c0) @ Tinv.T
    assert np.allclose(lat, np.round(lat), atol=1e-6), \
        'host centres are not on the e1/e2 lattice'
    return c0, T, Tinv


def hex_round(fq, fr):
    """Nearest axial lattice point (cube rounding)."""
    x, z = fq, fr
    y = -x - z
    rx, ry, rz = np.round(x), np.round(y), np.round(z)
    dx, dy, dz = abs(rx - x), abs(ry - y), abs(rz - z)
    if dx > dy and dx > dz:
        rx = -ry - rz
    elif dy <= dz:
        rz = -rx - ry
    return int(rx), int(rz)


def fine_owner(p, c0, Tinv):
    """Owning depth-B fine cell of point p, as integer axial coords ×2^B."""
    a, b = (p - c0) @ Tinv.T * (1 << B)
    return hex_round(a, b)


def walk_up(ij):
    """Fine cell (axial ints at depth B) -> (host axial ints, digit string).
    Centre child iff both coords even; else exactly one ± unit-vector pair
    reaches the coarser lattice and the P0-parity side owns (mode-0 rule)."""
    i, j = ij
    digits = []
    for _ in range(B):
        if i % 2 == 0 and j % 2 == 0:
            digits.append(0)
        else:
            cand = [t for t, (u, v) in enumerate(UNITS)
                    if (i + u) % 2 == 0 and (j + v) % 2 == 0]
            assert len(cand) == 2 and (cand[1] - cand[0]) == 3
            t = cand[0] if cand[0] % 2 == P0 else cand[1]
            digits.append(1 + t // 2)
            i, j = i + UNITS[t][0], j + UNITS[t][1]
        i, j = i // 2, j // 2
    return (i, j), ''.join(str(d) for d in reversed(digits))


def fold_down(host_ij, digits):
    """Inverse of walk_up: host axial ints + digit string -> fine ints."""
    i, j = host_ij
    for ch in digits:
        i, j = 2 * i, 2 * j
        d = int(ch)
        if d:
            t = 2 * (d - 1) + P0
            i, j = i - UNITS[t][0], j - UNITS[t][1]
    return i, j


def main():
    from hhg9 import Registrar, Points
    from hhg9.h9.uuid_address import (h9_encode, h9_bin, h9_bin_pts, h9_label,
                                      h9_cell_ancestor, _mode0_interior_pts,
                                      _batch_int_to_nibbles)
    from hhg9.h9.addressing import x_adr_to_d_adr

    reg = Registrar()
    b_oct = reg.domain('b_oct')
    rng = np.random.default_rng(9)
    lats = rng.uniform(BOX[0], BOX[1], N_PTS)
    lons = rng.uniform(BOX[2], BOX[3], N_PTS)
    deep = list(dict.fromkeys(h9_bin(h9_encode(lats, lons, reg), L_DEEP, reg)))
    pts = _mode0_interior_pts(deep, b_oct)
    assert len(np.unique(pts.oid)) == 1, 'sample must stay inside one octant'
    c0, T, Tinv = build_frame(reg, b_oct)

    # --- forward transcode: deep cell -> mode-0 point -> fine owner -> walk up
    fines = [fine_owner(p, c0, Tinv) for p in pts.coords[:, :2]]
    ups = [walk_up(f) for f in fines]
    host_ctrs = np.array([c0 + T @ np.array(h, float) for h, _ in ups])
    host_uuids = h9_bin_pts(Points(host_ctrs, domain=b_oct, oid=pts.oid), A)
    labels = [f'{h9_label(u, with_tail=False)}E{d}'
              for u, (_, d) in zip(host_uuids, ups)]
    print(f'transcoded {len(deep)} deep L{L_DEEP} cells -> Hex9+E4H '
          f'(A={A}, B={B}); e.g. {labels[0]}')

    # --- check 1: FOLD round trip
    ok = all(fold_down(h, d) == f for (h, d), f in zip(ups, fines))
    print(f'fold    : walk-up∘fold-down identity on {len(fines)} cells   '
          f'{"PASS" if ok else "FAIL"}')

    # --- check 2: COUNTS — enumerate a patch of fine cells, expect 4^B per host
    R = 6 * (1 << B)
    patch = [(i, j) for i in range(-R, R + 1) for j in range(-R, R + 1)
             if max(abs(i), abs(j), abs(i + j)) <= R]
    owned = {}
    for f in patch:
        h, _ = walk_up(f)
        owned.setdefault(h, []).append(f)
    interior = [h for h in owned
                if max(abs(h[0]), abs(h[1]), abs(h[0] + h[1])) <= R // (1 << B) - 2]
    bad = [h for h in interior if len(owned[h]) != 4 ** B]
    print(f'counts  : {len(interior)} interior hosts own '
          f'{"all exactly" if not bad else "NOT all"} 4^{B} = {4 ** B}   '
          f'{"PASS" if not bad else "FAIL"}')

    # --- check 3: VERBS — ownership host vs canonical ancestor vs lineage cut
    anc = h9_cell_ancestor(deep, A)
    own_agree = sum(u.int == a.int for u, a in zip(host_uuids, anc))
    nibs = _batch_int_to_nibbles([u.int for u in deep], n=32)
    d_adr = x_adr_to_d_adr(nibs)
    anc_nibs = _batch_int_to_nibbles([u.int for u in anc], n=32)
    d_anc = x_adr_to_d_adr(anc_nibs)
    cut_agree = int(np.sum(np.all(
        d_adr[:, :A + 1, 0] == d_anc[:, :A + 1, 0], axis=1)))
    n = len(deep)
    print(f'verbs   : E4H owner == h9_cell_ancestor for {own_agree}/{n} '
          f'({100 * own_agree / n:.1f}%); d_adr lineage cut == ancestor digits '
          f'for {cut_agree}/{n} ({100 * cut_agree / n:.1f}%)')
    return ok and not bad


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
