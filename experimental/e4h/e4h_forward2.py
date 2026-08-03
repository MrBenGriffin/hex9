# Part of the Hex9 (H9) Project
# Copyright ©2026, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""Symbolic forward transcode v2: H9 (deep) -> r_adr/d_adr thread -> Hex9+E4H.

v1 (e4h_forward.py) is the geometric oracle: floats, a probe grid, and
`_mode0_interior_pts` at runtime. v2 replaces the runtime geometry with
address arithmetic (Ben, 2026-07-29: "x_adr_to_r_adr should not need a
probe at all — but might need some sort of LUT"):

  * the deep cell's digit/mode chain (`x_adr_to_d_adr`) is folded into an
    EXACT lattice position (Fractions, denominator 3^k) relative to its
    layer-A lineage-cut anchor (`d_adr_to_x_adr` of the truncated chain —
    itself pure address arithmetic);
  * the fold is driven by two small calibrated LUTs — per (digit, mode):
    the child-centre displacement in child-pitch axial units, and per leaf
    context: the mode-0 representative offset. Calibration runs ONCE
    against hhg9's own decodes (h9_dec / _mode0_interior_pts) and snaps to
    exact rationals; it is LUT *generation*, not a runtime probe. Deriving
    the same tables straight from H9R/HEX_LUTS is the follow-up.
  * the E4H side (dyadic descent, walk-up, mode-0 ownership) is the same
    integer machinery as v1, now on exact Fractions — t-grid combinatorics
    only, per the substrate correction (E4H follows the t_cell grid, not
    the host's d-cell cut).

One geometric step remains at runtime: naming a host that is NOT the
anchor cell (lattice offset δ ≠ 0) still goes through h9_bin_pts; the
symbolic neighbour hop (region_neighbours cascade) is v3.

Verification: byte-identical host uuid + digit string against the v1
oracle over the full sample, plus exact-vs-float position residuals.

Run:  python experimental/e4h_forward2.py
"""
from fractions import Fraction

import numpy as np

from e4h_forward import (A, B, BOX, L_DEEP, N_PTS, UNITS,
                         build_frame, fine_owner, walk_up)

K = L_DEEP - A


def hex_round_frac(fq, fr):
    """Nearest axial lattice point, exact Fraction cube-rounding."""
    x, z = fq, fr
    y = -x - z
    rx, ry, rz = round(x), round(y), round(z)
    dx, dy, dz = abs(rx - x), abs(ry - y), abs(rz - z)
    if dx > dy and dx > dz:
        rx = -ry - rz
    elif dy <= dz:
        rz = -rx - ry
    return int(rx), int(rz)


def snap_int(v, tol=1e-6):
    r = int(round(v))
    assert abs(v - r) < tol, f'not an integer lattice value: {v}'
    return r


def snap_frac(v, den, tol=1e-6):
    f = Fraction(round(v * den), den)
    assert abs(v - float(f)) < tol, f'not a /{den} rational: {v}'
    return f


def calibrate(reg, b_oct, full, c0, Tinv):
    """Build the two LUTs from hhg9's own decodes (one-off LUT generation).

    disp[key]: child-centre minus parent-centre, axial ints in CHILD-pitch
    units, for each digit/mode context on the A..L_DEEP ladder.
    rep[key]:  mode-0 representative minus cell centre, axial Fractions in
    own-pitch units, keyed by the leaf context.
    Keys start as (digit, mode) and are extended with the leaf rid if the
    short key turns out not to determine the value."""
    from hhg9.h9.uuid_address import (h9_bin, h9_dec, _mode0_interior_pts,
                                      _batch_int_to_nibbles)
    from hhg9.h9.addressing import x_adr_to_d_adr, x_adr_to_r_adr

    lat = lambda xy: (np.atleast_2d(xy) - c0) @ Tinv.T

    def contexts(uuids, level):
        nibs = _batch_int_to_nibbles([u.int for u in uuids], n=32)
        dm = x_adr_to_d_adr(nibs)
        _, r_adr = x_adr_to_r_adr(nibs)
        return (dm[:, level, 0].astype(int), dm[:, level, 1].astype(int),
                r_adr[:, 1 + level].astype(int))

    # displacement pairs MUST be lineage pairs (child vs its d_adr-truncation
    # parent), not ownership pairs from same-point binning: a split child
    # (digit 6/7/8) straddles two cells, so its layer-L owner depends on the
    # sample point and the "displacement" would be two-valued by a full
    # parent step. Lineage is what the fold walks.
    import uuid as uuid_mod
    from hhg9.h9.uuid_address import batch_nibbles_to_int
    from hhg9.h9.addressing import d_adr_to_x_adr
    disp_by = {}
    kids = list(dict.fromkeys(h9_bin(full, L_DEEP, reg)))
    kn = _batch_int_to_nibbles([u.int for u in kids], n=32)
    # all CONTEXT comes from the full deep chain (interior rid semantics —
    # the truncated rebuild normalises its leaf rid to the "3" terminal and
    # would key differently from the runtime fold); truncations are used
    # only for the decoded lineage geometry. A bare digit truncation lacks
    # the right tail, so parent/child are rebuilt through d_adr.
    dmk = x_adr_to_d_adr(kn)
    _, ra_full = x_adr_to_r_adr(kn)

    def to_uuid(bt, top):
        nib = np.full((bt.shape[0], 32), 0x0F, dtype=np.uint8)
        nib[:, :top + 1] = bt[:, :-1]
        nib[:, -1] = bt[:, -1]
        return [uuid_mod.UUID(int=v) for v in batch_nibbles_to_int(nib)]

    for L in range(A, L_DEEP):
        cu = to_uuid(d_adr_to_x_adr(dmk[:, :L + 2, :]), L + 1)
        pu = to_uuid(d_adr_to_x_adr(dmk[:, :L + 1, :]), L)
        cc = lat(h9_dec(cu, b_oct).coords[:, :2])
        pc = lat(h9_dec(pu, b_oct).coords[:, :2])
        scale = 3 ** (L + 1 - A)
        for i in range(len(kids)):
            # child-pitch units; thirds, not ints: hex9's a9 children are
            # 6-interior + 3-split, so the fine lattice is offset from the
            # coarse one (the reason split digits 6/7/8 exist at all)
            d = tuple(snap_frac(v, 12) for v in (cc[i] - pc[i]) * scale)
            key = (int(dmk[i, L + 1, 0]), int(dmk[i, L + 1, 1]),
                   int(ra_full[i, 2 + L]),
                   int(dmk[i, L, 1]), int(ra_full[i, 1 + L]))
            disp_by.setdefault(key, set()).add(d)

    rep_by = {}
    deep = kids
    ctr = lat(h9_dec(deep, b_oct).coords[:, :2])
    rp = lat(_mode0_interior_pts(deep, b_oct).coords[:, :2])
    dg, mo, rid = contexts(deep, L_DEEP)
    for i in range(len(deep)):
        r = tuple(snap_frac(v, 60) for v in (rp[i] - ctr[i]) * 3 ** K)
        rep_by.setdefault((dg[i], mo[i], rid[i]), set()).add(r)

    def reduce_key(by, name):
        for keyfn, label in ((lambda k: k[:2], '(digit, mode)'),
                             (lambda k: k[:3], '(digit, mode, rid)'),
                             (lambda k: k, '(digit, mode, rid, p_mo, p_rid)')):
            lut = {}
            for k, vs in by.items():
                lut.setdefault(keyfn(k), set()).update(vs)
            if all(len(v) == 1 for v in lut.values()):
                print(f'calib   : {name} LUT keyed by {label}, '
                      f'{len(lut)} entries')
                return {k: next(iter(v)) for k, v in lut.items()}, keyfn
        raise AssertionError(f'{name} LUT: rid key insufficient — '
                             'needs more context')

    disp, disp_key = reduce_key(disp_by, 'displacement')
    rep, rep_key = reduce_key(rep_by, 'representative')
    return disp, disp_key, rep, rep_key


def symbolic_position(nibs_row, disp, disp_key, rep, rep_key):
    """Digit/mode chain -> exact axial position (layer-A pitch units)
    relative to the layer-A lineage-cut anchor centre. Pure arithmetic."""
    from hhg9.h9.addressing import x_adr_to_d_adr, x_adr_to_r_adr
    dm = x_adr_to_d_adr(nibs_row[None, :])[0]
    _, r_adr = x_adr_to_r_adr(nibs_row[None, :])
    p = [Fraction(0), Fraction(0)]
    for i in range(1, K + 1):
        key = disp_key((int(dm[A + i, 0]), int(dm[A + i, 1]),
                        int(r_adr[0, 1 + A + i]),
                        int(dm[A + i - 1, 1]), int(r_adr[0, A + i])))
        d = disp[key]
        p[0] += Fraction(d[0], 3 ** i)
        p[1] += Fraction(d[1], 3 ** i)
    key = rep_key((int(dm[A + K, 0]), int(dm[A + K, 1]),
                   int(r_adr[0, 1 + A + K])))
    r = rep[key]
    p[0] += r[0] / 3 ** K
    p[1] += r[1] / 3 ** K
    return p, dm


def anchor_uuid(dm):
    """Layer-A lineage-cut cell of one d_adr chain — pure address space."""
    import uuid as uuid_mod
    from hhg9.h9.addressing import d_adr_to_x_adr
    from hhg9.h9.uuid_address import batch_nibbles_to_int
    body_tail = d_adr_to_x_adr(dm[None, :A + 1, :])[0]
    nib = np.full((1, 32), 0x0F, dtype=np.uint8)
    nib[0, :A + 1] = body_tail[:-1]
    nib[0, -1] = body_tail[-1]
    return uuid_mod.UUID(int=batch_nibbles_to_int(nib)[0])


def main():
    from hhg9 import Registrar, Points
    from hhg9.h9.uuid_address import (h9_encode, h9_bin, h9_bin_pts, h9_dec,
                                      h9_label, _mode0_interior_pts,
                                      _batch_int_to_nibbles)

    reg = Registrar()
    b_oct = reg.domain('b_oct')
    rng = np.random.default_rng(9)
    lats = rng.uniform(BOX[0], BOX[1], N_PTS)
    lons = rng.uniform(BOX[2], BOX[3], N_PTS)
    full = h9_encode(lats, lons, reg)
    deep = list(dict.fromkeys(h9_bin(full, L_DEEP, reg)))
    c0, T, Tinv = build_frame(reg, b_oct)
    disp, disp_key, rep, rep_key = calibrate(reg, b_oct, full, c0, Tinv)

    # ---- v1 oracle (geometric): fine owner from the float mode-0 point
    pts = _mode0_interior_pts(deep, b_oct)
    v1_fines = [fine_owner(p, c0, Tinv) for p in pts.coords[:, :2]]
    v1_ups = [walk_up(f) for f in v1_fines]
    v1_hosts = h9_bin_pts(Points(
        np.array([c0 + T @ np.array(h, float) for h, _ in v1_ups]),
        domain=b_oct, oid=pts.oid), A)

    # ---- v2 symbolic: thread fold -> exact position -> descent -> walk-up
    nibs = _batch_int_to_nibbles([u.int for u in deep], n=32)
    owners, results, max_res = [], [], 0.0
    anchor_lat = {}
    for i, row in enumerate(nibs):
        p, dm = symbolic_position(row, disp, disp_key, rep, rep_key)
        anc = anchor_uuid(dm)
        if anc not in anchor_lat:            # one decode per unique anchor:
            axy = h9_dec([anc], b_oct).coords[0, :2]     # v2's one remaining
            al = (axy - c0) @ Tinv.T                     # geometric assist
            anchor_lat[anc] = (snap_int(al[0]), snap_int(al[1]))
        av = anchor_lat[anc]
        px, py = p[0] + av[0], p[1] + av[1]
        fine = hex_round_frac(px * (1 << B), py * (1 << B))
        owners.append(hex_round_frac(px, py))            # layer-A point owner
        v1_lat = (pts.coords[i, :2] - c0) @ Tinv.T
        max_res = max(max_res, abs(float(px) - v1_lat[0]),
                      abs(float(py) - v1_lat[1]))
        results.append(walk_up(fine))

    # host naming — symbolic: h9_cell_ancestor (pure address space) names
    # the hexagon at each deep cell's point-owner position, giving a
    # position -> canonical-uuid table; hosts are looked up by lattice
    # position. Geometric fallback only for positions no sample point owns.
    from hhg9.h9.uuid_address import h9_cell_ancestor
    import uuid as uuid_mod
    canon = h9_cell_ancestor(deep, A)
    table = {}
    for q, u in zip(owners, canon):
        prev = table.setdefault(q, u.int)
        assert prev == u.int, 'canonical-name table inconsistent'
    host_uuids, miss = [], []
    for i, (hij, _) in enumerate(results):
        v = table.get(hij)
        host_uuids.append(uuid_mod.UUID(int=v) if v is not None else None)
        if v is None:
            miss.append(i)
    if miss:
        ctrs = np.array([c0 + T @ np.array(results[i][0], float) for i in miss])
        named = h9_bin_pts(Points(ctrs, domain=b_oct,
                                  oid=np.full(len(miss), pts.oid[0])), A)
        for i, u in zip(miss, named):
            host_uuids[i] = u
    n = len(deep)
    print(f'position: exact thread-fold vs float oracle, max residual '
          f'{max_res:.2e} (layer-A pitch units)')
    print(f'naming  : {len(table)} hexagons named symbolically; '
          f'{n - len(miss)}/{n} hosts resolved from the table '
          f'({100 * (n - len(miss)) / n:.1f}%), {len(miss)} geometric fallback')

    match = sum(u.int == v.int and d == vd
                for u, (_, d), v, (_, vd)
                in zip(host_uuids, results, v1_hosts, v1_ups))
    ok = match == n
    ex = f'{h9_label(host_uuids[0], with_tail=False)}E{results[0][1]}'
    print(f'oracle  : v2 == v1 (host uuid + digits) for {match}/{n} '
          f'e.g. {ex}   {"PASS" if ok else "FAIL"}')
    return ok


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
