# Part of the Hex9 (H9) Project
# Copyright ©2026, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""Address-space forward transcode v3: H9 (deep) -> Hex9+E4H, no runtime
geometry.

Differences from v2 (e4h_forward2.py):

  * LUT GENERATION, not sample calibration: the displacement and
    representative tables are harvested from every cell of whole layers
    (L3 complete over a generous single-octant region, L4 as closure
    evidence), with contexts read mid-chain exactly as the runtime fold
    reads them. A closure check reports whether deeper layers add entries
    — the generated tables are complete-by-closure. (Deriving them
    algebraically from H9R/HEX_LUTS is the remaining purity step; the
    gen_fold_tables.py precedent applies.)
  * FULLY RELATIVE runtime: the fold position, dyadic descent and walk-up
    all run relative to the layer-A lineage anchor. This is sound because
    the walk-up is translation-invariant under even translations
    (absolute - relative = av * 2^j at depth j, even at every parity
    test), so the anchor's absolute lattice position is never needed —
    the v2 per-anchor h9_dec is gone.
  * LAYERED HOST NAMING with a negative result worth its weight: cross-
    cell naming is NOT suffix-local. Every walk-valid suffix extension of
    an anchor's chain stays inside the anchor's thread (mode-0 rows), and
    forcing mode-1 rows synthesises chains that are not addresses at all
    (bin(dec(u)) != u — machine-checked), so no split-child/BFS trick can
    name a neighbour from the anchor's digits alone. Real cross-cell
    addresses differ in the PREFIX: neighbour naming requires digit-carry
    up the chain — precisely the hex_step odometer redesign, now known to
    be the one irreducible missing primitive. Until it lands, naming is
    layered and counted: (1) a pure (anchor, δ) map taught by
    h9_cell_ancestor, (2) an absolute-position table costing one h9_dec
    per unique anchor, (3) a geometric h9_bin_pts fallback.

Verification: byte-identical host uuid + digit string against the v1
geometric oracle over the full sample.

Run:  python experimental/e4h_forward3.py [--viz out.png]
"""
import argparse
from fractions import Fraction

import numpy as np

from e4h_forward import (A, B, BOX, L_DEEP, N_PTS, build_frame, fine_owner,
                         walk_up)
from e4h_forward2 import (anchor_uuid, hex_round_frac, snap_frac,
                          symbolic_position)

K = L_DEEP - A
GENBOX = (6.0, 39.0, 16.0, 59.0)        # generous single-octant harvest region
GEN_N = 380                              # harvest grid is GEN_N x GEN_N


def gen_tables(reg, b_oct, c0, Tinv):
    """Harvest the fold LUTs from whole layers; closure-checked.

    For every enumerated cell (unique layer-L bins of a dense grid, chain
    confined to one octant), each chain step level i contributes one
    displacement entry keyed (digit, mode, rid) with the rid read from
    the FULL chain (interior semantics below the leaf, the normalised "3"
    terminal at it — both variants appear, exactly as the runtime fold
    queries them), and the leaf contributes one representative entry."""
    import uuid as uuid_mod
    from hhg9.h9.uuid_address import (h9_encode, h9_bin, h9_dec,
                                      _mode0_interior_pts,
                                      _batch_int_to_nibbles,
                                      batch_nibbles_to_int)
    from hhg9.h9.addressing import x_adr_to_d_adr, x_adr_to_r_adr, d_adr_to_x_adr

    lat = lambda xy: (np.atleast_2d(xy) - c0) @ Tinv.T
    g1 = np.linspace(GENBOX[0], GENBOX[1], GEN_N)
    g2 = np.linspace(GENBOX[2], GENBOX[3], GEN_N)
    gl, gn = np.meshgrid(g1, g2)
    full = h9_encode(gl.ravel(), gn.ravel(), reg)
    oid0 = h9_dec(h9_encode([(BOX[0] + BOX[1]) / 2], [(BOX[2] + BOX[3]) / 2],
                            reg), b_oct).oid[0]

    def to_uuid(bt, top):
        nib = np.full((bt.shape[0], 32), 0x0F, dtype=np.uint8)
        nib[:, :top + 1] = bt[:, :-1]
        nib[:, -1] = bt[:, -1]
        return [uuid_mod.UUID(int=v) for v in batch_nibbles_to_int(nib)]

    def harvest(cells, L):
        kn = _batch_int_to_nibbles([u.int for u in cells], n=32)
        dmk = x_adr_to_d_adr(kn)
        _, ra = x_adr_to_r_adr(kn)
        ctr, keep = {}, np.ones(len(cells), bool)
        for i in range(1, L + 1):                 # chain confined to octant;
            cu = to_uuid(d_adr_to_x_adr(dmk[:, :i + 1, :]), i)   # L0 skipped —
            pts = h9_dec(cu, b_oct)               # the runtime fold starts at A
            keep &= pts.oid == oid0
            ctr[i] = lat(pts.coords[:, :2])
        db, rb = {}, {}
        rp = lat(_mode0_interior_pts(cells, b_oct).coords[:, :2])
        for i in np.flatnonzero(keep):
            for lv in range(2, L + 1):
                # child-pitch units in the layer-A frame: 3^(lv - A), NOT
                # 3^lv — the table value is the universal per-step offset
                d = tuple(snap_frac(v, 12)
                          for v in (ctr[lv][i] - ctr[lv - 1][i]) * 3.0 ** (lv - A))
                key = (int(dmk[i, lv, 0]), int(dmk[i, lv, 1]),
                       int(ra[i, 1 + lv]))
                db.setdefault(key, set()).add(d)
            r = tuple(snap_frac(v, 60)
                      for v in (rp[i] - ctr[L][i]) * 3.0 ** (L - A))
            rb.setdefault((int(dmk[i, L, 0]), int(dmk[i, L, 1]),
                           int(ra[i, 1 + L])), set()).add(r)
        return db, rb, int(keep.sum())

    d3, r3, n3 = harvest(list(dict.fromkeys(h9_bin(full, 3, reg))), 3)
    d4, r4, n4 = harvest(list(dict.fromkeys(h9_bin(full[::3], 4, reg))), 4)
    d5, r5, n5 = harvest(list(dict.fromkeys(h9_bin(full[::9], 5, reg))), 5)

    def merge(parts, name):
        out = {}
        for by in parts:
            for k, vs in by.items():
                assert len(vs) == 1, f'{name} LUT conflicted at {k}'
                v = next(iter(vs))
                assert out.setdefault(k, v) == v, f'{name} closure broken: {k}'
        return out

    disp = merge((d3, d4, d5), 'displacement')
    rep = merge((r3, r4, r5), 'representative')
    a4 = len(set(d4) - set(d3))
    a5 = len(set(d5) - set(d3) - set(d4))
    a5r = len(set(r5) - set(r3) - set(r4))
    print(f'gen     : displacement {len(disp)} entries, representative '
          f'{len(rep)} entries ({n3} L3 + {n4} L4 + {n5} L5 cells); '
          f'closure: L4 adds {a4} disp keys, L5 adds {a5} disp / {a5r} rep '
          f'{"— CLOSED" if a5 == 0 and a5r == 0 else "— NOT closed"}')
    key3 = lambda k: k[:3]
    return disp, key3, rep, key3


def transcode(nibs_row, disp, dk, rep, rk):
    """One deep cell -> (anchor uuid, owner offset, host offset, digits).
    Pure address arithmetic + exact Fractions; nothing absolute."""
    p, dm = symbolic_position(nibs_row, disp, dk, rep, rk)
    anc = anchor_uuid(dm)
    d_own = hex_round_frac(p[0], p[1])
    fine = hex_round_frac(p[0] * (1 << B), p[1] * (1 << B))
    host_ij, digits = walk_up(fine)
    return anc, d_own, host_ij, digits, p


def main(viz_path=None):
    from hhg9 import Registrar, Points
    from hhg9.h9.uuid_address import (h9_encode, h9_bin, h9_bin_pts, h9_dec,
                                      h9_label, h9_cell_ancestor,
                                      _mode0_interior_pts,
                                      _batch_int_to_nibbles)
    import uuid as uuid_mod

    reg = Registrar()
    b_oct = reg.domain('b_oct')
    rng = np.random.default_rng(9)
    full = h9_encode(rng.uniform(BOX[0], BOX[1], N_PTS),
                     rng.uniform(BOX[2], BOX[3], N_PTS), reg)
    deep = list(dict.fromkeys(h9_bin(full, L_DEEP, reg)))
    c0, T, Tinv = build_frame(reg, b_oct)          # generator + oracle only
    disp, dk, rep, rk = gen_tables(reg, b_oct, c0, Tinv)

    # ---- v3 runtime: fully relative, no decodes in the transcode itself
    nibs = _batch_int_to_nibbles([u.int for u in deep], n=32)
    rows = [transcode(r, disp, dk, rep, rk) for r in nibs]

    # ---- host naming, layered (see docstring — cross-cell naming is NOT
    # suffix-local; full closure = the odometer carry primitive):
    #   1. pure (anchor, δ) map taught by h9_cell_ancestor (address space)
    #   2. absolute-position table (one h9_dec per unique anchor — counted)
    #   3. geometric h9_bin_pts fallback — counted
    canon = h9_cell_ancestor(deep, A)
    names_rel, names_abs, anc_av = {}, {}, {}
    from e4h_forward2 import snap_int
    for (anc, _, _, _, _) in rows:
        if anc.int not in anc_av:
            al = (h9_dec([anc], b_oct).coords[0, :2] - c0) @ Tinv.T
            anc_av[anc.int] = (snap_int(al[0]), snap_int(al[1]))
    for (anc, d_own, _, _, _), u in zip(rows, canon):
        prev = names_rel.setdefault((anc.int, d_own), u.int)
        assert prev == u.int, 'relative naming map inconsistent'
        av = anc_av[anc.int]
        prev = names_abs.setdefault((d_own[0] + av[0], d_own[1] + av[1]), u.int)
        assert prev == u.int, 'absolute naming table inconsistent'
    host_uuids, n_rel, n_abs, geo_i = [], 0, 0, []
    for i, (anc, _, hij, _, _) in enumerate(rows):
        v = names_rel.get((anc.int, hij))
        if v is not None:
            n_rel += 1
        else:
            av = anc_av[anc.int]
            v = names_abs.get((hij[0] + av[0], hij[1] + av[1]))
            if v is not None:
                n_abs += 1
            else:
                geo_i.append(i)
        host_uuids.append(uuid_mod.UUID(int=v) if v is not None else None)

    # ---- v1 oracle (geometric) for byte-comparison
    pts = _mode0_interior_pts(deep, b_oct)
    v1_ups = [walk_up(fine_owner(p, c0, Tinv)) for p in pts.coords[:, :2]]
    v1_hosts = h9_bin_pts(Points(
        np.array([c0 + T @ np.array(h, float) for h, _ in v1_ups]),
        domain=b_oct, oid=pts.oid), A)
    if geo_i:
        ctrs = np.array([c0 + T @ (np.array(rows[i][2], float)
                                   + np.array(anc_av[rows[i][0].int], float))
                         for i in geo_i])
        for i, u in zip(geo_i, h9_bin_pts(Points(
                ctrs, domain=b_oct, oid=np.full(len(geo_i), pts.oid[0])), A)):
            host_uuids[i] = u
    n = len(deep)
    print(f'naming  : {n_rel} pure (anchor, δ) / {n_abs} absolute table '
          f'({len(anc_av)} anchor decodes) / {len(geo_i)} geometric fallback '
          f'— full symbolic closure = odometer carry (see docstring)')
    match = sum(u is not None and u.int == v.int and r[3] == vd
                for u, r, v, (_, vd) in zip(host_uuids, rows, v1_hosts, v1_ups))
    ok = match == n
    i0 = next((i for i, r in enumerate(rows) if r[2] != (0, 0)), 0)
    ex = f'{h9_label(host_uuids[i0], with_tail=False)}E{rows[i0][3]}'
    print(f'oracle  : v3 == v1 (host uuid + digits) for {match}/{n} '
          f'e.g. {ex}   {"PASS" if ok else "FAIL"}')

    if viz_path:
        visualise(viz_path, reg, b_oct, c0, T, deep[i0], rows[i0],
                  host_uuids[i0], pts.coords[i0, :2])
        print(f'viz     : {viz_path}')
    return ok


def visualise(path, reg, b_oct, c0, T, deep_u, row, host_u, rep_xy):
    """Two panels. Left: the transcode in its layer-A neighbourhood — the
    depth-1 (a4) grid is the backdrop, depth-2 only inside the anchor and
    host. Right: leaf detail — the L9 cell among the adjacent depth-2 E4H
    cells, each labelled with its own digit suffix."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon as MP
    from hhg9.h9.uuid_address import (h9_dec, h9_label,
                                      _batch_int_to_nibbles,
                                      batch_nibbles_to_int)
    from hhg9.h9.addressing import x_adr_to_d_adr, d_adr_to_x_adr
    from e4h_forward import fold_down, UNITS, P0
    import uuid as uuid_mod

    BLUE, GOLD, RED = '#004488', '#DDAA33', '#BB5566'
    anc, d_own, host_ij, digits, p = row
    e1, e2 = T[:, 0], T[:, 1]
    pitch = np.linalg.norm(e1)
    hexv = lambda c, s: [np.asarray(c) + s * v for v in
                         [(e1 + e2) / 3, (2 * e2 - e1) / 3, (e2 - 2 * e1) / 3,
                          -(e1 + e2) / 3, (e1 - 2 * e2) / 3, (2 * e1 - e2) / 3]]

    nib = _batch_int_to_nibbles([deep_u.int], n=32)
    dm = x_adr_to_d_adr(nib)
    chain = []
    for lv in range(A, L_DEEP + 1):                # lineage cells A..deep
        bt = d_adr_to_x_adr(dm[:, :lv + 1, :])
        nb = np.full((1, 32), 0x0F, dtype=np.uint8)
        nb[0, :lv + 1] = bt[0, :-1]
        nb[0, -1] = bt[0, -1]
        u = uuid_mod.UUID(int=batch_nibbles_to_int(nb)[0])
        chain.append((lv, h9_dec([u], b_oct).coords[0, :2]))
    av = chain[0][1]
    R = 1 << B
    fine_ij = fold_down(host_ij, digits)
    fc = av + T @ (np.array(fine_ij, float) / R)
    hcc = av + T @ np.array(host_ij, float)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(16.5, 8.8),
                                   width_ratios=[1.08, 1])

    # ---------------- left: layer-A neighbourhood ----------------
    ring7 = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    for dij in ring7:
        c = av + T @ np.array(dij, float)
        axL.add_patch(MP(hexv(c, 1.0), closed=True, fc='none', ec='0.5',
                         lw=1.4))
    for i in range(-4, 5):                         # backdrop: depth-1 (a4)
        for j in range(-4, 5):
            if max(abs(i), abs(j), abs(i + j)) <= 4:
                c = av + T @ (np.array((i, j), float) / 2)
                axL.add_patch(MP(hexv(c, 0.5), closed=True, fc='none',
                                 ec='0.75', lw=0.7, zorder=0))
    for base in (np.array((0, 0)), np.array(host_ij)):
        for i in range(-R, R + 1):                 # depth-2 only in anchor+host
            for j in range(-R, R + 1):
                c = av + T @ (base + np.array((i, j), float) / R)
                if max(abs(i), abs(j), abs(i + j)) <= R + 1:
                    axL.add_patch(MP(hexv(c, 1 / R), closed=True, fc='none',
                                     ec='0.88', lw=0.45, zorder=0))
    axL.add_patch(MP(hexv(av, 1.0), closed=True, fc='none', ec=BLUE,
                     lw=1.8, ls='--'))
    axL.annotate('anchor (lineage cut, L6)', av - 0.95 * e2, color=BLUE,
                 fontsize=10, ha='center')
    axL.add_patch(MP(hexv(hcc, 1.0), closed=True, fc=BLUE, alpha=0.12,
                     ec=BLUE, lw=2.0))
    axL.annotate('host (E4H owner)', hcc + 0.55 * e2, color=BLUE,
                 fontsize=10, ha='center')

    up = [fc]                                      # E4H walk-up chain
    fi, fj = fine_ij
    for lv in range(B, 0, -1):
        if fi % 2 or fj % 2:
            cand = [t for t, (u, v) in enumerate(UNITS)
                    if (fi + u) % 2 == 0 and (fj + v) % 2 == 0]
            t = cand[0] if cand[0] % 2 == P0 else cand[1]
            fi, fj = fi + UNITS[t][0], fj + UNITS[t][1]
        fi, fj = fi // 2, fj // 2
        up.append(av + T @ (np.array((fi, fj), float) / (1 << (lv - 1))))
    axL.add_patch(MP(hexv(fc, 1 / R), closed=True, fc=GOLD, alpha=0.55,
                     ec='k', lw=1.2))
    up = np.array(up)
    axL.plot(up[:, 0], up[:, 1], color=GOLD, lw=2.2, marker='o', ms=5,
             zorder=6, label='E4H walk-up (mode-0 ownership)')
    dn = np.array([c for _, c in chain])
    axL.plot(dn[:, 0], dn[:, 1], color=BLUE, lw=2.2, marker='o', ms=5,
             zorder=6, label=f'H9 lineage fold L{A}\u2192L{L_DEEP}')
    for lv, c in chain[1:]:
        axL.add_patch(MP(hexv(c, 3.0 ** (A - lv)), closed=True, fc='none',
                         ec=BLUE, lw=1.0, alpha=0.7))
    axL.plot(*rep_xy, marker='*', color=RED, ms=16, zorder=7,
             label='mode-0 representative')
    axL.legend(handles=axL.get_legend_handles_labels()[0] + [
        Line2D([], [], color='0.75', lw=0.7, label='E4H depth 1 (a4)'),
        Line2D([], [], color='0.88', lw=0.45,
               label=f'E4H depth {B} (a4^{B}, the address depth)')],
        loc='lower right', fontsize=9)
    span = 1.9 * pitch
    axL.set_xlim(av[0] - span, av[0] + span)
    axL.set_ylim(av[1] - span, av[1] + span)
    axL.set_aspect('equal')
    axL.axis('off')
    axL.set_title('layer-A neighbourhood', fontsize=11)

    # ---------------- right: leaf detail ----------------
    hostd = str(host_ij)
    for i in range(fine_ij[0] - 3, fine_ij[0] + 4):
        for j in range(fine_ij[1] - 3, fine_ij[1] + 4):
            c = av + T @ (np.array((i, j), float) / R)
            if np.linalg.norm(c - fc) > 0.62 * pitch:
                continue
            dij, dgs = walk_up((i, j))
            same = dij == host_ij
            axR.add_patch(MP(hexv(c, 1 / R), closed=True,
                             fc=GOLD if (i, j) == tuple(fine_ij) else 'none',
                             alpha=0.45 if (i, j) == tuple(fine_ij) else 1.0,
                             ec='0.45' if same else '0.75', lw=1.1))
            axR.annotate(f'E{dgs}', c + 0.13 * pitch * np.array([0, -1]) / 4,
                         color=BLUE if same else '0.55', fontsize=9,
                         ha='center')
            if not same:
                axR.annotate('(other host)', c - 0.23 * pitch * np.array([0, 1]) / 4,
                             color='0.55', fontsize=7, ha='center')
    lv8, c8 = chain[2]
    lv9, c9 = chain[3]
    axR.add_patch(MP(hexv(c8, 3.0 ** (A - lv8)), closed=True, fc='none',
                     ec=BLUE, lw=1.0, alpha=0.6))
    axR.add_patch(MP(hexv(c9, 3.0 ** (A - lv9)), closed=True, fc=BLUE,
                     alpha=0.30, ec=BLUE, lw=1.4))
    axR.annotate(f'leaf H9 cell (L{L_DEEP})\nin its L{lv8} parent (outline)',
                 c9, xytext=c9 - 0.30 * pitch * np.array([1.0, 0.8]),
                 color=BLUE, fontsize=9, ha='center',
                 arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.0))
    axR.plot(*rep_xy, marker='*', color=RED, ms=15, zorder=7)
    axR.set_xlim(fc[0] - 0.66 * pitch, fc[0] + 0.66 * pitch)
    axR.set_ylim(fc[1] - 0.66 * pitch, fc[1] + 0.66 * pitch)
    axR.set_aspect('equal')
    axR.axis('off')
    axR.set_title(f'leaf detail: L{L_DEEP} H9 cell among the adjacent '
                  f'depth-{B} E4H cells', fontsize=11)

    lab = f'{h9_label(host_u, with_tail=False)}E{digits}'
    fig.suptitle(f'Hex9+E4H forward transcode (v3, address space)   '
                 f'{h9_label(deep_u, with_tail=False)} (L{L_DEEP})  '
                 f'\u2192  {lab}   [\u03b4={tuple(host_ij)}]', fontsize=13)
    fig.savefig(path, dpi=140, bbox_inches='tight')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--viz', metavar='PNG', help='write a one-cell walkthrough')
    args = ap.parse_args()
    raise SystemExit(0 if main(args.viz) else 1)
