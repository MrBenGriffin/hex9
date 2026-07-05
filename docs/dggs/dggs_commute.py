# Part of the Hex9 (H9) Project
# Copyright ©2026, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Hierarchy/geometry commutation across DGGS: does encode-then-truncate equal
encode-at-the-coarse-level?

Companion to dggs_nesting.py (the picture); this is the number. For a point p
and levels L, L+k, every hierarchical DGGS offers two routes to a coarse cell:

    direct     :  cell(p, L)                       — geometric binning
    via child  :  ancestor_L( cell(p, L+k) )       — index truncation

A hierarchy is layer-transitive when the two routes agree for every point:
bin once at the finest level, roll up for free; join datasets binned at
different resolutions by ancestry. Where they disagree, the user must choose
per task between geometric correctness (direct, but no cross-layer joins) and
index consistency (via child, but coarse counts geometrically misassigned).
The dggs_nesting.py red/gold cells are the geometric face of the same
disagreement; this script measures its rate directly — no polygons, no
approximation, each system through its own native operations:

    H3   : latlng_to_cell / cell_to_parent
    S2   : CellId.from_lat_lng / CellId.parent
    A5   : lonlat_to_cell / cell_to_parent
    Hex9 : h9_bin_pts (direct) / h9_bin on the child UUID alone (via child)

Sampling is uniform on the sphere (fixed seed; area-true via arcsin latitude).
Base levels match dggs_nesting.py's DEFAULT_RES (comparable anchor sizes);
the rate is a property of the refinement step, not of the level chosen.

Run:
    python docs/dggs/dggs_commute.py                    # N=10000, k=1..3
    python docs/dggs/dggs_commute.py --n 100000 --gens 4

Result (N=10000, seed 9, k=1..6): S2 0.00% at every k (congruent quadtree).
H3 is CONSTANT at ~6.5% regardless of k — the fixed shape mismatch between the
parent hexagon and the Gosper-island-like limit of its descendant set. A5 is
constant at ~35% (the rhombus limit of its pentagon refinement). Hex9 DECAYS:
16.76%, 5.88%, 1.94%, 0.62%, 0.23%, 0.07% — matching the predicted
(1/6)*3^(1-k) split-hex band (3 of 9 children split, half of each outside the
owner). The limit shape of a Hex9 cell's descendants IS the ancestor, so the
disagreement is a boundary band that thins with depth, and each disagreeing
point is assigned to a cell that shares the straddling hexagon with its
geometric parent — adjacent, never far. At the d_cell (half-hex) level the
Hex9 rate is zero at every depth; only the paired-hexagon layer carries the
band. --deep (bin at LMAX, roll up to the analysis layer): H3 15->5 6.50%,
S2 30->8 0.00%, A5 30->5 35.02%, Hex9 30->5 0.00% — at workload depth the
Hex9 band has decayed below any measurable rate ((1/6)*3^-24 ~ 1e-13) while
the H3/A5 constants stand. The deep row is Hex9's normal pipeline, not a
favourable corner: encode once to the full L30 address, then derive every
coarser bin from the address alone (truncation + tail canonicalisation, O(1),
no re-encoding from geometry). See §12 of the paper (Ancestry).

Last edited: 2026-07-05 (first version; --deep mode added).
"""
import argparse
import numpy as np


# ── per-system commutation trials ────────────────────────────────────────────
# Each returns the number of points where direct != via-child, for one k.

def h3_trial(lats, lngs, res, k):
    import h3
    bad = 0
    for lat, lng in zip(lats, lngs):
        direct = h3.latlng_to_cell(lat, lng, res)
        via = h3.cell_to_parent(h3.latlng_to_cell(lat, lng, res + k), res)
        bad += direct != via
    return bad


def s2_trial(lats, lngs, res, k):
    import s2sphere as s2
    bad = 0
    for lat, lng in zip(lats, lngs):
        leaf = s2.CellId.from_lat_lng(s2.LatLng.from_degrees(lat, lng))
        bad += leaf.parent(res) != leaf.parent(res + k).parent(res)
    return bad


def a5_trial(lats, lngs, res, k):
    import a5_fast as a5
    bad = 0
    for lat, lng in zip(lats, lngs):
        direct = a5.lonlat_to_cell(float(lng), float(lat), res)
        child = a5.lonlat_to_cell(float(lng), float(lat), res + k)
        bad += direct != a5.cell_to_parent(child, res)
    return bad


def hex9_trial(lats, lngs, res, k, _cache={}):
    """Direct = h9_bin_pts(p, L). Via-child = h9_bin(child_uuid, L) — the
    ancestor is derived from the child's 128-bit key alone (decode + re-bin
    through the canonical tail convention), never from the original point."""
    from hhg9 import Registrar, Points
    from hhg9.h9.uuid_address import h9_bin_pts, h9_bin
    if 'b_pts' not in _cache:
        reg = Registrar()
        g_gcd, b_oct = reg.domain('g_gcd'), reg.domain('b_oct')
        g = Points(np.column_stack([lats, lngs]).astype(float), g_gcd)
        _cache['reg'] = reg
        _cache['b_pts'] = reg.project(g, [g_gcd, b_oct])
    reg, b_pts = _cache['reg'], _cache['b_pts']
    direct = h9_bin_pts(b_pts, res)
    via = h9_bin(h9_bin_pts(b_pts, res + k), res, reg)
    return sum(d != v for d, v in zip(direct, via))


TRIALS = {'H3': h3_trial, 'S2': s2_trial, 'A5': a5_trial, 'Hex9': hex9_trial}
# Same anchors as dggs_nesting.py: comparable cell sizes, own numbering.
DEFAULT_RES = {'H3': 5, 'S2': 8, 'A5': 5, 'Hex9': 5}
# --deep: the workload-realistic case — bin at the system's maximum resolution,
# roll all the way up to the DEFAULT_RES analysis layer (k = LMAX - base).
LMAX = {'H3': 15, 'S2': 30, 'A5': 30, 'Hex9': 30}


def sample_sphere(n, seed=9):
    rng = np.random.default_rng(seed)
    lats = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, n)))
    lngs = rng.uniform(-180.0, 180.0, n)
    return lats, lngs


def main(n, gens, seed, deep):
    lats, lngs = sample_sphere(n, seed)
    print(f'N = {n} uniform sphere points, seed = {seed}')
    print(f'commutation check: cell(p, L)  vs  ancestor_L(cell(p, L+k))\n')
    if deep:
        print(f'{"system":>6} {"fine":>6} {"coarse":>7} {"k":>3} {"mismatch":>16}')
        for name, trial in TRIALS.items():
            res = DEFAULT_RES[name]
            k = LMAX[name] - res
            bad = trial(lats, lngs, res, k)
            print(f'{name:>6} {LMAX[name]:>6} {res:>7} {k:>3} '
                  f'{bad:>7} ({100.0 * bad / n:5.2f}%)')
        return
    header = f'{"system":>6} {"base":>6} ' + ''.join(f'{f"k={k}":>16}' for k in range(1, gens + 1))
    print(header)
    for name, trial in TRIALS.items():
        res = DEFAULT_RES[name]
        row = f'{name:>6} {res:>6} '
        for k in range(1, gens + 1):
            bad = trial(lats, lngs, res, k)
            row += f'{bad:>7} ({100.0 * bad / n:5.2f}%)'
        print(row)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--n', type=int, default=10000)
    ap.add_argument('--gens', type=int, default=3)
    ap.add_argument('--seed', type=int, default=9)
    ap.add_argument('--deep', action='store_true',
                    help='bin at each system\'s max resolution, roll up K levels')
    args = ap.parse_args()
    main(args.n, args.gens, args.seed, args.deep)