# Probe: when does region_neighbours' mcc2 lookup fall through to the
# geometric (nearest-centroid) tie-break?  Mirrors region.py:802-823.
#
# Three corpora:
#   1. interior  : random lat/lon (interior w.p. 1)            -> expect ~0
#   2. centroids : decode(addr) re-encoded (maximally interior)-> expect 0
#   3. edges     : midpoints of adjacent-cell centroids        -> expect > 0
#
# Corpus 2 is the load-bearing claim: traversal from a *stored* canonical
# address never re-enters the geometric fallback.
import sys
sys.path.insert(0, '.')
import numpy as np
from hhg9.h9.region import (region_neighbours, regions_xy, xy_regions, H9CTX, H9R,
                            recover_stats_reset, _RECOVER_STATS)
from hhg9 import Registrar, Points

C, R = H9CTX.c, H9CTX.r
BAD = int(R.invalid_region)


def fallback_mask(addr):
    """Replicates the `bad = (c2 == invalid_region)` test in region_neighbours."""
    cur = addr[:, -2]
    imo = C.mode[cur]
    term = addr[:, -1]
    c2 = R.mcc2[imo, term]
    return c2 == R.invalid_region


def latlon_to_addr(lats, lons, depth):
    reg = Registrar()
    g, b = reg.domain('g_gcd'), reg.domain('b_oct')
    bp = reg.project(Points(np.column_stack([lats, lons]), g), [g, b])
    oc, mode = bp.cm()
    return xy_regions(bp.coords, mode, depth)


def report(name, mask):
    n = mask.size
    k = int(mask.sum())
    print(f"  {name:<10s}: {k:>7,} / {n:>7,} fallback  ({100.0*k/max(n,1):.4f}%)")


def run(n, depth):
    print(f"\n[depth={depth}, n={n}]")
    rng = np.random.default_rng(42)
    lats = rng.uniform(-89.5, 89.5, n)
    lons = rng.uniform(-179.5, 179.5, n)

    addr = latlon_to_addr(lats, lons, depth)
    report("interior", fallback_mask(addr))

    # Corpus 2: decode -> re-encode the centroid (interior by construction).
    xym = regions_xy(addr)
    ctr_addr = xy_regions(xym[:, :2], xym[:, 2].astype(np.uint8), depth)
    report("centroids", fallback_mask(ctr_addr))

    # Corpus 3: midpoints of (cell, neighbour) centroids land on the shared edge.
    nb, _ = region_neighbours(addr)
    own = regions_xy(addr)
    oth = regions_xy(nb)
    hopped = addr[:, 0] != nb[:, 0]          # drop octant-spanning (y-flip) pairs
    keep = ~hopped
    mid = (own[keep, :2] + oth[keep, :2]) / 2.0
    mode_mid = own[keep, 2].astype(np.uint8)
    recover_stats_reset()
    edge_addr = xy_regions(mid, mode_mid, depth)
    report("edges", fallback_mask(edge_addr))
    s = _RECOVER_STATS
    print(f"    _recover on edge corpus: nonzero-calls={s['nonzero']:,}  "
          f"pts_in={s['pts_in']:,}  unrecovered={s['unrecovered']:,}")


def run_validity(n):
    """Hold geometry out entirely: vary only address *validity* of the (cur, term)
    terminal pair that mcc2 inspects.  Confirms the fallback is a guard against
    illegal addresses, not a float-edge tie-break."""
    print(f"\n[validity, n={n}] (geometry-free; random terminal pairs)")
    rng = np.random.default_rng(7)
    count = int(C.count)

    # Naive-random address: last two nibbles are arbitrary region-ids.
    cur = rng.integers(0, count, n).astype(np.uint8)
    term = rng.integers(0, count, n).astype(np.uint8)
    naive = np.zeros((n, 3), dtype=np.uint8)
    naive[:, -2] = cur
    naive[:, -1] = term
    report("naive-rand", fallback_mask(naive))

    # Legal-random address: pick a real cell, then a genuine c2-child terminal.
    valid_cells = np.unique(C.c2.reshape(-1))
    valid_cells = valid_cells[valid_cells != R.invalid_region]
    cur = rng.choice(valid_cells, n).astype(np.uint8)
    imo = C.mode[cur]
    # For each row pick a legal child from C.c2[mode] (shape (2, 3, k)).
    k = C.c2.shape[1] * C.c2.shape[2]
    flat = C.c2.reshape(2, -1)
    pick = rng.integers(0, k, n)
    term = flat[imo, pick].astype(np.uint8)
    legal = np.zeros((n, 3), dtype=np.uint8)
    legal[:, -2] = cur
    legal[:, -1] = term
    report("legal-rand", fallback_mask(legal))


if __name__ == '__main__':
    for d in (4, 15, 29):
        run(200_000, d)
    run_validity(1_000_000)
