"""
k-ring (4b): flood the combinatorial neighbour graph from a centre address.

SET-thread: needs only neighbours + carry, no cyclic order / global frame.
BFS over the region 3-edge adjacency (step_region from build_stepper, which uses
cmc2n + the pmn!=pmo carry), then collapse regions -> hexagon addresses for the
hex k-ring. Within-octant for now (a carry off the octant returns None -> the
seam step, item 3, will extend this).

Validation: a 2-D disk should grow ~quadratically (|disk(k)| ~ 1 + 3k(k+1) for hexes).
"""
import sys
import numpy as np
from collections import Counter
from build_stepper import make_chain, step_region
from hhg9.h9.addressing import reg_hex_digits, TailStyle
from hhg9 import Registrar

reg = Registrar(); b = reg.domain('b_oct')


def flood_regions(start_chain, oc, k):
    """Flood regions k deep; return (region dist, hexagon adjacency graph)."""
    start = tuple(int(x) for x in start_chain)
    dist = {start: 0}
    frontier = [start_chain]
    hex_adj = {}                       # hexkey -> set(neighbour hexkeys)
    hk = {start: hex_key(start_chain, oc)}
    for d in range(1, k + 1):
        nxt = []
        for ch in frontier:
            a = tuple(int(x) for x in ch)
            ka = hk[a]
            for edge in range(3):
                nb = step_region(np.asarray(ch, dtype=np.int64), edge)
                if nb is None:
                    continue
                t = tuple(int(x) for x in nb)
                kb = hk.get(t) or hex_key(nb, oc)
                hk[t] = kb
                if ka != kb:           # region step crossed a hexagon boundary
                    hex_adj.setdefault(ka, set()).add(kb)
                    hex_adj.setdefault(kb, set()).add(ka)
                if t not in dist:
                    dist[t] = d
                    nxt.append(nb)
        frontier = nxt
    return dist, hex_adj, hk[start]


def bfs_hex(hex_adj, centre, k):
    """BFS the hexagon graph -> {hexkey: hexagon-distance}."""
    dist = {centre: 0}
    frontier = [centre]
    for d in range(1, k + 1):
        nxt = []
        for h in frontier:
            for n in hex_adj.get(h, ()):
                if n not in dist:
                    dist[n] = d
                    nxt.append(n)
        frontier = nxt
    return dist


def hex_key(chain, oc):
    hv = reg_hex_digits(np.asarray(chain, dtype=np.uint8)[None, :], oc, b, TailStyle.key)
    return tuple(int(x) for x in hv[0])


def hexagon_centroids(region_chains, oc):
    """exact cell centroids (regions_xy), grouped+averaged per hexagon key."""
    from hhg9.h9.region import regions_xy
    arr = np.asarray(list(region_chains), dtype=np.uint8)
    xy = regions_xy(arr)[:, :2]
    acc = {}
    for ch, p in zip(region_chains, xy):
        k = hex_key(np.asarray(ch, dtype=np.uint8), oc)
        s, n = acc.get(k, (np.zeros(2), 0))
        acc[k] = (s + p, n + 1)
    return {k: s / n for k, (s, n) in acc.items()}


def main(lat, lon, K):
    from scipy.spatial import cKDTree
    chain, pb, oc = make_chain(lat, lon)
    rd, _adj, centre = flood_regions(chain, oc, 3 * K)
    print(f"\ncentre ({lat},{lon}) octant {int(oc[0])}  depth {len(chain) - 2}; "
          f"{len(rd)} regions flooded")

    # hexagon adjacency = 6-NN on exact cell centroids (matches the 6-neighbour LUT).
    cent = hexagon_centroids(rd.keys(), oc)
    keys = list(cent); P = np.array([cent[k] for k in keys])
    tree = cKDTree(P)
    # the 6 nearest cells (edge- AND vertex-neighbours span ~4x in distance, so
    # take k=6 rather than a single radius). Mutual-kNN keeps it symmetric.
    _, idx = tree.query(P, k=min(7, len(keys)))
    raw = {i: set(idx[i, 1:7]) for i in range(len(keys))}
    adj = {keys[i]: {keys[j] for j in raw[i] if i in raw[j]} for i in range(len(keys))}

    hd = bfs_hex(adj, centre, K)
    hc = Counter(hd.values())
    print("hexagon k-ring (6-NN on exact centroids):")
    cum = 0
    for d in sorted(hc):
        cum += hc[d]
        print(f"  ring {d}: +{hc[d]:<3d} (disk {cum},  ideal 6d={6*d if d else 1}, hex-disk {1+3*d*(d+1)})")


if __name__ == '__main__':
    lat = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    lon = float(sys.argv[2]) if len(sys.argv) > 2 else 25.0
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    main(lat, lon, K)
