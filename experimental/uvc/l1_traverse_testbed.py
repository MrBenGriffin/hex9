"""
Hex-step odometer test-bed (L1 = 108 cells, L2 = 972 cells).

Confirms the "placeholder stack" model: a directional step is a mixed-radix
odometer — each digit position cycles (period 9) at a phase offset set by the
position above it; a completed cycle carries into the prior digit; a carry off
the top rolls into the next octant via the mode flip.

Decisive metric: PHASE CARDINALITY = number of distinct neighbour-signatures a
leaf must distinguish for a fixed (leaf_digit, c2, r_mo).  If this stays BOUNDED
from L1 -> L2, the carried state is O(1) and the odometer is elegant.  If it
grows, the transition is ancestor-chain dependent.

Ground truth (adjacency) is purely geometric: 6-NN of cell centroids on the
sphere — independent of the UV machinery under review.
"""
import sys
import numpy as np
from collections import defaultdict
from hhg9 import Registrar, Points
from hhg9.h9.addressing import neighbours, hex_digits, TailStyle
from hhg9.h9.tail import tail_unpack_reversible


def latlon_to_xyz(lat_deg, lon_deg):
    la = np.radians(lat_deg); lo = np.radians(lon_deg)
    return np.column_stack([np.cos(la) * np.cos(lo),
                            np.cos(la) * np.sin(lo),
                            np.sin(la)])


def enumerate_cells(layer, density):
    reg = Registrar(); g = reg.domain('g_gcd'); b = reg.domain('b_oct')
    n = density
    lat = np.degrees(np.arcsin(np.linspace(-0.9995, 0.9995, n)))
    lon = np.linspace(-180, 180, 2 * n, endpoint=False)
    LA, LO = np.meshgrid(lat, lon)
    geo = np.column_stack([LA.ravel(), LO.ravel()])

    P = Points(geo, domain=g)
    PB = reg.project(P, [g, b])
    pts = neighbours(PB, layer=layer, coalesce=True)
    oid = pts.cm()[0].astype(np.int64)
    planar = np.asarray(pts.coords, dtype=np.float64)   # b_oct planar (per-octant frame)
    hv = np.asarray(hex_digits(pts, layer, TailStyle.reversible), dtype=np.uint8)
    body = hv[:, :-1]                      # (M, layer+1): col0..col_layer
    c2, r_mo, _p = tail_unpack_reversible(hv[:, -1])
    xyz = latlon_to_xyz(geo[:, 0], geo[:, 1])

    cells = {}
    for i in range(hv.shape[0]):
        key = (int(oid[i]), *(int(x) for x in body[i]))   # identity: octant + full body
        rec = cells.get(key)
        if rec is None:
            cells[key] = rec = {'xyz_sum': np.zeros(3), 'planar_sum': np.zeros(2), 'cnt': 0,
                                'oct': int(oid[i]), 'body': tuple(int(x) for x in body[i]),
                                'leaf': int(body[i, -1]), 'c2_votes': defaultdict(int),
                                'r_mo': int(r_mo[i])}
        rec['xyz_sum'] += xyz[i]; rec['planar_sum'] += planar[i]; rec['cnt'] += 1
        rec['c2_votes'][int(c2[i])] += 1
    out = list(cells.values())
    for r in out:
        c = r['xyz_sum'] / r['cnt']
        r['xyz'] = c / np.linalg.norm(c)
        r['planar'] = r['planar_sum'] / r['cnt']
        r['c2'] = max(r['c2_votes'], key=r['c2_votes'].get)
    return out


def build_cells(layer, density, verbose=True):
    """Enumerate cells and attach a geometric neighbour signature.

    Each returned cell dict gains:
      'sig'      : sorted tuple of 6 neighbour (leaf, c2) — None if not interior
      'interior' : non-aliased, degree-6, 2-ring-clean (away from vertices)
      'ring'     : indices of the 6 geometric neighbours
    Also returns body2idx: full-body tuple (incl. octant) -> cell index.
    """
    from scipy.spatial import cKDTree
    cells = enumerate_cells(layer, density)
    C = np.array([r['xyz'] for r in cells])

    # 6-NN via KD-tree on the unit sphere (chord distance). k=10 leaves room for an
    # alias at chord~0 plus the 6-ring. Thresholds are SCALE-RELATIVE (cell spacing
    # shrinks ~3x per layer): s = local neighbour spacing (median over k-1, robust
    # to a single alias at ~0). An alias is a twin hexagon at chord << s.
    tree = cKDTree(C)
    dist, idx = tree.query(C, k=min(10, len(cells)))
    s = np.median(dist[:, 1:], axis=1)
    aliased = dist[:, 1] < 0.4 * s
    ring = [[idx[i, t] for t in range(1, idx.shape[1])
             if 0.4 * s[i] <= dist[i, t] < 1.6 * s[i]][:6]
            for i in range(len(cells))]

    for i, r in enumerate(cells):
        r['ring'] = ring[i]
        r['spacing'] = float(s[i])
        r['nn_ratio'] = float(dist[i, 1] / s[i])      # nearest-twin closeness (<<1 => alias)
        r['twin'] = int(idx[i, 1]) if aliased[i] else None
        r['aliased'] = bool(aliased[i])
        r['interior'] = not (aliased[i] or len(ring[i]) != 6
                             or any(aliased[j] for j in ring[i]))
        r['sig'] = (tuple(sorted((cells[j]['leaf'], cells[j]['c2']) for j in ring[i]))
                    if r['interior'] else None)
    if verbose:
        ni = sum(r['interior'] for r in cells)
        print(f"L{layer}: {len(cells)} cells (expect {12 * 9 ** layer}); "
              f"aliased(vertex): {int(aliased.sum())}; interior: {ni}")
    body2idx = {(r['oct'], *r['body']): i for i, r in enumerate(cells)}
    return cells, body2idx


def analyse(layer, density):
    cells, _ = build_cells(layer, density)
    print(f"===== L{layer} context-freeness =====")
    good = np.array([r['interior'] for r in cells])
    nbr_ms = [r['sig'] for r in cells]

    def ctx_free(name, keyfn):
        groups = defaultdict(set)
        for i in range(len(cells)):
            if good[i]:
                groups[keyfn(i)].add(nbr_ms[i])
        free = sum(len(v) == 1 for v in groups.values())
        print(f"  key={name:<34} context-free: {free}/{len(groups)}")
        return free, len(groups)

    print("Context-freeness ladder (key -> neighbour (leaf',c2') multiset):")
    b = cells  # alias
    ctx_free("(leaf,c2)", lambda i: (b[i]['leaf'], b[i]['c2']))
    ctx_free("(leaf,c2,r_mo)", lambda i: (b[i]['leaf'], b[i]['c2'], b[i]['r_mo']))
    # ORDER-1: condition on the immediate parent only.
    ctx_free("(parent,leaf,c2,r_mo)  [order-1]",
             lambda i: (b[i]['body'][-2], b[i]['leaf'], b[i]['c2'], b[i]['r_mo']))
    if layer >= 3:
        # ORDER-2: add the grandparent. If order-1 already = 100%, this must NOT
        # split any class further -> grandparent is redundant -> parent screens off.
        ctx_free("(gp,parent,leaf,c2,r_mo) [order-2]",
                 lambda i: (b[i]['body'][-3], b[i]['body'][-2], b[i]['leaf'], b[i]['c2'], b[i]['r_mo']))

    phases = defaultdict(set)
    for i in range(len(cells)):
        if good[i]:
            phases[(b[i]['leaf'], b[i]['c2'], b[i]['r_mo'])].add(nbr_ms[i])
    card = {k: len(v) for k, v in phases.items()}
    if not card:
        print("PHASE CARDINALITY: no interior cells (enumeration too sparse?)")
        return 0
    hist = defaultdict(int)
    for v in card.values():
        hist[v] += 1
    print(f"PHASE CARDINALITY per (leaf,c2,r_mo): max={max(card.values())}  "
          f"distribution={dict(sorted(hist.items()))}")
    return max(card.values())


if __name__ == '__main__':
    if len(sys.argv) > 1:
        layer = int(sys.argv[1])
        density = int(sys.argv[2]) if len(sys.argv) > 2 else 360 * 3 ** (layer - 1)
        analyse(layer, density)
    else:
        analyse(1, density=360)
        analyse(2, density=900)
    print("\n>>> ORDER-1 CHECK: if (parent,leaf,c2,r_mo) = 100% AND order-2 adds no"
          " split, the parent screens off all higher ancestors => O(1) odometer.")
