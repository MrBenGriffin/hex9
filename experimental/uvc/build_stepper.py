"""
Step (2): a purely combinatorial region stepper on cmc2n.

State = region chain (cell ids). Step across c2-edge `edge` at the leaf:
  nbr, nbr_pmo = cmc2n[leaf_cell, parent_mode, edge]
  if nbr_pmo == parent_mode : no carry, done
  else                      : carry -> step the parent by the same edge (recurse up)
A carry past the protos = octant boundary (stop).

Validation: walk a fixed edge and read the leaf HEX digit each step; it should
cycle with period 9 (the 860671782-family).
"""
import numpy as np
from hhg9 import Registrar, Points
from hhg9.h9.region import H9R, xy_regions
from hhg9.h9.lattice import H9C
from hhg9.h9 import H9O
from hhg9.h9.addressing import reg_hex_digits, TailStyle

cmc2n = H9R.cmc2n
mode = H9C.mode
BAD = H9R.invalid_region
oid_nb = H9O.oid_nb
PROTO = H9R.proto                  # [0x49, 0x16] = mode0, mode1 root protos
C2_FLIP = np.array([0, 2, 1])      # seam c2 equivalence {0:0,1:2,2:1} (the y-flip)

reg = Registrar(); g = reg.domain('g_gcd'); b = reg.domain('b_oct')
DEPTH = 5


def make_chain(lat, lon):
    pt = Points(np.array([[float(lat), float(lon)]]), domain=g)
    pb = reg.project(pt, [g, b])
    oc, mo = pb.cm()
    cx = xy_regions(pb.coords, mo, DEPTH)      # (1, 2+DEPTH) cell ids
    return cx[0].astype(np.int64), pb, oc


def leaf_digit(chain, oc, pb):
    hv = reg_hex_digits(chain[None, :], oc, b, TailStyle.reversible)
    return int(hv[0, -2])                        # last body col = leaf hex digit


def step_region(chain, edge):
    chain = chain.copy()
    i = len(chain) - 1
    while i >= 2:                                 # cx[0],cx[1] are protos (octant frame)
        cell = int(chain[i]); par = int(chain[i - 1]); pmo = int(mode[par])
        nbr, npmo = int(cmc2n[cell, pmo, edge][0]), int(cmc2n[cell, pmo, edge][1])
        if nbr == BAD:
            return None
        chain[i] = nbr
        if npmo == pmo:
            return chain                         # no carry
        i -= 1                                    # carry: step parent too
    return None                                   # carried off the octant


def step_region_oct(chain, oct, edge):
    """Octant-aware step: a carry off the octant crosses the seam.
       oct -> oid_nb[oct, edge];  proto flips;  c2 follows {0:0,1:2,2:1} thereafter.
       Returns (chain, oct) or None."""
    chain = np.asarray(chain, dtype=np.int64).copy()
    i = len(chain) - 1
    while i >= 1:                                  # now descend into chain[1] (octant L0) too
        cell = int(chain[i]); par = int(chain[i - 1]); pmo = int(mode[par])
        e = cmc2n[cell, pmo, edge]
        nbr, npmo = int(e[0]), int(e[1])
        if nbr == BAD:
            return None
        chain[i] = nbr
        if npmo == pmo:
            return chain, oct                      # carry resolved within the octant
        i -= 1
    # carry reached the proto -> cross the seam
    new_oct = int(oid_nb[oct, edge])
    if new_oct < 0:
        return None
    pmo0 = int(mode[int(chain[0])])
    chain[0] = int(PROTO[1 - pmo0])                # flip root proto
    return chain, new_oct


def _demo():
    # walk each edge from a few starts, read leaf digit, look for period 9
    for lat, lon in [(0.0, 10.0), (0.0, 20.0), (12.0, 33.0)]:
        chain, pb, oc = make_chain(lat, lon)
        print(f"\nstart ({lat},{lon}) octant {int(oc[0])} chain={chain.tolist()}")
        for edge in range(3):
            seq, ch = [], chain.copy()
            ok = True
            for _ in range(20):
                seq.append(leaf_digit(ch, oc, pb))
                ch = step_region(ch, edge)
                if ch is None:
                    ok = False
                    break
            s = ''.join(str(d) for d in seq)
            per9 = len(seq) >= 18 and s[:9] == s[9:18]
            tag = 'PERIOD-9' if per9 else ('left-octant' if not ok else 'not-periodic')
            print(f"  edge {edge}: {s:<22} [{tag}]")


if __name__ == '__main__':
    _demo()
