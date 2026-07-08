"""Ray-descent automaton: decides whether ANY vertex-anchor FWD system on
the T1 dissection can be face-continuous.

At a corner handoff the two flanking descents are independent: each must
produce, at every level, a cell with a boundary edge along the shared-edge
ray r from the anchor point. Per side this is a walk in a finite automaton:

  node  (type t, side in {entry, exit}, ray r at that anchor corner)
  edge  choose (piece a, child type u) with the child's matching anchor at
        the tile corner, the child cell having a unit boundary edge along r
        there; recurse with ray pulled back through the child's rotation.

RELAXED transitions (any anchor-consistent child, ignoring whether a full
production exists) give an over-approximation: empty GFP here proves
face-continuity impossible for every vertex-anchor FWD system.
"""

import sys
from itertools import product

sys.path.insert(0, "/Users/ben/Documents/Projects/PyCharm/hex9/experimental/sfc")
sys.path.insert(0, "/private/tmp/claude-501/-Users-ben-Documents-Projects-PyCharm-hex9/"
                   "cfc1bd91-e8e0-48c9-8f26-a83d7c3b56fa/scratchpad")
from sfc_grammar import PARENT, rot
from sfc_edge import build_pieces, gfp, NAME, LABELS

# boundary rays (primitive lattice steps) at each anchor label
RAYS = {
    0: [(1, 0), (0, 1)],      # P0 -> toward P1, toward P3
    1: [(-1, 0), (-1, 1)],    # P1 -> toward P0, toward P2
    2: [(1, -1), (-1, 0)],    # P2 -> toward P1, toward P3
    3: [(1, 0), (0, -1)],     # P3 -> toward P2, toward P0
    4: [(1, 0), (-1, 0)],     # M  -> along the long edge both ways
}


def main():
    pieces = build_pieces()
    all_types = [t for t in product(LABELS, repeat=2)]
    surv = sorted(gfp(pieces, all_types))
    print(f"survivor types: {len(surv)}")

    # nodes
    nodes = set()
    for t in surv:
        for side, corner in (("entry", t[0]), ("exit", t[1])):
            for r in RAYS[corner]:
                nodes.add((t, side, r))

    def unit_edge_at(a, corner_pos, r):
        e = frozenset((corner_pos, (corner_pos[0] + r[0], corner_pos[1] + r[1])))
        return e in pieces[a]["bedges"]

    def transitions(node, alive):
        t, side, r = node
        corner = t[0] if side == "entry" else t[1]
        pos = PARENT[corner]
        out = []
        for a in range(9):
            lab = pieces[a]["labels"]
            j = pieces[a]["rot"]
            for u in surv:
                uc = u[0] if side == "entry" else u[1]
                if lab[uc] != pos:
                    continue
                if not unit_edge_at(a, pos, r):
                    continue
                rp = rot(r, (6 - j) % 6)   # pull ray back into child frame
                if rp not in RAYS[uc]:
                    continue
                nxt = (u, side, rp)
                if nxt in alive:
                    out.append((a, u, nxt))
        return out

    alive = set(nodes)
    while True:
        dead = {n for n in alive if not transitions(n, alive)}
        if not dead:
            break
        alive -= dead
    print(f"ray-descent GFP: {len(alive)}/{len(nodes)} nodes survive")
    if not alive:
        print("=> NO vertex-anchor FWD system can be face-continuous.")
        return
    by_side = {}
    for t, side, r in sorted(alive):
        by_side.setdefault(side, []).append((NAME[t[0]] + "->" + NAME[t[1]], r))
    for side, lst in by_side.items():
        print(f"  {side}: {len(lst)} viable (type, ray) states")
        for name, r in lst[:12]:
            print(f"    {name} ray {r}")


if __name__ == "__main__":
    main()
