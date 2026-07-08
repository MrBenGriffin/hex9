"""Decisive face-continuity analysis for vertex-anchor FWD systems:
full production enumeration + interleaved coinductive pruning.

sup(t, side, r): some surviving production of t whose first/last child
(a, u) has the unit boundary edge along r at the anchor AND sup(u, side,
pulled-back r). Computed as a GFP given current menus.

A production is pruned when some internal handoff has NO shared unit-edge
ray r with sup(exiting child type, exit, r-in-its-frame) AND
sup(entering child type, entry, r-in-its-frame).

Menus and sup-sets shrink monotonically; the fixpoint over-approximates
every real system. Empty => impossible, for ALL vertex-anchor FWD systems
over the named anchors. Nonempty => attempt consistent assignment by
randomized repair.
"""

import random
import sys
from itertools import product

sys.path.insert(0, "/Users/ben/Documents/Projects/PyCharm/hex9/experimental/sfc")
sys.path.insert(0, "/private/tmp/claude-501/-Users-ben-Documents-Projects-PyCharm-hex9/"
                   "cfc1bd91-e8e0-48c9-8f26-a83d7c3b56fa/scratchpad")
from sfc_grammar import PARENT, rot
from sfc_edge import build_pieces, gfp, edge_ok, NAME, LABELS

random.seed(4)

RAYS = {
    0: [(1, 0), (0, 1)],
    1: [(-1, 0), (-1, 1)],
    2: [(1, -1), (-1, 0)],
    3: [(1, 0), (0, -1)],
    4: [(1, 0), (-1, 0)],
}
UNITS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]


def main():
    pieces = build_pieces()
    all_types = [t for t in product(LABELS, repeat=2)]
    surv = sorted(gfp(pieces, all_types))
    print(f"types: {len(surv)}")

    def unit_edge(a, pos, r):
        return frozenset((pos, (pos[0]+r[0], pos[1]+r[1]))) in pieces[a]["bedges"]

    # ---- full enumeration ----
    menus = {}
    for t in surv:
        p, q = t
        start, goal = PARENT[p], PARENT[q]
        out = []
        def dfs(cur, used, k, prev, path):
            if k == 9:
                if cur == goal:
                    out.append(tuple(path))
                return
            for i in range(9):
                if used >> i & 1:
                    continue
                for u in surv:
                    lab = pieces[i]["labels"]
                    if lab[u[0]] == cur and (prev is None or
                                             edge_ok(pieces, prev, i, cur)):
                        dfs(lab[u[1]], used | 1 << i, k + 1, i,
                            path + [(i, u)])
        dfs(start, 0, 0, None, [])
        menus[t] = out
        print(f"  {NAME[t[0]]}->{NAME[t[1]]}: {len(out)} productions")
        sys.stdout.flush()

    def pull(r, a):
        return rot(r, (6 - pieces[a]["rot"]) % 6)

    def sup_gfp(menus):
        nodes = {(t, s, tuple(r)) for t in surv for s, c in
                 (("entry", t[0]), ("exit", t[1])) for r in RAYS[c]}
        alive = set(nodes)
        while True:
            dead = set()
            for (t, s, r) in alive:
                pos = PARENT[t[0] if s == "entry" else t[1]]
                ok = False
                for prod_ in menus[t]:
                    a, u = prod_[0] if s == "entry" else prod_[-1]
                    if not unit_edge(a, pos, r):
                        continue
                    rp = tuple(pull(r, a))
                    if (u, s, rp) in alive:
                        ok = True
                        break
                if not ok:
                    dead.add((t, s, r))
            if not dead:
                return alive
            alive -= dead

    def handoffs_ok(t, prod_, sup):
        for (a, u), (b, u2) in zip(prod_, prod_[1:]):
            pos = pieces[a]["labels"][u[1]]
            good = False
            for d in UNITS:
                if unit_edge(a, pos, d) and unit_edge(b, pos, d):
                    ra = tuple(pull(d, a))
                    rb = tuple(pull(d, b))
                    if (u, "exit", ra) in sup and (u2, "entry", rb) in sup:
                        good = True
                        break
            if not good:
                return False
        return True

    rounds = 0
    while True:
        rounds += 1
        sup = sup_gfp(menus)
        new_menus = {t: [p for p in menus[t] if handoffs_ok(t, p, sup)]
                     for t in surv}
        changed = any(len(new_menus[t]) != len(menus[t]) for t in surv)
        menus = new_menus
        sizes = {f"{NAME[t[0]]}->{NAME[t[1]]}": len(menus[t]) for t in surv}
        alivec = sum(1 for t in surv if menus[t])
        print(f"round {rounds}: sup={len(sup)} states, "
              f"{alivec}/{len(surv)} types with productions, "
              f"total prods {sum(len(m) for m in menus.values())}")
        sys.stdout.flush()
        if not changed:
            break
        if all(not menus[t] for t in surv):
            break

    live = [t for t in surv if menus[t]]
    if not live:
        print("\n=> EMPTY: no vertex-anchor FWD system over the named "
              "anchors can be face-continuous. Corner hops are unavoidable.")
        return
    print(f"\nfixpoint nonempty: {len(live)} live types; menu sizes: "
          f"{[(NAME[t[0]]+'->'+NAME[t[1]], len(menus[t])) for t in live]}")
    # randomized-repair consistent assignment over reduced menus
    def chain_sup(chosen):
        """Support sets under a FIXED choice per type (deterministic chains)."""
        nodes = {(t, s, tuple(r)) for t in chosen for s, c in
                 (("entry", t[0]), ("exit", t[1])) for r in RAYS[c]}
        alive = set(nodes)
        while True:
            dead = set()
            for (t, s, r) in alive:
                pos = PARENT[t[0] if s == "entry" else t[1]]
                pr = chosen[t]
                a, u = pr[0] if s == "entry" else pr[-1]
                if (not unit_edge(a, pos, r)) or u not in chosen or \
                        (u, s, tuple(pull(r, a))) not in alive:
                    dead.add((t, s, r))
            if not dead:
                return alive
            alive -= dead

    best = None
    for attempt in range(500):
        chosen = {t: random.choice(menus[t]) for t in live}
        for _ in range(60):
            sup2 = chain_sup(chosen)
            bad = [t for t in live if not handoffs_ok(t, chosen[t], sup2)
                   or any(u not in chosen for _, u in chosen[t])]
            nb = len(bad)
            if best is None or nb < best[0]:
                best = (nb, dict(chosen))
            if nb == 0:
                print("CONSISTENT FACE-CONTINUOUS SYSTEM FOUND:")
                for t in live:
                    print(f"  {NAME[t[0]]}->{NAME[t[1]]} ::= " +
                          " ".join(f"c{i}[{NAME[u[0]]}{NAME[u[1]]}]"
                                   for i, u in chosen[t]))
                return
            t = random.choice(bad)
            chosen[t] = random.choice(menus[t])
    print(f"repair best: {best[0]} violated types (of {len(live)}) — "
          f"needs CP-SAT for a definitive consistent-assignment verdict")


if __name__ == "__main__":
    main()
