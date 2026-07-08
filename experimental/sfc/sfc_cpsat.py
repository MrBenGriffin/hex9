"""CP-SAT decision: does a face-continuous vertex-anchor FWD system exist?

Variables:
  used[t]        type t participates
  x[t,p]         production p chosen for t (exactly one iff used)
  sup[t,s,r]     ray support: t's s-side descent keeps an edge along r

Constraints:
  sum_p x[t,p] == used[t]
  x[t,p] -> used[u] for every child type u in p
  sup[t,s,r] -> used[t]
  sup[t,s,r] & x[t,p] -> unit-edge at level 1 AND sup[child, s, pulled r]
  x[t,p] -> every internal handoff has a shared ray d with
            sup[exit child, exit, d_a] & sup[entry child, entry, d_b]
  at least one type used

The per-side ray condition is necessary AND sufficient for face-continuity
(convexity fixes the shared line; positive overlap at depth k <=> both
sides' depth-k cells own a unit edge from the handoff point along d).
So SAT => system exists (verify by expansion); UNSAT => theorem.
"""

import sys
from itertools import product

from ortools.sat.python import cp_model

sys.path.insert(0, "/Users/ben/Documents/Projects/PyCharm/hex9/experimental/sfc")
sys.path.insert(0, "/private/tmp/claude-501/-Users-ben-Documents-Projects-PyCharm-hex9/"
                   "cfc1bd91-e8e0-48c9-8f26-a83d7c3b56fa/scratchpad")
from sfc_grammar import PARENT, rot
from sfc_edge import (build_pieces, gfp, edge_ok, NAME, LABELS,
                      expand, cells_share_edge)

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

    def unit_edge(a, pos, r):
        return frozenset((pos, (pos[0]+r[0], pos[1]+r[1]))) in pieces[a]["bedges"]

    def pull(r, a):
        return tuple(rot(r, (6 - pieces[a]["rot"]) % 6))

    # full menus (same enumeration as sfc_prune)
    menus = {}
    for t in surv:
        start, goal = PARENT[t[0]], PARENT[t[1]]
        out = []
        def dfs(cur, used_, k, prev, path):
            if k == 9:
                if cur == goal:
                    out.append(tuple(path))
                return
            for i in range(9):
                if used_ >> i & 1:
                    continue
                for u in surv:
                    lab = pieces[i]["labels"]
                    if lab[u[0]] == cur and (prev is None or
                                             edge_ok(pieces, prev, i, cur)):
                        dfs(lab[u[1]], used_ | 1 << i, k + 1, i,
                            path + [(i, u)])
        dfs(start, 0, 0, None, [])
        menus[t] = out
    total = sum(len(m) for m in menus.values())
    print(f"types {len(surv)}, productions {total}")

    m = cp_model.CpModel()
    used = {t: m.NewBoolVar(f"used_{t}") for t in surv}
    x = {(t, i): m.NewBoolVar(f"x_{t}_{i}")
         for t in surv for i in range(len(menus[t]))}
    supv = {}
    for t in surv:
        for s, c in (("entry", t[0]), ("exit", t[1])):
            for r in RAYS[c]:
                supv[(t, s, tuple(r))] = m.NewBoolVar(f"sup_{t}_{s}_{r}")

    for t in surv:
        m.Add(sum(x[(t, i)] for i in range(len(menus[t]))) == 1) \
            .OnlyEnforceIf(used[t])
        m.Add(sum(x[(t, i)] for i in range(len(menus[t]))) == 0) \
            .OnlyEnforceIf(used[t].Not())

    for (t, s, r), sv in supv.items():
        m.AddImplication(sv, used[t])
        pos = PARENT[t[0] if s == "entry" else t[1]]
        for i, prod_ in enumerate(menus[t]):
            a, u = prod_[0] if s == "entry" else prod_[-1]
            if not unit_edge(a, pos, r):
                m.AddBoolOr([sv.Not(), x[(t, i)].Not()])
            else:
                rp = pull(r, a)
                key = (u, s, rp)
                if key in supv:
                    m.AddBoolOr([sv.Not(), x[(t, i)].Not(), supv[key]])
                else:
                    m.AddBoolOr([sv.Not(), x[(t, i)].Not()])

    for t in surv:
        for i, prod_ in enumerate(menus[t]):
            for (a, u), (b, u2) in zip(prod_, prod_[1:]):
                m.AddImplication(x[(t, i)], used[u])
                pos = pieces[a]["labels"][u[1]]
                alts = []
                for d in UNITS:
                    if unit_edge(a, pos, d) and unit_edge(b, pos, d):
                        ka = (u, "exit", pull(d, a))
                        kb = (u2, "entry", pull(d, b))
                        if ka in supv and kb in supv:
                            y = m.NewBoolVar(f"y_{t}_{i}_{a}_{b}_{d}")
                            m.AddImplication(y, supv[ka])
                            m.AddImplication(y, supv[kb])
                            alts.append(y)
                if alts:
                    m.AddBoolOr([x[(t, i)].Not()] + alts)
                else:
                    m.Add(x[(t, i)] == 0)
            m.AddImplication(x[(t, i)], used[prod_[-1][1]])

    m.AddBoolOr([used[t] for t in surv])

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 900
    solver.parameters.num_search_workers = 8
    status = solver.Solve(m)
    print("status:", solver.StatusName(status))
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        if status == cp_model.INFEASIBLE:
            print("=> THEOREM: no vertex-anchor FWD system over the named "
                  "anchors is face-continuous. Corner hops are unavoidable.")
        return
    chosen = {}
    for t in surv:
        if solver.Value(used[t]):
            for i in range(len(menus[t])):
                if solver.Value(x[(t, i)]):
                    chosen[t] = list(menus[t][i])
    print(f"SAT: {len(chosen)} used types")
    for t, pr in sorted(chosen.items()):
        print(f"  {NAME[t[0]]}->{NAME[t[1]]} ::= " +
              " ".join(f"c{i}[{NAME[u[0]]}{NAME[u[1]]}]" for i, u in pr))
    # independent verification by expansion
    prods = {t: pr for t, pr in chosen.items()}
    for t in sorted(chosen):
        ok = True
        for d in range(1, 6):
            seq = expand(t, prods, pieces, d)
            for (e1, x1, p1), (e2, x2, p2) in zip(seq, seq[1:]):
                assert x1 == e2
                if not cells_share_edge(p1, p2):
                    print(f"  !! {NAME[t[0]]}->{NAME[t[1]]} hops at depth {d}")
                    ok = False
                    break
            if not ok:
                break
        if ok:
            print(f"  {NAME[t[0]]}->{NAME[t[1]]}: face-continuous through "
                  f"depth 5 VERIFIED")


if __name__ == "__main__":
    main()
