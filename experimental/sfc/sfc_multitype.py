"""Multi-type traversal-grammar search (greatest fixed point) on the Hex9
d-cell 9-dissection, generalizing sfc_grammar.py:

- arc TYPES are endpoint pairs over {P0,P1,P2,P3,M}, including loop types (P,P)
- a type is VIABLE iff some ordering of the 9 children, each assigned a viable
  type (and direction / mirror-variant per regime), chains the parent's
  endpoints. Greatest fixed point: start with all types, prune until stable.
- if ANY type survives, a (multi-type) SFC exists; if none survives, no SFC
  with endpoints in the candidate set exists at all.

Also reports near-miss depth: the longest chain prefix achievable (out of 9).
"""

from itertools import product
from sfc_grammar import (TILINGS, PARENT, SIGMA_LAB, reconstruct, label_piece,
                         boundary_edges)

LABELS = [0, 1, 2, 3, 4]
NAME = {0: "P0", 1: "P1", 2: "P2", 3: "P3", 4: "M"}


def child_options(piece_labels, viable, regime):
    """(entry_coord, exit_coord, used_type) triples for one child."""
    opts = []
    for (p, q) in viable:
        forms = [(p, q)]
        if regime != "FWD":
            forms.append((q, p))
        if regime == "FWD+REV+MIRROR":
            forms.append((SIGMA_LAB[p], SIGMA_LAB[q]))
            forms.append((SIGMA_LAB[q], SIGMA_LAB[p]))
        for (e, x) in forms:
            opts.append((piece_labels[e], piece_labels[x], (p, q)))
    # dedupe on coords (keep first type tag)
    seen, ded = set(), []
    for e, x, t in opts:
        if (e, x) not in seen:
            seen.add((e, x))
            ded.append((e, x, t))
    return ded


def realizable(tau, pieces, viable, regime, want_example=False):
    p, q = tau
    start, goal = PARENT[p], PARENT[q]
    opts = [child_options(pc["labels"], viable, regime) for pc in pieces]
    best = {"depth": 0}
    found = []

    def dfs(cur, used, path):
        if found:
            return
        if len(path) > best["depth"]:
            best["depth"] = len(path)
        if len(path) == 9:
            if cur == goal:
                found.append(list(path))
            return
        for i in range(9):
            if used >> i & 1:
                continue
            for e, x, t in opts[i]:
                if e == cur:
                    dfs(x, used | 1 << i, path + [(i, t, e, x)])
                    if found:
                        return
    dfs(start, 0, [])
    if want_example:
        return (found[0] if found else None), best["depth"]
    return bool(found), best["depth"]


def run(tname, regime, ordered):
    sol = TILINGS[tname]
    pieces = []
    for k, cells in reconstruct(sol):
        j, labels = label_piece(cells)
        pieces.append({"orient": k, "rot": j, "labels": labels,
                       "bedges": boundary_edges(cells)})

    if ordered:  # FWD-only: types are ordered pairs (incl. loops)
        types = [(a, b) for a, b in product(LABELS, repeat=2)]
    else:        # unordered (reversal makes (p,q) ~ (q,p)); loops included
        types = [(a, b) for a in LABELS for b in LABELS if a <= b]

    viable = set(types)
    while True:
        dead = set()
        for tau in viable:
            ok, _ = realizable(tau, pieces, viable, regime)
            if not ok:
                dead.add(tau)
        if not dead:
            break
        viable -= dead

    # near-miss depth against the FULL type set (most generous measure)
    depths = {}
    for tau in types:
        _, d = realizable(tau, pieces, set(types), regime)
        depths[tau] = d
    dmax = max(depths.values())
    argmax = [f"{NAME[a]}->{NAME[b]}" for (a, b), d in depths.items() if d == dmax]
    print(f"  {tname} [{regime}]: surviving types = "
          f"{[f'{NAME[a]}-{NAME[b]}' for a, b in sorted(viable)] or 'NONE'}; "
          f"max chain prefix {dmax}/9 (types: {', '.join(argmax[:4])}"
          f"{'...' if len(argmax) > 4 else ''})")
    if viable:
        tau = sorted(viable)[0]
        ex, _ = realizable(tau, pieces, viable, regime, want_example=True)
        print(f"    example for {NAME[tau[0]]}-{NAME[tau[1]]}: " +
              " ".join(f"c{i}:{NAME[t[0]]}{NAME[t[1]]}" for i, t, e, x in ex))


if __name__ == "__main__":
    for tname in TILINGS:
        run(tname, "FWD", ordered=True)
        run(tname, "FWD+REV", ordered=False)
        run(tname, "FWD+REV+MIRROR", ordered=False)
