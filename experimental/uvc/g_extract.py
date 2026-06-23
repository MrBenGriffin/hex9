"""
g-extraction: is the phase-propagation rule a deterministic finite-state machine?

Model (Ben's linked list, fed from the leaf end): each digit position is a node
holding a minimal hidden 'phase'. The phase is SET top-down by a rule
    child_phase = g(parent_phase, child_digit, child_c2, r_mo)
A directional step is injected at the leaf and carries ripple up.

This script co-enumerates two consecutive layers, links each interior leaf cell
to its interior parent (by body prefix), and tests whether g is a function:

  g_full : (parent_sig, child_leaf, child_c2, r_mo) -> child_sig          deterministic?
  g_min  : (parent_leaf, parent_c2, parent_phase, child_leaf, child_c2, r_mo)
                                                    -> child_phase         deterministic?

where 'sig' = geometric 6-neighbour (leaf,c2) multiset, and 'phase' = index of a
cell's sig within its (leaf,c2,r_mo) bucket (the bounded <=4-valued hidden state).
g_min deterministic + small  =>  the minimal 2-bit linked-list node is real.
"""
import sys
import numpy as np
from collections import defaultdict
from l1_traverse_testbed import build_cells


def dump_g_table(g_min, path):
    """Write the deterministic phase-update FSM to a readable table."""
    lines = ["# g: (p_leaf,p_c2,p_phase, leaf,c2, r_mo) -> phase",
             "# p_* = parent node state; leaf/c2 = this node; phase = this node's hidden state"]
    for k in sorted(g_min):
        v = g_min[k]
        if len(v) == 1:
            pl, pc, pp, le, c2, rm = k
            lines.append(f"p({pl},{pc},{pp}) | this({le},{c2}) r_mo={rm} -> {next(iter(v))}")
    with open(path, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  g table ({sum(len(v)==1 for v in g_min.values())} rows) written to {path}")


def vertex_probe(cells):
    """Characterize the 6 octahedral vertices and the cells hugging them."""
    print("\n--- vertex / pentagon probe ---")
    C = np.array([r['xyz'] for r in cells])
    twins = [i for i, r in enumerate(cells) if r['nn_ratio'] < 0.15]   # true twins (chord~0)
    print(f"true-twin cells (nn_ratio<0.15): {len(twins)}")

    # cluster twin cells by centroid into vertex groups (greedy, coarse threshold).
    used = set(); verts = []
    med_s = float(np.median([cells[i]['spacing'] for i in range(len(cells))]))
    for i in twins:
        if i in used:
            continue
        grp = [j for j in twins if np.dot(C[i], C[j]) > 1 - (3 * med_s) ** 2 / 2]
        used.update(grp)
        verts.append(grp)
    print(f"vertex clusters found: {len(verts)} (expect 6)")

    # degree distribution among twin cells (pentagons should be degree 5).
    deg = defaultdict(int)
    for i in twins:
        deg[len(cells[i]['ring'])] += 1
    print(f"degree distribution of twin cells: {dict(sorted(deg.items()))}")

    # huggers = non-twin cells within ~2.5 spacings of a vertex centroid.
    if verts:
        vc = [np.mean([C[j] for j in g], axis=0) for g in verts]
        vc = [v / np.linalg.norm(v) for v in vc]
        thr = 1 - (2.5 * med_s) ** 2 / 2
        hug = [i for i in range(len(cells))
               if cells[i]['nn_ratio'] >= 0.15 and any(np.dot(C[i], v) > thr for v in vc)]
        hdeg = defaultdict(int)
        for i in hug:
            hdeg[len(cells[i]['ring'])] += 1
        n_hug_interior = sum(cells[i]['interior'] for i in hug)
        print(f"hugger cells (1-2 rings out): {len(hug)};  degree dist: {dict(sorted(hdeg.items()))};"
              f"  interior-flagged: {n_hug_interior}")


def assign_phases(*cell_sets):
    """Global sig->phase index per (leaf,c2,r_mo) bucket, shared across layers."""
    buckets = defaultdict(set)
    for cells in cell_sets:
        for r in cells:
            if r['interior']:
                buckets[(r['leaf'], r['c2'], r['r_mo'])].add(r['sig'])
    order = {k: {sig: i for i, sig in enumerate(sorted(v))} for k, v in buckets.items()}
    for cells in cell_sets:
        for r in cells:
            r['phase'] = order[(r['leaf'], r['c2'], r['r_mo'])][r['sig']] if r['interior'] else None
    return order


def det(rel):
    """rel: key -> set(value). Return (deterministic_keys, total_keys)."""
    det_keys = sum(1 for v in rel.values() if len(v) == 1)
    return det_keys, len(rel)


def main(child_layer, dens_parent, dens_child):
    parent_layer = child_layer - 1
    print(f"Co-enumerating L{parent_layer} (parent) and L{child_layer} (child)\n")
    pcells, pidx = build_cells(parent_layer, dens_parent)
    ccells, cidx = build_cells(child_layer, dens_child)
    order = assign_phases(pcells, ccells)

    def oct_interior(cells, r):
        # all 6 geometric neighbours share the cell's octant (no seam crossing).
        return r['interior'] and all(cells[j]['oct'] == r['oct'] for j in r['ring'])

    g_full = defaultdict(set)
    g_min = defaultdict(set)
    n_linked = n_orphan = n_seam = 0
    for c in ccells:
        if not c['interior']:
            continue
        if not oct_interior(ccells, c):
            n_seam += 1
            continue
        pkey = (c['oct'], *c['body'][:-1])      # parent = drop the leaf digit
        pj = pidx.get(pkey)
        if pj is None or not oct_interior(pcells, pcells[pj]):
            n_orphan += 1
            continue
        p = pcells[pj]
        n_linked += 1
        rmo = c['r_mo']
        g_full[(p['sig'], c['leaf'], c['c2'], rmo)].add(c['sig'])
        g_min[(p['leaf'], p['c2'], p['phase'], c['leaf'], c['c2'], rmo)].add(c['phase'])

    print(f"\nlinked octant-interior child->parent pairs: {n_linked}  "
          f"(seam-excluded: {n_seam}, orphans: {n_orphan})")

    df, tf = det(g_full)
    dm, tm = det(g_min)
    n_states_full = len({k[0] for k in g_full})        # distinct parent sigs used
    max_phase = max((p['phase'] for p in pcells if p['interior']), default=-1) + 1
    print("\n--- g determinism ---")
    print(f"  g_full  (parent_sig -> child_sig)          : {df}/{tf} deterministic")
    print(f"  g_min   (parent leaf,c2,phase -> phase)    : {dm}/{tm} deterministic")
    print(f"\n  distinct parent signatures (raw states)    : {n_states_full}")
    print(f"  max phase index (minimal hidden state size): {max_phase}  -> {max_phase} values")
    if dm == tm:
        print("\n  => g_min is a DETERMINISTIC FSM on the minimal (leaf,c2,phase) state.")
        print("     The linked-list node carries a bounded phase. Odometer confirmed.")
        dump_g_table(g_min, "experimental/uvc/g_table.txt")
        vertex_probe(ccells)
    else:
        print("\n  => g_min NOT fully deterministic; inspect residuals (seam noise vs real).")
        shown = 0
        for k, v in g_min.items():
            if len(v) > 1 and shown < 8:
                print(f"     parent(leaf={k[0]},c2={k[1]},ph={k[2]}) child(leaf={k[3]},c2={k[4]}) "
                      f"r_mo={k[5]} -> phases {sorted(v)}")
                shown += 1


if __name__ == '__main__':
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    dp = int(sys.argv[2]) if len(sys.argv) > 2 else 360 * 3 ** (L - 2)
    dc = int(sys.argv[3]) if len(sys.argv) > 3 else 360 * 3 ** (L - 1)
    main(L, dp, dc)
