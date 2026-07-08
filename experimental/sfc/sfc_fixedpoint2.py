"""Two-type SFC search with fixed-point (interior-allowed) anchors on the
Hex9 d-cell dissection. Types S, T; unknown anchors u = (e_S, x_S, e_T, x_T)
satisfy u_k = A_{i_k} u_{ref_k} + t_{i_k} where (i_k, ref_k) encode which
piece hosts the first/last child of each type's production and which anchor
it chains to. Functional graph -> exact cycle solve, then DFS the middle
children (piece, type, variant) with exact chaining.
"""

from fractions import Fraction
from itertools import product
import sys, time

sys.path.insert(0, "/Users/ben/Documents/Projects/PyCharm/hex9/experimental/sfc")
from sfc_grammar import TILINGS, PARENT, reconstruct, label_piece

F = Fraction
RHO = [[F(0), F(-1)], [F(1), F(1)]]


def mat_mul(A, B):
    return [[A[0][0]*B[0][0]+A[0][1]*B[1][0], A[0][0]*B[0][1]+A[0][1]*B[1][1]],
            [A[1][0]*B[0][0]+A[1][1]*B[1][0], A[1][0]*B[0][1]+A[1][1]*B[1][1]]]


def mat_vec(A, v):
    return (A[0][0]*v[0] + A[0][1]*v[1], A[1][0]*v[0] + A[1][1]*v[1])


def mat_pow(A, k):
    R = [[F(1), F(0)], [F(0), F(1)]]
    for _ in range(k):
        R = mat_mul(R, A)
    return R


def solve2(M, v):
    det = M[0][0]*M[1][1] - M[0][1]*M[1][0]
    if det == 0:
        return None
    return ((M[1][1]*v[0] - M[0][1]*v[1]) / det,
            (-M[1][0]*v[0] + M[0][0]*v[1]) / det)


def build():
    pieces = []
    for k, cells in reconstruct(TILINGS["T1"]):
        j, labels = label_piece(cells)
        A = [[c / 3 for c in row] for row in mat_pow(RHO, j)]
        m_img = mat_vec(A, (F(PARENT[4][0]), F(PARENT[4][1])))
        t = (F(labels[4][0]) - m_img[0], F(labels[4][1]) - m_img[1])
        pieces.append({"A": A, "t": t, "orient": k})
    return pieces


I2 = [[F(1), F(0)], [F(0), F(1)]]

# block indices: 0=e_S 1=x_S 2=e_T 3=x_T
# FIRST-child ref for (child_type, variant): entry anchor of that child
FIRST_REF = {("S", "f"): 0, ("S", "r"): 1, ("T", "f"): 2, ("T", "r"): 3}
# LAST-child ref: exit anchor of that child
LAST_REF = {("S", "f"): 1, ("S", "r"): 0, ("T", "f"): 3, ("T", "r"): 2}


def solve_anchors(pieces, cfg):
    """cfg = [(i0,r0),(i1,r1),(i2,r2),(i3,r3)] for blocks 0..3."""
    As = [pieces[i]["A"] for i, _ in cfg]
    ts = [pieces[i]["t"] for i, _ in cfg]
    refs = [r for _, r in cfg]
    u = [None] * 4
    state = [0] * 4  # 0 unvisited, 1 in-progress, 2 done

    def resolve(k, stack):
        if state[k] == 2:
            return True
        if state[k] == 1:
            # found cycle: stack from first occurrence of k
            ci = stack.index(k)
            cyc = stack[ci:]
            C = I2
            d = (F(0), F(0))
            # u_k = A_k u_{ref_k} + t_k ; compose around cycle starting at k
            M = [[F(1), F(0)], [F(0), F(1)]]
            v = (F(0), F(0))
            for node in cyc:
                # after loop: u_start = M u_start + v  built by expanding
                pass
            # explicit compose: u_{cyc[0]} = A_{cyc[0]} u_{cyc[1]} + t...
            M = [[F(1), F(0)], [F(0), F(1)]]
            v = (F(0), F(0))
            for node in cyc:
                Mv = mat_vec(M, ts[node])
                v = (v[0] + Mv[0], v[1] + Mv[1])
                M = mat_mul(M, As[node])
            sol = solve2([[I2[0][0]-M[0][0], I2[0][1]-M[0][1]],
                          [I2[1][0]-M[1][0], I2[1][1]-M[1][1]]], v)
            if sol is None:
                return False
            u[cyc[0]] = sol
            state[cyc[0]] = 2
            # back-fill remaining cycle nodes
            for node in reversed(cyc[1:]):
                nxt = refs[node]
                u[node] = tuple(a + b for a, b in
                                zip(mat_vec(As[node], u[nxt]), ts[node]))
                state[node] = 2
            return True
        state[k] = 1
        stack.append(k)
        if not resolve(refs[k], stack):
            return False
        if state[k] != 2:
            nxt = refs[k]
            u[k] = tuple(a + b for a, b in zip(mat_vec(As[k], u[nxt]), ts[k]))
            state[k] = 2
        stack.pop()
        return True

    for k in range(4):
        if state[k] != 2:
            if not resolve(k, []):
                return None
    return u


def run():
    pieces = build()
    fA = [p["A"] for p in pieces]
    ft = [p["t"] for p in pieces]

    def apply_i(i, p):
        v = mat_vec(fA[i], p)
        return (v[0] + ft[i][0], v[1] + ft[i][1])

    first_opts = [(i, r) for i in range(9) for r in range(4)]
    sols = []
    t0 = time.time()
    n = 0
    for c0, c1 in product(first_opts, repeat=2):        # first_S, last_S
        if c0[0] == c1[0]:
            continue
        for c2, c3 in product(first_opts, repeat=2):    # first_T, last_T
            if c2[0] == c3[0]:
                continue
            n += 1
            u = solve_anchors(pieces, [c0, c1, c2, c3])
            if u is None:
                continue
            eS, xS, eT, xT = u
            if eS == xS or eT == xT:
                continue

            def entry(i, typ, var):
                base = (eS if var == "f" else xS) if typ == "S" else \
                       (eT if var == "f" else xT)
                return apply_i(i, base)

            def exitp(i, typ, var):
                base = (xS if var == "f" else eS) if typ == "S" else \
                       (xT if var == "f" else eT)
                return apply_i(i, base)

            def dfs_prod(first_cfg, last_cfg, target_e, target_x):
                (i_f, r_f), (i_l, r_l) = first_cfg, last_cfg
                tf = ("S", "f") if r_f == 0 else ("S", "r") if r_f == 1 else \
                     ("T", "f") if r_f == 2 else ("T", "r")
                tl = ("S", "f") if r_l == 1 else ("S", "r") if r_l == 0 else \
                     ("T", "f") if r_l == 3 else ("T", "r")
                if entry(i_f, *tf) != target_e or exitp(i_l, *tl) != target_x:
                    return None
                middle = [i for i in range(9) if i not in (i_f, i_l)]
                goal = entry(i_l, *tl)
                out = []
                def dfs(cur, rem, path):
                    if out:
                        return
                    if not rem:
                        if cur == goal:
                            out.append(path[:])
                        return
                    for idx, i in enumerate(rem):
                        for typ, var in (("S","f"),("S","r"),("T","f"),("T","r")):
                            if entry(i, typ, var) == cur:
                                dfs(exitp(i, typ, var), rem[:idx]+rem[idx+1:],
                                    path + [(i, typ, var)])
                                if out:
                                    return
                dfs(exitp(i_f, *tf), middle, [])
                if out:
                    return [(i_f, *tf)] + out[0] + [(i_l, *tl)]
                return None

            pS = dfs_prod(c0, c1, eS, xS)
            if pS is None:
                continue
            pT = dfs_prod(c2, c3, eT, xT)
            if pT is None:
                continue
            sols.append((u, pS, pT))
            print("SOLUTION:")
            print("  anchors e_S,x_S,e_T,x_T =", u)
            print("  S:", pS)
            print("  T:", pT)
            if len(sols) >= 3:
                print(f"(stopping early after {n} configs, "
                      f"{time.time()-t0:.0f}s)")
                return sols
    print(f"checked {n} configs in {time.time()-t0:.0f}s; "
          f"solutions: {len(sols)}")
    return sols


if __name__ == "__main__":
    run()
