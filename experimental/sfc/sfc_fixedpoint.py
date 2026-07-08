"""Single-type SFC search with FIXED-POINT (interior) anchors on the Hex9
d-cell dissection — the 'starting at internals' construction from other_2.png.

Type T's entry anchor e and exit anchor x are not chosen from named points;
they are forced by the recursion:
    e = f_{i1}(e or x)   (entry lives inside the first child, recursively)
    x = f_{i9}(x or e)   (exit  lives inside the last  child, recursively)
('or x'/'or e' when the first/last child is traversed in reverse.)
These are linear over Q in Eisenstein (a,b) coords -> solve exactly, then
check the 8 chaining equalities  f_{ik}(exit_k) == f_{ik+1}(entry_{k+1}).
If all hold, the system closes at EVERY level (the parent-level chain
conditions are the same equations), connection points land on shared
boundaries automatically, and the tile's own endpoints may be interior.
"""

from fractions import Fraction
from itertools import product

import sys
sys.path.insert(0, "/Users/ben/Documents/Projects/PyCharm/hex9/experimental/sfc")
from sfc_grammar import TILINGS, PARENT, reconstruct, label_piece

F = Fraction
RHO = [[F(0), F(-1)], [F(1), F(1)]]  # 60-degree rotation in (a,b) basis


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


def sub_m(A, B):
    return [[A[0][0]-B[0][0], A[0][1]-B[0][1]], [A[1][0]-B[1][0], A[1][1]-B[1][1]]]


I2 = [[F(1), F(0)], [F(0), F(1)]]


def build():
    pieces = []
    for k, cells in reconstruct(TILINGS["T1"]):
        j, labels = label_piece(cells)
        A = [[c / 3 for c in row] for row in mat_pow(RHO, j)]
        m_img = mat_vec(A, (F(PARENT[4][0]), F(PARENT[4][1])))
        t = (F(labels[4][0]) - m_img[0], F(labels[4][1]) - m_img[1])
        pieces.append({"A": A, "t": t, "orient": k, "labels": labels})
    return pieces


def f_apply(piece, p):
    v = mat_vec(piece["A"], p)
    return (v[0] + piece["t"][0], v[1] + piece["t"][1])


def anchors(pieces, i1, v1, i9, v9):
    """Solve e = f_i1(e|x), x = f_i9(x|e). Returns (e, x) or None."""
    A1, t1 = pieces[i1]["A"], pieces[i1]["t"]
    A9, t9 = pieces[i9]["A"], pieces[i9]["t"]
    if v1 == "f":
        e = solve2(sub_m(I2, A1), t1)
        if e is None:
            return None
        if v9 == "f":
            x = solve2(sub_m(I2, A9), t9)
        else:
            x = tuple(a + b for a, b in zip(mat_vec(A9, e), t9))
    else:
        if v9 == "f":
            x = solve2(sub_m(I2, A9), t9)
            if x is None:
                return None
            e = tuple(a + b for a, b in zip(mat_vec(A1, x), t1))
        else:
            M = sub_m(I2, mat_mul(A9, A1))
            rhs = tuple(a + b for a, b in zip(mat_vec(A9, t1), t9))
            x = solve2(M, rhs)
            if x is None:
                return None
            e = tuple(a + b for a, b in zip(mat_vec(A1, x), t1))
    if x is None:
        return None
    return e, x


def search(pieces):
    sols = []
    for i1, i9 in product(range(9), repeat=2):
        if i1 == i9:
            continue
        for v1, v9 in product("fr", repeat=2):
            res = anchors(pieces, i1, v1, i9, v9)
            if res is None:
                continue
            e, x = res
            if e == x:
                continue  # closed-loop degenerate; note if needed
            # entry/exit plane point of child i under variant v
            def ept(i, v):
                return f_apply(pieces[i], e if v == "f" else x)
            def xpt(i, v):
                return f_apply(pieces[i], x if v == "f" else e)
            middle = [i for i in range(9) if i not in (i1, i9)]
            goal_start = ept(i9, v9)

            def dfs(cur, remaining, path):
                if not remaining:
                    if cur == goal_start:
                        sols.append((i1, v1, path[:], i9, v9, e, x))
                    return
                for idx, i in enumerate(remaining):
                    for v in "fr":
                        if ept(i, v) == cur:
                            dfs(xpt(i, v), remaining[:idx] + remaining[idx+1:],
                                path + [(i, v)])
            dfs(xpt(i1, v1), middle, [])
    return sols


if __name__ == "__main__":
    pieces = build()
    sols = search(pieces)
    print(f"single-type fixed-point-anchor systems: {len(sols)}")
    seen_fwd = [s for s in sols if s[1] == "f" and s[4] == "f"
                and all(v == "f" for _, v in s[2])]
    print(f"  of which forward-only: {len(seen_fwd)}")
    for s in (seen_fwd or sols)[:5]:
        i1, v1, mid, i9, v9, e, x = s
        order = [(i1, v1)] + mid + [(i9, v9)]
        print("  order:", " ".join(f"{i}{'' if v=='f' else chr(39)}" for i, v in order),
              f"| e={e} x={x}")
