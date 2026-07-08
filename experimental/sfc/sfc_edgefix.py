"""Mirror-closure SFC search restricted to EDGE-INTERIOR anchors
(entry-edge / exit-edge classes, per Ben): every anchor must lie strictly
inside a tile edge. Such systems are automatically face-continuous at all
depths. Derived from sfc_mirror_fixed.py.

sigma(p) = S p + c, S = [[-1,-1],[0,1]], c = (6,0):
  P0(0,0)<->P1(6,0), P3(0,3)<->P2(3,3), M(3,0) fixed;  sigma^2 = id.

A mirrored-type child inside piece i has effective map
  f_i(sigma(u)) = (A_i S) u + (A_i c + t_i).

Level 1: one essential shape + mirror  {B, MB}   (fast, inline)
Level 2: two essential shapes + mirror {A, MA, B, MB}  (large; run with
         argv[1] == 'big')
"""

from fractions import Fraction
from itertools import product
import sys, time

sys.path.insert(0, "/Users/ben/Documents/Projects/PyCharm/hex9/experimental/sfc")
from sfc_grammar import TILINGS, PARENT, reconstruct, label_piece

F = Fraction
RHO = [[F(0), F(-1)], [F(1), F(1)]]
I2 = [[F(1), F(0)], [F(0), F(1)]]
S_MIR = [[F(-1), F(-1)], [F(0), F(1)]]
C_MIR = (F(6), F(0))


def mat_mul(A, B):
    return [[A[0][0]*B[0][0]+A[0][1]*B[1][0], A[0][0]*B[0][1]+A[0][1]*B[1][1]],
            [A[1][0]*B[0][0]+A[1][1]*B[1][0], A[1][0]*B[0][1]+A[1][1]*B[1][1]]]


def mat_vec(A, v):
    return (A[0][0]*v[0] + A[0][1]*v[1], A[1][0]*v[0] + A[1][1]*v[1])


def mat_pow(A, k):
    R = I2
    for _ in range(k):
        R = mat_mul(R, A)
    return R


def solve2(M, v):
    det = M[0][0]*M[1][1] - M[0][1]*M[1][0]
    if det == 0:
        return None
    return ((M[1][1]*v[0] - M[0][1]*v[1]) / det,
            (-M[1][0]*v[0] + M[0][0]*v[1]) / det)




def edge_of(p):
    a, b = p
    if b == 0 and 0 < a < 6: return "long"
    if b == 3 and 0 < a < 3: return "short"
    if a == 0 and 0 < b < 3: return "left"
    if a + b == 6 and 0 < b < 3: return "right"
    return None

def build():
    """Per piece, plain and mirrored effective affine maps."""
    out = []
    for k, cells in reconstruct(TILINGS["T1"]):
        j, labels = label_piece(cells)
        A = [[cc / 3 for cc in row] for row in mat_pow(RHO, j)]
        m_img = mat_vec(A, (F(PARENT[4][0]), F(PARENT[4][1])))
        t = (F(labels[4][0]) - m_img[0], F(labels[4][1]) - m_img[1])
        Am = mat_mul(A, S_MIR)
        Ac = mat_vec(A, C_MIR)
        tm = (t[0] + Ac[0], t[1] + Ac[1])
        out.append(((A, t), (Am, tm)))
    return out


def sanity(pieces):
    """sigma involution + mirrored maps keep tile corners inside the tile."""
    s = mat_mul(S_MIR, S_MIR)
    assert s == I2
    p1 = mat_vec(S_MIR, (F(6), F(0)))
    assert (p1[0] + C_MIR[0], p1[1] + C_MIR[1]) == (F(0), F(0))
    corners = [(F(0), F(0)), (F(6), F(0)), (F(3), F(3)), (F(0), F(3))]
    for (A, t), (Am, tm) in pieces:
        for p in corners:
            for (M, T) in ((A, t), (Am, tm)):
                q = mat_vec(M, p)
                q = (q[0] + T[0], q[1] + T[1])
                # inside trapezoid: 0<=b<=3, a>=0, a+b<=6... b in [0,3]
                assert 0 <= q[1] <= 3 and q[0] >= 0 and q[0] + q[1]/1 <= 6, q


def search(pieces, n_real, tag, early_stop=10):
    """n_real essential shapes; effective types = each shape plain/mirrored.
    Blocks: e/x per real shape. etype k -> (shape s, mirrored m).
    entry(f) ref-block = e_s ; entry(r) = x_s ; exit flips."""
    n_blocks = 2 * n_real
    etypes = [(s, m) for s in range(n_real) for m in (0, 1)]
    fl_opts = [(i, k, v) for i in range(9) for k in range(len(etypes))
               for v in "fr"]

    def eff(i, m):
        return pieces[i][m]

    def entry_ref(k, v):
        s, m = etypes[k]
        return (2 * s if v == "f" else 2 * s + 1), m

    def exit_ref(k, v):
        s, m = etypes[k]
        return (2 * s + 1 if v == "f" else 2 * s), m

    t0 = time.time()
    n = 0
    sols = []
    # blocks: for shape s, block 2s = first-child eq, 2s+1 = last-child eq
    shape_fl = []
    for s in range(n_real):
        pairs = [(a, b) for a in fl_opts for b in fl_opts if a[0] != b[0]]
        shape_fl.append(pairs)

    for combo in product(*shape_fl):
        n += 1
        eqs = []
        ok = True
        for s, (first, last) in enumerate(combo):
            i, k, v = first
            rb, m = entry_ref(k, v)
            A, t = eff(i, m)
            eqs.append((A, rb, t))
            i, k, v = last
            rb, m = exit_ref(k, v)
            A, t = eff(i, m)
            eqs.append((A, rb, t))
        u = [None] * n_blocks
        state = [0] * n_blocks

        def resolve(k, stack):
            if state[k] == 2:
                return True
            if state[k] == 1:
                ci = stack.index(k)
                cyc = stack[ci:]
                M = I2
                vv = (F(0), F(0))
                for node in cyc:
                    Mv = mat_vec(M, eqs[node][2])
                    vv = (vv[0] + Mv[0], vv[1] + Mv[1])
                    M = mat_mul(M, eqs[node][0])
                sol = solve2([[1 - M[0][0], -M[0][1]],
                              [-M[1][0], 1 - M[1][1]]], vv)
                if sol is None:
                    return False
                u[cyc[0]] = sol
                state[cyc[0]] = 2
                for node in reversed(cyc[1:]):
                    nb = eqs[node][1]
                    u[node] = tuple(a + b for a, b in
                                    zip(mat_vec(eqs[node][0], u[nb]),
                                        eqs[node][2]))
                    state[node] = 2
                return True
            state[k] = 1
            stack.append(k)
            if not resolve(eqs[k][1], stack):
                return False
            if state[k] != 2:
                nb = eqs[k][1]
                u[k] = tuple(a + b for a, b in
                             zip(mat_vec(eqs[k][0], u[nb]), eqs[k][2]))
                state[k] = 2
            if stack and stack[-1] == k:
                stack.pop()
            return True

        for k in range(n_blocks):
            if state[k] != 2 and not resolve(k, []):
                ok = False
                break
        if not ok:
            continue
        if any(u[2*s] == u[2*s+1] for s in range(n_real)):
            continue
        if any(edge_of(u[k]) is None for k in range(n_blocks)):
            continue

        def entry(i, k, v):
            rb, m = entry_ref(k, v)
            A, t = eff(i, m)
            p = mat_vec(A, u[rb])
            return (p[0] + t[0], p[1] + t[1])

        def exitp(i, k, v):
            rb, m = exit_ref(k, v)
            A, t = eff(i, m)
            p = mat_vec(A, u[rb])
            return (p[0] + t[0], p[1] + t[1])

        good = True
        for s, (first, last) in enumerate(combo):
            if entry(*first) != u[2*s] or exitp(*last) != u[2*s+1]:
                good = False
                break
        if not good:
            continue

        prods = []
        for s, (first, last) in enumerate(combo):
            middle = [i for i in range(9) if i not in (first[0], last[0])]
            goal = entry(*last)
            out = []
            def dfs(cur, rem, path):
                if out:
                    return
                if not rem:
                    if cur == goal:
                        out.append(path[:])
                    return
                for idx, i in enumerate(rem):
                    for k in range(len(etypes)):
                        for v in "fr":
                            if entry(i, k, v) == cur:
                                dfs(exitp(i, k, v), rem[:idx] + rem[idx+1:],
                                    path + [(i, k, v)])
                                if out:
                                    return
            dfs(exitp(*first), middle, [])
            if not out:
                prods = None
                break
            prods.append([first] + out[0] + [last])
        if prods is None:
            continue
        sols.append((list(u), prods))
        names = {0: "B", 1: "MB"} if n_real == 1 else \
                {0: "A", 1: "MA", 2: "B", 3: "MB"}
        ec = [f"{edge_of(u[2*s])}->{edge_of(u[2*s+1])}" for s in range(n_real)]
        print(f"SOLUTION [{tag}] anchors: {u}  edge-classes: {ec}")
        for s, p in enumerate(prods):
            print(f"  shape {s}:",
                  [(i, names[k], v) for i, k, v in p])
        sys.stdout.flush()
        if len(sols) >= early_stop:
            print(f"(early stop, {n} configs, {time.time()-t0:.0f}s)")
            return sols
        continue
    print(f"[{tag}] done: {n} configs in {time.time()-t0:.0f}s; "
          f"solutions {len(sols)}")
    return sols


if __name__ == "__main__":
    pieces = build()
    sanity(pieces)
    if len(sys.argv) > 1 and sys.argv[1] == "big":
        search(pieces, 2, "2 shapes + mirror")
    else:
        search(pieces, 1, "1 shape + mirror")
