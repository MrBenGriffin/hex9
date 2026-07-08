"""Independent expansion verification of the two-shapes-plus-mirror SFC
system (solution 1 from sfc_mirror_fixed.py big). Twin-hierarchy semantics:
a mirrored child folds sigma into its frame map (its subtree uses the
mirrored dissection); a reversed child flips its subtree order and flips
each grandchild's direction. Checks at depths 1..3, exact rationals:
  - cell count 9^d and all cells distinct (they tile by construction)
  - continuity: consecutive exit == entry
  - global endpoints == the solved anchors
"""

from fractions import Fraction
import sys

sys.path.insert(0, "/Users/ben/Documents/Projects/PyCharm/hex9/experimental/sfc")
from sfc_grammar import TILINGS, PARENT, reconstruct, label_piece

F = Fraction
RHO = [[F(0), F(-1)], [F(1), F(1)]]
I2 = [[F(1), F(0)], [F(0), F(1)]]
S_MIR = [[F(-1), F(-1)], [F(0), F(1)]]
C_MIR = (F(6), F(0))
CORNERS = [(F(0), F(0)), (F(6), F(0)), (F(3), F(3)), (F(0), F(3))]


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


def compose(g, h):
    """(g o h)(x) = Mg(Mh x + th) + tg."""
    (Mg, tg), (Mh, th) = g, h
    M = mat_mul(Mg, Mh)
    v = mat_vec(Mg, th)
    return (M, (v[0] + tg[0], v[1] + tg[1]))


def apply(g, p):
    M, t = g
    v = mat_vec(M, p)
    return (v[0] + t[0], v[1] + t[1])


SIGMA = (S_MIR, C_MIR)


def build():
    out = []
    for k, cells in reconstruct(TILINGS["T1"]):
        j, labels = label_piece(cells)
        A = [[cc / 3 for cc in row] for row in mat_pow(RHO, j)]
        m_img = mat_vec(A, (F(PARENT[4][0]), F(PARENT[4][1])))
        t = (F(labels[4][0]) - m_img[0], F(labels[4][1]) - m_img[1])
        out.append((A, t))
    return out


# ---- solution 1 from the corrected big search ----
E = {"A": (F(0), F(0)), "B": (F(0), F(0))}
X = {"A": (F(3), F(3)), "B": (F(0), F(3))}
PROD = {
    "A": [(0, "MA", "f"), (5, "MA", "r"), (7, "B", "f"), (8, "MA", "f"),
          (4, "MA", "f"), (1, "MB", "r"), (2, "MB", "f"), (3, "B", "f"),
          (6, "MA", "r")],
    "B": [(0, "MA", "f"), (5, "B", "r"), (1, "MB", "r"), (2, "MB", "f"),
          (3, "B", "f"), (6, "MA", "r"), (8, "B", "f"), (4, "MB", "f"),
          (7, "A", "r")],
}


def plain(k):
    return k[-1]          # 'MA' -> 'A'


def mirrored(k):
    return k.startswith("M")


def expand(pieces, g, shape, d, depth, out):
    """Emit (cell_key, entry, exit) leaves in curve order."""
    if depth == 0:
        e = apply(g, E[shape] if d == "f" else X[shape])
        x = apply(g, X[shape] if d == "f" else E[shape])
        key = frozenset(apply(g, c) for c in CORNERS)
        out.append((key, e, x))
        return
    seq = PROD[shape]
    if d == "r":
        seq = [(i, k, ("r" if v == "f" else "f")) for (i, k, v) in
               reversed(seq)]
    for i, k, v in seq:
        h = compose(g, (pieces[i][0], pieces[i][1]))
        if mirrored(k):
            h = compose(h, SIGMA)
        expand(pieces, h, plain(k), v, depth - 1, out)


def main():
    pieces = build()
    IDG = (I2, (F(0), F(0)))
    for top in ("A", "B"):
        for depth in (1, 2, 3):
            out = []
            expand(pieces, IDG, top, "f", depth, out)
            assert len(out) == 9 ** depth, (top, depth, len(out))
            assert len({c for c, _, _ in out}) == len(out), "cells not distinct"
            assert out[0][1] == E[top], "start anchor wrong"
            assert out[-1][2] == X[top], "end anchor wrong"
            for (c1, e1, x1), (c2, e2, x2) in zip(out, out[1:]):
                assert x1 == e2, f"discontinuity at depth {depth}"
            print(f"type {top} depth {depth}: {len(out)} cells — "
                  f"continuity, coverage, endpoints VERIFIED")
    # also verify the mirrored types via their sigma-image (MA as top)
    out = []
    expand(pieces, SIGMA, "A", "f", 2, out)
    assert len(out) == 81 and len({c for c, _, _ in out}) == 81
    for (c1, e1, x1), (c2, e2, x2) in zip(out, out[1:]):
        assert x1 == e2
    print("type MA (sigma-image) depth 2: VERIFIED")


if __name__ == "__main__":
    main()
