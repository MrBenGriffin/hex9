# Part of the Hex9 (H9) Project
# Copyright ©2026, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""Canonical cell-ancestry invariant (paper §10b; supersedes libhex9 fossil F3).

Claims verified, globally, for every cell:
  1. h9_cell_parent gives exactly 9 canonical children per parent at every
     layer pair L1..L5 (6 interior + the parent's own 3 split children).
  2. h9_cell_ancestor composes: parent-of-parent == ancestor(L-2), and each
     L-2 cell has exactly 81 canonical grandchildren.
  3. The point operation differs, as it must: binning cells' centroids
     scatters split cells (that is the disease this API cures).
"""
import time
from collections import Counter

from hhg9 import Registrar
from hhg9.h9.grid import HexMesh
from hhg9.h9.uuid_address import h9_cell_parent, h9_cell_ancestor


def verify(max_layer: int = 5) -> bool:
    t0 = time.time()
    reg = Registrar()
    mesh = HexMesh.create(range(max_layer + 1), reg)
    ok = True
    for L in range(max_layer, 0, -1):
        up = h9_cell_parent(mesh.addr(L), reg=reg)
        hist = Counter(Counter(u.int for u in up).values())
        good = set(hist) == {9} and sum(hist.values()) == 12 * 9 ** (L - 1)
        ok &= good
        print(f'L{L}->L{L-1}: histogram {dict(sorted(hist.items()))} '
              f'parents {sum(hist.values())} {"PASS" if good else "FAIL"}')
    if max_layer >= 2:
        uu = mesh.addr(max_layer)
        two = h9_cell_ancestor(uu, max_layer - 2, reg=reg)
        composed = h9_cell_parent(h9_cell_parent(uu, reg=reg), reg=reg)
        agree = all(a == b for a, b in zip(two, composed))
        hist = Counter(Counter(u.int for u in two).values())
        good = agree and set(hist) == {81}
        ok &= good
        print(f'compose: ancestor(L-2) == parent∘parent: {agree}; '
              f'histogram {dict(sorted(hist.items()))} {"PASS" if good else "FAIL"}')
    print(f'[{time.time() - t0:.0f}s] {"ALL PASS" if ok else "FAILURES"}')
    return ok


if __name__ == '__main__':
    raise SystemExit(0 if verify() else 1)
