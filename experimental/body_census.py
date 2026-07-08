# Part of the Hex9 (H9) Project
# Copyright ©2026, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""Canonical-body uniqueness census (paper §10b).

Claim verified: across the full canonical address space, no two cells at the
same level share a tail-stripped digit body — checked exhaustively for
L0..L5 (12, 108, 972, 8,748, 78,732, 708,588 cells).

Cells are enumerated geometrically via HexMesh; addresses come from the
canonical h9_bin_pts pipeline (the same canonicalise/coalesce path as
h9_encode), applied to mode-safe face centroids. mesh.addr() now routes
through the same pipeline (its earlier hand-rolled packer mis-extracted the
tail nibble), but this census deliberately stands on the canonical encoder
alone rather than inheriting the mesh's address plumbing.

Also demonstrated (--london): lineage is not ancestry. 43585 cuts to 4358,
yet its canonical level-3 ancestor is the hexagon canonically named 4348 —
4358 being that hexagon's non-canonical (mode-1 parent) reading, while the
body 4358 canonically names a different hexagon elsewhere.
"""
import argparse
import time
from collections import Counter

import numpy as np

from hhg9 import Registrar, Points
from hhg9.h9 import H9O
from hhg9.h9.grid import HexMesh
from hhg9.h9.uuid_address import h9_bin_pts, h9_label, h9_encode, h9_bin


def centroids(mesh, b_oct, L):
    """Mode-safe face centroids as b_oct Points (mirrors grid.py's rule)."""
    fl = mesh[L]
    oid_v = mesh.pts.oid[fl]
    mo_v = H9O.oid_mo[oid_v]
    match = (mo_v == mo_v[:, :1])
    cv = mesh.pts.coords[fl]
    cxy = (cv * match[:, :, None]).sum(1) / match.sum(1, keepdims=True)
    return Points(cxy, b_oct, oid=oid_v[:, 0].astype(np.int32))


def census(max_layer: int = 5) -> bool:
    t0 = time.time()
    reg = Registrar()
    mesh = HexMesh.create(range(max_layer + 1), reg)
    b_oct = reg.domain('b_oct')
    ok = True
    for L in mesh.layers:
        uu = h9_bin_pts(centroids(mesh, b_oct, L), L)
        labels = {h9_label(u) for u in set(uu)}
        bodies = [l.split('.')[0] for l in labels]
        dups = {b: c for b, c in Counter(bodies).items() if c > 1}
        complete = len(set(uu)) == 12 * 9 ** L
        clean = not dups
        ok &= complete and clean
        print(f'L{L}: cells {len(set(uu)):>7} (complete: {complete})  '
              f'bodies {len(set(bodies)):>7}  collisions {len(dups)}')
        for b in sorted(dups)[:5]:
            print('   COLLISION:', sorted(l for l in labels if l.startswith(b + ".")))
    print(f'[{time.time() - t0:.1f}s] body census {"PASS" if ok else "FAIL"}')
    return ok


def london() -> None:
    """Lineage != ancestry demonstrator (figure in §10b)."""
    lats = np.linspace(50.2, 52.6, 500)
    lons = np.linspace(-2.0, 2.0, 500)
    LO, LA = np.meshgrid(lons, lats)
    pu = h9_encode(LA.ravel(), LO.ravel())
    bod3 = np.array([h9_label(u).split('.')[0] for u in h9_bin(pu, 3)])
    bod4 = np.array([h9_label(u).split('.')[0] for u in h9_bin(pu, 4)])
    m = bod3 == '4348'
    print('L4 lineage bodies over canonical L3 ancestor 4348:',
          sorted(set(bod4[m])))
    m3 = bod4 == '43585'
    print('43585 canonical L3 ancestor(s):', sorted(set(bod3[m3])),
          '(cuts to 4358 — a different canonical cell)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--depth', type=int, default=5)
    ap.add_argument('--london', action='store_true')
    args = ap.parse_args()
    passed = census(args.depth)
    if args.london:
        london()
    raise SystemExit(0 if passed else 1)
