# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""
PROBE: characterise canonicalisation (force leaf parent to its mode-0 half-hex).

Compares each raw address to its coalesced form (neighbours(coalesce=True)) and
classifies the change. Findings (L8, N~5k): ~50% of points need canonicalising;
of those ~57% are a pure TAIL p_mo 1->0 flip (no body change), ~43% fire a SHORT
body cascade (1-2 digits, wing digits 6/7/8 -> mode-0 sibling), octant-switch rare.
After canon, p_mo == 0 everywhere; already-canonical addresses are untouched.
"""
import numpy as np
from hhg9 import Registrar, Points
from hhg9.h9.addressing import hex_digits, neighbours
from hhg9.h9.tail import TailStyle, tail_unpack_reversible
from hhg9.h9 import H9C
import hhg9.h9.region as rg


def main(L=8, n=70):
    reg = Registrar()
    b_oct = reg.domain('b_oct'); g_gcd = reg.domain('g_gcd')
    lat = np.linspace(-80, 80, n); lon = np.linspace(-175, 175, n)
    LA, LO = np.meshgrid(lat, lon)
    bry = reg.project(Points(np.column_stack([LA.ravel(), LO.ravel()]), g_gcd), [g_gcd, b_oct])

    hx = np.asarray(hex_digits(bry, layer=L, tail_style=TailStyle.reversible), dtype=np.uint8)
    canon = neighbours(bry, layer=L, coalesce=True)
    hxc = np.asarray(hex_digits(canon, layer=L, tail_style=TailStyle.reversible), dtype=np.uint8)

    cx = np.asarray(rg.xy_regions(bry.coords, bry.cm()[1], L))
    need = H9C.mode[cx[:, -2]] == 1
    body_changed = np.any(hx[:, :-1] != hxc[:, :-1], axis=1)
    oct_switched = bry.cm()[0] != canon.cm()[0]
    _, _, p_moc = tail_unpack_reversible(hxc[:, -1])

    print(f"N={len(bry)} L={L}  need-canon={int(need.sum())} ({100*need.mean():.0f}%)")
    print(f"  tail-only p_mo flip : {int((need & ~body_changed & ~oct_switched).sum())}")
    print(f"  body cascade fired  : {int((need & body_changed).sum())}")
    print(f"  octant switched     : {int((need & oct_switched).sum())}")
    print(f"  p_mo after canon (needers): {sorted(set(p_moc[need].tolist()))}  (want [0])")
    print(f"  non-needers changed at all: {int((~need & (body_changed|oct_switched)).sum())}")
    casc = need & body_changed
    if casc.any():
        nchg = np.array([int(np.sum(hx[i, :-1] != hxc[i, :-1])) for i in np.flatnonzero(casc)])
        print(f"  cascade length: min={nchg.min()} max={nchg.max()} mean={nchg.mean():.2f}")


if __name__ == '__main__':
    main()
