# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""
PROBE: round-trip resolution of the uint64 address formats.

Confirms the single-nibble-tail fix: 'ua' (single u64) reaches L14 sub-metre;
'ua36' (multiword) and 'r35' (UR64) stay sub-nm. Use as a regression check.
"""
import numpy as np
from hhg9 import Registrar, Points
from hhg9.formats import OctahedralH9, DMS
from hhg9.algorithms.distance import wgs84
from hhg9.h9.addressing import Style


def main():
    reg = Registrar()
    h9f = OctahedralH9(reg)
    b_oct = reg.domain('b_oct'); g_gcd = reg.domain('g_gcd')
    b_oct.register_format(h9f); g_gcd.register_format(DMS(reg))
    ll = [(1.860751, 43.328763), (35.952813, 18.328413), (-22.9068, -43.1729),
          (51.4779, -0.0015), (28.2074, -177.3754), (-89.9982, -139.2723)]
    pos = Points(np.array(ll), g_gcd)
    bry = reg.project(pos, [g_gcd, b_oct])

    for fmt, style, unit in [('ua', Style.UH64A, 'm'), ('ua36', Style.UH64A, 'nm'),
                             ('r35', Style.UR64, 'nm')]:
        enc = h9f.format(bry, None, fmt)
        dec = reg.project(h9f.revert(enc, style), [b_oct, g_gcd])
        d = wgs84(pos.coords, dec.coords)
        d = d if unit == 'm' else d * 1e9
        w = len((enc.split('\n')[0]).split(';')[0])
        print(f"{fmt:5} ({w} hex/word): max error {d.max():.4f} {unit}")


if __name__ == '__main__':
    main()
