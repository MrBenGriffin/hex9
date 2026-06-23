# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""
PROBE: the key/address (k/a) distinction under the single-nibble tail.

Findings (run me): the k/a tails differ only by p_mo (bit 3).
  - RAW uint64 path (hex_pack, no coalesce): p_mo == 1 for ~50% of points -> 'ua'
    and a true key genuinely differ.
  - CANONICAL / UUID path (h9_enc, coalesces): p_mo == 0 for 100% -> address == bin.
  - The production key u64 branch is degenerate: hex_pack packs tail_ids>>4 but
    tail_pack_key is low-nibble, so 'uk' ends in 0 for every point (tail lost).
"""
import numpy as np
from hhg9 import Registrar, Points
from hhg9.formats import OctahedralH9, DMS
from hhg9.h9.addressing import hex_digits
from hhg9.h9.tail import TailStyle
from hhg9.h9.uuid_address import h9_enc, _batch_int_to_nibbles


def grid(n=60):
    lat = np.linspace(-85, 85, n); lon = np.linspace(-179, 179, n)
    LA, LO = np.meshgrid(lat, lon)
    return np.column_stack([LA.ravel(), LO.ravel()])


def main():
    reg = Registrar()
    h9f = OctahedralH9(reg)
    b_oct = reg.domain('b_oct'); g_gcd = reg.domain('g_gcd')
    b_oct.register_format(h9f); g_gcd.register_format(DMS(reg))
    bry = reg.project(Points(grid(), g_gcd), [g_gcd, b_oct])
    N = len(bry)
    print(f"=== sample N={N} ===")

    rev = np.asarray(hex_digits(bry, layer=14, tail_style=TailStyle.reversible), dtype=np.uint8)
    key = np.asarray(hex_digits(bry, layer=14, tail_style=TailStyle.key), dtype=np.uint8)
    p_mo = (rev[:, -1] >> 3) & 1
    print(f"RAW path: reversible-tail bit3 (p_mo) set: {int(p_mo.sum())}/{N} ({100*p_mo.mean():.0f}%)")
    print(f"          bodies identical (rev vs key): {np.array_equal(rev[:, :-1], key[:, :-1])}")

    # 'ua' is now the canonical u64 bin by default. ('uk'/UH64K was REMOVED — it
    # was redundant and its key tail packed as 0; legacy 'uk' now aliases to 'ua'.)
    ua = h9f.format(bry, None, 'ua').split('\n')
    print(f"          canonical 'ua' u64 bin sample: {ua[0]}")

    uuids = h9_enc(bry)
    tail = _batch_int_to_nibbles([u.int for u in uuids], n=32)[:, -1]
    print(f"CANONICAL (UUID): tail bit3 (p_mo) set: {int(((tail>>3)&1).sum())}/{N}; "
          f"tail values {sorted(set(tail.tolist()))}")


if __name__ == '__main__':
    main()
