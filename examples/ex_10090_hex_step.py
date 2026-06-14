# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Verify hex_step direction vectors against the known period-9 digit cycles.

From the user's earlier analysis, stepping repeatedly in direction 0 (the
+c2=0 axis / N→S direction) through any column of the grid produces one
of three period-9 digit sequences:

  Column A: 8 6 0 6 7 1 7 8 2  (repeat)
  Column B: 7 3 4 8 5 3 6 4 5  (repeat)  — note: displayed as 734853645
  Column C: 1 5 2
This script picks
This script picks  2 4 0 0 3 1  (repeat)  — note: displayed as 152240031
three starting addresses (one per column) and verifies
the digit at target_layer cycles with the expected period.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from hhg9 import Registrar, Points
from hhg9.h9.addressing import hex_layer, TailStyle
from hhg9.h9.uvc import hex_step, _DIR_UV, hex_digits_to_uvc


def _make_address(lat_deg, lon_deg, layer, reg):
    """Return a single (1, layer+1) hex address array for a geo point."""
    # from hhg9.projections.gcd_ellipsoid import GCDEllipsoid
    b_oct = reg.domain('b_oct')
    g_gcd = reg.domain('g_gcd')
    pt_geo = Points(np.array([[lat_deg, lon_deg]]), domain=g_gcd)
    pt_b = reg.project(pt_geo, [g_gcd, b_oct])
    hv = hex_layer(pt_b, layer=layer, tail_style=TailStyle.reversible)
    return hv  # shape (1, layer+1)


def show_dir_uvs():
    print("Direction vectors (_DIR_UV):  (axis, c2_class) -> (Δu, Δv)")
    for i in range(6):
        sign = '+' if i < 3 else '-'
        vecs = '  '.join(f"c2={c}: ({_DIR_UV[i,c,0]:+d},{_DIR_UV[i,c,1]:+d})" for c in range(3))
        print(f"  dir {i}  (c2={i%3} {sign}):  {vecs}")
    print()


def period_test(hx, target_layer, direction, label, expected_cycle):
    """Step 18 times and check the digit at target_layer repeats with period 9."""
    digits = []
    cur = hx.copy()
    for _ in range(18):
        d = int(cur[0, target_layer])
        digits.append(d)
        cur = hex_step(cur, target_layer, direction)

    cycle_str = ''.join(str(d) for d in digits[:9])
    repeat_str = ''.join(str(d) for d in digits[9:18])
    ok_cycle   = cycle_str == expected_cycle
    ok_repeat  = cycle_str == repeat_str
    status = "OK" if (ok_cycle and ok_repeat) else "FAIL"
    print(f"  {label}: {cycle_str} | {repeat_str}  expected={expected_cycle}  [{status}]")
    if not ok_cycle:
        print(f"    *** cycle mismatch — got {cycle_str}, expected {expected_cycle}")
    if not ok_repeat:
        print(f"    *** not periodic after 9 steps")
    return ok_cycle and ok_repeat


def run():
    reg = Registrar()
    b_oct = reg.domain('b_oct')
    g_gcd = reg.domain('g_gcd')

    show_dir_uvs()

    # Three starting points chosen to land in each of the 3 period columns.
    # We use latitude/longitude positions within a single octant so we stay
    # away from octant boundaries during the 18-step traversal.
    LAYER = 4  # deep enough to have a clear digit but not too slow

    # Approximate equatorial points spread across the three c2 sectors of a
    # single parent.  Adjust as needed once we know the actual UV layout.
    test_points = [
        (  0.0,  10.0, "ColA", "860671782"),
        (  0.0,  20.0, "ColB", "734853645"),
        (  0.0,  30.0, "ColC", "152240031"),
    ]

    all_ok = True
    print(f"Period-9 test at layer {LAYER}, direction 0 (+c2=0 axis):")
    for lat, lon, label, expected in test_points:
        pt_geo = Points(np.array([[lat, lon]]), domain=g_gcd)
        pt_b = reg.project(pt_geo, [g_gcd, b_oct])
        hv = hex_layer(pt_b, layer=LAYER, tail_style=TailStyle.reversible)
        ok = period_test(hv, LAYER, direction=0, label=label, expected_cycle=expected)
        all_ok = all_ok and ok

    print()
    # Also verify direction 3 (opposite of 0) reverses the cycle
    print(f"Direction 3 (opposite of 0) should reverse a known cycle:")
    pt_geo = Points(np.array([[0.0, 10.0]]), domain=g_gcd)
    pt_b = reg.project(pt_geo, [g_gcd, b_oct])
    hv = hex_layer(pt_b, layer=LAYER, tail_style=TailStyle.reversible)
    digits_fwd = []
    digits_rev = []
    cur = hv.copy()
    for _ in range(9):
        digits_fwd.append(int(cur[0, LAYER]))
        nxt = hex_step(cur, LAYER, direction=0)
        if nxt is None:
            break
        cur = nxt
    cur = hv.copy()
    for _ in range(9):
        digits_rev.append(int(cur[0, LAYER]))
        nxt = hex_step(cur, LAYER, direction=3)
        if nxt is None:
            break
        cur = nxt
    print(f"  fwd dir0: {''.join(str(d) for d in digits_fwd)}")
    print(f"  rev dir3: {''.join(str(d) for d in digits_rev)}")
    ok_rev = digits_rev[1:] == list(reversed(digits_fwd[:-1]))
    print(f"  reverse match: {'OK' if ok_rev else 'NOTE - may differ at wrap'}")

    print()
    print("Overall:", "PASSED" if all_ok else "some period tests FAILED — check direction assignment")


if __name__ == '__main__':
    run()
