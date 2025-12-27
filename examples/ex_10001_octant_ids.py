# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
uses H9 addresses and round-trips them.
This tests that octant ids are recorded the same everywhere.
The definitive calculation is in Points.

Last Tested
26 December 2025 0.1.0a4 (passed)
16 December 2025 0.1.0a3 (passed)
06 October 2025 √
"""
import numpy as np
from hhg9 import Points, Registrar
from hhg9.h9 import H9O

if __name__ == '__main__':
    reg = Registrar()
    b_oct = reg.domain('b_oct')
    b_lut = H9O.oid_cmp
    for i, s in enumerate(b_lut):
        pi = Points.calc_octant_ids(np.array([s]))
        p = Points(np.array([0]), b_oct, components=s)
        (c, m) = p.cm()
        print(s, i, pi, c, m)


