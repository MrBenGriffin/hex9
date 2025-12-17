# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
uses H9 addresses and round-trips them.
This tests that octant ids are recorded the same everywhere.
The definitive calculation is in Points.
16 December 2025 0.1.0a3 (passed)
06 October 2025 √
"""
import numpy as np
from hhg9 import Points, Registrar

if __name__ == '__main__':
    reg = Registrar()
    b_oct = reg.domain('b_oct')
    b_lut = b_oct.sign_to_id
    for (s, i) in b_lut.items():
        pi = Points.calc_octant_ids(np.array([s]))
        p = Points(np.array([0]), b_oct, components=s)
        (c, m) = p.cm()
        print(s, i, pi, c, m)


