# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
# Swap the warp data

import numpy as np
from hhg9 import Registrar, Points
from hhg9.algorithms.distance import wgs84


def test_seam_with_warp(warp_file):
    reg = Registrar()
    g_gcd = reg.domain('g_gcd')
    b_oct_d = reg.domain('b_oct')
    # Swap the warp data
    if warp_file:
        b_oct_d.set_warp(warp_file)
    else:
        b_oct_d.no_warp()

    results = []
    for lat, lon in [(51.4778301090125, 1e-11), (51.4778301090125, -1e-11),
                     (89.99, 0.0), (45.0, 90.0001e-11)]:
        pt = Points(np.array([[lat, lon]]), g_gcd)
        b = reg.project(pt, ['g_gcd', 'b_oct'])
        b2 = reg.project(b, ['b_oct', 'g_gcd'])
        err = wgs84(pt.coords, b2.coords)[0] * 1e9
        results.append((lat, lon, err))
    return results


if __name__ == '__main__':
    for label, path in [
        ('none', None),
        # ('v998', '../hhg9/data/v998_WGS84_l5_warp_data.npz'),
        ('v999', '../hhg9/data/v999_WGS84_l5_warp_data.npz'),
    ]:
        print(f'--- {label} ---')
        for lat, lon, err in test_seam_with_warp(path):
            print(f'  lat={lat:.4f} lon={lon:+.2e}: {err:.3f} nm')
